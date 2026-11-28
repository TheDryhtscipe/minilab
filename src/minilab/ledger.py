"""The cost ledger and the cost report.

Every model call and every review is logged from the first run (design doc-gmj,
"Cost ledger"). Each run has one ledger: an append-only file at
`runs/<run_id>/ledger.jsonl`, one JSON entry per line. The ledger, not the
record files, is the source of truth for costs. `Ledger.costs` gives one
record's costs; `Record.costs` is not filled in.

A charge names its domain pair and stage, and may name a record. A model call
made before any record exists, such as a generation call, or one that failed,
is still charged to its pair, so the per-pair budget covers every call. A
ledger opens only for a run with a manifest, and refuses a pair or a family
that the manifest does not list.

A caller that is about to make a model call first reserves tokens for it.
Inside `Ledger.pair_lock` it checks the budget with `pair_spend` and appends a
reservation; then it releases the lock, makes the call, and settles the
reservation, with a charge that names it or, if the call never started, a
release. So calls on one pair can run at once, and each check sees the tokens
the calls still in flight have reserved. `pair_spend` counts every charge and
every reservation not yet settled. A reservation left open by a crash stays
counted as an estimated cost of the tokens it reserved; the crashed call may
have spent more, and the ledger cannot know.
"""

import builtins
import fcntl
import os
import uuid
from collections import defaultdict
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    TypeAdapter,
    ValidationError,
    field_validator,
)

from minilab.records import (
    Costs,
    Family,
    RecordId,
    RecordStore,
    Stage,
    Status,
    TokenCount,
    check_record_id,
    check_run_id,
    run_of,
)
from minilab.run import RunManifest, load_manifest
from minilab.scheduler import Pair, canonical_pair

MAX_TOKENS = 10**12
"""Largest token count one charge may log."""
MAX_SECONDS = 10.0**7
"""Largest duration one charge may log (about 116 days)."""
MAX_MINUTES = 10.0**5
"""Largest review time one review may log (about 69 days)."""

ReservationId = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
"""A reservation's id: 32 lowercase hex digits (a UUID4)."""

QUEUEABLE: frozenset[Status] = frozenset({Status.survived, Status.proved, Status.killed})
"""Statuses a record in the review queue can have (doc-gmj, "Human queue")."""


class CorruptLedger(Exception):
    """A ledger line does not parse, or the file ends partway through a line."""


class _Entry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    at: AwareDatetime


class Charge(_Entry):
    """One model call's cost.

    `estimated` is true when the tool did not report its token counts and the
    adapter estimated them. `failed` is true when the call started but gave no
    reply; a charge logged before this field existed loads as false.
    `reservation` names the reservation this charge settles, if any.
    """

    kind: Literal["charge"] = "charge"
    pair: Pair
    stage: Stage
    family: Family
    record_id: RecordId | None
    tokens_in: Annotated[StrictInt, Field(ge=0, le=MAX_TOKENS)]
    tokens_out: Annotated[StrictInt, Field(ge=0, le=MAX_TOKENS)]
    seconds: Annotated[float, Field(strict=True, ge=0, le=MAX_SECONDS, allow_inf_nan=False)]
    estimated: StrictBool
    failed: StrictBool = False
    reservation: ReservationId | None = None

    @field_validator("pair")
    @classmethod
    def _canonical(cls, value: Pair) -> Pair:
        return canonical_pair(value)

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out


class Reservation(_Entry):
    """Tokens set aside for a model call that has not been charged yet.

    It counts toward its pair's spend until a charge or a release settles it.
    """

    kind: Literal["reservation"] = "reservation"
    id: ReservationId
    pair: Pair
    stage: Stage
    family: Family
    record_id: RecordId | None
    tokens: Annotated[StrictInt, Field(ge=0, le=MAX_TOKENS)]

    @field_validator("pair")
    @classmethod
    def _canonical(cls, value: Pair) -> Pair:
        return canonical_pair(value)


class Release(_Entry):
    """Settles a reservation whose call never started, so it cost nothing."""

    kind: Literal["release"] = "release"
    reservation: ReservationId


class Review(_Entry):
    """Minutes the user spent reviewing one queued record."""

    kind: Literal["review"] = "review"
    record_id: RecordId
    minutes: Annotated[float, Field(strict=True, ge=0, le=MAX_MINUTES, allow_inf_nan=False)]


AnyEntry = Charge | Reservation | Release | Review
Entry = Annotated[AnyEntry, Field(discriminator="kind")]
_ENTRY: TypeAdapter[AnyEntry] = TypeAdapter(Entry)


def _open_reservations(entries: Iterable[AnyEntry]) -> dict[str, Reservation]:
    """Return the reservations no later entry settles, by id, in the order logged."""
    reserved: dict[str, Reservation] = {}
    for entry in entries:
        if isinstance(entry, Reservation):
            reserved[entry.id] = entry
        elif isinstance(entry, Release):
            reserved.pop(entry.reservation, None)
        elif isinstance(entry, Charge) and entry.reservation is not None:
            reserved.pop(entry.reservation, None)
    return reserved


def _counted_charges(entries: builtins.list[AnyEntry]) -> builtins.list[Charge]:
    """Return the charges, plus each open reservation as an estimated charge of its tokens.

    An open reservation's call may have run and spent tokens that were never
    charged, so it counts as spend wherever costs are added up. No reply of
    its reached the run, so it counts as a failed call.
    """
    counted = [entry for entry in entries if isinstance(entry, Charge)]
    for held in _open_reservations(entries).values():
        counted.append(
            Charge(
                at=held.at,
                pair=held.pair,
                stage=held.stage,
                family=held.family,
                record_id=held.record_id,
                tokens_in=held.tokens,
                tokens_out=0,
                seconds=0.0,
                estimated=True,
                failed=True,
                reservation=held.id,
            )
        )
    return counted


class Ledger:
    """One run's costs, kept in `runs/<run_id>/ledger.jsonl` under the store's root.

    Each entry is appended with a single write under an exclusive `flock` on
    the ledger file, and reads take a shared `flock`, so processes on one local
    file system can log to and read the same ledger at once without seeing a
    half-written line. Reads parse the whole file. A line that does not parse,
    names a record of another run, settles a reservation that is not open, or
    is cut short by a crash raises `CorruptLedger` naming the line; nothing is
    skipped, because a skipped line would hide spend. An append after a
    cut-short line is refused too, so that a new entry is never joined onto a
    broken one.

    A charge or release that settles a reservation is checked against the
    file under the same exclusive lock as its append, so a reservation is
    settled at most once even when several processes try at the same time.

    Raise `FileNotFoundError` if the run has no manifest.
    """

    def __init__(self, store: RecordStore, run_id: str) -> None:
        self.store = store
        self.run_id = check_run_id(run_id)
        self.manifest: RunManifest = load_manifest(store.root, self.run_id)
        self.path = store.root / "runs" / self.run_id / "ledger.jsonl"

    def _own_record(self, record_id: str) -> str:
        record_id = check_record_id(record_id)
        if run_of(record_id) != self.run_id:
            raise ValueError(
                f"record {record_id} belongs to run {run_of(record_id)}, "
                f"not this ledger's run {self.run_id}"
            )
        return record_id

    def _append(
        self,
        entry: AnyEntry,
        check: Callable[[builtins.list[AnyEntry]], None] | None = None,
    ) -> None:
        """Append `entry`. If `check` is given, call it first with every entry
        in the file, under the same exclusive lock; it raises to refuse."""
        line = entry.model_dump_json().encode() + b"\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            size = os.fstat(fd).st_size
            if size and os.pread(fd, 1, size - 1) != b"\n":
                raise CorruptLedger(f"{self.path}: the last line is incomplete; refused to append")
            if check is not None:
                check(self._parse(_read_all(fd, size)))
            written = os.write(fd, line)
            if written != len(line):
                raise OSError(f"{self.path}: short write ({written} of {len(line)} bytes)")
            os.fsync(fd)
        finally:
            os.close(fd)

    def _check_target(self, entry: Charge | Reservation) -> None:
        if entry.pair not in self.manifest.pairs:
            raise ValueError(f"pair {entry.pair} is not in run {self.run_id}'s manifest")
        if entry.family not in self.manifest.families:
            raise ValueError(f"family {entry.family} is not in run {self.run_id}'s manifest")
        if entry.record_id is not None:
            record = self.store.load(self._own_record(entry.record_id))
            if canonical_pair(record.pair) != entry.pair:
                raise ValueError(
                    f"record {record.id} is for pair {record.pair}, not {entry.pair}"
                )

    def _find_open(self, entries: builtins.list[AnyEntry], reservation: str) -> Reservation:
        held = _open_reservations(entries).get(reservation)
        if held is None:
            raise ValueError(f"reservation {reservation} is not open in run {self.run_id}")
        return held

    def charge(
        self,
        *,
        pair: Pair,
        stage: Stage,
        family: Family,
        tokens_in: int,
        tokens_out: int,
        seconds: float,
        estimated: bool,
        failed: bool = False,
        record_id: str | None = None,
        reservation: str | None = None,
    ) -> None:
        """Log one model call's tokens and its duration in seconds.

        `failed` marks a call that started but gave no reply.

        If `reservation` is given, the charge settles that reservation: from
        then on the charge counts and the reservation does not.

        Raise `ValidationError` for an amount that is negative, non-finite,
        too large, or of the wrong type. Raise `ValueError` for an unknown pair,
        a pair or family the manifest does not list, a record of another run
        or another pair, or a reservation that is not open or was made for
        another pair, stage, family, or record. Raise `KeyError` if
        `record_id` names no stored record. Nothing is logged on any error.
        """
        entry = Charge(
            at=datetime.now(UTC),
            pair=pair,
            stage=stage,
            family=family,
            record_id=record_id,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            seconds=seconds,
            estimated=estimated,
            failed=failed,
            reservation=reservation,
        )
        self._check_target(entry)
        settles = entry.reservation
        if settles is None:
            self._append(entry)
            return

        def check(entries: builtins.list[AnyEntry]) -> None:
            held = self._find_open(entries, settles)
            if held.pair != entry.pair:
                raise ValueError(f"reservation {held.id} is for pair {held.pair}, not {entry.pair}")
            if held.family != entry.family:
                raise ValueError(
                    f"reservation {held.id} is for family {held.family}, not {entry.family}"
                )
            if held.stage != entry.stage:
                raise ValueError(
                    f"reservation {held.id} is for stage {held.stage}, not {entry.stage}"
                )
            if held.record_id != entry.record_id:
                raise ValueError(
                    f"reservation {held.id} is for record {held.record_id}, "
                    f"not {entry.record_id}"
                )

        self._append(entry, check)

    def reserve(
        self,
        *,
        pair: Pair,
        stage: Stage,
        family: Family,
        tokens: int,
        record_id: str | None = None,
    ) -> str:
        """Set aside `tokens` for a model call on `pair`, and return the reservation's id.

        The reservation counts toward the pair's spend until `charge` (with
        `reservation=`) or `release` settles it. To keep calls within a
        budget, check `pair_spend` and reserve inside one `pair_lock` block.
        Raise as `charge` does for a bad amount, pair, family, or record.
        """
        entry = Reservation(
            at=datetime.now(UTC),
            id=uuid.uuid4().hex,
            pair=pair,
            stage=stage,
            family=family,
            record_id=record_id,
            tokens=tokens,
        )
        self._check_target(entry)
        self._append(entry)
        return entry.id

    def release(self, reservation: str) -> None:
        """Settle a reservation whose call never started, so it costs nothing.

        Raise `ValueError` if the reservation is not open.
        """
        entry = Release(at=datetime.now(UTC), reservation=reservation)

        def check(entries: builtins.list[AnyEntry]) -> None:
            self._find_open(entries, entry.reservation)

        self._append(entry, check)

    def open_reservations(self) -> builtins.list[Reservation]:
        """Return the reservations no charge or release has settled, in the order logged."""
        return list(_open_reservations(self.entries()).values())

    def log_review(self, record_id: str, minutes: float) -> None:
        """Log the user's review minutes for a record.

        Raise `ValidationError` for minutes that are negative, non-finite, too
        large, or of the wrong type. Raise `ValueError` for a record of another
        run or one whose status can never be queued, and `KeyError` if no such
        record is stored.
        """
        entry = Review(at=datetime.now(UTC), record_id=record_id, minutes=minutes)
        record = self.store.load(self._own_record(entry.record_id))
        if record.status not in QUEUEABLE:
            raise ValueError(
                f"record {record.id} is {record.status}; only queue items are reviewed"
            )
        self._append(entry)

    def entries(self) -> builtins.list[AnyEntry]:
        """Return every entry in the order it was logged."""
        try:
            fd = os.open(self.path, os.O_RDONLY)
        except FileNotFoundError:
            return []
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
            data = _read_all(fd, os.fstat(fd).st_size)
        finally:
            os.close(fd)
        return self._parse(data)

    def _parse(self, data: bytes) -> builtins.list[AnyEntry]:
        """Parse the ledger file's bytes. Raise `CorruptLedger` for any bad line,
        and for a settlement of a reservation that is not open at that line."""
        if data and not data.endswith(b"\n"):
            count = data.count(b"\n") + 1
            raise CorruptLedger(f"{self.path}: line {count} is incomplete")
        entries: builtins.list[AnyEntry] = []
        reserved: set[str] = set()
        seen: set[str] = set()
        for number, line in enumerate(data.splitlines(), start=1):
            try:
                entry = _ENTRY.validate_json(line)
            except ValidationError as exc:
                raise CorruptLedger(f"{self.path}: line {number}: {exc}") from exc
            record_id = None if isinstance(entry, Release) else entry.record_id
            if record_id is not None and run_of(record_id) != self.run_id:
                raise CorruptLedger(
                    f"{self.path}: line {number}: record {record_id} is not in this run"
                )
            if isinstance(entry, Reservation):
                if entry.id in seen:
                    raise CorruptLedger(
                        f"{self.path}: line {number}: reservation {entry.id} is logged twice"
                    )
                seen.add(entry.id)
                reserved.add(entry.id)
            settles = entry.reservation if isinstance(entry, Charge | Release) else None
            if settles is not None:
                if settles not in reserved:
                    raise CorruptLedger(
                        f"{self.path}: line {number}: settles reservation {settles}, "
                        "which is not open"
                    )
                reserved.discard(settles)
            entries.append(entry)
        return entries

    @contextmanager
    def pair_lock(self, pair: Pair) -> Generator[None]:
        """Hold the pair's lock, so no other process using it can reserve on the pair.

        Check the budget and reserve tokens for the call inside one
        `with ledger.pair_lock(pair):` block; then no two processes can both
        pass the check before either reserves. Make the call and settle the
        reservation after the block, so other calls on the pair can start.
        The lock is an exclusive `flock` on `runs/<run_id>/locks/<pair>.lock`,
        so it holds across processes on one machine. It does not nest: taking
        a pair's lock again while holding it blocks forever. `reserve` and
        `charge` do not take the lock, and a caller that skips it is not
        stopped.

        Raise `ValueError` for a pair the manifest does not list.
        """
        pair = canonical_pair(pair)
        if pair not in self.manifest.pairs:
            raise ValueError(f"pair {pair} is not in run {self.run_id}'s manifest")
        path = self.path.parent / "locks" / f"{pair[0]}+{pair[1]}.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def charges(self) -> builtins.list[Charge]:
        """Return the charges logged, in order. Open reservations are not in
        this list, though `pair_spend`, `costs`, and `cost_report` count them."""
        return [entry for entry in self.entries() if isinstance(entry, Charge)]

    def reviews(self) -> builtins.list[Review]:
        return [entry for entry in self.entries() if isinstance(entry, Review)]

    def pair_spend(self, pair: Pair) -> int:
        """Return the tokens (in plus out) charged to `pair`, from every stage,
        plus the tokens of the pair's open reservations.

        Estimated counts are included.
        """
        pair = canonical_pair(pair)
        return sum(c.tokens for c in _counted_charges(self.entries()) if c.pair == pair)

    def costs(self, record_id: str) -> Costs:
        """Return one record's costs: tokens by family, seconds by stage, review minutes.

        An open reservation for the record counts as an estimated input cost.
        """
        record_id = self._own_record(record_id)
        tokens_in: dict[Family, int] = defaultdict(int)
        tokens_out: dict[Family, int] = defaultdict(int)
        estimated: dict[Family, bool] = defaultdict(bool)
        seconds: dict[Stage, float] = defaultdict(float)
        entries = self.entries()
        for charge in _counted_charges(entries):
            if charge.record_id != record_id:
                continue
            tokens_in[charge.family] += charge.tokens_in
            tokens_out[charge.family] += charge.tokens_out
            estimated[charge.family] |= charge.estimated
            if charge.seconds:
                seconds[charge.stage] += charge.seconds
        minutes = sum(
            entry.minutes
            for entry in entries
            if isinstance(entry, Review) and entry.record_id == record_id
        )
        return Costs(
            tokens={
                family: TokenCount(
                    tokens_in=tokens_in[family],
                    tokens_out=tokens_out[family],
                    estimated=estimated[family],
                )
                for family in tokens_in
            },
            seconds=dict(seconds),
            review_minutes=minutes,
        )


class CostReport(BaseModel):
    """A run's cost per human-reviewed survivor, and each family's share.

    A reviewed survivor is a record the user has reviewed at least once. The
    cost of a run is everything charged to it, including calls for records
    that were later killed and calls with no record. Hours are the summed
    durations of the model calls plus the user's review hours; both parts are
    also given on their own. Calls that ran in parallel each count in full, so
    machine hours can exceed the run's elapsed time. Per-survivor figures are
    `None` when nothing was reviewed.

    Only queue items are reviewed (`Ledger.log_review` refuses other
    statuses), so the reviewed survivors are the reviewed queue items:
    survivors, proved results, and certified kills.

    A reservation left open (its call crashed before it was charged) counts
    as an estimated, failed charge of the tokens it reserved, with no duration.
    `failed_tokens` is the part of `total_tokens` charged to failed calls; it
    stays in the total and in the per-survivor figures.

    `family_cost_share` is each charged family's share of the run's tokens; it
    is empty when the run has charged no tokens.
    `family_survivor_share` is each family's share of the reviewed survivors,
    credited to the family that wrote the record (`Provenance.family`).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    reviewed_survivors: int
    total_tokens: int
    unattributed_tokens: int
    """Tokens charged with no record, such as generation calls."""
    failed_tokens: int
    """Tokens charged to calls that started but gave no reply."""
    machine_hours: float
    review_hours: float
    tokens_per_reviewed_survivor: float | None
    machine_hours_per_reviewed_survivor: float | None
    review_hours_per_reviewed_survivor: float | None
    hours_per_reviewed_survivor: float | None
    family_cost_share: dict[Family, float]
    family_survivor_share: dict[Family, float]
    has_estimates: bool
    """True if any charge in the run used estimated token counts."""


def cost_report(store: RecordStore, ledger: Ledger, run_id: str) -> CostReport:
    """Build the cost report for a run from its ledger and its records.

    Raise `ValueError` if the ledger belongs to another run, or if it logs a
    review for a record that is not in the store.
    """
    run_id = check_run_id(run_id)
    if ledger.run_id != run_id:
        raise ValueError(f"ledger is for run {ledger.run_id}, not run {run_id}")
    entries = ledger.entries()
    charges = _counted_charges(entries)
    reviews = [entry for entry in entries if isinstance(entry, Review)]
    records = {record.id: record for record in store.list(run_id)}

    reviewed = sorted({review.record_id for review in reviews})
    missing = [record_id for record_id in reviewed if record_id not in records]
    if missing:
        raise ValueError(f"ledger logs reviews for records not in the store: {missing}")
    count = len(reviewed)

    total_tokens = sum(charge.tokens for charge in charges)
    unattributed = sum(charge.tokens for charge in charges if charge.record_id is None)
    failed = sum(charge.tokens for charge in charges if charge.failed)
    machine_hours = sum(charge.seconds for charge in charges) / 3600
    review_hours = sum(review.minutes for review in reviews) / 60

    family_tokens: dict[Family, int] = defaultdict(int)
    for charge in charges:
        family_tokens[charge.family] += charge.tokens
    cost_share = (
        {family: tokens / total_tokens for family, tokens in family_tokens.items()}
        if total_tokens
        else {}
    )
    survivor_count: dict[Family, int] = defaultdict(int)
    for record_id in reviewed:
        survivor_count[records[record_id].provenance.family] += 1
    survivor_share = {family: n / count for family, n in survivor_count.items()}

    def per(value: float) -> float | None:
        return value / count if count else None

    return CostReport(
        run_id=run_id,
        reviewed_survivors=count,
        total_tokens=total_tokens,
        unattributed_tokens=unattributed,
        failed_tokens=failed,
        machine_hours=machine_hours,
        review_hours=review_hours,
        tokens_per_reviewed_survivor=per(total_tokens),
        machine_hours_per_reviewed_survivor=per(machine_hours),
        review_hours_per_reviewed_survivor=per(review_hours),
        hours_per_reviewed_survivor=per(machine_hours + review_hours),
        family_cost_share=cost_share,
        family_survivor_share=survivor_share,
        has_estimates=any(charge.estimated for charge in charges),
    )


def _read_all(fd: int, size: int) -> bytes:
    """Read the first `size` bytes of the open file `fd`."""
    chunks: builtins.list[bytes] = []
    offset = 0
    while offset < size:
        chunk = os.pread(fd, size - offset, offset)
        if not chunk:
            break
        chunks.append(chunk)
        offset += len(chunk)
    return b"".join(chunks)
