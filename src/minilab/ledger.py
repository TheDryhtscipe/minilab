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

`pair_spend` only reads. A caller that checks the budget and then makes a call
must not let another process charge the same pair in between, or the pair can
overrun its budget.
"""

import builtins
import os
from collections import defaultdict
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
    adapter estimated them.
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

    @field_validator("pair")
    @classmethod
    def _canonical(cls, value: Pair) -> Pair:
        return canonical_pair(value)

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out


class Review(_Entry):
    """Minutes the user spent reviewing one queued record."""

    kind: Literal["review"] = "review"
    record_id: RecordId
    minutes: Annotated[float, Field(strict=True, ge=0, le=MAX_MINUTES, allow_inf_nan=False)]


Entry = Annotated[Charge | Review, Field(discriminator="kind")]
_ENTRY: TypeAdapter[Charge | Review] = TypeAdapter(Entry)


class Ledger:
    """One run's costs, kept in `runs/<run_id>/ledger.jsonl` under the store's root.

    Each entry is appended with a single write to a file opened for appending,
    so processes on one local file system can log to the same ledger at once.
    Reads parse the whole file. A line that does not parse, names a record of
    another run, or is cut short by a crash raises `CorruptLedger` naming the
    line; nothing is skipped, because a skipped line would hide spend. An
    append after a cut-short line is refused too, so that a new entry is never
    joined onto a broken one.

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

    def _append(self, entry: Charge | Review) -> None:
        line = entry.model_dump_json().encode() + b"\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            size = os.fstat(fd).st_size
            if size and os.pread(fd, 1, size - 1) != b"\n":
                raise CorruptLedger(f"{self.path}: the last line is incomplete; refused to append")
            written = os.write(fd, line)
            if written != len(line):
                raise OSError(f"{self.path}: short write ({written} of {len(line)} bytes)")
            os.fsync(fd)
        finally:
            os.close(fd)

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
        record_id: str | None = None,
    ) -> None:
        """Log one model call's tokens and its duration in seconds.

        Raise `ValidationError` for an amount that is negative, non-finite,
        too large, or of the wrong type. Raise `ValueError` for an unknown pair,
        a pair or family the manifest does not list, or a record of another run
        or another pair. Raise `KeyError` if `record_id` names no stored record.
        Nothing is logged on any error.
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
        )
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
        self._append(entry)

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

    def entries(self) -> builtins.list[Charge | Review]:
        """Return every entry in the order it was logged."""
        try:
            data = self.path.read_bytes()
        except FileNotFoundError:
            return []
        if data and not data.endswith(b"\n"):
            count = data.count(b"\n") + 1
            raise CorruptLedger(f"{self.path}: line {count} is incomplete")
        entries: builtins.list[Charge | Review] = []
        for number, line in enumerate(data.splitlines(), start=1):
            try:
                entry = _ENTRY.validate_json(line)
            except ValidationError as exc:
                raise CorruptLedger(f"{self.path}: line {number}: {exc}") from exc
            if entry.record_id is not None and run_of(entry.record_id) != self.run_id:
                raise CorruptLedger(
                    f"{self.path}: line {number}: record {entry.record_id} is not in this run"
                )
            entries.append(entry)
        return entries

    def charges(self) -> builtins.list[Charge]:
        return [entry for entry in self.entries() if isinstance(entry, Charge)]

    def reviews(self) -> builtins.list[Review]:
        return [entry for entry in self.entries() if isinstance(entry, Review)]

    def pair_spend(self, pair: Pair) -> int:
        """Return the tokens (in plus out) charged to `pair`, from every stage.

        Estimated counts are included.
        """
        pair = canonical_pair(pair)
        return sum(charge.tokens for charge in self.charges() if charge.pair == pair)

    def costs(self, record_id: str) -> Costs:
        """Return one record's costs: tokens by family, seconds by stage, review minutes."""
        record_id = self._own_record(record_id)
        tokens_in: dict[Family, int] = defaultdict(int)
        tokens_out: dict[Family, int] = defaultdict(int)
        estimated: dict[Family, bool] = defaultdict(bool)
        seconds: dict[Stage, float] = defaultdict(float)
        minutes = 0.0
        for entry in self.entries():
            if entry.record_id != record_id:
                continue
            if isinstance(entry, Charge):
                tokens_in[entry.family] += entry.tokens_in
                tokens_out[entry.family] += entry.tokens_out
                estimated[entry.family] |= entry.estimated
                seconds[entry.stage] += entry.seconds
            else:
                minutes += entry.minutes
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
    charges = [entry for entry in entries if isinstance(entry, Charge)]
    reviews = [entry for entry in entries if isinstance(entry, Review)]
    records = {record.id: record for record in store.list(run_id)}

    reviewed = sorted({review.record_id for review in reviews})
    missing = [record_id for record_id in reviewed if record_id not in records]
    if missing:
        raise ValueError(f"ledger logs reviews for records not in the store: {missing}")
    count = len(reviewed)

    total_tokens = sum(charge.tokens for charge in charges)
    unattributed = sum(charge.tokens for charge in charges if charge.record_id is None)
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
