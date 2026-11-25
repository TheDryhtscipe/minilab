"""The human queue: what the user reviews, and the review log.

The queue holds exactly three kinds of record (design doc-gmj, "Human queue"):

- survivors (`survived`) whose falsifier output holds a coverage report;
- `proved` records that the triviality screen did not flag;
- certified kills (`killed`) whose falsifier output holds a certificate and a
  mechanism that passed its test.

Nothing else is ever queued: not `untestable`, `ambiguous`, `known`,
`trivial`, `abandoned`, or `proposed` records, whatever evidence they hold.

The queue reads these fields, which the stages that write them must use:

- `Stage.falsifier`: `data["coverage"]`, an object with at least the keys in
  `COVERAGE_KEYS`; `data["certificate"]`, a non-empty object;
  `data["mechanism"]`, an object whose `passed` is the boolean `true`.
- `Stage.triviality`: `data["trivial"]` is `true` if the screen proved it
  (such a record is `trivial`, but the flag is checked as well).
- `Stage.formaliser`: `data["roundtrip"]` (text) and `data["audit"]`.
- `Stage.literature`: the stage's `notes`.

A review is stored on the record as its `Stage.queue` output (verdict, note,
minutes, and times); the verdict never changes the status. The minutes go to
the run's ledger with `Ledger.log_review`.

Review time is measured from when the item was shown. `minilab queue` marks
every item it prints as shown, in `runs/<run_id>/queue/shown.json`. Items are
usually read one after another, so the clock for an item starts at the later
of the time it was shown and the end of the run's previous review.
"""

import fcntl
import os
import tempfile
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue

from minilab.ledger import Ledger
from minilab.records import (
    Record,
    RecordId,
    RecordStore,
    Stage,
    StageOutput,
    Status,
    check_record_id,
    check_run_id,
    run_of,
)
from minilab.score import Score, score

COVERAGE_KEYS: frozenset[str] = frozenset(
    {"space", "dimensions", "starts", "worst_margin", "worst_point", "uncovered"}
)
"""Keys a coverage report must hold (build plan Task 8, `CoverageReport`)."""


class NotInQueue(Exception):
    """A review named a record that is not a queue item, or one not yet shown."""


class QueueKind(StrEnum):
    survivor = "survivor"
    proved = "proved"
    kill = "kill"


class Verdict(StrEnum):
    interesting = "interesting"
    not_interesting = "not-interesting"
    known = "known"


class QueueItem(BaseModel):
    """One record in the queue, with its kind and its ranking score."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record: Record
    kind: QueueKind
    score: Score


class Shown(BaseModel):
    """When each queue item was last shown, and when the last review ended."""

    model_config = ConfigDict(extra="forbid")

    shown: dict[RecordId, AwareDatetime] = Field(default_factory=lambda: {})
    last_review: AwareDatetime | None = None


def _data(record: Record, stage: Stage, key: str) -> JsonValue:
    output = record.stages.get(stage)
    return None if output is None else output.data.get(key)


def _has_coverage(record: Record) -> bool:
    coverage = _data(record, Stage.falsifier, "coverage")
    return isinstance(coverage, dict) and COVERAGE_KEYS <= coverage.keys()


def _has_certificate(record: Record) -> bool:
    certificate = _data(record, Stage.falsifier, "certificate")
    return isinstance(certificate, dict) and bool(certificate)


def _mechanism_passed(record: Record) -> bool:
    mechanism = _data(record, Stage.falsifier, "mechanism")
    return isinstance(mechanism, dict) and mechanism.get("passed") is True


def queue_kind(record: Record) -> QueueKind | None:
    """Return the record's kind of queue item, or None if it is not queued."""
    if record.status is Status.survived and _has_coverage(record):
        return QueueKind.survivor
    if (
        record.status is Status.proved
        and Stage.prover in record.stages
        and _data(record, Stage.triviality, "trivial") is not True
    ):
        return QueueKind.proved
    if record.status is Status.killed and _has_certificate(record) and _mechanism_passed(record):
        return QueueKind.kill
    return None


def queue_items(store: RecordStore, run_id: str) -> list[QueueItem]:
    """Return a run's queue items, best ranked first (see `Score.rank_key`)."""
    items: list[QueueItem] = []
    for record in store.list(run_id):
        kind = queue_kind(record)
        if kind is not None:
            items.append(QueueItem(record=record, kind=kind, score=score(record)))
    items.sort(key=lambda item: (item.score.rank_key(), item.record.id))
    return items


def _queue_dir(root: Path, run_id: str) -> Path:
    return root / "runs" / check_run_id(run_id) / "queue"


@contextmanager
def _locked(root: Path, run_id: str) -> Generator[Path]:
    """Hold the run's queue lock; yield the path of its shown file."""
    directory = _queue_dir(root, run_id)
    directory.mkdir(parents=True, exist_ok=True)
    fd = os.open(directory / ".lock", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield directory / "shown.json"
    finally:
        os.close(fd)


def _read_shown(path: Path) -> Shown:
    if not path.exists():
        return Shown()
    return Shown.model_validate_json(path.read_bytes())


def _write_shown(path: Path, shown: Shown) -> None:
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".shown.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(shown.model_dump_json(indent=2).encode())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def load_shown(root: Path, run_id: str) -> Shown:
    """Read when each of a run's queue items was last shown."""
    with _locked(root, run_id) as path:
        return _read_shown(path)


def mark_shown(root: Path, run_id: str, record_ids: Sequence[str], at: datetime) -> None:
    """Record that the given items of run `run_id` were shown at `at`."""
    run_id = check_run_id(run_id)
    for record_id in record_ids:
        if run_of(check_record_id(record_id)) != run_id:
            raise ValueError(f"record {record_id} is not in run {run_id}")
    with _locked(root, run_id) as path:
        shown = _read_shown(path)
        shown.shown.update(dict.fromkeys(record_ids, at))
        _write_shown(path, shown)


def review(
    store: RecordStore,
    ledger: Ledger,
    record_id: str,
    verdict: Verdict,
    note: str,
    *,
    now: datetime | None = None,
) -> float:
    """Store the user's verdict on a queue item and log the minutes it took.

    Return the minutes logged. Raise `NotInQueue` if the record is not a queue
    item or was never shown, `ValueError` if it was already reviewed or
    belongs to another run than the ledger's, and `KeyError` if no such
    record is stored.
    """
    record_id = check_record_id(record_id)
    if run_of(record_id) != ledger.run_id:
        raise ValueError(f"record {record_id} is not in the ledger's run {ledger.run_id}")
    now = datetime.now(UTC) if now is None else now
    with _locked(store.root, ledger.run_id) as path:
        record = store.load(record_id)
        if queue_kind(record) is None:
            raise NotInQueue(f"{record_id} ({record.status}) is not a queue item")
        if Stage.queue in record.stages:
            raise ValueError(f"{record_id} is already reviewed")
        shown = _read_shown(path)
        shown_at = shown.shown.get(record_id)
        if shown_at is None:
            raise NotInQueue(f"{record_id} has not been shown; run `minilab queue` first")
        start = shown_at
        if shown.last_review is not None and shown.last_review > start:
            start = shown.last_review
        minutes = max(0.0, (now - start).total_seconds() / 60)
        data: dict[str, JsonValue] = {
            "verdict": verdict.value,
            "note": note,
            "minutes": minutes,
            "shown_at": shown_at.isoformat(),
            "reviewed_at": now.isoformat(),
        }
        store.write_stage(record_id, Stage.queue, StageOutput(written_at=now, data=data))
        ledger.log_review(record_id, minutes)
        shown.last_review = now
        _write_shown(path, shown)
    return minutes
