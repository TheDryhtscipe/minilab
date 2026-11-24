"""Records and the record store.

Every conjecture is a record: a JSON file that each stage reads and extends
(design doc-gmj, "Records and stages"). A stage adds its own output and may set
the status; it never edits another stage's output, and nothing changes a final
status.
"""

import builtins
import os
import tempfile
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

# Record and run IDs become file and directory names, so they must be plain names.
SafeId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]
_SAFE_ID: TypeAdapter[str] = TypeAdapter(SafeId)


class Status(StrEnum):
    proposed = "proposed"
    untestable = "untestable"
    ambiguous = "ambiguous"
    killed = "killed"
    known = "known"
    trivial = "trivial"
    proved = "proved"
    survived = "survived"
    abandoned = "abandoned"


FINAL_STATUSES: frozenset[Status] = frozenset(
    {
        Status.killed,
        Status.known,
        Status.proved,
        Status.trivial,
        Status.abandoned,
        Status.untestable,
    }
)
"""Statuses that no later write may change."""


class Stage(StrEnum):
    generator = "generator"
    encoding = "encoding"
    falsifier = "falsifier"
    literature = "literature"
    formaliser = "formaliser"
    triviality = "triviality"
    prover = "prover"
    scorer = "scorer"
    queue = "queue"


class Family(StrEnum):
    claude = "claude"
    codex = "codex"
    gemini = "gemini"
    local = "local"


class StageAlreadyWritten(Exception):
    """A stage tried to write output that the record already holds."""


class FinalStatus(Exception):
    """A write tried to change a final status."""


class RecordExists(Exception):
    """A record with this ID already exists in the store."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Provenance(_Model):
    """Where a record came from."""

    family: Family
    model: str
    prompt_hash: str
    parents: list[SafeId] = Field(default_factory=lambda: [])
    """IDs of the records this one reformulates or refines."""


class StageOutput(_Model):
    """One stage's output on a record.

    `data` holds the stage's own fields. `evidence` holds paths, relative to the
    run directory, to files that back a verdict: a certificate, a coverage
    report, literature notes, or a Lean file.
    """

    written_at: datetime
    data: dict[str, JsonValue] = Field(default_factory=lambda: {})
    evidence: list[str] = Field(default_factory=lambda: [])
    notes: str = ""


class TokenCount(_Model):
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    estimated: bool = False
    """True if any count was estimated because the tool did not report it."""


class Costs(_Model):
    """Tokens by model family, wall-clock seconds by stage, and review minutes."""

    tokens: dict[Family, TokenCount] = Field(default_factory=lambda: {})
    seconds: dict[Stage, Annotated[float, Field(ge=0)]] = Field(default_factory=lambda: {})
    review_minutes: float = Field(default=0.0, ge=0)


class Record(_Model):
    """One conjecture and everything the stages have found about it.

    A record with no instance family (None or blank) is `untestable`: a new
    record without one gets that status, and any other status is refused.
    """

    id: SafeId
    run_id: SafeId
    pair: tuple[str, str]
    mode: Literal["plain", "structured"]
    statement: str
    objects: list[str]
    assumptions: list[str]
    domain_of_validity: str
    instance_family: str | None
    provenance: Provenance
    status: Status = Status.proposed
    stages: dict[Stage, StageOutput] = Field(default_factory=lambda: {})
    costs: Costs = Field(default_factory=Costs)

    @field_validator("instance_family")
    @classmethod
    def _blank_is_missing(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _untestable_without_instance_family(self) -> Self:
        if self.instance_family is None:
            if self.status is Status.proposed:
                self.status = Status.untestable
            elif self.status is not Status.untestable:
                raise ValueError(
                    f"a record with no instance family must be untestable, not {self.status}"
                )
        return self


def _check_id(value: str) -> str:
    try:
        return _SAFE_ID.validate_python(value)
    except ValidationError as exc:
        raise ValueError(f"not a valid record or run ID: {value!r}") from exc


class RecordStore:
    """Records on disk, one JSON file each at `runs/<run_id>/records/<id>.json`.

    Every write goes to a temporary file in the same directory, then replaces
    the target in one rename, so a reader sees either the old file or the new
    one. Record IDs are unique across runs, so `load` needs only the ID.

    Writes to one record are not locked against each other: two processes that
    write stages to the same record at once can lose one write. The pipeline
    runs each record's stages in order, one at a time.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def _records_dir(self, run_id: str) -> Path:
        return self.root / "runs" / run_id / "records"

    def _path(self, run_id: str, record_id: str) -> Path:
        return self._records_dir(run_id) / f"{record_id}.json"

    def _find(self, record_id: str) -> Path | None:
        matches = sorted((self.root / "runs").glob(f"*/records/{_check_id(record_id)}.json"))
        if len(matches) > 1:
            raise RuntimeError(f"record {record_id} exists in more than one run: {matches}")
        return matches[0] if matches else None

    @staticmethod
    def _write_temp(directory: Path, record: Record) -> Path:
        fd, name = tempfile.mkstemp(dir=directory, prefix=f".{record.id}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(record.model_dump_json(indent=2).encode())
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            Path(name).unlink(missing_ok=True)
            raise
        return Path(name)

    @staticmethod
    def _sync_dir(directory: Path) -> None:
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def create(self, record: Record) -> None:
        """Store a new record. Raise `RecordExists` if its ID is taken in any run."""
        if self._find(record.id) is not None:
            raise RecordExists(record.id)
        directory = self._records_dir(record.run_id)
        directory.mkdir(parents=True, exist_ok=True)
        temp = self._write_temp(directory, record)
        try:
            # link() fails if the target exists, so a racing create cannot overwrite.
            os.link(temp, self._path(record.run_id, record.id))
        except FileExistsError as exc:
            raise RecordExists(record.id) from exc
        finally:
            temp.unlink(missing_ok=True)
        self._sync_dir(directory)

    def load(self, record_id: str) -> Record:
        """Read a record. Raise `KeyError` if there is none with this ID."""
        path = self._find(record_id)
        if path is None:
            raise KeyError(record_id)
        return Record.model_validate_json(path.read_bytes())

    def write_stage(
        self, record_id: str, stage: Stage, output: StageOutput, status: Status | None = None
    ) -> Record:
        """Add one stage's output, and optionally set the status.

        Raise `StageAlreadyWritten` if the record already holds output for
        `stage`, and `FinalStatus` if `status` would change a final status. On
        either error the file is left unchanged. Return the updated record.
        """
        record = self.load(record_id)
        if stage in record.stages:
            raise StageAlreadyWritten(f"{record_id}: {stage}")
        new_status = record.status if status is None else status
        if record.status in FINAL_STATUSES and new_status is not record.status:
            raise FinalStatus(f"{record_id}: {record.status} is final, refused {new_status}")
        updated = Record.model_validate(
            {
                **record.model_dump(),
                "stages": {**record.stages, stage: output},
                "status": new_status,
            }
        )
        directory = self._records_dir(record.run_id)
        temp = self._write_temp(directory, updated)
        try:
            os.replace(temp, self._path(record.run_id, record.id))
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
        self._sync_dir(directory)
        return updated

    def list(self, run_id: str, status: Status | None = None) -> builtins.list[Record]:
        """Return a run's records sorted by ID, optionally only those with `status`."""
        directory = self._records_dir(_check_id(run_id))
        records = [
            Record.model_validate_json(path.read_bytes())
            for path in sorted(directory.glob("*.json"))
            if not path.name.startswith(".")
        ]
        return [r for r in records if status is None or r.status is status]
