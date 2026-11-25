"""The formaliser stage: formalise a survivor, lint it, read it back, and audit it.

Build plan doc-5wh, Task 11; design doc-gmj, stage 5.

It takes a `survived` record that has its falsifier and literature stages,
and:

1. asks a model for the Lean file (`formalise`, one retry with Lean's
   feedback);
2. runs the vacuity linter with the falsifier's worst point as the witness;
3. has another family translate the Lean back to English (`roundtrip`);
4. audits the hypotheses against the record (`audit`).

It writes the Lean source to `runs/<run_id>/formaliser/<id>.lean` and the
stage's data: `rejected`, `reasons`, `family`, `model`, `attempts`,
`lean_statement` (None when rejected, so the scorer falls back to the
record's own statement), `roundtrip` (the English text, which the queue
shows), `roundtrip_family`, `audit`, `weakenings`, `vacuity`, `binders`, and
`problems`.

A rejected formalisation does not change the status: the record stays
`survived`, and the stage's data says it was rejected and why (decision on
bead minilab-ec2916). If the file never compiled, the stage records that,
with no round trip or audit.

A model failure, a missing checker, or a Lean time-out raises before the
stage is written, so the record is unchanged and the stage can run again.
"""

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import cast

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from minilab.encoding.api import Instance
from minilab.formalise.audit import AuditResult, audit
from minilab.formalise.formaliser import DEFAULT_LEAN_TIMEOUT_S, LeanTimeout, formalise
from minilab.formalise.roundtrip import RoundTrip, roundtrip
from minilab.formalise.vacuity import VacuityResult, vacuity
from minilab.ledger import Ledger
from minilab.models.adapter import Budget
from minilab.models.backends import Backend
from minilab.records import Family, Record, Stage, StageOutput, Status


class FormaliserStageResult(BaseModel):
    """What the stage found, beyond what it wrote to the record."""

    model_config = ConfigDict(frozen=True)

    rejected: bool
    reasons: list[str]
    vacuity: VacuityResult | None
    roundtrip: RoundTrip | None
    audit: AuditResult | None


def _witness(record: Record) -> tuple[Instance | None, str]:
    """The falsifier's worst point, if the coverage report has a usable one."""
    output = record.stages.get(Stage.falsifier)
    coverage = output.data.get("coverage") if output is not None else None
    point = coverage.get("worst_point") if isinstance(coverage, dict) else None
    if point is None:
        return None, "The falsifier gave no worst point, so no witness was instantiated."
    try:
        return Instance.model_validate(point), ""
    except ValidationError as exc:
        return None, f"The falsifier's worst point is not an instance: {str(exc)[:200]}"


def _json(model: VacuityResult | AuditResult) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], model.model_dump(mode="json"))


def run_formaliser_stage(
    record_id: str,
    *,
    ledger: Ledger,
    budget: Budget,
    backends: Mapping[Family, Backend] | None = None,
    lean_timeout_s: float = DEFAULT_LEAN_TIMEOUT_S,
) -> FormaliserStageResult:
    """Formalise a survivor and write its formaliser stage, as the module describes.

    Raise ValueError for a record that is not `survived` with falsifier and
    literature stages, or that already has a formaliser stage.
    """
    store = ledger.store
    record = store.load(record_id)
    if (
        record.status is not Status.survived
        or Stage.falsifier not in record.stages
        or Stage.literature not in record.stages
    ):
        raise ValueError(
            f"{record_id} is {record.status}; only a survived record with falsifier and "
            "literature stages is formalised"
        )
    if Stage.formaliser in record.stages:
        raise ValueError(f"{record_id} already has a formaliser stage")

    formal = formalise(
        record, ledger=ledger, budget=budget, backends=backends, timeout_s=lean_timeout_s
    )
    notes: list[str] = []
    vac: VacuityResult | None = None
    trip: RoundTrip | None = None
    audited: AuditResult | None = None
    reasons: list[str] = []
    if not formal.ok:
        reasons.append(
            f"the formalisation did not compile after {formal.attempts} attempt(s): "
            + "; ".join(formal.problems)[:2000]
        )
    else:
        witness, note = _witness(record)
        if note:
            notes.append(note)
        vac = vacuity(formal, witness, timeout_s=lean_timeout_s)
        if vac.timed_out:
            raise LeanTimeout(f"{record_id}: the vacuity check timed out")
        trip = roundtrip(
            formal, formal.family, record=record, ledger=ledger, budget=budget, backends=backends
        )
        audited = audit(
            record, formal, ledger=ledger, budget=budget, backends=backends, vacuity=vac
        )
        reasons += vac.reasons + audited.reasons
    rejected = bool(reasons)

    run_dir = store.root / "runs" / record.run_id
    directory = run_dir / "formaliser"
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{record.id}.lean"
    (directory / name).write_text(formal.source, encoding="utf-8")
    data: dict[str, JsonValue] = {
        "rejected": rejected,
        "reasons": cast(list[JsonValue], reasons),
        "family": formal.family.value,
        "model": formal.model,
        "attempts": formal.attempts,
        "problems": cast(list[JsonValue], formal.problems),
        "lean_statement": None if rejected else formal.lean_statement,
        "roundtrip": trip.english if trip is not None else None,
        "roundtrip_family": trip.family.value if trip is not None else None,
        "audit": _json(audited) if audited is not None else None,
        "weakenings": cast(list[JsonValue], audited.weakenings if audited is not None else []),
        "vacuity": _json(vac) if vac is not None else None,
        "binders": cast(list[JsonValue], [b.model_dump(mode="json") for b in formal.binders]),
    }
    store.write_stage(
        record.id,
        Stage.formaliser,
        StageOutput(
            written_at=datetime.now(UTC),
            data=data,
            evidence=[f"formaliser/{name}"],
            notes=" ".join(notes),
        ),
    )
    return FormaliserStageResult(
        rejected=rejected, reasons=reasons, vacuity=vac, roundtrip=trip, audit=audited
    )
