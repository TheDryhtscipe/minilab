"""The falsifier stage: attack a record, and write `killed` or `survived`.

It takes a `proposed` record whose encoding stage agreed (decision doc-6ja),
rebuilds the two families' encodings as `Sandboxed`, and:

1. attacks the first encoding directly (`attack`);
2. with no kill, asks for up to `max_reformulations` reformulations
   (`reformulate_and_attack`);
3. with a kill, asks a checker family for a mechanism (`check_mechanism`).

Every certificate, direct, reformulated, or from a mechanism's instances, is
decided by `check_kill` with both encodings (decision doc-j6t).

It writes `killed` with the certificate file, or `survived` with the coverage
file, under `runs/<run_id>/falsifier/`. A model failure (`ModelUnavailable`,
`BudgetExceeded`), an unusable sandbox, or a search that could evaluate no
point at all raises before the stage is written, so the record's status is
unchanged and the stage can run again.
"""

import random
from collections.abc import Mapping
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict, JsonValue

from minilab.certify import CERTIFICATE, check_kill
from minilab.encoding.sandbox import DEFAULT_TIMEOUT_S, Sandboxed
from minilab.falsify.mechanism import MechanismResult, check_mechanism
from minilab.falsify.reformulate import CHECK_SAMPLES, reformulate_and_attack
from minilab.falsify.search import AttackResult, SearchBudget, attack
from minilab.ledger import Ledger
from minilab.models.adapter import Budget, NoChecker
from minilab.models.backends import Backend
from minilab.records import Family, Record, Stage, StageOutput, Status


class SearchFailed(Exception):
    """The search evaluated no valid point, so it has no coverage to report."""


class FalsifierResult(BaseModel):
    """What the falsifier stage did to one record."""

    model_config = ConfigDict(frozen=True)

    status: Status
    attack: AttackResult
    mechanism: MechanismResult | None


def _encodings(record: Record, run_dir: Path, timeout_s: float) -> tuple[Sandboxed, Sandboxed]:
    """The two agreed encodings from the record's encoding stage, as `Sandboxed`."""
    output = record.stages.get(Stage.encoding)
    if output is None or output.data.get("agreed") is not True:
        raise ValueError(f"{record.id} has no agreed encoding stage")
    families = output.data.get("families")
    if not isinstance(families, list) or len(families) != 2 or len(output.evidence) != 2:
        raise ValueError(f"{record.id}'s encoding stage does not name two encodings")
    made: list[Sandboxed] = []
    for family, name in zip(families, output.evidence, strict=True):
        path = (run_dir / name).resolve()
        if not isinstance(family, str) or not path.is_relative_to(run_dir.resolve()):
            raise ValueError(f"{record.id}'s encoding stage names an encoding it cannot hold")
        made.append(
            Sandboxed(
                source=path.read_text(encoding="utf-8"), family=Family(family), timeout_s=timeout_s
            )
        )
    return made[0], made[1]


def run_falsifier_stage(
    record_id: str,
    *,
    ledger: Ledger,
    budget: Budget,
    backends: Mapping[Family, Backend] | None = None,
    search: SearchBudget | None = None,
    max_reformulations: int = 3,
    checks: int = CHECK_SAMPLES,
    seed: int = 0,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> FalsifierResult:
    """Attack a record and write its falsifier stage, as the module describes.

    The search is seeded from the record's id and `seed`, so a rerun searches
    the same points. Raise ValueError for a record that is not `proposed`,
    already has a falsifier stage, or has no agreed encodings.
    """
    store = ledger.store
    record = store.load(record_id)
    if record.status is not Status.proposed:
        raise ValueError(f"{record_id} is {record.status}; only a proposed record is falsified")
    if Stage.falsifier in record.stages:
        raise ValueError(f"{record_id} already has a falsifier stage")
    run_dir = store.root / "runs" / record.run_id
    first, second = _encodings(record, run_dir, timeout_s)
    assumptions = len(record.assumptions)
    certify = partial(check_kill, first, second, assumptions=assumptions)
    rng = random.Random(f"{record.id}:{seed}")
    plan = search or SearchBudget()
    result = attack(first, plan, rng, assumptions=assumptions, certify=certify)
    notes: list[str] = []
    if not result.killed:
        try:
            result = reformulate_and_attack(
                record,
                first,
                certify=certify,
                ledger=ledger,
                budget=budget,
                backends=backends,
                search=plan,
                rng=rng,
                max_reformulations=max_reformulations,
                checks=checks,
                direct=result,
                original_source=first.source,
                timeout_s=timeout_s,
            )
        except NoChecker as exc:
            notes.append(f"No reformulation: {exc}.")
    mechanism: MechanismResult | None = None
    if result.certificate is not None:
        try:
            mechanism = check_mechanism(
                record,
                result.certificate,
                make=first,
                certify=certify,
                ledger=ledger,
                budget=budget,
                backends=backends,
                rng=rng,
                original_source=first.source,
                timeout_s=timeout_s,
            )
        except NoChecker as exc:
            notes.append(f"No mechanism: {exc}.")
    elif result.coverage.evaluations == 0:
        raise SearchFailed(f"{record_id}: the search evaluated no valid point")

    directory = run_dir / "falsifier"
    directory.mkdir(parents=True, exist_ok=True)
    attempts = [
        cast(dict[str, JsonValue], a.model_dump(mode="json")) for a in result.reformulations
    ]
    data: dict[str, JsonValue] = {
        "families": [first.family.value, second.family.value],
        "candidates": result.candidates,
        "rejected": cast(list[JsonValue], result.rejected),
        "reformulations": cast(list[JsonValue], attempts),
        "coverage": cast(dict[str, JsonValue], result.coverage.model_dump(mode="json")),
    }
    if result.certificate is not None:
        status = Status.killed
        name = f"{record.id}.certificate.json"
        (directory / name).write_bytes(CERTIFICATE.dump_json(result.certificate, indent=2))
        evidence = [f"falsifier/{name}"]
        data["certificate"] = cast(
            dict[str, JsonValue], CERTIFICATE.dump_python(result.certificate, mode="json")
        )
        data["check"] = result.check.reason if result.check is not None else None
        data["mechanism"] = (
            cast(dict[str, JsonValue], mechanism.model_dump(mode="json"))
            if mechanism is not None
            else None
        )
    else:
        status = Status.survived
        name = f"{record.id}.coverage.json"
        (directory / name).write_text(result.coverage.model_dump_json(indent=2), encoding="utf-8")
        evidence = [f"falsifier/{name}"]
    store.write_stage(
        record.id,
        Stage.falsifier,
        StageOutput(
            written_at=datetime.now(UTC), data=data, evidence=evidence, notes=" ".join(notes)
        ),
        status=status,
    )
    return FalsifierResult(status=status, attack=result, mechanism=mechanism)
