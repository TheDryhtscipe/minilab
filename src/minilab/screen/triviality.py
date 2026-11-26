"""The triviality screen: is the formalised conjecture easy to prove?

Build plan doc-5wh, Task 12; design doc-gmj, stage 6. The hold rule is bead
minilab-224a47.

`screen` tries, in order, and stops at the first proof Lean accepts:

1. five Mathlib tactics (`TACTICS`), each alone as the whole proof. `exact?`
   searches the library for one lemma that closes the goal; `positivity`
   runs after `intros`. `polyrith` is never used: it calls a web service.
2. three proof attempts by the local model (`Family.local`), each retry
   shown its last proof and Lean's errors. They are skipped, with a note, if
   the run has no local model.
3. one checker call (`pick_checker`, never the formaliser's family), asked
   whether the theorem is a corollary of a named Mathlib theorem, with a
   proof. A claim proves nothing until Lean checks the proof.

Every candidate proof goes into the formaliser's own file in place of its
`sorry` (`with_proof`) and is checked by `formal_from_source` in proved mode:
the same text gate, command check, probe, and axiom check as the formaliser
stage, and no `sorry` anywhere. A candidate counts only if the proved file
also matches the original: the same definitions (as Lean printed them),
binders, conclusion, `open`s, statement, and theorems (leaving out the
equation lemmas Lean makes for a definition when a proof uses it). Lean
fixes `conjecture`'s type before it reads the proof, so nothing a proof adds
can change what was proved; this comparison is a second guard, and keeps
the proved file the one the formaliser stage checked, with no new
definition or theorem after the proof. The prompts show what Lean
printed (`lean_view`), never the model-written source.

A Lean time-out on a candidate is that candidate's outcome (`timed_out`),
not an error. A model failure (`ModelUnavailable`) or a spent budget
(`BudgetExceeded`) raises.

`near_misses` (which the scorer counts, `score.py`) names each screen that
nearly succeeded: a tactic or model proof that timed out, and a model proof
that compiled with exactly one Lean error. A proof the text gate or the
command check refused (a parse error among them) is `refused` and is not a
near miss. Each name appears once.

`run_triviality_stage` runs the screen on a formalised survivor and writes
the record's triviality stage. A proof sets the status `trivial`, unless
the formaliser stage flagged the formalisation (`hold_reasons`): then the
record stays `survived`, held for the user's review with the proof and the
reasons (user decision on bead minilab-a32a1c). The prover (Task 13, bead
minilab-880beb) must skip a held record; nothing before it does.
"""

import json
import re
import textwrap
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError

from minilab.formalise.audit import AuditResult
from minilab.formalise.formaliser import (
    DEFAULT_LEAN_TIMEOUT_S,
    Binder,
    FormalOutput,
    LeanTimeout,
    extract_lean,
    fenced,
    formal_from_source,
    lean_view,
    sorry_proof_span,
    strip_comments,
)
from minilab.formalise.vacuity import VacuityResult
from minilab.ledger import Ledger
from minilab.models.adapter import Budget, NoChecker, Role, call, pick_checker
from minilab.models.backends import Backend
from minilab.records import Family, Record, Stage, StageOutput, Status

TACTICS: tuple[str, ...] = ("exact?", "aesop", "simp", "positivity", "nlinarith")
"""The tactic screens, in the order they run. Never `polyrith`."""

_TACTIC_TEXT: dict[str, str] = {"positivity": "intros\npositivity"}
"""The proof text of a screen whose name is not its whole proof."""

STARTUP_ALLOWANCE_S = 60.0
"""Seconds added to a tactic's time limit for Lean to start and load Mathlib."""

LOCAL_ATTEMPTS = 3

Outcome = Literal["proved", "failed", "timed_out", "refused", "no_proof"]

_LEAN_ERROR = re.compile(r"^\d+:\d+: ")


class ScreenAttempt(BaseModel):
    """One try at a proof: which screen, what happened, and why."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    screen: str
    outcome: Outcome
    detail: str = ""


class ScreenResult(BaseModel):
    """What the screen found.

    `proved_by` is a tactic's name, `local`, or `corollary`. `proof` is the
    proof text that went in place of the `sorry`, and `source` the whole
    proved file. `corollary_of` is the theorem the checker named, when its
    proof is the one that worked.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    proved: bool
    proved_by: str | None = None
    proof: str | None = None
    source: str | None = None
    corollary_of: str | None = None
    attempts: list[ScreenAttempt]
    near_misses: list[str]
    notes: list[str]


def with_proof(source: str, tactics: str) -> str:
    """`source` without comments, with `conjecture`'s proof replaced by `tactics`.

    Everything from the `:=` to the end of the one `sorry` becomes `:= by`
    and the tactic lines, dedented and indented two spaces. Comments are
    removed (Lean checks the file without them anyway); the rest is kept as
    it was. Raise ValueError if the file has no such `sorry`.
    """
    text = strip_comments(source)
    span = sorry_proof_span(source)
    if span is None:
        raise ValueError("the file has no `conjecture` proved by one `sorry`")
    body = textwrap.dedent(tactics).strip("\n")
    lines = "\n".join(f"  {line}" if line.strip() else "" for line in body.splitlines())
    return text[: span[0]] + ":= by\n" + lines + text[span[1] :]


_GENERATED = re.compile(
    r"(?:match_\d+\.)?(?:eq_\d+|eq_def|eq_unfold|congr_eq_\d+|induct|mutual_induct"
    r"|induct_unfolding|fun_cases|fun_cases_unfolding)"
)
"""The last parts of the equation and induction lemmas Lean makes on demand for
a definition, the first time a proof uses them (`simp [g]` makes `g.eq_1`).
Lean reserves these names, so a file cannot declare one."""


def _declared(formal: FormalOutput) -> list[str]:
    """The file's theorems, without the lemmas Lean made for its definitions."""
    return [
        name
        for name in formal.theorems
        if not (
            any(
                name.startswith(d + ".") and _GENERATED.fullmatch(name[len(d) + 1 :])
                for d in formal.definitions
            )
        )
    ]


def _mismatch(original: FormalOutput, proved: FormalOutput) -> list[str]:
    """The parts of the statement that differ between the two files."""

    def binders(formal: FormalOutput) -> list[tuple[str, str, bool, bool]]:
        return [(b.name, b.type, b.prop, b.instance) for b in formal.binders]

    differ: list[str] = []
    if original.definitions != proved.definitions or original.types != proved.types:
        differ.append("definitions")
    if original.definition_text != proved.definition_text:
        differ.append("definition text")
    if binders(original) != binders(proved):
        differ.append("binders")
    if (
        original.conclusion != proved.conclusion
        or original.conclusion_truncated != proved.conclusion_truncated
    ):
        differ.append("conclusion")
    if original.opens != proved.opens:
        differ.append("opens")
    if original.lean_statement != proved.lean_statement:
        differ.append("statement")
    if _declared(original) != _declared(proved):
        differ.append("theorems")
    return differ


class _Checked(BaseModel):
    model_config = ConfigDict(frozen=True)

    outcome: Outcome
    detail: str
    source: str | None = None
    lean_errors: int = 0


def _check(
    original: FormalOutput, tactics: str, timeout_s: float, cache: dict[str, _Checked]
) -> _Checked:
    """Check one candidate proof in Lean, as the module describes."""
    if not tactics.strip():
        return _Checked(outcome="no_proof", detail="no proof was given")
    try:
        candidate = with_proof(original.source, tactics)
    except ValueError as exc:
        return _Checked(outcome="refused", detail=str(exc))
    cached = cache.get(candidate)
    if cached is not None and cached.outcome != "timed_out":
        return cached
    try:
        out = formal_from_source(
            candidate, original.family, original.model, timeout_s=timeout_s, proved=True
        )
    except LeanTimeout as exc:
        checked = _Checked(outcome="timed_out", detail=str(exc))
    else:
        if out.ok:
            differ = _mismatch(original, out)
            if differ:
                checked = _Checked(
                    outcome="refused",
                    detail="the proved file does not state the same theorem: "
                    + ", ".join(differ)
                    + " differ",
                )
            else:
                checked = _Checked(outcome="proved", detail="", source=candidate)
        elif not out.conclusion:
            # The text gate or the command check refused it before the probe ran.
            checked = _Checked(outcome="refused", detail="; ".join(out.problems)[:2000])
        else:
            errors = sum(1 for p in out.problems if _LEAN_ERROR.match(p))
            checked = _Checked(
                outcome="failed", detail="; ".join(out.problems)[:2000], lean_errors=errors
            )
    cache[candidate] = checked
    return checked


def _prove_prompt(formal: FormalOutput, last: tuple[str, str] | None) -> str:
    prompt = f"""Prove this Lean 4 theorem. The file starts with `import Mathlib`, and the
definitions below are part of it.

{fenced(lean_view(formal))}

Reply with one ```lean code block that holds only the tactic proof: the
lines that go after `:= by`. Do not restate the theorem or write any other
declaration. Do not use `sorry`, `native_decide`, `set_option`, or comments."""
    if last is not None:
        proof, errors = last
        prompt += f"""

Your previous proof was:

{fenced(proof)}

Lean refused it:

{fenced(errors, "")}

Write a corrected proof."""
    return prompt


def _corollary_prompt(formal: FormalOutput) -> str:
    return f"""Is this Lean 4 theorem a direct corollary of a named theorem in Mathlib?
The file starts with `import Mathlib`, and the definitions below are part of it.

{fenced(lean_view(formal))}

Answer with one JSON object and nothing else:
{{"corollary": true or false,
 "theorem": "the full Mathlib name of the theorem, or empty",
 "proof": "a Lean 4 tactic proof that uses it (the lines after `:= by`), or empty"}}

Say true only if the theorem follows from that one named theorem in a few
steps. Do not use `sorry` or `native_decide` in the proof."""


class _CorollaryReply(BaseModel):
    model_config = ConfigDict(extra="ignore")

    corollary: bool
    theorem: str = ""
    proof: str = ""


_JSON_BLOCK = re.compile(r"```(?:json)?[ \t]*\n(.*?)\n?```", re.DOTALL)


def _parse_corollary(text: str) -> _CorollaryReply | None:
    candidates = [m.group(1) for m in _JSON_BLOCK.finditer(text)]
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            return _CorollaryReply.model_validate(json.loads(candidate))
        except (json.JSONDecodeError, ValidationError):
            continue
    return None


def _model_proof(text: str) -> str:
    """The tactic proof in a model's reply: its first Lean block, without a leading `by`."""
    proof = extract_lean(text).strip("\n")
    stripped = proof.lstrip()
    if re.match(r"by(?![\w'!?])", stripped):
        proof = stripped[2:]
    return proof


def screen(
    formal: FormalOutput,
    *,
    record: Record,
    ledger: Ledger,
    budget: Budget,
    backends: Mapping[Family, Backend] | None = None,
    tactic_timeout_s: float = 60.0,
    lean_timeout_s: float = DEFAULT_LEAN_TIMEOUT_S,
) -> ScreenResult:
    """Try to prove `formal` cheaply, as the module describes.

    `formal` must be an `ok` formalisation. Each tactic gets
    `tactic_timeout_s` plus `STARTUP_ALLOWANCE_S`; each model proof gets
    `lean_timeout_s`. Raise ValueError if `formal` is not `ok`, and what
    `call` raises (`ModelUnavailable`, `BudgetExceeded`, ...).
    """
    if not formal.ok:
        raise ValueError("only a formalisation that checked can be screened")
    attempts: list[ScreenAttempt] = []
    notes: list[str] = []
    misses: list[str] = []
    cache: dict[str, _Checked] = {}

    def miss(name: str) -> None:
        if name not in misses:
            misses.append(name)

    def result(
        proved_by: str | None = None,
        proof: str | None = None,
        source: str | None = None,
        corollary_of: str | None = None,
    ) -> ScreenResult:
        return ScreenResult(
            proved=proved_by is not None,
            proved_by=proved_by,
            proof=proof,
            source=source,
            corollary_of=corollary_of,
            attempts=attempts,
            near_misses=misses,
            notes=notes,
        )

    for name in TACTICS:
        text = _TACTIC_TEXT.get(name, name)
        checked = _check(formal, text, tactic_timeout_s + STARTUP_ALLOWANCE_S, cache)
        attempts.append(ScreenAttempt(screen=name, outcome=checked.outcome, detail=checked.detail))
        if checked.outcome == "timed_out":
            miss(name)
        if checked.outcome == "proved":
            return result(name, text, checked.source)

    families = ledger.manifest.families
    if Family.local in families:
        last: tuple[str, str] | None = None
        for _ in range(LOCAL_ATTEMPTS):
            reply = call(
                Family.local,
                Role.screen,
                _prove_prompt(formal, last),
                record_id=record.id,
                pair=record.pair,
                ledger=ledger,
                budget=budget,
                backends=backends,
            )
            proof = _model_proof(reply.text)
            checked = _check(formal, proof, lean_timeout_s, cache)
            attempts.append(
                ScreenAttempt(screen="local", outcome=checked.outcome, detail=checked.detail)
            )
            if checked.outcome == "proved":
                return result("local", proof, checked.source)
            if checked.outcome == "timed_out" or checked.lean_errors == 1:
                miss("local")
            last = (proof, checked.detail or "no proof was given")
    else:
        notes.append(
            f"the run has no local model, so its {LOCAL_ATTEMPTS} proof attempts were skipped"
        )

    try:
        checker = pick_checker(formal.family, families)
    except NoChecker as exc:
        notes.append(f"no corollary check: {exc}")
        return result()
    reply = call(
        checker,
        Role.screen,
        _corollary_prompt(formal),
        record_id=record.id,
        pair=record.pair,
        ledger=ledger,
        budget=budget,
        backends=backends,
    )
    claim = _parse_corollary(reply.text)
    if claim is None:
        attempts.append(
            ScreenAttempt(
                screen="corollary", outcome="no_proof", detail="the reply could not be read"
            )
        )
    elif not claim.corollary or not claim.proof.strip():
        attempts.append(
            ScreenAttempt(
                screen="corollary",
                outcome="no_proof",
                detail=f"the {checker.value} checker claimed no corollary with a proof",
            )
        )
    else:
        proof = _model_proof(claim.proof)
        checked = _check(formal, proof, lean_timeout_s, cache)
        attempts.append(
            ScreenAttempt(screen="corollary", outcome=checked.outcome, detail=checked.detail)
        )
        if checked.outcome == "proved":
            return result("corollary", proof, checked.source, claim.theorem.strip() or None)
        if checked.outcome == "timed_out" or checked.lean_errors == 1:
            miss("corollary")
    return result()


# --- the hold rule (bead minilab-224a47) ---


_BINDERS: TypeAdapter[list[Binder]] = TypeAdapter(list[Binder])


def _complete(raw: JsonValue, model: type[BaseModel]) -> bool:
    """Whether `raw` is an object that gives every field of `model`, defaults included."""
    return isinstance(raw, dict) and set(model.model_fields) <= raw.keys()


def hold_reasons(formaliser_data: Mapping[str, JsonValue]) -> list[str]:
    """Why a proof of this formalisation must be held for review, not set trivial.

    Reads the formaliser stage's `audit`, `vacuity`, and `weakenings`, and
    works from their raw fields, not only from their summaries. A reason is:
    any hypothesis the audit did not judge `matches` (a weakening among
    them), any hidden condition, any audit reason or rejection, an
    unreadable checker reply; contradictory or unused binders, a vacuity
    check that did not finish, and any doubt about the falsifier's witness
    (it failed a hypothesis, left one undecided or not instantiated, gave no
    value to a variable, or checked no hypothesis at all). A theorem with no
    hypotheses has nothing to check at the witness, so for it an empty
    witness list is not a reason (user decision on bead minilab-a32a1c);
    whether it has hypotheses is read from the binders Lean printed
    (`binders`), never from the linter's list. It fails closed:
    `audit` or `vacuity` that does not parse, or that leaves out a field
    (so a default would stand in for it), is a reason too. Empty if the
    formalisation is clean.
    """
    reasons: list[str] = []
    try:
        binders = _BINDERS.validate_python(formaliser_data.get("binders"))
    except ValidationError:
        reasons.append("the formaliser stage's binders could not be read")
        hypotheses = True
    else:
        hypotheses = any(b.prop for b in binders)
    top = formaliser_data.get("weakenings")
    if top != []:
        reasons.append(f"the formaliser stage lists weakenings: {json.dumps(top)[:500]}")
    raw_audit = formaliser_data.get("audit")
    try:
        if not _complete(raw_audit, AuditResult):
            raise ValueError("a field is missing")
        audit = AuditResult.model_validate(raw_audit)
    except (ValidationError, ValueError):
        reasons.append("the formaliser stage's audit could not be read")
    else:
        reasons += [f"the audit found a weakening: {w}" for w in audit.weakenings]
        reasons += [
            f"the audit found a hidden condition ({c.kind}) in {c.where}: {c.condition}"
            for c in audit.hidden
        ]
        reasons += [
            f"the audit judged hypothesis {h.label} ({h.type}) {h.kind}"
            for h in audit.hypotheses
            if h.kind != "matches"
        ]
        if audit.extra:
            reasons.append("the audit lists extra hypotheses: " + ", ".join(audit.extra))
        reasons += [f"the audit says: {r}" for r in audit.reasons]
        if not audit.judged:
            reasons.append("the audit's checker reply could not be read")
        if audit.rejected:
            reasons.append("the audit rejected the formalisation")
    raw_vacuity = formaliser_data.get("vacuity")
    try:
        if not _complete(raw_vacuity, VacuityResult):
            raise ValueError("a field is missing")
        vacuity = VacuityResult.model_validate(raw_vacuity)
    except (ValidationError, ValueError):
        reasons.append("the formaliser stage's vacuity check, with its witness, could not be read")
    else:
        if not vacuity.checked or vacuity.timed_out:
            reasons.append("the vacuity check did not finish, so the witness was not checked")
        if vacuity.contradictory:
            reasons.append("the hypotheses contradict each other")
        if vacuity.unused:
            reasons.append(
                "binders that cannot bear on the conclusion: " + ", ".join(vacuity.unused)
            )
        if hypotheses and not vacuity.satisfied_by_witness:
            reasons.append("the falsifier's witness does not satisfy the hypotheses")
        reasons += [
            f"the witness check of {w.label} is {w.state}"
            for w in vacuity.witness
            if w.state != "holds"
        ]
        if vacuity.unwitnessed:
            reasons.append("the witness gave no value to: " + ", ".join(vacuity.unwitnessed))
        if hypotheses and not vacuity.witness:
            reasons.append("no hypothesis was checked at the witness")
    return reasons


# --- the stage ---


def _unscreened(reasons: list[str]) -> dict[str, JsonValue]:
    return {
        "screened": False,
        "trivial": False,
        "held": False,
        "hold_reasons": [],
        "reasons": cast(list[JsonValue], reasons),
        "proved_by": None,
        "proof": None,
        "corollary_of": None,
        "near_misses": [],
        "attempts": [],
    }


def _formal(
    record: Record, data: Mapping[str, JsonValue], root: Path, timeout_s: float
) -> FormalOutput | str:
    """The formaliser's file, probed again; or why its statement cannot be trusted."""
    path = root / "runs" / record.run_id / "formaliser" / f"{record.id}.lean"
    if not path.is_file():
        return f"the formaliser's Lean file {path.name} is missing, so its statement is unknown"
    family, model, statement = data.get("family"), data.get("model"), data.get("lean_statement")
    if not isinstance(family, str) or family not in Family.__members__:
        return "the formaliser stage names no family, so its statement cannot be checked"
    formal = formal_from_source(
        path.read_text(encoding="utf-8"),
        Family(family),
        model if isinstance(model, str) else "",
        timeout_s=timeout_s,
    )
    if not formal.ok:
        return (
            "the formaliser's Lean file no longer checks, so its statement is unknown: "
            + ("; ".join(formal.problems)[:1000])
        )
    if formal.lean_statement != statement:
        return "the Lean file's statement differs from the one the formaliser stage recorded"
    return formal


def run_triviality_stage(
    record_id: str,
    *,
    ledger: Ledger,
    budget: Budget,
    backends: Mapping[Family, Backend] | None = None,
    tactic_timeout_s: float = 60.0,
    lean_timeout_s: float = DEFAULT_LEAN_TIMEOUT_S,
) -> ScreenResult | None:
    """Screen a formalised survivor and write its triviality stage.

    Raise ValueError for a record that is not `survived` with a formaliser
    stage, or that already has a triviality stage. A rejected formalisation,
    or a Lean file whose statement no longer matches the formaliser stage's,
    is not screened: the stage says why (`screened` false) and nothing else
    runs. Otherwise the stage records the screen's result; see the module
    for the status it sets. The proved file goes to
    `runs/<run_id>/triviality/<id>.lean`.

    The stage's data: `screened`, `trivial`, `held`, `hold_reasons`,
    `reasons`, `proved_by`, `proof`, `corollary_of`, `near_misses`, and
    `attempts`. A model failure or a Lean time-out on the re-check raises,
    leaving the record unchanged (the ledger still holds the charge of
    each model call that finished). Return the screen's result, or None if
    nothing was screened.
    """
    store = ledger.store
    record = store.load(record_id)
    formaliser = record.stages.get(Stage.formaliser)
    if record.status is not Status.survived or formaliser is None:
        raise ValueError(
            f"{record_id} is {record.status}; only a survived record with a formaliser "
            "stage is screened"
        )
    if Stage.triviality in record.stages:
        raise ValueError(f"{record_id} already has a triviality stage")

    data = formaliser.data
    now = datetime.now(UTC)
    rejected = data.get("rejected")
    if rejected is not False:
        why = (
            "the formaliser stage rejected the formalisation: "
            + "; ".join(str(r) for r in cast(list[JsonValue], data.get("reasons") or []))
            if rejected is True
            else "the formaliser stage does not say the formalisation was accepted, "
            "so it is treated as rejected"
        )
        store.write_stage(
            record.id, Stage.triviality, StageOutput(written_at=now, data=_unscreened([why]))
        )
        return None

    formal = _formal(record, data, store.root, lean_timeout_s)
    if isinstance(formal, str):
        store.write_stage(
            record.id, Stage.triviality, StageOutput(written_at=now, data=_unscreened([formal]))
        )
        return None

    found = screen(
        formal,
        record=record,
        ledger=ledger,
        budget=budget,
        backends=backends,
        tactic_timeout_s=tactic_timeout_s,
        lean_timeout_s=lean_timeout_s,
    )
    holds = hold_reasons(data)
    trivial = found.proved and not holds
    held = found.proved and bool(holds)
    evidence: list[str] = []
    if found.source is not None:
        directory = store.root / "runs" / record.run_id / "triviality"
        directory.mkdir(parents=True, exist_ok=True)
        name = f"{record.id}.lean"
        (directory / name).write_text(found.source, encoding="utf-8")
        evidence.append(f"triviality/{name}")
    out: dict[str, JsonValue] = {
        "screened": True,
        "trivial": trivial,
        "held": held,
        "hold_reasons": cast(list[JsonValue], holds),
        "reasons": [],
        "proved_by": found.proved_by,
        "proof": found.proof,
        "corollary_of": found.corollary_of,
        "near_misses": cast(list[JsonValue], found.near_misses),
        "attempts": cast(list[JsonValue], [a.model_dump(mode="json") for a in found.attempts]),
    }
    store.write_stage(
        record.id,
        Stage.triviality,
        StageOutput(
            written_at=datetime.now(UTC), data=out, evidence=evidence, notes=" ".join(found.notes)
        ),
        status=Status.trivial if trivial else None,
    )
    return found
