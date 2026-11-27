"""The prover loop: a model writes Lean proofs with compiler feedback, under a budget.

Build plan doc-5wh, Task 13; design doc-gmj, stage 7; bead minilab-880beb.

`prove` asks the prover family for a proof of the formaliser's `conjecture`,
and Lean checks every reply. The prover is Claude if the run has it, else
Gemini, else Codex (`prover_family`, the triviality screen's order). Each
retry is shown its last reply and why it was not accepted: Lean's errors, the
reason the file was refused, or the audit's reasons.

The loop stops at the first accepted proof, or when its budget
(`ProverBudget`) is spent: `max_attempts` model calls (default 8) or
`max_tokens` tokens (default 200,000), counting the input and output of the
prover's own calls. The token cap is checked before each call, so the last
call can overrun it by its own tokens. The audit's calls are charged to the
pair's budget (`Budget`) but not to this cap.

A reply is read in one of two ways:

- a tactic proof (`proof_from_reply`) goes in place of the file's `sorry`
  (`with_proof`);
- a reply whose first Lean block restates `theorem conjecture` (or `lemma
  conjecture`) replaces the file's whole `conjecture` declaration with the
  restated one, from its header to the end of the block. This is how the
  model adds hypotheses. A restatement is refused when Lean's printed
  statement was cut (a binder type or the conclusion over the probe's length
  limit): the comparison below reads only the printed text, so two different
  statements could look the same.

Lean checks the candidate in proved mode (`formal_from_source` with
`proved=True`): the text gate, the command check, the probe, no `sorry`
anywhere, and no axiom outside Lean's standard three. The prover checks the
axioms again itself (`axiom_problems`). Then it compares the candidate's
statement with the original (`statement_mismatch`):

- the same statement: a proof of the conjecture as stated;
- the original binders, in order and unchanged, with only hypotheses added
  (`added_binders`), and nothing else changed: a proof of a weaker
  statement. It goes through the hypothesis audit (`audit`), whose checker
  is never the prover's family and never Gemini. If the audit rejects it, the
  attempt is `audit_rejected` and the loop goes on. If the audit accepts it,
  the loop stops with a weakened proof, which is held for review;
- anything else (a changed conclusion, definition, or binder, a dropped
  binder, an added declaration): `refused`.

A Lean time-out on a candidate is that attempt's outcome (`timed_out`), never
a verdict. If every attempt timed out, `run_prover_stage` raises `LeanTimeout`
and writes nothing, so the prover can run again. A model failure
(`ModelUnavailable`) or a spent pair budget (`BudgetExceeded`) raises.

`run_prover_stage` runs the loop on a screened survivor and writes the
record's prover stage. The status it sets (user decisions on bead
minilab-880beb, 2026-11-26):

- a proof as stated, of a formalisation the formaliser stage did not flag
  (`hold_reasons`): `proved`;
- a proof that needed added hypotheses, or a proof of a flagged
  formalisation: the record stays `survived`, held for the user's review
  with the proof and the reasons;
- no proof: the status does not change.

A record the triviality screen held (it found a proof, but the formalisation
was flagged) or did not screen is skipped: the stage says why, and no model
is called.
"""

import re
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from minilab.formalise.audit import AuditResult, audit
from minilab.formalise.formaliser import (
    DEFAULT_LEAN_TIMEOUT_S,
    SORRY_AXIOM,
    STANDARD_AXIOMS,
    Binder,
    FormalOutput,
    LeanTimeout,
    extract_lean,
    fenced,
    formal_from_source,
    lean_view,
    restated_proof,
    sorry_proof_span,
    strip_comments,
)
from minilab.ledger import Ledger
from minilab.models.adapter import Budget, NoChecker, Role, call
from minilab.models.backends import Backend
from minilab.records import Family, Record, Stage, StageOutput, Status
from minilab.screen.triviality import (
    frontier_prover,
    hold_reasons,
    proof_from_reply,
    statement_mismatch,
    stored_formal,
    with_proof,
)

__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_MAX_TOKENS",
    "ProofAttempt",
    "ProofResult",
    "ProverBudget",
    "added_binders",
    "axiom_problems",
    "prove",
    "prover_family",
    "run_prover_stage",
]

DEFAULT_MAX_ATTEMPTS = 8
"""Prover calls per conjecture (build plan doc-5wh, Task 13)."""

DEFAULT_MAX_TOKENS = 200_000
"""Tokens, input and output, of the prover's calls per conjecture."""

Outcome = Literal[
    "proved", "weakened", "failed", "timed_out", "refused", "no_proof", "audit_rejected"
]
Stopped = Literal["proved", "weakened", "attempts", "tokens", "no_prover"]

_DECLARATION = re.compile(r"(?<![\w.'!?])(?:theorem|lemma)\s+conjecture(?![\w'!?.])")


class ProverBudget(BaseModel):
    """The prover's budget for one conjecture."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_attempts: int = Field(default=DEFAULT_MAX_ATTEMPTS, gt=0)
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, gt=0)


class ProofAttempt(BaseModel):
    """One prover call and what became of its proof."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome: Outcome
    detail: str = ""
    proof: str = ""
    tokens: int = 0


class ProofResult(BaseModel):
    """What the prover found.

    `proved`: Lean accepted a proof of the conjecture as stated and nothing
    holds it. `held`: Lean accepted a proof, but it is held for review
    (`hold_reasons` say why), so the status stays `survived`. `weakened`: the
    proof needed `added_hypotheses` (each `label : type`), and
    `weakened_statement` is the statement Lean proved; `audit` is the audit
    of it. `source` is the whole proved file and `axioms` the axioms
    `conjecture` uses. `tokens` counts the prover's own calls.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    proved: bool
    held: bool
    weakened: bool = False
    hold_reasons: list[str] = []
    prover: Family | None
    model: str | None = None
    proof: str | None = None
    source: str | None = None
    axioms: list[str] = []
    added_hypotheses: list[str] = []
    weakened_statement: str | None = None
    audit: AuditResult | None = None
    attempts: list[ProofAttempt]
    tokens: int
    stopped: Stopped
    notes: list[str] = []


def prover_family(families: Iterable[Family | str]) -> Family | None:
    """The family that writes proofs: Claude, else Gemini, else Codex, else None."""
    return frontier_prover(families)


def axiom_problems(axioms: Iterable[str]) -> list[str]:
    """Why a proof with these axioms cannot be accepted. Empty if it can."""
    used = set(axioms)
    problems: list[str] = []
    if SORRY_AXIOM in used:
        problems.append("the proof depends on `sorry`")
    extra = sorted(used - STANDARD_AXIOMS - {SORRY_AXIOM})
    if extra:
        problems.append(f"the proof uses axioms beyond Lean's standard ones: {', '.join(extra)}")
    return problems


def added_binders(original: FormalOutput, proved: FormalOutput) -> list[Binder] | None:
    """The hypotheses `proved` adds to `original`'s binders, or None.

    None unless every original binder appears in `proved`, in order and
    with the same name, type, and kind, and every other binder of `proved`
    is a hypothesis (a proposition, not an instance binder).
    """

    def key(b: Binder) -> tuple[str, str, bool, bool, bool]:
        return (b.name, b.type, b.prop, b.instance, b.truncated)

    wanted = [key(b) for b in original.binders]
    matched = 0
    added: list[Binder] = []
    for b in proved.binders:
        if matched < len(wanted) and key(b) == wanted[matched]:
            matched += 1
        elif b.prop and not b.instance:
            added.append(b)
        else:
            return None
    return added if matched == len(wanted) else None


def _restated(formal: FormalOutput, block: str) -> str | None:
    """`formal`'s file with its `conjecture` replaced by the one in `block`, or None.

    None if `block` does not restate `conjecture`. Comments are removed from both.
    """
    if restated_proof(block) is None:
        return None
    code = strip_comments(block)
    found = _DECLARATION.search(code)
    if found is None:
        return None
    text = strip_comments(formal.source)
    header = _DECLARATION.search(text)
    span = sorry_proof_span(formal.source)
    if header is None or span is None:
        return None
    return text[: header.start()] + code[found.start() :].strip() + "\n" + text[span[1] :]


class _Checked(BaseModel):
    model_config = ConfigDict(frozen=True)

    outcome: Outcome
    detail: str
    formal: FormalOutput | None = None
    added: list[Binder] = []


def _check(
    original: FormalOutput, candidate: str, family: Family, model: str, timeout_s: float
) -> _Checked:
    """Check one candidate file in Lean, as the module describes."""
    try:
        out = formal_from_source(candidate, family, model, timeout_s=timeout_s, proved=True)
    except LeanTimeout as exc:
        return _Checked(outcome="timed_out", detail=str(exc))
    if not out.ok:
        # Without a conclusion, the text gate or the command check refused it.
        outcome: Outcome = "failed" if out.conclusion else "refused"
        return _Checked(outcome=outcome, detail="; ".join(out.problems)[:2000])
    bad = axiom_problems(out.axioms)
    if bad:
        return _Checked(outcome="refused", detail="; ".join(bad))
    differ = statement_mismatch(original, out)
    if not differ:
        return _Checked(outcome="proved", detail="", formal=out)
    added = added_binders(original, out)
    if added and set(differ) <= {"binders", "statement"}:
        return _Checked(outcome="weakened", detail="", formal=out, added=added)
    return _Checked(
        outcome="refused",
        detail="the proved file does not state the same theorem: "
        + ", ".join(differ)
        + " differ; only hypotheses may be added, with every binder, the conclusion, "
        "and the definitions kept as they are",
    )


def _prompt(formal: FormalOutput, last: tuple[str, str] | None) -> str:
    prompt = f"""Write a Lean 4 proof of this theorem. The file starts with `import Mathlib`,
and the definitions below are part of it.

{fenced(lean_view(formal))}

Reply with one ```lean code block that holds only the tactic proof: the
lines that go after `:= by`. Do not use `sorry`, `native_decide`,
`set_option`, or comments.

If the theorem is false or cannot be proved as stated, you may instead
restate it with extra hypotheses: reply with the whole `theorem conjecture`
declaration, with every binder and the conclusion exactly as above and in the
same order, only hypotheses added, and its proof after `:= by`. Write no
other declaration. A proof with added hypotheses is audited and held for
review; it is not a proof of the theorem as stated, so prove the theorem as
stated if you can."""
    if last is not None:
        shown, why = last
        prompt += f"""

Your previous reply was:

{fenced(shown)}

{why}

Write a corrected proof."""
    return prompt


def _why(checked: _Checked) -> str:
    if checked.outcome == "no_proof":
        return "It held no proof."
    if checked.outcome == "audit_rejected":
        return "The hypothesis audit rejected the added hypotheses:\n\n" + fenced(
            checked.detail, ""
        )
    return "Lean refused it:\n\n" + fenced(checked.detail, "")


def _cut(formal: FormalOutput) -> bool:
    """Whether Lean's printed statement of `formal` was cut, so two different
    statements could print the same."""
    return formal.conclusion_truncated or any(b.truncated for b in formal.binders)


def _candidate(formal: FormalOutput, block: str, proof: str) -> tuple[str | None, _Checked | None]:
    """The file to check for a reply, or why there is none."""
    restated = _restated(formal, block)
    if restated is not None:
        if _cut(formal):
            return None, _Checked(
                outcome="refused",
                detail="the statement is too long for its printed text to be compared in full, "
                "so it may not be restated; give only the tactic proof",
            )
        return restated, None
    if not proof.strip():
        return None, _Checked(outcome="no_proof", detail="no proof was given")
    try:
        return with_proof(formal.source, proof), None
    except ValueError as exc:
        return None, _Checked(outcome="refused", detail=str(exc))


def prove(
    record: Record,
    formal: FormalOutput,
    budget: ProverBudget,
    *,
    ledger: Ledger,
    pair_budget: Budget,
    backends: Mapping[Family, Backend] | None = None,
    lean_timeout_s: float = DEFAULT_LEAN_TIMEOUT_S,
) -> ProofResult:
    """Try to prove `formal` within `budget`, as the module describes.

    `formal` must be an `ok` formalisation. Every model call is charged to
    the record's pair under `pair_budget`. Here `held` is true only for a
    weakened proof; `run_prover_stage` adds the formaliser stage's flags.
    Raise ValueError if `formal` is not `ok`, and what `call` raises
    (`ModelUnavailable`, `BudgetExceeded`, ...).
    """
    if not formal.ok:
        raise ValueError("only a formalisation that checked can be proved")
    family = prover_family(ledger.manifest.families)
    if family is None:
        return ProofResult(
            proved=False,
            held=False,
            prover=None,
            attempts=[],
            tokens=0,
            stopped="no_prover",
            notes=["the run has neither Claude, Gemini nor Codex, so nothing tried a proof"],
        )
    attempts: list[ProofAttempt] = []
    tokens = 0
    last: tuple[str, str] | None = None
    checked_files: dict[str, _Checked] = {}
    audits: dict[str, AuditResult] = {}
    stopped: Stopped = "attempts"
    for _ in range(budget.max_attempts):
        if tokens >= budget.max_tokens:
            stopped = "tokens"
            break
        reply = call(
            family,
            Role.prove,
            _prompt(formal, last),
            record_id=record.id,
            pair=record.pair,
            ledger=ledger,
            budget=pair_budget,
            backends=backends,
        )
        spent = reply.tokens_in + reply.tokens_out
        tokens += spent
        block = extract_lean(reply.text)
        proof = proof_from_reply(reply.text)
        candidate, checked = _candidate(formal, block, proof)
        if candidate is not None:
            cached = checked_files.get(candidate)
            if cached is not None and cached.outcome != "timed_out":
                checked = cached
            else:
                checked = _check(formal, candidate, family, reply.model, lean_timeout_s)
                checked_files[candidate] = checked
        assert checked is not None
        if checked.outcome == "weakened" and candidate is not None and checked.formal is not None:
            weakened = checked.formal
            found = audits.get(candidate)
            if found is None:
                try:
                    found = audit(
                        record, weakened, ledger=ledger, budget=pair_budget, backends=backends
                    )
                except NoChecker as exc:
                    checked = _Checked(
                        outcome="audit_rejected", detail=f"no family can audit them: {exc}"
                    )
                else:
                    audits[candidate] = found
            if found is not None and found.rejected:
                checked = _Checked(outcome="audit_rejected", detail="; ".join(found.reasons))
            elif found is not None:
                added = [f"{b.label} : {b.type}" for b in checked.added]
                attempts.append(ProofAttempt(outcome="weakened", proof=proof, tokens=spent))
                return ProofResult(
                    proved=False,
                    held=True,
                    weakened=True,
                    hold_reasons=[f"the proof needs added hypotheses: {', '.join(added)}"]
                    + [f"the audit records a weakening: {w}" for w in found.weakenings],
                    prover=family,
                    model=reply.model,
                    proof=proof,
                    source=candidate,
                    axioms=weakened.axioms,
                    added_hypotheses=added,
                    weakened_statement=weakened.lean_statement,
                    audit=found,
                    attempts=attempts,
                    tokens=tokens,
                    stopped="weakened",
                )
        attempts.append(
            ProofAttempt(outcome=checked.outcome, detail=checked.detail, proof=proof, tokens=spent)
        )
        if checked.outcome == "proved" and checked.formal is not None:
            return ProofResult(
                proved=True,
                held=False,
                prover=family,
                model=reply.model,
                proof=proof,
                source=candidate,
                axioms=checked.formal.axioms,
                attempts=attempts,
                tokens=tokens,
                stopped="proved",
            )
        last = (block if _restated(formal, block) is not None else proof, _why(checked))
    return ProofResult(
        proved=False, held=False, prover=family, attempts=attempts, tokens=tokens, stopped=stopped
    )


# --- the stage ---


def _skipped(reason: str) -> dict[str, JsonValue]:
    return {
        "attempted": False,
        "proved": False,
        "held": False,
        "hold_reasons": [],
        "weakened": False,
        "added_hypotheses": [],
        "weakened_statement": None,
        "audit": None,
        "reasons": [reason],
        "prover": None,
        "model": None,
        "proof": None,
        "axioms": [],
        "attempts": [],
        "tokens": 0,
        "stopped": None,
    }


def _skip_reason(screen: Mapping[str, JsonValue]) -> str | None:
    """Why the prover must not run after this triviality stage, or None."""
    if screen.get("screened") is not True:
        reasons = screen.get("reasons")
        why = "; ".join(str(r) for r in reasons) if isinstance(reasons, list) else ""
        return "the triviality screen did not screen it" + (f": {why}" if why else "")
    if screen.get("held") is True:
        return "the triviality screen proved it and held it for review"
    if screen.get("trivial") is not False or screen.get("held") is not False:
        return "the triviality stage does not say it found no proof, so the prover did not run"
    return None


def run_prover_stage(
    record_id: str,
    *,
    ledger: Ledger,
    budget: Budget,
    prover_budget: ProverBudget | None = None,
    backends: Mapping[Family, Backend] | None = None,
    lean_timeout_s: float = DEFAULT_LEAN_TIMEOUT_S,
) -> ProofResult | None:
    """Run the prover on a screened survivor and write its prover stage.

    Raise ValueError for a record that is not `survived` with formaliser and
    triviality stages, or that already has a prover stage. A record the
    screen held or did not screen, or whose formaliser file no longer
    matches its stage, is skipped: the stage says why (`attempted` false).
    Otherwise the stage records the loop's result and sets the status the
    module describes. The proved file goes to `runs/<run_id>/prover/<id>.lean`.

    The stage's data: `attempted`, `proved`, `held`, `hold_reasons`,
    `weakened`, `added_hypotheses`, `weakened_statement`, `audit`,
    `reasons`, `prover`, `model`, `proof`, `axioms`, `attempts`, `tokens`,
    and `stopped`. A model failure, a spent pair budget, a Lean time-out
    on the re-check, or Lean timing out on every proof attempt raises,
    leaving the record unchanged (the ledger still holds each finished
    call's charge). Return the result, or None if the
    record was skipped.
    """
    store = ledger.store
    record = store.load(record_id)
    formaliser = record.stages.get(Stage.formaliser)
    screen = record.stages.get(Stage.triviality)
    if record.status is not Status.survived or formaliser is None or screen is None:
        raise ValueError(
            f"{record_id} is {record.status}; only a survived record with formaliser and "
            "triviality stages is proved"
        )
    if Stage.prover in record.stages:
        raise ValueError(f"{record_id} already has a prover stage")

    def skip(reason: str) -> None:
        store.write_stage(
            record.id,
            Stage.prover,
            StageOutput(written_at=datetime.now(UTC), data=_skipped(reason)),
        )

    reason = _skip_reason(screen.data)
    if reason is not None:
        skip(reason)
        return None
    formal = stored_formal(record, formaliser.data, store.root, lean_timeout_s)
    if isinstance(formal, str):
        skip(formal)
        return None

    result = prove(
        record,
        formal,
        prover_budget or ProverBudget(),
        ledger=ledger,
        pair_budget=budget,
        backends=backends,
        lean_timeout_s=lean_timeout_s,
    )
    if result.attempts and all(a.outcome == "timed_out" for a in result.attempts):
        raise LeanTimeout(
            f"Lean timed out on all {len(result.attempts)} of the prover's proofs for "
            f"{record_id}; the record is unchanged, so the prover can run again"
        )
    if result.proved or result.held:
        flags = hold_reasons(formaliser.data)
        if flags:
            result = result.model_copy(
                update={
                    "proved": False,
                    "held": True,
                    "hold_reasons": [*result.hold_reasons, *flags],
                }
            )
    evidence: list[str] = []
    if result.source is not None:
        directory = store.root / "runs" / record.run_id / "prover"
        directory.mkdir(parents=True, exist_ok=True)
        name = f"{record.id}.lean"
        (directory / name).write_text(result.source, encoding="utf-8")
        evidence.append(f"prover/{name}")
    data: dict[str, JsonValue] = {
        "attempted": result.prover is not None,
        "proved": result.proved,
        "held": result.held,
        "hold_reasons": cast(list[JsonValue], result.hold_reasons),
        "weakened": result.weakened,
        "added_hypotheses": cast(list[JsonValue], result.added_hypotheses),
        "weakened_statement": result.weakened_statement,
        "audit": None
        if result.audit is None
        else cast(dict[str, JsonValue], result.audit.model_dump(mode="json")),
        "reasons": cast(list[JsonValue], result.notes),
        "prover": None if result.prover is None else result.prover.value,
        "model": result.model,
        "proof": result.proof,
        "axioms": cast(list[JsonValue], result.axioms),
        "attempts": cast(list[JsonValue], [a.model_dump(mode="json") for a in result.attempts]),
        "tokens": result.tokens,
        "stopped": result.stopped,
    }
    store.write_stage(
        record.id,
        Stage.prover,
        StageOutput(written_at=datetime.now(UTC), data=data, evidence=evidence),
        status=Status.proved if result.proved else None,
    )
    return result
