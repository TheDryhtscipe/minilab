"""The hypothesis audit: every Lean hypothesis the record does not state.

Build plan doc-5wh, Task 11; design doc-gmj, stage 5.

The script goes first. It matches a hypothesis to the record only when the
hypothesis's Lean type, as Lean prints it, is the same text as one of the
record's assumptions or its domain of validity (whitespace and Unicode
compatibility forms aside), and the type mentions none of the file's
definitions (the text of a definition's name says nothing about what the
definition says). It does not match by name: the names are the formaliser's
own labels, and trusting them would let the formaliser vouch for itself.

A checker model from a family other than the formaliser's (`pick_checker`)
is always called, because a condition can hide outside the hypotheses: in a
definition, in a variable's type (a subtype, `ℕ+`, `ℝ≥0`), in an instance
binder, or in the conclusion (`(0 < x → P x) ∧ True`). It is asked to judge
each hypothesis the script did not match and each the falsifier's witness
fails in Lean (a stated assumption holds at the witness, so a hypothesis
that fails there says more than the record does). It sorts each as
`matches` (it says what an assumption or the domain says), `weakening` (an
extra condition that may be stated as a weakening of the conjecture), or
`begs_question` (it assumes what the conjecture claims, or something close to
it). It also lists each condition hidden anywhere else, with `where` it is
and the same kinds; a listed condition of kind `matches` or `none` says
nothing extra and is dropped.

It fails closed. A hypothesis the checker does not judge is `unjudged`, and
an unreadable reply leaves every hypothesis it was asked about `unjudged`.
A hidden condition of any other kind is `unjudged`. The formalisation is
rejected if any hypothesis or hidden condition is `begs_question` or
`unjudged`. A weakening is not a rejection: its text goes
into `weakenings` for the stage to record. (A stage cannot edit the record's
assumptions, so the weakenings live in the formaliser stage's data.)

`extra` lists every hypothesis whose verdict is not `matches`.
"""

import json
import re
import unicodedata
from collections.abc import Mapping
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from minilab.formalise.formaliser import Binder, FormalOutput, lean_view
from minilab.formalise.vacuity import VacuityResult
from minilab.ledger import Ledger
from minilab.models.adapter import Budget, Role, call, pick_checker
from minilab.models.backends import Backend
from minilab.records import Family, Record

Kind = Literal["matches", "weakening", "begs_question", "unjudged"]

DOMAIN = "domain"


class AuditedHypothesis(BaseModel):
    """One hypothesis and the verdict on it. `by` says who gave the verdict."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    type: str
    kind: Kind
    counterpart: str | None = None
    reason: str = ""
    by: Literal["script", "checker"]


class HiddenCondition(BaseModel):
    """A condition outside the hypotheses, as the checker reported it.

    `where` names the definition, the binder, or "conclusion".
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    where: str
    condition: str
    kind: Literal["weakening", "begs_question", "unjudged"]
    reason: str = ""


class AuditResult(BaseModel):
    """The audit's verdicts, as the module describes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hypotheses: list[AuditedHypothesis]
    extra: list[str]
    weakenings: list[str]
    hidden: list[HiddenCondition]
    rejected: bool
    reasons: list[str]
    checker: Family
    judged: bool


def _normal(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", "", text).rstrip(".")
    while text.startswith("(") and text.endswith(")") and _balanced(text[1:-1]):
        text = text[1:-1]
    return text


def _balanced(text: str) -> bool:
    depth = 0
    for ch in text:
        depth += {"(": 1, ")": -1}.get(ch, 0)
        if depth < 0:
            return False
    return depth == 0


def script_matches(record: Record, formal: FormalOutput) -> dict[str, str]:
    """Hypothesis label to "assumption i" (1-based) or "domain", by exact stated text.

    A hypothesis whose type mentions a file definition is never matched.
    """
    stated: list[tuple[str, str]] = [
        (_normal(a), f"assumption {i}") for i, a in enumerate(record.assumptions, 1)
    ]
    stated.append((_normal(record.domain_of_validity), DOMAIN))
    found: dict[str, str] = {}
    for hypothesis in formal.hypotheses:
        if hypothesis.truncated or hypothesis.defs:
            continue
        lean = _normal(hypothesis.type)
        for text, where in stated:
            if lean and lean == text:
                found[hypothesis.label] = where
                break
    return found


def _shown(binder: Binder) -> str:
    """The binder's type, marked if the probe cut it."""
    return binder.type + (" …(cut)" if binder.truncated else "")


def audit_prompt(
    record: Record,
    formal: FormalOutput,
    asked: list[Binder],
    failing: set[str],
    *,
    unwitnessed: set[str] | None = None,
) -> str:
    assumptions = "\n".join(f"  {i}. {a}" for i, a in enumerate(record.assumptions, 1))
    listed = "\n".join(
        f"- {b.label} : {_shown(b)}"
        + ("  (this hypothesis fails at the falsifier's witness)" if b.label in failing else "")
        for b in asked
    )
    variables = "\n".join(
        f"- {b.label} : {_shown(b)}"
        + ("  (instance binder)" if b.instance else "")
        + (
            "  (no witness value reached it, so any condition its type carries was not "
            "checked: judge that condition as a hidden hypothesis)"
            if b.label in (unwitnessed or set())
            else ""
        )
        for b in formal.binders
        if not b.prop
    )
    conclusion = formal.conclusion + (" …(cut)" if formal.conclusion_truncated else "")
    return f"""You are doing a hypothesis audit of a Lean formalisation of a conjecture.

The conjecture as stated:
Statement: {record.statement}
Assumptions:
{assumptions or "  (none)"}
Domain of validity: {record.domain_of_validity}

The Lean file, as Lean prints it (definitions, then the theorem):
```lean
{lean_view(formal)}
```

Judge each of these hypotheses of `{formal.theorem}`:
{listed or "  (none)"}

Its variables:
{variables or "  (none)"}

Its conclusion, as Lean prints it:
{conclusion}

For each one, give its kind:
- "matches": it says what one stated assumption or the domain says. Give that
  counterpart as "assumption <number>" or "domain".
- "weakening": it adds a condition the statement does not have, and the
  conjecture could honestly be restated with it as an extra assumption.
- "begs_question": it assumes the conclusion, or a fact close to it that the
  conjecture is meant to establish.
A hypothesis that fails at the falsifier's witness cannot match an assumption
the witness satisfies.

Also list each condition the statement does not have that hides outside
these hypotheses: in a definition in the file, in a variable's type (a
subtype, `ℕ+`, `ℝ≥0` and the like), in an instance binder, or in the
conclusion (an implication or case split the statement does not make). Give
`where` as the definition's name, the variable's label, or "conclusion", and
kind "weakening" or "begs_question". List nothing that the statement says.

Reply with one JSON object and nothing else:
{{"hypotheses": [{{"label": "...", "kind": "...", "counterpart": "..." or null, "reason": "..."}}],
 "hidden": [{{"where": "...", "condition": "...", "kind": "...", "reason": "..."}}]}}"""


class _RawVerdict(BaseModel):
    model_config = ConfigDict(extra="ignore")

    label: str
    kind: str
    counterpart: str | None = None
    reason: str = ""


class _RawHidden(BaseModel):
    model_config = ConfigDict(extra="ignore")

    where: str
    condition: str
    kind: str
    reason: str = ""


class _RawReply(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hypotheses: list[_RawVerdict] = Field(default_factory=lambda: [])
    hidden: list[_RawHidden] = Field(default_factory=lambda: [])


_FENCE = re.compile(r"```[ \t]*[A-Za-z0-9_+-]*[ \t]*\n(?P<body>.*?)\n?```", re.DOTALL)


def parse_reply(text: str) -> _RawReply | None:
    """The checker's reply, or None if no candidate in it has the right shape."""
    candidates = [m.group("body") for m in _FENCE.finditer(text)] + [text]
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            value: object = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        try:
            return _RawReply.model_validate(cast(dict[str, JsonValue], value))
        except ValidationError:
            continue
    return None


_JUDGED: frozenset[str] = frozenset({"matches", "weakening", "begs_question"})

_NOTHING_HIDDEN: frozenset[str] = frozenset({"matches", "none"})


def audit(
    record: Record,
    formal: FormalOutput,
    *,
    ledger: Ledger,
    budget: Budget,
    backends: Mapping[Family, Backend] | None = None,
    vacuity: VacuityResult | None = None,
) -> AuditResult:
    """Audit `formal` against `record`, as the module describes.

    `vacuity` gives the witness results; a hypothesis the witness fails goes
    to the checker even if the script matched it. Raise NoChecker if no
    family can check, and what `call` raises.
    """
    matched = script_matches(record, formal)
    failing = {
        w.label for w in (vacuity.witness if vacuity is not None else []) if w.state == "fails"
    }
    asked = [h for h in formal.hypotheses if h.label not in matched or h.label in failing]
    # The script's matches are never re-asked; `script_matches` already
    # refuses a hypothesis that mentions a file definition.
    verdicts: dict[str, AuditedHypothesis] = {}
    for h in formal.hypotheses:
        if h.label in matched and h.label not in failing:
            verdicts[h.label] = AuditedHypothesis(
                label=h.label,
                type=h.type,
                kind="matches",
                counterpart=matched[h.label],
                reason="the Lean type is the stated text",
                by="script",
            )
    hidden: list[HiddenCondition] = []
    checker = pick_checker(formal.family, ledger.manifest.families)
    reply = call(
        checker,
        Role.audit,
        audit_prompt(
            record,
            formal,
            asked,
            failing,
            unwitnessed=set(vacuity.unwitnessed) if vacuity is not None else None,
        ),
        record_id=record.id,
        pair=record.pair,
        ledger=ledger,
        budget=budget,
        backends=backends,
    )
    parsed = parse_reply(reply.text)
    judged = parsed is not None
    given: dict[str, _RawVerdict] = {}
    if parsed is not None:
        for verdict in parsed.hypotheses:
            given.setdefault(verdict.label, verdict)
        for item in parsed.hidden:
            if item.kind in _NOTHING_HIDDEN:
                continue
            hidden_kind: Literal["weakening", "begs_question", "unjudged"] = "unjudged"
            if item.kind == "weakening" or item.kind == "begs_question":
                hidden_kind = item.kind
            hidden.append(
                HiddenCondition(
                    where=item.where,
                    condition=item.condition,
                    kind=hidden_kind,
                    reason=item.reason,
                )
            )
    for h in asked:
        verdict = given.get(h.label)
        kind: Kind = "unjudged"
        if verdict is not None and verdict.kind in _JUDGED:
            kind = cast(Kind, verdict.kind)
        verdicts[h.label] = AuditedHypothesis(
            label=h.label,
            type=h.type,
            kind=kind,
            counterpart=verdict.counterpart if verdict is not None else None,
            reason=verdict.reason if verdict is not None else "the checker gave no verdict",
            by="checker",
        )
    hypotheses = [verdicts[h.label] for h in formal.hypotheses]
    extra = [h.label for h in hypotheses if h.kind != "matches"]
    weakenings = [f"{h.label}: {h.type}" for h in hypotheses if h.kind == "weakening"]
    weakenings += [f"{c.where}: {c.condition}" for c in hidden if c.kind == "weakening"]
    reasons: list[str] = []
    if not judged:
        reasons.append("the checker's reply could not be read")
    for h in hypotheses:
        if h.kind == "begs_question":
            reasons.append(f"hypothesis {h.label} begs the question: {h.reason}")
        elif h.kind == "unjudged":
            reasons.append(f"hypothesis {h.label} was not judged")
    for c in hidden:
        if c.kind == "begs_question":
            reasons.append(f"a condition in {c.where} begs the question: {c.condition}")
        elif c.kind == "unjudged":
            reasons.append(f"a condition in {c.where} was not judged")
    return AuditResult(
        hypotheses=hypotheses,
        extra=extra,
        weakenings=weakenings,
        hidden=hidden,
        rejected=bool(reasons),
        reasons=reasons,
        checker=checker,
        judged=judged,
    )
