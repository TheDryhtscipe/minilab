"""The vacuity linter: are `conjecture`'s hypotheses consistent, satisfiable, and used?

Build plan doc-5wh, Task 11; design doc-gmj, stage 5.

One Lean run checks three things.

Every declaration the linter adds, and every marker it reads, carries a
random tag made for this run (`MinilabContraGoal_<tag>`), and a run counts
only if exactly one `READY` marker came back and every marker names a binder
of the right kind. The tag stops a file from declaring or printing the
linter's names ahead of time. It is no defence against model code that runs
while the file compiles, which could read the tag from the file; that is
the formaliser's command check's job, and the linter trusts an `ok`
formalisation from `formal_from_source` to have passed it.

**Contradictory hypotheses.** A Lean meta-program declares
`MinilabContraGoal_<tag>`, the statement `∀ binders, False`, and the linter tries
to prove it with a fixed list of tactics (`simp_all` and `norm_num` with the
file's definitions, `linarith`, `nlinarith`, `omega`, `tauto`). A proof that
Lean accepts, with only Lean's standard axioms, is a certificate that no
values satisfy the hypotheses. Failing to find a proof proves nothing: the
hypotheses may still contradict each other in a way the tactics miss.

**The falsifier's witness.** The falsifier's worst point gives values for
some variables. Each value is written by this module as a Lean numeral,
`(p / q)`, or a vector `![a, b, c]`, never copied from the record, and Lean
elaborates it against the variable's type with earlier values substituted.
Only variables whose type is ℕ, ℤ, ℚ, ℝ, or `Fin k →` one of those take a
value, and a fraction never goes into ℕ or ℤ (Lean would truncate it). For
each hypothesis whose variables all have values, the linter tries to prove
the instantiated hypothesis and its negation, so each is `holds`, `fails`,
`undecided`, or `not_instantiated`.

**Unused binders.** Lean's unused-variables linter cannot do this job: on a
theorem proved by `sorry`, every hypothesis is unused. So the linter works on
the probed statement instead, with a relevance closure over the binders and
the file's definitions (`unused_binders`). A binder is relevant if the
conclusion mentions it, if a relevant binder's type mentions it, or if it is
a hypothesis or instance binder whose type mentions a relevant binder or a
relevant definition. A definition is relevant if the conclusion or a
relevant binder mentions it, or a relevant definition uses it. So
`(L : ℝ) (hL : ∀ x, ‖deriv loss x‖ ≤ L)` is relevant to a conclusion about
`loss`. Every other binder is unused: it cannot bear on the conclusion. A
hypothesis that mentions no variable and no relevant definition, such as
`(1 : ℝ) < 2`, is unused too.

A rejected formalisation has contradictory hypotheses, an unused binder, or
a check that did not run. A Lean time-out is never a verdict: it sets
`timed_out`, decides nothing about the hypotheses, and rejects nothing.
"""

import json
import re
from fractions import Fraction
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from minilab.encoding.api import Instance, to_fraction
from minilab.formalise.formaliser import (
    DEFAULT_LEAN_TIMEOUT_S,
    STANDARD_AXIOMS,
    THEOREM,
    FormalOutput,
    checked_source,
    marker_messages,
    new_nonce,
)
from minilab.lean.runner import LeanResult, check_lean

WitnessState = Literal["holds", "fails", "undecided", "not_instantiated"]

VACUITY_MARKER = "MINILAB_VAC_"

CONTRA = "minilab_contra"

# The longest Lean text one witness value may become.
MAX_VALUE_TEXT = 1000

_NAME = re.compile(r"^[^\W\d]\w*'*$")


class WitnessCheck(BaseModel):
    """What the witness did to one hypothesis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    state: WitnessState
    detail: str = ""


class VacuityResult(BaseModel):
    """What the vacuity linter found.

    `checked` says the Lean meta-program ran, so `contradictory` and the
    witness checks mean something.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    contradictory: bool
    satisfied_by_witness: bool
    witness: list[WitnessCheck]
    unused: list[str]
    timed_out: bool
    notes: list[str]
    checked: bool = True

    @property
    def rejected(self) -> bool:
        return bool(self.reasons)

    @property
    def reasons(self) -> list[str]:
        """Why the formalisation is rejected; empty if it is not."""
        found: list[str] = []
        if self.contradictory:
            found.append("the hypotheses contradict each other (Lean proved False from them)")
        if self.unused:
            found.append("binders that cannot bear on the conclusion: " + ", ".join(self.unused))
        if not self.checked and not self.timed_out:
            found.append("the vacuity check did not run in Lean")
        return found


class _Marker(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    instantiated: bool | None = None
    ok: bool | None = None
    detail: str | None = None


def unused_binders(formal: FormalOutput) -> list[str]:
    """Labels of the binders that cannot bear on the conclusion, in order.

    The relevance closure, over binders and the file's definitions: the
    conclusion's binders and definitions are relevant; a relevant binder's
    binders and definitions are relevant; a relevant definition's definitions
    are relevant; and a hypothesis or instance binder that mentions a
    relevant binder or a relevant definition is relevant.
    """
    relevant = set(formal.conclusion_deps)
    defs = set(formal.conclusion_defs)
    changed = True
    while changed:
        changed = False
        for name in list(defs):
            more = set(formal.definition_uses.get(name, [])) - defs
            if more:
                defs |= more
                changed = True
        for binder in formal.binders:
            deps = set(binder.deps)
            if binder.index in relevant:
                if not deps <= relevant or not set(binder.defs) <= defs:
                    relevant |= deps
                    defs |= set(binder.defs)
                    changed = True
            elif (binder.prop or binder.instance) and (deps & relevant or set(binder.defs) & defs):
                relevant.add(binder.index)
                changed = True
    return [b.label for b in formal.binders if b.index not in relevant]


def _number(value: Fraction | str) -> tuple[str, bool]:
    """A value as Lean text, and whether it is an integer."""
    q = to_fraction(value)
    if q.denominator == 1:
        return (str(q.numerator) if q >= 0 else f"({q.numerator})"), True
    return f"({q.numerator} / {q.denominator})", False


def witness_terms(
    formal: FormalOutput, witness: Instance
) -> tuple[list[tuple[str, str, bool]], list[str]]:
    """(name, Lean text, integral) for each witness value that names a variable, and notes."""
    variables = {b.name for b in formal.binders if not b.prop}
    terms: list[tuple[str, str, bool]] = []
    notes: list[str] = []
    for name, value in sorted(witness.values.items()):
        if name not in variables or not _NAME.match(name):
            notes.append(f"the witness value {name!r} names no variable of the statement")
            continue
        try:
            if isinstance(value, list):
                parts = [_number(v) for v in value]
                text = "![" + ", ".join(p[0] for p in parts) + "]"
                integral = all(p[1] for p in parts)
            else:
                text, integral = _number(value)
        except (ValueError, ZeroDivisionError, OverflowError) as exc:
            notes.append(f"the witness value {name!r} is not an exact number: {exc}")
            continue
        if len(text) > MAX_VALUE_TEXT:
            notes.append(f"the witness value {name!r} is too long to write in Lean")
            continue
        terms.append((name, text, integral))
    return terms, notes


def _lean_string(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def _program(terms: list[tuple[str, str, bool]], nonce: str, arity: int) -> str:
    """The meta-program that declares the goals; see the module docstring."""
    witness = ", ".join(
        f"({_lean_string(n)}, {_lean_string(t)}, {'true' if i else 'false'})" for n, t, i in terms
    )
    m = f"{VACUITY_MARKER}{nonce}_"
    return f"""
open Lean Meta Elab Command in
#eval show CommandElabM Unit from do
  let env ← getEnv
  let some info := env.find? `{THEOREM} | throwError "no {THEOREM}"
  let witness : Array (String × String × Bool) := #[{witness}]
  let say : String → Json → TermElabM Unit := fun tag j =>
    logInfo (MessageData.ofFormat (Std.Format.text ("{m}" ++ tag ++ " " ++ j.compress)))
  let scalar : Expr → Bool := fun t =>
    t.isConstOf ``Real || t.isConstOf ``Rat || t.isConstOf ``Nat || t.isConstOf ``Int
  let integral : Expr → Bool := fun t => t.isConstOf ``Nat || t.isConstOf ``Int
  let define : Name → Expr → TermElabM Unit := fun name value => do
    let ps := (collectLevelParams {{}} value).params.toList
    let dv : DefinitionVal := {{ name := name, levelParams := ps, type := mkSort Level.zero,
                                 value := value, hints := ReducibilityHints.abbrev,
                                 safety := DefinitionSafety.safe }}
    addDecl (Declaration.defnDecl dv)
  liftTermElabM do
    forallBoundedTelescope info.type (some {arity}) fun xs _ => do
      if xs.size != {arity} then throwError "`{THEOREM}` has fewer binders than its header"
      define `MinilabContraGoal_{nonce} (← mkForallFVars xs (mkConst ``False))
      let mut fromV : Array Expr := #[]
      let mut toV : Array Expr := #[]
      for i in [0:xs.size] do
        let x := xs[i]!
        let d ← x.fvarId!.getDecl
        let raw ← instantiateMVars d.type
        let ty := raw.replaceFVars fromV toV
        let closed := !ty.hasFVar && !ty.hasMVar
        if ← isProp raw then
          define (Name.mkSimple s!"MinilabWitness_{nonce}_{{i}}")
            (if closed then ty else mkConst ``True)
          say "PROP" (Json.mkObj [("index", toJson i), ("instantiated", toJson closed)])
        else if d.binderInfo.isInstImplicit then
          if closed then
            match ← trySynthInstance ty with
            | LOption.some e =>
              fromV := fromV.push x
              toV := toV.push e
              say "SUBST" (Json.mkObj [("index", toJson i), ("ok", toJson true)])
            | _ => pure ()
        else
          let key := d.userName.eraseMacroScopes.toString
          match witness.find? (fun w => w.1 == key) with
          | none => pure ()
          | some (_, text, isInt) =>
            let codomain := if ty.isForall && !ty.bindingBody!.hasLooseBVars
              && ty.bindingDomain!.isAppOfArity ``Fin 1 then ty.bindingBody! else ty
            if !(closed && scalar codomain) then
              say "SUBST" (Json.mkObj [("index", toJson i), ("ok", toJson false),
                ("detail", toJson "its type is not ℕ, ℤ, ℚ, ℝ, or a vector of one of them")])
            else if !isInt && integral codomain then
              say "SUBST" (Json.mkObj [("index", toJson i), ("ok", toJson false),
                ("detail", toJson "the value is a fraction and the variable's type is ℕ or ℤ")])
            else
              match Parser.runParserCategory (← getEnv) `term text with
              | Except.error e =>
                say "SUBST" (Json.mkObj [("index", toJson i), ("ok", toJson false),
                  ("detail", toJson e)])
              | Except.ok stx =>
                try
                  let e ← Term.elabTermEnsuringType stx ty
                  Term.synthesizeSyntheticMVarsNoPostponing
                  let e ← instantiateMVars e
                  if e.hasMVar || e.hasFVar then throwError "the value did not elaborate fully"
                  fromV := fromV.push x
                  toV := toV.push e
                  say "SUBST" (Json.mkObj [("index", toJson i), ("ok", toJson true)])
                catch ex =>
                  say "SUBST" (Json.mkObj [("index", toJson i), ("ok", toJson false),
                    ("detail", toJson ((← ex.toMessageData.toString).take 300))])
      say "READY" Json.null
"""


_IDENT = re.compile(r"^[^\W\d][\w'.!?]*$")

_OPTIONS = """set_option autoImplicit false in
set_option linter.unusedTactic false in
set_option linter.unreachableTactic false in
"""


def _simp_set(names: list[str]) -> str:
    return f" [{', '.join(names)}]" if names else ""


def _contra_theorem(defs: list[str], nonce: str) -> str:
    simp = _simp_set(defs)
    return f"""{_OPTIONS}theorem _root_.{CONTRA}_{nonce} : MinilabContraGoal_{nonce} := by
  delta MinilabContraGoal_{nonce}
  intros
  first
    | (simp_all{simp}; done)
    | (norm_num{simp} at *; done)
    | linarith
    | nlinarith
    | omega
    | tauto
"""


def _witness_theorems(index: int, defs: list[str], nonce: str) -> tuple[str, str]:
    simp = _simp_set([*defs, "Fin.sum_univ_succ", "Fin.forall_fin_succ", "Fin.exists_fin_succ"])
    goal = f"MinilabWitness_{nonce}_{index}"
    return (
        f"""{_OPTIONS}theorem _root_.minilab_w_{nonce}_{index} : {goal} := by
  delta {goal}
  first
    | infer_instance
    | (norm_num{simp}; done)
    | decide
""",
        f"""{_OPTIONS}theorem _root_.minilab_wn_{nonce}_{index} : ¬ {goal} := by
  delta {goal}
  first
    | (norm_num{simp}; done)
    | decide
""",
    )


def _proved(result: LeanResult, name: str, lines: range) -> bool:
    """Lean accepted theorem `name`, written on `lines`, using only the standard axioms."""
    axioms = result.axioms_by_theorem.get(name)
    if axioms is None or not set(axioms) <= STANDARD_AXIOMS:
        return False
    return not any(
        m.severity == "error" and m.axioms_of is None and m.line in lines for m in result.messages
    )


def vacuity(
    formal: FormalOutput,
    witness: Instance | None,
    *,
    timeout_s: float = DEFAULT_LEAN_TIMEOUT_S,
) -> VacuityResult:
    """Run the vacuity linter on an `ok` formalisation, as the module describes."""
    if not formal.ok:
        raise ValueError("the vacuity linter needs a formalisation that compiled")
    unused = unused_binders(formal)
    notes: list[str] = []
    terms: list[tuple[str, str, bool]] = []
    if witness is not None:
        terms, notes = witness_terms(formal, witness)
    defs = [d for d in formal.definitions if _IDENT.match(d)]
    nonce = new_nonce()
    contra = f"{CONTRA}_{nonce}"

    body = checked_source(formal.source)
    text = body + _program(terms, nonce, len(formal.binders))
    contra_start = text.count("\n") + 1
    text += _contra_theorem(defs, nonce)
    contra_lines = range(contra_start, text.count("\n") + 1)
    lines: dict[str, range] = {}
    for binder in formal.hypotheses:
        holds, fails = _witness_theorems(binder.index, defs, nonce)
        for name, block in (
            (f"minilab_w_{nonce}_{binder.index}", holds),
            (f"minilab_wn_{nonce}_{binder.index}", fails),
        ):
            start = text.count("\n") + 1
            text += block
            lines[name] = range(start, text.count("\n") + 1)

    result = check_lean(text, timeout_s)
    if result.timed_out:
        return VacuityResult(
            contradictory=False,
            satisfied_by_witness=False,
            witness=[
                WitnessCheck(label=b.label, state="undecided", detail="Lean timed out")
                for b in formal.hypotheses
            ]
            if witness is not None
            else [],
            unused=unused,
            timed_out=True,
            notes=[*notes, f"Lean timed out after {timeout_s} s; nothing was decided"],
            checked=False,
        )

    source_lines = body.count("\n")
    instantiated: dict[int, bool] = {}
    substituted: set[int] = set()
    failed_values: dict[int, str] = {}
    by_index = {b.index: b for b in formal.binders}
    readies = 0
    malformed = 0
    for tag, payload in marker_messages(result, source_lines, f"{VACUITY_MARKER}{nonce}_"):
        if tag == "READY":
            readies += 1
            continue
        try:
            item = _Marker.model_validate_json(payload)
        except ValidationError:
            malformed += 1
            continue
        binder = by_index.get(item.index)
        if binder is None:
            malformed += 1
        elif tag == "PROP" and binder.prop and item.instantiated is not None:
            instantiated[item.index] = item.instantiated
        elif tag == "SUBST" and not binder.prop and item.ok is True:
            substituted.add(item.index)
        elif tag == "SUBST" and not binder.prop:
            failed_values[item.index] = item.detail or "not accepted"
        else:
            malformed += 1
    for index, detail in sorted(failed_values.items()):
        notes.append(f"the witness value for {by_index[index].label} was not used: {detail}")
    ready = readies == 1 and malformed == 0
    if not ready:
        notes.append(
            "the vacuity meta-program did not finish cleanly: " + "; ".join(result.errors)[:1000]
        )

    contradictory = ready and _proved(result, contra, contra_lines)
    checks: list[WitnessCheck] = []
    if witness is not None:
        for binder in formal.hypotheses:
            checks.append(
                _witness_check(
                    binder.index,
                    formal,
                    ready=ready,
                    instantiated=instantiated.get(binder.index, False),
                    substituted=substituted,
                    holds=_proved(
                        result,
                        f"minilab_w_{nonce}_{binder.index}",
                        lines[f"minilab_w_{nonce}_{binder.index}"],
                    ),
                    fails=_proved(
                        result,
                        f"minilab_wn_{nonce}_{binder.index}",
                        lines[f"minilab_wn_{nonce}_{binder.index}"],
                    ),
                )
            )
    return VacuityResult(
        contradictory=contradictory,
        satisfied_by_witness=bool(checks) and all(c.state == "holds" for c in checks),
        witness=checks,
        unused=unused,
        timed_out=False,
        notes=notes,
        checked=ready,
    )


def _witness_check(
    index: int,
    formal: FormalOutput,
    *,
    ready: bool,
    instantiated: bool,
    substituted: set[int],
    holds: bool,
    fails: bool,
) -> WitnessCheck:
    by_index = {b.index: b for b in formal.binders}
    binder = by_index[index]
    if not ready:
        return WitnessCheck(label=binder.label, state="undecided", detail="the check did not run")
    if not instantiated:
        missing = [
            by_index[d].label
            for d in binder.deps
            if d in by_index and d not in substituted and not by_index[d].prop
        ]
        detail = "no value for " + ", ".join(missing) if missing else "not closed by the witness"
        return WitnessCheck(label=binder.label, state="not_instantiated", detail=detail)
    if holds and not fails:
        return WitnessCheck(label=binder.label, state="holds")
    if fails and not holds:
        return WitnessCheck(label=binder.label, state="fails")
    return WitnessCheck(
        label=binder.label, state="undecided", detail="Lean proved neither it nor its negation"
    )
