"""The formaliser: a model writes the conjecture in Lean, and Lean checks the file.

Build plan doc-5wh, Task 11; design doc-gmj, stage 5.

A model from a family other than Gemini (Claude first, then Codex) writes one
Lean file: `import Mathlib`, the conjecture's concepts as `def`s, and one
top-level theorem named `conjecture` whose proof is `sorry`. Each assumption
and the domain of validity are named hypotheses of that theorem.

Before Lean runs, `source_problems` refuses a file that could change what
Lean or the later checks do: imports other than Mathlib, axioms, macros and
other syntax extensions, options, unsafe code, code run at compile time, any
name with `minilab` in it (the checks below add declarations with that
prefix), and any number of `sorry`s other than one. Comments and string
literals do not count.

`formal_from_source` then compiles the file once, with a probe appended. The
probe finds `conjecture` in Lean's environment and prints each binder of its
statement: name, type, whether it is a proposition (a hypothesis), whether it
is an instance binder, and which earlier binders its type mentions. It also
prints the conclusion and the file's definitions. These come from Lean's own
elaborated statement, not from reading the text, so the vacuity linter and
the hypothesis audit work on what Lean will prove.

The file is `ok` when Lean reports no error, `conjecture` is a theorem whose
axioms include `sorryAx`, nothing else in the file depends on `sorryAx`, and
no declaration uses an axiom outside Lean's standard three. A Lean time-out
raises `LeanTimeout`: it is never a verdict.
"""

import json
import re
from collections.abc import Iterable, Mapping
from typing import cast

from pydantic import BaseModel, ConfigDict, ValidationError

from minilab.lean.runner import LeanResult, check_lean
from minilab.ledger import Ledger
from minilab.models.adapter import Budget, Role, call
from minilab.models.backends import Backend
from minilab.records import Family, Record

DEFAULT_LEAN_TIMEOUT_S = 300.0

THEOREM = "conjecture"

SORRY_AXIOM = "sorryAx"

STANDARD_AXIOMS: frozenset[str] = frozenset({"propext", "Classical.choice", "Quot.sound"})
"""The axioms a file may use, besides `sorryAx` in `conjecture` itself."""

# The longest binder or conclusion type the probe prints, in characters. The
# Lean runner keeps at most 4000 characters of a message.
MAX_TYPE_CHARS = 1500


class LeanTimeout(Exception):
    """Lean hit its time limit, so nothing was decided."""


class NoFormaliser(Exception):
    """Neither Claude nor Codex is in the run's manifest."""


class Binder(BaseModel):
    """One binder of `conjecture`'s statement, as Lean elaborated it.

    `label` is the binder's name, made unique with `_<index>` when two binders
    share a name or a binder has none. `deps` are the indices of earlier
    binders that its type mentions. `truncated` says the type was cut to
    `MAX_TYPE_CHARS` characters.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int
    name: str
    label: str
    type: str
    prop: bool
    instance: bool
    deps: list[int]
    truncated: bool = False


class FormalOutput(BaseModel):
    """One formalisation: the Lean source and what the probe found in it.

    `lean_statement` is the text of `conjecture` without its proof.
    `problems` says why the file is not `ok`. `attempts` counts the model
    calls that led to this file.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    family: Family
    model: str
    source: str
    theorem: str = THEOREM
    lean_statement: str
    binders: list[Binder]
    conclusion: str
    conclusion_deps: list[int]
    definitions: list[str]
    ok: bool
    problems: list[str]
    attempts: int = 1

    @property
    def hypotheses(self) -> list[Binder]:
        """The binders whose types are propositions."""
        return [b for b in self.binders if b.prop]


# --- reading the source ---


def _ident_char(ch: str) -> bool:
    return ch.isalnum() or ch in "_'!?"


_CHAR_LITERAL = re.compile(r"'(?:\\(?:x[0-9a-fA-F]{2}|u\{[0-9a-fA-F]+\}|.)|[^'\\\n])'")
_RAW_STRING = re.compile(r"r(#*)\"")


def _scan(source: str) -> tuple[str, str]:
    """The source without comments, and the same text with string contents blanked.

    The two texts have the same length, so a position found in the second
    names the same place in the first. Newlines inside comments and strings
    are kept, so line numbers do not change.
    """
    kept: list[str] = []
    code: list[str] = []
    i, n = 0, len(source)
    while i < n:
        if source.startswith("--", i):
            end = source.find("\n", i)
            i = n if end < 0 else end
            continue
        if source.startswith("/-", i):
            depth, j, newlines = 1, i + 2, 0
            while j < n and depth:
                if source.startswith("/-", j):
                    depth, j = depth + 1, j + 2
                elif source.startswith("-/", j):
                    depth, j = depth - 1, j + 2
                else:
                    newlines += source[j] == "\n"
                    j += 1
            filler = "\n" * newlines or " "
            kept.append(filler)
            code.append(filler)
            i = j
            continue
        before = source[i - 1] if i else " "
        raw = _RAW_STRING.match(source, i) if not _ident_char(before) else None
        if raw is not None:
            close = '"' + raw.group(1)
            end = source.find(close, raw.end())
            end = n if end < 0 else end + len(close)
            text = source[i:end]
            kept.append(text)
            code.append(_blank(text))
            i = end
            continue
        if source[i] == '"':
            j = i + 1
            while j < n and source[j] != '"':
                j += 2 if source[j] == "\\" else 1
            end = min(j + 1, n)
            text = source[i:end]
            kept.append(text)
            code.append('"' + _blank(text[1:-1]) + '"' if len(text) > 1 else text)
            i = end
            continue
        if source[i] == "'" and not _ident_char(before):
            found = _CHAR_LITERAL.match(source, i)
            if found is not None:
                kept.append(found.group())
                code.append(_blank(found.group()))
                i = found.end()
                continue
        kept.append(source[i])
        code.append(source[i])
        i += 1
    return "".join(kept), "".join(code)


def _blank(text: str) -> str:
    return "".join(ch if ch == "\n" else " " for ch in text)


def strip_comments(source: str) -> str:
    """The source with every Lean comment removed (doc comments too); strings kept."""
    return _scan(source)[0]


def _word(word: str) -> re.Pattern[str]:
    return re.compile(r"(?<![\w.'!?])" + re.escape(word) + r"(?![\w'!?])")


FORBIDDEN: tuple[str, ...] = (
    "axiom",
    "opaque",
    "macro",
    "macro_rules",
    "syntax",
    "declare_syntax_cat",
    "elab",
    "elab_rules",
    "notation",
    "infix",
    "infixl",
    "infixr",
    "prefix",
    "postfix",
    "set_option",
    "unsafe",
    "partial",
    "implemented_by",
    "extern",
    "#eval",
    "#exit",
    "run_cmd",
    "run_elab",
    "run_meta",
    "initialize",
    "builtin_initialize",
    "admit",
    "include",
    "omit",
    "_root_",
    "simproc",
    "dsimproc",
    "command_elab",
    "term_elab",
    "tactic",
    "native_decide",
)
"""Words a formalisation may not use. Each could change what Lean or the
later checks do, or run code while the file compiles."""

_FORBIDDEN = [(word, _word(word)) for word in FORBIDDEN]
_SORRY = _word("sorry")
_MINILAB = re.compile(r"minilab", re.IGNORECASE)
_IMPORT = re.compile(r"^\s*import\s+(\S+)", re.MULTILINE)
_HEADER = re.compile(r"(?<![\w.'!?])(?:theorem|lemma)\s+" + re.escape(THEOREM) + r"(?![\w'!?.])")


def _statement_span(code: str) -> tuple[int, int] | None:
    """Where `conjecture`'s statement starts, and where its `:=` is, in `code`.

    The `:=` is the first one after the header outside any brackets.
    """
    header = _HEADER.search(code)
    if header is None:
        return None
    depth = 0
    for j in range(header.end(), len(code) - 1):
        ch = code[j]
        if ch in "([{⟨":
            depth += 1
        elif ch in ")]}⟩":
            depth -= 1
        elif depth == 0 and code.startswith(":=", j):
            return header.start(), j
    return None


def source_problems(source: str) -> list[str]:
    """Why the source cannot be checked, from its text alone. Empty if none.

    Comments and string literals are ignored.
    """
    _, code = _scan(source)
    problems: list[str] = []
    modules = _IMPORT.findall(code)
    if not modules:
        problems.append("the file must import Mathlib")
    for module in modules:
        if module != "Mathlib" and not module.startswith("Mathlib."):
            problems.append(f"import of {module}: only Mathlib may be imported")
    for word, pattern in _FORBIDDEN:
        if pattern.search(code):
            problems.append(f"forbidden construct `{word}`")
    if _MINILAB.search(code):
        problems.append("names containing `minilab` are reserved for the checks")
    headers = len(_HEADER.findall(code))
    if headers != 1:
        problems.append(f"the file must state one theorem named `{THEOREM}`; it states {headers}")
    sorries = [m.start() for m in _SORRY.finditer(code)]
    if len(sorries) != 1:
        problems.append(
            f"the file must use `sorry` exactly once, as the proof of `{THEOREM}`; "
            f"it uses it {len(sorries)} times"
        )
    span = _statement_span(code)
    if headers == 1 and span is None:
        problems.append(f"no `:=` ends the statement of `{THEOREM}`")
    if span is not None and len(sorries) == 1 and sorries[0] < span[1]:
        problems.append(f"the `sorry` must be in the proof of `{THEOREM}`")
    return problems


CHECK_OPTIONS = " set_option autoImplicit false set_option relaxedAutoImplicit false"
"""Added to the end of the last import line before Lean checks a file. The
Lean project leaves auto-bound implicits on, so a misspelt name would
otherwise become a new variable of the statement instead of an error."""


def checked_source(source: str) -> str:
    """The text Lean checks: comments removed and `CHECK_OPTIONS` after the imports.

    Lines keep their numbers, so Lean's messages point at the model's lines.
    """
    stripped, code = _scan(source)
    imports = list(_IMPORT.finditer(code))
    if not imports:
        return stripped if stripped.endswith("\n") else stripped + "\n"
    end = stripped.find("\n", imports[-1].end())
    end = len(stripped) if end < 0 else end
    text = stripped[:end] + CHECK_OPTIONS + stripped[end:]
    return text if text.endswith("\n") else text + "\n"


def lean_statement(source: str) -> str:
    """The text of `conjecture` without its proof or comments; empty if not found."""
    stripped, code = _scan(source)
    span = _statement_span(code)
    if span is None:
        return ""
    return stripped[span[0] : span[1]].strip()


# --- the probe ---

PROBE_MARKER = "MINILAB_PROBE_"

PROBE = f"""
open Lean Meta Elab Command in
#eval show CommandElabM Unit from do
  let say : String → Json → CommandElabM Unit := fun tag j =>
    logInfo (MessageData.ofFormat (Std.Format.text ("{PROBE_MARKER}" ++ tag ++ " " ++ j.compress)))
  let env ← getEnv
  match env.find? `{THEOREM} with
  | none => say "NONE" Json.null
  | some info =>
    let isThm := match info with
      | ConstantInfo.thmInfo _ => true
      | _ => false
    let mut defs : Array Json := #[]
    for (name, ci) in env.constants.map₂.toList do
      let user := (privateToUserName? name).getD name
      if user.isInternal then continue
      match ci with
      | ConstantInfo.defnInfo _ =>
        let ax ← collectAxioms name
        defs := defs.push (Json.mkObj [("name", toJson user.toString),
          ("sorry", toJson (ax.contains ``sorryAx))])
      | _ => pure ()
    say "DEFS" (Json.arr defs)
    liftTermElabM do
      forallTelescope info.type fun xs body => do
        for i in [0:xs.size] do
          let d ← xs[i]!.fvarId!.getDecl
          let ty ← instantiateMVars d.type
          let mut deps : Array Nat := #[]
          for j in [0:i] do
            if ty.containsFVar xs[j]!.fvarId! then deps := deps.push j
          let isP ← Meta.isProp ty
          let s := toString (← ppExpr ty)
          logInfo (MessageData.ofFormat (Std.Format.text ("{PROBE_MARKER}BINDER " ++
            (Json.mkObj [("index", toJson i),
              ("name", toJson d.userName.eraseMacroScopes.toString),
              ("prop", toJson isP), ("instance", toJson d.binderInfo.isInstImplicit),
              ("deps", toJson deps), ("type", toJson (s.take {MAX_TYPE_CHARS})),
              ("truncated", toJson (decide (s.length > {MAX_TYPE_CHARS})))]).compress)))
        let b ← instantiateMVars body
        let mut deps : Array Nat := #[]
        for j in [0:xs.size] do
          if b.containsFVar xs[j]!.fvarId! then deps := deps.push j
        let s := toString (← ppExpr b)
        logInfo (MessageData.ofFormat (Std.Format.text ("{PROBE_MARKER}BODY " ++
          (Json.mkObj [("theorem", toJson isThm), ("deps", toJson deps),
            ("type", toJson (s.take {MAX_TYPE_CHARS}))]).compress)))
"""
"""Appended after the model's source. It prints, as information messages
that start with `PROBE_MARKER`: `DEFS` (the file's definitions, and whether
each depends on `sorryAx`), one `BINDER` per binder of `conjecture`, and
`BODY` (the conclusion), or `NONE` if there is no `conjecture`. Names are
written without macro scopes, so an instance binder's name is readable."""


class _RawBinder(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    name: str
    prop: bool
    instance: bool
    deps: list[int]
    type: str
    truncated: bool


class _RawBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    theorem: bool
    deps: list[int]
    type: str


class _RawDef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    sorry: bool


def marker_messages(result: LeanResult, after_line: int, marker: str) -> list[tuple[str, str]]:
    """(tag, payload) of each information message from lines after `after_line`.

    Only lines the checks appended count, so the model's own source cannot
    forge a result.
    """
    found: list[tuple[str, str]] = []
    for message in result.messages:
        if message.severity != "information" or message.line <= after_line:
            continue
        if message.axioms_of is not None or not message.text.startswith(marker):
            continue
        tag, _, payload = message.text.removeprefix(marker).partition(" ")
        found.append((tag, payload))
    return found


def source_errors(result: LeanResult, source_lines: int) -> list[str]:
    """Lean's errors on the model's own lines, as `line:column: text`."""
    return [
        f"{m.line}:{m.column}: {m.text}"
        for m in result.messages
        if m.severity == "error" and m.axioms_of is None and m.line <= source_lines
    ]


def _labels(raw: list[_RawBinder]) -> list[str]:
    counts: dict[str, int] = {}
    for b in raw:
        counts[b.name] = counts.get(b.name, 0) + 1
    return [
        b.name
        if counts[b.name] == 1 and b.name not in ("", "[anonymous]")
        else f"{b.name}_{b.index}"
        for b in raw
    ]


def formal_from_source(
    source: str,
    family: Family | str,
    model: str,
    *,
    timeout_s: float = DEFAULT_LEAN_TIMEOUT_S,
    attempts: int = 1,
) -> FormalOutput:
    """Check `source` and probe its `conjecture`, as the module describes.

    Raise LeanTimeout if Lean hits `timeout_s`.
    """
    family = Family(family)
    statement = lean_statement(source)

    def refused(problems: list[str]) -> FormalOutput:
        return FormalOutput(
            family=family,
            model=model,
            source=source,
            lean_statement=statement,
            binders=[],
            conclusion="",
            conclusion_deps=[],
            definitions=[],
            ok=False,
            problems=problems,
            attempts=attempts,
        )

    problems = source_problems(source)
    if problems:
        return refused(problems)
    body = checked_source(source)
    source_lines = body.count("\n")
    result = check_lean(body + PROBE, timeout_s)
    if result.timed_out:
        raise LeanTimeout(f"Lean did not finish the formaliser check in {timeout_s} s")
    problems = source_errors(result, source_lines)
    probe_errors = [e for e in result.errors if e not in problems]
    raw_binders: list[_RawBinder] = []
    raw_body: _RawBody | None = None
    raw_defs: list[_RawDef] = []
    missing = False
    try:
        for tag, payload in marker_messages(result, source_lines, PROBE_MARKER):
            if tag == "NONE":
                missing = True
            elif tag == "DEFS":
                items = cast(list[object], json.loads(payload))
                raw_defs = [_RawDef.model_validate(item) for item in items]
            elif tag == "BINDER":
                raw_binders.append(_RawBinder.model_validate_json(payload))
            elif tag == "BODY":
                raw_body = _RawBody.model_validate_json(payload)
    except (ValidationError, json.JSONDecodeError, TypeError) as exc:
        problems.append(f"the probe's output could not be read: {str(exc)[:300]}")
    if missing:
        problems.append(f"there is no top-level theorem named `{THEOREM}`")
    elif raw_body is None:
        if not problems:
            problems.append("the probe gave no result: " + "; ".join(probe_errors)[:1000])
    elif not raw_body.theorem:
        problems.append(f"`{THEOREM}` is not a theorem")
    if not problems and probe_errors:
        problems.append("Lean reported errors: " + "; ".join(probe_errors)[:1000])
    own = result.axioms_by_theorem.get(THEOREM)
    if raw_body is not None and raw_body.theorem and (own is None or SORRY_AXIOM not in own):
        problems.append(f"`{THEOREM}` must be proved by `sorry` alone")
    for name, axioms in result.axioms_by_theorem.items():
        if name != THEOREM and SORRY_AXIOM in axioms:
            problems.append(f"`{name}` depends on `sorry`")
    for item in raw_defs:
        if item.sorry:
            problems.append(f"the definition `{item.name}` depends on `sorry`")
    extra = sorted(set(result.axioms) - STANDARD_AXIOMS - {SORRY_AXIOM})
    if extra:
        problems.append(f"the file uses axioms beyond Lean's standard ones: {', '.join(extra)}")
    raw_binders.sort(key=lambda b: b.index)
    binders = [
        Binder(
            index=b.index,
            name=b.name,
            label=label,
            type=b.type,
            prop=b.prop,
            instance=b.instance,
            deps=b.deps,
            truncated=b.truncated,
        )
        for b, label in zip(raw_binders, _labels(raw_binders), strict=True)
    ]
    return FormalOutput(
        family=family,
        model=model,
        source=source,
        lean_statement=statement,
        binders=binders,
        conclusion=raw_body.type if raw_body is not None else "",
        conclusion_deps=raw_body.deps if raw_body is not None else [],
        definitions=[d.name for d in raw_defs],
        ok=not problems,
        problems=problems,
        attempts=attempts,
    )


# --- the model call ---


def formaliser_family(available: Iterable[Family | str]) -> Family:
    """Claude if the run has it, else Codex. Never Gemini (it generates only) or local."""
    families = {Family(f) for f in available}
    for family in (Family.claude, Family.codex):
        if family in families:
            return family
    raise NoFormaliser(f"no formaliser family among {sorted(families)}")


def _record_text(record: Record) -> str:
    assumptions = "\n".join(f"  {i}. {a}" for i, a in enumerate(record.assumptions, 1))
    return f"""Statement: {record.statement}
Objects: {"; ".join(record.objects)}
Assumptions:
{assumptions or "  (none)"}
Domain of validity: {record.domain_of_validity}
Instance family: {record.instance_family or "(none)"}
"""


def formalise_prompt(record: Record) -> str:
    """The prompt that asks for the Lean file."""
    return f"""Write this conjecture as a Lean 4 file for Mathlib.

{_record_text(record)}
Rules for the file:
- Start with `import Mathlib` and import nothing else.
- Define each concept the conjecture uses as a `def`, in plain Mathlib terms.
- State the conjecture as one top-level `theorem {THEOREM}` whose proof is
  `sorry`. Use `sorry` nowhere else.
- Give each assumption, and the domain of validity, as its own named
  hypothesis of the theorem (`(h1 : ...)`). Add no hypothesis that the
  conjecture does not state.
- Name the variables after the parameters of the instance family.
- Write no comments. Use no `axiom`, `set_option`, `macro`, `notation`, or
  `#eval`, and no name containing "minilab".

Reply with the whole file in one ```lean code block."""


def retry_prompt(record: Record, source: str, problems: list[str]) -> str:
    """The prompt that shows the refused file and why it was refused."""
    listed = "\n".join(f"- {p}" for p in problems)
    return (
        formalise_prompt(record)
        + f"""

Your previous file was refused:

```lean
{source}
```

Problems:
{listed}

Write the whole file again, fixed."""
    )


_FENCE = re.compile(r"```[ \t]*(?P<lang>[A-Za-z0-9_+-]*)[ \t]*\n(?P<body>.*?)\n?```", re.DOTALL)


def extract_lean(text: str) -> str:
    """The first ```lean block of a reply (or an unlabelled one), else the whole reply."""
    for found in _FENCE.finditer(text):
        if found.group("lang").lower() in ("lean", "lean4", ""):
            body = found.group("body")
            return body if body.endswith("\n") else body + "\n"
    return text


def formalise(
    record: Record,
    *,
    ledger: Ledger,
    budget: Budget,
    backends: Mapping[Family, Backend] | None = None,
    family: Family | None = None,
    max_attempts: int = 2,
    timeout_s: float = DEFAULT_LEAN_TIMEOUT_S,
) -> FormalOutput:
    """Ask a model for the Lean file, retrying with Lean's feedback; return the last try.

    Raise what `call` raises (`ModelUnavailable`, `BudgetExceeded`, ...),
    LeanTimeout, or NoFormaliser. The result may not be `ok`.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    family = formaliser_family(ledger.manifest.families) if family is None else family
    prompt = formalise_prompt(record)
    formal: FormalOutput | None = None
    for attempt in range(1, max_attempts + 1):
        reply = call(
            family,
            Role.formalise,
            prompt,
            record_id=record.id,
            pair=record.pair,
            ledger=ledger,
            budget=budget,
            backends=backends,
        )
        source = extract_lean(reply.text)
        formal = formal_from_source(
            source, family, reply.model, timeout_s=timeout_s, attempts=attempt
        )
        if formal.ok:
            break
        prompt = retry_prompt(record, source, formal.problems)
    assert formal is not None
    return formal
