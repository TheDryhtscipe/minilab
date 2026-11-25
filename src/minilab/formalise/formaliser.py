"""The formaliser: a model writes the conjecture in Lean, and Lean checks the file.

Build plan doc-5wh, Task 11; design doc-gmj, stage 5.

A model from a family other than Gemini (Claude first, then Codex) writes one
Lean file: `import Mathlib`, the conjecture's concepts as `def`s, and one
top-level theorem named `conjecture` whose proof is `sorry`. Each assumption
and the domain of validity are named hypotheses of that theorem.

Model code must never run while Lean compiles the file: code that runs can
write the markers the checks read. Two gates come before any compile.
`source_problems` reads the text and refuses imports other than Mathlib, a
list of dangerous words, any name with `minilab` in it (the checks add
declarations with that prefix), and any number of `sorry`s other than one.
Comments and string literals do not count, and `«…»` names are opaque.
Then `command_check` has Lean parse the file, held only as a string, and
refuses every top-level command outside an allowlist (`open`, `namespace`,
`section`, `end`, `universe`, and `def`, `abbrev`, `theorem`, `lemma`,
`structure`, `inductive` declarations), any attribute, `instance`,
`variable`, `unsafe`, `partial`, `set_option`, `native_decide`, and any
`do` block (which `run_tac`, `by_elab` and their kin need). The allowlist is
the guard; the word list is a cheap first pass.

`formal_from_source` then compiles the file once, with a probe appended. The
probe finds `conjecture` in Lean's environment and prints each binder written
before its colon: name, type, whether it is a proposition (a hypothesis),
whether it is an instance binder, which earlier binders its type mentions,
and which file definitions. It also prints the conclusion (which keeps any
`∀` or `→` after the colon) and the definitions the file wrote. These come
from Lean's own elaborated statement, not from reading the text, so the
vacuity linter and the hypothesis audit work on what Lean will prove. Each
run's markers carry a random tag.

The file is `ok` when Lean reports no error, `conjecture` is a theorem whose
axioms include `sorryAx`, nothing else in the file depends on `sorryAx`, and
no declaration uses an axiom outside Lean's standard three. A Lean time-out
raises `LeanTimeout`: it is never a verdict.
"""

import json
import re
import secrets
from collections.abc import Iterable, Mapping

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

    `label` is the binder's name, or `name#index` when two binders share a
    name or a binder has none (`binder_labels`). `deps` are the indices of
    earlier binders that its type mentions, and `defs` the file definitions.
    `truncated` says the type was cut to `MAX_TYPE_CHARS` characters.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int
    name: str
    label: str
    type: str
    prop: bool
    instance: bool
    deps: list[int]
    defs: list[str] = []
    truncated: bool = False


class FormalOutput(BaseModel):
    """One formalisation: the Lean source and what the probe found in it.

    `lean_statement` is the text of `conjecture` without its proof.
    `binders` are the binders before the colon; `conclusion` is the rest.
    `definition_uses` maps each file definition to the file definitions its
    value mentions. `problems` says why the file is not `ok`. `attempts`
    counts the model calls that led to this file.
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
    conclusion_defs: list[str] = []
    conclusion_truncated: bool = False
    definitions: list[str]
    definition_uses: dict[str, list[str]] = {}
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
        if source[i] == "«":
            # A quoted identifier: opaque, so a `"` inside it opens no string.
            end = source.find("»", i + 1)
            end = n if end < 0 else end + 1
            text = source[i:end]
            kept.append(text)
            code.append("".join(ch if ch in "«»\n" else "x" for ch in text))
            i = end
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
    "#eval!",
    "#exit",
    "run_tac",
    "by_elab",
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
_LET = re.compile(r"(?<![\w.'!?])(?:let|have)(?![\w'!?])")
_HEADER = re.compile(r"(?<![\w.'!?])(?:theorem|lemma)\s+" + re.escape(THEOREM) + r"(?![\w'!?.])")


def _statement_span(code: str) -> tuple[int, int] | None:
    """Where `conjecture`'s statement starts, and where its `:=` is, in `code`.

    The `:=` is the first one after the header outside any brackets that no
    `let` or `have` in the statement claims.
    """
    header = _HEADER.search(code)
    if header is None:
        return None
    depth = 0
    pending = 0
    for j in range(header.end(), len(code) - 1):
        ch = code[j]
        if ch in "([{⟨⦃":
            depth += 1
        elif ch in ")]}⟩⦄":
            depth -= 1
        elif depth == 0 and _LET.match(code, j):
            pending += 1
        elif depth == 0 and code.startswith(":=", j):
            if pending:
                pending -= 1
                continue
            return header.start(), j
    return None


def source_problems(source: str) -> list[str]:
    """Why the source cannot be checked, from its text alone. Empty if none.

    Comments and string literals are ignored.
    """
    kept, code = _scan(source)
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
    if _MINILAB.search(kept):
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


# --- the command check ---


def lean_string_literal(text: str) -> str:
    """`text` as a Lean string literal."""
    out: list[str] = []
    for ch in text:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\x{ord(ch):02x}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


ALLOWED_COMMANDS: tuple[str, ...] = (
    "Lean.Parser.Command.open",
    "Lean.Parser.Command.namespace",
    "Lean.Parser.Command.section",
    "Lean.Parser.Command.end",
    "Lean.Parser.Command.universe",
    "Lean.Parser.Command.declaration",
    "lemma",
)
"""The top-level commands a formalisation may use. `open … in` is allowed
around an allowed command."""

ALLOWED_DECLARATIONS: tuple[str, ...] = (
    "Lean.Parser.Command.definition",
    "Lean.Parser.Command.theorem",
    "Lean.Parser.Command.abbrev",
    "Lean.Parser.Command.structure",
    "Lean.Parser.Command.inductive",
    "Lean.Parser.Command.classInductive",
)
"""The kinds of `declaration` a formalisation may use: no `instance`,
`axiom`, `opaque`, or `example`."""

DENIED_KINDS: tuple[str, ...] = (
    "Lean.Parser.Term.attributes",
    "Lean.Parser.Command.unsafe",
    "Lean.Parser.Command.partial",
    "Lean.Parser.Tactic.runTac",
    "Lean.Parser.Term.byElab",
    "Lean.Parser.Tactic.nativeDecide",
)
"""Syntax that may appear nowhere in a formalisation. Besides these, any
`do` block (a syntax kind whose last part starts with `doSeq`) and any
`set_option` are refused: `run_tac`, `by_elab` and their kin take a `do`
block and run it while the file compiles."""


def _lean_names(kinds: tuple[str, ...]) -> str:
    return "[" + ", ".join(f"`{k}" for k in kinds) + "]"


def _command_check(source: str, marker: str) -> str:
    """A Lean file that parses `source` without running it and lists what it may not use.

    The model's text is only a string literal here: Lean parses it with
    Mathlib's syntax, and elaborates only its `open`, `namespace`, `section`
    and `end` commands (so scoped notation parses). It prints `PROBLEM` for
    each refusal, then `ARITY` (how many binders `conjecture` has before its
    colon, or null), then `DONE`.
    """
    return f"""import Mathlib

open Lean Parser Elab Command

partial def minilabCheckKinds (s : Syntax) (acc : Array (SyntaxNodeKind × Syntax)) :
    Array (SyntaxNodeKind × Syntax) :=
  match s with
  | .node _ k args => args.foldl (fun a c => minilabCheckKinds c a) (acc.push (k, s))
  | _ => acc

def minilabCheckArity (sig : Syntax) : Nat :=
  sig[0].getArgs.foldl (fun n b =>
    let k := b.getKind
    if k == ``Lean.Parser.Term.explicitBinder || k == ``Lean.Parser.Term.implicitBinder
        || k == ``Lean.Parser.Term.strictImplicitBinder then n + b[1].getNumArgs
    else n + 1) 0

#eval show CommandElabM Unit from do
  let src : String := {lean_string_literal(source)}
  let say : String → Json → CommandElabM Unit := fun tag j =>
    logInfo (MessageData.ofFormat (Std.Format.text ("{marker}" ++ tag ++ " " ++ j.compress)))
  let allowedCmds : List Name := {_lean_names(ALLOWED_COMMANDS)}
  let allowedDecls : List Name := {_lean_names(ALLOWED_DECLARATIONS)}
  let denied : List Name := {_lean_names(DENIED_KINDS)}
  let inputCtx := Parser.mkInputContext src "<formalisation>"
  let (_, mps, msgs) ← Parser.parseHeader inputCtx
  let mut mps := mps
  let mut msgs := msgs
  let mut problems : Array String := #[]
  let mut depth : Int := 0
  let mut arity : Option Nat := none
  let lineOf : Syntax → Nat := fun s => (inputCtx.fileMap.toPosition (s.getPos?.getD 0)).line
  repeat
    let scope ← getScope
    let pmctx : Parser.ParserModuleContext :=
      {{ env := ← getEnv, options := scope.opts, currNamespace := scope.currNamespace,
         openDecls := scope.openDecls }}
    let (cmd, mps', msgs') := Parser.parseCommand inputCtx pmctx mps msgs
    mps := mps'
    msgs := msgs'
    if Parser.isTerminalCommand cmd then break
    let line := lineOf cmd
    -- `open … in` around a command: check the `open`, then the command.
    let mut inner := cmd
    let mut opens : Array Syntax := #[]
    while inner.getKind == ``Lean.Parser.Command.in do
      opens := opens.push inner[0]
      inner := inner[2]
    let k := inner.getKind
    let mut fine := true
    for o in opens do
      if o.getKind != ``Lean.Parser.Command.open then
        problems := problems.push s!"line {{line}}: only `open … in` may wrap a command"
        fine := false
    if !allowedCmds.contains k then
      problems := problems.push s!"line {{line}}: the command `{{k}}` is not allowed"
      fine := false
    else if k == ``Lean.Parser.Command.declaration && !allowedDecls.contains inner[1].getKind then
      let dk := inner[1].getKind
      problems := problems.push s!"line {{line}}: the declaration kind `{{dk}}` is not allowed"
      fine := false
    for (kind, node) in minilabCheckKinds cmd #[] do
      let last := match kind with
        | .str _ s => s
        | _ => ""
      let what := if last.startsWith "doSeq" then some "a `do` block"
        else if denied.contains kind || last == "set_option" then some s!"`{{kind}}`"
        else none
      if let some w := what then
        let ln := if last.startsWith "doSeq" then line else lineOf node
        let p := s!"line {{ln}}: {{w}} is not allowed"
        if !problems.contains p then problems := problems.push p
        fine := false
      if kind == ``Lean.Parser.Command.declId && depth == 0 && node[0].getId == `{THEOREM} then
        let sig := (minilabCheckKinds cmd #[]).find? (fun p => p.1 == ``Lean.Parser.Command.declSig)
        match sig with
        | some (_, s) => arity := some (minilabCheckArity s)
        | none => pure ()
    if fine && opens.isEmpty then
      if k == ``Lean.Parser.Command.namespace || k == ``Lean.Parser.Command.section then
        depth := depth + 1
      if k == ``Lean.Parser.Command.end then
        depth := depth - 1
      if k == ``Lean.Parser.Command.open || k == ``Lean.Parser.Command.namespace
          || k == ``Lean.Parser.Command.section || k == ``Lean.Parser.Command.end then
        try elabCommand cmd
        catch e => problems := problems.push s!"line {{line}}: {{← e.toMessageData.toString}}"
  for m in msgs.toList do
    if m.severity == .error then
      problems := problems.push s!"Lean could not parse the file: {{(← m.toString).take 300}}"
  if depth != 0 then
    problems := problems.push "every `namespace` and `section` must be closed with `end`"
  for p in problems.toList.take 20 do
    say "PROBLEM" (toJson (p.take 300))
  say "ARITY" (toJson arity)
  say "DONE" Json.null
"""


def command_check(
    source: str, *, timeout_s: float = DEFAULT_LEAN_TIMEOUT_S
) -> tuple[list[str], int | None]:
    """What the file may not use, and how many binders `conjecture` has before its colon.

    Lean parses the file without running any of it (`_command_check`). Only
    the commands in `ALLOWED_COMMANDS` and declarations in
    `ALLOWED_DECLARATIONS` may appear, and none of `DENIED_KINDS`, a `do`
    block, or `set_option`. This is the guard against model code running
    while the file compiles. Raise LeanTimeout if Lean hits `timeout_s`.
    """
    marker = f"MINILAB_CMD_{secrets.token_hex(8)}_"
    result = check_lean(_command_check(strip_comments(source), marker), timeout_s)
    if result.timed_out:
        raise LeanTimeout(f"Lean did not finish the command check in {timeout_s} s")
    problems: list[str] = []
    arity: int | None = None
    done = 0
    for tag, payload in marker_messages(result, 0, marker):
        try:
            value: object = json.loads(payload)
        except json.JSONDecodeError:
            problems.append("the command check's output could not be read")
            continue
        if tag == "PROBLEM" and isinstance(value, str):
            problems.append(value)
        elif tag == "ARITY" and (value is None or (isinstance(value, int) and value >= 0)):
            arity = value
        elif tag == "DONE":
            done += 1
    if done != 1:
        problems.append("the command check did not finish: " + "; ".join(result.errors)[:1000])
    return problems, arity


# --- the probe ---

PROBE_PREFIX = "MINILAB_PROBE_"

# The most file definitions one binder, the conclusion, or a definition lists.
MAX_DEFS_LISTED = 50


def new_nonce() -> str:
    """A random tag for one Lean run's markers and declaration names."""
    return secrets.token_hex(8)


def probe(marker: str, arity: int) -> str:
    """The Lean command appended after the model's source; see `formal_from_source`."""
    return f"""
open Lean Meta Elab Command in
#eval show CommandElabM Unit from do
  let say : String → Json → CommandElabM Unit := fun tag j =>
    logInfo (MessageData.ofFormat (Std.Format.text ("{marker}" ++ tag ++ " " ++ j.compress)))
  let env ← getEnv
  let mut fileDefs : Array Name := #[]
  for (name, ci) in env.constants.map₂.toList do
    let user := (privateToUserName? name).getD name
    if user.isInternal || user.isInternalDetail then continue
    if isAuxRecursor env name || isNoConfusion env name || isMatcherCore env name
        || (env.isProjectionFn name) || isInstanceCore env name then continue
    -- Lean records where each declaration the source wrote is; the ones it
    -- generates (`noConfusionType`, `ctorIdx`, deriving helpers) have no range.
    if (← findDeclarationRanges? name).isNone then continue
    match ci with
    | ConstantInfo.defnInfo _ => fileDefs := fileDefs.push name
    | _ => pure ()
  let userName : Name → String := fun n => ((privateToUserName? n).getD n).toString
  let mentions : Expr → Json := fun e =>
    toJson ((fileDefs.filter (fun n => e.containsConst (· == n))).toList.take {MAX_DEFS_LISTED}
      |>.map userName)
  for name in fileDefs do
    let some ci := env.find? name | continue
    let ax ← collectAxioms name
    say "DEF" (Json.mkObj [("name", toJson (userName name)),
      ("sorry", toJson (ax.contains ``sorryAx)), ("uses", mentions (ci.value?.getD ci.type))])
  match env.find? `{THEOREM} with
  | none => say "NONE" Json.null
  | some info =>
    let isThm := match info with
      | ConstantInfo.thmInfo _ => true
      | _ => false
    liftTermElabM do
      forallBoundedTelescope info.type (some {arity}) fun xs body => do
        if xs.size != {arity} then
          throwError "`{THEOREM}` has fewer binders than its header"
        for i in [0:xs.size] do
          let d ← xs[i]!.fvarId!.getDecl
          let ty ← instantiateMVars d.type
          let mut deps : Array Nat := #[]
          for j in [0:i] do
            if ty.containsFVar xs[j]!.fvarId! then deps := deps.push j
          let isP ← Meta.isProp ty
          let s := toString (← ppExpr ty)
          logInfo (MessageData.ofFormat (Std.Format.text ("{marker}BINDER " ++
            (Json.mkObj [("index", toJson i),
              ("name", toJson d.userName.eraseMacroScopes.toString),
              ("prop", toJson isP), ("instance", toJson d.binderInfo.isInstImplicit),
              ("deps", toJson deps), ("defs", mentions ty),
              ("type", toJson (s.take {MAX_TYPE_CHARS})),
              ("truncated", toJson (decide (s.length > {MAX_TYPE_CHARS})))]).compress)))
        let b ← instantiateMVars body
        let mut deps : Array Nat := #[]
        for j in [0:xs.size] do
          if b.containsFVar xs[j]!.fvarId! then deps := deps.push j
        let s := toString (← ppExpr b)
        logInfo (MessageData.ofFormat (Std.Format.text ("{marker}BODY " ++
          (Json.mkObj [("theorem", toJson isThm), ("deps", toJson deps), ("defs", mentions b),
            ("type", toJson (s.take {MAX_TYPE_CHARS})),
            ("truncated", toJson (decide (s.length > {MAX_TYPE_CHARS})))]).compress)))
"""


class _RawBinder(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    name: str
    prop: bool
    instance: bool
    deps: list[int]
    defs: list[str]
    type: str
    truncated: bool


class _RawBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    theorem: bool
    deps: list[int]
    defs: list[str]
    type: str
    truncated: bool


class _RawDef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    sorry: bool
    uses: list[str]


def marker_messages(result: LeanResult, after_line: int, marker: str) -> list[tuple[str, str]]:
    """(tag, payload) of each information message from lines after `after_line`.

    Only lines the checks appended count, and `marker` carries a random tag
    made for this run, so the model's source cannot write a result.
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


def binder_labels(names: list[str]) -> list[str]:
    """A distinct label for each binder name, in order.

    A name used once keeps its name. Others (repeats, and binders with no
    name) become `name#index`: `#` cannot appear in a printed Lean name
    without `«»`, so such a label never equals another binder's name.
    """
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    labels: list[str] = []
    used: set[str] = set()
    for index, name in enumerate(names):
        label = name
        if counts[name] > 1 or name in ("", "[anonymous]"):
            label = f"{name}#{index}"
        while label in used:
            label += "'"
        used.add(label)
        labels.append(label)
    return labels


def formal_from_source(
    source: str,
    family: Family | str,
    model: str,
    *,
    timeout_s: float = DEFAULT_LEAN_TIMEOUT_S,
    attempts: int = 1,
) -> FormalOutput:
    """Check `source` and probe its `conjecture`.

    Three gates, in order. `source_problems` reads the text. `command_check`
    has Lean parse the file without running it and refuses any command or
    syntax outside its allowlist; it also counts the binders of `conjecture`
    before its colon. Only then does Lean compile the file, once, with
    `probe` appended. The probe prints, as information messages that start
    with a marker made for this run: `DEF` for each definition the file
    wrote (not the matchers, projections, instances and other declarations
    Lean adds), with whether it depends on `sorryAx` and which other file
    definitions it uses; one `BINDER` per binder before the colon; and
    `BODY`, the conclusion, which keeps any `∀` or `→` written after the
    colon. Each binder and the conclusion list the file definitions they
    mention.

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
    problems, arity = command_check(source, timeout_s=timeout_s)
    if not problems and arity is None:
        problems.append(f"there is no top-level theorem named `{THEOREM}`")
    if problems or arity is None:
        return refused(problems)
    body = checked_source(source)
    source_lines = body.count("\n")
    marker = f"{PROBE_PREFIX}{new_nonce()}_"
    result = check_lean(body + probe(marker, arity), timeout_s)
    if result.timed_out:
        raise LeanTimeout(f"Lean did not finish the formaliser check in {timeout_s} s")
    problems = source_errors(result, source_lines)
    probe_errors = [e for e in result.errors if e not in problems]
    raw_binders: list[_RawBinder] = []
    bodies: list[_RawBody] = []
    raw_defs: list[_RawDef] = []
    missing = False
    try:
        for tag, payload in marker_messages(result, source_lines, marker):
            if tag == "NONE":
                missing = True
            elif tag == "DEF":
                raw_defs.append(_RawDef.model_validate_json(payload))
            elif tag == "BINDER":
                raw_binders.append(_RawBinder.model_validate_json(payload))
            elif tag == "BODY":
                bodies.append(_RawBody.model_validate_json(payload))
    except ValidationError as exc:
        problems.append(f"the probe's output could not be read: {str(exc)[:300]}")
    raw_body = bodies[0] if len(bodies) == 1 else None
    raw_binders.sort(key=lambda b: b.index)
    if missing:
        problems.append(f"there is no top-level theorem named `{THEOREM}`")
    elif raw_body is None:
        if not problems:
            problems.append("the probe gave no result: " + "; ".join(probe_errors)[:1000])
    elif not raw_body.theorem:
        problems.append(f"`{THEOREM}` is not a theorem")
    elif [b.index for b in raw_binders] != list(range(arity)):
        problems.append("the probe's binders do not match the theorem's header")
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
    binders = [
        Binder(
            index=b.index,
            name=b.name,
            label=label,
            type=b.type,
            prop=b.prop,
            instance=b.instance,
            deps=b.deps,
            defs=b.defs,
            truncated=b.truncated,
        )
        for b, label in zip(raw_binders, binder_labels([b.name for b in raw_binders]), strict=True)
    ]
    return FormalOutput(
        family=family,
        model=model,
        source=source,
        lean_statement=statement,
        binders=binders,
        conclusion=raw_body.type if raw_body is not None else "",
        conclusion_deps=raw_body.deps if raw_body is not None else [],
        conclusion_defs=raw_body.defs if raw_body is not None else [],
        conclusion_truncated=raw_body.truncated if raw_body is not None else False,
        definitions=[d.name for d in raw_defs],
        definition_uses={d.name: d.uses for d in raw_defs},
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
    family = formaliser_family(ledger.manifest.families if family is None else [family])
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
