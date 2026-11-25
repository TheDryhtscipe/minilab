"""The formaliser: a model writes the conjecture in Lean, and Lean checks the file.

Build plan doc-5wh, Task 11; design doc-gmj, stage 5.

A model from a family other than Gemini (Claude first, then Codex) writes one
Lean file: `import Mathlib`, the conjecture's concepts as `def`s, and one
top-level theorem named `conjecture` whose proof is `sorry`. Each assumption
and the domain of validity are named hypotheses of that theorem.

Model code must never run while Lean compiles the file: code that runs can
print a forged result, read the run's marker from the file, or exit early.
The Lean runner imports the guard (`lean/Minilab/Guard.lean`) into every
file it checks. The guard turns code generation off, so no declaration in
the file gets code and nothing it writes can run: not `#eval`, `eval%`,
`decide +native`, an Aesop rule, nor any elaborator that evaluates a term.
Our checks are the guard's own commands, compiled when it is built, given
their inputs as literals.

Two gates come before the compile, as a second guard and to give the model
reasons it can act on. `source_problems` reads the text and refuses imports
other than Mathlib, a list of dangerous words (`sorryAx` among them), any
name with `minilab` in it (the checks add declarations with that prefix),
and any number of the word `sorry` other than one. It skips comments and
plain string literals, and `«…»` names are opaque. It reads an interpolated
string (`s!"…{e}…"`) as a plain string, so it misses what is in the braces;
the next gate refuses interpolated strings. Then `command_check` has Lean
parse the file, held only as a string, and refuses every top-level command
outside an allowlist (`open`, `namespace`, `section`, `end`, `universe`, and
`def`, `abbrev`, `theorem`, `lemma`, `structure`, `inductive`
declarations), `open … in` around a command that opens or closes a scope,
an `open` outside any `section` or `namespace` (it would stay in force for
the checks appended after the file), any declaration Lean would treat as
internal (a part that starts with `_`, or a last part like `proof_1`), any
declaration other than the root one whose name ends in `conjecture`, any
attribute,
`instance`, `variable`, `unsafe`, `partial`, `set_option`, `native_decide`,
`eval%`, `plausible`, Aesop rule clauses, `decide` with a configuration, an
interpolated string, and any `do` block. Every `namespace` and `section`
must be closed. Lean checks the file with auto-bound implicits off
(`CHECK_OPTIONS`, given on Lean's command line).

`formal_from_source` then compiles the file once, with a probe appended. The
probe finds `conjecture` in Lean's environment and prints each binder written
before its colon: name, type, whether it is a proposition (a hypothesis),
whether it is an instance binder, which earlier binders its type mentions,
and which file definitions. It also prints the conclusion (which keeps any
`∀` or `→` after the colon) and the definitions the file wrote. These come
from Lean's own elaborated statement, not from reading the text, so the
vacuity linter and the hypothesis audit work on what Lean will prove. The
linter treats only the propositions before the colon as hypotheses: a
condition inside a variable's type (a subtype, a set, a structure's field)
or a premise after the colon is not one of them. Each
run's markers carry a random tag.

The file is `ok` when Lean reports no error, `conjecture` is a theorem whose
axioms include `sorryAx`, its statement does not depend on `sorryAx`, no
other constant in the file depends on `sorryAx` (internal and generated
names included), and no declaration uses an axiom outside Lean's standard
three. A Lean time-out
raises `LeanTimeout`: it is never a verdict.
"""

import json
import re
import secrets
from collections.abc import Iterable, Mapping
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

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

# The longest binder or conclusion type the probe prints, in characters: the
# guard's `maxTypeChars`. The Lean runner keeps at most 4000 characters of a
# message.
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

    `lean_statement` is the text of `conjecture` without its proof, as Lean
    parsed it (`command_check`); the text reading `lean_statement` only when
    Lean did not find it.
    `binders` are the binders before the colon; `conclusion` is the rest.
    `definitions` are the file's `def`s and the types it declares
    (`inductive`, `structure`, `class inductive`); `types` lists the latter.
    `definition_uses` maps each file definition to the file definitions its
    value (a type's: its constructors) mentions. `problems` says why the
    file is not `ok`. `attempts` counts the model calls that led to this
    file.
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
    types: list[str] = []
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
    "sorryAx",
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

    Comments and plain string literals are skipped. An interpolated string is
    read as a plain one, so the braces' contents are skipped too;
    `command_check` refuses interpolated strings.
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


CHECK_OPTIONS: dict[str, str] = {"autoImplicit": "false", "relaxedAutoImplicit": "false"}
"""Lean options for every check of a formalisation, given to Lean on its
command line (`check_lean`'s `options`), not as text in the file. The Lean
project leaves auto-bound implicits on, so a misspelt name would otherwise
become a new variable of the statement instead of an error. The model's
file cannot undo them: `set_option` is refused by both gates."""


def checked_source(source: str) -> str:
    """The text Lean checks: the source with comments removed, ending in a newline.

    Lines keep their numbers, so Lean's messages point at the model's lines.
    `command_check` parses this same text.
    """
    stripped = strip_comments(source)
    return stripped if stripped.endswith("\n") else stripped + "\n"


def lean_statement(source: str) -> str:
    """The text after the first `theorem conjecture` header, up to its `:=`; empty if none.

    This reads the text only, so it is a fallback for a file Lean never
    parsed. `FormalOutput.lean_statement` comes from Lean (`command_check`)
    whenever Lean found the root `conjecture`.
    """
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


def _command_check(source: str, marker: str) -> str:
    """A Lean file that parses `source` without running it and lists what it may not use.

    The model's text is only a string literal here, given to the guard's
    `#minilab_command_check` (`lean/Minilab/Guard.lean`, which holds the
    allowlist). Lean parses it with Mathlib's syntax and elaborates only its
    `open`, `namespace`, `section` and `end` commands (so scoped notation
    parses). It prints `PROBLEM` for each refusal, then `ARITY` (how many
    binders `conjecture` has before its colon, or null), then `DONE`.
    """
    return (
        "import Mathlib\n\n"
        f"#minilab_command_check {lean_string_literal(marker)} {lean_string_literal(source)}\n"
    )


_SPAN: TypeAdapter[tuple[int, int]] = TypeAdapter(tuple[int, int])


class CommandCheck(NamedTuple):
    """What `command_check` found.

    `problems` are its refusals. `arity` is how many binders `conjecture` has
    before its colon, or None if Lean found no `conjecture` at the root.
    `statement` is the text Lean parsed as `conjecture`, from its first token
    to the end of its type, or None.
    """

    problems: list[str]
    arity: int | None
    statement: str | None


def command_check(source: str, *, timeout_s: float = DEFAULT_LEAN_TIMEOUT_S) -> CommandCheck:
    """What the file may not use, how many binders `conjecture` has, and its statement.

    Lean parses the file without running any of it (`_command_check`). Only
    the commands and declaration kinds on the guard's allowlist may appear,
    and none of its denied syntax: attributes, `unsafe`, `partial`,
    `run_tac`, `by_elab`, `native_decide`, `eval%`, `plausible`, Aesop rule
    clauses, `decide` with a configuration, an interpolated string, a `do`
    block, or `set_option`. `open … in` may not wrap a scope command, an
    `open` must be inside a `section` or `namespace`, no declaration may
    have a name Lean treats as internal, and every `namespace` and `section`
    must be closed. Only the root `conjecture` may have that last name; one
    inside a `section` counts. The statement is the text Lean parsed as
    `conjecture`, from its first token to the end of its type.
    No model code runs in any Lean check, since the guard turns code
    generation off; the allowlist is a second guard that gives the model a
    reason it can act on. Raise LeanTimeout if Lean hits `timeout_s`.
    """
    text = checked_source(source).encode("utf-8")
    marker = f"MINILAB_CMD_{secrets.token_hex(8)}_"
    result = check_lean(
        _command_check(text.decode("utf-8"), marker), timeout_s, options=CHECK_OPTIONS
    )
    if result.timed_out:
        raise LeanTimeout(f"Lean did not finish the command check in {timeout_s} s")
    problems: list[str] = []
    arity: int | None = None
    statement: str | None = None
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
        elif tag == "STATEMENT" and value is not None:
            try:
                start, end = _SPAN.validate_python(value)
            except ValidationError:
                problems.append("the command check's statement range could not be read")
                continue
            if 0 <= start <= end <= len(text):
                statement = text[start:end].decode("utf-8", "replace").strip()
        elif tag == "DONE":
            done += 1
    if done != 1:
        problems.append("the command check did not finish: " + "; ".join(result.errors)[:1000])
    return CommandCheck(problems, arity, statement)


# --- the probe ---

PROBE_PREFIX = "MINILAB_PROBE_"

# The most file definitions one binder, the conclusion, or a definition
# lists: the guard's `maxDefsListed`.
MAX_DEFS_LISTED = 50


def new_nonce() -> str:
    """A random tag for one Lean run's markers and declaration names."""
    return secrets.token_hex(8)


def probe(marker: str, arity: int) -> str:
    """The command appended after the model's source; see `formal_from_source`.

    The probe is the guard's `#minilab_probe` (`lean/Minilab/Guard.lean`),
    compiled when the guard is built. It finds `conjecture` in Lean's
    environment and prints each of its first `arity` binders, its
    conclusion, and the file's definitions.
    """
    return f"\n#minilab_probe {lean_string_literal(marker)} {arity}\n"


class _RawBinder(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    name: str
    prop: bool
    instance: bool
    deps: list[int]
    defs: list[str]
    foreign: list[str]
    type: str
    truncated: bool


class _RawBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    theorem: bool
    sorry: bool
    deps: list[int]
    defs: list[str]
    foreign: list[str]
    type: str
    truncated: bool


_SORRY_LIST: TypeAdapter[list[str]] = TypeAdapter(list[str])


class _RawDef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: bool
    sorry: bool
    uses: list[str]


def _foreign_problem(where: str, shown: str) -> str:
    return (
        f"{where} uses the instance `{shown}`, which is not the one Lean's instance search "
        "finds there; the printed statement hides it, so write the statement with the "
        "usual instances"
    )


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
    definitions it uses; `SORRY`, every constant of the file other than
    `conjecture` that depends on `sorryAx`; one `BINDER` per binder before
    the colon; and `BODY`, the conclusion, which keeps any `∀` or `→`
    written after the colon, with whether the statement depends on
    `sorryAx`. Each binder and the conclusion list the file definitions they
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
    problems, arity, parsed = command_check(source, timeout_s=timeout_s)
    if parsed is not None:
        statement = parsed
    if not problems and arity is None:
        problems.append(f"there is no top-level theorem named `{THEOREM}`")
    if problems or arity is None:
        return refused(problems)
    body = checked_source(source)
    source_lines = body.count("\n")
    marker = f"{PROBE_PREFIX}{new_nonce()}_"
    result = check_lean(body + probe(marker, arity), timeout_s, options=CHECK_OPTIONS)
    if result.timed_out:
        raise LeanTimeout(f"Lean did not finish the formaliser check in {timeout_s} s")
    problems = source_errors(result, source_lines)
    probe_errors = [e for e in result.errors if e not in problems]
    raw_binders: list[_RawBinder] = []
    bodies: list[_RawBody] = []
    raw_defs: list[_RawDef] = []
    with_sorry: list[list[str]] = []
    missing = False
    try:
        for tag, payload in marker_messages(result, source_lines, marker):
            if tag == "NONE":
                missing = True
            elif tag == "SORRY":
                with_sorry.append(_SORRY_LIST.validate_json(payload))
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
    if len(with_sorry) != 1:
        problems.append("the probe did not report which declarations depend on `sorry`")
    if raw_body is not None and raw_body.sorry:
        problems.append(f"the statement of `{THEOREM}` depends on `sorry`")
    if not problems and probe_errors:
        problems.append("Lean reported errors: " + "; ".join(probe_errors)[:1000])
    own = result.axioms_by_theorem.get(THEOREM)
    if raw_body is not None and raw_body.theorem and (own is None or SORRY_AXIOM not in own):
        problems.append(f"`{THEOREM}` must be proved by `sorry` alone")
    sorry_names = {n for found in with_sorry for n in found}
    sorry_names |= {n for n, axioms in result.axioms_by_theorem.items() if SORRY_AXIOM in axioms}
    sorry_names |= {item.name for item in raw_defs if item.sorry}
    for name in sorted(sorry_names - {THEOREM}):
        problems.append(f"`{name}` depends on `sorry`")
    for b in raw_binders:
        for shown in b.foreign:
            problems.append(_foreign_problem(f"the type of binder {b.index}", shown))
    for shown in raw_body.foreign if raw_body is not None else []:
        problems.append(_foreign_problem("the conclusion", shown))
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
        types=[d.name for d in raw_defs if d.type],
        definition_uses={d.name: [u for u in d.uses if u != d.name] for d in raw_defs},
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
- Write `open` only as `open … in` before a declaration, or inside a
  `section … end`, and never with `renaming` or `hiding`. Do not put the
  theorem in a `namespace`.
- Use Mathlib's usual instances: build no instance in a statement
  (`haveI`, `letI`, `⟨…⟩` or `@` in an instance argument).
- Write no comments and no string literals. Use plain names, never `«…»`.
  Use no `axiom`, `set_option`, `macro`, `notation`, or `#eval`, and no
  name containing "minilab".

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
