"""Compile Lean source against the repo's mathlib (build plan doc-5wh, Task 9).

`check_lean` writes the source to a temporary file inside the repo's `lean/`
project, appends `#print axioms` for each theorem it declares, and runs
`lake env lean --json` on it. It reads Lean's messages back and gives a
`LeanResult`.

The runner uses `lake env lean` and not the installed `lean-beam` tools.
lean-beam keeps a Lean server running, but each check is a new file, so the
server imports mathlib again every time; on this machine that took about 12
seconds a check, against about 5 for `lake env lean` (bead minilab-335f82).

Lean source may come from a model, and `#eval` can run IO. So Lean runs under
bwrap: no network, a read-only view of the system, of the Lean toolchain, and
of the `lean/` project, none of the user's other files, and a private /tmp.
The whole process group is killed at the time limit.

A time-out, or a missing toolchain, is never a verdict. A time-out gives
`timed_out=True` and `ok=False` with no axioms; a missing toolchain raises
`LeanUnavailable`.
"""

import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import IO, Literal, cast

from pydantic import BaseModel, ConfigDict, ValidationError

import minilab

DEFAULT_TIMEOUT_S = 300.0

# The most output read back from Lean, in bytes.
MAX_OUTPUT = 16 * 1024**2

# The longest message text kept.
MAX_TEXT = 4000

# Temporary source files are named with this prefix inside `lean/`.
CHECK_PREFIX = "MinilabCheck_"

SORRY_WARNING = "declaration uses 'sorry'"
SORRY_AXIOM = "sorryAx"

Severity = Literal["error", "warning", "information"]


class LeanUnavailable(Exception):
    """Lean could not be run at all: no project, no `lake`, or no bwrap."""


class LeanMessage(BaseModel):
    """One message from Lean. `line` and `column` are 1-based in the source given.

    `axioms_of` names the theorem when the message came from an appended
    `#print axioms` line, not from the source.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    severity: Severity
    line: int
    column: int
    text: str
    axioms_of: str | None = None


class LeanResult(BaseModel):
    """What one compile gave.

    - `ok`: Lean exited cleanly with no error message and did not time out.
      A proof with `sorry` can still be `ok`; check `has_sorry`.
    - `errors`: each error as `"line:column: text"`.
    - `has_sorry`: Lean warned of `sorry`, or a theorem depends on `sorryAx`.
      A proof Lean rejected also depends on `sorryAx`.
    - `axioms`: every axiom any theorem depends on, sorted.
    - `axioms_by_theorem`: the axioms of each theorem, by its full name.
    - `timed_out`: Lean hit the time limit. Nothing else in the result counts.
    - `messages`: every message, errors, warnings, and information.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ok: bool
    errors: list[str]
    has_sorry: bool
    axioms: list[str]
    timed_out: bool
    axioms_by_theorem: dict[str, list[str]]
    messages: list[LeanMessage]


class _RawPosition(BaseModel):
    line: int
    column: int


class _RawMessage(BaseModel):
    severity: str
    pos: _RawPosition
    data: str


# --- finding the theorems ---


def _strip_comments(source: str) -> str:
    """Blank out comments and string literals, keeping every newline in place."""
    out: list[str] = []
    i = 0
    depth = 0
    n = len(source)
    while i < n:
        two = source[i : i + 2]
        ch = source[i]
        if depth > 0:
            if two == "/-":
                depth += 1
                out.append("  ")
                i += 2
            elif two == "-/":
                depth -= 1
                out.append("  ")
                i += 2
            else:
                out.append("\n" if ch == "\n" else " ")
                i += 1
        elif two == "/-":
            depth = 1
            out.append("  ")
            i += 2
        elif two == "--":
            end = source.find("\n", i)
            end = n if end == -1 else end
            out.append(" " * (end - i))
            i = end
        elif ch == '"':
            out.append(" ")
            i += 1
            while i < n and source[i] != '"':
                step = 2 if source[i] == "\\" else 1
                out.append("".join("\n" if c == "\n" else " " for c in source[i : i + step]))
                i += step
            out.append(" ")
            i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


_MODIFIERS = r"(?:(?:private|protected|public|noncomputable|nonrec|unsafe|partial)\s+)*"
_THEOREM = re.compile(
    r"^\s*(?:@\[[^\]]*\]\s*)?" + _MODIFIERS + r"(?:theorem|lemma)\s+([^\s(\[{:⦃]+)"
)
_NAMESPACE = re.compile(r"^\s*namespace\s+(\S+)")
_SECTION = re.compile(r"^\s*" + _MODIFIERS + r"section\b")
_END = re.compile(r"^\s*end\b")


def theorem_names(source: str) -> list[str]:
    """The full names of the theorems and lemmas `source` declares, in order.

    Follows `namespace`, `section`, and `end`, and skips comments.
    """
    scopes: list[str] = []
    names: list[str] = []
    for line in _strip_comments(source).splitlines():
        if match := _NAMESPACE.match(line):
            scopes.append(match.group(1))
        elif _SECTION.match(line):
            scopes.append("")
        elif _END.match(line):
            if scopes:
                scopes.pop()
        elif match := _THEOREM.match(line):
            name = match.group(1)
            if name.startswith("_root_."):
                full = name.removeprefix("_root_.")
            else:
                full = ".".join([s for s in scopes if s] + [name])
            if full not in names:
                names.append(full)
    return names


# --- running Lean ---


def lean_project() -> Path:
    """The repo's `lean/` project, found from where the package lives."""
    return Path(minilab.__file__).resolve().parents[2] / "lean"


def _elan_home() -> Path:
    return Path(os.environ.get("ELAN_HOME", str(Path.home() / ".elan")))


def _lake() -> Path:
    found = shutil.which("lake")
    if found is not None:
        return Path(found)
    lake = _elan_home() / "bin" / "lake"
    if lake.exists():
        return lake
    raise LeanUnavailable("lake is neither on PATH nor in the elan home")


def _read_only_paths(project: Path, lake: Path) -> list[Path]:
    """Paths Lean needs: the toolchain, the project, and the build if it is a link."""
    paths = {_elan_home(), lake.parent, lake.resolve().parent, project}
    build = project / ".lake"
    if build.is_symlink():
        paths.add(build.resolve())
    found: set[Path] = set()
    for path in paths:
        found.add(path)
        found.add(path.resolve())
    kept: list[Path] = []
    for path in sorted(found, key=lambda p: len(p.parts)):
        if path.exists() and not any(path.is_relative_to(k) for k in kept):
            kept.append(path)
    return kept


def _command(project: Path, lake: Path, source_file: Path) -> list[str]:
    if shutil.which("bwrap") is None:
        raise LeanUnavailable("bwrap is not installed; Lean is never run without it")
    args = [
        "bwrap",
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--cap-drop",
        "ALL",
        "--ro-bind",
        "/usr",
        "/usr",
    ]
    for link in ("/bin", "/lib", "/lib64", "/sbin"):
        if os.path.islink(link):
            args += ["--symlink", os.readlink(link), link]
        elif os.path.isdir(link):
            args += ["--ro-bind", link, link]
    for path in ("/etc/ld.so.cache", "/etc/ld.so.conf", "/etc/ld.so.conf.d"):
        args += ["--ro-bind-try", path, path]
    args += ["--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"]
    for path in _read_only_paths(project, lake):
        args += ["--ro-bind", str(path), str(path)]
    args += ["--remount-ro", "/", "--chdir", str(project), "--"]
    args += [str(lake), "env", "lean", "--json", source_file.name]
    return args


def _env(lake: Path) -> dict[str, str]:
    return {
        "PATH": f"{lake.parent}:{_elan_home() / 'bin'}:/usr/bin:/bin",
        "HOME": str(Path.home()),
        "ELAN_HOME": str(_elan_home()),
        "LANG": "C.UTF-8",
    }


def _clip(text: str) -> str:
    return text if len(text) <= MAX_TEXT else text[: MAX_TEXT - 3] + "..."


def _read(stream: IO[bytes]) -> str:
    stream.seek(0)
    return stream.read(MAX_OUTPUT).decode("utf-8", "replace")


def _severity(raw: str) -> Severity:
    if raw in ("error", "warning"):
        return cast(Severity, raw)
    return "information"


_DEPENDS = re.compile(r"depends on axioms:\s*\[(.*)\]", re.DOTALL)
_NO_AXIOMS = "does not depend on any axioms"


def _parse(
    stdout: str, source_lines: int, appended: dict[int, str]
) -> tuple[list[LeanMessage], list[str]]:
    """Lean's JSON messages, and any other lines it printed."""
    messages: list[LeanMessage] = []
    other: list[str] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            raw = _RawMessage.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValidationError):
            other.append(line)
            continue
        number = raw.pos.line
        theorem = appended.get(number)
        if theorem is None and number > source_lines:
            theorem = "(appended line)"
        messages.append(
            LeanMessage(
                severity=_severity(raw.severity),
                line=number,
                column=raw.pos.column + 1,
                text=_clip(raw.data.strip()),
                axioms_of=theorem,
            )
        )
    return messages, other


def _axioms(messages: list[LeanMessage], names: list[str]) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for message in messages:
        if message.axioms_of not in names or message.severity != "information":
            continue
        if _NO_AXIOMS in message.text:
            found[message.axioms_of] = []
        elif match := _DEPENDS.search(message.text):
            parts = [part.strip() for part in match.group(1).split(",")]
            found[message.axioms_of] = sorted(part for part in parts if part)
    return found


def _error_text(message: LeanMessage) -> str:
    if message.axioms_of is not None:
        return f"{message.line}:{message.column}: (axioms of {message.axioms_of}) {message.text}"
    return f"{message.line}:{message.column}: {message.text}"


def check_lean(
    source: str, timeout_s: float = DEFAULT_TIMEOUT_S, *, project: Path | None = None
) -> LeanResult:
    """Compile `source` in the `lean/` project and report errors, `sorry`, and axioms.

    Raise LeanUnavailable if Lean cannot be run at all.
    """
    project = lean_project() if project is None else project
    if not (project / "lakefile.toml").exists():
        raise LeanUnavailable(f"no Lean project at {project}")
    if not (project / ".lake").exists():
        raise LeanUnavailable(f"{project / '.lake'} is missing; run `lake exe cache get`")
    lake = _lake()

    body = source if source.endswith("\n") else source + "\n"
    source_lines = body.count("\n")
    names = theorem_names(body)
    appended: dict[int, str] = {}
    lines = [body, "\n"]
    for offset, name in enumerate(names):
        appended[source_lines + 2 + offset] = name
        lines.append(f"#print axioms {name}\n")
    text = "".join(lines)

    handle, path_text = tempfile.mkstemp(prefix=CHECK_PREFIX, suffix=".lean", dir=project)
    source_file = Path(path_text)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            out.write(text)
        command = _command(project, lake, source_file)
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                env=_env(lake),
                start_new_session=True,
            )
            try:
                code = proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                return LeanResult(
                    ok=False,
                    errors=[],
                    has_sorry=False,
                    axioms=[],
                    timed_out=True,
                    axioms_by_theorem={},
                    messages=[],
                )
            out_text = _read(stdout)
            err_text = _read(stderr)
    finally:
        source_file.unlink(missing_ok=True)

    messages, other = _parse(out_text, source_lines, appended)
    errors = [_error_text(m) for m in messages if m.severity == "error"]
    if code != 0 and not errors:
        detail = "\n".join(other + [err_text.strip()]).strip()
        errors.append(_clip(f"Lean exited with code {code}: {detail}"))
    by_theorem = _axioms(messages, names)
    axioms = sorted({axiom for found in by_theorem.values() for axiom in found})
    has_sorry = SORRY_AXIOM in axioms or any(
        m.severity == "warning" and SORRY_WARNING in m.text for m in messages
    )
    return LeanResult(
        ok=code == 0 and not errors,
        errors=errors,
        has_sorry=has_sorry,
        axioms=axioms,
        timed_out=False,
        axioms_by_theorem=by_theorem,
        messages=messages,
    )
