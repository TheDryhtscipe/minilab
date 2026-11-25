"""Compile Lean source against the repo's mathlib (build plan doc-5wh, Task 9).

`check_lean` writes the source to a temporary file inside the repo's `lean/`
project and runs `lake env lean --json` on it. It reads Lean's messages back
and gives a `LeanResult`.

Axioms are not found from the source text. The runner puts `import Lean`
before the source and appends one Lean command that walks every constant the
file declares and prints the axioms of each. So a theorem in any form, an
`axiom` declaration, and a definition all count. If that command gives no
result (for example after `#exit`), the result is not `ok`: a missing axiom
check never reads as "no axioms".

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

    `axioms_of` is set when the message came from the appended axiom check,
    not from the source.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    severity: Severity
    line: int
    column: int
    text: str
    axioms_of: str | None = None


class LeanResult(BaseModel):
    """What one compile gave.

    - `ok`: Lean exited cleanly with no error message, did not time out, and
      the axiom check ran. A proof with `sorry` can still be `ok`; check
      `has_sorry`.
    - `errors`: each error as `"line:column: text"`.
    - `has_sorry`: Lean warned of `sorry`, or a theorem depends on `sorryAx`.
      A proof Lean rejected also depends on `sorryAx`.
    - `axioms`: every axiom any declaration in the source depends on, sorted.
      An `axiom` the source declares is listed even if nothing uses it.
    - `axioms_by_theorem`: the axioms of each theorem, by its full name.
      Private theorems are listed by the name they were declared with.
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


class _Constant(BaseModel):
    """One constant from the axiom check: its name, theorem or not, and axioms."""

    name: str
    theorem: bool
    internal: bool
    axioms: list[str]


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


# The line put before the source, so the axiom check can use Lean's meta API.
# Mathlib imports Lean already, so for Mathlib sources this costs nothing.
IMPORT_LEAN = "import Lean\n"

# The axiom check prints one information message that starts with this.
AXIOM_MARKER = "MINILAB_AXIOMS "

# Appended after the source. It walks every constant the file declares (not
# the imported ones) and prints, for each, its name with any private prefix
# removed, whether it is a theorem, whether the name is internal, and the
# axioms it depends on. Names are written in full so that the source's own
# `open`s and namespaces do not change what they mean.
AXIOM_CHECK = """
#eval show Lean.Elab.Command.CommandElabM Unit from do
  let env ← Lean.getEnv
  let mut out : Array Lean.Json := #[]
  for (name, info) in env.constants.map₂.toList do
    let found ← Lean.collectAxioms name
    let user := (Lean.privateToUserName? name).getD name
    let isTheorem := match info with
      | Lean.ConstantInfo.thmInfo _ => true
      | _ => false
    out := out.push (Lean.Json.mkObj [
      ("name", Lean.toJson user.toString),
      ("theorem", Lean.toJson isTheorem),
      ("internal", Lean.toJson user.isInternal),
      ("axioms", Lean.toJson (found.map Lean.Name.toString))])
  Lean.logInfo (Lean.MessageData.ofFormat
    (Std.Format.text ("MINILAB_AXIOMS " ++ (Lean.Json.arr out).compress)))
"""

_AXIOM_CHECK_LABEL = "(axiom check)"


def _parse(
    stdout: str, source_lines: int
) -> tuple[list[LeanMessage], list[str], list[list[_Constant]]]:
    """Lean's JSON messages, any other lines it printed, and the axiom check results.

    Message lines are moved back by the one line of `IMPORT_LEAN`. Only an
    information message from the appended lines counts as an axiom check
    result, so the source cannot print one of its own.
    """
    messages: list[LeanMessage] = []
    other: list[str] = []
    checks: list[list[_Constant]] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            raw = _RawMessage.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValidationError):
            other.append(line)
            continue
        number = raw.pos.line - 1
        appended = number > source_lines
        severity = _severity(raw.severity)
        text = raw.data.strip()
        if appended and severity == "information" and text.startswith(AXIOM_MARKER):
            try:
                found = json.loads(text.removeprefix(AXIOM_MARKER))
                checks.append(
                    [_Constant.model_validate(item) for item in cast(list[object], found)]
                )
                continue
            except (json.JSONDecodeError, ValidationError, TypeError):
                pass
        messages.append(
            LeanMessage(
                severity=severity,
                line=number,
                column=raw.pos.column + 1,
                text=_clip(text),
                axioms_of=_AXIOM_CHECK_LABEL if appended else None,
            )
        )
    return messages, other, checks


def _error_text(message: LeanMessage) -> str:
    if message.axioms_of is not None:
        return f"{message.line}:{message.column}: {message.axioms_of} {message.text}"
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
    text = IMPORT_LEAN + body + AXIOM_CHECK

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

    messages, other, checks = _parse(out_text, source_lines)
    errors = [_error_text(m) for m in messages if m.severity == "error"]
    if code != 0 and not errors:
        detail = "\n".join(other + [err_text.strip()]).strip()
        errors.append(_clip(f"Lean exited with code {code}: {detail}"))
    if len(checks) != 1:
        errors.append(
            f"the axiom check gave {len(checks)} results, not 1, so the axioms are unknown"
        )
    constants = checks[0] if len(checks) == 1 else []
    by_theorem = {c.name: sorted(c.axioms) for c in constants if c.theorem and not c.internal}
    axioms = sorted({axiom for c in constants for axiom in c.axioms})
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
