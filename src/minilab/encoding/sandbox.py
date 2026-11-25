"""Run model-written encoding code in a sandbox (build plan doc-5wh, Task 7).

An encoding is a Python module written by a model. It must bind the name
`ENCODING` to a class whose instances implement the `Encoding` protocol (and
any of the capabilities `exact_margin`, `margin_ball`, `primal_dual`), and it
may import only the modules in `ALLOWED_MODULES`.

Two guards, in order:

1. `check_source` reads the module's syntax and refuses other imports and the
   built-ins that load code or open files. A determined module can get past a
   syntax check, so this only catches honest mistakes early.
2. `run_encoding` runs the module in a fresh child process under bwrap: no
   network, a read-only view of the system and of the Python install, none of
   the user's files, a small private /tmp, and a time limit. It runs in a
   systemd user scope (a cgroup) that caps the memory and the number of tasks
   of the whole process tree, tmpfs pages included; per-process limits alone
   let forked children add up. This is the guard that matters. bwrap was chosen over `unshare -rn`,
   which also works on this machine, because it also hides files and blocks
   writes. Hiding /run matters too: a Unix socket there (such as Docker's)
   stays reachable without a network.

The child sends back plain values (`minilab.encoding.wire`), and the parent
rebuilds them as fresh values of exact library types. `Sandboxed` wraps a
source as an `EncodingFactory` for the certificate checker: each encoding it
builds is one fresh child process, so code in one run cannot patch the checker,
the other encoding, or the next run (decision doc-j6t; minilab-1333bc).
"""

import ast
import os
import random
import resource
import signal
import site
import subprocess
import sys
import tempfile
import types
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import IO, Literal, cast

import flint
import mpmath
from pydantic import BaseModel, ConfigDict, Field, ValidationError

import minilab
from minilab.encoding.api import MAX_DIGITS, Encoding, Instance
from minilab.encoding.wire import WireValue, from_wire
from minilab.records import Family

# Modules an encoding may import, with their submodules. `random` is needed for
# the sampler's signature; the plan's list did not name it.
ALLOWED_MODULES = frozenset(
    {"minilab.encoding.api", "mpmath", "flint", "fractions", "math", "numpy", "random"}
)

# Built-in names that load code or open files.
BANNED_NAMES = frozenset({"__import__", "eval", "exec", "compile", "open", "breakpoint", "input"})

# The optional capabilities an encoding may have, beyond the protocol.
CAPABILITIES = ("exact_margin", "margin_ball", "primal_dual")

Method = Literal[
    "margin",
    "assumptions",
    "in_domain",
    "near_degenerate",
    "sample",
    "exact_margin",
    "margin_ball",
    "primal_dual",
    # A falsifier reformulation's maps and a mechanism's predicate
    # (minilab.falsify). They take a point, like the encoding methods.
    "there",
    "back",
    "predicate",
]

# Limits on the child process tree.
MEMORY_BYTES = 4 * 1024**3
OUTPUT_BYTES = 64 * 1024**2
TMP_BYTES = 64 * 1024**2
MAX_TASKS = 64

# The longest encoding source accepted, in characters.
MAX_SOURCE = 200_000

# The longest failure text kept.
MAX_FAILURE = 3_900

# The default time limit for one sandbox run, in seconds.
DEFAULT_TIMEOUT_S = 60.0


class ForbiddenSource(ValueError):
    """The encoding's source imports or calls something it may not."""


class SandboxFailure(Exception):
    """A sandbox run gave no usable answer: a time-out, a crash, or unreadable output."""


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Call(_Model):
    """One method call: on `point` (or, for `sample`, with a seeded rng), at `digits` digits."""

    method: Method
    point: Instance | None = None
    seed: int | None = None
    digits: int = Field(ge=1, le=2 * MAX_DIGITS)


EncodingCalls = Sequence[Call]


class CallResult(_Model):
    """A call's value, or the error the encoding raised."""

    value: WireValue | None = None
    error: str | None = Field(default=None, max_length=2000)

    def rebuild(self) -> object:
        """Rebuild the value in this process. Raise SandboxFailure for an error."""
        if self.error is not None:
            raise SandboxFailure(self.error)
        if self.value is None:
            raise SandboxFailure("the sandbox sent neither a value nor an error")
        try:
            return from_wire(self.value)
        except Exception as exc:  # noqa: BLE001 - the child wrote this value
            raise SandboxFailure(
                f"the sandbox sent a value that cannot be rebuilt: {describe(exc)}"
            ) from exc


class EncodingOutput(_Model):
    """What a sandbox run gave: one result per call, or a failure for the whole run."""

    results: list[CallResult] = Field(default_factory=lambda: [])
    capabilities: list[str] = Field(default_factory=lambda: [])
    failure: str | None = Field(default=None, max_length=4000)


class Request(_Model):
    source: str
    calls: list[Call]


def _clip(text: str) -> str:
    return text if len(text) <= MAX_FAILURE else text[: MAX_FAILURE - 3] + "..."


def describe(exc: BaseException) -> str:
    """Name an exception from encoding code, whose own text may raise."""
    try:
        text = str(exc)
    except Exception:  # noqa: BLE001 - the text is encoding code too
        text = "(its text raised)"
    return f"{type(exc).__name__}: {text[:500]}"


# --- the syntax check ---


def _allowed(module: str) -> bool:
    return any(module == m or module.startswith(m + ".") for m in ALLOWED_MODULES)


def check_source(source: str) -> None:
    """Refuse a module that imports outside `ALLOWED_MODULES`, uses a banned
    built-in by name, or does not bind `ENCODING` at the top level.

    Raise ForbiddenSource. This is a first check only; the sandbox is the guard.
    """
    if len(source) > MAX_SOURCE:
        raise ForbiddenSource(f"the encoding is longer than {MAX_SOURCE} characters")
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, MemoryError, RecursionError) as exc:
        raise ForbiddenSource(_clip(f"the encoding does not parse: {describe(exc)}")) from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _allowed(alias.name):
                    raise ForbiddenSource(f"forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise ForbiddenSource("forbidden import: a relative import")
            module = node.module or ""
            for alias in node.names:
                # `from minilab.encoding import api` imports minilab.encoding.api.
                if not (_allowed(module) or _allowed(f"{module}.{alias.name}")):
                    raise ForbiddenSource(f"forbidden import: {alias.name} from {module}")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            raise ForbiddenSource(f"forbidden name: {node.id}")
    binds = any(
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "ENCODING" for t in node.targets)
        for node in tree.body
    )
    if not binds:
        raise ForbiddenSource("the encoding must bind ENCODING to its class at the top level")


# --- the sandbox ---


def _bind_paths() -> list[Path]:
    """Paths the child needs, read-only: the Python install, the venv, and minilab."""
    package_root = Path(minilab.__file__).resolve().parents[1]
    paths = {Path(sys.prefix), Path(sys.base_prefix), package_root}
    paths.update(Path(p) for p in site.getsitepackages())
    found: set[Path] = set()
    for path in paths:
        found.add(path)
        found.add(path.resolve())
    # A path inside another is bound with it.
    ordered = sorted(found, key=lambda p: len(p.parts))
    kept: list[Path] = []
    for path in ordered:
        if not any(path.is_relative_to(k) for k in kept):
            kept.append(path)
    return kept


def _command() -> list[str]:
    args = [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "-p",
        f"MemoryMax={MEMORY_BYTES}",
        "-p",
        "MemorySwapMax=0",
        "-p",
        f"TasksMax={MAX_TASKS}",
        "bwrap",
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--cap-drop",
        "ALL",
        "--unsetenv",
        "XDG_RUNTIME_DIR",
        "--unsetenv",
        "DBUS_SESSION_BUS_ADDRESS",
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
    args += ["--dev", "/dev", "--proc", "/proc"]
    for tmp in ("/tmp", "/dev/shm"):
        args += ["--size", str(TMP_BYTES), "--tmpfs", tmp]
    for path in _bind_paths():
        args += ["--ro-bind", str(path), str(path)]
    # The root holding the mount points is a tmpfs; make it read-only too.
    args += ["--remount-ro", "/"]
    args += ["--chdir", "/tmp", "--", sys.executable, "-I", "-B", "-m", "minilab.encoding._child"]
    return args


_ENV = {
    "PATH": "/usr/bin:/bin",
    "HOME": "/tmp",
    "LANG": "C.UTF-8",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


# systemd-run needs these to reach the user's service manager; bwrap unsets
# them for the child.
_BUS_VARIABLES = ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS")


def _env() -> dict[str, str]:
    env = dict(_ENV)
    env.update({k: os.environ[k] for k in _BUS_VARIABLES if k in os.environ})
    return env


def _limits(timeout_s: float) -> None:
    """Set per-process limits before systemd-run starts. Runs between fork and exec.

    The cgroup scope caps the whole tree; these catch one runaway process early.
    """
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
    resource.setrlimit(resource.RLIMIT_FSIZE, (OUTPUT_BYTES, OUTPUT_BYTES))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    cpu = int(timeout_s) + 2
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))


def _tail(stream: IO[bytes]) -> str:
    stream.seek(0, os.SEEK_END)
    size = stream.tell()
    stream.seek(max(0, size - 1000))
    return stream.read().decode("utf-8", "replace").strip()


def _run(request: Request, timeout_s: float) -> EncodingOutput:
    with (
        tempfile.TemporaryFile() as stdin,
        tempfile.TemporaryFile() as stdout,
        tempfile.TemporaryFile() as stderr,
    ):
        stdin.write(request.model_dump_json().encode("utf-8"))
        stdin.seek(0)
        proc = subprocess.Popen(
            _command(),
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            env=_env(),
            start_new_session=True,
            preexec_fn=lambda: _limits(timeout_s),
        )
        try:
            code = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            return EncodingOutput(failure=f"the encoding timed out after {timeout_s:g}s")
        if code != 0:
            return EncodingOutput(
                failure=_clip(f"the sandbox exited with code {code}: {_tail(stderr)[-1500:]}")
            )
        stdout.seek(0)
        raw = stdout.read(OUTPUT_BYTES + 1)
    if len(raw) > OUTPUT_BYTES:
        return EncodingOutput(failure="the sandbox's output is too long")
    try:
        output = EncodingOutput.model_validate_json(raw)
    except ValidationError as exc:
        return EncodingOutput(
            failure=_clip(f"the sandbox's output is unreadable: {str(exc)[:500]}")
        )
    if output.failure is None and len(output.results) != len(request.calls):
        return EncodingOutput(
            failure=f"the sandbox gave {len(output.results)} results for {len(request.calls)} calls"
        )
    return output


def run_encoding(
    source: str, calls: EncodingCalls, timeout_s: float = DEFAULT_TIMEOUT_S
) -> EncodingOutput:
    """Check `source`, then make `calls` on one fresh encoding in a sandboxed child.

    Never raises for anything the encoding does: a forbidden source, a time-out,
    or a crash comes back as `failure`, and an error in one call as that call's
    `error`.
    """
    try:
        check_source(source)
    except ForbiddenSource as exc:
        return EncodingOutput(failure=_clip(str(exc)))
    return run_unchecked(source, calls, timeout_s)


def run_unchecked(
    source: str, calls: EncodingCalls, timeout_s: float = DEFAULT_TIMEOUT_S
) -> EncodingOutput:
    """`run_encoding` without the syntax check, to test the sandbox on its own."""
    return _run(Request(source=source, calls=list(calls)), timeout_s)


# --- the encoding factory for the certificate checker ---

_BASE_METHODS: tuple[Method, ...] = ("assumptions", "in_domain", "near_degenerate", "margin")


class _SandboxRun:
    """One run of an encoding: every call on a point is one fresh child process.

    The first call on a point makes every method call at once, at the point's
    precision, and later calls on the same point read the answers. The checker
    makes all its calls in one run on one point, so one run is one process.
    """

    def __init__(self, sandbox: "Sandboxed", capabilities: tuple[str, ...]) -> None:
        self._sandbox = sandbox
        self._capabilities = capabilities
        self._answers: dict[str, EncodingOutput] = {}

    def _call(self, method: Method, x: Instance) -> object:
        key = x.model_dump_json()
        methods: tuple[Method, ...] = _BASE_METHODS + cast(tuple[Method, ...], self._capabilities)
        if key not in self._answers:
            calls = [Call(method=m, point=x, digits=x.precision_digits) for m in methods]
            self._answers[key] = run_encoding(self._sandbox.source, calls, self._sandbox.timeout_s)
        output = self._answers[key]
        if output.failure is not None:
            raise SandboxFailure(output.failure)
        return output.results[methods.index(method)].rebuild()

    def margin(self, x: Instance) -> mpmath.mpf:
        return cast(mpmath.mpf, self._call("margin", x))

    def assumptions(self, x: Instance) -> list[bool]:
        return cast(list[bool], self._call("assumptions", x))

    def in_domain(self, x: Instance) -> bool:
        return cast(bool, self._call("in_domain", x))

    def near_degenerate(self, x: Instance) -> bool:
        return cast(bool, self._call("near_degenerate", x))

    def sample(self, rng: random.Random) -> Instance:
        seed = rng.getrandbits(63)
        output = run_encoding(
            self._sandbox.source,
            [Call(method="sample", seed=seed, digits=MAX_DIGITS)],
            self._sandbox.timeout_s,
        )
        if output.failure is not None:
            raise SandboxFailure(output.failure)
        value = output.results[0].rebuild()
        if type(value) is not Instance:
            raise SandboxFailure("the sampler did not give an Instance")
        return value


class _Exact(_SandboxRun):
    def exact_margin(self, x: Instance) -> object:
        return self._call("exact_margin", x)


class _Ball(_SandboxRun):
    def margin_ball(self, x: Instance) -> flint.arb:
        return cast(flint.arb, self._call("margin_ball", x))


class _Dual(_SandboxRun):
    def primal_dual(self, x: Instance) -> tuple[flint.arb, flint.arb]:
        return cast(tuple[flint.arb, flint.arb], self._call("primal_dual", x))


_MIXINS: dict[str, type[_SandboxRun]] = {
    "exact_margin": _Exact,
    "margin_ball": _Ball,
    "primal_dual": _Dual,
}


@cache
def _run_class(capabilities: tuple[str, ...]) -> type[_SandboxRun]:
    """A class with exactly the capability methods the encoding has.

    The checker asks `isinstance(enc, BallMargin)` and so on, which looks for
    the method by name.
    """
    bases = tuple(_MIXINS[c] for c in capabilities) or (_SandboxRun,)
    return cast(type[_SandboxRun], types.new_class("SandboxedEncoding", bases))


@dataclass(frozen=True)
class Sandboxed:
    """An encoding's source as an `EncodingFactory`: each call builds a fresh run.

    `family` is the model family that wrote the source; the caller
    (encode_twice) fills it in, and `check_kill` needs two different ones. Pass
    two of these to `check_kill`. The capabilities are read once, by a probe
    run; a later run that lacks one gives an error for that call.
    """

    source: str
    family: Family
    timeout_s: float = DEFAULT_TIMEOUT_S
    _probe: list[tuple[str, ...]] = field(default_factory=lambda: [], compare=False, repr=False)

    def _capabilities(self) -> tuple[str, ...]:
        if not self._probe:
            output = run_encoding(self.source, [], self.timeout_s)
            found = (
                () if output.failure else tuple(c for c in CAPABILITIES if c in output.capabilities)
            )
            self._probe.append(found)
        return self._probe[0]

    def __call__(self) -> Encoding:
        run = _run_class(self._capabilities())(self, self._capabilities())
        return cast(Encoding, run)
