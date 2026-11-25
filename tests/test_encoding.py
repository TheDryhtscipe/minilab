"""Tests for the encoding sandbox (build plan doc-5wh, Task 7, first half).

Model-written encoding code runs only in a sandbox: a fresh subprocess under
bwrap, with no network, a read-only filesystem that hides the user's files, and
a time limit. The certificate checker reads back plain values and makes every
comparison in its own process (decision doc-j6t; minilab-1333bc).
"""

import time
from fractions import Fraction
from pathlib import Path

import mpmath
import pytest

import minilab.certify.check as check_module
from minilab.certify import CERTIFICATE, Certificate, check_certificate, check_kill
from minilab.encoding.api import Instance
from minilab.encoding.sandbox import (
    Call,
    ForbiddenSource,
    Sandboxed,
    SandboxFailure,
    check_source,
    run_encoding,
    run_unchecked,
)
from minilab.records import Family

PILOT = Path(__file__).parent / "fixtures" / "pilot"
REPO = Path(__file__).resolve().parents[1]


def source_of(module: str, cls: str, extra: str = "") -> str:
    """A pilot fixture as encoding source: the module, then `ENCODING = cls`."""
    return (PILOT / f"{module}.py").read_text() + extra + f"\n\nENCODING = {cls}\n"


def load(name: str) -> Certificate:
    return CERTIFICATE.validate_json((PILOT / f"{name}.json").read_text())


Y2 = source_of("s7_y2", "Y2Encoding")
S1 = source_of("s1_side", "TsallisPinskerSide")
S7 = source_of("s7", "S7Encoding")

# A dishonest ball margin: the float value, wrapped as an exact arb. On its own
# it certifies the pilot's false grid_gap kill.
FLOAT_BALL = """

class FloatBall(Y2Encoding):
    def margin_ball(self, x: Instance) -> flint.arb:
        c, a1, b = (float(to_fraction(_scalar(x, k))) for k in ("c", "a1", "b"))
        return flint.arb(0.25 - rho_float(c, a1, b))
"""
FLOAT = source_of("s7_y2", "FloatBall", FLOAT_BALL)

POINT = Instance(values={"c": "0", "a1": "1", "b": "1"}, precision_digits=30)


def minimal(body: str, name: str = "Toy") -> str:
    """A small encoding whose margin method is `body`, indented as a method body."""
    lines = "\n".join("        " + line for line in body.strip().splitlines())
    return f"""
import random

import mpmath

from minilab.encoding.api import Instance


class {name}:
    def margin(self, x: Instance) -> mpmath.mpf:
{lines}

    def assumptions(self, x: Instance) -> list[bool]:
        return [True]

    def in_domain(self, x: Instance) -> bool:
        return True

    def near_degenerate(self, x: Instance) -> bool:
        return False

    def sample(self, rng: random.Random) -> Instance:
        return Instance(values={{"t": str(rng.randint(0, 9))}}, precision_digits=16)


ENCODING = {name}
"""


# --- the static import check ---


@pytest.mark.parametrize(
    "line",
    [
        "import os",
        "import socket",
        "import subprocess as sp",
        "from os import path",
        "import importlib",
        "from minilab.records import Record",
        "import minilab.certify.check",
        "from minilab import certify",
        "from . import neighbour",
        "x = __import__('os')",
        "y = eval('1')",
        "z = open('/etc/hostname')",
    ],
)
def test_forbidden_import_rejected(line: str) -> None:
    source = Y2.replace("import math\n", f"import math\n{line}\n", 1)
    assert line in source
    with pytest.raises(ForbiddenSource):
        check_source(source)
    out = run_encoding(source, [Call(method="sample", seed=1, digits=30)], timeout_s=30)
    assert out.failure is not None
    assert "forbidden" in out.failure
    assert out.results == []


@pytest.mark.parametrize(
    "line",
    ["import numpy.linalg", "from mpmath import mp", "import flint as fl", "from math import pi"],
)
def test_allowed_imports_pass(line: str) -> None:
    check_source(Y2.replace("import math\n", f"import math\n{line}\n", 1))


def test_source_must_bind_encoding() -> None:
    with pytest.raises(ForbiddenSource, match="ENCODING"):
        check_source((PILOT / "s7_y2.py").read_text())
    with pytest.raises(ForbiddenSource, match="parse"):
        check_source("def broken(:\n")


# --- the sandbox itself ---


def test_encoding_sandbox_timeout_and_no_network() -> None:
    loop = minimal("while True:\n    pass")
    started = time.monotonic()
    out = run_encoding(loop, [Call(method="margin", point=POINT, digits=30)], timeout_s=2)
    assert time.monotonic() - started < 15
    assert out.failure is not None
    assert "timed out" in out.failure

    # The import check stops `import socket`; run past it to test the sandbox.
    net = minimal(
        "import socket\n"
        "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
        "s.settimeout(3)\n"
        "s.connect(('1.1.1.1', 53))\n"
        "return mpmath.mpf(1)"
    )
    out = run_unchecked(net, [Call(method="margin", point=POINT, digits=30)], timeout_s=30)
    assert out.failure is None, out.failure
    [result] = out.results
    assert result.error is not None
    assert "OSError" in result.error or "unreachable" in result.error
    links = minimal(
        "import os\n"
        "names = [line.split(':')[0].strip() for line in open('/proc/net/dev').readlines()[2:]]\n"
        "if names != ['lo']:\n"
        "    raise RuntimeError('interfaces: ' + ', '.join(names))\n"
        "return mpmath.mpf(1)"
    )
    out = run_unchecked(links, [Call(method="margin", point=POINT, digits=30)], timeout_s=30)
    assert out.failure is None, out.failure
    assert out.results[0].error is None, out.results[0].error

    # A timed-out encoding never gives a verdict: the kill is refused.
    cert = load("grid_gap")
    result = check_kill(
        Sandboxed(loop, Family.claude, timeout_s=2),
        Sandboxed(Y2, Family.codex),
        cert,
        assumptions=2,
    )
    assert not result.accepted
    assert "timed out" in result.reason


def test_sandbox_hides_files_and_blocks_writes() -> None:
    home = Path.home()
    probe = minimal(
        "import os\n"
        "seen = []\n"
        f"for path in ({str(home / '.claude')!r}, {str(home / '.ssh')!r}, '/run', '/var/run'):\n"
        "    try:\n"
        "        os.listdir(path)\n"
        "        seen.append(path)\n"
        "    except OSError:\n"
        "        pass\n"
        f"for where in ({str(REPO)!r}, {str(REPO / 'src')!r}, {str(REPO / '.venv')!r}, '/usr'):\n"
        "    try:\n"
        "        open(where + '/sandbox-write-probe', 'w').write('x')\n"
        "        seen.append('wrote ' + where)\n"
        "    except OSError:\n"
        "        pass\n"
        "if seen:\n"
        "    raise RuntimeError('visible: ' + ', '.join(seen))\n"
        "return mpmath.mpf(1)"
    )
    out = run_unchecked(probe, [Call(method="margin", point=POINT, digits=30)], timeout_s=30)
    assert out.failure is None, out.failure
    [result] = out.results
    assert result.error is None, result.error
    for where in (REPO, REPO / "src", REPO / ".venv"):
        assert not (where / "sandbox-write-probe").exists()


def test_each_run_is_a_fresh_process() -> None:
    # Module state from one run must not reach the next.
    counter = minimal("global RUNS\nRUNS += 1\nreturn mpmath.mpf(RUNS)").replace(
        "class Toy:", "RUNS = 0\n\n\nclass Toy:"
    )
    calls = [Call(method="margin", point=POINT, digits=30)] * 2
    first = run_encoding(counter, calls, timeout_s=30)
    second = run_encoding(counter, calls, timeout_s=30)
    assert first.failure is None and second.failure is None
    assert [r.value for r in first.results] == [r.value for r in second.results]
    assert first.results[0].value != first.results[1].value  # state lives within a run


def test_values_come_back_exact() -> None:
    sample = run_encoding(S1, [Call(method="sample", seed=7, digits=30)], timeout_s=30)
    assert sample.failure is None, sample.failure
    point = sample.results[0].rebuild()
    assert isinstance(point, Instance)
    calls = [
        Call(method=m, point=point, digits=50)
        for m in ("margin", "margin_ball", "assumptions", "in_domain", "near_degenerate")
    ]
    out = run_encoding(S1, calls, timeout_s=30)
    assert out.failure is None, out.failure
    assert set(out.capabilities) == {"margin_ball"}
    margin, ball, holds, inside, degenerate = (r.rebuild() for r in out.results)
    import flint
    import mpmath

    assert type(margin) is mpmath.mpf
    assert type(ball) is flint.arb
    assert holds == [True, True, True, True]
    assert inside is True and degenerate is False
    # The ball and the mpf margin are the same quantity, computed two ways.
    assert abs(float(ball.mid()) - float(margin)) < 1e-12


def test_subclassed_values_are_refused() -> None:
    sly = (
        minimal("return mpmath.mpf(-1)")
        + """

import flint


class Sly(flint.arb):
    def __lt__(self, other: object) -> bool:
        return True


class SlyToy(Toy):
    def margin_ball(self, x: Instance) -> flint.arb:
        return Sly(1)


ENCODING = SlyToy
"""
    )
    out = run_encoding(sly, [Call(method="margin_ball", point=POINT, digits=30)], timeout_s=30)
    assert out.failure is None, out.failure
    value = out.results[0].rebuild()
    import flint

    assert type(value) is not flint.arb
    assert not isinstance(value, flint.arb)


def test_exact_rational_comes_back_exact() -> None:
    exact = (
        minimal("return mpmath.mpf(1)")
        + """

from fractions import Fraction


class Exact(Toy):
    def exact_margin(self, x: Instance) -> Fraction:
        return Fraction(-(10**400), 3)


ENCODING = Exact
"""
    )
    out = run_encoding(exact, [Call(method="exact_margin", point=POINT, digits=30)], timeout_s=30)
    assert out.failure is None, out.failure
    assert out.results[0].rebuild() == Fraction(-(10**400), 3)


# --- the certificate checker through the sandbox ---


def test_sandboxed_checks_match_the_pilot() -> None:
    assert check_certificate(
        Sandboxed(S7, Family.claude), load("s7_counterexample"), assumptions=2
    ).accepted
    result = check_certificate(
        Sandboxed(S7, Family.claude), load("warm_start_artefact"), assumptions=2
    )
    assert not result.accepted
    assert "upper bound on the margin" in result.reason


def test_kill_needs_both_encodings() -> None:
    cert = load("grid_gap")
    # On its own the dishonest encoding certifies the false violation.
    assert check_certificate(Sandboxed(FLOAT, Family.claude), cert, assumptions=2).accepted
    result = check_kill(
        Sandboxed(FLOAT, Family.claude), Sandboxed(Y2, Family.codex), cert, assumptions=2
    )
    assert not result.accepted
    assert "second encoding" in result.reason

    side = load("s1_side_counterexample")
    other = S1.replace("ENCODING =", "# written by the other family\nENCODING =")
    result = check_kill(
        Sandboxed(S1, Family.claude), Sandboxed(other, Family.codex), side, assumptions=4
    )
    assert result.accepted, result.reason


def test_kill_needs_two_families() -> None:
    side = load("s1_side_counterexample")
    other = S1 + "\n# written again\n"
    result = check_kill(
        Sandboxed(S1, Family.claude), Sandboxed(other, Family.claude), side, assumptions=4
    )
    assert not result.accepted
    assert "two families" in result.reason
    result = check_kill(
        Sandboxed(S1, Family.gemini), Sandboxed(other, Family.codex), side, assumptions=4
    )
    assert not result.accepted
    assert "Gemini" in result.reason


def test_kill_needs_sandboxed_encodings() -> None:
    from fixtures.pilot.s1_side import TsallisPinskerSide

    side = load("s1_side_counterexample")
    result = check_kill(
        TsallisPinskerSide,  # type: ignore[arg-type]
        Sandboxed(S1, Family.codex),
        side,
        assumptions=4,
    )
    assert not result.accepted
    assert "sandbox" in result.reason


# Module-level code that reaches the checker through a loophole the import
# check cannot see, and makes check_certificate accept everything.
TAMPER = """

_load = to_fraction.__globals__["__builtins__"]["__import__"]
_check = _load("minilab.certify.check", fromlist=["check_certificate"])
_result = _load("minilab.certify.result", fromlist=["accept"])
_check.check_certificate = lambda *a, **k: _result.accept("tampered")
"""


def test_encoding_cannot_tamper_with_the_checker(monkeypatch: pytest.MonkeyPatch) -> None:
    tamper = source_of("s7_y2", "FloatBall", FLOAT_BALL + TAMPER)
    check_source(tamper)  # the import check does not catch it
    cert = load("grid_gap")

    # In one process the tamper works: run it here and the false kill passes.
    monkeypatch.setattr(check_module, "check_certificate", check_module.check_certificate)
    exec(compile(tamper, "<tamper>", "exec"), {"__name__": "tamper"})  # noqa: S102
    assert check_module.check_certificate is not check_certificate
    assert check_kill(
        Sandboxed(FLOAT, Family.claude), Sandboxed(Y2, Family.codex), cert, assumptions=2
    ).accepted
    monkeypatch.undo()
    assert check_module.check_certificate is check_certificate

    # Through the sandbox it patches only its own process.
    result = check_kill(
        Sandboxed(tamper, Family.claude), Sandboxed(Y2, Family.codex), cert, assumptions=2
    )
    assert not result.accepted
    assert "second encoding refuses" in result.reason


def test_assumption_count_must_match() -> None:
    # S1's encoding gives four predicates; a conjecture with five is not what it encodes.
    side = load("s1_side_counterexample")
    result = check_certificate(Sandboxed(S1, Family.claude), side, assumptions=5)
    assert not result.accepted
    assert "4 assumption predicates" in result.reason
    none = S1.replace(
        "        return [\n            1 < q <= 2,",
        "        return []\n        [\n            1 < q <= 2,",
        1,
    )
    assert none != S1
    result = check_certificate(Sandboxed(none, Family.claude), side, assumptions=4)
    assert not result.accepted
    assert "0 assumption predicates" in result.reason


@pytest.mark.parametrize(
    "raise_line",
    ["raise SystemExit(3)", "raise KeyboardInterrupt", "raise Unprintable()"],
)
def test_sandboxed_kill_answers_whatever_the_encoding_raises(raise_line: str) -> None:
    unprintable = """

class Unprintable(Exception):
    def __str__(self) -> str:
        raise ValueError("cannot print")
"""
    boom = (
        Y2.replace("ENCODING = Y2Encoding", "")
        + unprintable
        + f"""

class Boom(Y2Encoding):
    def margin_ball(self, x: Instance) -> flint.arb:
        {raise_line}


ENCODING = Boom
"""
    )
    result = check_kill(
        Sandboxed(boom, Family.claude), Sandboxed(Y2, Family.codex), load("grid_gap"), assumptions=2
    )
    assert not result.accepted
    assert "first encoding refuses" in result.reason
    assert "raised" in result.reason


def test_sandbox_limits_memory() -> None:
    hog = minimal("block = bytearray(8 * 1024**3)\nreturn mpmath.mpf(len(block))")
    out = run_encoding(hog, [Call(method="margin", point=POINT, digits=30)], timeout_s=30)
    assert out.failure is None, out.failure
    [result] = out.results
    assert result.error is not None
    assert "MemoryError" in result.error


def test_child_rejects_precision_change() -> None:
    # An encoding that pins its own precision would make both checker runs the same.
    pinned = minimal("mpmath.mp.dps = 15\nreturn mpmath.mpf(-1)")
    out = run_encoding(pinned, [Call(method="margin", point=POINT, digits=30)], timeout_s=30)
    assert out.failure is None, out.failure
    [result] = out.results
    assert result.error is not None
    assert "changed the working precision" in result.error


def test_sandbox_limits_tasks_and_tmp() -> None:
    forks = minimal(
        "import os\n"
        "made = 0\n"
        "try:\n"
        "    for _ in range(200):\n"
        "        if os.fork() == 0:\n"
        "            import time\n"
        "            time.sleep(30)\n"
        "            os._exit(0)\n"
        "        made += 1\n"
        "except OSError:\n"
        "    pass\n"
        "return mpmath.mpf(made)"
    )
    out = run_unchecked(forks, [Call(method="margin", point=POINT, digits=30)], timeout_s=20)
    assert out.failure is None, out.failure
    made = out.results[0].rebuild()
    assert isinstance(made, mpmath.mpf)
    assert made < 64

    fill = minimal(
        "written = 0\n"
        "try:\n"
        "    for i in range(20):\n"
        "        with open(f'/tmp/f{i}', 'wb') as f:\n"
        "            f.write(bytes(32 * 1024**2))\n"
        "        written += 1\n"
        "except OSError:\n"
        "    pass\n"
        "return mpmath.mpf(written)"
    )
    out = run_unchecked(fill, [Call(method="margin", point=POINT, digits=30)], timeout_s=30)
    assert out.failure is None, out.failure
    assert out.results[0].rebuild() == mpmath.mpf(2)


def test_parent_never_raises() -> None:
    long_name = "import " + "a" * 5000
    out = run_encoding(long_name, [], timeout_s=30)
    assert out.failure is not None and "forbidden" in out.failure
    deep = (
        "x = " + "-" * 50000 + "1\nENCODING = int\n"
    )  # under the size cap; parsing raises MemoryError
    out = run_encoding(deep, [], timeout_s=30)
    assert out.failure is not None
    from minilab.encoding.sandbox import CallResult
    from minilab.encoding.wire import WireRational

    forged = CallResult(value=WireRational(numerator="0x1", denominator="0x0"))
    with pytest.raises(SandboxFailure, match="cannot be rebuilt"):
        forged.rebuild()
