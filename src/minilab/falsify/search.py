"""Falsifier search: try to drive a conjecture's margin below zero, then certify.

Search may suggest a kill; only a certificate makes one (design doc-gmj,
"Falsifier"). `attack` runs three searches on an encoding:

1. **Cold starts.** `starts` fresh samples from the encoding's sampler, each
   drawn from its own seed.
2. **Local optimisation.** From each start that is in the domain and meets
   every assumption, Nelder-Mead on the margin, with a penalty outside the
   domain. Every run starts cold from its own sample: nothing carries from
   one run to the next. The runs go in step: each round, every run that has
   not finished asks for one point, and one batch evaluates them all
   (decision doc-2nx). The optimiser only suggests points; a certificate is
   computed afresh by the library checker.
3. **Boundary sampling.** From valid points, rays in random directions: each
   ray is followed until it leaves the domain, then bisected to the last valid
   point before the edge.

The points with a negative margin are candidates. For the lowest few,
`build_certificate` asks the encoding for its exact margin, ball, or primal
and dual bounds at the certificate's precision, and the caller's certifier
decides. In the pipeline the certifier is `check_kill`, which asks both
encodings (decision doc-j6t). With no kill, the result carries a
`CoverageReport`.

An encoding runs in this process (`LocalEvaluator`, for library and test
encodings) or in the sandbox (`SandboxEvaluator`, for model-written ones).
The sandbox evaluates a batch of points in one child process, so samples,
boundary rays, and each round of the local runs are batched.
"""

import json
import math
import random
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import partial
from typing import Protocol, cast

import flint
import mpmath
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from minilab.certify import (
    Certificate,
    CheckResult,
    DualityCertificate,
    ExactCertificate,
    IntervalCertificate,
    check_certificate,
)
from minilab.certify.duality import MIN_DUALITY_DIGITS
from minilab.certify.result import working_digits
from minilab.encoding.agree import BATCH_CALLS, SandboxUnavailable
from minilab.encoding.api import (
    MAX_DIGITS,
    BallMargin,
    DualBounds,
    Encoding,
    EncodingFactory,
    ExactMargin,
    Instance,
    Value,
    to_fraction,
)
from minilab.encoding.sandbox import (
    DEFAULT_TIMEOUT_S,
    Call,
    CallResult,
    EncodingOutput,
    Method,
    Sandboxed,
    SandboxFailure,
    describe,
    run_encoding,
)
from minilab.falsify.coverage import CoverageReport
from minilab.records import Family

# Decides whether a certificate makes a kill. In the pipeline: `check_kill`
# with both encodings; for one trusted encoding: `check_certificate`.
Certifier = Callable[[Certificate], CheckResult]

# The working precision for drawing samples, in decimal digits.
SAMPLE_DIGITS = 50

# The objective outside the domain, or where the encoding fails.
PENALTY = 1e300

# Ray lengths tried along each boundary ray, as multiples of the direction.
RAY_STEPS = tuple(2.0**k for k in range(-6, 4))

# Bisection steps towards the edge of the domain on each ray.
BISECTIONS = 10

_PROBE = "class Probe:\n    pass\n\n\nENCODING = Probe\n"


class SearchBudget(BaseModel):
    """How hard the search tries. The defaults are the builder's choice (doc-gmj)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    starts: int = Field(default=16, ge=1)
    """Cold starts: fresh samples, each the start of one local optimisation."""
    local_evaluations: int = Field(default=100, ge=0)
    """The most points one local optimisation may evaluate."""
    boundary_rays: int = Field(default=16, ge=0)
    """Rays followed to the edge of the domain."""
    max_certify: int = Field(default=5, ge=1)
    """The most candidates sent to the certifier."""
    digits: int = Field(default=30, ge=1, le=MAX_DIGITS)
    """The precision of a certificate, in decimal digits."""


# --- evaluating points ---


@dataclass(frozen=True)
class Evaluation:
    """One point of the search space, and what the conjecture's encoding says there.

    `original` is the point in the conjecture's own variables: `point` itself,
    or its image under a reformulation's map back. `margin` is set only for a
    valid point: in the domain, with every assumption met.
    """

    point: Instance
    original: Instance | None
    valid: bool
    margin: mpmath.mpf | None
    error: str | None = None


def _failed(point: Instance, error: str, original: Instance | None = None) -> Evaluation:
    return Evaluation(point=point, original=original, valid=False, margin=None, error=error)


class Evaluator(Protocol):
    """Draws and evaluates points of a search space."""

    def sample(self, seeds: Sequence[int]) -> list[Instance | None]:
        """One sample per seed, or None where the sampler failed."""
        ...

    def evaluate(self, points: Sequence[Instance]) -> list[Evaluation]:
        """One evaluation per point, in order."""
        ...


def _finite_mpf(value: object) -> mpmath.mpf | None:
    if type(value) is mpmath.mpf and mpmath.isfinite(value):
        return value
    return None


def _judge(point: Instance, inside: object, held: object, margin: object) -> Evaluation:
    """Turn what an encoding returned for `point` into an evaluation."""
    if type(inside) is not bool:
        return _failed(point, "in_domain did not give a bool")
    if not isinstance(held, list) or not all(type(h) is bool for h in cast(list[object], held)):
        return _failed(point, "assumptions did not give a list of bools")
    if not inside or not all(cast(list[bool], held)):
        return Evaluation(point=point, original=point, valid=False, margin=None)
    value = _finite_mpf(margin)
    if value is None:
        return _failed(point, f"the margin is not a finite mpf: {type(margin).__name__}")
    return Evaluation(point=point, original=point, valid=True, margin=value)


class LocalEvaluator:
    """Evaluates an encoding in this process: for library and test encodings only."""

    def __init__(self, enc: Encoding) -> None:
        self.enc = enc

    def sample(self, seeds: Sequence[int]) -> list[Instance | None]:
        out: list[Instance | None] = []
        for seed in seeds:
            try:
                with working_digits(SAMPLE_DIGITS):
                    value = cast(object, self.enc.sample(random.Random(seed)))
            except Exception:  # noqa: BLE001 - encoding code may raise anything
                value = None
            out.append(value if type(value) is Instance else None)
        return out

    def evaluate(self, points: Sequence[Instance]) -> list[Evaluation]:
        out: list[Evaluation] = []
        for x in points:
            try:
                with working_digits(x.precision_digits):
                    inside = cast(object, self.enc.in_domain(x))
                    held = cast(object, self.enc.assumptions(x))
                    valid = (
                        inside is True
                        and isinstance(held, list)
                        and all(h is True for h in cast(list[object], held))
                    )
                    margin = cast(object, self.enc.margin(x)) if valid else None
                    out.append(_judge(x, inside, cast(object, held), margin))
            except Exception as exc:  # noqa: BLE001 - encoding code may raise anything
                out.append(_failed(x, f"the encoding raised {describe(exc)}"))
        return out


def _sandbox_ok() -> None:
    """Raise SandboxUnavailable unless the sandbox runs an empty encoding."""
    try:
        output = run_encoding(_PROBE, [], DEFAULT_TIMEOUT_S)
    except OSError as exc:
        raise SandboxUnavailable(f"the sandbox cannot start: {exc}") from exc
    if output.failure is not None:
        raise SandboxUnavailable(f"the sandbox cannot run an empty encoding: {output.failure}")


def sandbox_calls(
    source: str, calls: Sequence[Call], timeout_s: float, group: int = 1
) -> list[CallResult]:
    """Make `calls` on `source` in runs of at most BATCH_CALLS, each a fresh sandbox.

    If a run fails as a whole (a time-out, a crash, or bad output), one call in
    it may be the cause, so the run's calls are made again, `group` calls to a
    sandbox, and each group gets only its own failure. That retry happens only
    if `source` loads in a sandbox with no calls; if it does not, every call in
    the run gets the run's failure. If the sandbox cannot run an empty encoding
    either, raise SandboxUnavailable.
    """
    results: list[CallResult] = []
    checked = False
    loads: bool | None = None

    def run(chunk: Sequence[Call]) -> EncodingOutput:
        try:
            return run_encoding(source, chunk, timeout_s)
        except OSError as exc:
            raise SandboxUnavailable(f"the sandbox cannot start: {exc}") from exc

    def failed(chunk: Sequence[Call], failure: str) -> list[CallResult]:
        return [CallResult(error=failure[:2000]) for _ in chunk]

    for start in range(0, len(calls), BATCH_CALLS):
        chunk = calls[start : start + BATCH_CALLS]
        output = run(chunk)
        if output.failure is None:
            results.extend(output.results)
            continue
        if not checked:
            _sandbox_ok()
            checked = True
        if len(chunk) > group and loads is None:
            loads = run([]).failure is None
        if len(chunk) <= group or not loads:
            results.extend(failed(chunk, output.failure))
            continue
        for at in range(0, len(chunk), group):
            part = chunk[at : at + group]
            again = run(part)
            results.extend(again.results if again.failure is None else failed(part, again.failure))
    return results


def rebuild(result: CallResult, digits: int) -> tuple[object, str | None]:
    """Rebuild a sandbox call's value at `digits` digits: (value, None) or (None, error)."""
    try:
        with working_digits(digits):
            return result.rebuild(), None
    except SandboxFailure as exc:
        return None, str(exc)


def sandbox_samples(
    source: str, seeds: Sequence[int], timeout_s: float
) -> tuple[list[Instance | None], str | None]:
    """Draw one sample per seed in the sandbox; also return the first error, if any."""
    calls = [Call(method="sample", seed=s, digits=SAMPLE_DIGITS) for s in seeds]
    out: list[Instance | None] = []
    first: str | None = None
    for result in sandbox_calls(source, calls, timeout_s):
        value, error = rebuild(result, SAMPLE_DIGITS)
        if error is None and type(value) is not Instance:
            error = "the sampler did not give an Instance"
        first = first or error
        out.append(value if type(value) is Instance else None)
    return out, first


def sandbox_points(
    source: str, method: Method, points: Sequence[Instance | None], timeout_s: float
) -> tuple[list[object], list[str | None]]:
    """Call `method` on each point in the sandbox. A None point gives an error."""
    calls = [Call(method=method, point=x, digits=x.precision_digits) for x in points if x]
    results = iter(sandbox_calls(source, calls, timeout_s))
    values: list[object] = []
    errors: list[str | None] = []
    for x in points:
        if x is None:
            values.append(None)
            errors.append("no point to map")
            continue
        value, error = rebuild(next(results), x.precision_digits)
        values.append(value)
        errors.append(error)
    return values, errors


class SandboxEvaluator:
    """Evaluates a model-written encoding in the sandbox, a batch per child process."""

    def __init__(self, source: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self.source = source
        self.timeout_s = timeout_s

    def sample(self, seeds: Sequence[int]) -> list[Instance | None]:
        return sandbox_samples(self.source, seeds, self.timeout_s)[0]

    def evaluate(self, points: Sequence[Instance]) -> list[Evaluation]:
        methods: tuple[Method, ...] = ("in_domain", "assumptions", "margin")
        calls = [
            Call(method=m, point=x, digits=x.precision_digits) for x in points for m in methods
        ]
        results = sandbox_calls(self.source, calls, self.timeout_s, group=len(methods))
        out: list[Evaluation] = []
        for i, x in enumerate(points):
            got = [rebuild(r, x.precision_digits) for r in results[3 * i : 3 * i + 3]]
            (inside, e1), (held, e2), (margin, _) = got
            if e1 or e2:
                out.append(_failed(x, cast(str, e1 or e2)))
            else:
                out.append(_judge(x, inside, held, margin))
        return out


def evaluator_for(make: EncodingFactory) -> Evaluator:
    """A sandboxed encoding runs in the sandbox; any other in this process."""
    if isinstance(make, Sandboxed):
        return SandboxEvaluator(make.source, make.timeout_s)
    return LocalEvaluator(make())


# --- points as vectors ---

Coordinate = tuple[str, int | None]
"""A parameter name and, for a list value, the index of the entry."""


def _numeric(value: Fraction | str) -> Fraction | None:
    try:
        return to_fraction(value)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def coordinates(x: Instance) -> dict[Coordinate, Fraction]:
    """The numeric entries of a point, by coordinate."""
    found: dict[Coordinate, Fraction] = {}
    for key in sorted(x.values):
        value = x.values[key]
        items: list[tuple[int | None, Fraction | str]] = (
            list(enumerate(value)) if isinstance(value, list) else [(None, value)]
        )
        for index, item in items:
            number = _numeric(item)
            if number is not None:
                found[(key, index)] = number
    return found


@dataclass(frozen=True)
class Layout:
    """How one point maps to a vector: its numeric coordinates, in order.

    Coordinates that were integers in every sample stay integers; entries that
    are not numbers stay at the template's values.
    """

    template: Instance
    coords: tuple[Coordinate, ...]
    integer: tuple[bool, ...]

    @classmethod
    def of(cls, x: Instance, integral: set[Coordinate]) -> "Layout":
        found = coordinates(x)
        coords = tuple(found)
        return cls(template=x, coords=coords, integer=tuple(c in integral for c in coords))

    def vector(self, x: Instance | None = None) -> npt.NDArray[np.float64]:
        found = coordinates(x or self.template)
        return np.array([float(found[c]) for c in self.coords], dtype=np.float64)

    def build(self, v: npt.NDArray[np.float64]) -> Instance | None:
        """The point at `v`, with exact decimal values; None if `v` is not finite."""
        values: dict[str, Value] = {
            k: list(val) if isinstance(val, list) else val
            for k, val in self.template.values.items()
        }
        for (key, index), number, whole in zip(self.coords, v.tolist(), self.integer, strict=True):
            if not math.isfinite(number):
                return None
            text = str(round(number)) if whole else repr(float(number))
            if index is None:
                values[key] = text
            else:
                entries = values[key]
                assert isinstance(entries, list)
                entries[index] = text
        return Instance(values=values, precision_digits=self.template.precision_digits)


def _key(x: Instance) -> str:
    """A point's values, as text: two points with the same values have the same key."""
    return json.dumps({k: str(v) for k, v in sorted(x.values.items())}, sort_keys=True)


# --- the search ---


@dataclass
class SearchOutcome:
    """Every point a search evaluated, and what it found."""

    evaluations: list[Evaluation]
    starts: int
    invalid_starts: int
    dimensions: int
    boundary_points: int
    local_evaluations: int
    budget: SearchBudget
    sample_failures: int

    def candidates(self) -> list[Evaluation]:
        """Valid points with a negative margin, lowest first, one per original point."""
        seen: set[str] = set()
        out: list[Evaluation] = []
        found = [e for e in self.evaluations if e.valid and e.margin is not None and e.margin < 0]
        for e in sorted(found, key=lambda e: cast(mpmath.mpf, e.margin)):
            if e.original is None:
                continue
            key = _key(e.original)
            if key not in seen:
                seen.add(key)
                out.append(e)
        return out


class _Tracker:
    """Evaluates points once each and keeps every evaluation."""

    def __init__(self, evaluator: Evaluator) -> None:
        self.evaluator = evaluator
        self.cache: dict[str, Evaluation] = {}
        self.order: list[Evaluation] = []

    def evaluate(self, points: Sequence[Instance]) -> list[Evaluation]:
        keys = [_key(x) for x in points]
        fresh: dict[str, Instance] = {}
        for key, x in zip(keys, points, strict=True):
            if key not in self.cache and key not in fresh:
                fresh[key] = x
        if fresh:
            results = self.evaluator.evaluate(list(fresh.values()))
            for key, e in zip(fresh, results, strict=True):
                self.cache[key] = e
                self.order.append(e)
        return [self.cache[key] for key in keys]


def _objective(e: Evaluation) -> float:
    if not e.valid or e.margin is None:
        return PENALTY
    return max(-PENALTY, min(PENALTY, float(e.margin)))


# Nelder-Mead coefficients: reflection, expansion, contraction, and shrink
# (the standard values, as in scipy's default). The steps are scipy's too, except
# that ties in the ordering keep their order (a stable sort): many points score
# PENALTY, so ties are common here.
_RHO, _CHI, _PSI, _SIGMA = 1.0, 2.0, 0.5, 0.5
_XATOL, _FATOL = 1e-12, 1e-15

Vector = npt.NDArray[np.float64]
Steps = Generator[Vector, float]


class _Spent(Exception):
    """A Nelder-Mead run has asked for all the points its budget allows."""


def _nelder_mead(simplex: Vector, max_evaluations: int) -> Steps:
    """Nelder-Mead from `simplex`, one point at a time.

    The generator yields each point it wants scored and takes the score
    through `send`. It stops after `max_evaluations` points, or once the
    simplex has shrunk below the tolerances.
    """
    asked = 0

    def ask(v: Vector) -> Generator[Vector, float, float]:
        nonlocal asked
        if asked >= max_evaluations:
            raise _Spent
        asked += 1
        return (yield v.copy())

    sim = simplex.copy()
    n = sim.shape[1]
    try:
        fsim = np.empty(n + 1, dtype=np.float64)
        for j in range(n + 1):
            fsim[j] = yield from ask(sim[j])
        while True:
            order = np.argsort(fsim, kind="stable")
            sim, fsim = sim[order], fsim[order]
            if (
                float(np.max(np.abs(sim[1:] - sim[0]))) <= _XATOL
                and float(np.max(np.abs(fsim[1:] - fsim[0]))) <= _FATOL
            ):
                return
            centre = sim[:-1].mean(axis=0)
            worst = sim[-1]
            xr = (1 + _RHO) * centre - _RHO * worst
            fr = yield from ask(xr)
            if fr < fsim[0]:
                xe = (1 + _RHO * _CHI) * centre - _RHO * _CHI * worst
                fe = yield from ask(xe)
                sim[-1], fsim[-1] = (xe, fe) if fe < fr else (xr, fr)
                continue
            if fr < fsim[-2]:
                sim[-1], fsim[-1] = xr, fr
                continue
            if fr < fsim[-1]:
                xc = (1 + _PSI * _RHO) * centre - _PSI * _RHO * worst
                fc = yield from ask(xc)
                if fc <= fr:
                    sim[-1], fsim[-1] = xc, fc
                    continue
            else:
                xc = (1 - _PSI) * centre + _PSI * worst
                fc = yield from ask(xc)
                if fc < fsim[-1]:
                    sim[-1], fsim[-1] = xc, fc
                    continue
            for j in range(1, n + 1):
                sim[j] = sim[0] + _SIGMA * (sim[j] - sim[0])
                fsim[j] = yield from ask(sim[j])
    except _Spent:
        return


def _local(
    tracker: _Tracker,
    runs: Sequence[tuple[Layout, Vector]],
    max_evaluations: int,
) -> int:
    """Run Nelder-Mead from each layout's own point, all runs in step.

    Each round, every run that has not finished asks for one point, and one
    tracker batch evaluates them, in the order of `runs`. Return the number of
    points the runs asked for.
    """
    live: list[tuple[Layout, Steps, Vector]] = []
    for layout, steps in runs:
        x0 = layout.vector()
        simplex = np.vstack([x0] + [x0 + np.eye(len(x0))[i] * steps[i] for i in range(len(x0))])
        gen = _nelder_mead(simplex, max_evaluations)
        try:
            live.append((layout, gen, next(gen)))
        except StopIteration:
            continue
    asked = 0
    while live:
        asked += len(live)
        built = [layout.build(v) for layout, _, v in live]
        scored = iter(tracker.evaluate([x for x in built if x is not None]))
        following: list[tuple[Layout, Steps, Vector]] = []
        for (layout, gen, _), x in zip(live, built, strict=True):
            value = PENALTY if x is None else _objective(next(scored))
            try:
                following.append((layout, gen, gen.send(value)))
            except StopIteration:
                continue
        live = following
    return asked


def _spread(points: Sequence[Instance]) -> dict[Coordinate, float]:
    """The range of each coordinate over `points`, at least a small floor."""
    low: dict[Coordinate, float] = {}
    high: dict[Coordinate, float] = {}
    for x in points:
        for c, number in coordinates(x).items():
            value = float(number)
            low[c] = min(low.get(c, value), value)
            high[c] = max(high.get(c, value), value)
    return {c: high[c] - low[c] for c in low}


def _integral(points: Sequence[Instance]) -> set[Coordinate]:
    """Coordinates that are whole numbers in every point that has them."""
    whole: dict[Coordinate, bool] = {}
    for x in points:
        for c, number in coordinates(x).items():
            whole[c] = whole.get(c, True) and number.denominator == 1
    return {c for c, w in whole.items() if w}


def _steps(layout: Layout, spread: dict[Coordinate, float]) -> npt.NDArray[np.float64]:
    """A first step for each coordinate: a tenth of its spread, or of its size."""
    x0 = layout.vector()
    out: list[float] = []
    for c, value, whole in zip(layout.coords, x0.tolist(), layout.integer, strict=True):
        step = max(spread.get(c, 0.0) / 10, abs(value) / 10, 1e-2)
        out.append(max(1.0, round(step)) if whole else step)
    return np.array(out, dtype=np.float64)


def _boundary(
    tracker: _Tracker,
    bases: Sequence[Evaluation],
    integral: set[Coordinate],
    spread: dict[Coordinate, float],
    rays: int,
    rng: random.Random,
) -> int:
    """Follow `rays` random rays from valid points to the edge of the domain.

    Return the number of edge points found.
    """
    if not bases or rays == 0:
        return 0
    lines: list[tuple[Layout, npt.NDArray[np.float64], npt.NDArray[np.float64]]] = []
    for _ in range(rays):
        base = rng.choice(list(bases))
        layout = Layout.of(base.point, integral)
        if not layout.coords:
            continue
        x0 = layout.vector()
        scale = [
            max(spread.get(c, 0.0), abs(v), 1e-3)
            for c, v in zip(layout.coords, x0.tolist(), strict=True)
        ]
        direction = np.array([rng.gauss(0.0, 1.0) * s for s in scale], dtype=np.float64)
        lines.append((layout, x0, direction))
    # Step out along each ray, all rays in one batch.
    grid: list[tuple[int, float, Instance]] = []
    for r, (layout, x0, direction) in enumerate(lines):
        for t in RAY_STEPS:
            point = layout.build(x0 + t * direction)
            if point is not None:
                grid.append((r, t, point))
    results = tracker.evaluate([p for _, _, p in grid])
    brackets: dict[int, tuple[float, float]] = {}
    last_valid: dict[int, float] = {}
    for (r, t, _), e in zip(grid, results, strict=True):
        if r in brackets:
            continue
        if e.valid:
            last_valid[r] = t
        else:
            brackets[r] = (last_valid.get(r, 0.0), t)
    # Bisect every bracketed ray, one batch per step.
    for _ in range(BISECTIONS):
        middles: list[tuple[int, float, Instance]] = []
        for r, (lo, hi) in brackets.items():
            layout, x0, direction = lines[r]
            point = layout.build(x0 + (lo + hi) / 2 * direction)
            if point is not None:
                middles.append((r, (lo + hi) / 2, point))
        if not middles:
            break
        for (r, t, _), e in zip(middles, tracker.evaluate([p for _, _, p in middles]), strict=True):
            lo, hi = brackets[r]
            brackets[r] = (t, hi) if e.valid else (lo, t)
    found = 0
    for r, (lo, _) in brackets.items():
        layout, x0, direction = lines[r]
        point = layout.build(x0 + lo * direction)
        if point is not None and lo > 0 and tracker.evaluate([point])[0].valid:
            found += 1
    return found


def search(evaluator: Evaluator, budget: SearchBudget, rng: random.Random) -> SearchOutcome:
    """Cold starts, local optimisation, and boundary rays, as the module describes."""
    tracker = _Tracker(evaluator)
    seeds = [rng.getrandbits(63) for _ in range(budget.starts)]
    drawn = evaluator.sample(seeds)
    samples = [x for x in drawn if x is not None]
    starts = tracker.evaluate(samples)
    valid_starts = [e for e in starts if e.valid]
    integral = _integral(samples)
    spread = _spread(samples)
    dimensions = 0
    asked = 0
    if budget.local_evaluations > 0:
        runs: list[tuple[Layout, Vector]] = []
        for e in valid_starts:
            layout = Layout.of(e.point, integral)
            if not layout.coords:
                continue
            dimensions = max(dimensions, len(layout.coords))
            runs.append((layout, _steps(layout, spread)))
        asked = _local(tracker, runs, budget.local_evaluations)
    else:
        dimensions = max((len(coordinates(x)) for x in samples), default=0)
    edge = _boundary(tracker, valid_starts, integral, spread, budget.boundary_rays, rng)
    return SearchOutcome(
        evaluations=tracker.order,
        starts=len(samples),
        invalid_starts=len(samples) - len(valid_starts),
        dimensions=dimensions,
        boundary_points=edge,
        local_evaluations=asked,
        budget=budget,
        sample_failures=len(drawn) - len(samples),
    )


# --- coverage ---


def _show(value: float) -> str:
    return f"{value:.6g}"


def coverage_of(outcome: SearchOutcome) -> CoverageReport:
    """The coverage report for a search: what it reached and what it did not."""
    valid = [e for e in outcome.evaluations if e.valid and e.margin is not None]
    low: dict[str, float] = {}
    high: dict[str, float] = {}
    lengths: dict[str, set[int]] = {}
    fixed: set[str] = set()
    for e in valid:
        found = coordinates(e.point)
        for key, value in e.point.values.items():
            if isinstance(value, list):
                lengths.setdefault(key, set()).add(len(value))
            if not any(c[0] == key for c in found):
                fixed.add(key)
        for (key, _), number in found.items():
            v = float(number)
            low[key] = min(low.get(key, v), v)
            high[key] = max(high.get(key, v), v)
    parts: list[str] = []
    uncovered: list[str] = []
    for key in sorted(low):
        shape = ""
        if key in lengths:
            sizes = sorted(lengths[key])
            shape = (
                f"lists of {sizes[0]} to {sizes[-1]} entries, "
                if len(sizes) > 1
                else (f"lists of {sizes[0]} entries, ")
            )
        parts.append(f"{key}: {shape}values in [{_show(low[key])}, {_show(high[key])}]")
        uncovered.append(f"{key} outside [{_show(low[key])}, {_show(high[key])}]")
        if key in lengths:
            uncovered.append(f"{key} with lengths other than {sorted(lengths[key])}")
    for key in sorted(fixed - set(low)):
        parts.append(f"{key}: not numeric, held at sampled values")
        uncovered.append(f"{key} at values other than those sampled")
    if outcome.budget.local_evaluations == 0:
        uncovered.append("no local optimisation: only sampled and boundary points")
    if outcome.boundary_points == 0:
        uncovered.append("no points found at the edge of the domain")
    if outcome.invalid_starts:
        uncovered.append(
            f"{outcome.invalid_starts} of {outcome.starts} starts were outside the domain "
            "or failed an assumption, and were not optimised"
        )
    failed = sum(1 for e in outcome.evaluations if e.error is not None)
    if failed:
        uncovered.append(f"{failed} points where the encoding raised or the sandbox failed")
    if outcome.sample_failures:
        uncovered.append(f"{outcome.sample_failures} starts where the sampler failed")
    uncovered.append(
        "everything between the evaluated points: no counterexample found is not a proof"
    )
    worst = min(valid, key=lambda e: cast(mpmath.mpf, e.margin), default=None)
    worst_margin: str | None = None
    if worst is not None:
        with working_digits(worst.point.precision_digits):
            worst_margin = mpmath.nstr(cast(mpmath.mpf, worst.margin), 15)
    return CoverageReport(
        space="; ".join(parts) or "no valid point evaluated",
        dimensions=outcome.dimensions,
        starts=outcome.starts,
        worst_margin=_decimal(worst_margin),
        worst_point=worst.original if worst is not None else None,
        uncovered=uncovered,
        evaluations=len(valid),
        failed_evaluations=failed,
        boundary_points=outcome.boundary_points,
        local_evaluations=outcome.local_evaluations,
    )


def _decimal(text: str | None) -> str | None:
    """mpmath writes "1.0e+5"; keep it, but make sure it reads as an exact decimal."""
    if text is None:
        return None
    to_fraction(text)
    return text


# --- certificates ---


class CertificateUnavailable(Exception):
    """The encoding could not give the values a certificate needs at this point."""


def build_certificate(make: EncodingFactory, x: Instance, digits: int) -> Certificate:
    """Build a certificate at `x` from a fresh encoding's own certificate method.

    The kind follows the encoding's capability: exact margin, then ball, then
    primal and dual bounds. The certifier recomputes everything; this only
    says where to look. Raise CertificateUnavailable if the encoding cannot.
    """
    point = Instance(values=x.values, precision_digits=digits)
    try:
        enc = make()
        if isinstance(enc, ExactMargin):
            with working_digits(digits):
                margin = cast(object, enc.exact_margin(point))
            if type(margin) is not Fraction:
                raise CertificateUnavailable("the exact margin is not a Fraction")
            return ExactCertificate(point=point, margin=margin)
        if isinstance(enc, BallMargin):
            with working_digits(digits):
                ball = cast(object, enc.margin_ball(point))
                if type(ball) is not flint.arb or not ball.is_finite():
                    raise CertificateUnavailable("the margin ball is not a finite arb")
                ends = (
                    ball.lower().str(digits + 5, radius=False),
                    ball.upper().str(digits + 5, radius=False),
                )
            return IntervalCertificate(point=point, digits=digits, enclosure=ends)
        if isinstance(enc, DualBounds):
            high = max(digits, MIN_DUALITY_DIGITS)
            point = point.at_precision(high)
            with working_digits(high):
                bounds = cast(object, enc.primal_dual(point))
                if not (
                    type(bounds) is tuple
                    and len(cast(tuple[object, ...], bounds)) == 2
                    and all(type(b) is flint.arb for b in cast(tuple[object, ...], bounds))
                ):
                    raise CertificateUnavailable("the primal and dual bounds are not two arbs")
                primal, dual = cast(tuple[flint.arb, flint.arb], bounds)
                middle = (primal + dual) / 2
                return DualityCertificate(
                    point=point,
                    primal=primal.mid().str(high + 5, radius=False),
                    dual=dual.mid().str(high + 5, radius=False),
                    digits=high,
                    violation=middle.mid().str(high + 5, radius=False),
                )
    except CertificateUnavailable:
        raise
    except (Exception, SystemExit) as exc:  # noqa: BLE001 - encoding code may raise anything
        raise CertificateUnavailable(f"the encoding raised {describe(exc)}") from exc
    raise CertificateUnavailable("the encoding has no exact margin, ball, or primal-dual bounds")


class ReformulationAttempt(BaseModel):
    """One proposed reformulation: whether its maps passed, and what its search covered."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    family: Family
    accepted: bool
    reason: str | None
    source: str
    coverage: CoverageReport | None = None
    candidates: int = 0


class AttackResult(BaseModel):
    """A certified kill, or none, and the coverage of the search in the original variables."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    certificate: Certificate | None
    check: CheckResult | None
    """The certifier's answer to `certificate`."""
    coverage: CoverageReport
    candidates: int
    """Points with a negative margin that the search found."""
    rejected: list[str]
    """Why each candidate sent to the certifier was refused."""
    reformulations: list[ReformulationAttempt] = Field(default_factory=lambda: [])

    @model_validator(mode="after")
    def _kill_needs_acceptance(self) -> "AttackResult":
        if self.certificate is not None and (self.check is None or not self.check.accepted):
            raise ValueError("a certificate is kept only when the certifier accepted it")
        return self

    @property
    def killed(self) -> bool:
        return self.certificate is not None


def certify_candidates(
    make: EncodingFactory,
    candidates: Sequence[Evaluation],
    certify: Certifier,
    budget: SearchBudget,
) -> tuple[Certificate | None, CheckResult | None, list[str]]:
    """Certify the lowest candidates, in their original variables, until one is accepted."""
    rejected: list[str] = []
    for e in candidates[: budget.max_certify]:
        assert e.original is not None
        try:
            cert = build_certificate(make, e.original, budget.digits)
        except CertificateUnavailable as exc:
            rejected.append(f"no certificate at {dict(e.original.values)}: {exc}")
            continue
        result = certify(cert)
        if result.accepted:
            return cert, result, rejected
        rejected.append(result.reason)
    return None, None, rejected


def attack(
    make: EncodingFactory,
    budget: SearchBudget,
    rng: random.Random,
    *,
    assumptions: int,
    certify: Certifier | None = None,
) -> AttackResult:
    """Search the encoding `make` builds for a violation, and certify the best candidates.

    `certify` decides each certificate; by default `check_certificate` on
    `make` alone, which trusts that one encoding. A kill in the pipeline needs
    `check_kill` with both encodings. `assumptions` is the record's number of
    assumptions.
    """
    outcome = search(evaluator_for(make), budget, rng)
    checker = certify or partial(check_certificate, make, assumptions=assumptions)
    candidates = outcome.candidates()
    cert, check, rejected = certify_candidates(make, candidates, checker, budget)
    return AttackResult(
        certificate=cert,
        check=check,
        coverage=coverage_of(outcome),
        candidates=len(candidates),
        rejected=rejected,
    )
