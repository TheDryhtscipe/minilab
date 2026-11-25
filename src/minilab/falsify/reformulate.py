"""Reformulate, then attack (design doc-gmj, "Falsifier"; build plan doc-5wh, Task 8).

The pilot's random sampling missed S7's counterexample; searching over joint
laws in place of (source, distortion, slope) found it. So when the direct
search finds no kill, a model proposes a reformulation: a new parametrisation
of the same claim, as a module run in the sandbox. Its class (bound to
`ENCODING`, as the sandbox requires) has three methods:

- `sample(rng) -> Instance`: a point of the new space;
- `there(x) -> Instance`: the new point for an original point;
- `back(y) -> Instance`: the original point for a new point.

Before use, `check_reformulation` checks on `n` samples (half from each
sampler) that the maps agree with the original margin: a round trip
there-and-back keeps the point's validity and its margin, within `rtol`, and
the new sampler reaches the original domain often enough. A reformulation that
fails is refused and never searched.

An accepted one is searched with the original encoding's own margin and
predicates at `back(y)`, so the search measures the original claim. A
candidate is mapped back and certified in the original variables by the same
certifier as a direct kill.

The proposer is a checker family, never Gemini (Gemini only generates). Model
failures (`ModelUnavailable`, `BudgetExceeded`) are raised as `call` raises
them, and nothing is written.
"""

import random
from collections.abc import Mapping, Sequence
from typing import cast

import mpmath
from pydantic import BaseModel, ConfigDict, Field

from minilab.certify.result import working_digits
from minilab.encoding.api import EncodingFactory, Instance
from minilab.encoding.prompts import extract_source
from minilab.encoding.sandbox import ALLOWED_MODULES, DEFAULT_TIMEOUT_S, Method
from minilab.falsify.search import (
    AttackResult,
    Certifier,
    Evaluation,
    Evaluator,
    ReformulationAttempt,
    SearchBudget,
    attack,
    certify_candidates,
    coverage_of,
    evaluator_for,
    sandbox_points,
    sandbox_samples,
)
from minilab.falsify.search import search as run_search
from minilab.ledger import Ledger
from minilab.models.adapter import Budget, Role, call, pick_checker
from minilab.models.backends import Backend
from minilab.records import Family, Record

# Samples on which a reformulation's maps must agree with the original margin.
CHECK_SAMPLES = 200


class MapCheck(BaseModel):
    """Whether a reformulation's maps agree with the original margin, and if not, why."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    accepted: bool
    reason: str | None
    compared: int = Field(default=0, ge=0)
    """Round trips whose margins were compared."""


class SandboxReformulation:
    """A model-written reformulation, run in the sandbox."""

    def __init__(self, source: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self.source = source
        self.timeout_s = timeout_s

    def sample(self, seeds: Sequence[int]) -> tuple[list[Instance | None], str | None]:
        return sandbox_samples(self.source, seeds, self.timeout_s)

    def map(
        self, method: Method, points: Sequence[Instance | None]
    ) -> tuple[list[Instance | None], str | None]:
        """Map each point with `there` or `back`; None where it failed, and the first error."""
        values, errors = sandbox_points(self.source, method, points, self.timeout_s)
        out: list[Instance | None] = []
        first: str | None = None
        for value, error in zip(values, errors, strict=True):
            if error is None and type(value) is not Instance:
                error = f"{method} did not give an Instance"
            if error is not None and first is None:
                first = f"{method}: {error}"
            out.append(value if error is None and type(value) is Instance else None)
        return out, first


class ReformulatedEvaluator:
    """Searches the new space, judged by the original encoding at `back(y)`."""

    def __init__(self, reformulation: SandboxReformulation, original: Evaluator) -> None:
        self.reformulation = reformulation
        self.original = original

    def sample(self, seeds: Sequence[int]) -> list[Instance | None]:
        return self.reformulation.sample(seeds)[0]

    def evaluate(self, points: Sequence[Instance]) -> list[Evaluation]:
        backs, _ = self.reformulation.map("back", points)
        judged = iter(self.original.evaluate([x for x in backs if x is not None]))
        out: list[Evaluation] = []
        for y, x in zip(points, backs, strict=True):
            if x is None:
                out.append(
                    Evaluation(
                        point=y, original=None, valid=False, margin=None, error="back failed"
                    )
                )
                continue
            e = next(judged)
            out.append(
                Evaluation(point=y, original=x, valid=e.valid, margin=e.margin, error=e.error)
            )
        return out


def _close(a: mpmath.mpf, b: mpmath.mpf, rtol: float) -> bool:
    """Same sign, and within `rtol` of each other relative to the larger."""
    if (a > 0) != (b > 0) or (a < 0) != (b < 0):
        return False
    return bool(abs(a - b) <= rtol * max(abs(a), abs(b)))


def check_reformulation(
    source: str,
    original: Evaluator,
    rng: random.Random,
    n: int = CHECK_SAMPLES,
    rtol: float = 1e-9,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> MapCheck:
    """Check that a reformulation's maps agree with the original margin on `n` samples.

    Half the samples come from the original sampler (x against back(there(x)))
    and half from the reformulation's (back(y) against back(there(back(y)))).
    Each round trip must keep the point's validity, and, where valid, its
    margin within `rtol`. At least a quarter of the reformulation's samples
    must map back into the domain. Never raises for anything the reformulation
    does; raises SandboxUnavailable if the sandbox cannot run at all.
    """
    if n < 2:
        raise ValueError(f"the check needs at least 2 samples, got {n}")
    ref = SandboxReformulation(source, timeout_s)
    seeds = [rng.getrandbits(63) for _ in range(n)]
    half = n // 2
    xs = original.sample(seeds[:half])
    if any(x is None for x in xs):
        return MapCheck(accepted=False, reason="the original sampler failed")
    ys, error = ref.sample(seeds[half:])
    if error is not None:
        return MapCheck(accepted=False, reason=f"its sampler failed: {error}")
    steps: list[tuple[list[Instance | None], str | None]] = []
    ys_from_x, error = ref.map("there", xs)
    steps.append((ys_from_x, error))
    xs_round, error = ref.map("back", ys_from_x)
    steps.append((xs_round, error))
    xs_from_y, error = ref.map("back", ys)
    steps.append((xs_from_y, error))
    ys_round, error = ref.map("there", xs_from_y)
    steps.append((ys_round, error))
    xs_from_y_round, error = ref.map("back", ys_round)
    steps.append((xs_from_y_round, error))
    for _, failed in steps:
        if failed is not None:
            return MapCheck(accepted=False, reason=f"a map failed: {failed}")
    pairs = list(
        zip(
            cast(list[Instance], xs) + cast(list[Instance], xs_from_y),
            cast(list[Instance], xs_round) + cast(list[Instance], xs_from_y_round),
            strict=True,
        )
    )
    names = {frozenset(x.values) for x, _ in pairs} | {frozenset(x.values) for _, x in pairs}
    if len(names) > 1:
        return MapCheck(
            accepted=False, reason=f"back gives other parameter names: {sorted(map(sorted, names))}"
        )
    judged = original.evaluate([p for pair in pairs for p in pair])
    margin_gap: str | None = None
    domain_gap: str | None = None
    compared = 0
    for i, (x, x2) in enumerate(pairs):
        e, e2 = judged[2 * i], judged[2 * i + 1]
        where = f"at {dict(x.values)} and its round trip {dict(x2.values)}"
        if e.error is not None or e2.error is not None:
            return MapCheck(
                accepted=False,
                reason=f"the original encoding failed {where}: {e.error or e2.error}",
            )
        if e.valid != e2.valid:
            domain_gap = domain_gap or f"the round trip changes the domain or an assumption {where}"
            continue
        if e.valid and e.margin is not None and e2.margin is not None:
            compared += 1
            with working_digits(max(x.precision_digits, x2.precision_digits)):
                if not _close(e.margin, e2.margin, rtol):
                    margin_gap = margin_gap or (
                        f"the round trip changes the margin {where}: "
                        f"{mpmath.nstr(e.margin, 12)} against {mpmath.nstr(e2.margin, 12)}"
                    )
    if margin_gap or domain_gap:
        return MapCheck(accepted=False, reason=margin_gap or domain_gap, compared=compared)
    reached = sum(1 for e in judged[2 * half :: 2] if e.valid)
    if reached < max(1, (n - half) // 4):
        return MapCheck(
            accepted=False,
            reason=f"only {reached} of {n - half} of its samples map back into the domain",
            compared=compared,
        )
    return MapCheck(accepted=True, reason=None, compared=compared)


def _record_text(record: Record) -> str:
    assumptions = "\n".join(f"  {i}. {a}" for i, a in enumerate(record.assumptions, start=1))
    return (
        f"Statement: {record.statement}\n"
        f"Objects: {', '.join(record.objects)}\n"
        f"Assumptions, in order:\n{assumptions or '  (none)'}\n"
        f"Domain of validity: {record.domain_of_validity}\n"
        f"Instance family and its parameters: {record.instance_family}\n"
    )


def reformulation_prompt(
    record: Record,
    example: Instance | None,
    original_source: str | None,
    tried: Sequence[ReformulationAttempt],
    hint: str | None,
) -> str:
    """The prompt that asks for a reformulation. It shows the record, never another family's."""
    allowed = ", ".join(sorted(ALLOWED_MODULES))
    parts = [
        "A search for a counterexample to this conjecture found none. Propose a different "
        "parametrisation of the same claim, one where a counterexample is easier to find.\n",
        _record_text(record),
    ]
    if example is not None:
        parts.append(
            f"An instance in the original variables, as `Instance.values`: {dict(example.values)}\n"
        )
    if original_source is not None:
        parts.append(f"The original encoding:\n```python\n{original_source}\n```\n")
    if hint:
        parts.append(f"A hint: {hint}\n")
    for i, attempt in enumerate(tried, start=1):
        outcome = (
            f"refused: {attempt.reason}" if not attempt.accepted else "searched; no counterexample"
        )
        parts.append(f"Reformulation {i}, already tried, was {outcome}.\n")
    parts.append(
        f"""Write a Python module that binds the name ENCODING, at the top level, to a class
with these methods. `Instance` comes from `minilab.encoding.api`.

- sample(self, rng: random.Random) -> Instance: a random point of the new
  parametrisation, drawn only with `rng`.
- there(self, x: Instance) -> Instance: the new point for an original point x.
- back(self, y: Instance) -> Instance: the original point for a new point y. It must
  use exactly the original parameter names, and give values as exact decimal strings
  or Fractions, with precision_digits taken from y.

back(there(x)) must be the same instance of the claim as x. Most samples should map
back into the domain of validity. Compute in exact arithmetic where you can.

Import only these modules: {allowed}. `minilab.encoding.api` gives `Instance`,
`to_fraction`, `to_mpf`, and `to_arb` to read a value. Do not change the working
precision. Do not read files, use the network, or print.

Reply with the module only, in one ```python block."""
    )
    return "\n".join(parts)


def reformulate_and_attack(
    record: Record,
    make: EncodingFactory,
    *,
    certify: Certifier,
    ledger: Ledger,
    budget: Budget,
    backends: Mapping[Family, Backend] | None = None,
    search: SearchBudget,
    rng: random.Random,
    max_reformulations: int = 3,
    checks: int = CHECK_SAMPLES,
    direct: AttackResult | None = None,
    original_source: str | None = None,
    hint: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> AttackResult:
    """Attack `make` directly (unless `direct` is given), then through reformulations.

    Up to `max_reformulations` are asked for, one at a time, each told why the
    earlier ones failed. Each is checked by `check_reformulation` on `checks`
    samples; one that passes is searched, and its candidates are certified in
    the original variables by `certify`. Stops at the first kill. The result's
    coverage is the direct search's; each reformulation's is in its attempt.
    Raise NoChecker if no family may propose one, and model failures as `call`
    raises them.
    """
    first = direct or attack(
        make, search, rng, assumptions=len(record.assumptions), certify=certify
    )
    if first.killed:
        return first
    family = pick_checker(record.provenance.family, ledger.manifest.families)
    original = evaluator_for(make)
    examples = [x for x in original.sample([rng.getrandbits(63)]) if x is not None]
    attempts: list[ReformulationAttempt] = []
    rejected = list(first.rejected)
    candidates = first.candidates
    for _ in range(max_reformulations):
        prompt = reformulation_prompt(
            record, examples[0] if examples else None, original_source, attempts, hint
        )
        reply = call(
            family,
            Role.reformulate,
            prompt,
            record_id=record.id,
            pair=record.pair,
            ledger=ledger,
            budget=budget,
            backends=backends,
        )
        source = extract_source(reply.text)
        check = check_reformulation(source, original, rng, n=checks, timeout_s=timeout_s)
        if not check.accepted:
            attempts.append(
                ReformulationAttempt(
                    family=family, accepted=False, reason=check.reason, source=source
                )
            )
            continue
        evaluator = ReformulatedEvaluator(SandboxReformulation(source, timeout_s), original)
        outcome = run_search(evaluator, search, rng)
        found = outcome.candidates()
        candidates += len(found)
        cert, result, refused = certify_candidates(make, found, certify, search)
        rejected.extend(refused)
        attempts.append(
            ReformulationAttempt(
                family=family,
                accepted=True,
                reason=None,
                source=source,
                coverage=coverage_of(outcome),
                candidates=len(found),
            )
        )
        if cert is not None:
            return AttackResult(
                certificate=cert,
                check=result,
                coverage=first.coverage,
                candidates=candidates,
                rejected=rejected,
                reformulations=attempts,
            )
    return AttackResult(
        certificate=None,
        check=None,
        coverage=first.coverage,
        candidates=candidates,
        rejected=rejected,
        reformulations=attempts,
    )
