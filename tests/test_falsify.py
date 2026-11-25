"""Tests for the falsifier (build plan doc-5wh, Task 8; bead minilab-f132d7).

Search may suggest a kill; only a certificate makes one. A survival needs a
coverage report. A reformulation is used only after its maps agree with the
original margin, and a kill found through it is certified in the original
variables. A mechanism passes only if its predicted failure holds, certified,
on at least 9 of 10 fresh instances.

Every model here is a fake backend except in the live-marked test. Model-written
code (reformulations, mechanisms, and the stage's encodings) really runs in the
sandbox.
"""

import inspect
import json
import random
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from fractions import Fraction
from functools import partial
from pathlib import Path

import mpmath
import pytest
from fixtures.pilot import s7 as s7_module
from fixtures.pilot.s7 import S7Encoding

from minilab.certify import ExactCertificate, check_certificate
from minilab.encoding.api import Instance, to_fraction
from minilab.falsify.coverage import CoverageReport
from minilab.falsify.mechanism import check_mechanism
from minilab.falsify.reformulate import check_reformulation, reformulate_and_attack
from minilab.falsify.search import LocalEvaluator, SearchBudget, attack
from minilab.falsify.stage import run_falsifier_stage
from minilab.ledger import Ledger
from minilab.models.adapter import Budget, ModelReply, ModelUnavailable, default_backends
from minilab.models.backends import Backend
from minilab.records import Family, Provenance, Record, RecordStore, Stage, StageOutput, Status
from minilab.run import MemoryPolicy, RunManifest, save_manifest

RUN = "run-1"
PAIR = ("information-theory", "optimisation")
BUDGET = Budget(tokens_per_pair=1_000_000)


# --- toy encodings, in process ---


def scalar(x: Instance, key: str) -> Fraction:
    value = x.values[key]
    assert not isinstance(value, list)
    return to_fraction(value)


class Planted:
    """Over (a, b) in [0, 1]^2: a violation only in a disc of radius 1/100 around (7/10, 1/5).

    The sampler draws thousandths, so a sample almost never lands in the disc;
    the local optimiser has to find it.
    """

    def exact_margin(self, x: Instance) -> Fraction:
        a, b = scalar(x, "a"), scalar(x, "b")
        return (a - Fraction(7, 10)) ** 2 + (b - Fraction(1, 5)) ** 2 - Fraction(1, 10_000)

    def margin(self, x: Instance) -> mpmath.mpf:
        value = self.exact_margin(x)
        return mpmath.mpf(value.numerator) / value.denominator

    def assumptions(self, x: Instance) -> list[bool]:
        return [True]

    def in_domain(self, x: Instance) -> bool:
        return all(0 <= scalar(x, k) <= 1 for k in ("a", "b"))

    def near_degenerate(self, x: Instance) -> bool:
        return False

    def sample(self, rng: random.Random) -> Instance:
        return Instance(
            values={k: str(Fraction(rng.randint(0, 1000), 1000)) for k in ("a", "b")},
            precision_digits=30,
        )


class FloatLiar(Planted):
    """Its float margin is negative everywhere; its exact margin never is."""

    def margin(self, x: Instance) -> mpmath.mpf:
        return mpmath.mpf(-1)

    def exact_margin(self, x: Instance) -> Fraction:
        return Fraction(1)


class Integers:
    """Over t in [0, 10]: margin (t - 11/2)^2 - 1/100, violated only near t = 11/2.

    The sampler draws integers, so sampling alone never sees the violation.
    """

    def exact_margin(self, x: Instance) -> Fraction:
        return (scalar(x, "t") - Fraction(11, 2)) ** 2 - Fraction(1, 100)

    def margin(self, x: Instance) -> mpmath.mpf:
        value = self.exact_margin(x)
        return mpmath.mpf(value.numerator) / value.denominator

    def assumptions(self, x: Instance) -> list[bool]:
        return [True]

    def in_domain(self, x: Instance) -> bool:
        return 0 <= scalar(x, "t") <= 10

    def near_degenerate(self, x: Instance) -> bool:
        return False

    def sample(self, rng: random.Random) -> Instance:
        return Instance(values={"t": str(rng.randint(0, 10))}, precision_digits=30)


class Line:
    """Over real t: margin t, so every t < 0 is a violation."""

    def exact_margin(self, x: Instance) -> Fraction:
        return scalar(x, "t")

    def margin(self, x: Instance) -> mpmath.mpf:
        value = self.exact_margin(x)
        return mpmath.mpf(value.numerator) / value.denominator

    def assumptions(self, x: Instance) -> list[bool]:
        return [True]

    def in_domain(self, x: Instance) -> bool:
        return True

    def near_degenerate(self, x: Instance) -> bool:
        return False

    def sample(self, rng: random.Random) -> Instance:
        return Instance(values={"t": str(rng.randint(1, 100))}, precision_digits=30)


def no_search(starts: int = 8) -> SearchBudget:
    return SearchBudget(starts=starts, local_evaluations=0, boundary_rays=0)


# --- model-written code, as source text for the sandbox ---


def reformulation(there: str, back: str, sample: str) -> str:
    """A reformulation over one parameter `u`; `there` and `back` are Fraction expressions."""
    return f"""
import random
from fractions import Fraction

from minilab.encoding.api import Instance, to_fraction


class Reform:
    def sample(self, rng: random.Random) -> Instance:
        return Instance(values={{"u": str({sample})}}, precision_digits=30)

    def there(self, x: Instance) -> Instance:
        t = to_fraction(x.values["t"])
        return Instance(values={{"u": str({there})}}, precision_digits=x.precision_digits)

    def back(self, y: Instance) -> Instance:
        u = to_fraction(y.values["u"])
        return Instance(values={{"t": str({back})}}, precision_digits=y.precision_digits)


ENCODING = Reform
"""


HALVES = reformulation("2 * t", "u / 2", "rng.randint(0, 20)")
SHIFTED = reformulation("t", "u + 1", "rng.randint(0, 10)")


def mechanism(threshold: str) -> str:
    """A mechanism over `t` whose generator gives a violation when rng.random() < threshold."""
    return f"""
import random
from fractions import Fraction

from minilab.encoding.api import Instance


class Mechanism:
    def sample(self, rng: random.Random) -> Instance:
        r = Fraction(rng.random())
        t = -1 - r if r < Fraction({threshold}) else 1 + r
        return Instance(values={{"t": str(t)}}, precision_digits=30)

    def predicate(self, x: Instance) -> bool:
        return True


ENCODING = Mechanism
"""


def toy_source(margin: str, sample: str = "rng.randint(0, 9)") -> str:
    """An encoding over integer t in 0..9 with an exact margin in the Fraction `t`."""
    return f"""
import random
from fractions import Fraction

import mpmath

from minilab.encoding.api import Instance, to_fraction


class Toy:
    def exact_margin(self, x: Instance) -> Fraction:
        t = to_fraction(x.values["t"])
        return {margin}

    def margin(self, x: Instance) -> mpmath.mpf:
        value = self.exact_margin(x)
        return mpmath.mpf(value.numerator) / value.denominator

    def assumptions(self, x: Instance) -> list[bool]:
        return [True]

    def in_domain(self, x: Instance) -> bool:
        t = to_fraction(x.values["t"])
        return 0 <= t <= 9

    def near_degenerate(self, x: Instance) -> bool:
        return False

    def sample(self, rng: random.Random) -> Instance:
        return Instance(values={{"t": str({sample})}}, precision_digits=30)


ENCODING = Toy
"""


TRUE_TOY = toy_source("2 - t / 10")
FALSE_TOY = toy_source("t - 4")


# --- fakes and fixtures ---


Reply = Callable[[str], str]


class FakeBackend:
    """A backend whose reply text comes from `reply(prompt)`. It records each prompt."""

    def __init__(self, family: Family, reply: Reply) -> None:
        self.family = family
        self.reply = reply
        self.prompts: list[str] = []

    def run(
        self, prompt: str, *, cwd: Path, env: Mapping[str, str], timeout_s: float
    ) -> ModelReply:
        self.prompts.append(prompt)
        return ModelReply(
            text=self.reply(prompt), tokens_in=10, tokens_out=10, estimated=False, model="fake-1"
        )


class DownBackend(FakeBackend):
    """A backend whose tool has hit its usage limit."""

    def run(
        self, prompt: str, *, cwd: Path, env: Mapping[str, str], timeout_s: float
    ) -> ModelReply:
        self.prompts.append(prompt)
        raise ModelUnavailable("usage limit reached")


def fenced(source: str) -> str:
    return f"Here it is.\n\n```python\n{source}\n```\n"


def all_say(text: str) -> dict[Family, FakeBackend]:
    return {f: FakeBackend(f, lambda prompt: text) for f in Family}


def make_record(local: str = "a", assumptions: int = 1) -> Record:
    return Record(
        id=f"{RUN}.{local}",
        run_id=RUN,
        pair=PAIR,
        mode="plain",
        statement="For every t in the family, the margin is non-negative.",
        objects=["t"],
        assumptions=[f"assumption {i}" for i in range(assumptions)],
        domain_of_validity="the family's domain",
        instance_family="t: a number",
        provenance=Provenance(family=Family.gemini, model="m", prompt_hash="h"),
    )


@pytest.fixture
def store(tmp_path: Path) -> RecordStore:
    return RecordStore(tmp_path)


@pytest.fixture
def ledger(store: RecordStore) -> Ledger:
    save_manifest(
        store.root,
        RunManifest(
            run_id=RUN,
            mode="plain",
            pairs=(PAIR,),
            memory_policy=MemoryPolicy.blind,
            memory_before=None,
            literature_before=None,
            budget_tokens_per_pair=1_000_000,
            families=tuple(Family),
            started=datetime(2026, 11, 25, 12, 0, tzinfo=UTC),
        ),
    )
    return Ledger(store, RUN)


def encoded(store: RecordStore, source: str, local: str = "a") -> str:
    """Store a proposed record whose two agreed encodings are both `source`."""
    record = make_record(local)
    store.create(record)
    directory = store.root / "runs" / RUN / "encodings"
    directory.mkdir(parents=True, exist_ok=True)
    evidence: list[str] = []
    for family in ("claude", "codex"):
        name = f"{record.id}.{family}.py"
        (directory / name).write_text(source, encoding="utf-8")
        evidence.append(f"encodings/{name}")
    store.write_stage(
        record.id,
        Stage.encoding,
        StageOutput(
            written_at=datetime.now(UTC),
            data={"families": ["claude", "codex"], "agreed": True},
            evidence=evidence,
        ),
    )
    return record.id


# --- search and certificates ---


def test_planted_violation_found_and_certified() -> None:
    result = attack(
        Planted,
        SearchBudget(starts=8, local_evaluations=300, boundary_rays=4),
        random.Random(1),
        assumptions=1,
    )
    assert result.killed
    cert = result.certificate
    assert isinstance(cert, ExactCertificate)
    a, b = scalar(cert.point, "a"), scalar(cert.point, "b")
    assert (a - Fraction(7, 10)) ** 2 + (b - Fraction(1, 5)) ** 2 < Fraction(1, 10_000)
    # The library checker, run again on its own, accepts the certificate.
    assert check_certificate(Planted, cert, assumptions=1).accepted
    assert result.coverage.starts == 8 and result.coverage.dimensions == 2


def test_sampling_alone_misses_the_planted_violation() -> None:
    # The same seeds without the local optimiser: the kill above is the optimiser's.
    result = attack(Planted, no_search(), random.Random(1), assumptions=1)
    assert not result.killed
    assert result.coverage.worst_margin is not None
    assert Fraction(result.coverage.worst_margin) > 0


def test_float_violation_without_certificate_is_not_a_kill() -> None:
    result = attack(FloatLiar, no_search(), random.Random(1), assumptions=1)
    assert not result.killed
    assert result.certificate is None
    assert result.candidates > 0
    assert result.rejected and "not negative" in result.rejected[0]


def test_survived_always_has_coverage(store: RecordStore, ledger: Ledger) -> None:
    record_id = encoded(store, TRUE_TOY)
    fakes = all_say(fenced(HALVES))
    result = run_falsifier_stage(
        record_id,
        ledger=ledger,
        budget=BUDGET,
        backends=fakes,
        search=SearchBudget(starts=6, local_evaluations=0, boundary_rays=2),
        max_reformulations=1,
        checks=40,
    )
    assert result.status is Status.survived
    record = store.load(record_id)
    assert record.status is Status.survived
    output = record.stages[Stage.falsifier]
    assert output.evidence == [f"falsifier/{record_id}.coverage.json"]
    report = CoverageReport.model_validate_json(
        (store.root / "runs" / RUN / output.evidence[0]).read_text()
    )
    assert report.starts == 6
    assert report.dimensions == 1
    assert report.evaluations > 0
    assert report.worst_margin is not None and Fraction(report.worst_margin) > 0
    assert report.worst_point is not None
    assert report.uncovered
    assert report.space


def test_stage_kill_is_certified_by_both_encodings(store: RecordStore, ledger: Ledger) -> None:
    record_id = encoded(store, FALSE_TOY)
    fakes = all_say(fenced("import os\nENCODING = object\n"))
    result = run_falsifier_stage(
        record_id,
        ledger=ledger,
        budget=BUDGET,
        backends=fakes,
        search=SearchBudget(starts=6, local_evaluations=0, boundary_rays=0),
    )
    assert result.status is Status.killed
    record = store.load(record_id)
    assert record.status is Status.killed
    output = record.stages[Stage.falsifier]
    assert f"falsifier/{record_id}.certificate.json" in output.evidence
    check = output.data["check"]
    assert isinstance(check, str) and check.startswith("both encodings accept")
    # The mechanism was asked of a checker family and failed its sandbox check.
    mechanism_data = output.data["mechanism"]
    assert isinstance(mechanism_data, dict) and mechanism_data["passed"] is False
    assert fakes[Family.gemini].prompts == []


def test_model_unavailable_leaves_record_unchanged(store: RecordStore, ledger: Ledger) -> None:
    record_id = encoded(store, TRUE_TOY)
    path = store.root / "runs" / RUN / "records" / f"{record_id}.json"
    before = path.read_bytes()
    down: dict[Family, Backend] = {f: DownBackend(f, lambda prompt: "") for f in Family}
    with pytest.raises(ModelUnavailable):
        run_falsifier_stage(
            record_id,
            ledger=ledger,
            budget=BUDGET,
            backends=down,
            search=SearchBudget(starts=4, local_evaluations=0, boundary_rays=0),
        )
    assert path.read_bytes() == before
    assert store.load(record_id).status is Status.proposed


def test_stage_refuses_a_record_without_agreed_encodings(
    store: RecordStore, ledger: Ledger
) -> None:
    store.create(make_record())
    with pytest.raises(ValueError, match="encoding"):
        run_falsifier_stage(f"{RUN}.a", ledger=ledger, budget=BUDGET, backends=all_say(""))


# --- reformulation ---


def test_bad_reformulation_rejected() -> None:
    check = check_reformulation(SHIFTED, LocalEvaluator(Integers()), random.Random(0), n=40)
    assert not check.accepted
    assert check.reason is not None and "margin" in check.reason


def test_good_reformulation_accepted() -> None:
    check = check_reformulation(HALVES, LocalEvaluator(Integers()), random.Random(0), n=40)
    assert check.accepted, check.reason


def test_bad_reformulation_is_not_attacked(store: RecordStore, ledger: Ledger) -> None:
    store.create(make_record())
    certify = partial(check_certificate, Integers, assumptions=1)
    result = reformulate_and_attack(
        store.load(f"{RUN}.a"),
        Integers,
        certify=certify,
        ledger=ledger,
        budget=BUDGET,
        backends=all_say(fenced(SHIFTED)),
        search=no_search(),
        rng=random.Random(0),
        max_reformulations=2,
        checks=40,
    )
    assert not result.killed
    assert [r.accepted for r in result.reformulations] == [False, False]
    assert all(r.coverage is None for r in result.reformulations)


def test_reformulation_kill_is_certified_in_original_variables(
    store: RecordStore, ledger: Ledger
) -> None:
    store.create(make_record())
    certify = partial(check_certificate, Integers, assumptions=1)
    direct = attack(Integers, no_search(), random.Random(0), assumptions=1, certify=certify)
    assert not direct.killed
    fakes = all_say(fenced(HALVES))
    result = reformulate_and_attack(
        store.load(f"{RUN}.a"),
        Integers,
        certify=certify,
        ledger=ledger,
        budget=BUDGET,
        backends=fakes,
        search=no_search(starts=64),
        rng=random.Random(0),
        checks=40,
    )
    assert result.killed
    cert = result.certificate
    assert isinstance(cert, ExactCertificate)
    assert set(cert.point.values) == {"t"}
    assert scalar(cert.point, "t") == Fraction(11, 2)
    assert result.reformulations[0].accepted
    # The reformulation was asked of a checker family, never Gemini.
    assert fakes[Family.gemini].prompts == []
    assert len(fakes[Family.claude].prompts) == 1


@pytest.mark.live
def test_s7_joint_law_reformulation_kills(store: RecordStore, ledger: Ledger) -> None:
    """A live model, given S7 and the pilot's hint, finds a kill through a reformulation.

    The pilot found S7's counterexample only by searching over joint laws in
    place of (source, distortion, slope). Run with `-m live`.
    """
    record = Record(
        id=f"{RUN}.s7",
        run_id=RUN,
        pair=PAIR,
        mode="plain",
        statement=(
            "For any finite source p and any distortion d: X x Y -> [0, 1], "
            "R(D) - 2 D^2 is convex on [D_min, D_max]."
        ),
        objects=["p", "d", "R(D)"],
        assumptions=["p is a probability vector", "every d(x, y) lies in [0, 1]"],
        domain_of_validity="D_min <= D1 < D2 < D3 <= D_max",
        instance_family=(
            "p: n_x decimals; d: n_x * n_y decimals; s: 3 slopes; q: 3 output laws of n_y "
            "decimals; n_x, n_y in 2..3"
        ),
        provenance=Provenance(family=Family.gemini, model="m", prompt_hash="h"),
    )
    store.create(record)
    certify = partial(check_certificate, S7Encoding, assumptions=2)
    result = reformulate_and_attack(
        record,
        S7Encoding,
        certify=certify,
        ledger=ledger,
        budget=BUDGET,
        backends=default_backends(),
        search=SearchBudget(starts=32, local_evaluations=200, boundary_rays=8, digits=40),
        rng=random.Random(0),
        original_source=inspect.getsource(s7_module),
        hint="Search over joint laws of (X, Y) in place of (source, distortion, slope).",
    )
    assert result.killed, [r.reason for r in result.reformulations]


# --- mechanisms ---


class FixedSeeds(random.Random):
    """An rng whose getrandbits gives the listed seeds in turn."""

    def __init__(self, seeds: list[int]) -> None:
        super().__init__(0)
        self.seeds = list(seeds)

    def getrandbits(self, k: int, /) -> int:
        return self.seeds.pop(0)


def seeds_with(failing: int, total: int = 10) -> list[int]:
    """Seeds where exactly `failing` give random.Random(seed).random() < 1/2."""
    low = [s for s in range(1000) if random.Random(s).random() < 0.5][:failing]
    high = [s for s in range(1000) if random.Random(s).random() >= 0.5][: total - failing]
    return low + high


KILL = ExactCertificate(
    point=Instance(values={"t": "-1"}, precision_digits=30), margin=Fraction(-1)
)


@pytest.mark.parametrize(("failing", "passed"), [(10, True), (9, True), (8, False)])
def test_mechanism_needs_nine_of_ten(
    store: RecordStore, ledger: Ledger, failing: int, passed: bool
) -> None:
    store.create(make_record())
    fakes = all_say(fenced(mechanism("1/2")))
    result = check_mechanism(
        store.load(f"{RUN}.a"),
        KILL,
        make=Line,
        certify=partial(check_certificate, Line, assumptions=1),
        ledger=ledger,
        budget=BUDGET,
        backends=fakes,
        rng=FixedSeeds(seeds_with(failing)),
    )
    assert result.tried == 10
    assert result.held == failing
    assert result.passed is passed
    assert result.family is Family.claude
    assert fakes[Family.gemini].prompts == []


def test_mechanism_with_a_false_predicate_does_not_hold(store: RecordStore, ledger: Ledger) -> None:
    store.create(make_record())
    source = mechanism("1").replace("return True", "return False")
    result = check_mechanism(
        store.load(f"{RUN}.a"),
        KILL,
        make=Line,
        certify=partial(check_certificate, Line, assumptions=1),
        ledger=ledger,
        budget=BUDGET,
        backends=all_say(fenced(source)),
        rng=random.Random(0),
    )
    assert result.tried == 10
    assert result.held == 0 and not result.passed
    assert all("predicate" in reason for reason in result.reasons)


def test_coverage_report_round_trips() -> None:
    result = attack(Integers, no_search(), random.Random(0), assumptions=1)
    report = result.coverage
    assert report.starts == 8 and report.worst_point is not None
    assert CoverageReport.model_validate(json.loads(report.model_dump_json())) == report
