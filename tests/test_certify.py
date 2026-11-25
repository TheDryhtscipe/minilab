"""Tests for the certificate library (build plan doc-5wh, Task 6).

The pilot fixtures in tests/fixtures/pilot/ are rebuilt by hand from the
crossconj-ml pilot (see rebuild.py there). Each rejection test also shows that
the case is a real artefact, so that a checker that rejected everything could
not pass the suite: the acceptance tests cover the other side.
"""

import random
import signal
from collections.abc import Callable, Generator
from contextlib import contextmanager
from fractions import Fraction
from functools import partial
from pathlib import Path
from typing import cast

import flint
import mpmath
import numpy as np
import pytest
from fixtures.pilot.s1_side import TsallisPinskerSide
from fixtures.pilot.s7 import S7Encoding
from fixtures.pilot.s7_y2 import Y2Encoding, rho_float
from pydantic import ValidationError

from minilab.certify import (
    CERTIFICATE,
    Certificate,
    DualityCertificate,
    ExactCertificate,
    IntervalCertificate,
    check_certificate,
)
from minilab.certify.result import working_digits
from minilab.encoding.api import MAX_TEXT, Instance, to_fraction, to_mpf

PILOT = Path(__file__).parent / "fixtures" / "pilot"


def scalar(x: Instance, key: str) -> Fraction:
    value = x.values[key]
    assert not isinstance(value, list)
    return to_fraction(value)


def load(name: str) -> Certificate:
    return CERTIFICATE.validate_json((PILOT / f"{name}.json").read_text())


def test_rejects_warm_start_artefact() -> None:
    cert = load("warm_start_artefact")
    assert isinstance(cert, DualityCertificate)
    # The solver's own rates make R look concave: a falsifier trusting them
    # would report this as a violation.
    assert to_fraction(cert.violation) < -18
    result = check_certificate(S7Encoding, cert, assumptions=2)
    assert not result.accepted
    assert "upper bound on the margin" in result.reason


def test_rejects_grid_gap() -> None:
    cert = load("grid_gap")
    assert isinstance(cert, IntervalCertificate)
    c, a1, b = (float(scalar(cert.point, k)) for k in ("c", "a1", "b"))
    assert rho_float(c, a1, b) > 0.25  # the float grid shows a violation
    with mpmath.mp.workdps(100):
        assert Y2Encoding().margin(cert.point) > 0  # at 100 digits there is none
    result = check_certificate(Y2Encoding, cert, assumptions=2)
    assert not result.accepted
    assert "below zero" in result.reason


def test_rejects_out_of_domain() -> None:
    cert = load("out_of_domain")
    result = check_certificate(S7Encoding, cert, assumptions=2)
    assert not result.accepted
    assert "outside the domain" in result.reason

    class AnyDomain(S7Encoding):
        def in_domain(self, x: Instance) -> bool:
            return True

    # Only the domain check stops it: the bounds alone certify a violation.
    assert check_certificate(AnyDomain, cert, assumptions=2).accepted


def test_accepts_s7_counterexample() -> None:
    cert = load("s7_counterexample")
    assert isinstance(cert, DualityCertificate) and cert.digits == 40
    result = check_certificate(S7Encoding, cert, assumptions=2)
    assert result.accepted, result.reason


def test_accepts_s7prime_counterexample() -> None:
    cert = load("s7prime_counterexample")
    assert isinstance(cert, DualityCertificate) and cert.digits == 60
    result = check_certificate(S7Encoding, cert, assumptions=2)
    assert result.accepted, result.reason
    # The pilot's certified bound on 2 R[D1, D2, D3] was 0.18565 < 4.
    assert abs(to_fraction(cert.violation) + 4 - Fraction("0.18565335128737")) < Fraction(1, 10**12)


def test_accepts_s1_side_interval_certificate() -> None:
    cert = load("s1_side_counterexample")
    assert isinstance(cert, IntervalCertificate)
    result = check_certificate(TsallisPinskerSide, cert, assumptions=4)
    assert result.accepted, result.reason


def test_interval_claim_must_meet_the_enclosure() -> None:
    cert = load("s1_side_counterexample")
    assert isinstance(cert, IntervalCertificate)
    moved = cert.model_copy(update={"enclosure": ("-0.5", "-0.4")})
    result = check_certificate(TsallisPinskerSide, moved, assumptions=4)
    assert not result.accepted
    assert "does not meet the claimed" in result.reason


def test_duality_claim_must_match_recomputation() -> None:
    cert = load("s7_counterexample")
    assert isinstance(cert, DualityCertificate)
    tampered = cert.model_copy(update={"primal": "-5", "violation": "-5"})
    result = check_certificate(S7Encoding, tampered, assumptions=2)
    assert not result.accepted
    assert "do not match" in result.reason


def test_duality_needs_high_precision() -> None:
    cert = load("s7_counterexample")
    assert isinstance(cert, DualityCertificate)
    result = check_certificate(S7Encoding, cert.model_copy(update={"digits": 15}), assumptions=2)
    assert not result.accepted
    assert "at least 30 digits" in result.reason


def test_failed_assumption_rejected() -> None:
    cert = load("s7_counterexample")
    assert isinstance(cert, DualityCertificate)
    values = dict(cert.point.values)
    values["p"] = ["0.40", "0.61"]  # no longer sums to 1
    bad = cert.model_copy(update={"point": Instance(values=values, precision_digits=40)})
    result = check_certificate(S7Encoding, bad, assumptions=2)
    assert not result.accepted
    assert "fails assumption(s) [0]" in result.reason


class Quadratic:
    """Toy exact claim: x (1 - x) <= 1/5 on [0, 1]. False at x = 1/2."""

    def exact_margin(self, x: Instance) -> Fraction:
        v = scalar(x, "x")
        return Fraction(1, 5) - v * (1 - v)

    def margin(self, x: Instance) -> mpmath.mpf:
        m = self.exact_margin(x)
        return mpmath.mpf(m.numerator) / m.denominator

    def assumptions(self, x: Instance) -> list[bool]:
        return [0 <= to_fraction(x.values["x"]) <= 1]  # type: ignore[arg-type]

    def in_domain(self, x: Instance) -> bool:
        return True

    def near_degenerate(self, x: Instance) -> bool:
        return False

    def sample(self, rng: random.Random) -> Instance:
        return Instance(values={"x": str(Fraction(rng.randint(0, 100), 100))}, precision_digits=30)


def test_exact_certificate() -> None:
    half = Instance(values={"x": Fraction(1, 2)}, precision_digits=30)
    assert check_certificate(
        Quadratic, ExactCertificate(point=half, margin=Fraction(-1, 20)), assumptions=1
    ).accepted
    wrong = check_certificate(
        Quadratic, ExactCertificate(point=half, margin=Fraction(-1, 10)), assumptions=1
    )
    assert not wrong.accepted and "exact margin is -1/20" in wrong.reason
    tenth = Instance(values={"x": "0.1"}, precision_digits=30)
    positive = check_certificate(
        Quadratic, ExactCertificate(point=tenth, margin=Fraction(11, 100)), assumptions=1
    )
    assert not positive.accepted and "not negative" in positive.reason
    irrational = Instance(values={"x": "nan"}, precision_digits=30)
    assert not check_certificate(
        Quadratic, ExactCertificate(point=irrational, margin=Fraction(-1, 20)), assumptions=1
    ).accepted


def test_missing_capability_rejected() -> None:
    s7 = load("s7_counterexample")
    cert = IntervalCertificate(point=s7.point, digits=40, enclosure=("-5", "-3"))
    result = check_certificate(S7Encoding, cert, assumptions=2)  # S7 has no ball margin
    assert not result.accepted
    assert "no ball margin" in result.reason


def test_encoding_error_rejected() -> None:
    class Broken(S7Encoding):
        def primal_dual(self, x: Instance) -> tuple[flint.arb, flint.arb]:
            raise ZeroDivisionError("boom")

    result = check_certificate(Broken, load("s7_counterexample"), assumptions=2)
    assert not result.accepted
    assert "raised ZeroDivisionError" in result.reason


def test_certificate_json_round_trip() -> None:
    for name in ("warm_start_artefact", "grid_gap", "s1_side_counterexample"):
        cert = load(name)
        assert CERTIFICATE.validate_json(CERTIFICATE.dump_json(cert)) == cert


def test_check_restores_precision() -> None:
    before_mp, before_flint = mpmath.mp.dps, flint.ctx.prec
    check_certificate(S7Encoding, load("s7prime_counterexample"), assumptions=2)
    assert (mpmath.mp.dps, flint.ctx.prec) == (before_mp, before_flint)


@pytest.mark.parametrize("enc", [S7Encoding(), Y2Encoding(), TsallisPinskerSide()])
def test_fixture_encodings_sample_valid_points(
    enc: S7Encoding | Y2Encoding | TsallisPinskerSide,
) -> None:
    rng = random.Random(0)
    for _ in range(5):
        assert all(enc.assumptions(enc.sample(rng)))


# Configurable encodings for the checker's own guards. Each records the
# precisions it ran at in `seen`, which the instances of one factory share.

POINT = Instance(values={}, precision_digits=16)


class Toy:
    def __init__(
        self, *, holds: object = None, degenerate: object = False, seen: list[int] | None = None
    ) -> None:
        self.holds: object = [True] if holds is None else holds
        self.degenerate = degenerate
        self.seen: list[int] = [] if seen is None else seen

    def margin(self, x: Instance) -> mpmath.mpf:
        return mpmath.mpf(-1)

    def assumptions(self, x: Instance) -> list[bool]:
        self.seen.append(mpmath.mp.dps)
        return cast(list[bool], self.holds)

    def in_domain(self, x: Instance) -> bool:
        return True

    def near_degenerate(self, x: Instance) -> bool:
        return cast(bool, self.degenerate)

    def sample(self, rng: random.Random) -> Instance:
        return POINT


class DualToy(Toy):
    """Primal and dual bounds given by a function of the working precision."""

    def __init__(
        self, bounds: Callable[[], tuple[flint.arb, flint.arb]], *, seen: list[int] | None = None
    ) -> None:
        super().__init__(seen=seen)
        self.bounds = bounds

    def primal_dual(self, x: Instance) -> tuple[flint.arb, flint.arb]:
        return self.bounds()


class BallToy(Toy):
    """A ball margin given by a function of the working precision."""

    def __init__(
        self,
        ball: Callable[[], flint.arb],
        *,
        holds: object = None,
        degenerate: object = False,
        seen: list[int] | None = None,
    ) -> None:
        super().__init__(holds=holds, degenerate=degenerate, seen=seen)
        self.ball = ball

    def margin_ball(self, x: Instance) -> flint.arb:
        return self.ball()


def exact(value: str) -> tuple[flint.arb, flint.arb]:
    """Primal and dual bounds that are both the exact arb `value`."""
    ball = flint.arb(value)
    return ball, ball


def dual_cert(primal: str, dual: str, violation: str, digits: int = 30) -> DualityCertificate:
    point = Instance(values={}, precision_digits=digits)
    return DualityCertificate(
        point=point, primal=primal, dual=dual, digits=digits, violation=violation
    )


def ball_cert(low: str, high: str, digits: int = 16) -> IntervalCertificate:
    point = Instance(values={}, precision_digits=digits)
    return IntervalCertificate(point=point, digits=digits, enclosure=(low, high))


def test_exactly_zero_margin_rejected() -> None:
    # sqrt(30)^2 - 30 is 0. As mpf values it read as a tiny negative number at
    # any precision; as arb balls it contains 0.
    def zero() -> tuple[flint.arb, flint.arb]:
        m = flint.arb(30).sqrt() ** 2 - 30
        return m, m

    with mpmath.mp.workdps(30):
        claim = mpmath.nstr(mpmath.sqrt(30) ** 2 - 30, 30)
    assert to_fraction(claim) < 0
    result = check_certificate(
        partial(DualToy, zero), dual_cert(claim, claim, claim), assumptions=1
    )
    assert not result.accepted
    assert "not below zero" in result.reason


@pytest.mark.parametrize("power", [40, 100])
def test_cancellation_in_large_terms_rejected(power: int) -> None:
    # (10^n + 1) - 10^n - 1/2 reads as -1/2 in mpf at 30 and 60 digits when n
    # is 100; it is +1/2. The arb ball is wide enough to show it.
    def cancel() -> tuple[flint.arb, flint.arb]:
        big = flint.arb(10) ** power
        m = (big + 1) - big - flint.arb("0.5")
        return m, m

    result = check_certificate(
        partial(DualToy, cancel), dual_cert("-0.5", "-0.5", "-0.5"), assumptions=1
    )
    assert not result.accepted
    assert "not below zero" in result.reason


def test_repeat_check_must_pass() -> None:
    # A ball below zero at 16 digits and above it at 32.
    def flips() -> flint.arb:
        return flint.arb(-1) if mpmath.mp.dps <= 16 else flint.arb(1)

    result = check_certificate(partial(BallToy, flips), ball_cert("-1", "-1"), assumptions=1)
    assert not result.accepted
    assert "repeat check failed" in result.reason


def test_recheck_precision() -> None:
    # Away from zero and not degenerate: repeated at twice the digits.
    seen: list[int] = []
    make = partial(BallToy, lambda: flint.arb(-1), seen=seen)
    assert check_certificate(make, ball_cert("-1", "-1"), assumptions=1).accepted
    assert seen == [16, 32]
    # Near zero: repeated at 100 digits or more.
    seen = []
    make = partial(BallToy, lambda: flint.arb(-1) / 10**8, seen=seen)
    result = check_certificate(make, ball_cert("-1e-8", "-1e-8"), assumptions=1)
    assert result.accepted, result.reason
    assert max(seen) >= 100
    # Flagged near-degenerate: repeated at 100 digits or more.
    seen = []
    make = partial(BallToy, lambda: flint.arb(-1), degenerate=True, seen=seen)
    assert check_certificate(make, ball_cert("-1", "-1"), assumptions=1).accepted
    assert max(seen) >= 100


def drift() -> flint.arb:
    return flint.arb(-1) if mpmath.mp.dps <= 16 else flint.arb(-2)


def test_repeat_must_agree() -> None:
    # Both balls lie below zero and meet the claim, but they do not overlap:
    # the value is not stable.
    result = check_certificate(partial(BallToy, drift), ball_cert("-2", "-1"), assumptions=1)
    assert not result.accepted
    assert "do not agree" in result.reason


class Memo(BallToy):
    """Caches its ball by point, so a second run on the same object repeats the first."""

    def __init__(self, ball: Callable[[], flint.arb]) -> None:
        super().__init__(ball)
        self.cache: dict[str, flint.arb] = {}

    def margin_ball(self, x: Instance) -> flint.arb:
        key = x.model_dump_json(exclude={"precision_digits"})
        if key not in self.cache:
            self.cache[key] = self.ball()
        return self.cache[key]


def test_each_run_gets_a_fresh_encoding() -> None:
    cert = ball_cert("-2", "-1")
    result = check_certificate(partial(Memo, drift), cert, assumptions=1)
    assert not result.accepted
    assert "do not agree" in result.reason
    # A factory that hands back one object would let the cache hide the drift.
    shared = Memo(drift)
    result = check_certificate(lambda: shared, cert, assumptions=1)
    assert not result.accepted
    assert "same encoding object" in result.reason


def test_non_bool_degenerate_flag_rejected() -> None:
    make = partial(BallToy, lambda: flint.arb(-1), degenerate=np.True_)
    result = check_certificate(make, ball_cert("-1", "-1"), assumptions=1)
    assert not result.accepted
    assert "near-degeneracy flag" in result.reason


def test_non_bool_assumptions_rejected() -> None:
    make = partial(BallToy, lambda: flint.arb(-1), holds=[1])
    result = check_certificate(make, ball_cert("-1", "-1"), assumptions=1)
    assert not result.accepted
    assert "did not give booleans" in result.reason


def test_duality_gap_rule() -> None:
    def wide() -> tuple[flint.arb, flint.arb]:
        return flint.arb(-1), flint.arb("-1.01")

    result = check_certificate(
        partial(DualToy, wide), dual_cert("-1", "-1.01", "-1"), assumptions=1
    )
    assert not result.accepted
    assert "duality gap" in result.reason


def test_violation_must_lie_between_bounds() -> None:
    result = check_certificate(
        partial(DualToy, lambda: exact("-1")), dual_cert("-1", "-1", "-2"), assumptions=1
    )
    assert not result.accepted
    assert "outside the bounds" in result.reason


def test_claimed_violation_must_be_negative() -> None:
    # The bounds lie below zero; a positive claim within the tolerance of the
    # upper bound would otherwise pass.
    make = partial(DualToy, lambda: exact("-1e-28"))
    result = check_certificate(make, dual_cert("-1e-28", "-1e-28", "1e-27"), assumptions=1)
    assert not result.accepted
    assert "claimed violation" in result.reason and "not negative" in result.reason
    assert check_certificate(make, dual_cert("-1e-28", "-1e-28", "-1e-28"), assumptions=1).accepted


def test_duality_bounds_must_be_arb_balls() -> None:
    def floats() -> tuple[flint.arb, flint.arb]:
        return cast(flint.arb, mpmath.mpf(-1)), cast(flint.arb, mpmath.mpf(-1))

    result = check_certificate(partial(DualToy, floats), dual_cert("-1", "-1", "-1"), assumptions=1)
    assert not result.accepted
    assert "not two arb balls" in result.reason


def test_wide_ball_around_the_claim_is_not_a_mismatch() -> None:
    # The claim lies inside the ball, though further from its midpoint than the
    # tolerance: the claim is not certainly wrong.
    def wide() -> tuple[flint.arb, flint.arb]:
        ball = flint.arb("-1", "1e-20")
        return ball, ball

    result = check_certificate(
        partial(DualToy, wide), dual_cert("-1.000000000000000000001", "-1", "-1"), assumptions=1
    )
    assert result.accepted, result.reason


def test_non_finite_bounds_rejected() -> None:
    def nan() -> tuple[flint.arb, flint.arb]:
        return flint.arb("nan"), flint.arb("nan")

    result = check_certificate(partial(DualToy, nan), dual_cert("-1", "-1", "-1"), assumptions=1)
    assert not result.accepted
    assert "not finite" in result.reason

    make = partial(BallToy, lambda: flint.arb("nan"))
    result = check_certificate(make, ball_cert("-1", "-1"), assumptions=1)
    assert not result.accepted
    assert "not finite" in result.reason


class HugeExact(Toy):
    def exact_margin(self, x: Instance) -> Fraction:
        return Fraction(-(10**400))


def test_exact_margin_of_huge_magnitude() -> None:
    cert = ExactCertificate(point=POINT, margin=Fraction(-(10**400)))
    result = check_certificate(HugeExact, cert, assumptions=1)
    assert result.accepted, result.reason


def test_exact_certificate_needs_a_rational_point() -> None:
    # HugeExact's assumptions do not read the point, so only the rationality check stops it.
    irrational = Instance(values={"x": "nan"}, precision_digits=30)
    cert = ExactCertificate(point=irrational, margin=Fraction(-(10**400)))
    result = check_certificate(HugeExact, cert, assumptions=1)
    assert not result.accepted
    assert "needs a rational point" in result.reason


@contextmanager
def deadline(seconds: int) -> Generator[None]:
    """Fail, rather than hang, if the block runs longer than `seconds`."""

    def expire(signum: int, frame: object) -> None:
        raise TimeoutError(f"took longer than {seconds} s")

    previous = signal.signal(signal.SIGALRM, expire)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def test_huge_precision_and_exponents_refused() -> None:
    with pytest.raises(ValidationError):
        Instance(values={}, precision_digits=10**8)
    with pytest.raises(ValidationError):
        IntervalCertificate(point=POINT, digits=10**8, enclosure=("-1", "-1"))
    with pytest.raises(ValidationError):
        DualityCertificate(point=POINT, primal="-1", dual="-1", digits=10**8, violation="-1")
    with deadline(5):
        with pytest.raises(ValueError, match="exponent"):
            to_fraction("1e-99999999")
        assert not Instance(values={"x": "1e-99999999"}, precision_digits=30).is_rational()


@pytest.mark.parametrize(
    "text",
    [
        "-1e" + "\u0669" * 8,  # Arabic-Indic digits: Fraction reads 10^99999999
        "\uff11\uff12",  # fullwidth 12
        "0x10",
        "inf",
        "-Infinity",
        "nan",
        "1e5" + " " * 3 + "1",
    ],
)
def test_values_must_be_ascii_decimals(text: str) -> None:
    with deadline(5):
        with pytest.raises(ValueError):
            to_fraction(text)
        with pytest.raises(ValueError):
            to_mpf(text)


def test_long_values_refused() -> None:
    with pytest.raises(ValueError, match="too long"):
        to_fraction("-0." + "1" * (MAX_TEXT + 1))
    # Every value within the cap reads exactly: Python refuses to convert an
    # integer string of more than 4300 digits.
    assert to_fraction("-0." + "1" * (MAX_TEXT - 3)) < 0
    # Pydantic refuses an exponent in a Fraction field.
    with pytest.raises(ValidationError):
        CERTIFICATE.validate_json(
            '{"kind": "exact", "point": {"values": {}, "precision_digits": 30},'
            ' "margin": "1e-99999999"}'
        )


def test_check_never_raises() -> None:
    # Bypass validation, as a caller building a certificate by hand might.
    cert = IntervalCertificate.model_construct(
        kind="interval", point=POINT, digits=10**30, enclosure=("-1", "-1")
    )
    result = check_certificate(partial(BallToy, lambda: flint.arb(-1)), cert, assumptions=1)
    assert not result.accepted


def test_precision_restored_after_an_exception() -> None:
    before_mp, before_flint = mpmath.mp.dps, flint.ctx.prec
    with pytest.raises(RuntimeError), working_digits(200):
        raise RuntimeError("boom")
    assert (mpmath.mp.dps, flint.ctx.prec) == (before_mp, before_flint)


def test_encoding_must_not_change_the_precision() -> None:
    # An encoding that pins its own precision would make both runs the same.
    class Pinned(DualToy):
        def primal_dual(self, x: Instance) -> tuple[flint.arb, flint.arb]:
            mpmath.mp.dps = 30
            return self.bounds()

    result = check_certificate(
        partial(Pinned, lambda: exact("-0.5")), dual_cert("-0.5", "-0.5", "-0.5"), assumptions=1
    )
    assert not result.accepted
    assert "changed the working precision" in result.reason

    class PinnedBall(BallToy):
        def margin_ball(self, x: Instance) -> flint.arb:
            flint.ctx.dps = 16
            return self.ball()

    result = check_certificate(
        partial(PinnedBall, lambda: flint.arb(-1)), ball_cert("-1", "-1"), assumptions=1
    )
    assert not result.accepted
    assert "changed the working precision" in result.reason


def test_honest_ball_that_narrows_is_accepted() -> None:
    # A ball that tightens at higher precision is consistent: both contain the margin.
    def narrowing() -> flint.arb:
        rad = "0.4999" if mpmath.mp.dps <= 16 else "1e-10"
        return flint.arb("-0.5", rad)

    result = check_certificate(
        partial(BallToy, narrowing), ball_cert("-0.5", "-0.5"), assumptions=1
    )
    assert result.accepted, result.reason


def test_ball_margin_must_be_an_arb() -> None:
    make = partial(BallToy, lambda: cast(flint.arb, mpmath.mpf(-1)))
    result = check_certificate(make, ball_cert("-1", "-1"), assumptions=1)
    assert not result.accepted
    assert "not an arb" in result.reason


def test_empty_enclosure_rejected() -> None:
    result = check_certificate(
        partial(BallToy, lambda: flint.arb(-1)), ball_cert("-1", "-2"), assumptions=1
    )
    assert not result.accepted
    assert "enclosure is empty" in result.reason


class FloatExact(Toy):
    def exact_margin(self, x: Instance) -> Fraction:
        return cast(Fraction, -0.5)


def test_exact_margin_must_be_a_fraction() -> None:
    cert = ExactCertificate(point=POINT, margin=Fraction(-1, 2))
    result = check_certificate(FloatExact, cert, assumptions=1)
    assert not result.accepted
    assert "not a Fraction" in result.reason


class NumpyDomain(BallToy):
    def in_domain(self, x: Instance) -> bool:
        return cast(bool, np.True_)


def test_domain_predicate_must_give_a_bool() -> None:
    result = check_certificate(
        partial(NumpyDomain, lambda: flint.arb(-1)), ball_cert("-1", "-1"), assumptions=1
    )
    assert not result.accepted
    assert "domain predicate did not give a bool" in result.reason


def test_s7_refuses_distortions_it_cannot_order() -> None:
    # Three identical witnesses give three equal distortions: no weight has a sign.
    cert = load("s7_counterexample")
    values = dict(cert.point.values)
    s = values["s"]
    q = values["q"]
    assert isinstance(s, list) and isinstance(q, list)
    values["s"] = [s[0]] * 3
    values["q"] = q[:3] * 3
    same = Instance(values=values, precision_digits=40)
    with working_digits(40), pytest.raises(ValueError, match="too close"):
        S7Encoding().primal_dual(same)


class PointMemo(BallToy):
    """Caches its ball in the point it is given, which outlives a fresh encoding."""

    def margin_ball(self, x: Instance) -> flint.arb:
        values = cast(dict[str, object], x.values)
        if "_memo" not in values:
            values["_memo"] = self.ball()
        return cast(flint.arb, values["_memo"])


def test_each_run_gets_its_own_copy_of_the_point() -> None:
    cert = ball_cert("-2", "-1")
    result = check_certificate(partial(PointMemo, drift), cert, assumptions=1)
    assert not result.accepted
    assert "do not agree" in result.reason
    assert cert.point.values == {}


class Unprintable(Exception):
    def __str__(self) -> str:
        raise ValueError("cannot print")


@pytest.mark.parametrize("exc", [Unprintable(), SystemExit(3)])
def test_check_answers_whatever_the_encoding_raises(exc: BaseException) -> None:
    def boom() -> flint.arb:
        raise exc

    result = check_certificate(partial(BallToy, boom), ball_cert("-1", "-1"), assumptions=1)
    assert not result.accepted
    assert type(exc).__name__ in result.reason


class AlwaysBelow(flint.arb):
    """An arb whose comparisons say it lies below anything."""

    def __lt__(self, other: object) -> bool:
        return True


class AlwaysEqual(Fraction):
    def __eq__(self, other: object) -> bool:
        return True

    def __ne__(self, other: object) -> bool:
        return False

    def __ge__(self, other: object) -> bool:
        return False


class TupleLike(tuple[flint.arb, flint.arb]):
    pass


def test_subclasses_of_the_returned_types_rejected() -> None:
    result = check_certificate(
        partial(BallToy, lambda: AlwaysBelow(1)), ball_cert("1", "1"), assumptions=1
    )
    assert not result.accepted
    assert "not an arb" in result.reason

    class SubExact(Toy):
        def exact_margin(self, x: Instance) -> Fraction:
            return AlwaysEqual(1, 5)

    result = check_certificate(
        SubExact, ExactCertificate(point=POINT, margin=Fraction(-1)), assumptions=1
    )
    assert not result.accepted
    assert "not a Fraction" in result.reason

    def wrapped() -> tuple[flint.arb, flint.arb]:
        return TupleLike(exact("-1"))

    result = check_certificate(
        partial(DualToy, wrapped), dual_cert("-1", "-1", "-1"), assumptions=1
    )
    assert not result.accepted
    assert "not two arb balls" in result.reason


def test_factory_must_not_change_the_precision() -> None:
    before = (mpmath.mp.dps, flint.ctx.prec)

    def make() -> BallToy:
        mpmath.mp.dps = 7
        flint.ctx.dps = 7
        return BallToy(lambda: flint.arb(-1))

    result = check_certificate(make, ball_cert("-1", "-1"), assumptions=1)
    assert not result.accepted
    assert "changed the working precision" in result.reason
    assert (mpmath.mp.dps, flint.ctx.prec) == before


def test_duality_gap_is_measured_against_the_bounds() -> None:
    # Bounds -1e-30 and -1e-28: the gap is 99 times the upper bound. A claim
    # inflated within the tolerance must not make the gap look small.
    def wide() -> tuple[flint.arb, flint.arb]:
        return flint.arb("-1e-30"), flint.arb("-1e-28")

    result = check_certificate(
        partial(DualToy, wide), dual_cert("-1e-30", "-1e-28", "-1.0005e-25"), assumptions=1
    )
    assert not result.accepted
    assert "duality gap" in result.reason
