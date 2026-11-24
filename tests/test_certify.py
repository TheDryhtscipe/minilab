"""Tests for the certificate library (build plan doc-5wh, Task 6).

The pilot fixtures in tests/fixtures/pilot/ are rebuilt by hand from the
crossconj-ml pilot (see rebuild.py there). Each rejection test also shows that
the case is a real artefact, so that a checker that rejected everything could
not pass the suite: the acceptance tests cover the other side.
"""

import random
from fractions import Fraction
from pathlib import Path

import flint
import mpmath
import pytest
from fixtures.pilot.s1_side import TsallisPinskerSide
from fixtures.pilot.s7 import S7Encoding
from fixtures.pilot.s7_y2 import Y2Encoding, rho_float

from minilab.certify import (
    CERTIFICATE,
    Certificate,
    DualityCertificate,
    ExactCertificate,
    IntervalCertificate,
    check_certificate,
)
from minilab.encoding.api import Instance, to_fraction

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
    result = check_certificate(S7Encoding(), cert)
    assert not result.accepted
    assert "upper bound on the margin" in result.reason


def test_rejects_grid_gap() -> None:
    cert = load("grid_gap")
    assert isinstance(cert, IntervalCertificate)
    c, a1, b = (float(scalar(cert.point, k)) for k in ("c", "a1", "b"))
    assert rho_float(c, a1, b) > 0.25  # the float grid shows a violation
    with mpmath.mp.workdps(100):
        assert Y2Encoding().margin(cert.point) > 0  # at 100 digits there is none
    result = check_certificate(Y2Encoding(), cert)
    assert not result.accepted
    assert "below zero" in result.reason


def test_rejects_out_of_domain() -> None:
    cert = load("out_of_domain")
    result = check_certificate(S7Encoding(), cert)
    assert not result.accepted
    assert "outside the domain" in result.reason

    class AnyDomain(S7Encoding):
        def in_domain(self, x: Instance) -> bool:
            return True

    # Only the domain check stops it: the bounds alone certify a violation.
    assert check_certificate(AnyDomain(), cert).accepted


def test_accepts_s7_counterexample() -> None:
    cert = load("s7_counterexample")
    assert isinstance(cert, DualityCertificate) and cert.digits == 40
    result = check_certificate(S7Encoding(), cert)
    assert result.accepted, result.reason


def test_accepts_s7prime_counterexample() -> None:
    cert = load("s7prime_counterexample")
    assert isinstance(cert, DualityCertificate) and cert.digits == 60
    result = check_certificate(S7Encoding(), cert)
    assert result.accepted, result.reason
    # The pilot's certified bound on the divided difference was 0.18565 < 4.
    assert abs(to_fraction(cert.violation) + 4 - Fraction("0.18565335128737")) < Fraction(1, 10**12)


class Cancelling:
    """A margin of 10^-40 - offset that float-like precision gets wrong.

    At 30 digits, 1 + 10^-40 rounds to 1, so the margin reads as -offset.
    Records every precision it ran at.
    """

    def __init__(self, offset: str) -> None:
        self.offset = offset
        self.seen: list[int] = []

    def margin(self, x: Instance) -> mpmath.mpf:
        self.seen.append(mpmath.mp.dps)
        tiny = mpmath.mpf(10) ** -40
        return (1 + tiny) - 1 - mpmath.mpf(self.offset)

    def primal_dual(self, x: Instance) -> tuple[mpmath.mpf, mpmath.mpf]:
        m = self.margin(x)
        return m, m

    def assumptions(self, x: Instance) -> list[bool]:
        return [True]

    def in_domain(self, x: Instance) -> bool:
        return True

    def near_degenerate(self, x: Instance) -> bool:
        return False

    def sample(self, rng: random.Random) -> Instance:
        return Instance(values={}, precision_digits=30)


def _cancel_cert(offset: str, claimed: str) -> DualityCertificate:
    return DualityCertificate(
        point=Instance(values={}, precision_digits=30),
        primal=claimed,
        dual=claimed,
        digits=30,
        violation=claimed,
    )


def test_near_zero_margin_rechecked_at_100_digits() -> None:
    # False violation: -1e-41 at 30 digits, +9e-41 at 100.
    enc = Cancelling("1e-41")
    result = check_certificate(enc, _cancel_cert("1e-41", "-1e-41"))
    assert not result.accepted
    assert "near zero; the repeat check failed" in result.reason
    assert 30 in enc.seen and max(enc.seen) >= 100

    # A true violation near zero: -2e-40 at 30 digits, -1e-40 at 100. Accepted.
    enc = Cancelling("2e-40")
    result = check_certificate(enc, _cancel_cert("2e-40", "-2e-40"))
    assert result.accepted, result.reason
    assert "repeated at 100 digits" in result.reason
    assert max(enc.seen) >= 100


def test_accepts_s1_side_interval_certificate() -> None:
    cert = load("s1_side_counterexample")
    assert isinstance(cert, IntervalCertificate)
    result = check_certificate(TsallisPinskerSide(), cert)
    assert result.accepted, result.reason


def test_interval_claim_must_meet_the_enclosure() -> None:
    cert = load("s1_side_counterexample")
    assert isinstance(cert, IntervalCertificate)
    moved = cert.model_copy(update={"enclosure": ("-0.5", "-0.4")})
    result = check_certificate(TsallisPinskerSide(), moved)
    assert not result.accepted
    assert "does not meet the claimed" in result.reason


def test_duality_claim_must_match_recomputation() -> None:
    cert = load("s7_counterexample")
    assert isinstance(cert, DualityCertificate)
    tampered = cert.model_copy(update={"primal": "-5", "violation": "-5"})
    result = check_certificate(S7Encoding(), tampered)
    assert not result.accepted
    assert "do not match" in result.reason


def test_duality_needs_high_precision() -> None:
    cert = load("s7_counterexample")
    assert isinstance(cert, DualityCertificate)
    result = check_certificate(S7Encoding(), cert.model_copy(update={"digits": 15}))
    assert not result.accepted
    assert "at least 30 digits" in result.reason


def test_failed_assumption_rejected() -> None:
    cert = load("s7_counterexample")
    assert isinstance(cert, DualityCertificate)
    values = dict(cert.point.values)
    values["p"] = ["0.40", "0.61"]  # no longer sums to 1
    bad = cert.model_copy(update={"point": Instance(values=values, precision_digits=40)})
    result = check_certificate(S7Encoding(), bad)
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
        Quadratic(), ExactCertificate(point=half, margin=Fraction(-1, 20))
    ).accepted
    wrong = check_certificate(Quadratic(), ExactCertificate(point=half, margin=Fraction(-1, 10)))
    assert not wrong.accepted and "exact margin is -1/20" in wrong.reason
    tenth = Instance(values={"x": "0.1"}, precision_digits=30)
    positive = check_certificate(
        Quadratic(), ExactCertificate(point=tenth, margin=Fraction(11, 100))
    )
    assert not positive.accepted and "not negative" in positive.reason
    irrational = Instance(values={"x": "nan"}, precision_digits=30)
    assert not check_certificate(
        Quadratic(), ExactCertificate(point=irrational, margin=Fraction(-1, 20))
    ).accepted


def test_missing_capability_rejected() -> None:
    s7 = load("s7_counterexample")
    cert = IntervalCertificate(point=s7.point, digits=40, enclosure=("-5", "-3"))
    result = check_certificate(S7Encoding(), cert)  # S7 has no ball margin
    assert not result.accepted
    assert "no ball margin" in result.reason


def test_encoding_error_rejected() -> None:
    class Broken(S7Encoding):
        def primal_dual(self, x: Instance) -> tuple[mpmath.mpf, mpmath.mpf]:
            raise ZeroDivisionError("boom")

    result = check_certificate(Broken(), load("s7_counterexample"))
    assert not result.accepted
    assert "raised ZeroDivisionError" in result.reason


def test_certificate_json_round_trip() -> None:
    for name in ("warm_start_artefact", "grid_gap", "s1_side_counterexample"):
        cert = load(name)
        assert CERTIFICATE.validate_json(CERTIFICATE.dump_json(cert)) == cert


def test_check_restores_precision() -> None:
    before_mp, before_flint = mpmath.mp.dps, flint.ctx.prec
    check_certificate(S7Encoding(), load("s7prime_counterexample"))
    assert (mpmath.mp.dps, flint.ctx.prec) == (before_mp, before_flint)


@pytest.mark.parametrize("enc", [S7Encoding(), Y2Encoding(), TsallisPinskerSide()])
def test_fixture_encodings_sample_valid_points(
    enc: S7Encoding | Y2Encoding | TsallisPinskerSide,
) -> None:
    rng = random.Random(0)
    for _ in range(5):
        assert all(enc.assumptions(enc.sample(rng)))
