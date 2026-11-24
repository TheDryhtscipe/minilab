"""Certificates and the certificate checker.

Search may suggest a kill; only a certificate makes one (design doc-gmj,
"Falsifier"). `check_certificate` accepts a certificate only if all of these
hold (build plan doc-5wh, Task 6):

1. the point satisfies every assumption predicate and the domain predicate;
2. the margin is negative: exactly, by a rigorous enclosure, or between primal
   and dual bounds whose gap is below 1/1000 of the violation;
3. 1 and 2 hold again at twice the precision, or at 100 digits or more if the
   margin is within 1e-6 of zero or the encoding flags the point as
   near-degenerate; and the upper bounds on the margin from the two checks
   agree to within 1/1000. The repeat catches cancellation, which can make a
   positive margin read as negative well away from zero.

The checker never trusts a number in the certificate: it recomputes everything
through the encoding, in a fresh call, so no solver state is carried over (the
cold-start rule). Models never write this code.

The encoding's capabilities (`margin_ball`, `primal_dual`) are written by a
model, so one encoding's say-so is not enough for a kill: `check_kill` accepts
only when both independent encodings of the conjecture accept the certificate.
"""

from typing import Annotated, cast

import mpmath
from pydantic import Field, TypeAdapter

from minilab.certify.duality import DualityCertificate, check_duality_margin
from minilab.certify.exact import ExactCertificate, check_exact_margin
from minilab.certify.interval import IntervalCertificate, check_interval_margin
from minilab.certify.result import CheckResult, MarginCheck, accept, reject, working_digits
from minilab.encoding.api import Encoding, Instance

# A certificate is one of the three kinds, tagged by its `kind` field. Read one
# from JSON with `CERTIFICATE.validate_json`.
Certificate = Annotated[
    ExactCertificate | IntervalCertificate | DualityCertificate,
    Field(discriminator="kind"),
]
CERTIFICATE: TypeAdapter[Certificate] = TypeAdapter(Certificate)

# A margin this close to zero is checked again at higher precision.
NEAR_ZERO = 1e-6

# The precision of the repeat check near zero or near degeneracy, at the least.
RECHECK_DIGITS = 100

# The upper bounds from the first and the repeat check must agree to within
# this share of the larger.
AGREEMENT_SHARE = 1000


def _digits(cert: Certificate) -> int:
    if isinstance(cert, ExactCertificate):
        return cert.point.precision_digits
    return cert.digits


def _check_margin(
    enc: Encoding,
    cert: Certificate,
    point: Instance,
) -> MarginCheck:
    if isinstance(cert, ExactCertificate):
        return check_exact_margin(enc, cert, point)
    if isinstance(cert, IntervalCertificate):
        return check_interval_margin(enc, cert, point)
    return check_duality_margin(enc, cert, point)


def _check_at(
    enc: Encoding,
    cert: Certificate,
    digits: int,
) -> tuple[MarginCheck, bool]:
    """Run conditions 1 and 2 at `digits` digits.

    Returns the margin check and whether the encoding flags the point as
    near-degenerate.
    """
    point = cert.point.at_precision(digits)
    with working_digits(digits):
        try:
            # Encoding code is untrusted: check the shape of what it returns.
            holds = cast(object, enc.assumptions(point))
            if not isinstance(holds, list) or not all(
                isinstance(h, bool) for h in cast(list[object], holds)
            ):
                return MarginCheck.failed(
                    reject("the assumption predicates did not give booleans")
                ), False
            failed = [i for i, h in enumerate(cast(list[bool], holds)) if not h]
            if failed:
                return MarginCheck.failed(
                    reject(f"the point fails assumption(s) {failed} at {digits} digits")
                ), False
            if enc.in_domain(point) is not True:
                return MarginCheck.failed(
                    reject(f"the point is outside the domain at {digits} digits")
                ), False
            margin = _check_margin(enc, cert, point)
            if not margin.result.accepted:
                return margin, False
            flag = cast(object, enc.near_degenerate(point))
            if type(flag) is not bool:
                return MarginCheck.failed(
                    reject(f"the near-degeneracy flag is not a bool: {type(flag).__name__}")
                ), False
            degenerate = flag
        except Exception as exc:  # noqa: BLE001 - encoding code may raise anything
            return MarginCheck.failed(
                reject(f"the encoding raised {type(exc).__name__} at {digits} digits: {exc}")
            ), False
    return margin, degenerate


def check_certificate(enc: Encoding, cert: Certificate) -> CheckResult:
    """Accept `cert` as a violation of the conjecture `enc` encodes, or say why not.

    This trusts `enc`. A kill needs `check_kill`, which asks both encodings.
    """
    try:
        return _check_certificate(enc, cert)
    except Exception as exc:  # noqa: BLE001 - a check must answer, never raise
        return reject(f"the check raised {type(exc).__name__}: {exc}")


def _check_certificate(enc: Encoding, cert: Certificate) -> CheckResult:
    digits = _digits(cert)
    first, degenerate = _check_at(enc, cert, digits)
    if not first.result.accepted or first.upper is None:
        return first.result
    near_zero = abs(first.upper) < NEAR_ZERO
    recheck_digits = 2 * digits
    why = "repeated"
    if near_zero or degenerate:
        recheck_digits = max(RECHECK_DIGITS, recheck_digits)
        why = "near zero, so repeated" if near_zero else "near-degenerate, so repeated"
    second, _ = _check_at(enc, cert, recheck_digits)
    if not second.result.accepted or second.upper is None:
        return reject(
            f"{why} at {recheck_digits} digits; the repeat check failed: {second.result.reason}"
        )
    with mpmath.mp.workdps(recheck_digits):
        diff = abs(first.upper - second.upper)
        allowed = max(abs(first.upper), abs(second.upper)) / AGREEMENT_SHARE
        if diff > allowed:
            return reject(
                f"the upper bounds at {digits} and {recheck_digits} digits do not agree: "
                f"{mpmath.nstr(first.upper, 15)} and {mpmath.nstr(second.upper, 15)}"
            )
    return accept(
        f"{first.result.reason}; {why} at {recheck_digits} digits: {second.result.reason}"
    )


def check_kill(first: Encoding, second: Encoding, cert: Certificate) -> CheckResult:
    """Accept `cert` as a kill only if both independent encodings accept it.

    The two encodings are the two families' encodings of one conjecture (design
    doc-gmj, "Encoding"). Model code computes the ball and duality bounds, so a
    single encoding could certify a violation that is not there.
    """
    if first is second:
        return reject("a kill needs two independent encodings, not the same encoding twice")
    results = [check_certificate(enc, cert) for enc in (first, second)]
    for name, result in zip(("first", "second"), results, strict=True):
        if not result.accepted:
            return reject(f"the {name} encoding refuses: {result.reason}")
    return accept(f"both encodings accept. First: {results[0].reason}. Second: {results[1].reason}")
