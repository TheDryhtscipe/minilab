"""Certificates and the certificate checker.

Search may suggest a kill; only a certificate makes one (design doc-gmj,
"Falsifier"). `check_certificate` accepts a certificate only if all of these
hold (build plan doc-5wh, Task 6):

1. the point satisfies every assumption predicate and the domain predicate;
2. the margin is negative: exactly, by a rigorous enclosure, or between primal
   and dual bounds whose gap is below 1/1000 of the violation;
3. if the margin is within 1e-6 of zero, or the encoding flags the point as
   near-degenerate, 1 and 2 hold again at 100 digits or more.

The checker never trusts a number in the certificate: it recomputes everything
through the encoding, in a fresh call, so no solver state is carried over (the
cold-start rule). Models never write this code.
"""

from typing import Annotated, cast

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

# The precision of the repeat check, at the least.
RECHECK_DIGITS = 100


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
            degenerate = enc.near_degenerate(point) is True if margin.result.accepted else False
        except Exception as exc:  # noqa: BLE001 - encoding code may raise anything
            return MarginCheck.failed(
                reject(f"the encoding raised {type(exc).__name__} at {digits} digits: {exc}")
            ), False
    return margin, degenerate


def check_certificate(enc: Encoding, cert: Certificate) -> CheckResult:
    """Accept `cert` as a kill of the conjecture `enc` encodes, or say why not."""
    digits = _digits(cert)
    first, degenerate = _check_at(enc, cert, digits)
    if not first.result.accepted:
        return first.result
    near_zero = abs(first.upper) < NEAR_ZERO
    if not (near_zero or degenerate):
        return first.result
    recheck_digits = max(RECHECK_DIGITS, 2 * digits)
    why = "near zero" if near_zero else "near-degenerate"
    second, _ = _check_at(enc, cert, recheck_digits)
    if not second.result.accepted:
        return reject(f"{why}; the repeat check failed: {second.result.reason}")
    return accept(
        f"{first.result.reason}; {why}, so repeated at {recheck_digits} digits: "
        f"{second.result.reason}"
    )
