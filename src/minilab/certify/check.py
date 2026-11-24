"""Certificates and the certificate checker.

Search may suggest a kill; only a certificate makes one (design doc-gmj,
"Falsifier"). `check_certificate` accepts a certificate only if all of these
hold (build plan doc-5wh, Task 6):

1. the point satisfies every assumption predicate and the domain predicate;
2. the margin is negative: exactly, by a rigorous enclosure, or between primal
   and dual bounds whose gap is below 1/1000 of the violation;
3. 1 and 2 hold again at twice the precision, or at 100 digits or more if the
   margin is within 1e-6 of zero or the encoding flags the point as
   near-degenerate; and the two checks agree on the margin.

For interval and exact certificates the sign is rigorous, if the encoding's
ball or exact margin is honest. Duality bounds are mpf values, so the repeat is
a heuristic: it catches cancellation that changes the result between the two
precisions, not cancellation in terms larger than both.

The checker never trusts a number in the certificate: it recomputes everything
through the encoding. It calls the encoding object it is given for both runs,
so state that object keeps (a cache, a solver's warm start) carries from the
first run to the repeat. It rejects a certificate if an encoding call changes
the working precision. Models never write this code.

The encoding's capabilities (`margin_ball`, `primal_dual`) are written by a
model, so one encoding's say-so is not enough for a kill: `check_kill` accepts
only when both independent encodings of the conjecture accept the certificate.
"""

from typing import Annotated, cast

import flint
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

# Outside interval certificates, the upper bounds from the first and the
# repeat check must agree to within this share of the larger.
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
        expected = (mpmath.mp.prec, flint.ctx.prec)

        def moved(step: str) -> MarginCheck | None:
            # An encoding that sets its own precision would make the repeat
            # check run at the same precision as the first.
            if (mpmath.mp.prec, flint.ctx.prec) == expected:
                return None
            return MarginCheck.failed(
                reject(f"the encoding changed the working precision in {step}")
            )

        try:
            # Encoding code is untrusted: check the shape of what it returns.
            holds = cast(object, enc.assumptions(point))
            if bad := moved("the assumptions"):
                return bad, False
            if not isinstance(holds, list) or not all(
                type(h) is bool for h in cast(list[object], holds)
            ):
                return MarginCheck.failed(
                    reject("the assumption predicates did not give booleans")
                ), False
            failed = [i for i, h in enumerate(cast(list[bool], holds)) if not h]
            if failed:
                return MarginCheck.failed(
                    reject(f"the point fails assumption(s) {failed} at {digits} digits")
                ), False
            inside = cast(object, enc.in_domain(point))
            if bad := moved("the domain predicate"):
                return bad, False
            if type(inside) is not bool:
                return MarginCheck.failed(
                    reject(f"the domain predicate did not give a bool: {type(inside).__name__}")
                ), False
            if not inside:
                return MarginCheck.failed(
                    reject(f"the point is outside the domain at {digits} digits")
                ), False
            margin = _check_margin(enc, cert, point)
            if bad := moved("the margin"):
                return bad, False
            if not margin.result.accepted:
                return margin, False
            flag = cast(object, enc.near_degenerate(point))
            if bad := moved("the near-degeneracy flag"):
                return bad, False
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
    if not first.result.accepted:
        return first.result
    assert first.upper is not None
    near_zero = abs(first.upper) < NEAR_ZERO
    recheck_digits = 2 * digits
    why = "repeated"
    if near_zero or degenerate:
        recheck_digits = max(RECHECK_DIGITS, recheck_digits)
        why = "near zero, so repeated" if near_zero else "near-degenerate, so repeated"
    second, _ = _check_at(enc, cert, recheck_digits)
    if not second.result.accepted:
        return reject(
            f"{why} at {recheck_digits} digits; the repeat check failed: {second.result.reason}"
        )
    if not _agree(cert, first, second, recheck_digits):
        return reject(
            f"the margin bounds at {digits} and {recheck_digits} digits do not agree: "
            f"{_bounds(first)} and {_bounds(second)}"
        )
    return accept(
        f"{first.result.reason}; {why} at {recheck_digits} digits: {second.result.reason}"
    )


def _bounds(check: MarginCheck) -> str:
    if check.lower is None or check.upper is None:
        return "none"
    return f"[{mpmath.nstr(check.lower, 15)}, {mpmath.nstr(check.upper, 15)}]"


def _agree(cert: Certificate, first: MarginCheck, second: MarginCheck, digits: int) -> bool:
    """Say whether the first check and the repeat agree on the margin.

    Two balls agree if they overlap: an honest ball may narrow as the precision
    grows. Other bounds agree if the upper bounds are within 1/1000 of each
    other.
    """
    assert first.lower is not None and first.upper is not None
    assert second.lower is not None and second.upper is not None
    if isinstance(cert, IntervalCertificate):
        return first.lower <= second.upper and second.lower <= first.upper
    with mpmath.mp.workdps(digits):
        diff = abs(first.upper - second.upper)
        return diff <= max(abs(first.upper), abs(second.upper)) / AGREEMENT_SHARE


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
