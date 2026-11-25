"""Certificates and the certificate checker.

Search may suggest a kill; only a certificate makes one (design doc-gmj,
"Falsifier"). `check_certificate` accepts a certificate only if all of these
hold (build plan doc-5wh, Task 6):

1. the point satisfies every assumption predicate and the domain predicate;
2. the margin is negative: exactly, by a rigorous enclosure, or between primal
   and dual bounds whose gap is below 1/1000 of the upper bound;
3. 1 and 2 hold again at twice the precision, or at 100 digits or more if the
   margin is within 1e-6 of zero or the encoding flags the point as
   near-degenerate; and the two checks agree on the margin.

The sign is rigorous for every kind, if the encoding is honest: an exact
margin, an arb ball, or two arb balls for the primal and dual bounds. The two
checks agree if their bounds on the margin overlap.

The checker never trusts a number in the certificate: it recomputes everything
through the encoding. It takes a factory and builds a fresh encoding for each
run, so state kept on the encoding object (a cache, a solver's warm start)
cannot carry from the first run to the repeat. It cannot see state kept
elsewhere, such as on the class or in a module. It rejects a certificate if an
encoding call or the factory leaves the working precision changed. It cannot
see an encoding that sets its own precision and restores it before it returns.
Models never write this code.

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
from minilab.certify.result import CheckResult, MarginCheck, accept, reject, show, working_digits
from minilab.encoding.api import Encoding, EncodingFactory, Instance

# A certificate is one of the three kinds, tagged by its `kind` field. Read one
# from JSON with `CERTIFICATE.validate_json`.
Certificate = Annotated[
    ExactCertificate | IntervalCertificate | DualityCertificate,
    Field(discriminator="kind"),
]
CERTIFICATE: TypeAdapter[Certificate] = TypeAdapter(Certificate)

# A margin not certainly below -NEAR_ZERO is checked again at higher precision.
NEAR_ZERO = flint.arb("1e-6")

# The precision of the repeat check near zero or near degeneracy, at the least.
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
    # A deep copy: an encoding could write into the point's values, and the
    # repeat, the other encoding, and the caller must not see that.
    point = cert.point.model_copy(deep=True).at_precision(digits)
    with working_digits(digits):
        expected = (mpmath.mp.prec, flint.ctx.prec)

        def moved(step: str) -> MarginCheck | None:
            # An encoding that leaves its own precision set would make the
            # repeat check run at the same precision as the first.
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
        except (Exception, SystemExit) as exc:  # noqa: BLE001 - encoding code may raise anything
            return MarginCheck.failed(
                reject(f"the encoding raised {_describe(exc)} at {digits} digits")
            ), False
    return margin, degenerate


def check_certificate(make: EncodingFactory, cert: Certificate) -> CheckResult:
    """Accept `cert` as a violation of the conjecture `make` encodes, or say why not.

    `make` builds a fresh encoding; it is called once for each run. This trusts
    the encoding. A kill needs `check_kill`, which asks both encodings.
    """
    try:
        return _check_certificate(make, cert)
    except (Exception, SystemExit) as exc:  # noqa: BLE001 - answer, except to KeyboardInterrupt
        return reject(f"the check raised {_describe(exc)}")


def _describe(exc: BaseException) -> str:
    """Name an exception from encoding code, whose own text may raise."""
    try:
        text = str(exc)
    except Exception:  # noqa: BLE001 - the text is encoding code too
        text = "(its text raised)"
    return f"{type(exc).__name__}: {text[:500]}"


def _build(make: EncodingFactory, digits: int) -> Encoding | CheckResult:
    """Call the factory at `digits` digits; reject it if it changes the precision."""
    with working_digits(digits):
        expected = (mpmath.mp.prec, flint.ctx.prec)
        enc = make()
        if (mpmath.mp.prec, flint.ctx.prec) != expected:
            return reject("the factory changed the working precision")
    return enc


def _check_certificate(make: EncodingFactory, cert: Certificate) -> CheckResult:
    digits = _digits(cert)
    enc = _build(make, digits)
    if isinstance(enc, CheckResult):
        return enc
    first, degenerate = _check_at(enc, cert, digits)
    if not first.result.accepted:
        return first.result
    assert first.enclosure is not None
    near_zero = not first.enclosure.upper() < -NEAR_ZERO
    recheck_digits = 2 * digits
    why = "repeated"
    if near_zero or degenerate:
        recheck_digits = max(RECHECK_DIGITS, recheck_digits)
        why = "near zero, so repeated" if near_zero else "near-degenerate, so repeated"
    fresh = _build(make, recheck_digits)
    if isinstance(fresh, CheckResult):
        return fresh
    if fresh is enc:
        return reject("the factory returned the same encoding object twice")
    second, _ = _check_at(fresh, cert, recheck_digits)
    if not second.result.accepted:
        return reject(
            f"{why} at {recheck_digits} digits; the repeat check failed: {second.result.reason}"
        )
    if not _agree(first, second):
        return reject(
            f"the margin bounds at {digits} and {recheck_digits} digits do not agree: "
            f"{_bounds(first)} and {_bounds(second)}"
        )
    return accept(
        f"{first.result.reason}; {why} at {recheck_digits} digits: {second.result.reason}"
    )


def _bounds(check: MarginCheck) -> str:
    if check.enclosure is None:
        return "none"
    return f"[{show(check.enclosure.lower())}, {show(check.enclosure.upper())}]"


def _agree(first: MarginCheck, second: MarginCheck) -> bool:
    """Say whether the first check and the repeat agree on the margin.

    They agree if their bounds on the margin overlap: honest bounds may narrow
    as the precision grows, but both contain the margin.
    """
    assert first.enclosure is not None and second.enclosure is not None
    return first.enclosure.overlaps(second.enclosure)


def check_kill(
    make_first: EncodingFactory, make_second: EncodingFactory, cert: Certificate
) -> CheckResult:
    """Accept `cert` as a kill only if both independent encodings accept it.

    The two factories build the two families' encodings of one conjecture
    (design doc-gmj, "Encoding"). Model code computes the ball and duality
    bounds, so a single encoding could certify a violation that is not there.
    This refuses one factory passed twice, but cannot tell which family wrote
    an encoding: the caller (Task 7's encode_twice) must give one per family.
    """
    if make_first is make_second:
        return reject("a kill needs two independent encodings, not the same encoding twice")
    results = [check_certificate(make, cert) for make in (make_first, make_second)]
    for name, result in zip(("first", "second"), results, strict=True):
        if not result.accepted:
            return reject(f"the {name} encoding refuses: {result.reason}")
    return accept(f"both encodings accept. First: {results[0].reason}. Second: {results[1].reason}")
