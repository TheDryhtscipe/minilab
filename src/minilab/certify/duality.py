"""Duality certificates: matching primal and dual bounds on an optimisation-defined margin."""

from typing import Literal

import mpmath
from pydantic import BaseModel, ConfigDict, Field

from minilab.certify.result import CheckResult, MarginCheck, reject
from minilab.encoding.api import MAX_DIGITS, DualBounds, Encoding, Instance, to_mpf

# A duality certificate must be computed at high precision (design doc-gmj,
# "Falsifier"). Float precision gave the pilot its spurious ratios.
MIN_DUALITY_DIGITS = 30

# The gap between the bounds must be smaller than this share of the violation.
GAP_SHARE = 1000

# An upper bound must lie below -scale * 10^-(digits - NOISE_DIGITS), where
# scale is the size of the bounds. Above that, rounding at the last step could
# have made a zero margin negative.
NOISE_DIGITS = 10


class DualityCertificate(BaseModel):
    """A point, the primal and dual bounds on the margin there, and the claimed violation.

    The values are decimal strings computed at `digits` digits.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["duality"] = "duality"
    point: Instance
    primal: str
    dual: str
    digits: int = Field(ge=1, le=MAX_DIGITS)
    violation: str


def check_duality_margin(enc: Encoding, cert: DualityCertificate, point: Instance) -> MarginCheck:
    """Recompute both bounds and check them against the claim.

    Requires: the recomputed bounds match the claimed ones; both lie below
    -scale * 10^-(digits - 10), where scale is the size of the bounds; the
    claimed violation lies between them, so it is negative too; and the gap
    between them is smaller than 1/1000 of the violation.

    The bounds are mpf values, not enclosures. The floor catches a zero margin
    that rounding at the final step makes negative. It cannot catch rounding
    in large intermediate terms; the repeat check in `check` is a heuristic
    against that, not a proof.
    """
    if not isinstance(enc, DualBounds):
        return MarginCheck.failed(
            reject(
                "the encoding has no primal and dual bounds, so a duality certificate "
                "cannot be checked"
            )
        )
    if cert.digits < MIN_DUALITY_DIGITS:
        return MarginCheck.failed(
            reject(
                f"a duality certificate needs at least {MIN_DUALITY_DIGITS} digits, "
                f"not {cert.digits}"
            )
        )
    try:
        claimed_primal, claimed_dual, violation = (
            to_mpf(v) for v in (cert.primal, cert.dual, cert.violation)
        )
    except (ValueError, ZeroDivisionError) as exc:
        return MarginCheck.failed(reject(f"a claimed value is not a number: {exc}"))
    primal, dual = enc.primal_dual(point)
    values = (claimed_primal, claimed_dual, violation, primal, dual)
    if not all(mpmath.isfinite(v) for v in values):
        return MarginCheck.failed(reject("a bound or the violation is not finite"))
    scale = max(mpmath.mpf(1), abs(claimed_primal), abs(claimed_dual))
    tol = scale * mpmath.mpf(10) ** -(cert.digits - 5)
    if abs(primal - claimed_primal) > tol or abs(dual - claimed_dual) > tol:
        return MarginCheck.failed(
            reject(
                f"the recomputed bounds ({mpmath.nstr(primal, 15)}, {mpmath.nstr(dual, 15)}) "
                f"do not match the claimed ({cert.primal}, {cert.dual})"
            )
        )
    high, low = max(primal, dual), min(primal, dual)
    if high >= 0:
        return MarginCheck.failed(
            reject(f"the upper bound on the margin, {mpmath.nstr(high, 15)}, is not negative")
        )
    floor = -scale * mpmath.mpf(10) ** -(cert.digits - NOISE_DIGITS)
    if high >= floor:
        return MarginCheck.failed(
            reject(
                f"the upper bound on the margin, {mpmath.nstr(high, 15)}, is not "
                f"distinguishable from zero at {cert.digits} digits"
            )
        )
    if not low - tol <= violation <= high + tol:
        return MarginCheck.failed(
            reject(
                f"the claimed violation {cert.violation} lies outside the bounds "
                f"[{mpmath.nstr(low, 15)}, {mpmath.nstr(high, 15)}]"
            )
        )
    gap = high - low
    if gap * GAP_SHARE >= abs(violation):
        return MarginCheck.failed(
            reject(
                f"the duality gap {mpmath.nstr(gap, 5)} is not below 1/{GAP_SHARE} of the "
                f"violation {mpmath.nstr(violation, 5)}"
            )
        )
    return MarginCheck(
        result=CheckResult(
            accepted=True,
            reason=f"margin in [{mpmath.nstr(low, 15)}, {mpmath.nstr(high, 15)}], "
            f"gap {mpmath.nstr(gap, 3)}",
        ),
        lower=low,
        upper=high,
    )
