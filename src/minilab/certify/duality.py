"""Duality certificates: matching primal and dual bounds on an optimisation-defined margin."""

from fractions import Fraction
from typing import Literal, cast

import flint
from pydantic import BaseModel, ConfigDict, Field

from minilab.certify.result import CheckResult, MarginCheck, reject, show
from minilab.encoding.api import MAX_DIGITS, DualBounds, Encoding, Instance, to_arb, to_fraction

# A duality certificate must be computed at high precision (design doc-gmj,
# "Falsifier"). Float precision gave the pilot its spurious ratios.
MIN_DUALITY_DIGITS = 30

# The gap between the bounds must be smaller than this share of the upper bound.
GAP_SHARE = 1000

# A claimed bound matches the recomputed ball if it lies within
# scale * 10^-(digits - MATCH_DIGITS) of it, where scale is the size of the
# claimed bounds and at least 1.
MATCH_DIGITS = 5


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
    """Recompute both bounds as arb balls and check them against the claim.

    Requires: no claimed bound lies certainly further than the tolerance from
    its recomputed ball; the highest point of the two balls lies below zero; the
    claimed violation is negative and lies between the bounds, give or take the
    tolerance; and the gap between the lowest and highest points of the balls
    is certainly smaller than 1/1000 of the upper bound's distance from zero.

    The sign test is rigorous if the encoding's balls are honest: the margin
    lies at or below the highest point of the two balls.
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
            to_fraction(v) for v in (cert.primal, cert.dual, cert.violation)
        )
    except (ValueError, ZeroDivisionError) as exc:
        return MarginCheck.failed(reject(f"a claimed value is not a number: {exc}"))
    # Encoding code is untrusted: check the shape of what it returns.
    bounds = cast(object, enc.primal_dual(point))
    # Exact types: a subclass could override iteration or the comparisons below.
    if not (
        type(bounds) is tuple
        and len(cast(tuple[object, ...], bounds)) == 2
        and all(type(b) is flint.arb for b in cast(tuple[object, ...], bounds))
    ):
        return MarginCheck.failed(
            reject(f"the primal and dual bounds are not two arb balls: {bounds!r:.100}")
        )
    primal, dual = cast(tuple[flint.arb, flint.arb], bounds)
    if not (primal.is_finite() and dual.is_finite()):
        return MarginCheck.failed(reject(f"a bound is not finite: {primal}, {dual}"))
    scale = max(Fraction(1), abs(claimed_primal), abs(claimed_dual))
    tol = to_arb(scale * Fraction(1, 10 ** (cert.digits - MATCH_DIGITS)))
    if abs(primal - to_arb(claimed_primal)) > tol or abs(dual - to_arb(claimed_dual)) > tol:
        return MarginCheck.failed(
            reject(
                f"the recomputed bounds ({primal}, {dual}) "
                f"do not match the claimed ({cert.primal}, {cert.dual})"
            )
        )
    # Exact arbs (radius 0), so these comparisons are decided.
    high = primal.upper() if primal.upper() > dual.upper() else dual.upper()
    low = primal.lower() if primal.lower() < dual.lower() else dual.lower()
    if not high < 0:
        return MarginCheck.failed(
            reject(f"the upper bound on the margin, {show(high)}, is not below zero")
        )
    claim = to_arb(violation)
    if not claim < 0:
        return MarginCheck.failed(reject(f"the claimed violation {cert.violation} is not negative"))
    if claim < low - tol or claim > high + tol:
        return MarginCheck.failed(
            reject(
                f"the claimed violation {cert.violation} lies outside the bounds "
                f"[{show(low)}, {show(high)}]"
            )
        )
    gap = high - low
    # Against the recomputed upper bound, not the claim: the tolerance would
    # let a claim be inflated past both bounds.
    if not gap * GAP_SHARE < abs(high):
        return MarginCheck.failed(
            reject(
                f"the duality gap {gap.upper().str(5, radius=False)} is not below "
                f"1/{GAP_SHARE} of the upper bound {show(high)}"
            )
        )
    return MarginCheck(
        result=CheckResult(
            accepted=True,
            reason=f"margin in [{show(low)}, {show(high)}], gap {gap.upper().str(3, radius=False)}",
        ),
        enclosure=low.union(high),
    )
