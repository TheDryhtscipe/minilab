"""Interval certificates: a rigorous enclosure of the margin lies below zero."""

from typing import Literal

import flint
from pydantic import BaseModel, ConfigDict, Field

from minilab.certify.result import CheckResult, MarginCheck, reject
from minilab.encoding.api import MAX_DIGITS, BallMargin, Encoding, Instance, to_arb, to_fraction


class IntervalCertificate(BaseModel):
    """A point, a precision, and a claimed enclosure [lower, upper] of the margin."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["interval"] = "interval"
    point: Instance
    digits: int = Field(ge=1, le=MAX_DIGITS)
    enclosure: tuple[str, str]


def check_interval_margin(enc: Encoding, cert: IntervalCertificate, point: Instance) -> MarginCheck:
    """Enclose the margin with arb at the current precision.

    The ball must lie wholly below zero and meet the claimed enclosure.
    """
    if not isinstance(enc, BallMargin):
        return MarginCheck.failed(
            reject("the encoding has no ball margin, so an interval certificate cannot be checked")
        )
    try:
        low, high = (to_fraction(v) for v in cert.enclosure)
    except (ValueError, ZeroDivisionError) as exc:
        return MarginCheck.failed(reject(f"the claimed enclosure is not two rationals: {exc}"))
    if low > high:
        return MarginCheck.failed(reject(f"the claimed enclosure is empty: [{low}, {high}]"))
    ball = enc.margin_ball(point)
    # Exactly an arb: a subclass could override the comparisons below.
    if type(ball) is not flint.arb:  # pyright: ignore[reportUnnecessaryComparison]
        return MarginCheck.failed(reject(f"the ball margin is not an arb: {type(ball).__name__}"))
    if not ball.is_finite():
        return MarginCheck.failed(reject(f"the margin enclosure is not finite: {ball}"))
    if not ball < 0:
        return MarginCheck.failed(reject(f"the margin enclosure {ball} does not lie below zero"))
    claimed = to_arb(cert.enclosure[0]).union(to_arb(cert.enclosure[1]))
    if not ball.overlaps(claimed):
        return MarginCheck.failed(
            reject(f"the margin enclosure {ball} does not meet the claimed [{low}, {high}]")
        )
    return MarginCheck(
        result=CheckResult(accepted=True, reason=f"margin enclosure {ball} < 0"),
        enclosure=ball,
    )
