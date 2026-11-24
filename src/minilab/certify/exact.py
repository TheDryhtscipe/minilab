"""Exact certificates: the violation holds in exact rational arithmetic."""

from fractions import Fraction
from typing import Literal

from pydantic import BaseModel, ConfigDict

from minilab.certify.result import CheckResult, MarginCheck, reject
from minilab.encoding.api import Encoding, ExactMargin, Instance, to_arb


class ExactCertificate(BaseModel):
    """A rational point and the exact margin claimed there."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["exact"] = "exact"
    point: Instance
    margin: Fraction


def check_exact_margin(enc: Encoding, cert: ExactCertificate, point: Instance) -> MarginCheck:
    """Recompute the margin exactly; it must equal the claim and be negative."""
    if not isinstance(enc, ExactMargin):
        return MarginCheck.failed(
            reject("the encoding has no exact margin, so an exact certificate cannot be checked")
        )
    if not point.is_rational():
        return MarginCheck.failed(reject("an exact certificate needs a rational point"))
    margin = enc.exact_margin(point)
    if not isinstance(margin, Fraction):  # pyright: ignore[reportUnnecessaryIsInstance]
        return MarginCheck.failed(
            reject(f"the exact margin is not a Fraction: {type(margin).__name__}")
        )
    if margin != cert.margin:
        return MarginCheck.failed(
            reject(f"claimed margin {cert.margin} but the exact margin is {margin}")
        )
    if margin >= 0:
        return MarginCheck.failed(reject(f"the exact margin {margin} is not negative"))
    return MarginCheck(
        result=CheckResult(accepted=True, reason=f"exact margin {margin} < 0"),
        enclosure=to_arb(margin),
    )
