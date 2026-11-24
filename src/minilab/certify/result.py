"""The result of a certificate check, and the precision context checks run in."""

from collections.abc import Generator
from contextlib import contextmanager

import flint
import mpmath
from pydantic import BaseModel, ConfigDict


class CheckResult(BaseModel):
    """Whether a certificate was accepted, and why."""

    model_config = ConfigDict(frozen=True)

    accepted: bool
    reason: str


def accept(reason: str) -> CheckResult:
    return CheckResult(accepted=True, reason=reason)


def reject(reason: str) -> CheckResult:
    return CheckResult(accepted=False, reason=reason)


class MarginCheck(BaseModel):
    """The margin part of a check, with a certified upper bound on the margin.

    `upper` is used only to decide whether the margin is close enough to zero
    that the check must be repeated at higher precision.
    """

    model_config = ConfigDict(frozen=True)

    result: CheckResult
    upper: float

    @classmethod
    def failed(cls, result: CheckResult) -> "MarginCheck":
        return cls(result=result, upper=float("nan"))


@contextmanager
def working_digits(digits: int) -> Generator[None]:
    """Run mpmath and python-flint at `digits` decimal digits, then restore both.

    Both libraries keep their precision in global state, so checks must not run
    in parallel threads of one process.
    """
    saved_flint = flint.ctx.prec
    try:
        flint.ctx.dps = digits
        with mpmath.mp.workdps(digits):
            yield
    finally:
        flint.ctx.prec = saved_flint
