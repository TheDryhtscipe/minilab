"""The result of a certificate check, and the precision context checks run in."""

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass

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


@dataclass(frozen=True)
class MarginCheck:
    """The margin part of a check, with an arb ball that contains the margin.

    The ball decides whether the margin is close enough to zero that the repeat
    check needs at least 100 digits, and whether the two checks agree. It is
    None when the check failed. A ball keeps the full range and exactness of an
    exact margin, which a float would overflow and an mpf would round.
    """

    result: CheckResult
    enclosure: flint.arb | None

    @classmethod
    def failed(cls, result: CheckResult) -> "MarginCheck":
        return cls(result=result, enclosure=None)


def show(exact: flint.arb) -> str:
    """Write an exact arb (radius 0, such as `ball.upper()`) to 15 digits."""
    return exact.str(15, radius=False)


@contextmanager
def working_digits(digits: int) -> Generator[None]:
    """Run mpmath and python-flint at `digits` decimal digits, then restore both.

    Both libraries keep their precision in global state, so checks must not run
    in parallel threads of one process. Encoding code can change it too; the
    checker rejects a certificate if an encoding call leaves it changed.
    """
    saved_flint = flint.ctx.prec
    try:
        flint.ctx.dps = digits
        with mpmath.mp.workdps(digits):
            yield
    finally:
        flint.ctx.prec = saved_flint
