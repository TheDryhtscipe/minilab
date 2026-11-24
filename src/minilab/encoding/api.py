"""The encoding API: instances, and the protocols an encoding implements.

An encoding is the executable form of a conjecture: a margin (negative means
violated), a predicate for each assumption, a domain predicate, a sampler, and a
near-degeneracy flag (design doc-gmj, "Encoding").

This is the minimal version that the certificate library needs (build plan
doc-5wh, Task 6). Task 7 completes it. Beyond the plan's `Encoding` protocol it
defines three optional capabilities. Each certificate kind needs one, because
the checker recomputes the margin itself and never trusts a number it is given:

- `ExactMargin`: the margin as an exact rational, for exact certificates;
- `BallMargin`: the margin as an arb ball (python-flint), for interval
  certificates;
- `DualBounds`: a primal and a dual bound on the margin, each an arb ball, for
  quantities defined by an optimisation, for duality certificates.

The checker takes an `EncodingFactory`, not an encoding, and builds a fresh
encoding for each run, so that state kept by one run cannot carry into the next.
"""

import random
import re
from collections.abc import Callable
from fractions import Fraction
from typing import Protocol, runtime_checkable

import flint
import mpmath
from pydantic import BaseModel, ConfigDict, Field

# A value is an exact rational, a decimal string, or a list of decimal strings.
# A string holds an exact decimal or an mpf value written out at the stated
# precision; "a/b" is also accepted, because a Fraction is written that way in
# JSON.
Value = Fraction | str | list[str]

# The most decimal digits a point or a certificate may ask for. The repeat
# check runs at up to twice this.
MAX_DIGITS = 5_000

# Limits on a value written as text, so that reading it is quick and exact:
# Fraction expands "1e-99999999" into a hundred-million-digit integer, and
# Python refuses to read an integer string of more than 4300 digits.
MAX_TEXT = 4_000
MAX_EXPONENT = 10_000

# A value written as text is an ASCII decimal ("-1.5e3", underscores allowed
# between digits) or a fraction of two ASCII integers ("3/4"). Python's own
# readers also take other Unicode digits, hexadecimal, "inf", and "nan".
_DECIMAL = re.compile(
    r"[+-]?(?=[0-9]|\.[0-9])[0-9_]*(?:\.[0-9_]*)?(?:[eE](?P<exp>[+-]?[0-9][0-9_]*))?",
    re.ASCII,
)
_RATIO = re.compile(r"[+-]?[0-9][0-9_]*\s*/\s*[0-9][0-9_]*", re.ASCII)


class Instance(BaseModel):
    """One point of a conjecture's finite family of test instances."""

    model_config = ConfigDict(frozen=True)

    values: dict[str, Value]
    precision_digits: int = Field(ge=1, le=MAX_DIGITS)

    def at_precision(self, digits: int) -> "Instance":
        """Return the same point, to be evaluated at `digits` decimal digits.

        The values do not change, but a decimal read with `to_mpf` is rounded
        to the working precision, so the point evaluated moves by that rounding.
        """
        return self.model_copy(update={"precision_digits": digits})

    def is_rational(self) -> bool:
        """Say whether every value is an exact rational (a Fraction or a finite decimal)."""
        try:
            for value in self.values.values():
                items: list[Fraction | str] = list(value) if isinstance(value, list) else [value]
                for item in items:
                    to_fraction(item)
        except (ValueError, ZeroDivisionError, OverflowError):
            return False
        return True


def to_fraction(value: Fraction | str) -> Fraction:
    """Read a value as an exact rational. Raise ValueError if it is not one."""
    if isinstance(value, Fraction):
        return value
    return Fraction(_checked_text(value))


def _checked_text(value: str) -> str:
    """Strip `value` and check it is a decimal or a ratio within the limits.

    Raise ValueError if it is not.
    """
    text = value.strip()
    if len(text) > MAX_TEXT:
        raise ValueError(f"a value of {len(text)} characters is too long")
    if _RATIO.fullmatch(text):
        return text
    found = _DECIMAL.fullmatch(text)
    if not found:
        raise ValueError(f"not an ASCII decimal or a ratio of integers: {text[:50]!r}")
    exponent = found.group("exp")
    if exponent is not None and abs(int(exponent)) > MAX_EXPONENT:
        raise ValueError(f"the exponent of {text[:50]!r} is beyond {MAX_EXPONENT}")
    return text


def to_mpf(value: Fraction | str) -> mpmath.mpf:
    """Read a value as an mpf at the current working precision."""
    if isinstance(value, Fraction) or "/" in value:
        exact = to_fraction(value)
        return mpmath.mpf(exact.numerator) / exact.denominator
    return mpmath.mpf(_checked_text(value))


def to_arb(value: Fraction | str) -> flint.arb:
    """Read an exact rational value as an arb ball that contains it exactly."""
    exact = to_fraction(value)
    return flint.arb(flint.fmpq(exact.numerator, exact.denominator))


@runtime_checkable
class Encoding(Protocol):
    """The executable form of a conjecture. Margins below zero are violations."""

    def margin(self, x: Instance) -> mpmath.mpf: ...

    def assumptions(self, x: Instance) -> list[bool]: ...

    def in_domain(self, x: Instance) -> bool: ...

    def sample(self, rng: random.Random) -> Instance: ...

    def near_degenerate(self, x: Instance) -> bool: ...


@runtime_checkable
class ExactMargin(Protocol):
    """An encoding whose margin is rational at rational points."""

    def exact_margin(self, x: Instance) -> Fraction: ...


@runtime_checkable
class BallMargin(Protocol):
    """An encoding that can enclose its margin in an arb ball.

    The ball is computed at python-flint's current precision, which the checker
    sets. It must contain the true margin at `x`.
    """

    def margin_ball(self, x: Instance) -> flint.arb: ...


@runtime_checkable
class DualBounds(Protocol):
    """An encoding whose margin is defined by an optimisation.

    `primal_dual` returns two arb balls, computed at python-flint's current
    precision: one bounds the margin from above (from a primal-feasible point)
    and one from below (from a dual-feasible point). The true margin lies
    between the lowest point of the two balls and the highest, whichever ball
    is which.
    """

    def primal_dual(self, x: Instance) -> tuple[flint.arb, flint.arb]: ...


# Builds a fresh encoding. The checker calls it once for each run.
EncodingFactory = Callable[[], Encoding]
