"""Encoding of the pilot's S7 check for |Y| = 2, rebuilt by hand as a test fixture.

For a binary output, the pilot reduced S7 to a closed form rho(c, a1, b) in
three real parameters (s7_y2.py): S7 for |Y| = 2 holds if and only if
rho <= 1/4 for all real c and all a1, b > 0. The margin is 1/4 - rho.

With sigma the logistic function:
    l0, l1, l2 = sigma(c), sigma(c + a1), sigma(c - b)
    m1 = (l1 - l0) / a1,  m2 = (l0 - l2) / b,  S = a1 m1 + b m2
    w1 = l1 (1 - l1),     w2 = l2 (1 - l2)
    F = a1 b / S * (m2 w1 a1 + m1 w2 b + (m2 w1 - m1 w2)^2 / (m1 m2 S))
    rho = F / max(a1, b)^2

For small a1 and b (near independence) m1 and m2 come from differences of
nearly equal numbers. In float, the pilot's grid showed rho = 0.2500111 at
c = 12, a1 = b = 0.00107, a false violation; at 60 digits rho stays below 1/4.

Instance values: `c`, `a1`, `b`, each an exact decimal.
"""

import math
import random
from fractions import Fraction

import flint
import mpmath

from minilab.encoding.api import Instance, to_arb, to_fraction, to_mpf


def _scalar(x: Instance, key: str) -> Fraction | str:
    value = x.values[key]
    if isinstance(value, list):
        raise ValueError(f"{key} must be a single value")
    return value


def rho_float(c: float, a1: float, b: float) -> float:
    """The closed form in IEEE double, as the pilot's grid evaluated it."""

    def sg(t: float) -> float:
        return 1 / (1 + math.exp(-t))

    l0, l1, l2 = sg(c), sg(c + a1), sg(c - b)
    m1 = (l1 - l0) / a1
    m2 = (l0 - l2) / b
    w1 = l1 * (1 - l1)
    w2 = l2 * (1 - l2)
    s = a1 * m1 + b * m2
    f = a1 * b / s * (m2 * w1 * a1 + m1 * w2 * b + (m2 * w1 - m1 * w2) ** 2 / (m1 * m2 * s))
    return f / max(a1, b) ** 2


class Y2Encoding:
    """rho(c, a1, b) <= 1/4: S7 restricted to a binary output alphabet."""

    def margin(self, x: Instance) -> mpmath.mpf:
        c, a1, b = (to_mpf(_scalar(x, k)) for k in ("c", "a1", "b"))

        def sg(t: mpmath.mpf) -> mpmath.mpf:
            return 1 / (1 + mpmath.exp(-t))

        l0, l1, l2 = sg(c), sg(c + a1), sg(c - b)
        m1 = (l1 - l0) / a1
        m2 = (l0 - l2) / b
        w1 = l1 * (1 - l1)
        w2 = l2 * (1 - l2)
        s = a1 * m1 + b * m2
        f = a1 * b / s * (m2 * w1 * a1 + m1 * w2 * b + (m2 * w1 - m1 * w2) ** 2 / (m1 * m2 * s))
        return mpmath.mpf(1) / 4 - f / max(a1, b) ** 2

    def margin_ball(self, x: Instance) -> flint.arb:
        c, a1, b = (to_arb(_scalar(x, k)) for k in ("c", "a1", "b"))
        big = max(to_fraction(_scalar(x, "a1")), to_fraction(_scalar(x, "b")))

        def sg(t: flint.arb) -> flint.arb:
            return 1 / (1 + (-t).exp())

        l0, l1, l2 = sg(c), sg(c + a1), sg(c - b)
        m1 = (l1 - l0) / a1
        m2 = (l0 - l2) / b
        w1 = l1 * (1 - l1)
        w2 = l2 * (1 - l2)
        s = a1 * m1 + b * m2
        f = a1 * b / s * (m2 * w1 * a1 + m1 * w2 * b + (m2 * w1 - m1 * w2) ** 2 / (m1 * m2 * s))
        return flint.arb(1) / 4 - f / to_arb(big) ** 2

    def assumptions(self, x: Instance) -> list[bool]:
        to_fraction(_scalar(x, "c"))
        return [to_fraction(_scalar(x, "a1")) > 0, to_fraction(_scalar(x, "b")) > 0]

    def in_domain(self, x: Instance) -> bool:
        return True

    def near_degenerate(self, x: Instance) -> bool:
        """Near independence: a small log-likelihood gap on either side."""
        small = Fraction(1, 100)
        return to_fraction(_scalar(x, "a1")) < small or to_fraction(_scalar(x, "b")) < small

    def sample(self, rng: random.Random) -> Instance:
        values: dict[str, Fraction | str | list[str]] = {
            "c": str(Fraction(rng.randint(-1200, 1200), 100)),
            "a1": str(Fraction(rng.randint(1, 4000), 100)),
            "b": str(Fraction(rng.randint(1, 4000), 100)),
        }
        return Instance(values=values, precision_digits=30)
