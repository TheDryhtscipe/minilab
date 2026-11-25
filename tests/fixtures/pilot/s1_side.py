"""Encoding of the pilot's S1 side test, rebuilt by hand as a test fixture.

S1 (pilot run, 23 November 2026), side test: for q in (1, 2], the Bregman
divergence of phi_q(p) = (sum_i p_i^q - 1) / (q - 1) on the simplex satisfies
B_q(p, r) >= (q / 2) ||p - r||_1^2. The pilot found counterexamples (for
example q = 1.05, p = (0.3801, 0.6199), r = (0.5026, 0.4974), margin -0.0008).
This is a genuine kill: it is the interval-certificate case that must be
accepted.

The margin is B_q(p, r) - (q / 2) ||p - r||_1^2, with
B_q(p, r) = phi_q(p) - phi_q(r) - sum_i q r_i^(q-1) (p_i - r_i) / (q - 1).

Instance values: `q` (a decimal), `p` and `r` (lists of decimals).
"""

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


def _list(x: Instance, key: str) -> list[str]:
    value = x.values[key]
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


class TsallisPinskerSide:
    """B_q(p, r) >= (q/2) ||p - r||_1^2 for q in (1, 2]."""

    def margin(self, x: Instance) -> mpmath.mpf:
        q = to_mpf(_scalar(x, "q"))
        p = [to_mpf(v) for v in _list(x, "p")]
        r = [to_mpf(v) for v in _list(x, "r")]

        def phi(u: list[mpmath.mpf]) -> mpmath.mpf:
            return (mpmath.fsum(v**q for v in u) - 1) / (q - 1)

        grad = [q * v ** (q - 1) / (q - 1) for v in r]
        bregman = (
            phi(p) - phi(r) - mpmath.fsum(g * (a - b) for g, a, b in zip(grad, p, r, strict=True))
        )
        l1 = mpmath.fsum(abs(a - b) for a, b in zip(p, r, strict=True))
        return bregman - q / 2 * l1**2

    def margin_ball(self, x: Instance) -> flint.arb:
        q = to_arb(_scalar(x, "q"))
        p = [to_arb(v) for v in _list(x, "p")]
        r = [to_arb(v) for v in _list(x, "r")]

        def power(v: flint.arb, e: flint.arb) -> flint.arb:
            return (e * v.log()).exp()

        def phi(u: list[flint.arb]) -> flint.arb:
            total = flint.arb(0)
            for v in u:
                total += power(v, q)
            return (total - 1) / (q - 1)

        inner = flint.arb(0)
        l1 = flint.arb(0)
        for a, b in zip(p, r, strict=True):
            inner += q * power(b, q - 1) / (q - 1) * (a - b)
            l1 += abs(a - b)
        return phi(p) - phi(r) - inner - q / 2 * l1**2

    def assumptions(self, x: Instance) -> list[bool]:
        q = to_fraction(_scalar(x, "q"))
        p = [to_fraction(v) for v in _list(x, "p")]
        r = [to_fraction(v) for v in _list(x, "r")]
        return [
            1 < q <= 2,
            len(p) == len(r) and len(p) >= 2,
            all(v > 0 for v in p) and sum(p, Fraction(0)) == 1,
            all(v > 0 for v in r) and sum(r, Fraction(0)) == 1,
        ]

    def in_domain(self, x: Instance) -> bool:
        return True

    def near_degenerate(self, x: Instance) -> bool:
        """p and r almost equal: both sides of the inequality vanish together."""
        p = [to_fraction(v) for v in _list(x, "p")]
        r = [to_fraction(v) for v in _list(x, "r")]
        return sum((abs(a - b) for a, b in zip(p, r, strict=True)), Fraction(0)) < Fraction(
            1, 10**6
        )

    def sample(self, rng: random.Random) -> Instance:
        a, b = rng.randint(1, 999), rng.randint(1, 999)
        values: dict[str, Fraction | str | list[str]] = {
            "q": str(Fraction(rng.randint(101, 200), 100)),
            "p": [str(Fraction(a, 1000)), str(Fraction(1000 - a, 1000))],
            "r": [str(Fraction(b, 1000)), str(Fraction(1000 - b, 1000))],
        }
        return Instance(values=values, precision_digits=30)
