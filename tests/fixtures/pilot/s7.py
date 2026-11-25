"""Encoding of the pilot's conjecture S7, rebuilt by hand as a test fixture.

S7 (pilot run, 23 November 2026): for any finite source p and any distortion
d: X x Y -> [0, 1], R(D) - 2 D^2 is convex on [D_min, D_max].

Equivalently, at any three distortions D1 < D2 < D3 in [D_min, D_max] the
second divided difference R[D1, D2, D3] is at least 2, the divided difference
of 2 D^2. The weights below carry a factor of 2, so the margin is
2 R[D1, D2, D3] - 4. R is defined by an optimisation, so the encoding gives a
primal and a dual bound on the margin, as arb balls, from witnesses carried in
the instance:

- for each of the three points, a slope s >= 0 and an output law q > 0. They
  define the channel W(y|x) = q_y exp(-s d(x,y)) / Z_x, whose distortion is D_i;
- upper bound: R(D_i) <= I(p, W), because W is a channel with distortion D_i;
- lower bound (Csiszar dual, valid for any s >= 0 and q > 0):
  R(D_i) >= sum_x p_x log(1/Z_x) - log max_y c_y - s D_i,
  with c_y = sum_x p_x exp(-s d(x,y)) / Z_x.

The pilot's own certificate for S7' (s7_counter3.py) uses the same bounds.

The divided difference is sum_i w_i R(D_i) with w_i = 2 / prod_{j != i}
(D_i - D_j), in any order of the points. An upper bound on it takes the upper
rate bound where w_i > 0 and the lower one where w_i < 0; a lower bound takes
the reverse. In ball arithmetic each ball contains the true value, so the
highest point of the upper ball lies above the true margin and the lowest point
of the lower ball below it.

Instance values: `p` (n_x decimals), `d` (n_x * n_y decimals, row by row),
`s` (3 decimals), `q` (3 * n_y decimals, one law after another; each is
normalised before use). All are exact decimals. The assumptions read them
exactly; the ball bounds read them with to_arb, as balls that contain them.
`margin`, the domain check and the near-degeneracy flag read them with to_mpf,
rounded to the working precision, so the point they evaluate moves by that
rounding.
"""

import random
from fractions import Fraction

import flint
import mpmath

from minilab.encoding.api import Instance, to_arb, to_fraction, to_mpf


class S7Point:
    """The distortion and the two rate bounds at one witness (s, q), in mpf."""

    def __init__(self, distortion: mpmath.mpf, upper: mpmath.mpf, lower: mpmath.mpf) -> None:
        self.distortion = distortion
        self.upper = upper
        self.lower = lower


class S7Ball:
    """The distortion and the two rate bounds at one witness (s, q), as arb balls."""

    def __init__(self, distortion: flint.arb, upper: flint.arb, lower: flint.arb) -> None:
        self.distortion = distortion
        self.upper = upper
        self.lower = lower


def _arb_sum(terms: list[flint.arb]) -> flint.arb:
    total = flint.arb(0)
    for term in terms:
        total += term
    return total


def _list(x: Instance, key: str) -> list[str]:
    value = x.values[key]
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


class S7Encoding:
    """R(D) - 2 D^2 is convex, encoded through three witnessed points of R."""

    def _shape(self, x: Instance) -> tuple[int, int]:
        nx = len(_list(x, "p"))
        nd = len(_list(x, "d"))
        if nx == 0 or nd % nx != 0:
            raise ValueError("d must have n_x rows of equal length")
        ny = nd // nx
        if len(_list(x, "s")) != 3 or len(_list(x, "q")) != 3 * ny:
            raise ValueError("need three slopes and three output laws")
        return nx, ny

    def points(self, x: Instance) -> list[S7Point]:
        """Evaluate the three witnessed points at the current mpmath precision."""
        nx, ny = self._shape(x)
        p = [to_mpf(v) for v in _list(x, "p")]
        d_flat = [to_mpf(v) for v in _list(x, "d")]
        d = [d_flat[i * ny : (i + 1) * ny] for i in range(nx)]
        out: list[S7Point] = []
        for k, s_text in enumerate(_list(x, "s")):
            s = to_mpf(s_text)
            q_raw = [to_mpf(v) for v in _list(x, "q")[k * ny : (k + 1) * ny]]
            if s < 0 or min(q_raw) <= 0:
                raise ValueError("witness needs s >= 0 and q > 0")
            total = mpmath.fsum(q_raw)
            q = [v / total for v in q_raw]
            e = [[mpmath.exp(-s * d[i][j]) for j in range(ny)] for i in range(nx)]
            z = [mpmath.fsum(q[j] * e[i][j] for j in range(ny)) for i in range(nx)]
            w = [[q[j] * e[i][j] / z[i] for j in range(ny)] for i in range(nx)]
            c = [mpmath.fsum(p[i] * e[i][j] / z[i] for i in range(nx)) for j in range(ny)]
            r = [mpmath.fsum(p[i] * w[i][j] for i in range(nx)) for j in range(ny)]
            distortion = mpmath.fsum(p[i] * w[i][j] * d[i][j] for i in range(nx) for j in range(ny))
            rate = mpmath.fsum(
                p[i] * w[i][j] * mpmath.log(w[i][j] / r[j])
                for i in range(nx)
                for j in range(ny)
                if p[i] > 0
            )
            dual = (
                mpmath.fsum(p[i] * mpmath.log(1 / z[i]) for i in range(nx))
                - mpmath.log(max(c))
                - s * distortion
            )
            out.append(S7Point(distortion, rate, dual))
        return sorted(out, key=lambda pt: pt.distortion)

    def balls(self, x: Instance) -> list[S7Ball]:
        """Enclose the three witnessed points at python-flint's current precision."""
        nx, ny = self._shape(x)
        p = [to_arb(v) for v in _list(x, "p")]
        d_flat = [to_arb(v) for v in _list(x, "d")]
        d = [d_flat[i * ny : (i + 1) * ny] for i in range(nx)]
        out: list[S7Ball] = []
        for k, s_text in enumerate(_list(x, "s")):
            s = to_arb(s_text)
            q_raw = [to_arb(v) for v in _list(x, "q")[k * ny : (k + 1) * ny]]
            if not (s >= 0 and all(v > 0 for v in q_raw)):
                raise ValueError("witness needs s >= 0 and q > 0")
            total = _arb_sum(q_raw)
            q = [v / total for v in q_raw]
            e = [[(-s * d[i][j]).exp() for j in range(ny)] for i in range(nx)]
            z = [_arb_sum([q[j] * e[i][j] for j in range(ny)]) for i in range(nx)]
            w = [[q[j] * e[i][j] / z[i] for j in range(ny)] for i in range(nx)]
            c = [_arb_sum([p[i] * e[i][j] / z[i] for i in range(nx)]) for j in range(ny)]
            r = [_arb_sum([p[i] * w[i][j] for i in range(nx)]) for j in range(ny)]
            distortion = _arb_sum([p[i] * w[i][j] * d[i][j] for i in range(nx) for j in range(ny)])
            rate = _arb_sum(
                [
                    p[i] * w[i][j] * (w[i][j] / r[j]).log()
                    for i in range(nx)
                    for j in range(ny)
                    if not p[i] == 0
                ]
            )
            # An upper bound on max_y c_y: the largest upper end of the balls.
            c_max = c[0].upper()
            for cy in c[1:]:
                if cy.upper() > c_max:
                    c_max = cy.upper()
            dual = (
                _arb_sum([p[i] * (1 / z[i]).log() for i in range(nx)])
                - c_max.log()
                - s * distortion
            )
            out.append(S7Ball(distortion, rate, dual))
        return out

    def primal_dual(self, x: Instance) -> tuple[flint.arb, flint.arb]:
        """Return (upper, lower) balls bounding the second divided difference minus 4."""
        pts = self.balls(x)
        upper = flint.arb(0)
        lower = flint.arb(0)
        for i, pt in enumerate(pts):
            denominator = flint.arb(1)
            for j, other in enumerate(pts):
                if j != i:
                    denominator *= pt.distortion - other.distortion
            weight = 2 / denominator
            if weight > 0:
                upper += weight * pt.upper
                lower += weight * pt.lower
            elif weight < 0:
                upper += weight * pt.lower
                lower += weight * pt.upper
            else:
                raise ValueError("the distortions are too close to order")
        return upper - 4, lower - 4

    def margin(self, x: Instance) -> mpmath.mpf:
        """The upper bound on the margin, in mpf at the current precision."""
        a, b, c = self.points(x)
        d1, d2, d3 = a.distortion, b.distortion, c.distortion
        w1 = 2 / ((d2 - d1) * (d3 - d1))
        w2 = 2 / ((d2 - d1) * (d3 - d2))
        w3 = 2 / ((d3 - d2) * (d3 - d1))
        return w1 * a.upper - w2 * b.lower + w3 * c.upper - 4

    def assumptions(self, x: Instance) -> list[bool]:
        self._shape(x)
        p = [to_fraction(v) for v in _list(x, "p")]
        d = [to_fraction(v) for v in _list(x, "d")]
        return [
            all(v >= 0 for v in p) and sum(p, Fraction(0)) == 1,
            all(0 <= v <= 1 for v in d),
        ]

    def in_domain(self, x: Instance) -> bool:
        """D_min <= D1 < D2 < D3 <= D_max, at the current precision."""
        nx, ny = self._shape(x)
        p = [to_mpf(v) for v in _list(x, "p")]
        d_flat = [to_mpf(v) for v in _list(x, "d")]
        d = [d_flat[i * ny : (i + 1) * ny] for i in range(nx)]
        d_min = mpmath.fsum(p[i] * min(d[i]) for i in range(nx))
        d_max = min(mpmath.fsum(p[i] * d[i][j] for i in range(nx)) for j in range(ny))
        a, b, c = (pt.distortion for pt in self.points(x))
        return d_min <= a < b < c <= d_max

    def near_degenerate(self, x: Instance) -> bool:
        """Near independence (a rate near 0) or near the edge of q's support."""
        _, ny = self._shape(x)
        q = [to_mpf(v) for v in _list(x, "q")]
        rates = [pt.upper for pt in self.points(x)]
        smallest_q = min(
            q[k * ny + j] / mpmath.fsum(q[k * ny : (k + 1) * ny])
            for k in range(3)
            for j in range(ny)
        )
        return min(rates) < mpmath.mpf("1e-6") or smallest_q < mpmath.mpf("1e-6")

    def sample(self, rng: random.Random) -> Instance:
        nx, ny = rng.randint(2, 3), rng.randint(2, 3)
        weights = [Fraction(rng.randint(1, 20)) for _ in range(nx)]
        total = sum(weights, Fraction(0))
        values: dict[str, Fraction | str | list[str]] = {
            "p": [str(w / total) for w in weights],
            "d": [str(Fraction(rng.randint(0, 100), 100)) for _ in range(nx * ny)],
            "s": [str(Fraction(rng.randint(1, 400), 100)) for _ in range(3)],
            "q": [str(Fraction(rng.randint(1, 20), 20)) for _ in range(3 * ny)],
        }
        return Instance(values=values, precision_digits=30)
