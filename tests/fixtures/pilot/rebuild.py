"""Rebuild the pilot certificate fixtures (build plan doc-5wh, Task 6).

Run from the repo root: `PYTHONPATH=tests uv run python tests/fixtures/pilot/rebuild.py`.
It writes one JSON certificate per case next to this file. Each certificate is
what a falsifier would submit; the tests check that the library accepts or
rejects it. Sources, all in ~/dev/sci/experiments/crossconj-ml (read only):

- warm_start_artefact: s7_fixed.py/.out. Seed 7, trial 6: a 3-letter source
  with Hamming distortion, Blahut-Arimoto warm-started along the slope grid.
  At s = 3.876 and 4.041 the warm q keeps q_1 near 0 (1e-42, 1e-21) where the
  optimum has full support. p is rounded to 4 places; the artefact persists.
- grid_gap: s7_y2.py/.out. The float grid cell c = 12, a1 = b = 0.001069 gave
  rho = 0.2500111 > 1/4; mpmath does not.
- out_of_domain: structured.py/.out, S7. The kill used s = 0, where the
  distortion is 0.5 > D_max = 0.4625 for p = (0.5375, 0.4625), Hamming d.
- s7_counterexample: s7_counter.py/.out. P = [[.20,.15,.05],[.05,.15,.40]],
  s = K = 1.97408, three points on the linear segment (t at 0.1, 0.5, 0.9 of
  its range), 40 digits.
- s7prime_counterexample: s7_counter3.py/.out. The 2x3 base plus a third
  letter of mass 0.01 with d row (0, 1/2, 1), slopes 0.9755K, 0.9775K,
  0.9795K, 60 digits.
- s1_side_counterexample: structured.py/.out, S1 side test, q = 1.05.
"""

import json
import math
from pathlib import Path

import mpmath

from fixtures.pilot.s1_side import TsallisPinskerSide
from fixtures.pilot.s7 import S7Encoding
from fixtures.pilot.s7_y2 import rho_float
from minilab.certify.result import working_digits
from minilab.encoding.api import Instance

HERE = Path(__file__).parent
F = mpmath.mpf


def _write(name: str, cert: dict[str, object]) -> None:
    (HERE / f"{name}.json").write_text(json.dumps(cert, indent=2) + "\n")


def _duality(name: str, x: Instance, digits: int, naive: bool = False) -> None:
    """Write a duality certificate with the bounds the encoding gives at x.

    With `naive`, the claimed violation is what a falsifier that trusts its
    solver would report: the divided difference of the primal rates alone.
    """
    with working_digits(digits):
        primal_ball, dual_ball = S7Encoding().primal_dual(x)
        primal, dual = (
            mpmath.mpf(ball.mid().str(digits + 5, radius=False))
            for ball in (primal_ball, dual_ball)
        )
        violation = primal
        if naive:
            a, b, c = S7Encoding().points(x)
            d1, d2, d3 = a.distortion, b.distortion, c.distortion
            violation = (
                2 * a.upper / ((d2 - d1) * (d3 - d1))
                - 2 * b.upper / ((d2 - d1) * (d3 - d2))
                + 2 * c.upper / ((d3 - d2) * (d3 - d1))
                - 4
            )
        cert: dict[str, object] = {
            "kind": "duality",
            "point": x.model_dump(mode="json"),
            "primal": mpmath.nstr(primal, digits),
            "dual": mpmath.nstr(dual, digits),
            "digits": digits,
            "violation": mpmath.nstr(violation, digits),
        }
    _write(name, cert)


def _base_2x3() -> tuple[list[list[mpmath.mpf]], mpmath.mpf, list[mpmath.mpf], list[mpmath.mpf]]:
    """d, K, q and the kernel vector h of the pilot's 2x3 joint law."""
    joint = [[F("0.20"), F("0.15"), F("0.05")], [F("0.05"), F("0.15"), F("0.40")]]
    p = [sum(r, F(0)) for r in joint]
    q = [joint[0][y] + joint[1][y] for y in range(3)]
    ll = [[mpmath.log(joint[x][y] / (p[x] * q[y])) for y in range(3)] for x in range(2)]
    k = max(max(r) - min(r) for r in ll)
    d = [[(max(ll[x]) - ll[x][y]) / k for y in range(3)] for x in range(2)]
    w = [[joint[x][y] / p[x] for y in range(3)] for x in range(2)]
    h = [
        w[0][1] * w[1][2] - w[0][2] * w[1][1],
        w[0][2] * w[1][0] - w[0][0] * w[1][2],
        w[0][0] * w[1][1] - w[0][1] * w[1][0],
    ]
    return d, k, q, h


def s7_counterexample() -> None:
    with mpmath.mp.workdps(60):
        d, k, q, h = _base_2x3()
        tmin = max(-1 / h[y] for y in range(3) if h[y] > 0)
        tmax = min(-1 / h[y] for y in range(3) if h[y] < 0)
        qs: list[str] = []
        for frac in ("0.1", "0.5", "0.9"):
            t = tmin + F(frac) * (tmax - tmin)
            qs += [mpmath.nstr(q[y] * (1 + t * h[y]), 45) for y in range(3)]
        x = Instance(
            values={
                "p": ["0.40", "0.60"],
                "d": [mpmath.nstr(v, 45) for r in d for v in r],
                "s": [mpmath.nstr(k, 45)] * 3,
                "q": qs,
            },
            precision_digits=40,
        )
    _duality("s7_counterexample", x, 40)


def _kkt_q(p: list[mpmath.mpf], d: list[list[mpmath.mpf]], s: mpmath.mpf) -> list[mpmath.mpf]:
    """The optimal output law at slope s: Blahut-Arimoto from uniform, then Newton on c(q) = 1."""
    n = len(d[0])
    e = [[mpmath.exp(-s * v) for v in row] for row in d]
    q = [F(1) / n] * n
    for _ in range(3000):
        z = [mpmath.fsum(q[j] * row[j] for j in range(n)) for row in e]
        q = [mpmath.fsum(p[i] * q[j] * e[i][j] / z[i] for i in range(len(p))) for j in range(n)]
    for _ in range(60):
        z = [mpmath.fsum(q[j] * row[j] for j in range(n)) for row in e]
        c = [mpmath.fsum(p[i] * e[i][j] / z[i] for i in range(len(p))) - 1 for j in range(n)]
        jac = [
            [
                -mpmath.fsum(p[i] * e[i][j] * e[i][m] / z[i] ** 2 for i in range(len(p)))
                for m in range(n)
            ]
            for j in range(n)
        ]
        # Solve jac * step = c by Gaussian elimination with partial pivoting.
        a = [row[:] + [c[j]] for j, row in enumerate(jac)]
        for col in range(n):
            piv = max(range(col, n), key=lambda r: abs(a[r][col]))
            a[col], a[piv] = a[piv], a[col]
            for r in range(col + 1, n):
                f = a[r][col] / a[col][col]
                a[r] = [u - f * v for u, v in zip(a[r], a[col], strict=True)]
        step = [F(0)] * n
        for r in reversed(range(n)):
            step[r] = (a[r][n] - mpmath.fsum(a[r][m] * step[m] for m in range(r + 1, n))) / a[r][r]
        q = [qj - sj for qj, sj in zip(q, step, strict=True)]
    return q


def s7prime_counterexample() -> None:
    with mpmath.mp.workdps(80):
        d0, k, _, _ = _base_2x3()
        d_text = [[mpmath.nstr(v, 65) for v in r] for r in d0] + [["0", "0.5", "1"]]
        d = [[F(v) for v in r] for r in d_text]
        p_text = ["0.396", "0.594", "0.01"]
        p = [F(v) for v in p_text]
        s_text = [mpmath.nstr(k * F(f), 65) for f in ("0.9755", "0.9775", "0.9795")]
        qs: list[str] = []
        for st in s_text:
            qs += [mpmath.nstr(v, 70) for v in _kkt_q(p, d, F(st))]
        x = Instance(
            values={"p": p_text, "d": [v for r in d_text for v in r], "s": s_text, "q": qs},
            precision_digits=60,
        )
    _duality("s7prime_counterexample", x, 60)


def warm_start_artefact() -> None:
    x = Instance(
        values={
            "p": ["0.3551", "0.1898", "0.4551"],
            "d": ["0", "1", "1", "1", "0", "1", "1", "1", "0"],
            "s": ["3.876278466736944", "4.0412505117891735", "4.213243666362372"],
            "q": [
                "0.4356743325452064",
                "2.3550173279583027e-42",
                "0.5643256674547936",
                "0.43607875784031175",
                "1.1541330947999721e-21",
                "0.5639212421596882",
                "0.3560808431173619",
                "0.18333214324140876",
                "0.4605870136412294",
            ],
        },
        precision_digits=30,
    )
    _duality("warm_start_artefact", x, 30, naive=True)


def out_of_domain() -> None:
    x = Instance(
        values={
            "p": ["0.5375", "0.4625"],
            "d": ["0", "1", "1", "0"],
            "s": ["0", "0", "0"],
            "q": ["0.75", "0.25", "0.5", "0.5", "0.25", "0.75"],
        },
        precision_digits=30,
    )
    _duality("out_of_domain", x, 30)


def grid_gap() -> None:
    # The pilot's grid was np.exp(np.linspace(log(1e-3), log(40), 160)); this is its cell 1.
    start = math.log(1e-3)
    a = math.exp(1 * ((math.log(40) - start) / 159) + start)
    margin = 0.25 - rho_float(12.0, a, a)
    x = Instance(values={"c": "12.0", "a1": repr(a), "b": repr(a)}, precision_digits=16)
    cert: dict[str, object] = {
        "kind": "interval",
        "point": x.model_dump(mode="json"),
        "digits": 16,
        "enclosure": [repr(margin - 1e-9), repr(margin + 1e-9)],
    }
    _write("grid_gap", cert)


def s1_side_counterexample() -> None:
    import flint

    x = Instance(
        values={"q": "1.05", "p": ["0.3801", "0.6199"], "r": ["0.5026", "0.4974"]},
        precision_digits=30,
    )
    flint.ctx.dps = 30
    ball = TsallisPinskerSide().margin_ball(x)
    cert: dict[str, object] = {
        "kind": "interval",
        "point": x.model_dump(mode="json"),
        "digits": 30,
        "enclosure": [
            ball.lower().mid().str(30, radius=False),
            ball.upper().mid().str(30, radius=False),
        ],
    }
    _write("s1_side_counterexample", cert)


if __name__ == "__main__":
    warm_start_artefact()
    grid_gap()
    out_of_domain()
    s7_counterexample()
    s7prime_counterexample()
    s1_side_counterexample()
