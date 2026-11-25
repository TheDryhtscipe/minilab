---
id: doc-j6t
title: The certificate checker recomputes every number, and a kill needs both encodings
type: decision
scope: internal
created_at: '2026-11-24T22:46:10.037625Z'
updated_at: '2026-11-25T00:02:45.035058Z'
---

# The certificate checker recomputes every number, and a kill needs both encodings

## Decision

The certificate checker (`minilab.certify`) trusts no number in a certificate. It recomputes the margin and the predicates through the encoding. A certificate says where to look and what to expect; the encoding gives the values.

Each certificate kind needs one optional capability beyond the `Encoding` protocol in the build plan (doc-5wh, Task 6):

| Certificate | Capability | Returns |
|---|---|---|
| `ExactCertificate` | `ExactMargin.exact_margin` | the margin as a `Fraction` |
| `IntervalCertificate` | `BallMargin.margin_ball` | an arb ball (python-flint) that contains the margin |
| `DualityCertificate` | `DualBounds.primal_dual` | two arb balls: a primal bound on the margin and a dual bound |

An encoding without the capability cannot have that kind of certificate checked. The protocols live in `minilab.encoding.api`, which Task 7 completes.

`check_certificate(make, cert)` takes an encoding factory, not an encoding. `make` takes no arguments and returns a fresh encoding. The checker calls it once for each run, so state kept on the encoding object, such as a cache or a solver's warm start, cannot carry from the first run to the repeat. The checker rejects a factory that returns the same object twice. It cannot see state kept on the class or in a module.

`check_certificate` accepts only if all of these hold:

1. The point satisfies every assumption predicate and the domain predicate. Each predicate must return a Python `bool`.
2. The margin is negative. For a duality certificate, the highest point of the two balls must lie below zero. The claimed violation must be negative and lie between the bounds, within the tolerance. The gap between the lowest and highest points of the balls must be below 1/1000 of the distance from the highest point to zero. Duality certificates need at least 30 digits.
3. Steps 1 and 2 hold again at twice the precision, with a fresh encoding. The repeat runs at 100 digits or more if the margin is not certainly below -1e-6, or if the encoding flags the point as near-degenerate. The flag must be a Python `bool`.
4. The two runs agree on the margin: each run's enclosure of the margin overlaps the other's. An honest enclosure may narrow as the precision grows, but it still contains the margin.
5. No encoding call, and no call to the factory, leaves the working precision of mpmath or python-flint changed. An encoding that pinned its own precision would make the repeat run at the same precision as the first run. The checker cannot see an encoding that sets its own precision and restores it before it returns.

Each run gets its own deep copy of the certificate's point, so an encoding that writes into the point's values cannot pass anything to the repeat, to the other encoding, or to the caller. The checker requires the exact types `flint.arb`, `Fraction`, and `tuple` for what an encoding returns, because a subclass could override the comparisons the checker makes.

A claimed duality bound is rejected only if it lies certainly further than `scale * 10^-(digits - 5)` from its recomputed ball, where `scale` is the size of the claimed bounds and at least 1.

The sign test is rigorous for every kind if the encoding is honest: an exact margin, an arb ball that contains the margin, or two arb balls that bound it from above and below. Each run records an arb enclosure of the margin, and rules 3 and 4 use it.

`check_kill(make_first, make_second, cert)` accepts a kill only if each of two factories' encodings accepts the certificate. The two encodings are the two model families' encodings of one conjecture (design doc-gmj, "Encoding"). `check_certificate` alone is not enough for a kill. `check_kill` refuses one factory passed twice, but it cannot tell which family wrote an encoding.

Limits: a point or a certificate may ask for at most 5,000 digits, so the repeat runs at most at 10,000. A value written as text must be an ASCII decimal (such as `-1.5e3`, with underscores allowed between digits) or a ratio of two ASCII integers (such as `3/4`). It may have at most 4,000 characters, because Python refuses to read an integer string of more than 4,300 digits. Its exponent may be at most 10,000. Python's own readers also accept other Unicode digits, hexadecimal, `inf`, and `nan`; minilab refuses them. `check_certificate` turns any error from encoding code, `SystemExit` included, into a rejection. `KeyboardInterrupt` still stops it.

The checker runs the encodings in its own process. Until Task 7's sandbox runs each encoding run in its own subprocess, encoding code can change the other encoding or the checker itself, keep state on a class or in a module, or loop and hang the check.

## Context

The design (doc-gmj, "Falsifier") says only a certificate from library code can make a kill. The pilot showed three ways a search reports a false kill: a solver's warm-start state, a float grid, and a point outside the domain. The fixtures in `tests/fixtures/pilot/` rebuild each case, and the checker rejects each one.

An adversarial review of the first version (commit f9309f8) found these holes:

- An exactly zero margin, `sqrt(30)^2 - 30`, reads as a tiny negative number at any precision. The checker accepted it.
- Cancellation can make a positive margin read as negative far from zero: `(10^40 + 1) - 10^40 - 0.5` reads as -0.5 at 30 digits. The repeat check ran only near zero, so it never caught this.
- A near-degeneracy flag of `np.True_` skipped the repeat check.
- Very large precisions or exponents hung the checker or raised out of it.
- `margin_ball` and `primal_dual` are written by a model. A `margin_ball` that wrapped a float value in an arb ball made the false grid kill pass.

Rules 2 to 4 and the limits closed the first four. `check_kill` answers the last one: the user chose to require both encodings (bead minilab-814b03, 24 November 2026).

A second adversarial review (commit 4e2038f) found more:

- Duality bounds were mpf values, so cancellation larger than both precisions passed: `(10^100 + 1) - 10^100 - 0.5` read as -0.5 at 30 and at 60 digits. The user chose arb balls for duality bounds, so the ball now contains zero and the check rejects it.
- An encoding that set `mpmath.mp.dps` itself made both runs use the same precision. Rule 5 closes this.
- The checker used one encoding object for both runs, so an encoding that cached its results by point carried the first run's numbers into the repeat. The user chose encoding factories, which close this.
- An exponent written in non-ASCII digits, such as Arabic-Indic digits, got past the exponent limit, and reading it hung. The ASCII rule closes this.
- An honest ball that narrowed at the higher precision was rejected. The overlap rule closes this.
- `check_kill` checked only that the two encodings were different objects, so two instances of one class passed. The user chose to leave this to Task 7: `encode_twice` must give one encoding for each model family.

A third adversarial review (commit 3927b86) found more:

- In one process, a dishonest first encoding patched the second encoding's class, or replaced `check_certificate`, and `check_kill` accepted the false grid kill. The user chose to close this in Task 7's sandbox: each encoding run goes in its own subprocess with a timeout, and the parent rebuilds the returned values and makes every comparison itself (bead minilab-1333bc).
- Every run shared one copy of the point's values, so an encoding that cached into the point beat the factory. Each run now gets a deep copy.
- An exception whose text raised, and `SystemExit`, escaped the checker. They are now rejections.
- Subclasses of `flint.arb`, `Fraction`, and `tuple` passed the type checks, and their overridden comparisons ran the checker's own tests. The checker now requires the exact types.
- A factory could leave the precision changed for the caller. The factory now runs inside the precision context and is rejected if it changes it.
- The gap rule compared against the claimed violation, which the tolerance let a claim inflate past both bounds. It now compares against the recomputed upper bound.
- `assumptions()` returning an empty list passes condition 1. The checker needs the conjecture's number of assumptions, which Task 7's specification gives.

## Options considered

For model-written capabilities:

1. **Require both encodings to accept (chosen).** One family's faulty bound code cannot make a kill alone. It costs one more check for each kill.
2. **Check `margin_ball` and `primal_dual` against `margin` at twice the precision.** This catches a ball that misses the margin, but `margin` is also model code.
3. **Library code evaluates the assumptions exactly.** This closes a gap in the predicates only, not in the bounds.

For the repeat check:

1. **Always repeat, and require agreement (chosen).** Each check costs about twice as much.
2. **Repeat only near zero or near degeneracy.** This missed cancellation away from zero.

For duality bounds:

1. **Two arb balls (chosen).** The sign test is rigorous, like the interval one. Encodings must compute their bounds in ball arithmetic.
2. **mpf values with a noise floor and a repeat.** This is a heuristic and missed cancellation larger than both precisions.

For encoding state:

1. **A factory, called once for each run (chosen).** State on the object cannot carry over. State on the class or in a module still can.
2. **One encoding object for both runs.** A cache keyed on the point hides drift between the runs.

## Consequences

- Task 7's plan has `check_certificate` take one encoding. It now takes a factory, and the falsifier must call `check_kill` with one factory for each of `encode_twice`'s families for any kill.
- `encode_twice` must give one encoding for each model family. `check_kill` refuses only one factory passed twice.
- Encodings that can be killed need `exact_margin`, `margin_ball`, or `primal_dual`. A `primal_dual` must return arb balls. Task 7's encoding prompt must ask for at least one capability, written in exact or ball arithmetic.
- Both encodings must support the certificate's kind, or the kill fails.
- The two encodings could share a mistake if the two families make the same error.
- Checks set the global precision of mpmath and python-flint, so they must not run in parallel threads of one process.
