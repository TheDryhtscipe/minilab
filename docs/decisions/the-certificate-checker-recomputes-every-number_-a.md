---
id: doc-j6t
title: The certificate checker recomputes every number, and a kill needs both encodings
type: decision
scope: internal
created_at: '2026-11-24T22:46:10.037625Z'
updated_at: '2026-11-24T22:59:41.416311Z'
---

# The certificate checker recomputes every number, and a kill needs both encodings

## Decision

The certificate checker (`minilab.certify`) trusts no number in a certificate. It recomputes the margin and the predicates through the encoding, in a fresh call. A certificate says where to look and what to expect; the encoding gives the values.

Each certificate kind needs one optional capability beyond the `Encoding` protocol in the build plan (doc-5wh, Task 6):

| Certificate | Capability | Returns |
|---|---|---|
| `ExactCertificate` | `ExactMargin.exact_margin` | the margin as a `Fraction` |
| `IntervalCertificate` | `BallMargin.margin_ball` | an arb ball (python-flint) that contains the margin |
| `DualityCertificate` | `DualBounds.primal_dual` | a primal and a dual bound, with the margin between them |

An encoding without the capability cannot have that kind of certificate checked. The protocols live in `minilab.encoding.api`, which Task 7 completes.

`check_certificate(enc, cert)` accepts only if all of these hold:

1. The point satisfies every assumption predicate and the domain predicate. Each predicate must return a Python `bool`.
2. The margin is negative. For a duality certificate, the upper bound must lie below `-scale * 10^-(digits - 10)`, where `scale` is the size of the bounds. The gap between the bounds must be below 1/1000 of the violation. Duality certificates need at least 30 digits.
3. Steps 1 and 2 hold again at twice the precision. The repeat runs at 100 digits or more if the margin is within 1e-6 of zero, or the encoding flags the point as near-degenerate. The flag must be a Python `bool`.
4. The two runs agree on the margin. For an interval certificate, the two balls overlap: an honest ball may narrow as the precision grows. For the other kinds, the upper bounds agree to within 1/1000.
5. No encoding call changes the working precision of mpmath or python-flint. An encoding that pinned its own precision would make the repeat run at the same precision as the first run.

For interval and exact certificates the sign is rigorous if the encoding's ball or exact margin is honest. Duality bounds are mpf values, not enclosures, so for them the repeat is a heuristic. It catches cancellation that changes the result between the two precisions. It does not catch cancellation in terms larger than both precisions can hold: `(10^100 + 1) - 10^100 - 0.5` reads as -0.5 at 30 and at 60 digits.

The checker calls the encoding object it is given for both runs. State that object keeps, such as a cache or a solver's warm start, carries from the first run to the repeat.

`check_kill(first, second, cert)` accepts a kill only if two distinct encodings each accept the certificate. The two encodings are the two model families' encodings of one conjecture (design doc-gmj, "Encoding"). `check_certificate` alone is not enough for a kill.

Limits: a point or a certificate may ask for at most 5,000 digits, so the repeat runs at most at 10,000. A value written as text must be an ASCII decimal (such as `-1.5e3`, with underscores allowed between digits) or a ratio of two ASCII integers (such as `3/4`). It may have at most 4,000 characters, because Python refuses to read an integer string of more than 4,300 digits. Its exponent may be at most 10,000. Python's own readers also accept other Unicode digits, hexadecimal, `inf`, and `nan`; minilab refuses them. `check_certificate` never raises: it turns any error into a rejection.

## Context

The design (doc-gmj, "Falsifier") says only a certificate from library code can make a kill. The pilot showed three ways a search reports a false kill: a solver's warm-start state, a float grid, and a point outside the domain. The fixtures in `tests/fixtures/pilot/` rebuild each case, and the checker rejects each one.

An adversarial review of the first version (commit f9309f8) found these holes:

- An exactly zero margin, `sqrt(30)^2 - 30`, reads as a tiny negative number at any precision. The checker accepted it.
- Cancellation can make a positive margin read as negative far from zero: `(10^40 + 1) - 10^40 - 0.5` reads as -0.5 at 30 digits. The repeat check ran only near zero, so it never caught this.
- A near-degeneracy flag of `np.True_` skipped the repeat check.
- Very large precisions or exponents hung the checker or raised out of it.
- `margin_ball` and `primal_dual` are written by a model. A `margin_ball` that wrapped a float value in an arb ball made the false grid kill pass.

Rules 2 to 4 and the limits close the first four. `check_kill` answers the last one: the user chose to require both encodings (bead minilab-814b03, 24 November 2026).

A second adversarial review (commit 4e2038f) found more:

- Cancellation larger than both precisions still passes a duality certificate. This is still open.
- An encoding that set `mpmath.mp.dps` itself made both runs use the same precision. Rule 5 closes this.
- An encoding that caches its results by point carries the first run's numbers into the repeat. This is still open.
- An exponent written in non-ASCII digits, such as Arabic-Indic digits, got past the exponent limit, and reading it hung. The ASCII rule closes this.
- An honest ball that narrowed at the higher precision was rejected. The overlap rule for interval certificates closes this.
- `check_kill` checks only that the two encodings are different objects. Two instances of one class pass.

## Options considered

For model-written capabilities:

1. **Require both encodings to accept (chosen).** One family's faulty bound code cannot make a kill alone. It costs one more check for each kill.
2. **Check `margin_ball` and `primal_dual` against `margin` at twice the precision.** This catches a ball that misses the margin, but `margin` is also model code.
3. **Library code evaluates the assumptions exactly.** This closes a gap in the predicates only, not in the bounds.

For the repeat check:

1. **Always repeat, and require agreement (chosen).** Each check costs about twice as much.
2. **Repeat only near zero or near degeneracy.** This missed cancellation away from zero.

## Consequences

- Task 7's plan has `check_certificate` take one encoding. The falsifier must call `check_kill` with both of `encode_twice`'s encodings for any kill.
- Encodings that can be killed need `exact_margin`, `margin_ball`, or `primal_dual`. Task 7's encoding prompt must ask for at least one.
- Both encodings must support the certificate's kind, or the kill fails.
- The two encodings could share a mistake if the two families make the same error. `check_kill` checks only that they are different objects.
- Checks set the global precision of mpmath and python-flint, so they must not run in parallel threads of one process.
