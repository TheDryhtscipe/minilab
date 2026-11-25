---
id: doc-emx
title: Encodings run in a bwrap sandbox, and kills come only from sandboxed encodings
type: decision
scope: internal
created_at: '2026-11-25T00:31:21.753404Z'
updated_at: '2026-11-25T00:54:03.911065Z'
---

# Encodings run in a bwrap sandbox, and kills come only from sandboxed encodings

Bead: minilab-1333bc (build plan doc-5wh, Task 7, first half). Builds on decision doc-j6t (the certificate checker).

## Context

A model writes each conjecture's encoding as a Python module. The third review of the certificate checker (minilab-814b03) showed that one process is not enough. A dishonest encoding patched the other encoding's class, or `check_certificate` itself, and `check_kill` accepted the false grid_gap kill. The same review found three more gaps: a looping encoding hangs the checker, state on a class or module carries between runs, and `assumptions()` returning `[]` passes the assumption check.

The user decided (24 Nov 2026) that each run the checker makes runs in its own fresh subprocess, with a timeout. The child sends back plain values. The parent rebuilds them and makes every comparison itself.

## Decision

1. **The encoding contract.** An encoding is a module that binds `ENCODING` to its class. It may import only `minilab.encoding.api`, `mpmath`, `flint`, `fractions`, `math`, `numpy`, and `random`. The plan's list did not name `random`; the sampler's signature needs it.
2. **A syntax check first.** `check_source` refuses other imports, relative imports, and the built-ins `__import__`, `eval`, `exec`, `compile`, `open`, `breakpoint`, and `input`. A determined module can get past it, so it only catches honest mistakes.
3. **The sandbox is the guard.** `run_encoding` runs the module in a fresh child process under bwrap:
   - `--unshare-all` (no network) and `--cap-drop ALL`;
   - read-only binds of `/usr`, the Python install, the venv, and minilab's `src` only, so the user's files and `/run` are hidden;
   - a private `/tmp` and `/dev/shm` of 64 MiB each, and the sandbox root remounted read-only;
   - a systemd user scope (a cgroup) around bwrap, with `MemoryMax` 4 GiB, `MemorySwapMax` 0, and `TasksMax` 64. The cgroup caps the whole process tree, tmpfs pages included. Per-process limits alone let forked children add up past them (review of 4737bff);
   - per-process limits on memory, output (64 MiB), and CPU time, and a time limit that kills the whole process group.

   `run_encoding` never raises: a long or unparsable source, a time-out, a crash, or unreadable output comes back as a failure. If the user's service manager cannot be reached, every run fails, so no verdict comes from an unguarded run.
4. **Exact values on the wire.** The child writes each result as a typed record (`minilab.encoding.wire`). An mpf travels as its mantissa and exponent, an arb as its midpoint and radius in the same form, and a Fraction as two integers, all in hexadecimal. The parent rebuilds fresh values of exact library types. A value of any other type, or a subclass, comes back as `Opaque`, which no check accepts.
5. **`Sandboxed` is an encoding factory.** Each encoding it builds is one run: one child process makes every call on the checker's point. Code in one run cannot reach the checker, the other encoding, or the next run.
6. **Kills need sandboxed encodings from two families.** `check_kill(first, second, cert, *, assumptions)` refuses anything but two `Sandboxed` encodings. Each `Sandboxed` carries the model family that wrote it. `check_kill` refuses two from one family, and any from Gemini, which only generates. The user chose this on 25 Nov 2026, after the review showed that comparing sources is beaten by adding one character. `check_kill` trusts the family it is given, so `encode_twice` (minilab-e47de8) must fill it in truthfully.
7. **The assumption count comes from the record.** `check_certificate` and `check_kill` take `assumptions`, the number of assumptions the record states. An encoding must give exactly that many predicates.

## Alternatives considered

- **`unshare -rn` instead of bwrap.** It works on this machine, but it gives only a network block. bwrap also hides files and blocks writes. A network block alone also leaves Unix sockets under `/run` (such as Docker's) reachable.
- **Bind `/` read-only and hide parts.** This shows every file not hidden by name. An allowlist of paths shows only what the child needs.
- **Restructure the checker to read a record of values.** The checker keeps its `Encoding` interface. The sandbox supplies a proxy encoding instead, so every guard from doc-j6t still applies unchanged.

## Consequences

- A certificate check costs three child processes per encoding: a probe for the encoding's capabilities, then one for each run. A kill check costs six, a few seconds in all.
- `check_certificate` still takes in-process factories, for tests and the search. Such an encoding can reach anything in that process, so it never makes a kill.
- An encoding's batch evaluates every method at the point, `margin` included. A broken `margin` therefore stops a kill even when the certificate does not need it.
- The second half of Task 7 (minilab-e47de8: `encode_twice`, `agree`, and the ambiguous rounds) builds on `run_encoding` and `Sandboxed`.
