---
id: doc-wdh
title: The cost ledger is a per-run log charged by pair and stage
type: decision
scope: internal
created_at: '2026-11-24T16:53:02.062400Z'
updated_at: '2026-11-25T00:36:54.758380Z'
---

# The cost ledger is a per-run log charged by pair and stage

## Decision

Each run keeps its costs in one append-only file, `runs/<run_id>/ledger.jsonl`, written by `minilab.ledger.Ledger`. That file is the only record of costs. `Ledger.costs(record_id)` builds one record's `Costs` from it. `Record.costs` is not filled in.

`Ledger.charge` takes keyword arguments: `pair`, `stage`, `family`, `tokens_in`, `tokens_out`, `seconds`, `estimated`, and an optional `record_id`. A call with no record, such as a generation call or a failed call, is still charged to its pair. When a charge names a record, the record must belong to the ledger's run and to the same pair.

A ledger opens only for a run that has a manifest (`runs/<run_id>/manifest.json`). It refuses a pair or a model family that the manifest does not list.

`Ledger.log_review` accepts only records whose status can be in the review queue: `survived`, `proved`, or `killed`. So the cost report's "reviewed survivors" are the reviewed queue items.

The user made the first two rulings on 24 Nov 2026 (bead minilab-93cad4). The last two came from the adversarial review of that bead.

## Context

The build plan (doc-5wh, Task 3) gives `charge(record_id, family, tokens_in, tokens_out, seconds, estimated)`. The design (doc-gmj, "Cost ledger" and "Retrodiction benchmark") needs two things that signature cannot hold:

- wall-clock time per stage, so a charge must name its stage;
- a token budget per pair that covers every model call for the pair, including generation calls made before any record exists, so a charge must name its pair.

Records are frozen, and the record store only appends stage output. Writing costs into each record would mean rewriting the record file on every charge, also after a final status.

## Options considered

1. **One append-only log per run (chosen).** Each append is one write under a lock on the file, so several processes can charge at once. There is one source of truth. A reader of one record file does not see its costs there.
2. **The log plus a copy in each record.** A record file would hold its own costs. Every charge would rewrite the record, and the copy could drift from the log.

## Consequences

- The model adapter (Task 4) passes the pair and the stage to `charge`. It can charge a call before it knows the record.
- A caller reserves tokens before each model call. Inside one `with ledger.pair_lock(pair):` block it checks `pair_spend` and appends a reservation with `reserve`. It then leaves the block, makes the call, and settles the reservation. A charge with `reservation=` settles it for a call that ran. `release` settles it for a call that never started. So calls on one pair can run at the same time (minilab-294360).
- `pair_spend` counts every charge and every open reservation. A reservation that a crash left open keeps counting as spend. The cost report and `costs` count it as an estimated input charge with no duration.
- The model adapter reserves the prompt's token estimate plus `Budget.reserve_output_tokens` (32,000 by default). It refuses a call once the pair's spend is at or over the budget. So a pair can overrun its budget by the last admitted call's tokens. Each call in flight with it adds what it spent beyond its own reservation. The user ruled for this bound on 25 Nov 2026 (bead minilab-294360).
- A charge or release that settles a reservation is checked under the same exclusive lock as its append. So a reservation is settled at most once, even when several processes try at once. The reservation must still be open, and must be for the same pair, stage, family, and record.
- A call that fails after its tool started is charged the output the tool reported. If the tool reported none, as on a timeout, the call is charged its output allowance, so a failed call never frees the output it reserved. A call stopped for printing more than `MAX_OUTPUT_BYTES` (16 MiB) is charged the output it printed, at 4 bytes per token.
- The pair lock is an exclusive `flock` for each pair. It holds only on one machine. `reserve` and `charge` do not take it, so nothing stops a caller that skips it. Taking a pair's lock while you hold it blocks forever.
- Each append takes an exclusive `flock` on the ledger file, and each read takes a shared one. No reader or appender sees another process's half-written line.
- A ledger line that does not parse, names another run's record, or was cut short by a crash makes every read fail with `CorruptLedger`. Appends after a cut-short line are refused. Nothing is skipped, because a skipped line would hide spend.
- One charge may log at most 10^12 tokens and 10^7 seconds. One review may log at most 10^5 minutes.
- The cost report's machine hours are the summed call durations. Calls that ran in parallel each count in full.
- The manifest's pairs and families are tuples, so a loaded manifest cannot be changed in memory. A `blind` run may not set `memory_before`.
