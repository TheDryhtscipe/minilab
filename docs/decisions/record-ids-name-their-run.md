---
id: doc-b9w
title: Record IDs name their run
type: decision
scope: internal
created_at: '2026-11-24T14:53:18.426918Z'
updated_at: '2026-11-24T14:53:18.426918Z'
---

# Record IDs name their run

## Decision

A record ID has the form `<run_id>.<local>`, for example `run-1.r7`. The record store refuses a record whose ID does not start with its own run ID and a dot. Run IDs and local names use only `a-z`, `0-9`, `_`, and `-`, and are at most 64 characters each. Neither may contain the `.` separator.

The store works out a record's file path from its ID alone: `runs/<run_id>/records/<run_id>.<local>.json`. `load` and `write_stage` no longer search every run.

## Context

The design (doc-gmj, "Records and stages") says each stage takes record IDs. The build plan (doc-5wh, Task 2) gives `RecordStore.load(record_id)` with no run ID. So a record ID must be unique across all runs.

The first version of the store let callers choose any ID. `create` checked every run for the ID, then wrote the file. That check was not atomic across runs. Two runs could create the same ID at the same moment, and both would succeed. After that, `load` and `write_stage` failed for that ID.

Records are committed to the repo after each run. Runs in separate git worktrees can also make the same ID. The clash then appears only when the branches merge.

## Options considered

1. **Put the run ID in the record ID (chosen).** Two runs cannot make the same ID, on one machine or on separate branches. Within one run, `create` adds the file with `os.link`, which fails if the file exists. So of two creates with one ID, exactly one succeeds. IDs are longer to type, for example in `minilab review`.
2. **A lock file around the check and the write.** This works only for processes on one machine and one file system. It does not cover clashes at merge time.
3. **Claim each ID in a shared directory with an exclusive link.** This has the same gap at merge time. A crash between the claim and the write leaves a stray claim.
4. **Random IDs made by the store.** Clashes become very unlikely, but IDs are hard to read in the review queue. IDs that callers choose would still race.

## Consequences

- Later tasks that make records, such as the generator, must build IDs as `<run_id>.<local>`. `minilab.records.run_of` returns the run ID that a record ID names.
- `Provenance.parents` holds full record IDs, so a parent may be in another run.
- A copy of a record file in another run's directory does not affect `load` or `write_stage`. `list` for that run reports it as corrupt.
- Two processes that write stages to the same record at once can still lose one write. The pipeline runs each record's stages one at a time.
