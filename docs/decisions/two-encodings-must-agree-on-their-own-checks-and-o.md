---
id: doc-6ja
title: Two encodings must agree on their own checks and on each other, and an ambiguous
  record is sharpened as a child record
type: decision
scope: internal
created_at: '2026-11-25T02:00:15.904636Z'
updated_at: '2026-11-25T02:00:15.904636Z'
---

# Two encodings must agree on their own checks and on each other, and an ambiguous record is sharpened as a child record

Status: accepted, 25 November 2026. Bead minilab-e47de8. Code: `src/minilab/encoding/agree.py` and `src/minilab/encoding/prompts.py`. Design doc-gmj, "Encoding". Build plan doc-5wh, Task 7, second half.

## Context

An encoding is the code form of a conjecture: a margin, the assumption predicates, a domain predicate, and a sampler. A model writes it, so wrong code could lead to a certified kill of the wrong claim. The design guards this step: two model families each write an encoding from the record alone, and the two must agree on shared samples. If they disagree, the record becomes `ambiguous` and its statement is made more precise. After a fixed number of rounds the record is `abandoned`.

The record store allows one write per stage. A record cannot take the encoding stage twice, so rounds cannot happen on one record.

## Decision

### Who encodes

`encode_twice` asks two families. The first is `pick_checker(author)`; the second is `pick_checker` of the first. Gemini never encodes. Both families get the same prompt, built from the record only, so neither sees the other's code. The author's own family may write one of the two encodings: the design asks for two different families, not two families other than the author (user decision, 25 November 2026). Each encoding comes back as a `Sandboxed` that names the family that wrote it, because `check_kill` trusts that field.

The prompt names the parameters from the record's instance family, and both encoders must use those names (user decision, 25 November 2026). It asks for the margin as the right side minus the left side of the stated inequality, unscaled, so two margins can be compared. It asks for at least one of `exact_margin`, `margin_ball`, or `primal_dual`, in exact or ball arithmetic.

### What agreement means

`agree` runs both encodings in the sandbox on `n` shared samples. Each sampler draws half of them. First it checks each encoding on its own. An encoding is broken if any of these fails:

- it runs in the sandbox;
- its values have the right types;
- each of its certificate methods overlaps its own margin, widened by `rtol`;
- its sampler gives at least two different points;
- it gives one assumption result for each assumption in the record.

Then it compares the two encodings. They disagree if any of these differ:

- the parameter names in their samples;
- `in_domain` on any sample;
- the assumption results on any sample;
- the margin on a sample that both put in the domain. Margins agree only if they have the same sign and are within `rtol` of each other, relative to the larger. There is no absolute tolerance, so two tiny margins of opposite sign disagree.

Fewer than a quarter of the samples in the domain is also a disagreement. Margins are rebuilt and compared at the sample's own precision.

Margins and certificate methods run only on points that both encodings put in the domain. Calls run in batches of 60, each in a fresh sandbox with its own time limit.

### What happens after a disagreement

1. If one encoding is broken, that family is asked once more. The retry prompt shows only its own failure, never the other encoding (user decision, 25 November 2026).
2. If the encodings still disagree and the record has two ambiguous ancestors, the record is `abandoned` with no further model call.
3. Otherwise the generator family is asked for a sharper statement, at most twice. If neither reply can be read, the record is `abandoned` (user decision, 25 November 2026).
4. If a reply can be read, the encoding stage is written with status `ambiguous` and the sharper statement in its data. `make_child` then stores it as a new record whose provenance names the ambiguous record as its parent (user decision, 25 November 2026).

The round of a record is the number of ambiguous ancestors it has. The child's ID comes from the parent's ID alone. `make_child` makes no model call, so it is safe to run again after a crash. If another record already holds the child's ID, `make_child` refuses it.

If the sandbox cannot run an empty encoding, `agree` raises `SandboxUnavailable`. The stage then writes nothing, and the record stays `proposed`. A model failure has the same effect.

## Consequences

- An ambiguous record stays `ambiguous`, because that status is not final. The pipeline driver (Task 16) must treat an ambiguous record with a child as finished.
- Each encoding is saved under the run at `encodings/<record_id>.<family>.py` and named in the stage's evidence. The falsifier rebuilds the `Sandboxed` pair from these files and the stage's `families` field.
- If a model call fails partway through the stage, the stage runs again from the start and pays for the encodings again.
- Parameter names are compared between the two encodings, not with the instance family, because the instance family is free text.
- `near_degenerate` is not compared.
- The review found that the sandbox wire dropped the sign of every mpf. Commit 83ae75c fixes it, and PR 10 (minilab-1333bc) needs that fix.
