---
id: doc-gmj
title: 'Minilab design: a filter-first search loop for cross-domain ML mathematics'
type: decision
scope: internal
created_at: '2026-11-24T01:06:00.969784Z'
updated_at: '2026-11-24T01:19:53.972944Z'
---

# Minilab design: a filter-first search loop for cross-domain ML mathematics

Status: draft for approval, 24 November 2026.

## Intent

Minilab is a small, standalone search loop for exploratory mathematics that bears on machine learning. It picks a pair of domains, has language models propose conjectures that cross them, and puts every conjecture through a chain of filters. The user reviews only what reaches the human queue.

The principle comes from the user's harness diagram: **the filters decide the value, not the generator**. Generators are cheap and unreliable. Every verdict that matters (killed, known, proved, trivial) is made by code or by Lean, and comes with evidence that can be checked again. One step needs a model: turning an informal conjecture into executable code and into Lean. The design guards that step with independent encodings, a round-trip check, and a hypothesis audit (see "Encoding" and stage 5).

If the loop proves itself, a later project may let other ML projects start "hunter" sessions that use it to look for ways around a stuck problem. This design does not build that, but it keeps the door open (see "Records and stages").

## Why now

A pilot run (information theory × convex optimisation, 23 November 2026; files in `~/dev/sci/experiments/crossconj-ml`; MuninnDB vault `sci`, memory `01M85D7A6NM38RQ9EV6JAFK16A`) showed that the loop can produce real work. It also showed where an unfiltered loop goes wrong. Each lesson below is a requirement in this design.

| Pilot lesson | Where the design answers it |
|---|---|
| The falsifier gave three wrong verdicts from numerical artefacts: a warm-started Blahut–Arimoto solver, grid gaps next to uncertified points, and points outside the claimed domain. | Certified kills; domain check; cold-start rule (Falsifier) |
| Random sampling missed the counterexample; searching over joint laws found it. | Reformulate, then attack (Falsifier) |
| The most interesting result was a kill with a mechanism (R(D) is linear when the optimal output law is not unique). | Mechanism-bearing kills enter the human queue |
| Both Lean proofs were real but trivial, and one smuggled a KKT fact in as a hypothesis. | Triviality screen before the prover budget; hypothesis audit |
| The literature check needed local `pdftotext`; Loogle and `exact?` only see mathlib. | Known-result filter reads arXiv and MathOverflow, converts PDFs locally |
| Retrodiction can be contaminated by literature search and memory. | Date cap, blinded memory, isolated model calls (Benchmark) |
| One run and its follow-ups filled about one session context. | Cost ledger: tokens and hours per survivor from day one |

## Goals

1. Produce conjectures that cross two of the ten domains, and give each one a verdict backed by evidence that can be checked again.
2. Put in front of the user only conjectures that survived the filters, and kills that come with a mechanism.
3. Measure, before any scale-up, whether structured generation recovers more hidden post-cutoff results than plain prompting (the retrodiction benchmark).
4. Track hours and tokens per human-reviewed survivor from the first run.

## Non-goals

- Hunter sessions for other projects. The record format allows them later; nothing is built for them now.
- A web interface. The human queue is a command-line view.
- A Sage install. Exact rationals and interval arithmetic cover the pilot's needs. Sage is an add-on if a domain needs it later.
- More than three model families.
- Scale-up of any kind before the benchmark decision.

## Anti-goals

The lab has failed, even if it runs, if any of these happen:

- It records a kill without a certificate, or a "survived" verdict without a coverage report.
- A model chooses which domain pair to work on.
- A benchmark run can see a hidden result through literature search, memory, a model's web access, or the sealed set.
- A run's cost is not known afterwards.
- A model from the family that wrote a Lean statement also checks that statement.
- The build touches `~/dev/sci/experiments/crossconj`.

## Constraints

- **Lane:** focused (the default). Only the user sets lanes.
- **Location:** a git repo at `~/dev/sci/minilab`, with beads (prefix `minilab`) and DOCA. Memory goes in a new MuninnDB vault, `minilab`.
- **Python:** managed with `uv`, fully typed, no `Any`, checked with a strict type checker.
- **Lean:** a Lean project inside the repo, on the same toolchain as `crossconj` (`leanprover/lean4:v4.37.0-rc1`) and its mathlib version, so the built mathlib cache is reused.
- **Model access:** subscriptions only. Claude through `claude -p`; Codex (GPT-6 Sol, Plus plan) through `codex exec`; Gemini (4 Neon, Pro plan) through Antigravity CLI, which has to be installed. Codex Plus has the tightest limits.
- **Isolation:** every model call runs with no file tools, no web tools, and an empty working directory. The recipe for `claude -p` is in MuninnDB `common` (memory `01M3WPMQETWRFJQPDGFY0ZVBJG`): no `--bare`, `--tools ""`, `--setting-sources ""`, `--strict-mcp-config`, an empty cwd, and `ANTHROPIC_*` variables stripped.

## Domains

The ten domains, agreed with the user:

1. Convex and nonconvex optimisation
2. Information theory
3. Probability and concentration
4. High-dimensional statistics
5. Random matrices and linear algebra
6. Optimal transport
7. Kernels and approximation theory
8. Gradient flows and dynamical systems
9. Information geometry
10. Graph theory and spectral methods

This gives 45 unordered pairs.

## Approaches considered

| Approach | Trade-off |
|---|---|
| A. One Python program runs every stage, with fixed prompts. | Most predictable and easiest to replay. Judgement steps such as reformulation become fixed prompts with no one steering. |
| B. Claude Code skills and workflow scripts, one skill per stage. | Quick to build, and close to the later hunter idea. Hard to replay, costs are hard to attribute, and the filters become prompts that can talk themselves into a verdict. |
| **C. Split by role (chosen).** Verdicts are made by plain Python and Lean, from encodings that two model families wrote independently. Judgement steps go through one adapter over the three model tools. A thin skill lets a Claude session drive a run. | Replayable and auditable where it matters, flexible where judgement is needed. More parts to build than B. |

The user chose C because it keeps the filters as code, which is the point of the design, and still lets a Claude session steer a run.

## Records and stages

Every conjecture is a **record**: a JSON file that each stage reads and extends. Each stage is a command that takes record IDs and writes new fields. A stage never edits another stage's fields. This lets a later hunter session enter at any stage with its own problem.

A record holds at least:

- the run ID, the domain pair, and the generation mode (plain or structured);
- the informal statement, the objects it uses, and an explicit list of assumptions, including the domain of validity;
- provenance: model family, model name, prompt hash, and parent records (for reformulations and refinements);
- a finite family of test instances (see stage 2);
- a status: `proposed`, `untestable`, `ambiguous`, `killed`, `known`, `trivial`, `proved`, `survived`, or `abandoned`;
- for each verdict, its evidence: a certificate file, a coverage report, literature notes, or a Lean file;
- for kills, an optional mechanism with its own check;
- costs: tokens by family, wall-clock time, and the user's review minutes.

The canonical store is the repo (`runs/<run-id>/records/`), committed after each run. When a record reaches a final status, a summary goes to the `minilab` MuninnDB vault with its provenance and a pointer to the files.

The pipeline runs the stages in this order. It differs from the user's diagram in one place: a cheap triviality screen runs before the prover budget, as the pilot showed it should.

1. Pair scheduler
2. Generator
3. Falsifier (including reformulate, then attack)
4. Known-result filter
5. Formaliser, linter, round-trip check, and hypothesis audit
6. Triviality screen
7. Prover loop
8. Scorer
9. Human queue

### 1. Pair scheduler

The visit counts for the 45 pairs are kept in a state file in the repo, outside any model. The scheduler picks a pair with weight that falls as a pair's visit count grows. It also takes an explicit pair list, which the benchmark uses. A model never picks or changes the pair.

### 2. Generator

There are two modes:

- **Plain:** one prompt asking for conjectures that link domain A and domain B.
- **Structured:** three steps. *Elicit* the central objects, operations, and inequalities of each domain. *Abstract* them to their structure. *Cross* that structure into the other domain and state conjectures.

Both modes run across the three families: Claude, Codex, and Gemini. Gemini only generates, because the user has seen a high hallucination rate from Neon. The filters catch what it gets wrong. Each conjecture must come with an explicit assumption list and a stated domain of validity, because the falsifier checks counterexamples against them.

Each conjecture must also come with a **finite family of test instances**: a finite-dimensional, computable version of the claim that the falsifier can search. For an asymptotic or infinite-dimensional claim this might be a discretisation, a fixed dimension, or a parametric family. A conjecture without one gets the status `untestable`. Untestable conjectures are kept on a separate list and never enter the human queue as survivors. This keeps untested claims out of the queue and out of the benchmark. (Agreed with the user.)

### Encoding

Before the falsifier runs, a conjecture needs executable form: a margin function (negative means violated), predicates for each assumption and for the domain of validity, and a parametrisation of the instance family. Only a model can write this, so it is the place where a model sits inside every verdict. Wrong code would give a certified kill of the wrong claim.

The guard:

1. Two different model families each write the encoding independently, from the record alone. Neither sees the other's.
2. A script evaluates both encodings on the same random instances: margins and predicates must agree within tolerance.
3. Where a Lean statement exists (stage 5), the falsifier's witness is also checked against it.
4. If the encodings disagree, the conjecture gets the status `ambiguous` and goes back for its statement to be made precise, at most a fixed number of times. It never goes on to a verdict.

Encodings use a small, fixed library API for exact arithmetic, intervals, and certificates. The certificate checkers are part of that library. Models never write them.

### 3. Falsifier

The falsifier reads a record and tries to kill it. It works in Python, with `mpmath`, `python-flint` (interval arithmetic), exact rationals, and `numpy`/`scipy` for search.

**Search may suggest a kill. Only a certificate makes one.** A kill is recorded only when a library checker, separate from the search code and never written by a model, confirms one of these on the agreed encoding:

- the violation holds in exact rational arithmetic;
- an interval enclosure shows the violation;
- for an optimisation-defined quantity, matching primal and dual bounds at high precision, with the gap far smaller than the violation.

The checker also confirms in exact arithmetic that the counterexample satisfies every assumption in the record, including the domain of validity. An iterative solver used for a verdict must start cold. A warm start may speed up search but never a certificate. When a margin is tiny or the point is near a degenerate case (for example near independence), the checker repeats the check at 100 or more digits. The pilot's huge spurious ratios came from cancellation at float precision.

**Survival needs effort and a coverage report.** A conjecture survives only after an adversarial optimiser has tried to drive its margin negative: several starts, a local optimiser on the violation margin, and sampling of the boundary. The coverage report states the parameter space and dimensions searched, the number of starts, the worst margin found and where, and the regions that were not covered. "Survived" means "no counterexample found under this coverage". It never means "true".

**Reformulate, then attack.** If the direct search fails, a model proposes up to a fixed number of reformulations: a different parametrisation of the same claim, such as searching over joint laws in place of (source, distortion, slope). Before the falsifier uses one, a check confirms on random instances that the maps there and back agree with the original claim. A kill found in a reformulation is mapped back to the original variables and certified there.

**Mechanisms.** For a certified kill, a checker model (not the generator's family) may propose a mechanism: a structural reason for the failure. The falsifier tests the mechanism's prediction on a small family of new instances. A kill with a tested mechanism goes to the human queue.

### 4. Known-result filter

The filter looks for the result in this order:

1. Loogle and `exact?` against mathlib.
2. The arXiv API, with a date cap on submission date.
3. MathOverflow through the Stack Exchange API, with a date cap.

PDFs are downloaded to a cache in the repo and converted locally with `pdftotext`. Web fetches returned raw PDF bytes in the pilot. A checker model reads the top candidates and writes literature notes: the paper, the statement it matches, and how close the match is. A close match sets the status to `known`. Every query and result is logged on the record. The filter never uses a general web search engine, because its results cannot be date-capped.

### 5. Formaliser, linter, round-trip check, and hypothesis audit

A model writes the conjecture in Lean, with its concepts as definitions. Then:

- **Vacuity linter:** checks that the hypotheses can be satisfied, by building the falsifier's witness instance in Lean where possible. It rejects hypotheses that contradict each other or are unused.
- **Round-trip check:** a model from a different family from the formaliser translates the Lean statement back to English without seeing the original. The record keeps both texts. Neon never does this check.
- **Hypothesis audit:** a script and the checker model list every assumption in the Lean statement that is not in the record's assumption list. Each extra assumption either goes back into the record as a stated weakening, or the formalisation is rejected. This catches the pilot's smuggled KKT hypothesis.

### 6. Triviality screen

This screen runs before any prover budget is spent. It tries, each with a short time limit, `exact?`, `aesop`, `simp`, `positivity`, and `nlinarith`. (`polyrith` is left out: it calls an external service.) It also has the local weak model (`qwen3.5:4b` through ollama) attempt a Lean proof, which Lean checks. A checker model asks whether the statement is a few-line corollary of a named theorem, and the screen tests that by trying the corollary in Lean. A result proved here gets the status `trivial`.

### 7. Prover loop

A model writes Lean proofs with compiler feedback, under a budget per conjecture: a cap on attempts and tokens. A proof that needs an added hypothesis goes back through the hypothesis audit. A proved result gets the status `proved`, with its Lean file and the list of axioms it uses.

### 8. Scorer

The scorer records triviality signals (which screens nearly succeeded, proof length, how many library lemmas the proof uses) and a compression measure (how much the statement says compared with what it costs to prove). These rank the queue. They do not remove anything from it.

### 9. Human queue

A command shows each queued item with its statement, round-trip text, hypothesis audit, literature notes, evidence (certificate or coverage report), and, for kills, the mechanism. The user marks each item as interesting, not interesting, or known, adds a note, and the tool logs the review time. The queue holds:

- survivors with a coverage report;
- proved results that are not `trivial`;
- certified kills with a tested mechanism.

## Memory read policy

Each run declares how it may read the `minilab` vault:

- `blind`: no reads.
- `pair`: only records for the same domain pair.
- `all`: every record.

Each policy can carry a date cap, which uses MuninnDB's `before` filter. The policy is written into the run's manifest. The pilot vault `sci` is never read by a run.

## Retrodiction benchmark (first milestone)

The benchmark tests whether structured mode recovers more hidden results than plain prompting. Scale-up happens only if structured mode wins.

**The hidden set.** A separate curator session collects 20 to 30 cross-domain results whose first arXiv version is on or after **1 July 2026**. That is the day after the latest training cutoff among the three generator models: Claude June 2026, GPT-6 Sol 20 April 2026, Gemini 4 Neon March 2026. Each result touches two of the ten domains and can be stated in a few lines. The results come from no more than 8 domain pairs, to keep the cost in bounds. (Agreed with the user; scale back further if it costs too much.) The curator stores the set outside the repo in a sealed folder that no run reads. The curator tells the runs only which domain pairs the set covers, not the results.

**Blinding.** During benchmark runs:

- the literature date cap is 30 June 2026;
- the memory read policy is `blind`;
- every model call has no web or file tools. This must be verified for each of the three model tools before the benchmark starts. A tool that cannot be isolated is left out of both benchmark modes, but may still be used in normal runs;
- the plain and structured runs cannot see each other's records.

**The comparison.** Both modes run on the same domain pairs with the same token budget per pair. The budget covers every model call for that pair: generation, encoding, reformulation, formalising, checking, and proving. Each mode's human queue is graded against the hidden set.

**Grading.** A grader session sees the hidden set and both queues, without knowing which queue came from which mode. It proposes matches of two kinds:

- **strict:** the queue holds a statement equivalent to the hidden result, or one that implies it;
- **near-miss:** the right objects and the right direction, with a weaker or wrong statement.

The user confirms each proposed match.

**Decision rule** (agreed with the user):

1. Scale up if structured mode has at least 2 strict matches and more strict matches than plain mode.
2. If the strict counts tie (including 0 to 0), compare near-misses per queued item, so that a mode with a bigger queue does not win on volume. Scale up if structured mode is higher.
3. If that also ties, do not scale up, and record the result as inconclusive.

The result, with queue sizes and costs, goes into a DOCA decision record.

## Cost ledger

From the first run, each record and each run logs tokens by model family (from the tools' JSON output), wall-clock time per stage, and the user's review minutes. A report gives hours and tokens per human-reviewed survivor, and each family's share of the cost and of the survivors. This shows whether Gemini earns its place.

## Validation against the pilot

Before the benchmark, the filters must reproduce the pilot's lessons as tests:

- the three wrong pilot verdicts (warm-start artefact, grid gap next to an uncertified point, point outside the domain) are rejected as kills;
- the certified S7 counterexample is accepted;
- S8's KKT hypothesis is flagged by the hypothesis audit;
- S1 and S8 are flagged by the triviality screen.

The pilot has scripts and output logs, not records. The builder rebuilds these cases by hand as record fixtures from the pilot scripts (`s7_*.py`, `verify_kills.py`, and their `.out` files) and the MuninnDB memory `01M85D7A6NM38RQ9EV6JAFK16A`, and reads but never changes the pilot files. Checks done by code or Lean must pass every time. Checks that depend on a model must pass in most of several tries; the plan sets the numbers.

## Decided by the user

- A standalone search loop first; hunter use later, if it proves itself.
- The ten domains above.
- Claude, Codex, and Gemini as the three model families; installing Antigravity CLI is fine.
- Benchmark scoring: strict plus near-misses, deciding on strict.
- Approach C.
- Installing what the lab needs is fine.
- Conjectures without a finite family of test instances are `untestable` and never queued as survivors.
- The benchmark decision rule above.
- The hidden set comes from no more than 8 domain pairs, and each pair's budget covers every model call. Scale back if it costs too much.

## Calls made in the design (the user may change these)

- Gemini generates only and never checks.
- Codex's tight limits go first to short checking calls, then to generation.
- The local weak model is `qwen3.5:4b`.
- `python-flint` and exact rationals in place of Sage.
- Repo location `~/dev/sci/minilab` and vault `minilab`.
- The triviality screen moves ahead of the prover.
- Benchmark runs are told which domain pairs the hidden set covers, so both modes get the same hint.
- The curator, grader, and run sessions are separate sessions.
- Two families write each conjecture's encoding independently; disagreement gives `ambiguous`.
- A tool that cannot be isolated is left out of both benchmark modes but kept for normal runs. Costs a tool does not report are estimated and marked as estimates.
- Pilot validation cases are rebuilt by hand as fixtures.
- `polyrith` is left out of the triviality screen.

## Left to the builder

- Prompt wording for each judgement step.
- Exact field names in the record schema, within the list above.
- The choice of adversarial optimiser and the default number of starts.
- The Lean runner (for example the installed `lean-beam` tools, or `lake env lean` directly).
- The formulas for the triviality and compression scores.
- Default budgets per conjecture and per run.
