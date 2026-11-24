---
id: doc-5wh
title: Minilab build plan
type: guide
scope: internal
created_at: '2026-11-24T01:22:50.443418Z'
updated_at: '2026-11-24T13:43:59.228010Z'
---

# Minilab build plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to work this plan task by task. Each task is one bead; work it with the `/bead` skill.

**Goal:** build the minilab search loop and filters described in doc-gmj, validate them against the pilot, and run the retrodiction benchmark that decides whether to scale up.

**Architecture:** a typed Python package (`minilab`) holds records, the scheduler, the certificate library, the filters, the Lean runner, the cost ledger, and the pipeline driver. Verdicts come from code and Lean. Judgement steps go through one model adapter over `claude -p`, `codex exec`, Antigravity CLI, and ollama. A user-level skill lets a Claude session drive a run.

**Tech stack:** Python 3.13 with `uv`; pydantic v2 for records; `mpmath`, `python-flint`, `fractions`, `numpy`, `scipy`; `pytest`; `pyright` (strict); `ruff`. Lean 4 `leanprover/lean4:v4.37.0-rc1` with mathlib at `v4.37.0-rc1` (rev `31ce06eaadf4a200c0e1407c5f10953cf606b8f8`). `pdftotext` (poppler). MuninnDB vault `minilab`.

**Spec:** doc-gmj, "Minilab design: a filter-first search loop for cross-domain ML mathematics". Read it before any task. Where this plan and the spec disagree, the spec wins; flag the conflict on the bead.

## Global constraints

- Repo: `/home/thedryhtscipe/dev/sci/minilab`. Beads prefix `minilab`. Lane: focused.
- No `Any` types. `uv run pyright` in strict mode must pass on `src/` and `tests/`.
- Never write markdown files except README.md, AGENTS.md, CLAUDE.md. Documentation goes in DOCA.
- Never read or change `~/dev/sci/experiments/crossconj`. Pilot files in `~/dev/sci/experiments/crossconj-ml` are read only.
- Every model call runs isolated: empty temporary working directory, no file tools, no web tools, no MCP. For `claude -p`: no `--bare`, `--tools ""`, `--setting-sources ""`, `--strict-mcp-config`, `--no-session-persistence`, `ANTHROPIC_*` stripped from the environment (MuninnDB `common` memory `01M3WPMQETWRFJQPDGFY0ZVBJG`).
- Model roles: Gemini generates only. A checker is never from the family that wrote the thing it checks. Codex budget goes to checking calls before generation.
- A kill needs a certificate from library code. A survival needs a coverage report.
- Benchmark constants: hidden results first posted on arXiv on or after `2026-07-01`; literature date cap `2026-06-30`; at most 8 domain pairs; memory policy `blind`.
- Tests never call a live model or a live web API by default. Live checks are marked `@pytest.mark.live` and run only when asked.
- Commit messages state the change and may name the bead. No AI co-author lines.
- Remote: the public repo `TheDryhtscipe/minilab` on GitHub (use the TheDryhtscipe account). Push before closing a bead. Because the repo is public, nothing from the sealed hidden set (Task 18) may ever be committed.

## Review focus

These failures are implied by the spec but no single feature test covers them. Each has a test in the task named.

1. **Model-written encoding code misbehaves** (infinite loop, file writes, network). Expect a timeout or a refusal and the status `ambiguous`, never a verdict. Test: `test_encoding_sandbox_timeout_and_no_network` (Task 7).
2. **A model tool hits its usage limit or fails mid-run** (Codex Plus is likeliest). Expect the record to stay at its current stage and the run to be resumable; never a verdict by default. Test: `test_adapter_failure_leaves_record_unchanged` (Task 4) and `test_run_resume_after_failure` (Task 16).
3. **arXiv or Stack Exchange is down or rate-limited.** Expect literature status `unchecked`, never "not found". Test: `test_literature_outage_is_unchecked` (Task 10).
4. **Lean times out.** Expect `LeanResult.timed_out`, with no status change to `trivial`, `proved`, or `killed`. Test: `test_lean_timeout_is_not_a_verdict` (Task 11).
5. **An `ambiguous` conjecture loops.** Expect `abandoned` after the fixed number of rounds (default 2). Test: `test_ambiguous_rounds_end_in_abandoned` (Task 7).

## How to work each task

For every task: write the failing tests named in the task; run them and see them fail on their assertions; write the code; run them and see them pass; run `uv run pyright` and `uv run ruff check`; commit. Run only the tests for the area you changed (`orc test run` where it applies). Record the outcome on the bead.

---

## Phase 0: foundations

### Task 1: Repo scaffold and Lean project

**Files:** create `pyproject.toml`, `src/minilab/__init__.py`, `tests/conftest.py`, `.gitignore`, `CLAUDE.md`, `lean/lakefile.toml`, `lean/lean-toolchain`, `lean/Minilab.lean`, `lean/Minilab/Basic.lean`.

- `pyproject.toml`: package `minilab`, Python `>=3.13`, dependencies `pydantic>=2`, `mpmath`, `python-flint`, `numpy`, `scipy`, `httpx`, `typer`; dev `pytest`, `pyright`, `ruff`. Console script `minilab = "minilab.cli:app"`. Pyright `typeCheckingMode = "strict"`. Pytest marker `live`.
- `lean/`: require mathlib `v4.37.0-rc1`; toolchain `leanprover/lean4:v4.37.0-rc1`. Run `lake exe cache get`, then `lake build`.
- `.gitignore`: `.venv/`, `lean/.lake/`, `cache/`, `runs/*/tmp/`.
- `CLAUDE.md`: one paragraph pointing to doc-gmj and this guide, and the Global constraints above.

**Done when:** `uv run pytest` reports no tests and exits 0; `uv run pyright` passes; `lake build` in `lean/` succeeds; `Minilab/Basic.lean` holds `theorem minilab_smoke : (1 : ℕ) + 1 = 2 := rfl`.

### Task 2: Records and record store

**Files:** create `src/minilab/records.py`, `tests/test_records.py`.

**Produces:**
- `class Status(StrEnum)`: `proposed`, `untestable`, `ambiguous`, `killed`, `known`, `trivial`, `proved`, `survived`, `abandoned`.
- `class Stage(StrEnum)`: `generator`, `encoding`, `falsifier`, `literature`, `formaliser`, `triviality`, `prover`, `scorer`, `queue`.
- `class Family(StrEnum)`: `claude`, `codex`, `gemini`, `local`.
- `class Record(BaseModel)`: `id: str`, `run_id: str`, `pair: tuple[str, str]`, `mode: Literal["plain", "structured"]`, `statement: str`, `objects: list[str]`, `assumptions: list[str]`, `domain_of_validity: str`, `instance_family: str | None`, `provenance: Provenance`, `status: Status`, `stages: dict[Stage, StageOutput]`, `costs: Costs`.
- `class RecordStore`: `__init__(self, root: Path)`, `create(self, record: Record) -> None`, `load(self, record_id: str) -> Record`, `write_stage(self, record_id: str, stage: Stage, output: StageOutput, status: Status | None = None) -> Record`, `list(self, run_id: str, status: Status | None = None) -> list[Record]`.
- Storage: one JSON file per record at `runs/<run_id>/records/<id>.json`, written atomically (temp file, then rename).

**Tests:**
- `test_round_trip`: a record written and loaded compares equal.
- `test_stage_cannot_overwrite_other_stage`: `write_stage` for `falsifier` leaves `stages[Stage.generator]` byte-identical; writing `generator` twice raises `StageAlreadyWritten`.
- `test_status_transitions`: `killed`, `known`, `proved`, `trivial`, `abandoned`, `untestable` are final; a write that changes a final status raises `FinalStatus`.
- `test_untestable_without_instance_family`: creating a record with `instance_family=None` sets `status == Status.untestable`.

### Task 3: Run manifest and cost ledger

**Files:** create `src/minilab/run.py`, `src/minilab/ledger.py`, `tests/test_ledger.py`.

**Consumes:** `Family`, `Record`, `RecordStore` (Task 2).

**Produces:**
- `class MemoryPolicy(StrEnum)`: `blind`, `pair`, `all`.
- `class RunManifest(BaseModel)`: `run_id`, `mode`, `pairs: list[tuple[str, str]]`, `memory_policy: MemoryPolicy`, `memory_before: datetime | None`, `literature_before: date | None`, `budget_tokens_per_pair: int`, `families: list[Family]`, `started: datetime`. Saved as `runs/<run_id>/manifest.json`.
- `class Ledger`: `charge(self, record_id: str, family: Family, tokens_in: int, tokens_out: int, seconds: float, estimated: bool) -> None`; `log_review(self, record_id: str, minutes: float) -> None`; `pair_spend(self, pair: tuple[str, str]) -> int`.
- `def cost_report(store: RecordStore, ledger: Ledger, run_id: str) -> CostReport` with `hours_per_reviewed_survivor`, `tokens_per_reviewed_survivor`, and per-family shares of cost and of survivors.

**Tests:**
- `test_cost_report_per_reviewed_survivor`: 3 queued records, 2 reviewed, known charges; the report gives the exact hours and tokens divided by 2.
- `test_estimated_costs_are_flagged`: a charge with `estimated=True` makes `CostReport.has_estimates` true.
- `test_pair_spend_sums_all_stages`: charges from generator, encoding, and prover calls for one pair all count towards `pair_spend`.

### Task 4: Model adapter and isolation probe

**Files:** create `src/minilab/models/adapter.py`, `src/minilab/models/backends.py`, `scripts/probe_isolation.py`, `tests/test_adapter.py`.

Install Antigravity CLI first. Find the official install route from Google's documentation, and record the version on the bead.

**Consumes:** `Family`, `Ledger` (Task 3).

**Produces:**
- `class ModelReply(BaseModel)`: `text: str`, `tokens_in: int`, `tokens_out: int`, `estimated: bool`, `model: str`.
- `class Role(StrEnum)`: `generate`, `encode`, `reformulate`, `mechanism`, `formalise`, `roundtrip`, `audit`, `literature`, `prove`, `curate`, `grade`.
- `def call(family: Family, role: Role, prompt: str, *, record_id: str, pair: tuple[str, str], ledger: Ledger, budget: Budget) -> ModelReply`. It raises `BudgetExceeded` before calling if the pair's spend is at the budget. It raises `ModelUnavailable` on a non-zero exit, a timeout, or a usage-limit message.
- `def pick_checker(author: Family, available: list[Family]) -> Family`: never `author`, never `gemini`, prefers `claude` over `codex` except for short checking calls when Codex budget remains.
- Backends: Claude (`claude -p --output-format json` with the isolation flags), Codex (`codex exec` with a read-only sandbox, web search off, JSON output), Gemini (Antigravity CLI with tools off), Local (ollama HTTP API, `qwen3.5:4b`). Each runs in a fresh `tempfile.mkdtemp()` working directory with a scrubbed environment. Tokens come from each tool's JSON output; where a tool gives none, estimate at 4 characters per token and set `estimated=True`.

**Tests (fake backends):**
- `test_budget_exceeded_before_call`.
- `test_adapter_failure_leaves_record_unchanged`: a backend raising `ModelUnavailable` leaves the record file byte-identical.
- `test_pick_checker_never_author_never_gemini`: for every author family.
- `test_scrubbed_env_and_empty_cwd`: the fake backend sees no `ANTHROPIC_*` variables and an empty cwd.

**Live probe:** `scripts/probe_isolation.py` places a canary file next to the repo and asks each tool to read it and to fetch a URL. It also asks about a fact that only a web search could answer. It writes `runs/probes/<date>.json` with pass or fail per tool. A tool passes only if all three attempts fail. Run it and record the result on the bead. A tool that fails is marked "normal runs only" in `src/minilab/models/isolation.json`.

### Task 5: Pair scheduler

**Files:** create `src/minilab/scheduler.py`, `tests/test_scheduler.py`, `state/pairs.json` (created on first use).

**Produces:**
- `DOMAINS: tuple[str, ...]`, holding the ten domains from doc-gmj in that order, with slugs: `optimisation`, `information-theory`, `probability`, `hd-statistics`, `random-matrices`, `optimal-transport`, `kernels`, `gradient-flows`, `information-geometry`, `graphs-spectral`.
- `def all_pairs() -> list[tuple[str, str]]`: 45 sorted unordered pairs.
- `class Scheduler`: `__init__(self, state_path: Path, rng: random.Random)`, `pick(self) -> tuple[str, str]` with weight `1 / (1 + visits)`, `record_visit(self, pair: tuple[str, str]) -> None`.

**Tests:** `test_45_pairs_no_self_pairs`; `test_weight_falls_with_visits` (over 10,000 seeded picks, a pair visited 9 times is picked less than a fresh one); `test_seeded_pick_is_deterministic`; `test_unknown_domain_rejected`.

---

## Phase 1: filters

### Task 6: Certificate library and pilot fixtures

**Files:** create `src/minilab/certify/exact.py`, `src/minilab/certify/interval.py`, `src/minilab/certify/duality.py`, `src/minilab/certify/check.py`, `tests/fixtures/pilot/`, `tests/test_certify.py`.

**Produces:**
- `class Certificate(BaseModel)`, a tagged union: `ExactCertificate` (rational point, exact margin), `IntervalCertificate` (point, digits, enclosure of the margin), `DualityCertificate` (point, primal value, dual value, digits, claimed violation).
- `def check_certificate(enc: Encoding, cert: Certificate) -> CheckResult`. `CheckResult` has `accepted: bool` and `reason: str`. It accepts only if all of these hold:
  1. the point satisfies every assumption predicate and the domain predicate, in exact arithmetic or by interval enclosure;
  2. the margin is negative: exactly, by enclosure, or with `primal - dual` smaller than 1/1000 of the violation;
  3. if the margin is within `1e-6` of zero, or the encoding flags the point as near-degenerate, it holds again at 100 digits.
- `Encoding` is defined in Task 7. This task uses a minimal stub of the protocol and replaces it when Task 7 lands.

**Pilot fixtures:** rebuild by hand, from `crossconj-ml/s7_*.py`, `verify_kills.py`, their `.out` files, and MuninnDB `sci` memory `01M85D7A6NM38RQ9EV6JAFK16A`:
- `warm_start_artefact`: a Blahut–Arimoto value from a warm start that falsely shows a violation;
- `grid_gap`: an uncertified grid point that falsely shows a violation;
- `out_of_domain`: a point outside the claimed domain;
- `s7_counterexample`: P = [[.20,.15,.05],[.05,.15,.40]] at s = 1.97408, with matching primal and dual bounds (Csiszár dual);
- `s7prime_counterexample`: from `s7_counter3.py`, at 60 digits.

If a pilot case cannot be rebuilt, note why on the bead and pick the nearest case the pilot does document.

**Tests:** `test_rejects_warm_start_artefact`, `test_rejects_grid_gap`, `test_rejects_out_of_domain`, `test_accepts_s7_counterexample`, `test_accepts_s7prime_counterexample`, `test_near_zero_margin_rechecked_at_100_digits`.

### Task 7: Encodings, sandbox, and dual-encoding agreement

**Files:** create `src/minilab/encoding/api.py`, `src/minilab/encoding/sandbox.py`, `src/minilab/encoding/agree.py`, `tests/test_encoding.py`.

**Consumes:** `call`, `pick_checker`, `Role.encode` (Task 4); `RecordStore` (Task 2).

**Produces:**
- `class Instance(BaseModel)`, frozen: `values: dict[str, Fraction | str | list[str]]`, where a `str` holds an exact decimal or `mpf` value at the stated precision, and `precision_digits: int`.
- `class Encoding(Protocol)`: `margin(self, x: Instance) -> mpf`, `assumptions(self, x: Instance) -> list[bool]`, `in_domain(self, x: Instance) -> bool`, `sample(self, rng: random.Random) -> Instance`, `near_degenerate(self, x: Instance) -> bool`.
- Model-written encodings are Python modules that may import only `minilab.encoding.api`, `mpmath`, `flint`, `fractions`, `math`, and `numpy`.
- `def run_encoding(source: str, calls: EncodingCalls, timeout_s: float) -> EncodingOutput`: runs the module in a subprocess with a time limit, a temporary directory, and no network (use `unshare -rn` if it works on this machine, otherwise `bwrap`; record which on the bead).
- `def encode_twice(record: Record) -> tuple[str, str]`: two families, neither sees the other's output.
- `def agree(a: str, b: str, n: int = 200, rtol: float = 1e-9) -> Agreement`: compares margins and predicates on `n` shared samples.
- Disagreement sets the status to `ambiguous` and asks the generator family for a sharper statement. After 2 rounds the status becomes `abandoned`.

**Tests:** `test_agreeing_encodings_pass`; `test_disagreeing_encodings_give_ambiguous`; `test_ambiguous_rounds_end_in_abandoned`; `test_encoding_sandbox_timeout_and_no_network` (an encoding with `while True: pass` times out, and one that opens a socket fails; both give `ambiguous`, never a verdict); `test_forbidden_import_rejected`.

### Task 8: Falsifier search, coverage, reformulation, and mechanisms

**Files:** create `src/minilab/falsify/search.py`, `src/minilab/falsify/coverage.py`, `src/minilab/falsify/reformulate.py`, `src/minilab/falsify/mechanism.py`, `tests/test_falsify.py`.

**Consumes:** `Encoding`, `run_encoding` (Task 7); `check_certificate` (Task 6); `call` (Task 4).

**Produces:**
- `def attack(enc: Encoding, budget: SearchBudget, rng: random.Random) -> AttackResult`: seeded multi-start, a local optimiser on the margin (`scipy.optimize.minimize`, cold starts only for anything used in a certificate), and boundary sampling. On a candidate violation, it builds a certificate and calls `check_certificate`.
- `class CoverageReport(BaseModel)`: `space`, `dimensions`, `starts`, `worst_margin`, `worst_point`, `uncovered: list[str]`.
- `def reformulate_and_attack(record: Record, enc: Encoding, max_reformulations: int = 3) -> AttackResult`: a model proposes a reparametrisation with maps there and back. The maps must agree with the original margin on 200 samples before use. A kill is mapped back and certified in the original variables.
- `def check_mechanism(record: Record, kill: Certificate) -> MechanismResult`: a checker family proposes a mechanism as a predicate plus an instance generator. It passes if the predicted failure holds on at least 9 of 10 fresh instances, each certified.
- The falsifier writes `killed` (with certificate), `survived` (with `CoverageReport`), or leaves the status unchanged on `ModelUnavailable`.

**Tests:** `test_planted_violation_found_and_certified` (a toy encoding with a known counterexample); `test_survived_always_has_coverage`; `test_bad_reformulation_rejected` (maps that disagree are refused); `test_s7_joint_law_reformulation_kills` (live-marked; uses the pilot's joint-law idea); `test_mechanism_needs_nine_of_ten`.

### Task 9: Lean runner

**Files:** create `src/minilab/lean/runner.py`, `tests/test_lean_runner.py`.

**Produces:** `def check_lean(source: str, timeout_s: float = 300) -> LeanResult`. `LeanResult` has `ok`, `errors: list[str]`, `has_sorry`, `axioms: list[str]`, and `timed_out`. It writes the source to a temporary file inside `lean/` and runs `lake env lean`. It appends `#print axioms` for each theorem. (Check whether the installed `lean-beam` tools are faster; use them if so, and say which on the bead.)

**Tests:** `test_smoke_theorem_ok`; `test_sorry_detected`; `test_axioms_listed` (a proof by `Classical.choice` lists it); `test_lean_timeout_is_not_a_verdict`.

### Task 10: Known-result filter

**Files:** create `src/minilab/literature/loogle.py`, `src/minilab/literature/arxiv.py`, `src/minilab/literature/mathoverflow.py`, `src/minilab/literature/pdf.py`, `src/minilab/literature/filter.py`, `tests/test_literature.py`, `tests/fixtures/sample.pdf`.

**Consumes:** `check_lean` (Task 9) for `exact?`; `call` (Task 4) for notes.

**Produces:**
- `def search_mathlib(statement_lean: str | None, query: str) -> list[Hit]` (Loogle API, then `exact?` through `check_lean`).
- `def search_arxiv(query: str, before: date | None) -> list[Paper]` (arXiv export API, `submittedDate` range). It also drops any result whose first version is after `before`, even if the API returns it.
- `def search_mathoverflow(query: str, before: date | None) -> list[Post]` (Stack Exchange API, `site=mathoverflow`, `todate`).
- `def pdf_text(url: str) -> str`: downloads to `cache/pdf/`, then runs `pdftotext -layout`.
- `def literature_check(record: Record, before: date | None) -> LiteratureOutput`: status `known` on a close match, with notes from a checker family; otherwise notes and no status change. Every query and hit is logged.
- No general web search engine anywhere in the package.

**Tests (recorded HTTP responses):** `test_date_cap_drops_late_papers`; `test_mathoverflow_date_cap`; `test_pdf_text_from_fixture`; `test_literature_outage_is_unchecked`; `test_no_general_web_search` (a static check that the package imports no search-engine client and calls no host except `arxiv.org`, `export.arxiv.org`, `api.stackexchange.com`, and `loogle.lean-lang.org`).

### Task 11: Formaliser, vacuity linter, round-trip check, and hypothesis audit

**Files:** create `src/minilab/formalise/formaliser.py`, `src/minilab/formalise/vacuity.py`, `src/minilab/formalise/roundtrip.py`, `src/minilab/formalise/audit.py`, `tests/test_formalise.py`, `tests/fixtures/pilot/s8.lean`.

**Consumes:** `check_lean` (Task 9); `call`, `pick_checker` (Task 4); the falsifier's witness from the record (Task 8).

**Produces:**
- `def formalise(record: Record) -> FormalOutput`: the Lean statement, with concepts as `def`s, compiled with `sorry` as the proof.
- `def vacuity(formal: FormalOutput, witness: Instance | None) -> VacuityResult`: hypotheses satisfiable (instantiate the witness in Lean where possible), none contradictory, none unused (Lean's unused-variables linter).
- `def roundtrip(formal: FormalOutput, author: Family) -> RoundTrip`: a family from `pick_checker(author, ...)` translates back to English without the original. Both texts are stored.
- `def audit(record: Record, formal: FormalOutput) -> AuditResult`: lists each Lean hypothesis with no counterpart in `record.assumptions`. The script matches by name and type first, and a checker model judges the remainder. Each extra hypothesis is added to the record as a stated weakening, or the formalisation is rejected.

**Tests:** `test_audit_flags_s8_kkt_hypothesis` (rebuild S8 from `crossconj/CrossconjML/SupportRank.lean` by reading it, never changing it; the KKT hypothesis must be listed); `test_roundtrip_family_differs_from_author`; `test_contradictory_hypotheses_rejected`; `test_unused_hypothesis_flagged`.

### Task 12: Triviality screen

**Files:** create `src/minilab/screen/triviality.py`, `tests/test_triviality.py`.

**Consumes:** `check_lean` (Task 9); `call` with `Family.local` and a checker family (Task 4).

**Produces:** `def screen(formal: FormalOutput) -> ScreenResult`, trying in order, each with a 60-second limit: `exact?`, `aesop`, `simp`, `positivity`, `nlinarith`; then 3 Lean proof attempts by `qwen3.5:4b`; then a checker's "corollary of named theorem" attempt, checked in Lean. Any success sets `trivial` and stores the proof. Not `polyrith`.

**Tests:** `test_tactic_screen_proves_easy_goal` (deterministic, for example `0 ≤ x^2`); `test_s1_flagged_trivial` and `test_s8_flagged_trivial` (live-marked; must flag in at least 2 of 3 runs; record the runs on the bead); `test_screen_failure_leaves_status`.

### Task 13: Prover loop

**Files:** create `src/minilab/prove/loop.py`, `tests/test_prover.py`.

**Consumes:** `check_lean` (Task 9); `call` (Task 4); `audit` (Task 11).

**Produces:** `def prove(record: Record, formal: FormalOutput, budget: ProverBudget) -> ProofResult`. A compiler-feedback loop, by default at most 8 attempts and 200,000 tokens per conjecture. On success: status `proved`, with the Lean file and `axioms`. A proof that changes the statement or adds a hypothesis goes back through `audit`.

**Tests:** `test_budget_cap_stops_loop`; `test_added_hypothesis_triggers_audit`; `test_proof_with_sorry_not_accepted`; `test_nonstandard_axiom_blocks_proved`.

### Task 14: Scorer and human queue

**Files:** create `src/minilab/score.py`, `src/minilab/queue.py`, `src/minilab/cli.py` (the `queue` and `review` commands), `tests/test_queue.py`.

**Consumes:** `RecordStore` (Task 2); `Ledger.log_review` (Task 3).

**Produces:**
- `def score(record: Record) -> Score`: triviality signals and a compression measure (statement length in tokens divided by proof length in tokens; the builder may refine it and must say how on the bead). Used for ranking only.
- `def queue_items(store: RecordStore, run_id: str) -> list[QueueItem]`: exactly survivors with a coverage report, `proved` records that are not `trivial`, and certified kills with a passed mechanism.
- `minilab queue <run_id>` prints each item: statement, round-trip text, audit, literature notes, evidence, mechanism. `minilab review <record_id> --verdict interesting|not-interesting|known --note TEXT` stores the verdict and logs the minutes since the item was shown.

**Tests:** `test_queue_membership` (one record per status; only the three allowed kinds appear); `test_untestable_and_ambiguous_never_queued`; `test_review_logs_minutes`.

---

## Phase 2: generation and runs

### Task 15: Generator

**Files:** create `src/minilab/generate/plain.py`, `src/minilab/generate/structured.py`, `src/minilab/generate/parse.py`, `tests/test_generate.py`.

**Consumes:** `call`, `Role.generate` (Task 4); `Record`, `RecordStore` (Task 2).

**Produces:**
- `def generate_plain(pair: tuple[str, str], family: Family, n: int) -> list[Record]`.
- `def generate_structured(pair: tuple[str, str], family: Family, n: int) -> list[Record]`: elicit, abstract, cross, with each step's prompt and reply logged on the records' provenance.
- Both ask for JSON with `statement`, `objects`, `assumptions`, `domain_of_validity`, and `instance_family`. Parse failures are retried once, then dropped and logged.

**Tests (fake models):** `test_missing_instance_family_gives_untestable`; `test_structured_logs_three_steps`; `test_gemini_allowed_only_for_generate` (the adapter refuses `Family.gemini` for any other role); `test_malformed_json_retried_then_dropped`.

### Task 16: Pipeline driver, memory policy, MuninnDB sync, and the run skill

**Files:** create `src/minilab/pipeline.py`, `src/minilab/memory.py`, extend `src/minilab/cli.py` (`run`, `resume`, `report`), `tests/test_pipeline.py`. User-level skill at `~/.claude/skills/minilab-run/SKILL.md` (allowed under `~/.claude/`).

**Consumes:** every earlier task.

**Produces:**
- `minilab run --mode plain|structured --pairs auto|<file> --memory blind|pair|all [--memory-before DATE] [--literature-before DATE] --budget-tokens-per-pair N`. It writes the manifest and runs records through the stages in the spec's order, stopping each record at its first final status.
- `minilab resume <run_id>` continues every record from its last written stage.
- `minilab report <run_id>` prints the `CostReport`.
- `class MemoryClient`: wraps MuninnDB's local API. It enforces the manifest's policy: with `blind` it raises on any read; with `pair` it filters by the pair tag; `memory_before` becomes the `before` filter. It writes one memory per final record to vault `minilab`, with the run ID, status, and a path to the record. It never reads vault `sci`.
- The skill tells a Claude session how to start a run, read the queue, and review items. It names the safeguards and must not repeat the spec.

**Tests:** `test_blind_policy_makes_no_reads` (a spy client); `test_memory_before_passed_through`; `test_never_reads_sci_vault`; `test_run_resume_after_failure` (a fake `ModelUnavailable` halfway, then `resume` finishes with no duplicate stage writes); `test_stage_order_matches_spec`.

### Task 17: Pilot validation and first smoke run

**Files:** create `tests/test_pilot_validation.py`.

Run the whole pipeline on the pilot fixtures from Tasks 6, 11, and 12. Then do one live smoke run on the pair `information-theory` × `optimisation` in structured mode, with memory `blind`, `--budget-tokens-per-pair` 300000.

**Done when:**
- the validation tests pass: three false kills rejected, S7 and S7′ accepted, S8's KKT hypothesis flagged, S1 and S8 flagged trivial (2 of 3);
- the smoke run finishes or stops cleanly on budget;
- `minilab report` gives hours and tokens per reviewed survivor after the user reviews the queue;
- a short DOCA note (type `context`, linked to this bead) records the costs and any filter that misfired.

---

## Phase 3: retrodiction benchmark

Start Phase 3 only after Task 17 is done and the user has reviewed the smoke run.

### Task 18: Curate the sealed hidden set

The task runs in a **separate session** that never runs the pipeline.

- Choose 20 to 30 results whose first arXiv version is on or after 2026-07-01. Each must touch two of the ten domains, fit in a few lines, and come from no more than 8 domain pairs in total.
- Store them in `/home/thedryhtscipe/dev/sci/minilab-sealed/hidden.json`, outside the repo. Give each entry the arXiv ID, the first-version date, the statement, and its pair.
- Write only the pair list to the repo: `benchmark/pairs.json`.
- Before closing, confirm that Task 4's probe passed for each family used in the benchmark. Record which families qualify.

**Done when:** `benchmark/pairs.json` lists at most 8 pairs, `hidden.json` exists outside the repo, and the bead notes list the qualifying families. The bead never holds hidden statements.

### Task 19: Run both modes

Run `minilab run --mode plain` and `minilab run --mode structured` on `benchmark/pairs.json`. Use `--memory blind --literature-before 2026-06-30`, the qualifying families only, and the same `--budget-tokens-per-pair` for both. Start with 300000 tokens per pair, and scale back if the user says it costs too much. Keep each mode's run directory separate. Neither run may read the other's records.

**Done when:** both runs have finished or stopped on budget, both queues exist, and both cost reports are on the bead.

### Task 20: Grade and decide

The task runs in a **separate grader session**.

1. Label the two queues A and B at random, and keep the key in the sealed folder.
2. The grader compares each queue with `hidden.json` and proposes strict and near-miss matches with reasons.
3. The user confirms or rejects each match.
4. Unblind and apply the decision rule from doc-gmj.
5. Write the result as a DOCA `decision` document: the counts, the queue sizes, near-misses per queued item, the costs from both reports, and the verdict (scale up, don't scale up, or inconclusive). Link it to this bead.

---

## Dependencies

- Tasks 2 to 5 need Task 1.
- Task 3 needs Task 2.
- Task 4 needs Task 3.
- Tasks 6 and 9 need Task 1.
- Task 7 needs Tasks 4 and 6.
- Task 8 needs Task 7.
- Task 10 needs Tasks 4 and 9.
- Task 11 needs Tasks 8 and 9.
- Task 12 needs Task 11.
- Task 13 needs Task 12.
- Task 14 needs Task 3.
- Task 15 needs Task 4.
- Task 16 needs Tasks 5, 8, 10, 13, 14, and 15.
- Task 17 needs Task 16.
- Task 18 needs Tasks 4 and 17.
- Task 19 needs Task 18.
- Task 20 needs Task 19.

Tasks that can run at the same time once their inputs exist: 5 alongside 3 to 4; 6 and 9 alongside 2 to 4; 10 alongside 7 to 8; 14 and 15 alongside Phase 1.
