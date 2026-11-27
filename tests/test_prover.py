"""Tests for the prover loop.

Build plan doc-5wh, Task 13; bead minilab-880beb. Model calls go to fake
backends; Lean runs for real against the repo's mathlib (`lean/.lake`).

User decisions on the bead (2026-11-26): a proof that needs an added
hypothesis, which the audit accepts, is held for review and the record stays
`survived`; so is a proof of a formalisation the formaliser stage flagged;
the prover is Claude, else Gemini, else Codex.
"""

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from pydantic import JsonValue

from minilab.formalise.audit import AuditResult
from minilab.formalise.formaliser import (
    STANDARD_AXIOMS,
    FormalOutput,
    LeanTimeout,
    formal_from_source,
)
from minilab.formalise.vacuity import VacuityResult, WitnessCheck
from minilab.ledger import Ledger
from minilab.models.adapter import GEMINI_ROLES, Budget, Role
from minilab.models.backends import ModelReply, ModelUnavailable
from minilab.prove import loop
from minilab.prove.loop import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_TOKENS,
    ProofResult,
    ProverBudget,
    prove,
    prover_family,
    run_prover_stage,
)
from minilab.records import Family, Provenance, Record, RecordStore, Stage, StageOutput, Status
from minilab.run import MemoryPolicy, RunManifest, save_manifest

RUN = "prv"
PAIR = ("information-theory", "optimisation")
LEAN_TIMEOUT = 240.0
BUDGET = Budget(tokens_per_pair=10_000_000)

# None of the screen's tactics proves this; induction does.
NEEDS_INDUCTION = """import Mathlib

def g : ℕ → ℕ
  | 0 => 1
  | (n + 1) => g n + 1

theorem conjecture (n : ℕ) : 0 < g n := by
  sorry
"""

INDUCTION_PROOF = "induction n <;> simp [g] <;> omega"

# False (x = 2), so it cannot be proved as stated; with `x < 1` added it can.
FALSE = """import Mathlib

theorem conjecture (x : ℝ) (hx : 0 < x) : x ^ 2 < x := by
  sorry
"""

WEAKENED = """theorem conjecture (x : ℝ) (hx : 0 < x) (h1 : x < 1) : x ^ 2 < x := by
  nlinarith"""

EASY = """import Mathlib

theorem conjecture (x : ℝ) : 0 ≤ x ^ 2 := by
  sorry
"""


# --- fakes ---


class FakeBackend:
    """Replies by the kind of prompt; records every prompt it is sent."""

    def __init__(self, family: Family, replies: Mapping[str, list[str]], tokens: int = 10) -> None:
        self.family = family
        self.replies = {k: list(v) for k, v in replies.items()}
        self.tokens = tokens
        self.prompts: list[tuple[str, str]] = []
        self.fail = False

    def run(
        self, prompt: str, *, cwd: Path, env: Mapping[str, str], timeout_s: float
    ) -> ModelReply:
        kind = kind_of(prompt)
        self.prompts.append((kind, prompt))
        if self.fail:
            raise ModelUnavailable("usage limit reached")
        queue = self.replies.get(kind, [])
        text = queue.pop(0) if len(queue) > 1 else (queue[0] if queue else "")
        return ModelReply(
            text=text,
            tokens_in=self.tokens,
            tokens_out=self.tokens,
            estimated=False,
            model="fake-1",
        )


def kind_of(prompt: str) -> str:
    if "Write a Lean 4 proof of this theorem" in prompt:
        return "prove"
    if "hypothesis audit" in prompt:
        return "audit"
    return "other"


def fakes(
    by_family: Mapping[Family, Mapping[str, list[str]]], tokens: int = 10
) -> dict[Family, FakeBackend]:
    return {f: FakeBackend(f, by_family.get(f, {}), tokens) for f in Family}


def lean_block(text: str) -> str:
    return f"Here is a proof.\n\n```lean\n{text}\n```\n"


def calls(backends: Mapping[Family, FakeBackend], kind: str) -> list[tuple[Family, str]]:
    return [(f, p) for f, b in backends.items() for k, p in b.prompts if k == kind]


def audit_reply(**kinds: str) -> str:
    return json.dumps(
        {
            "hypotheses": [
                {"label": label, "kind": kind, "counterpart": None, "reason": f"{label} is {kind}"}
                for label, kind in kinds.items()
            ],
            "hidden": [],
        }
    )


# --- records ---


def make_record(local: str = "a") -> Record:
    return Record(
        id=f"{RUN}.{local}",
        run_id=RUN,
        pair=PAIR,
        mode="structured",
        statement="The conjecture.",
        objects=["x"],
        assumptions=["x is positive"],
        domain_of_validity="the reals",
        instance_family="x in [-10, 10]",
        provenance=Provenance(family=Family.gemini, model="m", prompt_hash="h"),
    )


@pytest.fixture
def store(tmp_path: Path) -> RecordStore:
    return RecordStore(tmp_path)


def make_ledger(store: RecordStore, families: tuple[Family, ...] = tuple(Family)) -> Ledger:
    save_manifest(
        store.root,
        RunManifest(
            run_id=RUN,
            mode="structured",
            pairs=(PAIR,),
            memory_policy=MemoryPolicy.blind,
            memory_before=None,
            literature_before=None,
            budget_tokens_per_pair=BUDGET.tokens_per_pair,
            families=families,
            started=datetime(2026, 11, 26, 12, 0, tzinfo=UTC),
        ),
    )
    return Ledger(store, RUN)


_probed: dict[str, FormalOutput] = {}


def probed(source: str) -> FormalOutput:
    """The source compiled and probed by Lean, as Codex's formalisation (cached)."""
    if source not in _probed:
        _probed[source] = formal_from_source(source, Family.codex, "fake-1", timeout_s=LEAN_TIMEOUT)
    assert _probed[source].ok, _probed[source].problems
    return _probed[source]


def clean_audit() -> AuditResult:
    return AuditResult(
        hypotheses=[],
        extra=[],
        weakenings=[],
        hidden=[],
        rejected=False,
        reasons=[],
        checker=Family.claude,
        judged=True,
    )


def clean_vacuity(hypotheses: bool) -> VacuityResult:
    return VacuityResult(
        contradictory=False,
        satisfied_by_witness=hypotheses,
        witness=[WitnessCheck(label="hx", state="holds")] if hypotheses else [],
        unused=[],
        timed_out=False,
        notes=[],
    )


def formaliser_data(source: str, audit: AuditResult | None = None) -> dict[str, JsonValue]:
    formal = probed(source)
    a = clean_audit() if audit is None else audit
    return {
        "rejected": False,
        "reasons": [],
        "family": Family.codex.value,
        "model": "fake-1",
        "attempts": 1,
        "problems": [],
        "lean_statement": formal.lean_statement,
        "roundtrip": "text",
        "roundtrip_family": Family.claude.value,
        "audit": cast(dict[str, JsonValue], a.model_dump(mode="json")),
        "weakenings": cast(list[JsonValue], list(a.weakenings)),
        "vacuity": cast(
            dict[str, JsonValue], clean_vacuity(bool(formal.hypotheses)).model_dump(mode="json")
        ),
        "binders": cast(list[JsonValue], [b.model_dump(mode="json") for b in formal.binders]),
    }


def screen_data(*, screened: bool = True, held: bool = False) -> dict[str, JsonValue]:
    return {
        "screened": screened,
        "trivial": False,
        "held": held,
        "hold_reasons": ["a flag"] if held else [],
        "reasons": [] if screened else ["the formaliser stage rejected the formalisation"],
        "proved_by": "simp" if held else None,
        "proof": "simp" if held else None,
        "corollary_of": None,
        "near_misses": [],
        "attempts": [],
    }


def screened_record(
    store: RecordStore,
    source: str,
    *,
    local: str = "a",
    audit: AuditResult | None = None,
    screen: dict[str, JsonValue] | None = None,
) -> Record:
    """A survivor with falsifier, literature, formaliser, and triviality stages."""
    record = make_record(local)
    store.create(record)
    now = datetime(2026, 11, 26, 13, 0, tzinfo=UTC)
    store.write_stage(
        record.id,
        Stage.falsifier,
        StageOutput(written_at=now, data={"coverage": {"worst_point": None}}),
        status=Status.survived,
    )
    store.write_stage(record.id, Stage.literature, StageOutput(written_at=now, notes="none"))
    directory = store.root / "runs" / RUN / "formaliser"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{record.id}.lean").write_text(source, encoding="utf-8")
    store.write_stage(
        record.id,
        Stage.formaliser,
        StageOutput(
            written_at=now,
            data=formaliser_data(source, audit),
            evidence=[f"formaliser/{record.id}.lean"],
        ),
    )
    store.write_stage(
        record.id,
        Stage.triviality,
        StageOutput(written_at=now, data=screen if screen is not None else screen_data()),
    )
    return store.load(record.id)


def run_stage(
    store: RecordStore,
    record: Record,
    backends: Mapping[Family, FakeBackend],
    budget: ProverBudget | None = None,
    families: tuple[Family, ...] = tuple(Family),
) -> ProofResult | None:
    return run_prover_stage(
        record.id,
        ledger=make_ledger(store, families),
        budget=BUDGET,
        prover_budget=budget or ProverBudget(),
        backends=backends,
        lean_timeout_s=LEAN_TIMEOUT,
    )


def prover_data(store: RecordStore, record: Record) -> dict[str, JsonValue]:
    return store.load(record.id).stages[Stage.prover].data


# --- the loop ---


def test_default_budget_is_eight_attempts_and_200k_tokens() -> None:
    assert (DEFAULT_MAX_ATTEMPTS, DEFAULT_MAX_TOKENS) == (8, 200_000)
    assert ProverBudget() == ProverBudget(max_attempts=8, max_tokens=200_000)


def test_proof_as_stated_sets_proved_with_file_and_axioms(store: RecordStore) -> None:
    record = screened_record(store, NEEDS_INDUCTION)
    backends = fakes({Family.claude: {"prove": [lean_block(INDUCTION_PROOF)]}})
    result = run_stage(store, record, backends)
    assert result is not None and result.proved and not result.held
    saved = store.load(record.id)
    assert saved.status is Status.proved
    data = saved.stages[Stage.prover].data
    assert data["proved"] is True and data["held"] is False
    assert data["proof"] == INDUCTION_PROOF
    axioms = data["axioms"]
    assert isinstance(axioms, list) and axioms
    assert set(cast(list[str], axioms)) <= STANDARD_AXIOMS
    assert saved.stages[Stage.prover].evidence == [f"prover/{record.id}.lean"]
    lean_file = store.root / "runs" / RUN / "prover" / f"{record.id}.lean"
    assert INDUCTION_PROOF in lean_file.read_text(encoding="utf-8")
    assert "sorry" not in lean_file.read_text(encoding="utf-8")


def test_axioms_come_from_lean(store: RecordStore) -> None:
    # A proof about the reals goes through classical logic.
    record = screened_record(store, EASY)
    backends = fakes({Family.claude: {"prove": [lean_block("positivity")]}})
    result = run_stage(store, record, backends)
    assert result is not None and result.proved
    assert "Classical.choice" in result.axioms


def test_compiler_feedback_reaches_the_next_attempt(store: RecordStore) -> None:
    record = screened_record(store, NEEDS_INDUCTION)
    backends = fakes({Family.claude: {"prove": [lean_block("simp"), lean_block(INDUCTION_PROOF)]}})
    result = run_stage(store, record, backends)
    assert result is not None and result.proved
    prompts = [p for _, p in calls(backends, "prove")]
    assert len(prompts) == 2
    assert "Lean refused it" not in prompts[0]
    assert "Lean refused it" in prompts[1] and "simp" in prompts[1]
    assert [a.outcome for a in result.attempts] == ["failed", "proved"]


def test_budget_cap_stops_loop(store: RecordStore) -> None:
    # No reply holds a proof, so only the cap can stop the loop.
    record = screened_record(store, FALSE)
    backends = fakes({Family.claude: {"prove": ["I cannot prove this."]}})
    result = run_stage(store, record, backends)
    assert result is not None and not result.proved
    assert len(calls(backends, "prove")) == DEFAULT_MAX_ATTEMPTS
    assert result.stopped == "attempts"
    saved = store.load(record.id)
    assert saved.status is Status.survived
    assert saved.stages[Stage.prover].data["stopped"] == "attempts"


def test_token_cap_stops_loop(store: RecordStore) -> None:
    # Each call spends 60,000 tokens: after 4 calls (240,000) the cap of 200,000 is spent.
    record = screened_record(store, FALSE)
    backends = fakes({Family.claude: {"prove": ["I cannot prove this."]}}, tokens=30_000)
    result = run_stage(store, record, backends)
    assert result is not None and not result.proved
    assert len(calls(backends, "prove")) == 4
    assert result.stopped == "tokens"
    assert result.tokens == 240_000


def test_smaller_budget_is_obeyed(store: RecordStore) -> None:
    record = screened_record(store, FALSE)
    backends = fakes({Family.claude: {"prove": ["no"]}})
    result = run_stage(store, record, backends, ProverBudget(max_attempts=2, max_tokens=10**9))
    assert result is not None and len(calls(backends, "prove")) == 2


def test_proof_with_sorry_not_accepted(store: RecordStore) -> None:
    record = screened_record(store, FALSE)
    backends = fakes({Family.claude: {"prove": [lean_block("sorry")]}})
    result = run_stage(store, record, backends, ProverBudget(max_attempts=1))
    assert result is not None and not result.proved and not result.held
    assert result.attempts[0].outcome == "refused"
    assert store.load(record.id).status is Status.survived


def test_nonstandard_axiom_blocks_proved(
    store: RecordStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The formaliser's proved mode already refuses such a file; this checks the
    # prover's own guard, in case a file ever reaches it with one.
    real = loop.formal_from_source

    def with_extra_axiom(
        source: str,
        family: Family | str,
        model: str,
        *,
        timeout_s: float = LEAN_TIMEOUT,
        attempts: int = 1,
        proved: bool = False,
    ) -> FormalOutput:
        out = real(source, family, model, timeout_s=timeout_s, attempts=attempts, proved=proved)
        if proved:
            return out.model_copy(update={"axioms": [*out.axioms, "Minilab.trustMe"]})
        return out

    monkeypatch.setattr(loop, "formal_from_source", with_extra_axiom)
    record = screened_record(store, NEEDS_INDUCTION)
    backends = fakes({Family.claude: {"prove": [lean_block(INDUCTION_PROOF)]}})
    result = run_stage(store, record, backends, ProverBudget(max_attempts=1))
    assert result is not None and not result.proved
    assert result.attempts[0].outcome == "refused"
    assert "Minilab.trustMe" in result.attempts[0].detail
    assert store.load(record.id).status is Status.survived


def test_prover_cannot_add_a_declaration(store: RecordStore) -> None:
    record = screened_record(store, NEEDS_INDUCTION)
    sneaky = (
        "theorem conjecture (n : ℕ) : 0 < g n := by\n"
        f"  {INDUCTION_PROOF}\n\ntheorem extra : True := trivial"
    )
    backends = fakes({Family.claude: {"prove": [lean_block(sneaky)]}})
    result = run_stage(store, record, backends, ProverBudget(max_attempts=1))
    assert result is not None and not result.proved
    assert result.attempts[0].outcome == "refused"


def test_restated_theorem_with_the_same_statement_is_a_proof(store: RecordStore) -> None:
    record = screened_record(store, NEEDS_INDUCTION)
    restated = f"theorem conjecture (n : ℕ) : 0 < g n := by\n  {INDUCTION_PROOF}"
    backends = fakes({Family.claude: {"prove": [lean_block(restated)]}})
    result = run_stage(store, record, backends)
    assert result is not None and result.proved and not result.weakened
    assert store.load(record.id).status is Status.proved


# --- added hypotheses go through the audit ---


def test_added_hypothesis_triggers_audit(store: RecordStore) -> None:
    record = screened_record(store, FALSE)
    backends = fakes(
        {
            Family.claude: {"prove": [lean_block(WEAKENED)]},
            Family.codex: {"audit": [audit_reply(hx="matches", h1="weakening")]},
        }
    )
    result = run_stage(store, record, backends)
    assert result is not None
    audits = calls(backends, "audit")
    # The prover is Claude, so the audit's checker is Codex, never Gemini.
    assert [f for f, _ in audits] == [Family.codex]
    assert "h1" in audits[0][1]
    assert result.weakened and result.held and not result.proved
    assert result.added_hypotheses == ["h1 : x < (1 : ℝ)"]
    assert result.audit is not None and result.audit.weakenings == ["h1: x < (1 : ℝ)"]
    saved = store.load(record.id)
    assert saved.status is Status.survived
    data = saved.stages[Stage.prover].data
    assert data["held"] is True and data["proved"] is False and data["weakened"] is True
    assert data["added_hypotheses"] == ["h1 : x < (1 : ℝ)"]
    reasons = data["hold_reasons"]
    assert isinstance(reasons, list) and any("h1 : x < (1 : ℝ)" in str(r) for r in reasons)
    assert isinstance(data["audit"], dict)
    statement = data["weakened_statement"]
    assert isinstance(statement, str) and "h1 : x < (1 : ℝ)" in statement


def test_added_hypothesis_the_audit_rejects_is_not_held(store: RecordStore) -> None:
    record = screened_record(store, FALSE)
    backends = fakes(
        {
            Family.claude: {"prove": [lean_block(WEAKENED), ""]},
            Family.codex: {"audit": [audit_reply(hx="matches", h1="begs_question")]},
        }
    )
    result = run_stage(store, record, backends, ProverBudget(max_attempts=2))
    assert result is not None and not result.held and not result.proved
    assert [a.outcome for a in result.attempts] == ["audit_rejected", "no_proof"]
    prompts = [p for _, p in calls(backends, "prove")]
    assert "begs the question" in prompts[1]
    saved = store.load(record.id)
    assert saved.status is Status.survived
    assert saved.stages[Stage.prover].data["held"] is False


def test_changed_conclusion_is_refused_without_audit(store: RecordStore) -> None:
    record = screened_record(store, FALSE)
    changed = "theorem conjecture (x : ℝ) (hx : 0 < x) : 0 < x := by\n  exact hx"
    backends = fakes({Family.claude: {"prove": [lean_block(changed)]}})
    result = run_stage(store, record, backends, ProverBudget(max_attempts=1))
    assert result is not None and not result.proved and not result.held
    assert result.attempts[0].outcome == "refused"
    assert "conclusion" in result.attempts[0].detail
    assert calls(backends, "audit") == []


def test_dropped_hypothesis_is_refused(store: RecordStore) -> None:
    record = screened_record(store, FALSE)
    dropped = "theorem conjecture (x : ℝ) (h1 : x < 1) (h2 : 0 < x) : x ^ 2 < x := by\n  nlinarith"
    backends = fakes({Family.claude: {"prove": [lean_block(dropped)]}})
    result = run_stage(store, record, backends, ProverBudget(max_attempts=1))
    assert result is not None and not result.held
    assert result.attempts[0].outcome == "refused"
    assert calls(backends, "audit") == []


# --- the stage ---


def test_held_screen_record_is_skipped(store: RecordStore) -> None:
    record = screened_record(store, NEEDS_INDUCTION, screen=screen_data(held=True))
    backends = fakes({Family.claude: {"prove": [lean_block(INDUCTION_PROOF)]}})
    assert run_stage(store, record, backends) is None
    assert calls(backends, "prove") == []
    saved = store.load(record.id)
    assert saved.status is Status.survived
    data = saved.stages[Stage.prover].data
    assert data["attempted"] is False and data["proved"] is False
    assert any("held" in str(r) for r in cast(list[JsonValue], data["reasons"]))


def test_unscreened_record_is_skipped(store: RecordStore) -> None:
    record = screened_record(store, NEEDS_INDUCTION, screen=screen_data(screened=False))
    backends = fakes({Family.claude: {"prove": [lean_block(INDUCTION_PROOF)]}})
    assert run_stage(store, record, backends) is None
    assert calls(backends, "prove") == []
    assert store.load(record.id).stages[Stage.prover].data["attempted"] is False


def test_flagged_formalisation_proof_is_held(store: RecordStore) -> None:
    flagged = clean_audit().model_copy(update={"reasons": ["something looks off"]})
    record = screened_record(store, NEEDS_INDUCTION, audit=flagged)
    backends = fakes({Family.claude: {"prove": [lean_block(INDUCTION_PROOF)]}})
    result = run_stage(store, record, backends)
    assert result is not None and result.held and not result.proved
    saved = store.load(record.id)
    assert saved.status is Status.survived
    data = saved.stages[Stage.prover].data
    assert data["held"] is True and data["proof"] == INDUCTION_PROOF
    assert any("something looks off" in str(r) for r in cast(list[JsonValue], data["hold_reasons"]))


def test_model_failure_leaves_record_unchanged(store: RecordStore) -> None:
    record = screened_record(store, NEEDS_INDUCTION)
    path = store.root / "runs" / RUN / "records" / f"{record.id}.json"
    before = path.read_bytes()
    backends = fakes({Family.claude: {"prove": [lean_block(INDUCTION_PROOF)]}})
    backends[Family.claude].fail = True
    with pytest.raises(ModelUnavailable):
        run_stage(store, record, backends)
    assert path.read_bytes() == before


def test_stage_needs_a_screened_survivor(store: RecordStore) -> None:
    record = make_record()
    store.create(record)
    with pytest.raises(ValueError):
        run_stage(store, record, fakes({}))


def test_stage_runs_once(store: RecordStore) -> None:
    record = screened_record(store, FALSE)
    backends = fakes({Family.claude: {"prove": ["no"]}})
    ledger = make_ledger(store)

    def once() -> None:
        run_prover_stage(
            record.id,
            ledger=ledger,
            budget=BUDGET,
            prover_budget=ProverBudget(max_attempts=1),
            backends=backends,
            lean_timeout_s=LEAN_TIMEOUT,
        )

    once()
    with pytest.raises(ValueError):
        once()


def test_prove_refuses_a_formalisation_that_did_not_check(store: RecordStore) -> None:
    record = make_record()
    store.create(record)
    bad = probed(FALSE).model_copy(update={"ok": False})
    with pytest.raises(ValueError):
        prove(
            record,
            bad,
            ProverBudget(),
            ledger=make_ledger(store),
            pair_budget=BUDGET,
            backends=fakes({}),
        )


# --- who proves ---


@pytest.mark.parametrize(
    ("families", "family"),
    [
        ((Family.claude, Family.gemini, Family.codex), Family.claude),
        ((Family.gemini, Family.codex), Family.gemini),
        ((Family.codex, Family.local), Family.codex),
        ((Family.local,), None),
    ],
)
def test_prover_family_order(families: tuple[Family, ...], family: Family | None) -> None:
    assert prover_family(families) is family


def test_gemini_may_prove_and_still_checks_nothing() -> None:
    assert Role.prove in GEMINI_ROLES
    assert GEMINI_ROLES == frozenset({Role.generate, Role.screen, Role.prove})


def test_gemini_proves_when_claude_is_absent(store: RecordStore) -> None:
    record = screened_record(store, NEEDS_INDUCTION)
    backends = fakes({Family.gemini: {"prove": [lean_block(INDUCTION_PROOF)]}})
    result = run_stage(store, record, backends, families=(Family.gemini, Family.codex))
    assert result is not None and result.proved and result.prover is Family.gemini
    assert [f for f, _ in calls(backends, "prove")] == [Family.gemini]


def test_no_prover_family_writes_an_unattempted_stage(store: RecordStore) -> None:
    record = screened_record(store, NEEDS_INDUCTION)
    backends = fakes({})
    result = run_stage(store, record, backends, families=(Family.local,))
    assert result is not None and not result.proved and result.prover is None
    assert calls(backends, "prove") == []
    assert store.load(record.id).status is Status.survived


# --- found in review ---

PAD = " ∧ ".join(["(1 : ℕ) = 1"] * 160)
"""Long enough that Lean's printed conclusion is cut (the probe keeps 1500 characters)."""

CUT_FALSE = f"""import Mathlib

theorem conjecture (x : ℕ) : {PAD} ∧ x = x + 1 := by
  sorry
"""


def test_cut_statement_cannot_be_restated(store: RecordStore) -> None:
    # The two conclusions print the same once cut, so only the refusal stops this.
    assert probed(CUT_FALSE).conclusion_truncated
    record = screened_record(store, CUT_FALSE)
    swapped = f"theorem conjecture (x : ℕ) : {PAD} ∧ x = x := by\n  simp"
    backends = fakes({Family.claude: {"prove": [lean_block(swapped)]}})
    result = run_stage(store, record, backends, ProverBudget(max_attempts=1))
    assert result is not None and not result.proved and not result.held
    assert result.attempts[0].outcome == "refused"
    assert "too long" in result.attempts[0].detail
    assert store.load(record.id).status is Status.survived


def test_every_attempt_timed_out_raises_and_writes_nothing(
    store: RecordStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = loop.formal_from_source

    def slow(
        source: str,
        family: Family | str,
        model: str,
        *,
        timeout_s: float = LEAN_TIMEOUT,
        attempts: int = 1,
        proved: bool = False,
    ) -> FormalOutput:
        if proved:
            raise LeanTimeout("Lean hit its time limit")
        return real(source, family, model, timeout_s=timeout_s, attempts=attempts, proved=proved)

    monkeypatch.setattr(loop, "formal_from_source", slow)
    record = screened_record(store, NEEDS_INDUCTION)
    path = store.root / "runs" / RUN / "records" / f"{record.id}.json"
    before = path.read_bytes()
    backends = fakes({Family.claude: {"prove": [lean_block(INDUCTION_PROOF)]}})
    with pytest.raises(LeanTimeout):
        run_stage(store, record, backends, ProverBudget(max_attempts=2))
    assert path.read_bytes() == before


def test_added_hypothesis_with_changed_conclusion_is_refused(store: RecordStore) -> None:
    record = screened_record(store, FALSE)
    changed = "theorem conjecture (x : ℝ) (hx : 0 < x) (h1 : x < 1) : 0 < x := by\n  exact hx"
    backends = fakes({Family.claude: {"prove": [lean_block(changed)]}})
    result = run_stage(store, record, backends, ProverBudget(max_attempts=1))
    assert result is not None and not result.held and not result.proved
    assert result.attempts[0].outcome == "refused"
    assert "conclusion" in result.attempts[0].detail
    assert calls(backends, "audit") == []


def test_gemini_weakened_proof_is_audited_by_codex(store: RecordStore) -> None:
    record = screened_record(store, FALSE)
    backends = fakes(
        {
            Family.gemini: {"prove": [lean_block(WEAKENED)]},
            Family.codex: {"audit": [audit_reply(hx="matches", h1="weakening")]},
        }
    )
    result = run_stage(store, record, backends, families=(Family.gemini, Family.codex))
    assert result is not None and result.held and result.prover is Family.gemini
    assert [f for f, _ in calls(backends, "audit")] == [Family.codex]
