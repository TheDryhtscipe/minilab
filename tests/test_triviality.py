"""Tests for the triviality screen.

Build plan doc-5wh, Task 12; bead minilab-a32a1c, with the guard of bead
minilab-224a47 (a statement the formaliser stage flagged is held for review,
never set `trivial`). Model calls go to fake backends; Lean runs for real
against the repo's mathlib (`lean/.lake`), about 4 seconds a check.

`tests/fixtures/pilot/s1.lean` and `s8.lean` are reconstructions of the
pilot's statements from their English text; the pilot's own Lean files are in
`~/dev/sci/experiments/crossconj`, which this project never reads.
"""

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from pydantic import JsonValue

from minilab.formalise.audit import AuditResult, HiddenCondition
from minilab.formalise.formaliser import FormalOutput, formal_from_source
from minilab.formalise.vacuity import VacuityResult, WitnessCheck
from minilab.ledger import Ledger
from minilab.models.adapter import Budget
from minilab.models.backends import ModelReply, ModelUnavailable
from minilab.records import (
    Family,
    Provenance,
    Record,
    RecordStore,
    Stage,
    StageOutput,
    Status,
)
from minilab.run import MemoryPolicy, RunManifest, save_manifest
from minilab.screen.triviality import (
    TACTICS,
    ScreenResult,
    hold_reasons,
    run_triviality_stage,
    screen,
    with_proof,
)

FIXTURES = Path(__file__).parent / "fixtures" / "pilot"
S1 = (FIXTURES / "s1.lean").read_text(encoding="utf-8")
S8 = (FIXTURES / "s8.lean").read_text(encoding="utf-8")
RUN = "scr"
PAIR = ("information-theory", "optimisation")
LEAN_TIMEOUT = 240.0
TACTIC_TIMEOUT = 60.0

EASY = """import Mathlib

theorem conjecture (x : ℝ) : 0 ≤ x ^ 2 := by
  sorry
"""

# None of the five tactics proves this (checked in Lean); induction does.
NEEDS_INDUCTION = """import Mathlib

def g : ℕ → ℕ
  | 0 => 1
  | (n + 1) => g n + 1

theorem conjecture (n : ℕ) : 0 < g n := by
  sorry
"""

INDUCTION_PROOF = "induction n <;> simp [g] <;> omega"

# False (x = 2), so nothing can prove it.
FALSE = """import Mathlib

theorem conjecture (x : ℝ) (hx : 0 < x) : x ^ 2 < x := by
  sorry
"""


# --- fakes ---


class FakeBackend:
    """Replies by the kind of prompt; records every prompt it is sent."""

    def __init__(self, family: Family, replies: Mapping[str, list[str]]) -> None:
        self.family = family
        self.replies = {k: list(v) for k, v in replies.items()}
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
        return ModelReply(text=text, tokens_in=10, tokens_out=10, estimated=False, model="fake-1")


def kind_of(prompt: str) -> str:
    if "corollary of a named theorem" in prompt:
        return "corollary"
    if "Prove this Lean 4 theorem" in prompt:
        return "prove"
    return "other"


def fakes(replies: Mapping[str, list[str]]) -> dict[Family, FakeBackend]:
    return {f: FakeBackend(f, replies) for f in Family}


def lean_block(text: str) -> str:
    return f"Here is a proof.\n\n```lean\n{text}\n```\n"


def corollary_reply(corollary: bool, theorem: str = "", proof: str = "") -> str:
    return json.dumps({"corollary": corollary, "theorem": theorem, "proof": proof})


def calls(backends: Mapping[Family, FakeBackend], kind: str) -> list[tuple[Family, str]]:
    return [(f, p) for f, b in backends.items() for k, p in b.prompts if k == kind]


def make_record(local: str = "a") -> Record:
    return Record(
        id=f"{RUN}.{local}",
        run_id=RUN,
        pair=PAIR,
        mode="structured",
        statement="Every real square is non-negative.",
        objects=["x"],
        assumptions=["x is real"],
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
            budget_tokens_per_pair=1_000_000,
            families=families,
            started=datetime(2026, 11, 26, 12, 0, tzinfo=UTC),
        ),
    )
    return Ledger(store, RUN)


BUDGET = Budget(tokens_per_pair=1_000_000)

_probed: dict[str, FormalOutput] = {}


def probed(source: str) -> FormalOutput:
    """The source compiled and probed by Lean, as Claude's formalisation (cached)."""
    if source not in _probed:
        _probed[source] = formal_from_source(
            source, Family.claude, "fake-1", timeout_s=LEAN_TIMEOUT
        )
    assert _probed[source].ok, _probed[source].problems
    return _probed[source]


def run_screen(
    store: RecordStore,
    source: str,
    backends: Mapping[Family, FakeBackend],
    families: tuple[Family, ...] = tuple(Family),
) -> ScreenResult:
    ledger = make_ledger(store, families)
    record = make_record()
    store.create(record)
    return screen(
        probed(source),
        record=record,
        ledger=ledger,
        budget=BUDGET,
        backends=backends,
        tactic_timeout_s=TACTIC_TIMEOUT,
        lean_timeout_s=LEAN_TIMEOUT,
    )


# --- the screen ---


def test_tactic_screen_proves_easy_goal(store: RecordStore) -> None:
    backends = fakes({})
    result = run_screen(store, EASY, backends)
    assert result.proved
    assert result.proved_by in TACTICS
    assert result.source is not None and "sorry" not in result.source
    assert all(not b.prompts for b in backends.values()), "no model call after a tactic proof"
    # The proof is a full Lean file that checks with no sorry.
    again = formal_from_source(
        result.source, Family.claude, "m", timeout_s=LEAN_TIMEOUT, proved=True
    )
    assert again.ok, again.problems


def test_tactics_run_in_order_and_stop_at_the_first_success(store: RecordStore) -> None:
    result = run_screen(store, EASY, fakes({}))
    names = [a.screen for a in result.attempts]
    assert names == list(TACTICS[: len(names)])
    assert [a.outcome for a in result.attempts][-1] == "proved"
    assert all(a.outcome != "proved" for a in result.attempts[:-1])


def test_polyrith_is_not_a_screen() -> None:
    assert "polyrith" not in " ".join(TACTICS)


def test_local_model_proof_accepted_after_tactics_fail(store: RecordStore) -> None:
    backends = fakes({"prove": [lean_block(INDUCTION_PROOF)]})
    result = run_screen(store, NEEDS_INDUCTION, backends)
    assert result.proved
    assert result.proved_by == "local"
    assert result.proof is not None and "induction" in result.proof
    assert [f for f, _ in calls(backends, "prove")] == [Family.local]
    assert calls(backends, "corollary") == []


def test_local_model_gets_three_attempts_with_feedback(store: RecordStore) -> None:
    backends = fakes(
        {"prove": [lean_block("simp"), lean_block("omega"), lean_block(INDUCTION_PROOF)]}
    )
    result = run_screen(store, NEEDS_INDUCTION, backends)
    prompts = [p for _, p in calls(backends, "prove")]
    assert len(prompts) == 3
    assert "simp" in prompts[1] and "made no progress" in prompts[1]
    assert result.proved and result.proved_by == "local"


def test_screen_prompts_show_what_lean_printed_not_the_source(store: RecordStore) -> None:
    source = NEEDS_INDUCTION.replace("def g", "-- a note for the prover\ndef g")
    backends = fakes({"prove": [lean_block("simp")], "corollary": [corollary_reply(False)]})
    run_screen(store, source, backends)
    for _, prompt in calls(backends, "prove") + calls(backends, "corollary"):
        assert "a note for the prover" not in prompt
        assert "conjecture" in prompt


def test_proof_with_sorry_is_not_accepted(store: RecordStore) -> None:
    backends = fakes({"prove": [lean_block("sorry")], "corollary": [corollary_reply(False)]})
    result = run_screen(store, FALSE, backends)
    assert not result.proved
    local = [a for a in result.attempts if a.screen == "local"]
    assert local and all(a.outcome == "refused" for a in local)


@pytest.mark.parametrize(
    "proof",
    [
        # A real proof, then a new definition after it: the file is no longer the
        # one the formaliser stage checked.
        INDUCTION_PROOF + "\n\ndef extra : ℕ := 0",
        # Leaves the proof to close the theorem early and state another one.
        INDUCTION_PROOF + "\n\ntheorem conjecture' : True := trivial",
        "native_decide",
    ],
)
def test_model_proof_cannot_change_the_file(store: RecordStore, proof: str) -> None:
    backends = fakes({"prove": [lean_block(proof)], "corollary": [corollary_reply(False)]})
    result = run_screen(store, NEEDS_INDUCTION, backends)
    assert not result.proved
    assert any(a.outcome == "refused" for a in result.attempts if a.screen == "local")


def test_corollary_is_checked_in_lean(store: RecordStore) -> None:
    backends = fakes(
        {
            "prove": [lean_block("simp")],
            "corollary": [corollary_reply(True, "Nat.rec on g", INDUCTION_PROOF)],
        }
    )
    result = run_screen(store, NEEDS_INDUCTION, backends)
    assert result.proved
    assert result.proved_by == "corollary"
    assert result.corollary_of == "Nat.rec on g"
    checkers = [f for f, _ in calls(backends, "corollary")]
    assert checkers == [Family.codex], "the checker is never the formaliser's family"


def test_corollary_claim_that_fails_in_lean_proves_nothing(store: RecordStore) -> None:
    backends = fakes(
        {"prove": [lean_block("simp")], "corollary": [corollary_reply(True, "sq_nonneg", "simp")]}
    )
    result = run_screen(store, FALSE, backends)
    assert not result.proved
    assert result.corollary_of is None
    assert [a.outcome for a in result.attempts if a.screen == "corollary"] == ["failed"]


def test_no_local_model_in_the_run_skips_its_attempts(store: RecordStore) -> None:
    backends = fakes(
        {"prove": [lean_block(INDUCTION_PROOF)], "corollary": [corollary_reply(False)]}
    )
    result = run_screen(store, NEEDS_INDUCTION, backends, (Family.claude, Family.codex))
    assert not result.proved
    assert calls(backends, "prove") == []
    assert any("local" in note for note in result.notes)


def test_model_failure_raises(store: RecordStore) -> None:
    backends = fakes({})
    backends[Family.local].fail = True
    with pytest.raises(ModelUnavailable):
        run_screen(store, NEEDS_INDUCTION, backends)


def test_with_proof_replaces_only_the_sorry() -> None:
    proved = with_proof(NEEDS_INDUCTION, "  " + INDUCTION_PROOF)
    assert "sorry" not in proved
    assert proved.startswith(NEEDS_INDUCTION.split(":= by")[0])
    assert proved.rstrip().endswith(INDUCTION_PROOF)


def test_proved_mode_refuses_sorry_and_needs_a_proof() -> None:
    sorry_file = formal_from_source(EASY, Family.claude, "m", timeout_s=LEAN_TIMEOUT, proved=True)
    assert not sorry_file.ok
    proved = formal_from_source(
        with_proof(EASY, "nlinarith [sq_nonneg x]"),
        Family.claude,
        "m",
        timeout_s=LEAN_TIMEOUT,
        proved=True,
    )
    assert proved.ok, proved.problems
    wrong = formal_from_source(
        with_proof(FALSE, "nlinarith"), Family.claude, "m", timeout_s=LEAN_TIMEOUT, proved=True
    )
    assert not wrong.ok


# --- the hold rule (bead minilab-224a47) ---


def clean_audit() -> AuditResult:
    return AuditResult(
        hypotheses=[],
        extra=[],
        weakenings=[],
        hidden=[],
        rejected=False,
        reasons=[],
        checker=Family.codex,
        judged=True,
    )


def clean_vacuity() -> VacuityResult:
    return VacuityResult(
        contradictory=False,
        satisfied_by_witness=True,
        witness=[WitnessCheck(label="hx", state="holds")],
        unused=[],
        timed_out=False,
        notes=[],
    )


def formaliser_data(
    audit: AuditResult | None = None,
    vacuity: VacuityResult | None = None,
    *,
    rejected: bool = False,
) -> dict[str, JsonValue]:
    a = clean_audit() if audit is None else audit
    v = clean_vacuity() if vacuity is None else vacuity
    return {
        "rejected": rejected,
        "reasons": cast(list[JsonValue], ["the hypotheses contradict"] if rejected else []),
        "family": Family.claude.value,
        "model": "fake-1",
        "attempts": 1,
        "problems": [],
        "lean_statement": None if rejected else "conjecture …",
        "roundtrip": "text",
        "roundtrip_family": Family.codex.value,
        "audit": cast(dict[str, JsonValue], a.model_dump(mode="json")),
        "weakenings": cast(list[JsonValue], list(a.weakenings)),
        "vacuity": cast(dict[str, JsonValue], v.model_dump(mode="json")),
        "binders": [],
    }


def test_clean_formalisation_has_no_hold_reasons() -> None:
    assert hold_reasons(formaliser_data()) == []


@pytest.mark.parametrize(
    ("data", "word"),
    [
        (
            formaliser_data(clean_audit().model_copy(update={"weakenings": ["x is at most 1"]})),
            "weakening",
        ),
        (
            formaliser_data(
                clean_audit().model_copy(
                    update={
                        "hidden": [HiddenCondition(where="f", condition="x > 0", kind="weakening")]
                    }
                )
            ),
            "hidden",
        ),
        (
            formaliser_data(
                vacuity=clean_vacuity().model_copy(
                    update={
                        "satisfied_by_witness": False,
                        "witness": [WitnessCheck(label="hx", state="fails")],
                    }
                )
            ),
            "witness",
        ),
        (
            formaliser_data(
                vacuity=clean_vacuity().model_copy(
                    update={"satisfied_by_witness": False, "unwitnessed": ["v"]}
                )
            ),
            "witness",
        ),
        (
            formaliser_data(
                vacuity=clean_vacuity().model_copy(
                    update={"witness": [WitnessCheck(label="hx", state="undecided")]}
                )
            ),
            "witness",
        ),
        (
            formaliser_data(vacuity=clean_vacuity().model_copy(update={"witness": []})),
            "witness",
        ),
        ({"rejected": False, "audit": "garbled", "vacuity": None}, "could not be read"),
    ],
)
def test_flags_are_hold_reasons(data: dict[str, JsonValue], word: str) -> None:
    reasons = hold_reasons(data)
    assert reasons and any(word in r for r in reasons), reasons


# --- the stage ---


def screened_record(
    store: RecordStore, source: str, data: dict[str, JsonValue], local: str = "a"
) -> Record:
    """A survivor with falsifier, literature, and formaliser stages.

    An accepted formalisation records the statement Lean printed for
    `source`, as the formaliser stage does.
    """
    if data.get("rejected") is False:
        data = {**data, "lean_statement": probed(source).lean_statement}
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
        StageOutput(written_at=now, data=data, evidence=[f"formaliser/{record.id}.lean"]),
    )
    return store.load(record.id)


def run_stage(store: RecordStore, record: Record, backends: Mapping[Family, FakeBackend]) -> None:
    run_triviality_stage(
        record.id,
        ledger=make_ledger(store),
        budget=BUDGET,
        backends=backends,
        tactic_timeout_s=TACTIC_TIMEOUT,
        lean_timeout_s=LEAN_TIMEOUT,
    )


def test_unflagged_proof_sets_trivial(store: RecordStore) -> None:
    record = screened_record(store, EASY, formaliser_data())
    run_stage(store, record, fakes({}))
    after = store.load(record.id)
    assert after.status is Status.trivial
    out = after.stages[Stage.triviality]
    assert out.data["trivial"] is True
    assert out.data["held"] is False
    assert out.data["screened"] is True
    assert out.data["proved_by"] in TACTICS
    proof_file = store.root / "runs" / RUN / out.evidence[0]
    assert "sorry" not in proof_file.read_text(encoding="utf-8")


def test_flagged_proof_is_held_not_trivial(store: RecordStore) -> None:
    audit = clean_audit().model_copy(update={"weakenings": ["x is at most 1"]})
    record = screened_record(store, EASY, formaliser_data(audit))
    run_stage(store, record, fakes({}))
    after = store.load(record.id)
    assert after.status is Status.survived
    data = after.stages[Stage.triviality].data
    assert data["trivial"] is False
    assert data["held"] is True
    assert isinstance(data["proof"], str) and data["proof"]
    reasons = data["hold_reasons"]
    assert isinstance(reasons, list) and any("weakening" in str(r) for r in reasons)


def test_screen_failure_leaves_status(store: RecordStore) -> None:
    record = screened_record(store, FALSE, formaliser_data())
    backends = fakes({"prove": [lean_block("nlinarith")], "corollary": [corollary_reply(False)]})
    run_stage(store, record, backends)
    after = store.load(record.id)
    assert after.status is Status.survived
    data = after.stages[Stage.triviality].data
    assert data["trivial"] is False
    assert data["held"] is False
    assert data["proof"] is None
    assert isinstance(data["near_misses"], list)
    assert isinstance(data["attempts"], list) and data["attempts"]


def test_rejected_formalisation_is_not_screened(store: RecordStore) -> None:
    record = screened_record(store, EASY, formaliser_data(rejected=True))
    backends = fakes({})
    run_stage(store, record, backends)
    after = store.load(record.id)
    assert after.status is Status.survived
    data = after.stages[Stage.triviality].data
    assert data["screened"] is False
    assert data["trivial"] is False
    reasons = data["reasons"]
    assert isinstance(reasons, list) and any("rejected" in str(r) for r in reasons)
    assert all(not b.prompts for b in backends.values())


def test_stage_model_failure_leaves_record_unchanged(store: RecordStore) -> None:
    record = screened_record(store, NEEDS_INDUCTION, formaliser_data())
    backends = fakes({})
    backends[Family.local].fail = True
    with pytest.raises(ModelUnavailable):
        run_stage(store, record, backends)
    assert Stage.triviality not in store.load(record.id).stages


def test_stage_needs_a_formalised_survivor(store: RecordStore) -> None:
    record = make_record()
    store.create(record)
    with pytest.raises(ValueError, match="formaliser"):
        run_stage(store, record, fakes({}))


def test_stage_refuses_a_lean_file_that_changed(store: RecordStore) -> None:
    record = screened_record(store, EASY, formaliser_data())
    lean_file = store.root / "runs" / RUN / "formaliser" / f"{record.id}.lean"
    lean_file.write_text(FALSE, encoding="utf-8")
    run_stage(store, record, fakes({"corollary": [corollary_reply(False)]}))
    data = store.load(record.id).stages[Stage.triviality].data
    assert data["screened"] is False
    reasons = data["reasons"]
    assert isinstance(reasons, list) and any("statement" in str(r) for r in reasons)


# --- live: the pilot cases (design doc-gmj, "Pilot validation") ---


def _flag_runs(store: RecordStore, source: str, runs: int = 3) -> list[ScreenResult]:
    families = (Family.claude, Family.codex, Family.local)
    results: list[ScreenResult] = []
    for i in range(runs):
        ledger = make_ledger(store, families)
        record = make_record(f"live{i}")
        store.create(record)
        results.append(
            screen(
                probed(source),
                record=record,
                ledger=ledger,
                budget=BUDGET,
                tactic_timeout_s=TACTIC_TIMEOUT,
                lean_timeout_s=LEAN_TIMEOUT,
            )
        )
    for result in results:
        print(result.model_dump_json(include={"proved", "proved_by", "corollary_of", "attempts"}))
    return results


@pytest.mark.live
def test_s1_flagged_trivial(store: RecordStore) -> None:
    results = _flag_runs(store, S1)
    assert sum(r.proved for r in results) >= 2


@pytest.mark.live
def test_s8_flagged_trivial(store: RecordStore) -> None:
    results = _flag_runs(store, S8)
    assert sum(r.proved for r in results) >= 2
