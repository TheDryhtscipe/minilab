"""Tests for the formaliser, vacuity linter, round-trip check, and hypothesis audit.

Build plan doc-5wh, Task 11. Model calls go to fake backends. Lean runs for
real against the repo's mathlib (`lean/.lake`), so each Lean check takes a
few seconds; the probed statements are cached per module.

`tests/fixtures/pilot/s8.lean` is a reconstruction of the pilot's S8, written
from its English statement and run notes. The pilot's own Lean file is in
`~/dev/sci/experiments/crossconj`, which this project never reads (bead
minilab-ec2916).
"""

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

import pytest

from minilab.encoding.api import Instance
from minilab.formalise.audit import AuditResult, audit, script_matches
from minilab.formalise.formaliser import (
    FormalOutput,
    formal_from_source,
    formalise,
    formaliser_family,
    source_problems,
    strip_comments,
)
from minilab.formalise.roundtrip import roundtrip
from minilab.formalise.stage import run_formaliser_stage
from minilab.formalise.vacuity import VacuityResult, vacuity
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

FIXTURES = Path(__file__).parent / "fixtures"
S8 = (FIXTURES / "pilot" / "s8.lean").read_text(encoding="utf-8")
RUN = "fml"
PAIR = ("information-theory", "optimisation")
LEAN_TIMEOUT = 240.0

S8_ASSUMPTIONS = [
    "W is a discrete memoryless channel: a matrix with m input rows and n output columns, "
    "each row a probability distribution",
    "p is a capacity-achieving input distribution for W",
]

SIMPLE = """import Mathlib

theorem conjecture (x : ℝ) (hx : 0 ≤ x) : 0 ≤ x ^ 3 := by
  sorry
"""

CONTRADICTORY = """import Mathlib

theorem conjecture (x : ℝ) (hpos : 0 < x) (hneg : x < 0) : x ^ 2 = 5 := by
  sorry
"""

UNUSED = """import Mathlib

theorem conjecture (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hc : (1 : ℝ) < 2) : 0 ≤ x ^ 3 := by
  sorry
"""

WITNESS = """import Mathlib

def IsProb (v : Fin 3 → ℝ) : Prop := (∀ i, 0 ≤ v i) ∧ ∑ i, v i = 1

theorem conjecture (s : ℝ) (v : Fin 3 → ℝ) (hs : 1 < s) (hv : IsProb v) (hbig : 10 < s) :
    0 ≤ s * v 0 := by
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
    if "Write this conjecture as a Lean 4" in prompt:
        return "formalise"
    if "Translate the Lean theorem" in prompt:
        return "roundtrip"
    if "hypothesis audit" in prompt:
        return "audit"
    return "other"


def fakes(replies: Mapping[str, list[str]]) -> dict[Family, FakeBackend]:
    return {f: FakeBackend(f, replies) for f in Family}


def lean_block(source: str) -> str:
    return f"Here it is.\n\n```lean\n{source}```\n"


def audit_reply(kinds: Mapping[str, tuple[str, str | None]], hidden: list[str] | None = None) -> str:
    return json.dumps(
        {
            "hypotheses": [
                {"label": label, "kind": kind, "counterpart": counterpart, "reason": "r"}
                for label, (kind, counterpart) in kinds.items()
            ],
            "hidden": [
                {"definition": d, "condition": "c", "kind": "weakening", "reason": "r"}
                for d in (hidden or [])
            ],
        }
    )


def make_record(
    local: str = "a",
    *,
    assumptions: list[str] | None = None,
    statement: str = "Every DMC has a capacity-achieving input with support at most rank(W).",
    family: Family = Family.gemini,
    status: Status = Status.proposed,
) -> Record:
    return Record(
        id=f"{RUN}.{local}",
        run_id=RUN,
        pair=PAIR,
        mode="structured",
        statement=statement,
        objects=["channel W", "capacity-achieving input p"],
        assumptions=S8_ASSUMPTIONS if assumptions is None else assumptions,
        domain_of_validity="finite input and output alphabets",
        instance_family="channels with 2 to 4 inputs and outputs",
        provenance=Provenance(family=family, model="m", prompt_hash="h"),
        status=status,
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
            started=datetime(2026, 11, 25, 12, 0, tzinfo=UTC),
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
    return _probed[source]


# --- formaliser ---


def test_probe_lists_binders_and_conclusion() -> None:
    formal = probed(S8)
    assert formal.ok, formal.problems
    names = [b.name for b in formal.binders]
    assert names == ["m", "n", "W", "hW", "p", "hp", "hKKT"]
    assert [h.label for h in formal.hypotheses] == ["hW", "hp", "hKKT"]
    kkt = next(b for b in formal.binders if b.name == "hKKT")
    # hKKT mentions W and p.
    assert set(kkt.deps) == {names.index("W"), names.index("p"), 0, 1}
    assert "IsCapacityAchieving" in formal.conclusion
    assert {"IsChannel", "IsCapacityAchieving", "rowKL"} <= set(formal.definitions)
    assert formal.lean_statement.startswith("theorem conjecture")
    assert "sorry" not in formal.lean_statement


@pytest.mark.parametrize(
    ("bad", "problem"),
    [
        ("axiom cheat : False\n", "axiom"),
        ("macro \"easy\" : tactic => `(tactic| sorry)\n", "macro"),
        ("set_option maxHeartbeats 0\n", "set_option"),
        ("theorem minilab_contra : True := trivial\n", "minilab"),
        ("theorem helper : 1 = 1 := by sorry\n", "sorry"),
    ],
)
def test_forbidden_constructs_refused(bad: str, problem: str) -> None:
    source = SIMPLE.replace("theorem conjecture", bad + "\ntheorem conjecture")
    problems = source_problems(source)
    assert any(problem in p for p in problems), problems


def test_comments_and_strings_do_not_count_as_constructs() -> None:
    source = SIMPLE.replace(
        "theorem conjecture", '/-- an axiom in a comment -/\n-- macro here\ntheorem conjecture'
    )
    assert source_problems(source) == []
    assert "axiom" not in strip_comments(source)


def test_missing_conjecture_is_a_problem() -> None:
    formal = probed(SIMPLE.replace("theorem conjecture", "theorem other"))
    assert not formal.ok
    assert any("conjecture" in p for p in formal.problems)


def test_formalise_retries_with_feedback_then_succeeds(store: RecordStore) -> None:
    ledger = make_ledger(store)
    record = make_record(assumptions=["0 ≤ x"])
    store.create(record)
    broken = SIMPLE.replace("0 ≤ x ^ 3", "0 ≤ x ^ 3 + undefined_thing")
    backends = fakes({"formalise": [lean_block(broken), lean_block(SIMPLE)]})
    formal = formalise(
        record, ledger=ledger, budget=BUDGET, backends=backends, timeout_s=LEAN_TIMEOUT
    )
    assert formal.ok, formal.problems
    assert formal.attempts == 2
    assert formal.family is Family.claude
    prompts = [p for k, p in backends[Family.claude].prompts if k == "formalise"]
    assert len(prompts) == 2
    assert "undefined_thing" in prompts[1]  # the second prompt shows the refused file


def test_formaliser_family_is_never_gemini_or_local() -> None:
    assert formaliser_family([Family.gemini, Family.codex, Family.local]) is Family.codex
    assert formaliser_family(list(Family)) is Family.claude


# --- vacuity ---


def test_contradictory_hypotheses_rejected() -> None:
    result = vacuity(probed(CONTRADICTORY), None, timeout_s=LEAN_TIMEOUT)
    assert result.contradictory
    assert result.rejected
    # A consistent statement is not called contradictory.
    fine = vacuity(probed(SIMPLE), None, timeout_s=LEAN_TIMEOUT)
    assert not fine.contradictory
    assert not fine.rejected


def test_unused_hypothesis_flagged() -> None:
    result = vacuity(probed(UNUSED), None, timeout_s=LEAN_TIMEOUT)
    # y appears only in hy, and hc mentions no variable: none bear on the conclusion.
    assert sorted(result.unused) == ["hc", "hy", "y"]
    assert result.rejected
    assert vacuity(probed(SIMPLE), None, timeout_s=LEAN_TIMEOUT).unused == []


def test_witness_instantiated_in_lean() -> None:
    formal = probed(WITNESS)
    witness = Instance(
        values={"s": Fraction(5, 2), "v": ["0.5", "0.25", "1/4"]}, precision_digits=30
    )
    result = vacuity(formal, witness, timeout_s=LEAN_TIMEOUT)
    states = {w.label: w.state for w in result.witness}
    assert states == {"hs": "holds", "hv": "holds", "hbig": "fails"}
    assert not result.satisfied_by_witness
    good = Instance(values={"s": "12", "v": ["0.5", "0.25", "0.25"]}, precision_digits=30)
    result = vacuity(formal, good, timeout_s=LEAN_TIMEOUT)
    assert {w.state for w in result.witness} == {"holds"}
    assert result.satisfied_by_witness
    assert not result.contradictory


def test_witness_with_unknown_names_is_not_instantiated() -> None:
    witness = Instance(values={"other": "1"}, precision_digits=30)
    result = vacuity(probed(WITNESS), witness, timeout_s=LEAN_TIMEOUT)
    assert {w.state for w in result.witness} == {"not_instantiated"}
    assert not result.satisfied_by_witness


def test_lean_timeout_in_vacuity_is_not_a_verdict() -> None:
    result = vacuity(probed(CONTRADICTORY), None, timeout_s=0.5)
    assert result.timed_out
    assert not result.contradictory
    assert not result.rejected


# --- round trip ---


@pytest.mark.parametrize("author", [Family.claude, Family.codex])
def test_roundtrip_family_differs_from_author(store: RecordStore, author: Family) -> None:
    ledger = make_ledger(store)
    record = make_record(statement="The secret original wording of S8.")
    store.create(record)
    formal = probed(S8).model_copy(update={"family": author})
    backends = fakes({"roundtrip": ["For every channel W ..."]})
    trip = roundtrip(formal, author, record=record, ledger=ledger, budget=BUDGET, backends=backends)
    assert trip.family is not author
    assert trip.family is not Family.gemini
    assert trip.english == "For every channel W ..."
    assert trip.original == record.statement
    called = [f for f, b in backends.items() if b.prompts]
    assert called == [trip.family]
    prompt = backends[trip.family].prompts[0][1]
    # The translator never sees the original, nor the fixture's comments.
    assert record.statement not in prompt
    assert "reconstruction" not in prompt
    assert "theorem conjecture" in prompt


# --- audit ---


def test_audit_flags_s8_kkt_hypothesis(store: RecordStore) -> None:
    ledger = make_ledger(store)
    record = make_record()
    store.create(record)
    formal = probed(S8)
    # No assumption in the record states KKT, so the script cannot match it.
    assert "hKKT" not in script_matches(record, formal)
    verdicts = {
        "hW": ("matches", "assumption 1"),
        "hp": ("matches", "assumption 2"),
        "hKKT": ("begs_question", None),
    }
    backends = fakes({"audit": [audit_reply(verdicts)]})
    result = audit(record, formal, ledger=ledger, budget=BUDGET, backends=backends)
    assert result.extra == ["hKKT"]
    assert result.rejected
    prompt = next(p for _, p in backends[result.checker or Family.claude].prompts)
    assert "hKKT" in prompt


def test_audit_fails_closed_when_checker_skips_a_hypothesis(store: RecordStore) -> None:
    ledger = make_ledger(store)
    record = make_record()
    store.create(record)
    # The checker says nothing about hKKT: it stays listed, unjudged, and the
    # formalisation is rejected.
    reply = audit_reply({"hW": ("matches", "assumption 1"), "hp": ("matches", "assumption 2")})
    result = audit(record, probed(S8), ledger=ledger, budget=BUDGET, backends=fakes({"audit": [reply]}))
    assert result.extra == ["hKKT"]
    assert result.rejected
    kkt = next(h for h in result.hypotheses if h.label == "hKKT")
    assert kkt.kind == "unjudged"


def test_audit_unreadable_reply_rejects(store: RecordStore) -> None:
    ledger = make_ledger(store)
    record = make_record()
    store.create(record)
    result = audit(
        record, probed(S8), ledger=ledger, budget=BUDGET, backends=fakes({"audit": ["no json"]})
    )
    assert result.rejected
    assert sorted(result.extra) == ["hKKT", "hW", "hp"]


def test_audit_weakening_is_stated_not_rejected(store: RecordStore) -> None:
    ledger = make_ledger(store)
    record = make_record()
    store.create(record)
    verdicts = {
        "hW": ("matches", "assumption 1"),
        "hp": ("matches", "assumption 2"),
        "hKKT": ("weakening", None),
    }
    result = audit(
        record,
        probed(S8),
        ledger=ledger,
        budget=BUDGET,
        backends=fakes({"audit": [audit_reply(verdicts, hidden=["rowKL"])]}),
    )
    assert not result.rejected
    assert result.extra == ["hKKT"]
    assert len(result.weakenings) == 2  # hKKT, and the condition hidden in rowKL
    assert any("hKKT" in w for w in result.weakenings)


def test_script_matches_stated_text_and_checker_skips_it(store: RecordStore) -> None:
    ledger = make_ledger(store)
    record = make_record(assumptions=["0 ≤ x"])
    store.create(record)
    formal = probed(SIMPLE)
    assert script_matches(record, formal) == {"hx": "assumption 1"}
    backends = fakes({"audit": ["unused"]})
    result = audit(record, formal, ledger=ledger, budget=BUDGET, backends=backends)
    assert not result.rejected
    assert result.extra == []
    # Nothing was left for the checker, so no call was made.
    assert all(not b.prompts for b in backends.values())


def test_witness_failure_overrides_script_match(store: RecordStore) -> None:
    ledger = make_ledger(store)
    record = make_record(assumptions=["1 < s", "10 < s"])
    store.create(record)
    formal = probed(WITNESS)
    vac = VacuityResult.model_validate(
        {
            "contradictory": False,
            "satisfied_by_witness": False,
            "witness": [
                {"label": "hs", "state": "holds"},
                {"label": "hv", "state": "holds"},
                {"label": "hbig", "state": "fails"},
            ],
            "unused": [],
            "timed_out": False,
            "notes": [],
        }
    )
    verdicts = {"hv": ("matches", "domain"), "hbig": ("weakening", None)}
    backends = fakes({"audit": [audit_reply(verdicts)]})
    result: AuditResult = audit(
        record, formal, ledger=ledger, budget=BUDGET, backends=backends, vacuity=vac
    )
    assert "hbig" in result.extra
    prompt = next(p for _, p in backends[result.checker or Family.claude].prompts)
    assert "hbig" in prompt and "fails" in prompt


@pytest.mark.live
def test_audit_flags_s8_kkt_hypothesis_live(store: RecordStore) -> None:
    ledger = make_ledger(store, (Family.claude, Family.codex))
    record = make_record()
    store.create(record)
    result = audit(record, probed(S8), ledger=ledger, budget=BUDGET)
    assert "hKKT" in result.extra


# --- the stage ---


def survived_record(store: RecordStore, local: str = "a") -> Record:
    record = make_record(local, assumptions=["0 ≤ x"])
    store.create(record)
    now = datetime(2026, 11, 25, 13, 0, tzinfo=UTC)
    coverage = {
        "space": "x in [0, 10]",
        "dimensions": 1,
        "starts": 4,
        "worst_margin": "0",
        "worst_point": {"values": {"x": "0.5"}, "precision_digits": 30},
        "uncovered": [],
    }
    store.write_stage(
        record.id,
        Stage.falsifier,
        StageOutput(written_at=now, data={"coverage": coverage}),
        status=Status.survived,
    )
    store.write_stage(record.id, Stage.literature, StageOutput(written_at=now, notes="none"))
    return store.load(record.id)


def test_stage_writes_roundtrip_and_audit(store: RecordStore) -> None:
    ledger = make_ledger(store)
    record = survived_record(store)
    backends = fakes({"formalise": [lean_block(SIMPLE)], "roundtrip": ["For x ≥ 0, x³ ≥ 0."]})
    run_formaliser_stage(
        record.id, ledger=ledger, budget=BUDGET, backends=backends, lean_timeout_s=LEAN_TIMEOUT
    )
    after = store.load(record.id)
    assert after.status is Status.survived
    out = after.stages[Stage.formaliser]
    assert out.data["roundtrip"] == "For x ≥ 0, x³ ≥ 0."
    assert isinstance(out.data["audit"], dict)
    assert out.data["rejected"] is False
    assert isinstance(out.data["lean_statement"], str)
    vac = out.data["vacuity"]
    assert isinstance(vac, dict) and vac["satisfied_by_witness"] is True
    lean_file = store.root / "runs" / RUN / out.evidence[0]
    assert lean_file.read_text(encoding="utf-8") == SIMPLE


def test_stage_records_rejection_without_changing_status(store: RecordStore) -> None:
    ledger = make_ledger(store)
    record = survived_record(store)
    audit_ok = audit_reply({"hpos": ("matches", "assumption 1"), "hneg": ("weakening", None)})
    backends = fakes(
        {"formalise": [lean_block(CONTRADICTORY)], "roundtrip": ["text"], "audit": [audit_ok]}
    )
    run_formaliser_stage(
        record.id, ledger=ledger, budget=BUDGET, backends=backends, lean_timeout_s=LEAN_TIMEOUT
    )
    after = store.load(record.id)
    assert after.status is Status.survived
    data = after.stages[Stage.formaliser].data
    assert data["rejected"] is True
    reasons = data["reasons"]
    assert isinstance(reasons, list) and any("contradict" in str(r) for r in reasons)


def test_model_failure_leaves_record_unchanged(store: RecordStore) -> None:
    ledger = make_ledger(store)
    record = survived_record(store)
    path = store.root / "runs" / RUN / "records" / f"{record.id}.json"
    before = path.read_bytes()
    backends = fakes({"formalise": [lean_block(SIMPLE)]})
    for backend in backends.values():
        backend.fail = True
    with pytest.raises(ModelUnavailable):
        run_formaliser_stage(record.id, ledger=ledger, budget=BUDGET, backends=backends)
    assert path.read_bytes() == before


def test_stage_needs_a_survivor_with_literature(store: RecordStore) -> None:
    ledger = make_ledger(store)
    record = make_record()
    store.create(record)
    with pytest.raises(ValueError, match="survived"):
        run_formaliser_stage(record.id, ledger=ledger, budget=BUDGET, backends=fakes({}))
