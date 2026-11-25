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
from pydantic import JsonValue

from minilab.encoding.api import Instance
from minilab.formalise.audit import AuditResult, audit, script_matches
from minilab.formalise.formaliser import (
    CommandCheck,
    FormalOutput,
    LeanTimeout,
    NoFormaliser,
    command_check,
    formal_from_source,
    formalise,
    formaliser_family,
    lean_statement,
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


def audit_reply(
    kinds: Mapping[str, tuple[str, str | None]], hidden: list[str] | None = None
) -> str:
    return json.dumps(
        {
            "hypotheses": [
                {"label": label, "kind": kind, "counterpart": counterpart, "reason": "r"}
                for label, (kind, counterpart) in kinds.items()
            ],
            "hidden": [
                {"where": d, "condition": "c", "kind": "weakening", "reason": "r"}
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
        ('macro "easy" : tactic => `(tactic| sorry)\n', "macro"),
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
        "theorem conjecture", "/-- an axiom in a comment -/\n-- macro here\ntheorem conjecture"
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
    result = audit(
        record, probed(S8), ledger=ledger, budget=BUDGET, backends=fakes({"audit": [reply]})
    )
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
    backends = fakes({"audit": [audit_reply({})]})
    result = audit(record, formal, ledger=ledger, budget=BUDGET, backends=backends)
    assert not result.rejected
    assert result.extra == []
    hx = next(h for h in result.hypotheses if h.label == "hx")
    assert hx.by == "script"
    # The checker is still called (a condition can hide outside the
    # hypotheses), but it is not asked to judge hx.
    prompt = next(p for _, p in backends[result.checker].prompts)
    judged_part = prompt.split("Judge each of these hypotheses")[1].split("Its variables")[0]
    assert "hx" not in judged_part


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
    coverage: dict[str, JsonValue] = {
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
    backends = fakes(
        {
            "formalise": [lean_block(SIMPLE)],
            "roundtrip": ["For x ≥ 0, x³ ≥ 0."],
            "audit": [audit_reply({})],
        }
    )
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


# --- review findings (bead minilab-ec2916) ---

ODD_NAMES = """import Mathlib
def «a"» : Nat := 1
axiom bad : False
#eval (1 : Nat)
def «b"» : Nat := 2
theorem conjecture (x : ℝ) (h : 0 < x) : 0 < x ^ 2 := sorry
"""


def test_quoted_names_do_not_open_strings() -> None:
    # A `"` inside «» once opened a fake string that hid the axiom and #eval.
    problems = source_problems(ODD_NAMES)
    assert any("axiom" in p for p in problems), problems
    assert any("#eval" in p for p in problems), problems


@pytest.mark.parametrize(
    ("bad", "word"),
    [
        ("open Lean in\n#eval! (pure () : IO Unit)\n", "#eval!"),
        ('def helper : 1 = 1 := by\n  run_tac Lean.logInfo "ran"\n  rfl\n', "run_tac"),
    ],
)
def test_word_list_catches_eval_bang_and_run_tac(bad: str, word: str) -> None:
    source = SIMPLE.replace("theorem conjecture", bad + "\ntheorem conjecture")
    assert any(word in p for p in source_problems(source))


@pytest.mark.parametrize(
    ("bad", "problem"),
    [
        ("instance : Inhabited ℝ := ⟨0⟩\n", "instance"),
        ("@[simp] def q : ℕ := 1\n", "attributes"),
        ("variable (n : ℕ)\n", "variable"),
        ("#check (1 : ℕ)\n", "check"),
        ("def t : ℕ := Id.run do\n  return 1\n", "do"),
        # Mathlib's `run_conv` takes a `do` block and runs it; no word list knows it.
        (
            'theorem helper : (1 : ℕ) = 1 := by\n  conv =>\n    run_conv Lean.logInfo "ran code"\n',
            "do",
        ),
        ("namespace Open\nend Open\nsection\n", "end"),
        ("def q : ℕ := eval% 3\n", "eval_expr"),
        ("theorem helper : True := by aesop (add safe trivial)\n", "Aesop rule clause"),
        ("theorem helper : 3 = 3 := by decide +native\n", "configuration"),
        ("theorem helper : 3 = 3 := by decide (config := {})\n", "configuration"),
    ],
)
def test_command_allowlist_refuses_before_compiling(bad: str, problem: str) -> None:
    source = SIMPLE.replace("theorem conjecture", bad + "\ntheorem conjecture")
    assert source_problems(source) == []
    formal = formal_from_source(source, Family.claude, "m", timeout_s=LEAN_TIMEOUT)
    assert not formal.ok
    assert any(problem in p for p in formal.problems), formal.problems


ORDINARY = """import Mathlib

noncomputable section

open Real

namespace Foo
def g (x : ℝ) : ℝ := π * x
end Foo

structure S where
  a : ℝ
  deriving Inhabited

def s : String := "axiom #eval"

def f : ℕ → ℕ
  | 0 => 1
  | n + 1 => n

end

open Finset in
theorem conjecture {m : ℕ} (x y : ℝ) [Fact (1 < m)] ⦃z : ℝ⦄ (h : 0 < Foo.g x) :
    ∀ ε > 0, let t := ε; t * x > -1 := by
  sorry
"""


def test_ordinary_file_passes_and_only_header_binders_count() -> None:
    formal = probed(ORDINARY)
    assert formal.ok, formal.problems
    # ε and `ε > 0` belong to the conclusion, not to the header.
    assert [b.name for b in formal.binders] == ["m", "x", "y", "", "z", "h"]
    # The unnamed instance binder has no name (Lean's `inst✝` is not one).
    assert formal.binders[3].label == "#3"
    assert formal.conclusion.startswith("∀ ε > 0")
    # Only the definitions the source wrote: no matchers, no deriving helpers.
    assert sorted(formal.definitions) == ["Foo.g", "f", "s"]
    assert next(b for b in formal.binders if b.name == "h").defs == ["Foo.g"]
    assert lean_statement(ORDINARY).endswith("t * x > -1")


def test_statement_span_skips_let_in_the_statement() -> None:
    source = SIMPLE.replace("0 ≤ x ^ 3 := by", "let y := x; 0 ≤ y ^ 3 := by")
    assert lean_statement(source).endswith("0 ≤ y ^ 3")


def test_misspelt_name_is_an_error_not_a_binder() -> None:
    formal = probed(SIMPLE.replace("(hx : 0 ≤ x)", "(hx : 0 ≤ xx)"))
    assert not formal.ok
    assert all(b.name != "xx" for b in formal.binders)


def test_duplicate_binder_names_get_distinct_labels() -> None:
    source = """import Mathlib

theorem conjecture (x : ℝ) (h_2 : 0 ≤ x) (h : x ^ 2 ≤ 7) (h : 0 ≤ x) : 0 ≤ x ^ 3 := by
  sorry
"""
    formal = probed(source)
    assert formal.ok, formal.problems
    labels = [b.label for b in formal.binders]
    assert len(set(labels)) == len(labels), labels
    assert labels[1] == "h_2"


MANY_DEFS = (
    "import Mathlib\n\n"
    + "".join(f"def someRatherLongDefinitionName{i} : ℕ := {i}\n" for i in range(120))
    + "\ntheorem conjecture (x : ℝ) (hx : 0 ≤ x) : 0 ≤ x ^ 3 := by\n  sorry\n"
)


def test_many_definitions_are_all_read() -> None:
    formal = probed(MANY_DEFS)
    assert formal.ok, formal.problems
    assert len(formal.definitions) == 120


LOSS = """import Mathlib

def loss (x : ℝ) : ℝ := x ^ 2

def risk (x : ℝ) : ℝ := loss x + 1

theorem conjecture (L : ℝ) (hL : ∀ x, |loss x| ≤ L) (h0 : ∀ x, 0 ≤ loss x) :
    ∀ x, 0 < risk x := by
  sorry
"""


def test_hypotheses_about_the_conclusions_definitions_are_used() -> None:
    # hL is an ∃-L style bound and h0 a closed fact, both about `loss`, which
    # `risk` in the conclusion uses: neither is unused.
    result = vacuity(probed(LOSS), None, timeout_s=LEAN_TIMEOUT)
    assert result.unused == []


FORGE = """import Mathlib
open Lean Elab Command in
#eval! show CommandElabM Unit from do
  let src := (← getFileMap).source
  if (src.splitOn ("MINILAB_" ++ "VAC_")).length > 1 then
    let fname ← getFileName
    let mk := fun (s : String) (v : Expr) => liftCoreM (addDecl (Declaration.defnDecl
      (mkDefinitionValEx (Name.mkSimple s) [] (mkSort Level.zero) v .abbrev .safe [])))
    mk ("Minilab" ++ "ContraGoal") (mkConst ``False)
    for i in [0:3] do mk (s!"Minilab" ++ s!"Witness_{i}") (mkConst ``True)
    let say := fun (t : String) => do
      let m : Message := { fileName := fname, pos := ⟨20, 0⟩, data := t, severity := .information }
      modify fun s => { s with messages := s.messages.add m }
    say "MINILAB_VAC_PROP {\\"index\\":1,\\"instantiated\\":true}"
    say "MINILAB_VAC_PROP {\\"index\\":2,\\"instantiated\\":true}"
    say "MINILAB_VAC_READY null"
theorem conjecture (x : ℝ) (h1 : x < 0) (h2 : 0 < x) : x ^ 2 < 0 := sorry
"""


def test_forged_vacuity_markers_are_ignored() -> None:
    # The formaliser refuses this file; here it is forced through, as if a
    # guard had failed, to show the linter's own markers and names hold.
    assert source_problems(FORGE)
    clean = FORGE.split("\n", 1)[0] + "\n" + FORGE.splitlines()[-1] + "\n"
    formal = probed(clean).model_copy(update={"source": FORGE})
    result = vacuity(formal, None, timeout_s=LEAN_TIMEOUT)
    assert result.checked
    assert result.contradictory


def test_out_of_range_marker_indices_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    import re

    from minilab.formalise import vacuity as vacuity_module
    from minilab.lean.runner import LeanMessage, LeanResult

    def fake_check_lean(
        text: str, timeout_s: float, *, options: Mapping[str, str] | None = None
    ) -> LeanResult:
        found = re.search(r"MINILAB_VAC_[0-9a-f]+_", text)
        marker = found.group() if found is not None else "MINILAB_VAC_"
        far = text.count("\n")

        def info(body: str) -> LeanMessage:
            return LeanMessage(severity="information", line=far, column=0, text=marker + body)

        return LeanResult(
            ok=True,
            errors=[],
            has_sorry=True,
            axioms=[],
            timed_out=False,
            axioms_by_theorem={},
            messages=[
                info('SUBST {"index":99,"ok":false}'),
                info('PROP {"index":0,"instantiated":true}'),
                info("READY null"),
            ],
        )

    monkeypatch.setattr(vacuity_module, "check_lean", fake_check_lean)
    witness = Instance(values={"x": "1"}, precision_digits=30)
    result = vacuity(probed(SIMPLE), witness, timeout_s=LEAN_TIMEOUT)
    assert not result.checked
    assert result.rejected


def test_fraction_into_a_natural_number_is_named() -> None:
    source = """import Mathlib

theorem conjecture (n : ℕ) (hn : 2 ≤ n) : 0 < n := by
  sorry
"""
    witness = Instance(values={"n": "5/2"}, precision_digits=30)
    result = vacuity(probed(source), witness, timeout_s=LEAN_TIMEOUT)
    assert any("fraction" in note for note in result.notes), result.notes


@pytest.mark.parametrize(
    ("source", "shown"),
    [
        (
            "import Mathlib\ntheorem conjecture (x : {t : ℝ // 1 < t}) (n : ℕ+) :\n"
            "    (1 : ℝ) < (x : ℝ) ^ (n : ℕ) := sorry\n",
            "1 < t",
        ),
        (
            "import Mathlib\ntheorem conjecture (x : ℝ) : (0 < x → x ^ 3 > 0) ∧ True := sorry\n",
            "0 < x →",
        ),
    ],
)
def test_audit_sees_variable_types_and_the_conclusion(
    store: RecordStore, source: str, shown: str
) -> None:
    ledger = make_ledger(store)
    record = make_record(assumptions=[])
    store.create(record)
    backends = fakes({"audit": ["no json"]})
    result = audit(record, probed(source), ledger=ledger, budget=BUDGET, backends=backends)
    prompts = [p for b in backends.values() for _, p in b.prompts]
    assert len(prompts) == 1
    assert shown in prompts[0]
    assert result.rejected


def test_script_never_matches_a_hypothesis_about_a_definition(store: RecordStore) -> None:
    source = """import Mathlib

def smooth (x : ℝ) : Prop := True

theorem conjecture (x : ℝ) (hs : smooth x) : 0 ≤ x ^ 2 := by
  sorry
"""
    record = make_record(assumptions=["smooth x"])
    assert script_matches(record, probed(source)) == {}


def test_hidden_condition_that_matches_is_not_a_rejection(store: RecordStore) -> None:
    ledger = make_ledger(store)
    record = make_record()
    store.create(record)
    reply = json.dumps(
        {
            "hypotheses": [
                {"label": "hW", "kind": "matches", "counterpart": "assumption 1"},
                {"label": "hp", "kind": "matches", "counterpart": "assumption 2"},
                {"label": "hKKT", "kind": "weakening"},
            ],
            "hidden": [{"where": "IsDist", "condition": "sums to 1", "kind": "matches"}],
        }
    )
    result = audit(
        record, probed(S8), ledger=ledger, budget=BUDGET, backends=fakes({"audit": [reply]})
    )
    assert not result.rejected, result.reasons
    assert result.hidden == []


def test_formalise_refuses_a_generator_family(store: RecordStore) -> None:
    ledger = make_ledger(store)
    record = make_record()
    store.create(record)
    backends = fakes({"formalise": [lean_block(SIMPLE)]})
    with pytest.raises(NoFormaliser):
        formalise(record, ledger=ledger, budget=BUDGET, backends=backends, family=Family.gemini)
    assert all(not b.prompts for b in backends.values())


def test_vacuity_that_did_not_run_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    from minilab.formalise import vacuity as vacuity_module
    from minilab.lean.runner import LeanResult

    def silent(
        text: str, timeout_s: float, *, options: Mapping[str, str] | None = None
    ) -> LeanResult:
        return LeanResult(
            ok=False,
            errors=["1:0: boom"],
            has_sorry=True,
            axioms=[],
            timed_out=False,
            axioms_by_theorem={},
            messages=[],
        )

    monkeypatch.setattr(vacuity_module, "check_lean", silent)
    result = vacuity(probed(SIMPLE), None, timeout_s=LEAN_TIMEOUT)
    assert not result.checked
    assert any("did not run" in r for r in result.reasons)


def test_lean_timeout_in_formalise_leaves_record_unchanged(store: RecordStore) -> None:
    ledger = make_ledger(store)
    record = survived_record(store)
    path = store.root / "runs" / RUN / "records" / f"{record.id}.json"
    before = path.read_bytes()
    backends = fakes({"formalise": [lean_block(SIMPLE)]})
    with pytest.raises(LeanTimeout):
        run_formaliser_stage(
            record.id, ledger=ledger, budget=BUDGET, backends=backends, lean_timeout_s=0.5
        )
    assert path.read_bytes() == before


# --- model code that runs while Lean compiles (second adversarial review) ---

ADVERSARIAL = FIXTURES / "adversarial"


def _past_the_allowlist(monkeypatch: pytest.MonkeyPatch, name: str) -> FormalOutput:
    """The fixture compiled and probed as if the command allowlist had a hole.

    Each fixture hides an IO action in an Aesop rule clause. When it runs, it
    reads the check file through /proc/self/cmdline, takes the run's marker
    from it, prints forged results to stdout, and exits. The allowlist now
    refuses the clause, so it is switched off here: the test is that no model
    code runs at all, whatever the allowlist misses.
    """
    import minilab.formalise.formaliser as formaliser

    source = (ADVERSARIAL / name).read_text(encoding="utf-8")
    assert source_problems(source) == []

    def no_problems(_source: str, *, timeout_s: float) -> CommandCheck:
        return CommandCheck([], 2, None)

    monkeypatch.setattr(formaliser, "command_check", no_problems)
    return formal_from_source(source, Family.claude, "m", timeout_s=LEAN_TIMEOUT)


def test_model_code_cannot_forge_the_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    formal = _past_the_allowlist(monkeypatch, "forge_probe.lean")
    # The forged probe claims `(h : 0 ≤ n)` and `n + 0 = n`; Lean has `n < 0` and `n = 5`.
    assert [b.type for b in formal.binders] == ["ℕ", "n < 0"], formal
    assert formal.conclusion == "n = 5"
    assert not formal.ok
    assert any("aesop" in p for p in formal.problems), formal.problems


def test_model_code_cannot_forge_the_vacuity_check(monkeypatch: pytest.MonkeyPatch) -> None:
    formal = _past_the_allowlist(monkeypatch, "forge_vacuity.lean")
    assert [b.type for b in formal.binders] == ["ℕ", "n < 0"], formal
    # Forced through, as if every earlier guard had passed it.
    forced = formal.model_copy(update={"ok": True, "problems": []})
    witness = Instance(values={"n": "3"}, precision_digits=30)
    result = vacuity(forced, witness, timeout_s=LEAN_TIMEOUT)
    # The forged run claims the witness n = 3 satisfies `n < 0`.
    assert result.checked
    assert [(c.label, c.state) for c in result.witness] == [("h", "fails")], result
    assert not result.satisfied_by_witness


# --- result forging without running code (third adversarial review) ---


def _past_the_text_gate(monkeypatch: pytest.MonkeyPatch, source: str) -> FormalOutput:
    """`source` checked by Lean alone: the text gate is switched off.

    The text gate is a second guard. These tests show that Lean's own checks
    hold when the text gate misses something.
    """
    import minilab.formalise.formaliser as formaliser

    def no_problems(_source: str) -> list[str]:
        return []

    monkeypatch.setattr(formaliser, "source_problems", no_problems)
    return formal_from_source(source, Family.claude, "m", timeout_s=LEAN_TIMEOUT)


def test_auto_bound_variables_stay_off_when_the_import_line_is_shared() -> None:
    # The options used to be appended to the last import line as text; here
    # that line also holds the theorem, so `n` was bound automatically and
    # the hypothesis `n < 0` was hidden in the conclusion.
    source = "import Mathlib theorem conjecture (h0 : 0 ≤ n) (h : n < 0) : n = 5 := sorry\n"
    assert source_problems(source) == []
    formal = formal_from_source(source, Family.claude, "m", timeout_s=LEAN_TIMEOUT)
    assert not formal.ok
    assert all(b.name != "n" for b in formal.binders), formal.binders
    assert any("n" in p and "Unknown identifier" in p for p in formal.problems), formal.problems


def test_auto_bound_variables_stay_off_in_the_vacuity_check() -> None:
    # A file whose `ok` was forced, as if the formaliser had passed it.
    source = "import Mathlib theorem conjecture (h0 : 0 ≤ n) (h : n < 0) : n = 5 := sorry\n"
    formal = probed(SIMPLE).model_copy(update={"source": source})
    result = vacuity(formal, None, timeout_s=LEAN_TIMEOUT)
    assert not result.checked
    assert result.rejected


def test_sorry_axiom_in_the_statement_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    source = (
        "import Mathlib\n\ntheorem conjecture (n : ℕ) (h : n < sorryAx ℕ false) : n < 5 := sorry\n"
    )
    assert any("sorryAx" in p for p in source_problems(source))
    formal = _past_the_text_gate(monkeypatch, source)
    assert not formal.ok
    assert any("statement" in p and "sorry" in p for p in formal.problems), formal.problems


INTERNAL_SORRY = """import Mathlib

def c : ℕ := sorryAx ℕ false

theorem conjecture (n : ℕ) (h : n < c) : n < 5 := sorry
"""


def test_sorry_in_a_definition_the_statement_uses_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    formal = _past_the_text_gate(monkeypatch, INTERNAL_SORRY)
    assert not formal.ok
    assert any("statement" in p and "sorry" in p for p in formal.problems), formal.problems


def test_internal_names_cannot_hide_sorry(monkeypatch: pytest.MonkeyPatch) -> None:
    source = """import Mathlib

def _c : ℕ := sorryAx ℕ false

theorem _lem : (1 : ℕ) = 2 := sorryAx _ false

theorem conjecture (n : ℕ) (h : n < 3) : n < 5 := sorry
"""
    problems, _arity, _statement = command_check(source, timeout_s=LEAN_TIMEOUT)
    assert sum("`_c`" in p or "`_lem`" in p for p in problems) == 2, problems
    # Past the text gate and the allowlist, Lean's probe still names both.
    import minilab.formalise.formaliser as formaliser

    def allowed(_source: str, *, timeout_s: float) -> CommandCheck:
        return CommandCheck([], 2, None)

    monkeypatch.setattr(formaliser, "command_check", allowed)
    formal = _past_the_text_gate(monkeypatch, source)
    assert not formal.ok
    assert any("_c" in p and "sorry" in p for p in formal.problems), formal.problems
    assert any("_lem" in p and "sorry" in p for p in formal.problems), formal.problems


@pytest.mark.parametrize(
    "note",
    [
        's!"{(sorry : ℕ)} {(by admit : ℕ)}"',
        's!"{\'"\'}"',
        'm!"{(1 : ℕ)}"',
    ],
)
def test_interpolated_strings_are_refused(note: str) -> None:
    source = SIMPLE.replace(
        "theorem conjecture", f"def note : String := toString {note}\n\ntheorem conjecture"
    )
    problems, _arity, _statement = command_check(source, timeout_s=LEAN_TIMEOUT)
    assert any("interpolated string" in p for p in problems), problems


def test_open_in_cannot_wrap_a_scope_command() -> None:
    source = "import Mathlib\n\nopen Real in section\n\n" + SIMPLE.split("\n", 2)[2]
    problems, _arity, _statement = command_check(source, timeout_s=LEAN_TIMEOUT)
    assert any("open … in" in p for p in problems), problems


def test_conjecture_inside_a_section_is_top_level() -> None:
    source = "import Mathlib\n\nsection\n\n" + SIMPLE.split("\n", 2)[2] + "\nend\n"
    formal = probed(source)
    assert formal.ok, formal.problems
    assert [b.name for b in formal.binders] == ["x", "hx"]


def test_conjecture_inside_a_namespace_is_refused() -> None:
    source = "import Mathlib\n\nnamespace Foo\n\n" + SIMPLE.split("\n", 2)[2] + "\nend Foo\n"
    problems, arity, _statement = command_check(source, timeout_s=LEAN_TIMEOUT)
    assert any("Foo.conjecture" in p for p in problems), problems
    assert arity is None


# --- statements and names Lean and the text read differently (fourth adversarial review) ---

DECOY = """import Mathlib

namespace Shown
theorem conjecture (n : ℕ) (h : n < 3) : n < 5 := by omega
end Shown

theorem «conjecture» (n : ℕ) (h : n < 3) : n < 2 := sorry
"""


def test_a_conjecture_outside_the_root_namespace_is_refused() -> None:
    problems, _arity, _statement = command_check(DECOY, timeout_s=LEAN_TIMEOUT)
    assert any("Shown.conjecture" in p for p in problems), problems
    formal = formal_from_source(DECOY, Family.claude, "m", timeout_s=LEAN_TIMEOUT)
    assert not formal.ok


def test_the_statement_is_the_text_lean_read_as_conjecture() -> None:
    source = (
        "import Mathlib\n\ntheorem conjecture (n : ℕ) (h : n < 3) : "
        "letI k : ℕ := 0; n < 5 + k ∧ n = 7 := sorry\n"
    )
    formal = probed(source)
    assert formal.ok, formal.problems
    assert formal.lean_statement == (
        "theorem conjecture (n : ℕ) (h : n < 3) : letI k : ℕ := 0; n < 5 + k ∧ n = 7"
    )


OPEN_LEAK = """import Mathlib

def bad (x : ℝ) : Prop := x < x

theorem conjecture (x : ℝ) (hx : bad x) : x = 5 := sorry

namespace Other
def bad : ℕ := 0
theorem Fin.sum_univ_succ : True := trivial
end Other

open Other
"""


def test_a_top_level_open_is_refused() -> None:
    problems, _arity, _statement = command_check(OPEN_LEAK, timeout_s=LEAN_TIMEOUT)
    assert any("top-level `open`" in p for p in problems), problems


def test_open_inside_a_section_or_with_in_is_allowed() -> None:
    source = """import Mathlib

section
open Real
def g (x : ℝ) : ℝ := π * x
end

open Finset in
theorem conjecture (x : ℝ) (h : 0 < g x) : 0 < x := sorry
"""
    problems, arity, _statement = command_check(source, timeout_s=LEAN_TIMEOUT)
    assert problems == []
    assert arity == 2


@pytest.mark.parametrize("name", ["proof_1", "match_2", "eq_1", "omega_3", "Foo._x"])
def test_names_lean_treats_as_internal_are_refused(name: str) -> None:
    source = f"""import Mathlib

def {name} (x : ℝ) : Prop := x < x

theorem conjecture (x : ℝ) (hx : {name} x) : x = 5 := sorry
"""
    problems, _arity, _statement = command_check(source, timeout_s=LEAN_TIMEOUT)
    assert any(f"`{name}`" in p and "internal" in p for p in problems), problems


def test_a_cut_binder_type_is_marked_in_the_audit_prompt(store: RecordStore) -> None:
    from minilab.formalise.audit import audit_prompt

    formal = probed(SIMPLE)
    cut = [b.model_copy(update={"truncated": True}) for b in formal.binders]
    prompt = audit_prompt(
        survived_record(store), formal.model_copy(update={"binders": cut}), cut[1:], set()
    )
    assert prompt.count("…(cut)") == 2, prompt


# --- conditions in a variable's type (review 4, F4; user decision: flag for the audit) ---

SUBTYPE = """import Mathlib

theorem conjecture (r : {t : ℝ // t * t + 1 < t}) (x : ℝ) (hx : 0 < x) : x ≠ (r : ℝ) := sorry
"""


def test_a_variable_with_no_witness_value_blocks_satisfied_by_witness() -> None:
    formal = probed(SUBTYPE)
    assert formal.ok, formal.problems
    witness = Instance(values={"x": "1"}, precision_digits=30)
    result = vacuity(formal, witness, timeout_s=LEAN_TIMEOUT)
    assert result.checked
    assert [(c.label, c.state) for c in result.witness] == [("hx", "holds")]
    assert result.unwitnessed == ["r"]
    assert not result.satisfied_by_witness
    # Flagged for the audit, not rejected.
    assert not result.rejected


def test_the_audit_prompt_marks_variables_with_no_witness_value(store: RecordStore) -> None:
    from minilab.formalise.audit import audit_prompt

    formal = probed(SIMPLE)
    prompt = audit_prompt(
        survived_record(store), formal, formal.hypotheses, set(), unwitnessed={"x"}
    )
    line = next(row for row in prompt.splitlines() if row.startswith("- x :"))
    assert "no witness value" in line, prompt
