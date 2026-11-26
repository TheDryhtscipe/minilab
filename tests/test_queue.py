"""Tests for the scorer and the human queue (build plan doc-5wh, Task 14).

The queue holds exactly three kinds of record (doc-gmj, "Human queue"):
survivors with a coverage report, proved results that are not trivial, and
certified kills with a tested mechanism. The scorer only ranks them.
`minilab review` stores the user's verdict and logs the minutes since the
item was shown.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import JsonValue
from typer.testing import CliRunner

from minilab.cli import app
from minilab.ledger import Ledger
from minilab.queue import (
    NotInQueue,
    QueueKind,
    Verdict,
    load_shown,
    mark_shown,
    queue_items,
    review,
)
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
from minilab.score import score, token_count

RUN = "run-1"
PAIR = ("information-theory", "optimisation")
T0 = datetime(2026, 11, 25, 9, 0, tzinfo=UTC)

COVERAGE: dict[str, JsonValue] = {
    "space": "P in simplex(3x2), s in (0, 5)",
    "dimensions": 6,
    "starts": 64,
    "worst_margin": "0.0123",
    "worst_point": {"s": "1.2"},
    "uncovered": ["s > 5"],
}
CERTIFICATE: dict[str, JsonValue] = {"kind": "exact", "point": {"t": "3/7"}, "margin": "-1/49"}
MECHANISM_PASSED: dict[str, JsonValue] = {
    "passed": True,
    "held": 10,
    "tried": 10,
    "description": "R(D) is linear when the optimal output law is not unique",
}
MECHANISM_FAILED: dict[str, JsonValue] = {**MECHANISM_PASSED, "passed": False, "held": 6}


def _out(
    data: dict[str, JsonValue] | None = None,
    *,
    evidence: list[str] | None = None,
    notes: str = "",
) -> StageOutput:
    return StageOutput(written_at=T0, data=data or {}, evidence=evidence or [], notes=notes)


def _record(local: str, *, instance_family: str | None = "n in 1..4") -> Record:
    return Record(
        id=f"{RUN}.{local}",
        run_id=RUN,
        pair=PAIR,
        mode="structured",
        statement=f"Statement of {local}: R(D) is convex in D for every source P.",
        objects=["R(D)", "P"],
        assumptions=["P has full support"],
        domain_of_validity="0 <= D <= D_max",
        instance_family=instance_family,
        provenance=Provenance(family=Family.claude, model="m", prompt_hash="h"),
    )


def _new(store: RecordStore, local: str, *, instance_family: str | None = "n in 1..4") -> str:
    record = _record(local, instance_family=instance_family)
    store.create(record)
    return record.id


def _survivor(store: RecordStore, local: str, coverage: dict[str, JsonValue] | None) -> str:
    rid = _new(store, local)
    data: dict[str, JsonValue] = {} if coverage is None else {"coverage": coverage}
    store.write_stage(rid, Stage.falsifier, _out(data), status=Status.survived)
    return rid


def _kill(store: RecordStore, local: str, mechanism: dict[str, JsonValue] | None) -> str:
    rid = _new(store, local)
    data: dict[str, JsonValue] = {"certificate": CERTIFICATE}
    if mechanism is not None:
        data["mechanism"] = mechanism
    store.write_stage(
        rid, Stage.falsifier, _out(data, evidence=["certificates/k.json"]), status=Status.killed
    )
    return rid


def _formalised(store: RecordStore, local: str) -> str:
    rid = _survivor(store, local, COVERAGE)
    store.write_stage(
        rid,
        Stage.literature,
        _out({"hits": []}, notes="No close match on arXiv before 2026-06-30."),
    )
    store.write_stage(
        rid,
        Stage.formaliser,
        _out(
            {
                "lean_statement": "theorem t (x : ℝ) (h : 0 < x) : 0 < x ^ 2 + x",
                "roundtrip": "For every positive real x, x squared plus x is positive.",
                "audit": {"extra_hypotheses": []},
            }
        ),
    )
    return rid


def _proved(store: RecordStore, local: str, *, near_misses: list[str] | None = None) -> str:
    rid = _formalised(store, local)
    misses: list[JsonValue] = list(near_misses or [])
    store.write_stage(rid, Stage.triviality, _out({"near_misses": misses}))
    store.write_stage(
        rid,
        Stage.prover,
        _out(
            {"proof": "by\n  have h2 := sq_nonneg x\n  nlinarith [Real.add_pos h h]", "axioms": []},
            evidence=["lean/p.lean"],
        ),
        status=Status.proved,
    )
    return rid


def _trivial(store: RecordStore, local: str) -> str:
    rid = _formalised(store, local)
    store.write_stage(
        rid, Stage.triviality, _out({"proof": "by positivity"}), status=Status.trivial
    )
    return rid


def _with_status(store: RecordStore, local: str, status: Status) -> str:
    """A record that reached `status` by a stage that holds every kind of queue evidence."""
    rid = _new(store, local)
    full: dict[str, JsonValue] = {
        "coverage": COVERAGE,
        "certificate": CERTIFICATE,
        "mechanism": MECHANISM_PASSED,
    }
    store.write_stage(rid, Stage.falsifier, _out(full), status=status)
    return rid


@pytest.fixture
def store(tmp_path: Path) -> RecordStore:
    return RecordStore(tmp_path)


@pytest.fixture
def ledger(store: RecordStore) -> Ledger:
    save_manifest(
        store.root,
        RunManifest(
            run_id=RUN,
            mode="structured",
            pairs=(PAIR,),
            memory_policy=MemoryPolicy.blind,
            memory_before=None,
            literature_before=None,
            budget_tokens_per_pair=300_000,
            families=(Family.claude, Family.codex),
            started=T0,
        ),
    )
    return Ledger(store, RUN)


def test_queue_membership(store: RecordStore) -> None:
    queued = {
        _survivor(store, "survivor", COVERAGE): QueueKind.survivor,
        _proved(store, "proved"): QueueKind.proved,
        _kill(store, "kill-mech", MECHANISM_PASSED): QueueKind.kill,
    }
    left_out = [
        _new(store, "proposed"),
        _new(store, "untestable", instance_family=None),
        _with_status(store, "ambiguous", Status.ambiguous),
        _with_status(store, "known", Status.known),
        _with_status(store, "abandoned", Status.abandoned),
        _trivial(store, "trivial"),
        _survivor(store, "survivor-no-coverage", None),
        _survivor(store, "survivor-thin-coverage", {"space": "s", "starts": 3}),
        _kill(store, "kill-no-mech", None),
        _kill(store, "kill-failed-mech", MECHANISM_FAILED),
        _kill(store, "kill-truthy-mech", {**MECHANISM_PASSED, "passed": "yes"}),
    ]
    statuses = {r.status for r in store.list(RUN)}
    assert statuses == set(Status)  # one record (at least) per status

    items = queue_items(store, RUN)

    assert {item.record.id: item.kind for item in items} == queued
    assert not {item.record.id for item in items} & set(left_out)


def test_proved_with_trivial_triviality_stage_not_queued(store: RecordStore) -> None:
    rid = _formalised(store, "flagged")
    store.write_stage(rid, Stage.triviality, _out({"trivial": True, "near_misses": []}))
    store.write_stage(rid, Stage.prover, _out({"proof": "by simp"}), status=Status.proved)

    assert queue_items(store, RUN) == []


def test_untestable_and_ambiguous_never_queued(store: RecordStore, ledger: Ledger) -> None:
    untestable = _new(store, "untestable", instance_family=None)
    ambiguous = _with_status(store, "ambiguous", Status.ambiguous)
    control = _survivor(store, "control", COVERAGE)

    assert [item.record.id for item in queue_items(store, RUN)] == [control]

    result = CliRunner().invoke(app, ["queue", RUN, "--root", str(store.root)])
    assert result.exit_code == 0, result.output
    assert control in result.output
    assert untestable not in result.output
    assert ambiguous not in result.output
    shown = load_shown(store.root, RUN)
    assert set(shown.shown) == {control}

    for rid in (untestable, ambiguous):
        mark_shown(store.root, RUN, [rid], T0)  # even if somehow shown
        with pytest.raises(NotInQueue):
            review(store, ledger, rid, Verdict.interesting, "", now=T0 + timedelta(minutes=1))
    assert ledger.reviews() == []


def test_review_logs_minutes(store: RecordStore, ledger: Ledger) -> None:
    first = _survivor(store, "first", COVERAGE)
    second = _kill(store, "second", MECHANISM_PASSED)
    mark_shown(store.root, RUN, [first, second], T0)

    review(store, ledger, first, Verdict.interesting, "nice", now=T0 + timedelta(minutes=7))
    # The second item's clock starts when the first review ended.
    review(store, ledger, second, Verdict.known, "", now=T0 + timedelta(minutes=10))

    minutes = {r.record_id: r.minutes for r in ledger.reviews()}
    assert minutes == pytest.approx({first: 7.0, second: 3.0})
    stored = store.load(first).stages[Stage.queue].data
    assert stored["verdict"] == "interesting"
    assert stored["note"] == "nice"
    assert stored["minutes"] == pytest.approx(7.0)
    assert store.load(second).stages[Stage.queue].data["verdict"] == "known"
    # A verdict does not change the status.
    assert store.load(second).status is Status.killed


def test_review_cli_logs_minutes_since_shown(store: RecordStore, ledger: Ledger) -> None:
    rid = _survivor(store, "item", COVERAGE)
    runner = CliRunner()
    shown = runner.invoke(app, ["queue", RUN, "--root", str(store.root)])
    assert shown.exit_code == 0, shown.output
    # Pretend the user read the item for 12 minutes.
    mark_shown(store.root, RUN, [rid], datetime.now(UTC) - timedelta(minutes=12))

    result = runner.invoke(
        app,
        [
            "review",
            rid,
            "--verdict",
            "not-interesting",
            "--note",
            "already folklore",
            "--root",
            str(store.root),
        ],
    )

    assert result.exit_code == 0, result.output
    (logged,) = ledger.reviews()
    assert logged.record_id == rid
    assert logged.minutes == pytest.approx(12.0, abs=0.1)
    data = store.load(rid).stages[Stage.queue].data
    assert data["verdict"] == "not-interesting"
    assert data["note"] == "already folklore"


def test_review_needs_item_shown_first(store: RecordStore, ledger: Ledger) -> None:
    rid = _survivor(store, "unseen", COVERAGE)
    with pytest.raises(NotInQueue, match="shown"):
        review(store, ledger, rid, Verdict.interesting, "", now=T0)
    assert ledger.reviews() == []


def test_review_twice_refused(store: RecordStore, ledger: Ledger) -> None:
    rid = _survivor(store, "twice", COVERAGE)
    mark_shown(store.root, RUN, [rid], T0)
    review(store, ledger, rid, Verdict.interesting, "", now=T0 + timedelta(minutes=2))
    with pytest.raises(ValueError, match="already reviewed"):
        review(store, ledger, rid, Verdict.known, "", now=T0 + timedelta(minutes=4))
    assert len(ledger.reviews()) == 1


def test_queue_cli_prints_every_section(store: RecordStore) -> None:
    proved = _proved(store, "proved")
    kill = _kill(store, "kill", MECHANISM_PASSED)

    result = CliRunner().invoke(app, ["queue", RUN, "--root", str(store.root)])

    assert result.exit_code == 0, result.output
    out = result.output
    assert proved in out and kill in out
    assert "x squared plus x is positive" in out  # round-trip text
    assert "extra_hypotheses" in out  # audit
    assert "No close match on arXiv" in out  # literature notes
    assert "lean/p.lean" in out  # evidence
    assert "certificates/k.json" in out  # evidence
    assert "output law is not unique" in out  # mechanism


def test_queue_cli_shows_a_held_proof_and_its_flags(store: RecordStore) -> None:
    held = _formalised(store, "held")
    store.write_stage(
        held,
        Stage.triviality,
        _out(
            {
                "screened": True,
                "trivial": False,
                "held": True,
                "proved_by": "nlinarith",
                "proof": "nlinarith [sq_nonneg x]",
                "hold_reasons": ["the audit found a weakening: h1: x < 1"],
                "near_misses": [],
            }
        ),
    )
    plain = _formalised(store, "plain")
    store.write_stage(
        plain,
        Stage.triviality,
        _out({"screened": True, "trivial": False, "held": False, "proof": None}),
    )

    result = CliRunner().invoke(app, ["queue", RUN, "--root", str(store.root)])

    assert result.exit_code == 0, result.output
    items = {chunk.split()[1]: chunk for chunk in result.output.split("\n[")}
    held_text, plain_text = items[held], items[plain]
    assert "Held" in held_text
    assert "nlinarith [sq_nonneg x]" in held_text
    assert "the audit found a weakening: h1: x < 1" in held_text
    assert "Held" not in plain_text


def test_token_count() -> None:
    assert token_count("") == 0
    assert token_count("0 < x ^ 2 + x") == 7
    assert token_count("Real.add_pos h h") == 3


def test_score_compression_and_signals(store: RecordStore) -> None:
    rid = _proved(store, "scored", near_misses=["nlinarith"])
    s = score(store.load(rid))

    lean = "theorem t (x : ℝ) (h : 0 < x) : 0 < x ^ 2 + x"
    proof = "by\n  have h2 := sq_nonneg x\n  nlinarith [Real.add_pos h h]"
    assert s.statement_tokens == token_count(lean)
    assert s.proof_tokens == token_count(proof)
    assert s.compression == pytest.approx(token_count(lean) / token_count(proof))
    assert s.near_misses == ["nlinarith"]
    assert s.library_lemmas == ["Real.add_pos", "sq_nonneg"]


def test_score_without_proof_has_no_compression(store: RecordStore) -> None:
    s = score(store.load(_survivor(store, "s", COVERAGE)))
    assert s.proof_tokens is None
    assert s.compression is None
    assert s.near_misses == []


def test_ranking_puts_near_misses_last(store: RecordStore) -> None:
    near = _proved(store, "a-near", near_misses=["simp", "aesop"])
    clean = _proved(store, "b-clean")

    assert [item.record.id for item in queue_items(store, RUN)] == [clean, near]


def test_shown_file_round_trip(store: RecordStore) -> None:
    mark_shown(store.root, RUN, [f"{RUN}.a"], T0)
    mark_shown(store.root, RUN, [f"{RUN}.b"], T0 + timedelta(minutes=1))
    shown = load_shown(store.root, RUN)
    assert shown.shown == {f"{RUN}.a": T0, f"{RUN}.b": T0 + timedelta(minutes=1)}
    raw = json.loads((store.root / "runs" / RUN / "queue" / "shown.json").read_text())
    assert set(raw["shown"]) == {f"{RUN}.a", f"{RUN}.b"}
