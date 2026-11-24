"""Tests for records and the record store (build plan Task 2)."""

import json
import math
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from minilab.records import (
    FINAL_STATUSES,
    CorruptRecord,
    Costs,
    Family,
    FinalStatus,
    Provenance,
    Record,
    RecordExists,
    RecordStore,
    Stage,
    StageAlreadyWritten,
    StageOutput,
    Status,
    TokenCount,
)

_NOW = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)


def _record(
    record_id: str = "r1",
    *,
    run_id: str = "run-1",
    instance_family: str | None = "n in 1..4, P a 2x3 stochastic matrix",
    status: Status = Status.proposed,
) -> Record:
    return Record(
        id=record_id,
        run_id=run_id,
        pair=("information-theory", "optimisation"),
        mode="structured",
        statement="R(D) is convex in D for every source P.",
        objects=["R(D)", "P"],
        assumptions=["P has full support"],
        domain_of_validity="0 <= D <= D_max",
        instance_family=instance_family,
        provenance=Provenance(
            family=Family.claude, model="claude-opus-5-5", prompt_hash="abc123", parents=[]
        ),
        status=status,
        stages={},
        costs=Costs(
            tokens={Family.claude: TokenCount(tokens_in=100, tokens_out=50, estimated=False)},
            seconds={Stage.generator: 1.25},
            review_minutes=0.0,
        ),
    )


def _output(note: str = "") -> StageOutput:
    return StageOutput(
        written_at=_NOW,
        data={"margin": "-0.125", "starts": 32, "nested": {"ok": True, "xs": [1.5, None]}},
        evidence=["certificates/r1.json"],
        notes=note,
    )


def _record_path(root: Path, run_id: str, record_id: str) -> Path:
    return root / "runs" / run_id / "records" / f"{record_id}.json"


def test_round_trip(tmp_path: Path) -> None:
    store = RecordStore(tmp_path)
    record = _record()
    store.create(record)
    store.write_stage("r1", Stage.generator, _output("generated"))
    loaded = store.load("r1")
    assert loaded == record.model_copy(update={"stages": {Stage.generator: _output("generated")}})
    assert _record_path(tmp_path, "run-1", "r1").is_file()
    # Loading reads the file, not a cache: a fresh store sees the same record.
    assert RecordStore(tmp_path).load("r1") == loaded


def test_stage_cannot_overwrite_other_stage(tmp_path: Path) -> None:
    store = RecordStore(tmp_path)
    store.create(_record())
    store.write_stage("r1", Stage.generator, _output("generated"))
    path = _record_path(tmp_path, "run-1", "r1")
    before = Record.model_validate_json(path.read_bytes())
    gen_before = before.stages[Stage.generator].model_dump_json().encode()

    store.write_stage("r1", Stage.falsifier, _output("attacked"))
    after = Record.model_validate_json(path.read_bytes())
    assert after.stages[Stage.generator].model_dump_json().encode() == gen_before
    assert after.stages[Stage.falsifier].notes == "attacked"

    with pytest.raises(StageAlreadyWritten):
        store.write_stage("r1", Stage.generator, _output("again"))
    # The refused write left the file as it was.
    assert Record.model_validate_json(path.read_bytes()) == after


@pytest.mark.parametrize("final", sorted(FINAL_STATUSES))
def test_status_transitions(tmp_path: Path, final: Status) -> None:
    assert FINAL_STATUSES == {
        Status.killed,
        Status.known,
        Status.proved,
        Status.trivial,
        Status.abandoned,
        Status.untestable,
    }
    store = RecordStore(tmp_path)
    store.create(_record())
    if final is Status.untestable:
        # A record with an instance family reaches untestable only by a stage verdict.
        store.write_stage("r1", Stage.encoding, _output(), status=final)
    else:
        store.write_stage("r1", Stage.falsifier, _output(), status=final)
    path = _record_path(tmp_path, "run-1", "r1")
    frozen = path.read_bytes()

    for other in Status:
        if other is final:
            continue
        with pytest.raises(FinalStatus):
            store.write_stage("r1", Stage.scorer, _output(), status=other)
        assert path.read_bytes() == frozen

    # A later stage may still add its output if it leaves the status alone.
    updated = store.write_stage("r1", Stage.scorer, _output("scored"))
    assert updated.status is final
    assert store.write_stage("r1", Stage.queue, _output(), status=final).status is final


def test_non_final_status_can_change(tmp_path: Path) -> None:
    store = RecordStore(tmp_path)
    store.create(_record())
    assert store.write_stage("r1", Stage.encoding, _output(), status=Status.ambiguous).status is (
        Status.ambiguous
    )
    assert store.write_stage("r1", Stage.falsifier, _output(), status=Status.survived).status is (
        Status.survived
    )
    assert store.write_stage("r1", Stage.literature, _output(), status=Status.known).status is (
        Status.known
    )


def test_untestable_without_instance_family(tmp_path: Path) -> None:
    record = _record(instance_family=None)
    assert record.status == Status.untestable
    store = RecordStore(tmp_path)
    store.create(record)
    assert store.load("r1").status == Status.untestable
    # A record with no instance family cannot claim any other verdict.
    with pytest.raises(ValidationError):
        _record(instance_family=None, status=Status.survived)


def test_create_refuses_existing_record(tmp_path: Path) -> None:
    store = RecordStore(tmp_path)
    store.create(_record())
    path = _record_path(tmp_path, "run-1", "r1")
    original = path.read_bytes()
    with pytest.raises(RecordExists):
        store.create(_record().model_copy(update={"statement": "something else"}))
    assert path.read_bytes() == original


def test_create_refuses_id_used_in_another_run(tmp_path: Path) -> None:
    store = RecordStore(tmp_path)
    store.create(_record())
    with pytest.raises(RecordExists):
        store.create(_record(run_id="run-2"))
    assert not (tmp_path / "runs" / "run-2" / "records" / "r1.json").exists()
    assert store.load("r1").run_id == "run-1"


def test_list_filters_by_run_and_status(tmp_path: Path) -> None:
    store = RecordStore(tmp_path)
    store.create(_record("b"))
    store.create(_record("a"))
    store.create(_record("c", instance_family=None))
    store.create(_record("z", run_id="run-2"))
    assert [r.id for r in store.list("run-1")] == ["a", "b", "c"]
    assert [r.id for r in store.list("run-1", Status.untestable)] == ["c"]
    assert [r.id for r in store.list("run-2")] == ["z"]
    assert store.list("run-3") == []


def test_no_temp_files_left_behind(tmp_path: Path) -> None:
    store = RecordStore(tmp_path)
    store.create(_record())
    store.write_stage("r1", Stage.generator, _output())
    records_dir = tmp_path / "runs" / "run-1" / "records"
    assert sorted(p.name for p in records_dir.iterdir()) == ["r1.json"]


@pytest.mark.parametrize("bad", ["", "../x", "a/b", ".hidden", "a\\b", "a b"])
def test_unsafe_ids_rejected(bad: str) -> None:
    with pytest.raises(ValidationError):
        _record(bad)
    with pytest.raises(ValidationError):
        _record(run_id=bad)


def test_load_unknown_record_raises(tmp_path: Path) -> None:
    store = RecordStore(tmp_path)
    store.create(_record())
    with pytest.raises(KeyError):
        store.load("nope")
    with pytest.raises(ValueError):
        store.load("../r1")


def test_blank_instance_family_is_untestable() -> None:
    assert _record(instance_family="   ").status is Status.untestable
    assert _record(instance_family="   ").instance_family is None


def test_stage_bytes_unchanged_in_raw_file(tmp_path: Path) -> None:
    store = RecordStore(tmp_path)
    store.create(_record())
    store.write_stage("r1", Stage.generator, _output("generated"))
    path = _record_path(tmp_path, "run-1", "r1")
    raw_before = json.loads(path.read_bytes())["stages"]["generator"]
    store.write_stage("r1", Stage.falsifier, _output("attacked"))
    store.write_stage("r1", Stage.literature, _output("read"))
    assert json.loads(path.read_bytes())["stages"]["generator"] == raw_before


def test_misplaced_file_is_refused_not_redirected(tmp_path: Path) -> None:
    store = RecordStore(tmp_path)
    store.create(_record("a"))
    a_path = _record_path(tmp_path, "run-1", "a")
    x_path = _record_path(tmp_path, "run-1", "x")
    shutil.copy(a_path, x_path)
    a_bytes = a_path.read_bytes()
    with pytest.raises(CorruptRecord):
        store.load("x")
    with pytest.raises(CorruptRecord):
        store.write_stage("x", Stage.generator, _output())
    assert a_path.read_bytes() == a_bytes


def test_create_revalidates_record(tmp_path: Path) -> None:
    store = RecordStore(tmp_path)
    with pytest.raises(ValidationError):
        _record().run_id = "../../escaped"  # records are frozen
    # Get past the frozen check the way a careless caller could.
    bad_run = _record()
    object.__setattr__(bad_run, "run_id", "../../escaped")
    with pytest.raises(ValidationError):
        store.create(bad_run)
    bad_status = _record(instance_family=None)
    object.__setattr__(bad_status, "status", Status.survived)
    with pytest.raises(ValidationError):
        store.create(bad_status)
    assert not (tmp_path.parent / "escaped").exists()
    assert not (tmp_path / "runs").exists() or store.list("run-1") == []


def test_corrupt_file_named_in_list_error(tmp_path: Path) -> None:
    store = RecordStore(tmp_path)
    store.create(_record())
    (tmp_path / "runs" / "run-1" / "records" / "partial.json").write_text('{"id": "par')
    with pytest.raises(CorruptRecord, match="partial.json"):
        store.list("run-1")


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_floats_refused(value: float) -> None:
    with pytest.raises(ValidationError):
        StageOutput(written_at=_NOW, data={"margin": value})
    with pytest.raises(ValidationError):
        StageOutput(written_at=_NOW, data={"deep": {"xs": [1.0, value]}})
    with pytest.raises(ValidationError):
        Costs(seconds={Stage.generator: value})


def test_status_given_as_string(tmp_path: Path) -> None:
    store = RecordStore(tmp_path)
    store.create(_record())
    store.write_stage("r1", Stage.falsifier, _output(), status=Status("killed"))
    as_text: str = "killed"
    record = store.write_stage(
        "r1",
        Stage.scorer,
        _output(),
        status=as_text,  # pyright: ignore[reportArgumentType]
    )
    assert record.status is Status.killed


@pytest.mark.parametrize(
    "stage",
    [s for s in Stage if s not in (Stage.scorer, Stage.queue)],
)
def test_verdict_stages_refused_after_final_status(tmp_path: Path, stage: Stage) -> None:
    store = RecordStore(tmp_path)
    store.create(_record())
    first = Stage.encoding if stage is not Stage.encoding else Stage.falsifier
    store.write_stage("r1", first, _output(), status=Status.killed)
    path = _record_path(tmp_path, "run-1", "r1")
    frozen = path.read_bytes()
    if stage is not first:
        with pytest.raises(FinalStatus):
            store.write_stage("r1", stage, _output("proved"))
        assert path.read_bytes() == frozen


def test_failed_link_leaves_no_temp_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = RecordStore(tmp_path)
    store.create(_record("a"))

    def refuse(src: str | Path, dst: str | Path) -> None:
        raise FileExistsError(dst)

    monkeypatch.setattr(os, "link", refuse)
    with pytest.raises(RecordExists):
        store.create(_record("b"))
    records_dir = tmp_path / "runs" / "run-1" / "records"
    assert sorted(p.name for p in records_dir.iterdir()) == ["a.json"]


@pytest.mark.parametrize("bad", ["R1", "Run-1"])
def test_upper_case_ids_rejected(bad: str) -> None:
    with pytest.raises(ValidationError):
        _record(bad)
