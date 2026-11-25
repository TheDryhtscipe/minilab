"""Tests for the run manifest and the cost ledger (build plan Task 3)."""

import math
import multiprocessing
import os
import shutil
import time
from datetime import UTC, date, datetime
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from minilab.ledger import CorruptLedger, Ledger, cost_report
from minilab.records import Family, Provenance, Record, RecordStore, Stage, Status, TokenCount
from minilab.run import (
    ManifestExists,
    MemoryPolicy,
    RunManifest,
    load_manifest,
    manifest_path,
    save_manifest,
)

RUN = "run-1"
PAIR = ("information-theory", "optimisation")
OTHER_PAIR = ("kernels", "probability")
_STARTED = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)


def _record(
    local: str,
    *,
    run_id: str = RUN,
    pair: tuple[str, str] = PAIR,
    family: Family = Family.claude,
    status: Status = Status.survived,
) -> Record:
    return Record(
        id=f"{run_id}.{local}",
        run_id=run_id,
        pair=pair,
        mode="structured",
        statement="R(D) is convex in D for every source P.",
        objects=["R(D)", "P"],
        assumptions=["P has full support"],
        domain_of_validity="0 <= D <= D_max",
        instance_family="n in 1..4",
        provenance=Provenance(family=family, model="m", prompt_hash="h"),
        status=status,
    )


@pytest.fixture
def store(tmp_path: Path) -> RecordStore:
    return RecordStore(tmp_path)


def _ledger(store: RecordStore, run_id: str = RUN) -> Ledger:
    """Open a run's ledger, first saving a manifest for the run if it has none."""
    if not manifest_path(store.root, run_id).exists():
        save_manifest(
            store.root,
            _manifest(run_id=run_id, families=list(Family), pairs=[PAIR, OTHER_PAIR]),
        )
    return Ledger(store, run_id)


def _charge(
    ledger: Ledger,
    *,
    record_id: str | None,
    family: Family = Family.claude,
    stage: Stage = Stage.generator,
    pair: tuple[str, str] = PAIR,
    tokens_in: int = 0,
    tokens_out: int = 0,
    seconds: float = 0.0,
    estimated: bool = False,
) -> None:
    ledger.charge(
        pair=pair,
        stage=stage,
        family=family,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        seconds=seconds,
        estimated=estimated,
        record_id=record_id,
    )


# --- The three tests the plan names -------------------------------------------------------


def test_cost_report_per_reviewed_survivor(store: RecordStore) -> None:
    for local in ("a", "b", "c"):
        store.create(_record(local))
    ledger = _ledger(store)
    _charge(ledger, record_id=None, tokens_in=1000, tokens_out=500, seconds=1800.0)
    _charge(ledger, record_id="run-1.a", stage=Stage.encoding, tokens_in=300, tokens_out=200,
            seconds=900.0, family=Family.codex)
    _charge(ledger, record_id="run-1.b", stage=Stage.prover, tokens_in=2000, tokens_out=1000,
            seconds=3600.0)
    _charge(ledger, record_id="run-1.c", stage=Stage.falsifier, tokens_in=0, tokens_out=0,
            seconds=900.0, family=Family.local)
    ledger.log_review("run-1.a", 15.0)
    ledger.log_review("run-1.b", 45.0)

    report = cost_report(store, ledger, RUN)

    # 5000 tokens; 7200 s of model time = 2 h; 60 review minutes = 1 h; 2 reviewed.
    assert report.reviewed_survivors == 2
    assert report.total_tokens == 5000
    assert report.unattributed_tokens == 1500
    assert report.tokens_per_reviewed_survivor == 2500.0
    assert report.machine_hours_per_reviewed_survivor == 1.0
    assert report.review_hours_per_reviewed_survivor == 0.5
    assert report.hours_per_reviewed_survivor == 1.5
    assert report.has_estimates is False


def test_estimated_costs_are_flagged(store: RecordStore) -> None:
    store.create(_record("a"))
    ledger = _ledger(store)
    _charge(ledger, record_id="run-1.a", tokens_in=10, tokens_out=10)
    assert cost_report(store, ledger, RUN).has_estimates is False

    _charge(ledger, record_id="run-1.a", stage=Stage.encoding, family=Family.gemini,
            tokens_in=10, tokens_out=10, estimated=True)
    assert cost_report(store, ledger, RUN).has_estimates is True
    costs = ledger.costs("run-1.a")
    assert costs.tokens[Family.gemini].estimated is True
    assert costs.tokens[Family.claude].estimated is False


def test_pair_spend_sums_all_stages(store: RecordStore) -> None:
    store.create(_record("a"))
    store.create(_record("z", pair=OTHER_PAIR))
    ledger = _ledger(store)
    # A generation call made before any record exists.
    _charge(ledger, record_id=None, stage=Stage.generator, tokens_in=100, tokens_out=1)
    _charge(ledger, record_id="run-1.a", stage=Stage.encoding, tokens_in=20, tokens_out=2,
            family=Family.codex)
    # The same pair named in the other order counts towards the same budget.
    _charge(ledger, record_id="run-1.a", stage=Stage.prover, tokens_in=3000, tokens_out=300,
            pair=(PAIR[1], PAIR[0]), estimated=True)
    # Another pair's spend does not count.
    _charge(ledger, record_id="run-1.z", pair=OTHER_PAIR, tokens_in=99999, tokens_out=1)

    assert ledger.pair_spend(PAIR) == 100 + 1 + 20 + 2 + 3000 + 300
    assert ledger.pair_spend((PAIR[1], PAIR[0])) == ledger.pair_spend(PAIR)
    assert ledger.pair_spend(OTHER_PAIR) == 100000


# --- Ledger behaviour -----------------------------------------------------------------------


def test_costs_for_record_by_family_and_stage(store: RecordStore) -> None:
    store.create(_record("a"))
    store.create(_record("b"))
    ledger = _ledger(store)
    _charge(ledger, record_id="run-1.a", stage=Stage.encoding, tokens_in=10, tokens_out=1,
            seconds=2.5)
    _charge(ledger, record_id="run-1.a", stage=Stage.encoding, tokens_in=5, tokens_out=4,
            seconds=0.5, family=Family.codex)
    _charge(ledger, record_id="run-1.a", stage=Stage.prover, tokens_in=7, tokens_out=3,
            seconds=4.0)
    _charge(ledger, record_id="run-1.b", stage=Stage.prover, tokens_in=1000, tokens_out=1000,
            seconds=100.0)
    ledger.log_review("run-1.a", 3.0)
    ledger.log_review("run-1.a", 2.0)

    costs = ledger.costs("run-1.a")
    assert costs.tokens == {
        Family.claude: TokenCount(tokens_in=17, tokens_out=4),
        Family.codex: TokenCount(tokens_in=5, tokens_out=4),
    }
    assert costs.seconds == {Stage.encoding: 3.0, Stage.prover: 4.0}
    assert costs.review_minutes == 5.0


def test_ledger_persists_across_instances(store: RecordStore, tmp_path: Path) -> None:
    store.create(_record("a"))
    _charge(_ledger(store), record_id="run-1.a", tokens_in=40, tokens_out=2)
    _ledger(store).log_review("run-1.a", 1.5)

    reopened = _ledger(store)
    assert reopened.pair_spend(PAIR) == 42
    assert reopened.costs("run-1.a").review_minutes == 1.5
    assert (tmp_path / "runs" / RUN / "ledger.jsonl").is_file()


def test_charge_refuses_record_of_other_pair(store: RecordStore) -> None:
    store.create(_record("a"))
    ledger = _ledger(store)
    with pytest.raises(ValueError, match="pair"):
        _charge(ledger, record_id="run-1.a", pair=OTHER_PAIR, tokens_in=1)
    assert ledger.pair_spend(OTHER_PAIR) == 0
    assert ledger.pair_spend(PAIR) == 0


def test_charge_and_review_refuse_records_of_other_runs(store: RecordStore) -> None:
    store.create(_record("a", run_id="run-2"))
    ledger = _ledger(store)
    with pytest.raises(ValueError, match="run-2"):
        _charge(ledger, record_id="run-2.a", tokens_in=1)
    with pytest.raises(ValueError, match="run-2"):
        ledger.log_review("run-2.a", 1.0)
    assert ledger.pair_spend(PAIR) == 0


def test_charge_and_review_refuse_unknown_records(store: RecordStore) -> None:
    ledger = _ledger(store)
    with pytest.raises(KeyError):
        _charge(ledger, record_id="run-1.missing", tokens_in=1)
    with pytest.raises(KeyError):
        ledger.log_review("run-1.missing", 1.0)
    assert ledger.pair_spend(PAIR) == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [("tokens_in", -1), ("tokens_out", -1), ("seconds", -0.5), ("seconds", math.nan),
     ("seconds", math.inf)],
)
def test_charge_refuses_bad_amounts(store: RecordStore, field: str, value: float) -> None:
    ledger = _ledger(store)
    amounts: dict[str, float] = {"tokens_in": 1, "tokens_out": 1, "seconds": 1.0}
    amounts[field] = value
    with pytest.raises(ValidationError):
        ledger.charge(pair=PAIR, stage=Stage.generator, family=Family.claude,
                      tokens_in=int(amounts["tokens_in"]), tokens_out=int(amounts["tokens_out"]),
                      seconds=amounts["seconds"], estimated=False)
    assert ledger.pair_spend(PAIR) == 0


@pytest.mark.parametrize("minutes", [-1.0, math.nan, math.inf])
def test_review_refuses_bad_minutes(store: RecordStore, minutes: float) -> None:
    store.create(_record("a"))
    ledger = _ledger(store)
    with pytest.raises(ValidationError):
        ledger.log_review("run-1.a", minutes)
    assert ledger.costs("run-1.a").review_minutes == 0.0


def test_charge_refuses_unknown_pair(store: RecordStore) -> None:
    ledger = _ledger(store)
    with pytest.raises(ValueError, match="domain"):
        _charge(ledger, record_id=None, pair=("kernels", "astrology"), tokens_in=1)


def test_corrupt_ledger_line_is_reported(store: RecordStore, tmp_path: Path) -> None:
    store.create(_record("a"))
    ledger = _ledger(store)
    _charge(ledger, record_id="run-1.a", tokens_in=1)
    path = tmp_path / "runs" / RUN / "ledger.jsonl"
    with path.open("a") as handle:
        handle.write('{"kind": "charge", "tokens_in": "lots"}\n')
    with pytest.raises(CorruptLedger, match="line 2"):
        ledger.pair_spend(PAIR)


def test_truncated_last_line_is_reported(store: RecordStore, tmp_path: Path) -> None:
    store.create(_record("a"))
    ledger = _ledger(store)
    _charge(ledger, record_id="run-1.a", tokens_in=1)
    path = tmp_path / "runs" / RUN / "ledger.jsonl"
    data = path.read_bytes()
    path.write_bytes(data[:-5])
    with pytest.raises(CorruptLedger):
        ledger.pair_spend(PAIR)


# --- The cost report ------------------------------------------------------------------------


def test_cost_report_family_shares(store: RecordStore) -> None:
    store.create(_record("a", family=Family.gemini))
    store.create(_record("b", family=Family.claude))
    store.create(_record("c", family=Family.claude))
    store.create(_record("d", family=Family.gemini))
    ledger = _ledger(store)
    _charge(ledger, record_id=None, family=Family.gemini, tokens_in=600, tokens_out=0)
    _charge(ledger, record_id="run-1.b", family=Family.claude, tokens_in=200, tokens_out=0)
    _charge(ledger, record_id="run-1.c", family=Family.codex, tokens_in=100, tokens_out=100)
    for record_id in ("run-1.a", "run-1.b", "run-1.c", "run-1.d"):
        ledger.log_review(record_id, 1.0)

    report = cost_report(store, ledger, RUN)
    assert report.family_cost_share == {
        Family.gemini: 0.6,
        Family.claude: 0.2,
        Family.codex: 0.2,
    }
    # Survivors are credited to the family that wrote them.
    assert report.family_survivor_share == {Family.gemini: 0.5, Family.claude: 0.5}


def test_cost_report_with_nothing_reviewed(store: RecordStore) -> None:
    store.create(_record("a"))
    ledger = _ledger(store)
    _charge(ledger, record_id="run-1.a", tokens_in=10, seconds=60.0)

    report = cost_report(store, ledger, RUN)
    assert report.reviewed_survivors == 0
    assert report.total_tokens == 10
    assert report.tokens_per_reviewed_survivor is None
    assert report.hours_per_reviewed_survivor is None
    assert report.family_survivor_share == {}


def test_cost_report_refuses_ledger_of_other_run(store: RecordStore) -> None:
    with pytest.raises(ValueError, match="run-2"):
        cost_report(store, _ledger(store, "run-2"), RUN)


# --- The run manifest -----------------------------------------------------------------------


def _manifest(**changes: object) -> RunManifest:
    fields: dict[str, object] = {
        "run_id": RUN,
        "mode": "structured",
        "pairs": [PAIR, OTHER_PAIR],
        "memory_policy": MemoryPolicy.blind,
        "memory_before": None,
        "literature_before": date(2026, 6, 30),
        "budget_tokens_per_pair": 300000,
        "families": [Family.claude, Family.codex],
        "started": _STARTED,
    }
    fields.update(changes)
    return RunManifest.model_validate(fields)


def test_manifest_round_trip(tmp_path: Path) -> None:
    manifest = _manifest(
        memory_policy=MemoryPolicy.pair, memory_before=datetime(2026, 6, 30, tzinfo=UTC)
    )
    path = save_manifest(tmp_path, manifest)
    assert path == tmp_path / "runs" / RUN / "manifest.json"
    assert load_manifest(tmp_path, RUN) == manifest


def test_manifest_is_written_once(tmp_path: Path) -> None:
    save_manifest(tmp_path, _manifest())
    before = (tmp_path / "runs" / RUN / "manifest.json").read_bytes()
    with pytest.raises(ManifestExists):
        save_manifest(tmp_path, _manifest(budget_tokens_per_pair=1))
    assert (tmp_path / "runs" / RUN / "manifest.json").read_bytes() == before


def test_manifest_pairs_are_canonical_and_unique() -> None:
    manifest = _manifest(pairs=[(PAIR[1], PAIR[0])])
    assert manifest.pairs == (PAIR,)
    with pytest.raises(ValidationError, match="more than once"):
        _manifest(pairs=[PAIR, (PAIR[1], PAIR[0])])
    with pytest.raises(ValidationError):
        _manifest(pairs=[])
    with pytest.raises(ValidationError, match="domain"):
        _manifest(pairs=[("kernels", "astrology")])


@pytest.mark.parametrize(
    "changes",
    [
        {"budget_tokens_per_pair": 0},
        {"families": []},
        {"families": [Family.claude, Family.claude]},
        {"started": datetime(2026, 11, 24, 12, 0)},
        {"memory_before": datetime(2026, 6, 30)},
        {"run_id": "Run 1"},
        {"mode": "fancy"},
        {"memory_policy": "some"},
        {"memory_before": datetime(2026, 6, 30, tzinfo=UTC)},
        {"budget_tokens_per_pair": True},
        {"budget_tokens_per_pair": 5.0},
        {"started": 0},
    ],
)
def test_manifest_refuses_bad_fields(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _manifest(**changes)


def test_load_manifest_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path, RUN)


# --- Hardening found in review ------------------------------------------------------------


def test_ledger_needs_a_manifest(store: RecordStore) -> None:
    with pytest.raises(FileNotFoundError):
        Ledger(store, "run-9")


def test_charge_refuses_pair_or_family_outside_manifest(store: RecordStore) -> None:
    save_manifest(store.root, _manifest(pairs=[PAIR], families=[Family.claude]))
    ledger = Ledger(store, RUN)
    with pytest.raises(ValueError, match="manifest"):
        _charge(ledger, record_id=None, pair=OTHER_PAIR, tokens_in=1)
    with pytest.raises(ValueError, match="manifest"):
        _charge(ledger, record_id=None, family=Family.gemini, tokens_in=1)
    assert ledger.entries() == []


@pytest.mark.parametrize(
    "status",
    [Status.proposed, Status.ambiguous, Status.abandoned, Status.known, Status.trivial,
     Status.untestable],
)
def test_review_refuses_records_that_are_never_queued(store: RecordStore, status: Status) -> None:
    store.create(_record("a", status=status))
    ledger = _ledger(store)
    with pytest.raises(ValueError, match="queue"):
        ledger.log_review("run-1.a", 1.0)
    assert ledger.entries() == []


@pytest.mark.parametrize("status", [Status.survived, Status.proved, Status.killed])
def test_review_accepts_queueable_records(store: RecordStore, status: Status) -> None:
    store.create(_record("a", status=status))
    ledger = _ledger(store)
    ledger.log_review("run-1.a", 1.0)
    assert ledger.costs("run-1.a").review_minutes == 1.0


def test_append_refuses_after_an_incomplete_line(store: RecordStore, tmp_path: Path) -> None:
    store.create(_record("a"))
    ledger = _ledger(store)
    _charge(ledger, record_id="run-1.a", tokens_in=1)
    path = tmp_path / "runs" / RUN / "ledger.jsonl"
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    broken = path.read_bytes()
    with pytest.raises(CorruptLedger, match="incomplete"):
        ledger.pair_spend(PAIR)
    with pytest.raises(CorruptLedger, match="incomplete"):
        _charge(ledger, record_id="run-1.a", tokens_in=1)
    assert path.read_bytes() == broken


def test_ledger_line_for_another_run_is_corrupt(store: RecordStore, tmp_path: Path) -> None:
    store.create(_record("a"))
    ledger = _ledger(store)
    _charge(ledger, record_id="run-1.a", tokens_in=1)
    path = tmp_path / "runs" / RUN / "ledger.jsonl"
    line = path.read_text().replace("run-1.a", "run-2.a")
    with path.open("a") as handle:
        handle.write(line)
    with pytest.raises(CorruptLedger, match="line 2"):
        ledger.entries()


def test_report_refuses_review_of_missing_record(store: RecordStore, tmp_path: Path) -> None:
    store.create(_record("a"))
    ledger = _ledger(store)
    ledger.log_review("run-1.a", 1.0)
    path = tmp_path / "runs" / RUN / "ledger.jsonl"
    path.write_text(path.read_text().replace("run-1.a", "run-1.ghost"))
    with pytest.raises(ValueError, match="ghost"):
        cost_report(store, ledger, RUN)


@pytest.mark.parametrize(
    "changes",
    [
        {"tokens_in": True},
        {"tokens_in": 1.0},
        {"tokens_in": 10**13},
        {"estimated": 1},
        {"seconds": True},
        {"seconds": "2.5"},
        {"seconds": 1e308},
    ],
)
def test_charge_refuses_loose_or_huge_values(
    store: RecordStore, changes: dict[str, object]
) -> None:
    ledger = _ledger(store)
    fields: dict[str, object] = {"tokens_in": 1, "tokens_out": 1, "seconds": 1.0,
                                 "estimated": False}
    fields.update(changes)
    with pytest.raises(ValidationError):
        ledger.charge(
            pair=PAIR,
            stage=Stage.generator,
            family=Family.claude,
            tokens_in=cast(int, fields["tokens_in"]),
            tokens_out=cast(int, fields["tokens_out"]),
            seconds=cast(float, fields["seconds"]),
            estimated=cast(bool, fields["estimated"]),
        )
    assert ledger.entries() == []


@pytest.mark.parametrize("minutes", [True, "3", 1e9])
def test_review_refuses_loose_or_huge_minutes(store: RecordStore, minutes: object) -> None:
    store.create(_record("a"))
    ledger = _ledger(store)
    with pytest.raises(ValidationError):
        ledger.log_review("run-1.a", cast(float, minutes))
    assert ledger.entries() == []


def test_estimated_flag_survives_a_later_exact_charge(store: RecordStore) -> None:
    store.create(_record("a"))
    ledger = _ledger(store)
    _charge(ledger, record_id="run-1.a", tokens_in=5, estimated=True)
    _charge(ledger, record_id="run-1.a", stage=Stage.encoding, tokens_in=5, estimated=False)
    assert ledger.costs("run-1.a").tokens[Family.claude].estimated is True


def test_save_manifest_validates_again(tmp_path: Path) -> None:
    good = _manifest()
    # model_copy does not validate, so this builds a manifest with a bad budget.
    bad = good.model_copy(update={"budget_tokens_per_pair": 0})
    with pytest.raises(ValidationError):
        save_manifest(tmp_path, bad)
    assert not manifest_path(tmp_path, RUN).exists()


def test_manifest_collections_cannot_change() -> None:
    manifest = _manifest()
    assert isinstance(manifest.pairs, tuple)
    assert isinstance(manifest.families, tuple)


def test_load_manifest_refuses_a_manifest_of_another_run(tmp_path: Path) -> None:
    source = save_manifest(tmp_path, _manifest(run_id="run-2"))
    target = manifest_path(tmp_path, RUN)
    target.parent.mkdir(parents=True)
    shutil.copy(source, target)
    with pytest.raises(ValueError, match="run-2"):
        load_manifest(tmp_path, RUN)


# --- Several processes at once, and writes cut short ------------------------------------

_FORK = multiprocessing.get_context("fork")
_STEP = 10
_BUDGET = 50


def _spend_up_to_budget(root: str, start: Event) -> None:
    """Charge `_STEP` tokens to PAIR while the pair's spend stays within `_BUDGET`."""
    start.wait()
    ledger = Ledger(RecordStore(Path(root)), RUN)
    while True:
        with ledger.pair_lock(PAIR):
            if ledger.pair_spend(PAIR) + _STEP > _BUDGET:
                return
            time.sleep(0.01)  # the model call
            _charge(ledger, record_id=None, tokens_in=_STEP)


def test_pair_lock_keeps_processes_within_the_budget(store: RecordStore) -> None:
    _ledger(store)
    start = _FORK.Event()
    workers = [
        _FORK.Process(target=_spend_up_to_budget, args=(str(store.root), start))
        for _ in range(6)
    ]
    for worker in workers:
        worker.start()
    start.set()
    for worker in workers:
        worker.join(timeout=60)
        assert worker.exitcode == 0
    assert Ledger(store, RUN).pair_spend(PAIR) == _BUDGET


def test_pair_lock_refuses_a_pair_outside_the_manifest(store: RecordStore) -> None:
    ledger = _ledger(store)
    with pytest.raises(ValueError, match="manifest"):
        with ledger.pair_lock(("kernels", "optimisation")):
            pass


def _charge_many(root: str, start: Event, family: Family) -> None:
    start.wait()
    ledger = Ledger(RecordStore(Path(root)), RUN)
    for _ in range(50):
        _charge(ledger, record_id=None, family=family, tokens_in=1)


def _read_until_done(root: str, start: Event, done: Event) -> None:
    start.wait()
    ledger = Ledger(RecordStore(Path(root)), RUN)
    while not done.is_set():
        ledger.pair_spend(PAIR)


def test_concurrent_charges_and_reads_never_see_a_broken_line(store: RecordStore) -> None:
    _ledger(store)
    start = _FORK.Event()
    done = _FORK.Event()
    families = list(Family)
    writers = [
        _FORK.Process(
            target=_charge_many, args=(str(store.root), start, families[n % len(families)])
        )
        for n in range(8)
    ]
    readers = [
        _FORK.Process(target=_read_until_done, args=(str(store.root), start, done))
        for _ in range(2)
    ]
    for worker in writers + readers:
        worker.start()
    start.set()
    for worker in writers:
        worker.join(timeout=60)
    done.set()
    for worker in readers:
        worker.join(timeout=60)
    assert [worker.exitcode for worker in writers + readers] == [0] * 10
    ledger = Ledger(store, RUN)
    assert len(ledger.charges()) == 400
    assert ledger.pair_spend(PAIR) == 400


def _save_one(root: str, start: Event, budget: int, results: "Queue[int]") -> None:
    start.wait()
    try:
        save_manifest(Path(root), _manifest(budget_tokens_per_pair=budget))
    except ManifestExists:
        return
    results.put(budget)


def test_concurrent_manifest_saves_have_one_winner(tmp_path: Path) -> None:
    for trial in range(5):
        root = tmp_path / str(trial)
        start = _FORK.Event()
        results: Queue[int] = _FORK.Queue()
        workers = [
            _FORK.Process(target=_save_one, args=(str(root), start, budget, results))
            for budget in range(1, 9)
        ]
        for worker in workers:
            worker.start()
        start.set()
        for worker in workers:
            worker.join(timeout=60)
            assert worker.exitcode == 0
        winners: list[int] = []
        while not results.empty():
            winners.append(results.get())
        assert len(winners) == 1
        assert load_manifest(root, RUN).budget_tokens_per_pair == winners[0]


def test_manifest_write_cut_short_leaves_no_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(fd: int) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", fail)
    with pytest.raises(OSError, match="disk full"):
        save_manifest(tmp_path, _manifest())
    monkeypatch.undo()
    assert sorted((tmp_path / "runs" / RUN).iterdir()) == []
    save_manifest(tmp_path, _manifest())
    assert load_manifest(tmp_path, RUN) == _manifest()


def test_temp_file_left_by_a_crash_does_not_block_the_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / RUN
    run_dir.mkdir(parents=True)
    (run_dir / ".manifest.abc.tmp").write_bytes(b'{"run_id": "run-1", "mo')
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path, RUN)
    save_manifest(tmp_path, _manifest())
    assert load_manifest(tmp_path, RUN) == _manifest()


def test_ledger_write_cut_short_is_refused_not_joined(store: RecordStore, tmp_path: Path) -> None:
    store.create(_record("a"))
    ledger = _ledger(store)
    _charge(ledger, record_id="run-1.a", tokens_in=1)
    path = tmp_path / "runs" / RUN / "ledger.jsonl"
    whole = path.read_bytes()
    path.write_bytes(whole + whole[: len(whole) // 2])
    with pytest.raises(CorruptLedger, match="line 2 is incomplete"):
        ledger.entries()
    with pytest.raises(CorruptLedger, match="incomplete"):
        _charge(ledger, record_id="run-1.a", tokens_in=1)


# --- Reservations (minilab-294360) -------------------------------------------------------


def _reserve(
    ledger: Ledger,
    tokens: int,
    *,
    record_id: str | None = None,
    pair: tuple[str, str] = PAIR,
    family: Family = Family.claude,
) -> str:
    return ledger.reserve(
        pair=pair, stage=Stage.encoding, family=family, tokens=tokens, record_id=record_id
    )


def test_open_reservation_counts_until_settled(store: RecordStore) -> None:
    ledger = _ledger(store)
    reservation = _reserve(ledger, 500)
    assert ledger.pair_spend(PAIR) == 500
    assert [r.id for r in ledger.open_reservations()] == [reservation]
    ledger.charge(
        pair=PAIR, stage=Stage.encoding, family=Family.claude, tokens_in=30, tokens_out=7,
        seconds=1.0, estimated=False, reservation=reservation,
    )
    assert ledger.pair_spend(PAIR) == 37
    assert ledger.open_reservations() == []


def test_released_reservation_costs_nothing(store: RecordStore) -> None:
    ledger = _ledger(store)
    reservation = _reserve(ledger, 500)
    ledger.release(reservation)
    assert ledger.pair_spend(PAIR) == 0
    assert ledger.open_reservations() == []
    assert cost_report(store, ledger, RUN).total_tokens == 0


def test_reservation_is_settled_at_most_once(store: RecordStore) -> None:
    ledger = _ledger(store)
    reservation = _reserve(ledger, 50)
    ledger.release(reservation)
    with pytest.raises(ValueError, match="not open"):
        ledger.release(reservation)
    with pytest.raises(ValueError, match="not open"):
        ledger.charge(
            pair=PAIR, stage=Stage.encoding, family=Family.claude, tokens_in=1, tokens_out=0,
            seconds=0.0, estimated=False, reservation=reservation,
        )
    with pytest.raises(ValueError, match="not open"):
        ledger.release("0" * 32)
    assert len(ledger.entries()) == 2


def test_settling_charge_must_match_the_reservation(store: RecordStore) -> None:
    ledger = _ledger(store)
    reservation = _reserve(ledger, 50)
    with pytest.raises(ValueError, match="pair"):
        ledger.charge(
            pair=OTHER_PAIR, stage=Stage.encoding, family=Family.claude, tokens_in=1,
            tokens_out=0, seconds=0.0, estimated=False, reservation=reservation,
        )
    with pytest.raises(ValueError, match="family"):
        ledger.charge(
            pair=PAIR, stage=Stage.encoding, family=Family.codex, tokens_in=1,
            tokens_out=0, seconds=0.0, estimated=False, reservation=reservation,
        )
    assert ledger.pair_spend(PAIR) == 50


def test_reserve_refuses_what_charge_refuses(store: RecordStore) -> None:
    store.create(_record("a", pair=OTHER_PAIR))
    ledger = _ledger(store)
    with pytest.raises(ValueError, match="pair"):
        _reserve(ledger, 1, record_id="run-1.a")
    with pytest.raises(ValueError, match="manifest"):
        _reserve(ledger, 1, pair=("kernels", "optimisation"))
    with pytest.raises(ValidationError):
        _reserve(ledger, -1)
    assert ledger.entries() == []


def test_open_reservation_is_an_estimated_cost(store: RecordStore) -> None:
    store.create(_record("a"))
    ledger = _ledger(store)
    _reserve(ledger, 40, record_id="run-1.a", family=Family.codex)
    _reserve(ledger, 60)
    report = cost_report(store, ledger, RUN)
    assert report.total_tokens == 100
    assert report.unattributed_tokens == 60
    assert report.has_estimates
    costs = ledger.costs("run-1.a")
    assert costs.tokens[Family.codex] == TokenCount(tokens_in=40, tokens_out=0, estimated=True)


def test_settling_an_unknown_reservation_on_disk_is_corrupt(
    store: RecordStore, tmp_path: Path
) -> None:
    ledger = _ledger(store)
    reservation = _reserve(ledger, 5)
    ledger.release(reservation)
    path = tmp_path / "runs" / RUN / "ledger.jsonl"
    release_line = path.read_bytes().splitlines()[1] + b"\n"
    path.write_bytes(path.read_bytes() + release_line)
    with pytest.raises(CorruptLedger, match="line 3"):
        ledger.entries()


def _release_once(root: str, start: Event, reservation: str, results: "Queue[str]") -> None:
    start.wait()
    ledger = Ledger(RecordStore(Path(root)), RUN)
    try:
        ledger.release(reservation)
        results.put("released")
    except ValueError:
        results.put("refused")


def test_concurrent_settles_of_one_reservation_have_one_winner(store: RecordStore) -> None:
    ledger = _ledger(store)
    reservation = _reserve(ledger, 5)
    start = _FORK.Event()
    results: Queue[str] = _FORK.Queue()
    workers = [
        _FORK.Process(target=_release_once, args=(str(store.root), start, reservation, results))
        for _ in range(6)
    ]
    for worker in workers:
        worker.start()
    start.set()
    for worker in workers:
        worker.join(timeout=60)
        assert worker.exitcode == 0
    outcomes = sorted(results.get(timeout=5) for _ in workers)
    assert outcomes == ["refused"] * 5 + ["released"]
    ledger.entries()
