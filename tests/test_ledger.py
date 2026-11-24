"""Tests for the run manifest and the cost ledger (build plan Task 3)."""

import math
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from minilab.ledger import CorruptLedger, Ledger, cost_report
from minilab.records import Family, Provenance, Record, RecordStore, Stage, Status, TokenCount
from minilab.run import ManifestExists, MemoryPolicy, RunManifest, load_manifest, save_manifest

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
    ledger = Ledger(store, RUN)
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
    assert report.tokens_per_reviewed_survivor == 2500.0
    assert report.machine_hours_per_reviewed_survivor == 1.0
    assert report.review_hours_per_reviewed_survivor == 0.5
    assert report.hours_per_reviewed_survivor == 1.5
    assert report.has_estimates is False


def test_estimated_costs_are_flagged(store: RecordStore) -> None:
    store.create(_record("a"))
    ledger = Ledger(store, RUN)
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
    ledger = Ledger(store, RUN)
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
    ledger = Ledger(store, RUN)
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
    _charge(Ledger(store, RUN), record_id="run-1.a", tokens_in=40, tokens_out=2)
    Ledger(store, RUN).log_review("run-1.a", 1.5)

    reopened = Ledger(store, RUN)
    assert reopened.pair_spend(PAIR) == 42
    assert reopened.costs("run-1.a").review_minutes == 1.5
    assert (tmp_path / "runs" / RUN / "ledger.jsonl").is_file()


def test_charge_refuses_record_of_other_pair(store: RecordStore) -> None:
    store.create(_record("a"))
    ledger = Ledger(store, RUN)
    with pytest.raises(ValueError, match="pair"):
        _charge(ledger, record_id="run-1.a", pair=OTHER_PAIR, tokens_in=1)
    assert ledger.pair_spend(OTHER_PAIR) == 0
    assert ledger.pair_spend(PAIR) == 0


def test_charge_and_review_refuse_records_of_other_runs(store: RecordStore) -> None:
    store.create(_record("a", run_id="run-2"))
    ledger = Ledger(store, RUN)
    with pytest.raises(ValueError, match="run-2"):
        _charge(ledger, record_id="run-2.a", tokens_in=1)
    with pytest.raises(ValueError, match="run-2"):
        ledger.log_review("run-2.a", 1.0)
    assert ledger.pair_spend(PAIR) == 0


def test_charge_and_review_refuse_unknown_records(store: RecordStore) -> None:
    ledger = Ledger(store, RUN)
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
    ledger = Ledger(store, RUN)
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
    ledger = Ledger(store, RUN)
    with pytest.raises(ValidationError):
        ledger.log_review("run-1.a", minutes)
    assert ledger.costs("run-1.a").review_minutes == 0.0


def test_charge_refuses_unknown_pair(store: RecordStore) -> None:
    ledger = Ledger(store, RUN)
    with pytest.raises(ValueError, match="domain"):
        _charge(ledger, record_id=None, pair=("kernels", "astrology"), tokens_in=1)


def test_corrupt_ledger_line_is_reported(store: RecordStore, tmp_path: Path) -> None:
    store.create(_record("a"))
    ledger = Ledger(store, RUN)
    _charge(ledger, record_id="run-1.a", tokens_in=1)
    path = tmp_path / "runs" / RUN / "ledger.jsonl"
    with path.open("a") as handle:
        handle.write('{"kind": "charge", "tokens_in": "lots"}\n')
    with pytest.raises(CorruptLedger, match="line 2"):
        ledger.pair_spend(PAIR)


def test_truncated_last_line_is_reported(store: RecordStore, tmp_path: Path) -> None:
    store.create(_record("a"))
    ledger = Ledger(store, RUN)
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
    ledger = Ledger(store, RUN)
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
    ledger = Ledger(store, RUN)
    _charge(ledger, record_id="run-1.a", tokens_in=10, seconds=60.0)

    report = cost_report(store, ledger, RUN)
    assert report.reviewed_survivors == 0
    assert report.total_tokens == 10
    assert report.tokens_per_reviewed_survivor is None
    assert report.hours_per_reviewed_survivor is None
    assert report.family_survivor_share == {}


def test_cost_report_refuses_ledger_of_other_run(store: RecordStore) -> None:
    with pytest.raises(ValueError, match="run-2"):
        cost_report(store, Ledger(store, "run-2"), RUN)


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
    manifest = _manifest(memory_before=datetime(2026, 6, 30, tzinfo=UTC))
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
    assert manifest.pairs == [PAIR]
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
    ],
)
def test_manifest_refuses_bad_fields(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _manifest(**changes)


def test_load_manifest_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path, RUN)
