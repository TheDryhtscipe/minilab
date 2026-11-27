"""Tests for the pipeline driver and the memory client (build plan doc-5wh, Task 16).

The driver runs a run's records through the stages in the design's order
(doc-gmj, "Records and stages"), stops each record at its first final status,
and can be resumed after a failure without writing any stage twice. The
memory client enforces the manifest's read policy and never reads vault `sci`.

Every stage here is a fake that writes plausible stage output; every model
and every MuninnDB call is a fake. Nothing here calls a live service.
"""

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

import pytest
from pydantic import JsonValue
from typer.testing import CliRunner

from minilab.cli import app
from minilab.memory import (
    VAULT,
    MemoryClient,
    MemoryReadRefused,
    MemoryUnavailable,
    pair_tag,
)
from minilab.models.adapter import BudgetExceeded
from minilab.models.backends import ModelUnavailable
from minilab.pipeline import (
    BENCHMARK_LITERATURE_CAP,
    RECORD_STAGES,
    STAGE_ORDER,
    BenchmarkRefused,
    DriverSettings,
    StageContext,
    StageRunner,
    check_benchmark,
    drive,
    next_stage,
    start_run,
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
from minilab.run import MemoryPolicy, RunManifest, default_models
from minilab.scheduler import Pair

RUN = "run-1"
PAIR: Pair = ("information-theory", "optimisation")
OTHER: Pair = ("kernels", "probability")
FAMILIES: tuple[Family, ...] = (Family.claude, Family.codex, Family.gemini, Family.local)
NOW = datetime(2026, 11, 27, 1, 0, tzinfo=UTC)

COVERAGE: dict[str, JsonValue] = {
    "space": "p in (0, 1)",
    "dimensions": 1,
    "starts": 16,
    "worst_margin": "0.1",
    "worst_point": {"p": "1/2"},
    "uncovered": [],
}

Outcome = Literal["survive", "kill", "kill-mechanism", "known", "trivial", "proved", "ambiguous"]


def manifest(
    *,
    run_id: str = RUN,
    mode: Literal["plain", "structured"] = "plain",
    pairs: tuple[Pair, ...] = (PAIR,),
    policy: MemoryPolicy = MemoryPolicy.blind,
    memory_before: datetime | None = None,
    literature_before: date | None = None,
    families: tuple[Family, ...] = FAMILIES,
) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        mode=mode,
        pairs=pairs,
        memory_policy=policy,
        memory_before=memory_before,
        literature_before=literature_before,
        budget_tokens_per_pair=1_000_000,
        families=families,
        models=default_models(families),
        started=NOW,
    )


def out(**data: JsonValue) -> StageOutput:
    return StageOutput(written_at=NOW, data=dict(data))


class FakeGenerator:
    """Makes one record per call, whose statement names its outcome for the fake stages."""

    def __init__(self, outcomes: Sequence[Outcome], fail_first: set[Family] | None = None) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[Pair, Family, int, tuple[str, ...]]] = []
        self.fail_first = set(fail_first or set())

    def __call__(
        self, pair: Pair, family: Family, n: int, earlier: Sequence[str], ctx: StageContext
    ) -> list[Record]:
        self.calls.append((pair, family, n, tuple(earlier)))
        if family in self.fail_first:
            self.fail_first.discard(family)
            raise ModelUnavailable(f"{family} hit its usage limit")
        if not self.outcomes:
            return []
        outcome = self.outcomes.pop(0)
        record = Record(
            id=f"{ctx.ledger.run_id}.g-{family.value}-{pair[0][:4]}-{len(self.calls)}",
            run_id=ctx.ledger.run_id,
            pair=pair,
            mode=ctx.ledger.manifest.mode,
            statement=f"outcome:{outcome}",
            objects=["x"],
            assumptions=["x > 0"],
            domain_of_validity="x > 0",
            instance_family="x: a rational",
            provenance=Provenance(family=family, model="fake", prompt_hash="h"),
            stages={Stage.generator: out(family=family.value, mode=ctx.ledger.manifest.mode)},
        )
        ctx.ledger.store.create(record)
        return [record]


class FakeStages:
    """Fake stage runners. Each writes its stage the way the real one would for the
    outcome named in the record's statement, and logs (record id, stage) calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Stage]] = []
        self.written: list[tuple[str, Stage]] = []
        self.fail: dict[tuple[str, Stage], BaseException] = {}

    def runners(self) -> dict[Stage, StageRunner]:
        def make(stage: Stage) -> StageRunner:
            def run(record_id: str, ctx: StageContext) -> None:
                self.calls.append((record_id, stage))
                error = self.fail.pop((record_id, stage), None)
                if error is not None:
                    raise error
                self._write(ctx.ledger.store, record_id, stage)
                self.written.append((record_id, stage))

            return run

        return {stage: make(stage) for stage in RECORD_STAGES}

    @staticmethod
    def _write(store: RecordStore, record_id: str, stage: Stage) -> None:
        record = store.load(record_id)
        outcome = record.statement.removeprefix("outcome:")
        status: Status | None = None
        data: dict[str, JsonValue] = {}
        if stage is Stage.encoding:
            data = {"agreed": outcome != "ambiguous"}
            if outcome == "ambiguous":
                status = Status.ambiguous
        elif stage is Stage.falsifier:
            if outcome in ("kill", "kill-mechanism"):
                status = Status.killed
                data = {
                    "certificate": {"kind": "exact"},
                    "mechanism": {"passed": outcome == "kill-mechanism"},
                }
            else:
                status = Status.survived
                data = {"coverage": COVERAGE}
        elif stage is Stage.literature:
            if outcome == "known":
                status = Status.known
        elif stage is Stage.formaliser:
            data = {"lean_statement": "theorem conjecture : True"}
        elif stage is Stage.triviality:
            data = {"trivial": outcome == "trivial"}
            if outcome == "trivial":
                status = Status.trivial
        elif stage is Stage.prover:
            data = {"attempted": True, "proof": "trivial"}
            if outcome == "proved":
                status = Status.proved
        store.write_stage(record_id, stage, out(**data), status=status)


class SpyTransport:
    """A fake MuninnDB transport: logs every tool call, answers recall with `memories`."""

    def __init__(self, memories: Sequence[Mapping[str, JsonValue]] = ()) -> None:
        self.calls: list[tuple[str, dict[str, JsonValue]]] = []
        self.memories: list[JsonValue] = [dict(m) for m in memories]
        self.down = False

    def call_tool(self, name: str, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        self.calls.append((name, arguments))
        if self.down:
            raise MemoryUnavailable("MuninnDB is down")
        if name == "muninn_remember":
            return {"id": f"m-{len(self.calls)}"}
        if name == "muninn_recall":
            return {"memories": list(self.memories)}
        return {}

    def reads(self) -> list[tuple[str, dict[str, JsonValue]]]:
        return [(n, a) for n, a in self.calls if n != "muninn_remember"]


def memory_content(
    record_id: str, statement: str, *, run_id: str = "old-run", pair: Pair = PAIR
) -> str:
    return (
        f"Minilab record {record_id}\nRun: {run_id}\nPair: {pair[0]}/{pair[1]}\n"
        f"Mode: plain\nStatus: survived\nRecord file: runs/{run_id}/records/{record_id}.json\n"
        f"Statement: {statement}"
    )


def start(root: Path, m: RunManifest, settings: DriverSettings | None = None) -> RecordStore:
    store = RecordStore(root)
    start_run(store, m, settings or DriverSettings(conjectures_per_family=1))
    return store


# --- stage order ---------------------------------------------------------------


def test_stage_order_matches_spec(tmp_path: Path) -> None:
    # doc-gmj, "Records and stages": generator, falsifier, known-result filter,
    # formaliser, triviality screen, prover, scorer, human queue. The encoding
    # runs between generation and the falsifier ("Encoding": "Before the
    # falsifier runs").
    assert STAGE_ORDER == (
        Stage.generator,
        Stage.encoding,
        Stage.falsifier,
        Stage.literature,
        Stage.formaliser,
        Stage.triviality,
        Stage.prover,
        Stage.scorer,
        Stage.queue,
    )
    store = start(tmp_path, manifest(families=(Family.claude, Family.codex)))
    stages = FakeStages()
    result = drive(
        store, RUN, runners=stages.runners(), generator=FakeGenerator(["survive", "proved"])
    )
    assert result.finished
    for record in store.list(RUN):
        called = [stage for rid, stage in stages.calls if rid == record.id]
        assert called == list(RECORD_STAGES)
        assert list(record.stages)[: len(RECORD_STAGES) + 1] == [Stage.generator, *RECORD_STAGES]
        # Both outcomes are queue items, so both are scored.
        assert Stage.scorer in record.stages


def test_record_stops_at_first_final_status(tmp_path: Path) -> None:
    store = start(tmp_path, manifest(families=(Family.claude, Family.codex, Family.gemini)))
    stages = FakeStages()
    drive(
        store,
        RUN,
        runners=stages.runners(),
        generator=FakeGenerator(["kill", "known", "trivial"]),
    )
    by_outcome = {r.statement.removeprefix("outcome:"): r for r in store.list(RUN)}
    assert list(by_outcome["kill"].stages) == [Stage.generator, Stage.encoding, Stage.falsifier]
    assert by_outcome["known"].status is Status.known
    assert Stage.formaliser not in by_outcome["known"].stages
    assert by_outcome["trivial"].status is Status.trivial
    assert Stage.prover not in by_outcome["trivial"].stages
    # A kill without a passed mechanism is not queued, so it is not scored.
    assert Stage.scorer not in by_outcome["kill"].stages


def test_next_stage() -> None:
    record = Record(
        id=f"{RUN}.r1",
        run_id=RUN,
        pair=PAIR,
        mode="plain",
        statement="s",
        objects=[],
        assumptions=[],
        domain_of_validity="d",
        instance_family="f",
        provenance=Provenance(family=Family.claude, model="m", prompt_hash="h"),
    )
    assert next_stage(record) is Stage.encoding
    encoded = record.model_copy(update={"stages": {Stage.encoding: out()}})
    assert next_stage(encoded) is Stage.falsifier
    ambiguous = encoded.model_copy(update={"status": Status.ambiguous})
    assert next_stage(ambiguous) is None
    untestable = record.model_copy(update={"status": Status.untestable})
    assert next_stage(untestable) is None
    # A proposed record with a falsifier stage is not something the driver can
    # move on: the falsifier always sets killed or survived.
    odd = record.model_copy(update={"stages": {Stage.encoding: out(), Stage.falsifier: out()}})
    assert next_stage(odd) is None
    survived = record.model_copy(
        update={
            "status": Status.survived,
            "stages": {s: out() for s in RECORD_STAGES[:3]},
        }
    )
    assert next_stage(survived) is Stage.formaliser


# --- resume ---------------------------------------------------------------------


def test_run_resume_after_failure(tmp_path: Path) -> None:
    families = (Family.claude, Family.codex, Family.gemini, Family.local)
    store = start(tmp_path, manifest(families=families))
    stages = FakeStages()
    generator = FakeGenerator(["survive", "proved", "survive"], fail_first={Family.gemini})
    runners = stages.runners()

    # Generation for Gemini fails (usage limit): the run stops before any stage runs.
    first = drive(store, RUN, runners=runners, generator=generator)
    assert not first.finished
    assert first.stopped is not None and "usage limit" in first.stopped
    made = store.list(RUN)
    assert len(made) == 2
    assert stages.calls == []

    # On resume, the existing records run first; a model failure halfway
    # through the second one's stages stops the run before Gemini's batch.
    gemini_id = f"{RUN}.g-gemini-info-4"
    stages.fail[(made[1].id, Stage.literature)] = ModelUnavailable("codex usage limit")
    second = drive(store, RUN, runners=runners, generator=generator)
    assert not second.finished
    assert len(generator.calls) == 3
    assert next_stage(store.load(made[0].id)) is None
    halted = store.load(made[1].id)
    assert halted.status is Status.survived
    assert Stage.literature not in halted.stages

    # A second resume finishes: Gemini's batch runs again, Claude's and Codex's
    # do not, and every stage of every record is written exactly once.
    third = drive(store, RUN, runners=runners, generator=generator)
    assert third.finished
    assert [c[1] for c in generator.calls] == [
        Family.claude,
        Family.codex,
        Family.gemini,
        Family.gemini,
    ]
    assert len(stages.written) == len(set(stages.written))
    records = store.list(RUN)
    assert {r.id for r in records} == {made[0].id, made[1].id, gemini_id}
    for record in records:
        assert next_stage(record) is None
        assert [s for rid, s in stages.written if rid == record.id] == list(RECORD_STAGES)


def test_generation_batch_with_records_is_not_rerun(tmp_path: Path) -> None:
    # A crash after the records were stored but before the batch was logged as
    # done must not generate the batch again.
    store = start(tmp_path, manifest(families=(Family.claude, Family.local)))
    generator = FakeGenerator(["survive", "survive"])
    record = generator(PAIR, Family.claude, 1, (), StageContext.for_run(store, RUN))
    assert record
    drive(store, RUN, runners=FakeStages().runners(), generator=generator)
    assert len(generator.calls) == 1
    assert len(store.list(RUN)) == 1


def test_budget_exceeded_stops_only_that_pair(tmp_path: Path) -> None:
    store = start(tmp_path, manifest(pairs=(PAIR, OTHER), families=(Family.claude, Family.local)))
    stages = FakeStages()
    generator = FakeGenerator(["survive", "survive"])
    drive(store, RUN, runners={}, generator=generator)  # generate only
    first, second = sorted(store.list(RUN), key=lambda r: r.pair != PAIR)
    assert first.pair == PAIR and second.pair == OTHER
    stages.fail[(first.id, Stage.falsifier)] = BudgetExceeded("pair spent")
    result = drive(store, RUN, runners=stages.runners(), generator=generator)
    assert result.over_budget == [PAIR]
    # The budget is a clean stop, not a failure: the run counts as finished.
    assert result.finished
    assert next_stage(store.load(second.id)) is None
    assert store.load(first.id).status is Status.proposed
    assert (first.id, Stage.literature) not in stages.calls


def test_unchecked_literature_skips_the_record_and_goes_on(tmp_path: Path) -> None:
    from minilab.literature.filter import LiteratureOutput, LiteratureUnchecked

    store = start(tmp_path, manifest(families=(Family.claude, Family.codex)))
    stages = FakeStages()
    generator = FakeGenerator(["survive", "survive"])
    drive(store, RUN, runners={}, generator=generator)
    first, second = store.list(RUN)
    stages.fail[(first.id, Stage.literature)] = LiteratureUnchecked(
        LiteratureOutput.model_construct(notes="arXiv is down")
    )
    result = drive(store, RUN, runners=stages.runners(), generator=generator)
    assert not result.finished
    assert result.stopped is None
    assert [(s.record_id, s.stage) for s in result.skipped] == [(first.id, Stage.literature)]
    assert "arXiv is down" in result.skipped[0].error
    assert Stage.literature not in store.load(first.id).stages
    assert next_stage(store.load(second.id)) is None
    # Resume tries the skipped stage again.
    again = drive(store, RUN, runners=stages.runners(), generator=generator)
    assert again.finished
    assert next_stage(store.load(first.id)) is None


def test_unexpected_error_is_not_swallowed(tmp_path: Path) -> None:
    store = start(tmp_path, manifest(families=(Family.claude, Family.local)))
    stages = FakeStages()
    generator = FakeGenerator(["survive"])
    drive(store, RUN, runners={}, generator=generator)
    (record,) = store.list(RUN)
    stages.fail[(record.id, Stage.encoding)] = RuntimeError("a bug")
    with pytest.raises(RuntimeError, match="a bug"):
        drive(store, RUN, runners=stages.runners(), generator=generator)


def test_ambiguous_child_is_run(tmp_path: Path) -> None:
    store = start(tmp_path, manifest(families=(Family.claude, Family.local)))
    stages = FakeStages()
    runners = dict(stages.runners())
    plain_encoding = runners[Stage.encoding]

    def encoding_with_child(record_id: str, ctx: StageContext) -> None:
        plain_encoding(record_id, ctx)
        parent = ctx.ledger.store.load(record_id)
        if parent.status is Status.ambiguous:
            child = parent.model_copy(
                update={
                    "id": f"{record_id}-c1",
                    "statement": "outcome:survive",
                    "status": Status.proposed,
                    "stages": {Stage.generator: parent.stages[Stage.generator]},
                    "provenance": parent.provenance.model_copy(update={"parents": [record_id]}),
                }
            )
            ctx.ledger.store.create(child)

    runners[Stage.encoding] = encoding_with_child
    result = drive(store, RUN, runners=runners, generator=FakeGenerator(["ambiguous"]))
    assert result.finished
    child = store.load(f"{RUN}.g-claude-info-1-c1")
    assert child.status is Status.survived
    assert next_stage(child) is None


def test_run_lock_refuses_a_second_driver(tmp_path: Path) -> None:
    from minilab.pipeline import RunBusy, run_lock

    store = start(tmp_path, manifest(families=(Family.claude, Family.local)))
    with run_lock(store, RUN):
        with pytest.raises(RunBusy):
            drive(store, RUN, runners={}, generator=FakeGenerator([]))


# --- memory policy ----------------------------------------------------------------


def test_blind_policy_makes_no_reads(tmp_path: Path) -> None:
    m = manifest(families=(Family.claude, Family.codex))
    store = start(tmp_path, m)
    spy = SpyTransport([{"id": "x", "content": memory_content("old.r1", "an old claim")}])
    memory = MemoryClient(m, spy)
    generator = FakeGenerator(["survive", "kill"])
    result = drive(store, RUN, runners=FakeStages().runners(), generator=generator, memory=memory)
    assert result.finished
    assert spy.reads() == []
    assert all(earlier == () for *_, earlier in generator.calls)
    with pytest.raises(MemoryReadRefused):
        memory.recall(PAIR)
    assert spy.reads() == []
    # Writes still happen under blind: one per finished record.
    written = [a for n, a in spy.calls if n == "muninn_remember"]
    assert len(written) == 2


def test_memory_before_passed_through(tmp_path: Path) -> None:
    before = datetime(2026, 7, 1, tzinfo=UTC)
    m = manifest(
        policy=MemoryPolicy.pair, memory_before=before, families=(Family.claude, Family.local)
    )
    store = start(tmp_path, m)
    spy = SpyTransport(
        [
            {
                "id": "a",
                "content": memory_content("old.r1", "early claim"),
                "created_at": "2026-06-01T00:00:00Z",
            },
            # Created after the cap: a server that ignored `before` must not leak it.
            {
                "id": "b",
                "content": memory_content("old.r2", "late claim"),
                "created_at": "2026-08-01T00:00:00Z",
            },
            # No creation time: it cannot be shown to be before the cap.
            {"id": "c", "content": memory_content("old.r3", "undated claim")},
            # Another pair: a server that ignored the tag filter must not leak it.
            {
                "id": "d",
                "content": memory_content("old.r4", "other pair", pair=OTHER),
                "created_at": "2026-06-01T00:00:00Z",
            },
            # This run's own record.
            {
                "id": "e",
                "content": memory_content("run-1.r5", "own run", run_id=RUN),
                "created_at": "2026-06-01T00:00:00Z",
            },
        ]
    )
    generator = FakeGenerator(["survive"])
    drive(store, RUN, runners={}, generator=generator, memory=MemoryClient(m, spy))
    ((name, args),) = spy.reads()
    assert name == "muninn_recall"
    assert args["vault"] == VAULT
    assert args["before"] == "2026-07-01T00:00:00Z"
    assert args["tags_all"] == ["minilab-record", pair_tag(PAIR)]
    assert args["read_only"] is True
    assert generator.calls[0][3] == ("early claim",)


def test_all_policy_reads_every_pair(tmp_path: Path) -> None:
    m = manifest(policy=MemoryPolicy.all, families=(Family.claude, Family.local))
    store = start(tmp_path, m)
    spy = SpyTransport(
        [
            {"id": "a", "content": memory_content("old.r1", "this pair")},
            {"id": "b", "content": memory_content("old.r2", "other pair", pair=OTHER)},
        ]
    )
    generator = FakeGenerator(["survive"])
    drive(store, RUN, runners={}, generator=generator, memory=MemoryClient(m, spy))
    ((_, args),) = spy.reads()
    assert args["tags_all"] == ["minilab-record"]
    assert "before" not in args
    assert generator.calls[0][3] == ("this pair", "other pair")


def test_never_reads_sci_vault(tmp_path: Path) -> None:
    for policy in MemoryPolicy:
        root = tmp_path / policy.value
        m = manifest(policy=policy, families=(Family.claude, Family.local))
        store = start(root, m)
        spy = SpyTransport([{"id": "a", "content": memory_content("old.r1", "claim")}])
        drive(
            store,
            RUN,
            runners=FakeStages().runners(),
            generator=FakeGenerator(["survive"]),
            memory=MemoryClient(m, spy),
        )
        assert spy.calls, policy
        assert {str(a["vault"]) for _, a in spy.calls} == {"minilab"}
    assert VAULT == "minilab"


def test_non_blind_run_needs_a_memory_client(tmp_path: Path) -> None:
    m = manifest(policy=MemoryPolicy.pair, families=(Family.claude, Family.local))
    store = start(tmp_path, m)
    with pytest.raises(ValueError, match="memory"):
        drive(store, RUN, runners={}, generator=FakeGenerator(["survive"]))


def test_memory_outage_during_recall_stops_the_run(tmp_path: Path) -> None:
    m = manifest(policy=MemoryPolicy.pair, families=(Family.claude, Family.local))
    store = start(tmp_path, m)
    spy = SpyTransport()
    spy.down = True
    generator = FakeGenerator(["survive"])
    result = drive(store, RUN, runners={}, generator=generator, memory=MemoryClient(m, spy))
    assert not result.finished
    assert result.stopped is not None and "MuninnDB" in result.stopped
    assert generator.calls == []


def test_memory_sync_once_per_record_and_retried_after_outage(tmp_path: Path) -> None:
    m = manifest(families=(Family.claude, Family.codex))
    store = start(tmp_path, m)
    spy = SpyTransport()
    spy.down = True
    stages = FakeStages()
    generator = FakeGenerator(["survive", "known"])
    first = drive(
        store, RUN, runners=stages.runners(), generator=generator, memory=MemoryClient(m, spy)
    )
    assert first.unsynced == 2
    spy.down = False
    second = drive(
        store, RUN, runners=stages.runners(), generator=generator, memory=MemoryClient(m, spy)
    )
    assert second.unsynced == 0
    third = drive(
        store, RUN, runners=stages.runners(), generator=generator, memory=MemoryClient(m, spy)
    )
    assert third.unsynced == 0
    written = [a for n, a in spy.calls if n == "muninn_remember"]
    # Two failed attempts on the first drive, then one write each.
    assert len(written) == 4
    ok = written[2:]
    assert {str(a["op_id"]) for a in ok} == {f"minilab-record:{r.id}" for r in store.list(RUN)}
    for args in ok:
        assert args["upsert_mode"] is True
        tags = args["tags"]
        assert isinstance(tags, list)
        assert "minilab-record" in tags and pair_tag(PAIR) in tags and f"run:{RUN}" in tags
        content = args["content"]
        assert isinstance(content, str) and f"Run: {RUN}" in content


# --- benchmark checks -----------------------------------------------------------------


ISOLATED: dict[str, dict[str, JsonValue]] = {
    f.value: {"status": "isolated", "benchmark": True} for f in Family
}


def benchmark_manifest(**changes: object) -> RunManifest:
    base = manifest(literature_before=BENCHMARK_LITERATURE_CAP)
    return base.model_copy(update=changes)


def test_benchmark_accepts_isolated_blind_capped_run() -> None:
    check_benchmark(benchmark_manifest(), ISOLATED)


def test_benchmark_refuses_family_not_isolated() -> None:
    isolation = {**ISOLATED, "gemini": {"status": "normal runs only", "benchmark": False}}
    with pytest.raises(BenchmarkRefused, match="gemini"):
        check_benchmark(benchmark_manifest(), isolation)
    missing = {k: v for k, v in ISOLATED.items() if k != "codex"}
    with pytest.raises(BenchmarkRefused, match="codex"):
        check_benchmark(benchmark_manifest(), missing)


def test_benchmark_refuses_reads_and_late_or_missing_cap() -> None:
    with pytest.raises(BenchmarkRefused, match="blind"):
        check_benchmark(benchmark_manifest(memory_policy=MemoryPolicy.all), ISOLATED)
    with pytest.raises(BenchmarkRefused, match="literature"):
        check_benchmark(benchmark_manifest(literature_before=None), ISOLATED)
    with pytest.raises(BenchmarkRefused, match="literature"):
        check_benchmark(benchmark_manifest(literature_before=date(2026, 7, 1)), ISOLATED)


def test_benchmark_refuses_more_than_eight_pairs() -> None:
    from minilab.scheduler import all_pairs

    with pytest.raises(BenchmarkRefused, match="8"):
        check_benchmark(benchmark_manifest(pairs=tuple(all_pairs()[:9])), ISOLATED)


def test_start_run_checks_benchmark_before_saving(tmp_path: Path) -> None:
    store = RecordStore(tmp_path)
    with pytest.raises(BenchmarkRefused):
        start_run(store, manifest(), DriverSettings(benchmark=True))
    assert not (tmp_path / "runs" / RUN).exists()


# --- command line ---------------------------------------------------------------------


def test_cli_report_prints_the_cost_report(tmp_path: Path) -> None:
    store = start(tmp_path, manifest(families=(Family.claude, Family.local)))
    drive(store, RUN, runners={}, generator=FakeGenerator([]))
    result = CliRunner().invoke(app, ["report", RUN, "--root", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "reviewed survivors: 0" in result.output
    assert "tokens per reviewed survivor: none reviewed" in result.output


def test_cli_run_refuses_a_benchmark_run_before_any_call(tmp_path: Path) -> None:
    pairs = tmp_path / "pairs.json"
    pairs.write_text(json.dumps([list(PAIR)]))
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--root",
            str(tmp_path),
            "--run-id",
            "bench-1",
            "--mode",
            "plain",
            "--pairs",
            str(pairs),
            "--memory",
            "pair",
            "--budget-tokens-per-pair",
            "1000",
            "--benchmark",
        ],
    )
    assert result.exit_code == 1, result.output
    assert "blind" in result.output
    assert not (tmp_path / "runs" / "bench-1").exists()


def test_cli_run_pairs_auto_uses_the_scheduler(tmp_path: Path) -> None:
    from minilab.cli import choose_pairs

    state = tmp_path / "state" / "pairs.json"
    picked = choose_pairs("auto", count=3, state_path=state, seed=1)
    assert len(set(picked)) == 3
    again = choose_pairs("auto", count=3, state_path=tmp_path / "other.json", seed=1)
    assert again == picked
    with pytest.raises(ValueError):
        choose_pairs("auto", count=46, state_path=state, seed=1)


def test_cli_pairs_file(tmp_path: Path) -> None:
    from minilab.cli import choose_pairs

    path = tmp_path / "pairs.json"
    path.write_text(
        json.dumps([["optimisation", "information-theory"], ["kernels", "probability"]])
    )
    assert choose_pairs(str(path), count=1, state_path=tmp_path / "s.json", seed=0) == [
        ("information-theory", "optimisation"),
        ("kernels", "probability"),
    ]
    path.write_text(json.dumps([["optimisation", "not-a-domain"]]))
    with pytest.raises(ValueError):
        choose_pairs(str(path), count=1, state_path=tmp_path / "s.json", seed=0)


# --- guards found by review ------------------------------------------------------------


def test_logged_batch_with_no_records_is_not_rerun(tmp_path: Path) -> None:
    # A reply that could not be read makes no records; the batch is still done.
    store = start(tmp_path, manifest(families=(Family.claude, Family.local)))
    generator = FakeGenerator([])
    drive(store, RUN, runners={}, generator=generator)
    drive(store, RUN, runners={}, generator=generator)
    assert len(generator.calls) == 1


def test_a_record_that_stopped_the_run_goes_last(tmp_path: Path) -> None:
    store = start(tmp_path, manifest(families=(Family.claude, Family.codex)))
    stages = FakeStages()
    generator = FakeGenerator(["survive", "survive"])
    drive(store, RUN, runners={}, generator=generator)
    first, second = store.list(RUN)
    for _ in range(3):
        stages.fail[(first.id, Stage.encoding)] = ModelUnavailable("call timed out")
        result = drive(store, RUN, runners=stages.runners(), generator=generator)
        assert result.stopped is not None
    # Without the reordering, the first record would fail first every time.
    assert next_stage(store.load(second.id)) is None
    assert drive(store, RUN, runners=stages.runners(), generator=generator).finished


def test_runner_that_writes_nothing_is_a_bug(tmp_path: Path) -> None:
    store = start(tmp_path, manifest(families=(Family.claude, Family.local)))
    drive(store, RUN, runners={}, generator=FakeGenerator(["survive"]))
    runners = dict(FakeStages().runners())
    runners[Stage.encoding] = lambda record_id, ctx: None
    with pytest.raises(RuntimeError, match="without writing"):
        drive(store, RUN, runners=runners, generator=FakeGenerator([]))


@pytest.mark.parametrize(
    "error",
    [
        pytest.param("SearchFailed", id="search"),
        pytest.param("CertificationFailed", id="certification"),
        pytest.param("CertificateUnavailable", id="certificate"),
        pytest.param("LeanTimeout", id="lean-timeout"),
        pytest.param("NoChecker", id="no-checker"),
        pytest.param("NoFormaliser", id="no-formaliser"),
    ],
)
def test_record_failures_skip_the_record(tmp_path: Path, error: str) -> None:
    from minilab import pipeline

    kind = next(k for k in pipeline.SKIP_RECORD if k.__name__ == error)
    store = start(tmp_path, manifest(families=(Family.claude, Family.codex)))
    stages = FakeStages()
    generator = FakeGenerator(["survive", "survive"])
    drive(store, RUN, runners={}, generator=generator)
    first, second = store.list(RUN)
    stages.fail[(first.id, Stage.falsifier)] = kind("failed")
    result = drive(store, RUN, runners=stages.runners(), generator=generator)
    assert result.stopped is None
    assert [s.record_id for s in result.skipped] == [first.id]
    assert next_stage(store.load(second.id)) is None


@pytest.mark.parametrize("error", ["SandboxUnavailable", "LeanUnavailable"])
def test_tool_failures_stop_the_run(tmp_path: Path, error: str) -> None:
    from minilab import pipeline

    kind = next(k for k in pipeline.STOP_RUN if k.__name__ == error)
    store = start(tmp_path, manifest(families=(Family.claude, Family.codex)))
    stages = FakeStages()
    generator = FakeGenerator(["survive", "survive"])
    drive(store, RUN, runners={}, generator=generator)
    first, second = store.list(RUN)
    stages.fail[(first.id, Stage.encoding)] = kind("tool down")
    result = drive(store, RUN, runners=stages.runners(), generator=generator)
    assert result.stopped is not None and "tool down" in result.stopped
    assert Stage.encoding not in store.load(second.id).stages


def test_ambiguous_parent_whose_child_was_lost_gets_it(tmp_path: Path) -> None:
    store = start(tmp_path, manifest(families=(Family.claude, Family.local)))
    drive(store, RUN, runners={}, generator=FakeGenerator(["survive"]))
    (record,) = store.list(RUN)
    sharper: dict[str, JsonValue] = {
        "statement": "outcome:survive, sharper",
        "objects": ["x"],
        "assumptions": ["x > 0"],
        "domain_of_validity": "x > 0",
        "instance_family": "x: a rational",
    }
    # The encoding stage wrote the parent, then the process died before make_child.
    store.write_stage(
        record.id, Stage.encoding, out(agreed=False, sharper=sharper), status=Status.ambiguous
    )
    result = drive(store, RUN, runners=FakeStages().runners(), generator=FakeGenerator([]))
    assert result.finished
    children = [r for r in store.list(RUN) if r.provenance.parents == [record.id]]
    assert len(children) == 1
    assert children[0].status is Status.survived


def test_scorer_waits_until_the_record_is_done(tmp_path: Path) -> None:
    from minilab.literature.filter import LiteratureOutput, LiteratureUnchecked

    store = start(tmp_path, manifest(families=(Family.claude, Family.local)))
    stages = FakeStages()
    drive(store, RUN, runners={}, generator=FakeGenerator(["proved"]))
    (record,) = store.list(RUN)
    stages.fail[(record.id, Stage.literature)] = LiteratureUnchecked(
        LiteratureOutput.model_construct(notes="down")
    )
    drive(store, RUN, runners=stages.runners(), generator=FakeGenerator([]))
    assert Stage.scorer not in store.load(record.id).stages
    drive(store, RUN, runners=stages.runners(), generator=FakeGenerator([]))
    done = store.load(record.id)
    assert done.status is Status.proved
    assert done.stages[Stage.scorer].data["kind"] == "proved"


def test_missing_settings_refuse_to_drive(tmp_path: Path) -> None:
    store = start(tmp_path, manifest(families=(Family.claude, Family.local)))
    (tmp_path / "runs" / RUN / "driver" / "settings.json").unlink()
    with pytest.raises(FileNotFoundError):
        drive(store, RUN, runners={}, generator=FakeGenerator([]))


def test_benchmark_is_checked_again_on_resume(tmp_path: Path) -> None:
    from minilab import pipeline

    m = manifest(literature_before=BENCHMARK_LITERATURE_CAP)
    store = RecordStore(tmp_path)
    pipeline_isolation = pipeline.load_isolation
    start_run(store, m, DriverSettings(benchmark=True))
    try:
        pipeline.load_isolation = lambda: {  # type: ignore[assignment]
            **ISOLATED,
            "gemini": {"status": "normal runs only", "benchmark": False},
        }
        with pytest.raises(BenchmarkRefused, match="gemini"):
            drive(store, RUN, runners={}, generator=FakeGenerator([]))
    finally:
        pipeline.load_isolation = pipeline_isolation


def test_memory_client_must_hold_the_runs_manifest(tmp_path: Path) -> None:
    m = manifest(policy=MemoryPolicy.pair, families=(Family.claude, Family.local))
    store = start(tmp_path, m)
    wider = MemoryClient(m.model_copy(update={"memory_policy": MemoryPolicy.all}), SpyTransport())
    with pytest.raises(ValueError, match="manifest"):
        drive(store, RUN, runners={}, generator=FakeGenerator([]), memory=wider)


def test_superseded_memories_are_dropped(tmp_path: Path) -> None:
    m = manifest(policy=MemoryPolicy.all, families=(Family.claude, Family.local))
    spy = SpyTransport(
        [
            {
                "id": "a",
                "content": memory_content("old.r1", "old form"),
                "annotations": {"superseded_by": "b"},
            },
            {"id": "b", "content": memory_content("old.r1", "new form")},
        ]
    )
    assert [e.statement for e in MemoryClient(m, spy).recall(PAIR)] == ["new form"]
