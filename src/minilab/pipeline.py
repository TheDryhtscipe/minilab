"""The pipeline driver: run a run's records through the stages, and resume it.

Design doc-gmj, "Records and stages"; build plan doc-5wh, Task 16. `start_run`
writes a run's manifest and driver settings. `drive` then does the work, and
can be called again (`minilab resume`) to continue after any stop:

1. **Generate.** For each pair in the manifest and each generating family
   in it (Claude, Codex, Gemini; not the local model), it asks for
   `conjectures_per_family` conjectures, once. A batch counts as done when
   it is logged in `runs/<run_id>/driver/generated.jsonl`, or when a record
   from it exists (a crash between storing records and logging the batch
   must not generate it again). Under the `pair` or `all` memory policy, the
   pair's earlier records are recalled first and their statements go into
   the prompt; under `blind`, nothing is read.
2. **Run each record's stages** in `RECORD_STAGES` order: encoding,
   falsifier, literature, formaliser, triviality, prover. Each record
   stops at its first final status, or when it is `ambiguous` (its sharper
   child record, made by the encoding stage, is run instead). Child records
   that appear while the driver runs are picked up in the same call.
3. **Score** every queue item that has no scorer stage.
4. **Sync memory:** one MuninnDB memory per finished record (`memory.py`).

What stops what:

- A model failure (`ModelUnavailable`, including a usage limit), an
  unusable sandbox, Lean not starting, or MuninnDB failing on a read stops
  the whole run. The record being worked on is left as it was, since every
  stage raises before it writes. `RunOutcome.stopped` says why.
- `BudgetExceeded` stops that pair only. Its records stay where they are;
  the other pairs go on. A run whose only unfinished work is on spent pairs
  counts as finished ("stops cleanly on budget").
- A failure that belongs to one record (the literature sources being
  down, a search or certificate run that failed, a Lean time-out, no checker
  family) skips that record for the rest of this call and is listed in
  `RunOutcome.skipped`. A later `drive` tries it again.
- Anything else is a bug and is raised.

Only one driver may work on a run at a time (`run_lock`).
"""

import fcntl
import json
import os
from collections.abc import Callable, Mapping, Sequence
from collections.abc import Generator as GeneratorType
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from importlib import resources
from pathlib import Path
from typing import Annotated, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from minilab.encoding.agree import SandboxUnavailable, run_encoding_stage
from minilab.falsify.search import CertificateUnavailable
from minilab.falsify.stage import CertificationFailed, SearchFailed, run_falsifier_stage
from minilab.formalise.formaliser import LeanTimeout, NoFormaliser
from minilab.formalise.stage import run_formaliser_stage
from minilab.generate.core import GENERATOR_FAMILIES
from minilab.generate.plain import generate_plain
from minilab.generate.structured import generate_structured
from minilab.lean.runner import LeanUnavailable
from minilab.ledger import Ledger
from minilab.literature.filter import LiteratureUnchecked, run_literature_stage
from minilab.memory import MemoryClient, MemoryUnavailable
from minilab.models.adapter import Budget, BudgetExceeded, NoChecker
from minilab.models.backends import Backend, ModelUnavailable
from minilab.prove.loop import run_prover_stage
from minilab.queue import queue_items
from minilab.records import (
    FINAL_STATUSES,
    Family,
    Record,
    RecordStore,
    Stage,
    StageOutput,
    Status,
    check_run_id,
)
from minilab.run import MemoryPolicy, RunManifest, load_manifest, manifest_path, save_manifest
from minilab.scheduler import Pair, canonical_pair
from minilab.screen.triviality import run_triviality_stage

__all__ = [
    "BENCHMARK_LITERATURE_CAP",
    "BENCHMARK_MAX_PAIRS",
    "RECORD_STAGES",
    "STAGE_ORDER",
    "BenchmarkRefused",
    "DriverSettings",
    "Generator",
    "RunBusy",
    "RunOutcome",
    "Skipped",
    "StageContext",
    "StageRunner",
    "check_benchmark",
    "default_generator",
    "default_runners",
    "drive",
    "load_isolation",
    "load_settings",
    "next_stage",
    "run_lock",
    "start_run",
]

STAGE_ORDER: tuple[Stage, ...] = (
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
"""Every stage, in the design's order (doc-gmj, "Records and stages")."""

RECORD_STAGES: tuple[Stage, ...] = STAGE_ORDER[1:7]
"""The stages the driver runs on each record, after generation and before scoring."""

_PROPOSED_STAGES: frozenset[Stage] = frozenset({Stage.encoding, Stage.falsifier})
"""The stages that take a `proposed` record. The falsifier always sets killed
or survived, so the later stages take only `survived` records."""

BENCHMARK_LITERATURE_CAP = date(2026, 6, 30)
"""A benchmark run's literature date cap may be no later than this (doc-gmj, "Blinding")."""

BENCHMARK_MAX_PAIRS = 8
"""A benchmark run covers at most this many domain pairs."""

STOP_RUN: tuple[type[BaseException], ...] = (
    ModelUnavailable,
    SandboxUnavailable,
    LeanUnavailable,
)
"""Failures that stop the whole run: the next record would fail the same way."""

SKIP_RECORD: tuple[type[BaseException], ...] = (
    LiteratureUnchecked,
    SearchFailed,
    CertificationFailed,
    CertificateUnavailable,
    LeanTimeout,
    NoChecker,
    NoFormaliser,
)
"""Failures that skip one record for the rest of a `drive` call."""


class BenchmarkRefused(Exception):
    """The run does not meet the benchmark's blinding rules."""


class RunBusy(Exception):
    """Another driver is working on this run."""


class DriverSettings(BaseModel):
    """How the driver runs one run, saved with it so `resume` runs it the same way."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    conjectures_per_family: Annotated[int, Field(ge=1, le=50)] = 4
    """Conjectures asked of each generating family for each pair."""
    seed: int = 0
    """Seeds the falsifier's search."""
    benchmark: bool = False
    """A benchmark run: `check_benchmark` must pass when it starts and on every resume."""


def _driver_dir(root: Path, run_id: str) -> Path:
    return root / "runs" / check_run_id(run_id) / "driver"


def _settings_path(root: Path, run_id: str) -> Path:
    return _driver_dir(root, run_id) / "settings.json"


def load_settings(store: RecordStore, run_id: str) -> DriverSettings:
    """Read a run's driver settings (written by `start_run`)."""
    return DriverSettings.model_validate_json(_settings_path(store.root, run_id).read_bytes())


@dataclass(frozen=True)
class StageContext:
    """What every stage runner gets besides the record id."""

    ledger: Ledger
    budget: Budget
    backends: Mapping[Family, Backend] | None
    settings: DriverSettings

    @classmethod
    def for_run(
        cls,
        store: RecordStore,
        run_id: str,
        backends: Mapping[Family, Backend] | None = None,
    ) -> "StageContext":
        ledger = Ledger(store, run_id)
        path = _settings_path(store.root, run_id)
        settings = load_settings(store, run_id) if path.exists() else DriverSettings()
        return cls(ledger, Budget.from_manifest(ledger.manifest), backends, settings)


StageRunner = Callable[[str, StageContext], object]
"""Runs one stage on one record and writes it, or raises before writing."""

Generator = Callable[[Pair, Family, int, Sequence[str], StageContext], Sequence[Record]]
"""Makes and stores one batch of records: pair, family, count, earlier statements."""


def default_runners() -> dict[Stage, StageRunner]:
    """The real stage functions, with the run's ledger, budget, and backends."""

    def encoding(record_id: str, ctx: StageContext) -> object:
        return run_encoding_stage(
            record_id, ledger=ctx.ledger, budget=ctx.budget, backends=ctx.backends
        )

    def falsifier(record_id: str, ctx: StageContext) -> object:
        return run_falsifier_stage(
            record_id,
            ledger=ctx.ledger,
            budget=ctx.budget,
            backends=ctx.backends,
            seed=ctx.settings.seed,
        )

    def literature(record_id: str, ctx: StageContext) -> object:
        return run_literature_stage(
            record_id, ledger=ctx.ledger, budget=ctx.budget, backends=ctx.backends
        )

    def formaliser(record_id: str, ctx: StageContext) -> object:
        return run_formaliser_stage(
            record_id, ledger=ctx.ledger, budget=ctx.budget, backends=ctx.backends
        )

    def triviality(record_id: str, ctx: StageContext) -> object:
        return run_triviality_stage(
            record_id, ledger=ctx.ledger, budget=ctx.budget, backends=ctx.backends
        )

    def prover(record_id: str, ctx: StageContext) -> object:
        return run_prover_stage(
            record_id, ledger=ctx.ledger, budget=ctx.budget, backends=ctx.backends
        )

    return {
        Stage.encoding: encoding,
        Stage.falsifier: falsifier,
        Stage.literature: literature,
        Stage.formaliser: formaliser,
        Stage.triviality: triviality,
        Stage.prover: prover,
    }


def default_generator(
    pair: Pair, family: Family, n: int, earlier: Sequence[str], ctx: StageContext
) -> Sequence[Record]:
    """Generate in the run's mode (plain or structured)."""
    generate = generate_plain if ctx.ledger.manifest.mode == "plain" else generate_structured
    return generate(
        pair,
        family,
        n,
        ledger=ctx.ledger,
        budget=ctx.budget,
        backends=ctx.backends,
        earlier=earlier,
    )


def next_stage(record: Record) -> Stage | None:
    """The stage the driver runs next on `record`, or None if the record is done.

    A record is done at a final status, when `ambiguous` (its child goes on),
    and when every stage in `RECORD_STAGES` is written.
    """
    if record.status in FINAL_STATUSES or record.status not in (
        Status.proposed,
        Status.survived,
    ):
        return None
    for stage in RECORD_STAGES:
        if stage not in record.stages:
            if record.status is Status.proposed and stage not in _PROPOSED_STAGES:
                return None
            return stage
    return None


def load_isolation() -> dict[str, dict[str, JsonValue]]:
    """Read `minilab/models/isolation.json`: each family's isolation probe result."""
    text = resources.files("minilab.models").joinpath("isolation.json").read_text("utf-8")
    raw: JsonValue = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError("isolation.json does not hold a JSON object")
    table: dict[str, dict[str, JsonValue]] = {}
    for family, entry in raw.items():
        if isinstance(entry, dict):
            table[family] = entry
    return table


def check_benchmark(
    manifest: RunManifest, isolation: Mapping[str, Mapping[str, JsonValue]] | None = None
) -> None:
    """Raise `BenchmarkRefused` unless the run meets the benchmark's blinding rules.

    Design doc-gmj, "Blinding": memory `blind`; a literature date cap no later
    than `BENCHMARK_LITERATURE_CAP`; at most `BENCHMARK_MAX_PAIRS` pairs; and
    only families whose isolation probe passed (`isolation.json` status
    `isolated` and `benchmark` true).
    """
    if manifest.memory_policy is not MemoryPolicy.blind:
        raise BenchmarkRefused(
            f"a benchmark run must use memory policy blind, not {manifest.memory_policy}"
        )
    cap = manifest.literature_before
    if cap is None or cap > BENCHMARK_LITERATURE_CAP:
        raise BenchmarkRefused(
            f"a benchmark run needs a literature date cap on or before "
            f"{BENCHMARK_LITERATURE_CAP.isoformat()}, not {cap}"
        )
    if len(manifest.pairs) > BENCHMARK_MAX_PAIRS:
        raise BenchmarkRefused(
            f"a benchmark run covers at most {BENCHMARK_MAX_PAIRS} pairs, not {len(manifest.pairs)}"
        )
    table = load_isolation() if isolation is None else isolation
    for family in manifest.families:
        entry = table.get(family.value)
        if entry is None or entry.get("status") != "isolated" or entry.get("benchmark") is not True:
            status = None if entry is None else entry.get("status")
            raise BenchmarkRefused(
                f"{family.value} did not pass the isolation probe (status {status!r}); "
                "leave it out of benchmark runs"
            )


def start_run(store: RecordStore, manifest: RunManifest, settings: DriverSettings) -> None:
    """Save a new run's manifest and driver settings. Nothing is generated yet.

    A benchmark run is checked first (`check_benchmark`); a refused run
    leaves nothing on disk. Raise `ManifestExists` if the run already exists.
    """
    if settings.benchmark:
        check_benchmark(manifest)
    save_manifest(store.root, manifest)
    path = _settings_path(store.root, manifest.run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, settings.model_dump_json(indent=2))


def _write_json(path: Path, text: str) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


@contextmanager
def run_lock(store: RecordStore, run_id: str) -> GeneratorType[None]:
    """Hold the run's driver lock; raise `RunBusy` if another driver holds it."""
    path = _driver_dir(store.root, run_id) / "lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunBusy(f"another driver is working on run {run_id}") from exc
        yield
    finally:
        os.close(fd)


class Skipped(BaseModel):
    """A record stage that failed in a way that belongs to that record alone."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str
    stage: Stage
    error: str


class RunOutcome(BaseModel):
    """What one `drive` call did."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    finished: bool
    """True when no record has work left, except on pairs whose budget is spent."""
    stopped: str | None = None
    """Why the run stopped early (a model or tool failure), or None."""
    over_budget: list[Pair] = Field(default_factory=lambda: [])
    skipped: list[Skipped] = Field(default_factory=lambda: [])
    waiting: list[str] = Field(default_factory=lambda: [])
    """Records whose next stage has no runner in this call."""
    unsynced: int = 0
    """Finished records whose memory could not be written; a later call retries."""


def _log_event(root: Path, run_id: str, **fields: JsonValue) -> None:
    path = _driver_dir(root, run_id) / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    entry: dict[str, JsonValue] = {"at": datetime.now(UTC).isoformat(), **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, JsonValue]]:
    if not path.exists():
        return []
    entries: list[dict[str, JsonValue]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry: JsonValue = json.loads(line)
        except ValueError:
            continue  # A line cut short by a crash; its batch or sync is redone.
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


def _batch_key(pair: Pair, family: Family) -> str:
    a, b = canonical_pair(pair)
    return f"{a}/{b}:{family.value}"


def _batches_done(store: RecordStore, run_id: str) -> set[str]:
    done: set[str] = set()
    for entry in _read_jsonl(_driver_dir(store.root, run_id) / "generated.jsonl"):
        key = entry.get("batch")
        if isinstance(key, str):
            done.add(key)
    for record in store.list(run_id):
        generator = record.stages.get(Stage.generator)
        family = None if generator is None else generator.data.get("family")
        if isinstance(family, str) and family in Family.__members__:
            done.add(_batch_key(record.pair, Family(family)))
    return done


def _status_value(record: Record) -> str:
    return record.status.value


def _finished(record: Record) -> bool:
    """A record whose memory the driver writes: final, or survived with nothing left."""
    if record.status in FINAL_STATUSES:
        return True
    return record.status is Status.survived and next_stage(record) is None


def _sync_memory(store: RecordStore, run_id: str, memory: MemoryClient | None) -> int:
    """Write a memory for each finished record not yet synced at its status. Return
    how many are left unsynced."""
    path = _driver_dir(store.root, run_id) / "memory.jsonl"
    synced = {(e.get("record_id"), e.get("status")) for e in _read_jsonl(path)}
    pending = [
        r for r in store.list(run_id) if _finished(r) and (r.id, _status_value(r)) not in synced
    ]
    if memory is None:
        return len(pending)
    left = 0
    for record in pending:
        try:
            memory_id = memory.remember(record)
        except MemoryUnavailable as exc:
            left += 1
            _log_event(
                store.root, run_id, event="memory-unsynced", record_id=record.id, error=str(exc)
            )
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            line: dict[str, JsonValue] = {
                "record_id": record.id,
                "status": _status_value(record),
                "memory_id": memory_id,
                "at": datetime.now(UTC).isoformat(),
            }
            handle.write(json.dumps(line) + "\n")
    return left


def _score_queue(store: RecordStore, run_id: str) -> None:
    for item in queue_items(store, run_id):
        if Stage.scorer in item.record.stages:
            continue
        data = cast(dict[str, JsonValue], item.score.model_dump(mode="json"))
        store.write_stage(
            item.record.id,
            Stage.scorer,
            StageOutput(written_at=datetime.now(UTC), data={**data, "kind": item.kind.value}),
        )


class _Stop(Exception):
    """Internal: the run must stop; the message says why."""


def _generate(
    store: RecordStore,
    ctx: StageContext,
    generator: Generator,
    memory: MemoryClient | None,
    over_budget: list[Pair],
) -> None:
    run_id = ctx.ledger.run_id
    manifest = ctx.ledger.manifest
    families = [f for f in manifest.families if f in GENERATOR_FAMILIES]
    for pair in manifest.pairs:
        done = _batches_done(store, run_id)
        todo = [f for f in families if _batch_key(pair, f) not in done]
        if not todo:
            continue
        earlier: list[str] = []
        if manifest.memory_policy is not MemoryPolicy.blind:
            if memory is None:
                raise ValueError(
                    f"run {run_id} reads memory ({manifest.memory_policy}); pass a memory client"
                )
            try:
                earlier = [e.statement for e in memory.recall(pair)]
            except MemoryUnavailable as exc:
                raise _Stop(f"MuninnDB failed while reading memory for {pair}: {exc}") from exc
        for family in todo:
            try:
                made = generator(pair, family, ctx.settings.conjectures_per_family, earlier, ctx)
            except BudgetExceeded as exc:
                over_budget.append(pair)
                _log_event(store.root, run_id, event="over-budget", pair=list(pair), error=str(exc))
                break
            except STOP_RUN as exc:
                raise _Stop(f"generation by {family.value} on {pair} failed: {exc}") from exc
            path = _driver_dir(store.root, run_id) / "generated.jsonl"
            entry: dict[str, JsonValue] = {
                "batch": _batch_key(pair, family),
                "records": [r.id for r in made],
                "at": datetime.now(UTC).isoformat(),
            }
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")


def _run_records(
    store: RecordStore,
    ctx: StageContext,
    runners: Mapping[Stage, StageRunner],
    outcome: RunOutcome,
) -> None:
    run_id = ctx.ledger.run_id
    blocked: set[str] = set()
    while True:
        progressed = False
        for listed in store.list(run_id):
            if listed.id in blocked or canonical_pair(listed.pair) in outcome.over_budget:
                continue
            record = listed
            while (stage := next_stage(record)) is not None:
                runner = runners.get(stage)
                if runner is None:
                    outcome.waiting.append(record.id)
                    blocked.add(record.id)
                    break
                try:
                    runner(record.id, ctx)
                except BudgetExceeded as exc:
                    outcome.over_budget.append(canonical_pair(record.pair))
                    _log_event(
                        store.root,
                        run_id,
                        event="over-budget",
                        pair=list(canonical_pair(record.pair)),
                        record_id=record.id,
                        stage=stage.value,
                        error=str(exc),
                    )
                    break
                except STOP_RUN as exc:
                    raise _Stop(f"{stage.value} on {record.id} failed: {exc}") from exc
                except SKIP_RECORD as exc:
                    skipped = Skipped(
                        record_id=record.id, stage=stage, error=f"{type(exc).__name__}: {exc}"
                    )
                    outcome.skipped.append(skipped)
                    blocked.add(record.id)
                    _log_event(
                        store.root, run_id, event="skipped", **skipped.model_dump(mode="json")
                    )
                    break
                progressed = True
                after = store.load(record.id)
                if stage not in after.stages:
                    raise RuntimeError(
                        f"the {stage.value} runner returned without writing {record.id}'s stage"
                    )
                record = after
        if not progressed:
            return


def drive(
    store: RecordStore,
    run_id: str,
    *,
    runners: Mapping[Stage, StageRunner] | None = None,
    generator: Generator | None = None,
    backends: Mapping[Family, Backend] | None = None,
    memory: MemoryClient | None = None,
) -> RunOutcome:
    """Do a run's remaining work, as the module describes, and say what happened.

    `runners` and `generator` default to the real stages. `memory` is needed
    by a run that reads memory; without one, nothing is synced and
    `RunOutcome.unsynced` counts what is waiting. Raise `RunBusy` if another
    driver holds the run, `FileNotFoundError` if it has no manifest, and
    `BenchmarkRefused` if a benchmark run no longer meets the rules.
    """
    if not manifest_path(store.root, run_id).exists():
        raise FileNotFoundError(f"run {run_id} has no manifest")
    with run_lock(store, run_id):
        ctx = StageContext.for_run(store, run_id, backends)
        if ctx.settings.benchmark:
            check_benchmark(load_manifest(store.root, run_id))
        if memory is not None and memory.manifest.run_id != run_id:
            raise ValueError(f"the memory client is for run {memory.manifest.run_id}")
        outcome = RunOutcome(run_id=run_id, finished=False)
        _log_event(store.root, run_id, event="drive-start")
        try:
            _generate(store, ctx, generator or default_generator, memory, outcome.over_budget)
            _run_records(store, ctx, default_runners() if runners is None else runners, outcome)
        except _Stop as stop:
            outcome.stopped = str(stop)
            _log_event(store.root, run_id, event="stopped", reason=outcome.stopped)
        outcome.over_budget = sorted(set(outcome.over_budget))
        _score_queue(store, run_id)
        outcome.unsynced = _sync_memory(store, run_id, memory)
        outcome.finished = outcome.stopped is None and not outcome.skipped and not outcome.waiting
        _log_event(
            store.root,
            run_id,
            event="drive-end",
            finished=outcome.finished,
            stopped=outcome.stopped,
        )
        return outcome
