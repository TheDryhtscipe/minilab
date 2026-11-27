"""The `minilab` command line.

`run` starts a run and `resume` continues one (both through
`minilab.pipeline.drive`); `report` prints a run's cost report; `queue` and
`review` show the human queue and store verdicts. Every command takes
`--root`, the directory that holds `runs/` and `state/` (by default the
current directory).
"""

import json
import random
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated, Literal

import typer
from pydantic import JsonValue

from minilab.ledger import CostReport, Ledger, cost_report
from minilab.memory import McpTransport, MemoryClient
from minilab.pipeline import (
    BenchmarkRefused,
    DriverSettings,
    RunBusy,
    RunOutcome,
    drive,
    start_run,
)
from minilab.queue import NotInQueue, QueueItem, Verdict, mark_shown, queue_items
from minilab.queue import review as review_item
from minilab.records import Family, RecordStore, Stage, run_of
from minilab.run import ManifestExists, MemoryPolicy, RunManifest, default_models, load_manifest
from minilab.scheduler import Pair, Scheduler, all_pairs, canonical_pair

app = typer.Typer(no_args_is_help=True, help="Minilab: a filter-first search loop.")

RootOption = Annotated[
    Path | None,
    typer.Option("--root", help="Directory that holds runs/ (default: the current directory)."),
]


def _root(root: Path | None) -> Path:
    return Path.cwd() if root is None else root


def _json(value: JsonValue) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def _section(title: str, body: str) -> str:
    text = body.strip() or "(none)"
    indented = "\n".join(f"    {line}" for line in text.splitlines())
    return f"  {title}:\n{indented}"


def format_item(position: int, item: QueueItem) -> str:
    """Render one queue item for the terminal."""
    record = item.record
    stages = record.stages

    def data(stage: Stage, key: str) -> JsonValue:
        output = stages.get(stage)
        return None if output is None else output.data.get(key)

    roundtrip = data(Stage.formaliser, "roundtrip")
    audit = data(Stage.formaliser, "audit")
    literature = stages.get(Stage.literature)
    evidence: list[str] = []
    for key in ("coverage", "certificate"):
        value = data(Stage.falsifier, key)
        if value is not None:
            evidence.append(f"{key}: {_json(value)}")
    for stage, output in stages.items():
        evidence.extend(f"{stage.value} file: {path}" for path in output.evidence)
    axioms = data(Stage.prover, "axioms")
    if axioms is not None:
        evidence.append(f"axioms: {_json(axioms)}")
    mechanism = data(Stage.falsifier, "mechanism")
    s = item.score
    compression = "n/a" if s.compression is None else f"{s.compression:.3g}"
    lines = [
        f"[{position}] {record.id}  {item.kind.value} ({record.status.value})  "
        f"pair {record.pair[0]} x {record.pair[1]}",
        _section("Statement", record.statement),
        _section("Round-trip text", roundtrip if isinstance(roundtrip, str) else ""),
        _section("Hypothesis audit", "" if audit is None else _json(audit)),
        _section("Literature notes", "" if literature is None else literature.notes),
        _section("Evidence", "\n".join(evidence)),
    ]
    if mechanism is not None:
        lines.append(_section("Mechanism", _json(mechanism)))
    if data(Stage.triviality, "held") is True:
        proof = data(Stage.triviality, "proof")
        reasons = data(Stage.triviality, "hold_reasons")
        listed = reasons if isinstance(reasons, list) else ["(the reasons could not be read)"]
        corollary = data(Stage.triviality, "corollary_of")
        lines.append(
            _section(
                "Held: the triviality screen proved it, but the formalisation was flagged",
                f"proved by: {data(Stage.triviality, 'proved_by')}"
                + (f" (corollary of {corollary})" if isinstance(corollary, str) else "")
                + "\n"
                f"proof:\n{proof if isinstance(proof, str) else _json(proof)}\n"
                "flags:\n" + "\n".join(f"- {r}" for r in listed),
            )
        )
    if data(Stage.prover, "held") is True:
        proof = data(Stage.prover, "proof")
        reasons = data(Stage.prover, "hold_reasons")
        listed = reasons if isinstance(reasons, list) else ["(the reasons could not be read)"]
        weakened = data(Stage.prover, "weakened_statement")
        lines.append(
            _section(
                "Held: the prover proved it, but the proof is not of the statement as "
                "formalised, or the formalisation was flagged",
                (f"statement proved:\n{weakened}\n" if isinstance(weakened, str) else "")
                + f"proof:\n{proof if isinstance(proof, str) else _json(proof)}\n"
                "flags:\n" + "\n".join(f"- {r}" for r in listed),
            )
        )
    lines.append(
        f"  Score: compression {compression}, near misses {s.near_misses or 'none'}, "
        f"library lemmas {len(s.library_lemmas)}"
    )
    queued = stages.get(Stage.queue)
    if queued is not None:
        lines.append(f"  Reviewed: {queued.data.get('verdict')} ({queued.data.get('note')})")
    return "\n".join(lines)


@app.command()
def queue(run_id: str, root: RootOption = None) -> None:
    """Show a run's queue items, best ranked first, and start their review clocks."""
    store = RecordStore(_root(root))
    items = queue_items(store, run_id)
    if not items:
        typer.echo(f"Run {run_id} has no queue items.")
        return
    mark_shown(store.root, run_id, [item.record.id for item in items], datetime.now(UTC))
    typer.echo("\n\n".join(format_item(n, item) for n, item in enumerate(items, start=1)))


@app.command()
def review(
    record_id: str,
    verdict: Annotated[Verdict, typer.Option("--verdict", help="Your verdict on the item.")],
    note: Annotated[str, typer.Option("--note", help="A note to store with it.")] = "",
    root: RootOption = None,
) -> None:
    """Store your verdict on a queue item and log the minutes since it was shown."""
    store = RecordStore(_root(root))
    try:
        ledger = Ledger(store, run_of(record_id))
        minutes = review_item(store, ledger, record_id, verdict, note)
    except (NotInQueue, ValueError, KeyError, FileNotFoundError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Stored {verdict.value} for {record_id}; logged {minutes:.1f} minutes.")


DEFAULT_FAMILIES: tuple[Family, ...] = (Family.claude, Family.codex, Family.gemini, Family.local)
"""A run's families unless `--families` says otherwise. The local model is needed
by the triviality screen."""


def choose_pairs(spec: str, *, count: int, state_path: Path, seed: int | None) -> list[Pair]:
    """The run's pairs: `auto` picks `count` distinct pairs with the scheduler;
    anything else is a JSON file holding a list of two-domain lists.

    Picking does not record visits; `run` records them once the run is saved.
    Raise ValueError for an unknown domain, a bad file, or too many pairs.
    """
    if spec == "auto":
        pool = all_pairs()
        if not 1 <= count <= len(pool):
            raise ValueError(f"--pair-count must be between 1 and {len(pool)}, not {count}")
        scheduler = Scheduler(state_path, random.Random(seed))
        picked: list[Pair] = []
        for _ in range(count):
            pair = scheduler.pick([p for p in pool if p not in picked])
            picked.append(pair)
        return picked
    raw: JsonValue = json.loads(Path(spec).read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{spec}: expected a non-empty JSON list of pairs")
    pairs: list[Pair] = []
    for item in raw:
        if not (
            isinstance(item, list) and len(item) == 2 and all(isinstance(d, str) for d in item)
        ):
            raise ValueError(f"{spec}: not a pair of domain names: {item!r}")
        a, b = item
        pairs.append(canonical_pair((str(a), str(b))))
    return pairs


def _families(text: str) -> tuple[Family, ...]:
    try:
        return tuple(Family(name.strip()) for name in text.split(",") if name.strip())
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _print_outcome(outcome: RunOutcome, root: Path) -> None:
    run_id = outcome.run_id
    if outcome.stopped is not None:
        typer.echo(f"Run {run_id} stopped: {outcome.stopped}")
    for skipped in outcome.skipped:
        typer.echo(f"Skipped {skipped.record_id} at {skipped.stage.value}: {skipped.error}")
    for pair in outcome.over_budget:
        typer.echo(f"Pair {pair[0]} x {pair[1]} reached its token budget.")
    if outcome.unsynced:
        typer.echo(f"{outcome.unsynced} finished records are not yet in MuninnDB; resume retries.")
    if outcome.finished:
        typer.echo(f"Run {run_id} finished. See the queue: minilab queue {run_id} --root {root}")
    else:
        typer.echo(
            f"Run {run_id} is not finished. Continue it: minilab resume {run_id} --root {root}"
        )


def _drive(store: RecordStore, run_id: str) -> RunOutcome:
    try:
        memory = MemoryClient(load_manifest(store.root, run_id), McpTransport())
        return drive(store, run_id, memory=memory)
    except (RunBusy, BenchmarkRefused, FileNotFoundError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def run(
    mode: Annotated[Literal["plain", "structured"], typer.Option("--mode")],
    budget_tokens_per_pair: Annotated[
        int, typer.Option("--budget-tokens-per-pair", min=1, help="Tokens per pair, all calls.")
    ],
    pairs: Annotated[
        str, typer.Option("--pairs", help="auto, or a JSON file of [domain, domain] lists.")
    ] = "auto",
    pair_count: Annotated[int, typer.Option("--pair-count", help="Pairs to pick with auto.")] = 1,
    memory: Annotated[MemoryPolicy, typer.Option("--memory")] = MemoryPolicy.blind,
    memory_before: Annotated[
        datetime | None,
        typer.Option("--memory-before", help="Read only memories made before this."),
    ] = None,
    literature_before: Annotated[
        datetime | None, typer.Option("--literature-before", formats=["%Y-%m-%d"])
    ] = None,
    families: Annotated[
        str, typer.Option("--families", help="Comma-separated families.")
    ] = ",".join(f.value for f in DEFAULT_FAMILIES),
    conjectures_per_family: Annotated[int, typer.Option("--conjectures-per-family")] = 4,
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    seed: Annotated[int, typer.Option("--seed", help="Seeds pair picks and search.")] = 0,
    benchmark: Annotated[
        bool,
        typer.Option(
            "--benchmark", help="Refuse the run unless it meets the benchmark's blinding."
        ),
    ] = False,
    root: RootOption = None,
) -> None:
    """Start a run: write its manifest, then generate and run every record's stages."""
    base = _root(root)
    store = RecordStore(base)
    state_path = base / "state" / "pairs.json"
    chosen_families = _families(families)
    try:
        chosen = choose_pairs(pairs, count=pair_count, state_path=state_path, seed=seed)
        before = memory_before
        if before is not None and before.tzinfo is None:
            before = before.replace(tzinfo=UTC)
        cap: date | None = None if literature_before is None else literature_before.date()
        manifest = RunManifest(
            run_id=run_id or f"{mode}-{datetime.now(UTC):%Y%m%d-%H%M%S}",
            mode=mode,
            pairs=tuple(chosen),
            memory_policy=memory,
            memory_before=before,
            literature_before=cap,
            budget_tokens_per_pair=budget_tokens_per_pair,
            families=chosen_families,
            models=default_models(chosen_families),
            started=datetime.now(UTC),
        )
        settings = DriverSettings(
            conjectures_per_family=conjectures_per_family, seed=seed, benchmark=benchmark
        )
        start_run(store, manifest, settings)
    except (BenchmarkRefused, ManifestExists, ValueError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if pairs == "auto":
        scheduler = Scheduler(state_path, random.Random(seed))
        for pair in manifest.pairs:
            scheduler.record_visit(pair)
    typer.echo(f"Started run {manifest.run_id} on {len(manifest.pairs)} pairs.")
    outcome = _drive(store, manifest.run_id)
    _print_outcome(outcome, base)
    if not outcome.finished:
        raise typer.Exit(code=2)


@app.command()
def resume(run_id: str, root: RootOption = None) -> None:
    """Continue a run from each record's last written stage."""
    base = _root(root)
    outcome = _drive(RecordStore(base), run_id)
    _print_outcome(outcome, base)
    if not outcome.finished:
        raise typer.Exit(code=2)


def format_report(report: CostReport) -> str:
    """The cost report as plain lines."""

    def per(value: float | None, unit: str) -> str:
        return "none reviewed" if value is None else f"{value:,.2f} {unit}"

    lines = [
        f"Run {report.run_id}",
        f"  reviewed survivors: {report.reviewed_survivors}",
        f"  total tokens: {report.total_tokens:,} ({report.unattributed_tokens:,} with no record)",
        f"  machine hours: {report.machine_hours:.2f}; review hours: {report.review_hours:.2f}",
        f"  tokens per reviewed survivor: {per(report.tokens_per_reviewed_survivor, 'tokens')}",
        f"  hours per reviewed survivor: {per(report.hours_per_reviewed_survivor, 'hours')}",
    ]
    if report.family_cost_share:
        shares = ", ".join(
            f"{f.value} {v:.0%}" for f, v in sorted(report.family_cost_share.items())
        )
        lines.append(f"  share of tokens: {shares}")
    if report.family_survivor_share:
        shares = ", ".join(
            f"{f.value} {v:.0%}" for f, v in sorted(report.family_survivor_share.items())
        )
        lines.append(f"  share of reviewed survivors: {shares}")
    if report.has_estimates:
        lines.append("  Some token counts are estimates (a tool reported none).")
    return "\n".join(lines)


@app.command()
def report(run_id: str, root: RootOption = None) -> None:
    """Print a run's cost report: tokens and hours per reviewed survivor."""
    store = RecordStore(_root(root))
    try:
        ledger = Ledger(store, run_id)
        result = cost_report(store, ledger, run_id)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(format_report(result))
