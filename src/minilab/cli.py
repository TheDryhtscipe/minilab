"""The `minilab` command line.

A plain typer app; later tasks add their commands to `app`. Every command
takes `--root`, the directory that holds `runs/` (by default the current
directory).
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from pydantic import JsonValue

from minilab.ledger import Ledger
from minilab.queue import NotInQueue, QueueItem, Verdict, mark_shown, queue_items
from minilab.queue import review as review_item
from minilab.records import RecordStore, Stage, run_of

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
