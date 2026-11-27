"""What both generator modes share: the calls, the step log, and making records.

A generator run asks one family for conjectures on one domain pair. Every
call it makes is a `Step`, and each record it makes holds the whole list of
steps in its generator stage (`data["steps"]`): the prompt, the reply, the
model, and the prompt's hash. That is the record's generation provenance; the
`Provenance` model itself keeps only the family, the model, and the hash of the
final prompt.

The final step asks for JSON. A reply that cannot be read is asked for once
more, with what was wrong; if that fails too, the batch is dropped. Every
dropped reply, every item that could not be read, and every duplicate is
logged as one JSON line in `runs/<run_id>/generator/dropped.jsonl`.
"""

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, JsonValue

from minilab.generate.parse import Conjecture, ParseFailure, parse_conjectures
from minilab.ledger import Ledger
from minilab.models.adapter import Budget, Role, call
from minilab.models.backends import Backend
from minilab.records import (
    Family,
    Provenance,
    Record,
    RecordExists,
    Stage,
    StageOutput,
)
from minilab.scheduler import Pair, canonical_pair

Mode = Literal["plain", "structured"]

GENERATOR_FAMILIES: frozenset[Family] = frozenset({Family.claude, Family.codex, Family.gemini})
"""The families that generate (design doc-gmj, "Generator"). The local weak model does not."""

MAX_PARSE_ATTEMPTS = 2
"""A reply that cannot be read is asked for once more, then dropped."""

DOMAIN_NAMES: dict[str, str] = {
    "optimisation": "Convex and nonconvex optimisation",
    "information-theory": "Information theory",
    "probability": "Probability and concentration",
    "hd-statistics": "High-dimensional statistics",
    "random-matrices": "Random matrices and linear algebra",
    "optimal-transport": "Optimal transport",
    "kernels": "Kernels and approximation theory",
    "gradient-flows": "Gradient flows and dynamical systems",
    "information-geometry": "Information geometry",
    "graphs-spectral": "Graph theory and spectral methods",
}
"""The ten domains' names as the design gives them, by scheduler slug."""


def domain_name(slug: str) -> str:
    return DOMAIN_NAMES[slug]


JSON_INSTRUCTIONS = """Reply with one JSON object and nothing else, of the form
{"conjectures": [ ... ]}. Each conjecture is an object with these keys:

- "statement": the conjecture in a few precise sentences, with the direction of
  every inequality stated.
- "objects": a list of strings, the mathematical objects it uses.
- "assumptions": a list of strings, every assumption it needs, one per item.
- "domain_of_validity": the exact range of the quantities where it is claimed.
- "instance_family": a finite, computable family of test instances on which code
  can check the claim, such as a fixed dimension, a discretisation, or a
  parametric family. Name each parameter as a plain identifier (letters, digits,
  and underscores) and give its type and range, for example
  "p: a rational in (0, 1); n: an integer in 2..8". Code that tests the
  conjecture will use exactly these names. Use null if the claim has no such
  family."""


def json_instructions(n: int) -> str:
    return f"State at most {n} conjectures.\n\n{JSON_INSTRUCTIONS}"


MAX_EARLIER = 20
"""At most this many earlier statements go into a prompt."""

MAX_EARLIER_CHARS = 400
"""Each earlier statement is cut to this many characters."""


def earlier_section(earlier: Sequence[str]) -> str:
    """The prompt paragraph that lists earlier runs' conjectures, or "" if none.

    The driver passes statements only when the run's memory policy allows
    reads (design doc-gmj, "Memory read policy"). With none, the prompt is
    the same as without this paragraph, so a `blind` run's prompts do not
    change.
    """
    items = [" ".join(s.split())[:MAX_EARLIER_CHARS] for s in earlier[:MAX_EARLIER]]
    items = [item for item in items if item]
    if not items:
        return ""
    listed = "\n".join(f"- {item}" for item in items)
    return (
        "Earlier runs already proposed these conjectures. Do not repeat them; "
        f"propose different ones.\n\n{listed}\n\n"
    )


def retry_prompt(prompt: str, failure: str) -> str:
    return (
        prompt
        + f"""

Your previous reply could not be read: {failure[:500]}
Reply again with the JSON object only."""
    )


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


class Step(BaseModel):
    """One generator call: what was asked, and what came back."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step: str
    prompt: str
    prompt_hash: str
    reply: str
    model: str


class Generation:
    """One generator run for one family on one pair: its calls, its log, and its records."""

    def __init__(
        self,
        mode: Mode,
        pair: Pair,
        family: Family | str,
        n: int,
        *,
        ledger: Ledger,
        budget: Budget,
        backends: Mapping[Family, Backend] | None,
        earlier: Sequence[str] = (),
    ) -> None:
        family = Family(family)
        if family not in GENERATOR_FAMILIES:
            raise ValueError(f"family {family.value} does not generate")
        if ledger.manifest.mode != mode:
            raise ValueError(
                f"run {ledger.run_id} is a {ledger.manifest.mode} run; refused {mode} generation"
            )
        if n < 1:
            raise ValueError(f"asked for {n} conjectures")
        self.mode: Mode = mode
        self.pair = canonical_pair(pair)
        self.family = family
        self.n = n
        self.ledger = ledger
        self.budget = budget
        self.backends = backends
        self.steps: list[Step] = []
        self.earlier: tuple[str, ...] = tuple(earlier)

    @property
    def run_id(self) -> str:
        return self.ledger.run_id

    def ask(self, step: str, prompt: str) -> Step:
        """Make one generate call and log it as a step."""
        reply = call(
            self.family,
            Role.generate,
            prompt,
            record_id=None,
            pair=self.pair,
            ledger=self.ledger,
            budget=self.budget,
            backends=self.backends,
        )
        logged = Step(
            step=step,
            prompt=prompt,
            prompt_hash=prompt_hash(prompt),
            reply=reply.text,
            model=reply.model,
        )
        self.steps.append(logged)
        return logged

    def _log_path(self) -> Path:
        return self.ledger.store.root / "runs" / self.run_id / "generator" / "dropped.jsonl"

    def log_dropped(self, what: str, **fields: str) -> None:
        """Append one line to the run's log of dropped generator output."""
        entry: dict[str, str] = {
            "at": datetime.now(UTC).isoformat(),
            "what": what,
            "mode": self.mode,
            "family": self.family.value,
            "pair": "/".join(self.pair),
            **fields,
        }
        path = self._log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    def ask_for_conjectures(self, step: str, prompt: str) -> tuple[Step, list[Conjecture]] | None:
        """Ask for the JSON list, once more if it cannot be read. Return None if dropped."""
        asked = prompt
        for _ in range(MAX_PARSE_ATTEMPTS):
            logged = self.ask(step, asked)
            try:
                conjectures, failures = parse_conjectures(logged.reply)
            except ParseFailure as exc:
                self.log_dropped(
                    "reply", reply=logged.reply, prompt_hash=logged.prompt_hash, error=str(exc)
                )
                asked = retry_prompt(prompt, str(exc))
                continue
            for failure in failures:
                self.log_dropped("item", item=failure, prompt_hash=logged.prompt_hash)
            return logged, conjectures
        return None

    def _record_id(self, conjecture: Conjecture) -> str:
        content = json.dumps(
            [self.mode, self.family.value, list(self.pair), conjecture.model_dump(mode="json")],
            sort_keys=True,
        )
        return f"{self.run_id}.g-{hashlib.sha256(content.encode()).hexdigest()[:16]}"

    def make_records(self, final: Step, conjectures: list[Conjecture]) -> list[Record]:
        """Store the first `n` conjectures as records, each holding the whole step log.

        A conjecture whose record already exists (the same text from the same
        family, mode, and pair) is logged as a duplicate and skipped.
        """
        steps = cast(list[JsonValue], [s.model_dump(mode="json") for s in self.steps])
        records: list[Record] = []
        for index, conjecture in enumerate(conjectures[: self.n]):
            record = Record(
                id=self._record_id(conjecture),
                run_id=self.run_id,
                pair=self.pair,
                mode=self.mode,
                statement=conjecture.statement,
                objects=conjecture.objects,
                assumptions=conjecture.assumptions,
                domain_of_validity=conjecture.domain_of_validity,
                instance_family=conjecture.instance_family,
                provenance=Provenance(
                    family=self.family, model=final.model, prompt_hash=final.prompt_hash
                ),
                stages={
                    Stage.generator: StageOutput(
                        written_at=datetime.now(UTC),
                        data={
                            "mode": self.mode,
                            "family": self.family.value,
                            "index": index,
                            "steps": steps,
                        },
                    )
                },
            )
            try:
                self.ledger.store.create(record)
            except RecordExists:
                self.log_dropped("duplicate", record_id=record.id, statement=record.statement)
                continue
            records.append(record)
        return records


def run_generation(
    generation: Generation, final_prompt: Callable[[Generation], tuple[str, str]]
) -> list[Record]:
    """Run the steps `final_prompt` makes, ask for the JSON list, and store the records.

    `final_prompt` makes any earlier steps through `generation.ask` and returns
    the final step's name and prompt.
    """
    step, prompt = final_prompt(generation)
    asked = generation.ask_for_conjectures(step, prompt)
    if asked is None:
        return []
    final, conjectures = asked
    return generation.make_records(final, conjectures)
