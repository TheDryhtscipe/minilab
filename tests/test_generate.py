"""Tests for the generator (build plan doc-5wh, Task 15; minilab-d1f549).

Plain mode asks once for conjectures that link two domains. Structured mode
asks in three steps (elicit, abstract, cross) and logs each step's prompt and
reply on the records. Both ask for JSON with `statement`, `objects`,
`assumptions`, `domain_of_validity`, and `instance_family`; a conjecture with
no instance family is `untestable`. A reply that cannot be read is asked for
once more, then dropped and logged.

Every model here is a fake backend.
"""

import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest

from minilab.generate.parse import parse_conjectures
from minilab.generate.plain import generate_plain, plain_prompt
from minilab.generate.structured import cross_prompt, generate_structured
from minilab.ledger import Ledger
from minilab.models.adapter import (
    GEMINI_ROLES,
    Budget,
    ModelReply,
    Role,
    RoleNotAllowed,
    call,
)
from minilab.records import Family, RecordStore, Stage, Status
from minilab.run import MemoryPolicy, RunManifest, save_manifest

RUN = "run-1"
PAIR = ("optimisation", "information-theory")
BUDGET = Budget(tokens_per_pair=1_000_000)

GOOD = {
    "statement": "For every p in (0, 1), H(p) <= log 2.",
    "objects": ["binary entropy H"],
    "assumptions": ["p is a probability"],
    "domain_of_validity": "0 < p < 1",
    "instance_family": "p: a rational in (0, 1)",
}


def conjecture(**changes: str | None) -> dict[str, str | list[str] | None]:
    item: dict[str, str | list[str] | None] = dict(GOOD)
    for key, value in changes.items():
        item[key] = value
    return item


def reply_of(*items: Mapping[str, object]) -> str:
    return "```json\n" + json.dumps({"conjectures": list(items)}) + "\n```"


Reply = Callable[[str], str]


class FakeBackend:
    """A backend whose reply text comes from `reply(prompt)`. It records each prompt."""

    def __init__(self, family: Family, reply: Reply) -> None:
        self.family = family
        self.reply = reply
        self.prompts: list[str] = []
        self.replies: list[str] = []

    def run(
        self, prompt: str, *, cwd: Path, env: Mapping[str, str], timeout_s: float
    ) -> ModelReply:
        self.prompts.append(prompt)
        text = self.reply(prompt)
        self.replies.append(text)
        return ModelReply(text=text, tokens_in=10, tokens_out=10, estimated=False, model="fake-1")


def in_turn(*texts: str) -> Reply:
    """Each call gives the next text; the last one repeats."""
    queue = list(texts)

    def reply(prompt: str) -> str:
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return reply


def make_ledger(root: Path, mode: Literal["plain", "structured"]) -> Ledger:
    store = RecordStore(root)
    save_manifest(
        root,
        RunManifest(
            run_id=RUN,
            mode=mode,
            pairs=(PAIR,),
            memory_policy=MemoryPolicy.blind,
            memory_before=None,
            literature_before=None,
            budget_tokens_per_pair=1_000_000,
            families=tuple(Family),
            started=datetime(2026, 11, 25, 12, 0, tzinfo=UTC),
        ),
    )
    return Ledger(store, RUN)


@pytest.fixture
def plain_ledger(tmp_path: Path) -> Ledger:
    return make_ledger(tmp_path, "plain")


@pytest.fixture
def structured_ledger(tmp_path: Path) -> Ledger:
    return make_ledger(tmp_path, "structured")


def dropped_log(ledger: Ledger) -> list[dict[str, object]]:
    path = ledger.store.root / "runs" / RUN / "generator" / "dropped.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


# --- the record and its instance family ---


def test_missing_instance_family_gives_untestable(plain_ledger: Ledger) -> None:
    no_key = conjecture()
    del no_key["instance_family"]
    text = reply_of(
        conjecture(statement="A"),
        conjecture(statement="B", instance_family=None),
        conjecture(statement="C", instance_family="  "),
        no_key | {"statement": "D"},
    )
    backend = FakeBackend(Family.gemini, in_turn(text))
    records = generate_plain(
        PAIR,
        Family.gemini,
        4,
        ledger=plain_ledger,
        budget=BUDGET,
        backends={Family.gemini: backend},
    )
    by_statement = {r.statement: r for r in records}
    assert set(by_statement) == {"A", "B", "C", "D"}
    assert by_statement["A"].status is Status.proposed
    for name in ("B", "C", "D"):
        assert by_statement[name].status is Status.untestable
        assert by_statement[name].instance_family is None
    stored = {r.id: r for r in plain_ledger.store.list(RUN)}
    assert {r.id for r in records} == set(stored)
    assert stored[by_statement["D"].id].status is Status.untestable


def test_plain_records_carry_pair_mode_and_provenance(plain_ledger: Ledger) -> None:
    backend = FakeBackend(Family.claude, in_turn(reply_of(conjecture(), conjecture(statement="B"))))
    records = generate_plain(
        PAIR,
        Family.claude,
        2,
        ledger=plain_ledger,
        budget=BUDGET,
        backends={Family.claude: backend},
    )
    assert len(records) == 2
    for record in records:
        assert record.pair == ("information-theory", "optimisation")
        assert record.mode == "plain"
        assert record.provenance.family is Family.claude
        assert record.provenance.model == "fake-1"
        assert record.provenance.parents == []
        steps = record.stages[Stage.generator].data["steps"]
        assert isinstance(steps, list) and len(steps) == 1
    assert plain_ledger.pair_spend(PAIR) == 20


def test_at_most_n_records(plain_ledger: Ledger) -> None:
    items = [conjecture(statement=f"S{i}") for i in range(5)]
    backend = FakeBackend(Family.claude, in_turn(reply_of(*items)))
    records = generate_plain(
        PAIR,
        Family.claude,
        3,
        ledger=plain_ledger,
        budget=BUDGET,
        backends={Family.claude: backend},
    )
    assert [r.statement for r in records] == ["S0", "S1", "S2"]


def test_prompts_ask_for_named_parameters() -> None:
    # Both encoders use the parameter names the instance family gives (doc-6ja).
    for prompt in (plain_prompt(PAIR, 3), cross_prompt(PAIR, "some structures", 3)):
        for key in ("statement", "objects", "assumptions", "domain_of_validity", "instance_family"):
            assert f'"{key}"' in prompt
        assert "name each parameter" in prompt.lower()
        assert "use exactly these names" in prompt
        assert "Convex and nonconvex optimisation" in prompt and "Information theory" in prompt


def test_mode_must_match_the_manifest(plain_ledger: Ledger) -> None:
    backend = FakeBackend(Family.claude, in_turn(reply_of(conjecture())))
    with pytest.raises(ValueError, match="plain"):
        generate_structured(
            PAIR,
            Family.claude,
            1,
            ledger=plain_ledger,
            budget=BUDGET,
            backends={Family.claude: backend},
        )
    assert backend.prompts == []


def test_local_model_does_not_generate(plain_ledger: Ledger) -> None:
    backend = FakeBackend(Family.local, in_turn(reply_of(conjecture())))
    with pytest.raises(ValueError, match="local"):
        generate_plain(
            PAIR,
            Family.local,
            1,
            ledger=plain_ledger,
            budget=BUDGET,
            backends={Family.local: backend},
        )
    assert backend.prompts == []


# --- structured mode ---


def test_structured_logs_three_steps(structured_ledger: Ledger) -> None:
    elicit = "Optimisation: convex duality. Information theory: entropy, KL divergence."
    abstract = "Both share a variational form: a value is a supremum over a convex set."
    final = reply_of(conjecture(statement="A"), conjecture(statement="B"))
    backend = FakeBackend(Family.codex, in_turn(elicit, abstract, final))
    records = generate_structured(
        PAIR,
        Family.codex,
        2,
        ledger=structured_ledger,
        budget=BUDGET,
        backends={Family.codex: backend},
    )
    assert len(records) == 2 and len(backend.prompts) == 3
    # Each step builds on the reply before it.
    assert elicit in backend.prompts[1]
    assert abstract in backend.prompts[2]
    for record in records:
        assert record.mode == "structured"
        steps = record.stages[Stage.generator].data["steps"]
        assert isinstance(steps, list)
        logged = [s for s in steps if isinstance(s, dict)]
        assert [s["step"] for s in logged] == ["elicit", "abstract", "cross"]
        assert [s["prompt"] for s in logged] == backend.prompts
        assert [s["reply"] for s in logged] == [elicit, abstract, final]
    assert structured_ledger.pair_spend(PAIR) == 60


# --- roles ---


def test_gemini_allowed_only_for_generate_and_screen_proofs(plain_ledger: Ledger) -> None:
    # Screen proofs (`Role.screen`) are tested in test_adapter and test_triviality.
    backend = FakeBackend(Family.gemini, in_turn(reply_of(conjecture())))
    backends = {Family.gemini: backend}
    for role in Role:
        if role in GEMINI_ROLES:
            continue
        with pytest.raises(RoleNotAllowed):
            call(
                Family.gemini,
                role,
                "check this",
                record_id=None,
                pair=PAIR,
                ledger=plain_ledger,
                budget=BUDGET,
                backends=backends,
            )
    assert backend.prompts == []
    records = generate_plain(
        PAIR, Family.gemini, 1, ledger=plain_ledger, budget=BUDGET, backends=backends
    )
    assert len(records) == 1 and records[0].provenance.family is Family.gemini


# --- parse failures ---


def test_malformed_json_retried_then_dropped(plain_ledger: Ledger) -> None:
    backend = FakeBackend(Family.claude, in_turn("not json at all", '{"conjectures": [}'))
    records = generate_plain(
        PAIR,
        Family.claude,
        2,
        ledger=plain_ledger,
        budget=BUDGET,
        backends={Family.claude: backend},
    )
    assert records == []
    assert len(backend.prompts) == 2
    assert plain_ledger.store.list(RUN) == []
    log = dropped_log(plain_ledger)
    assert [entry["reply"] for entry in log] == ["not json at all", '{"conjectures": [}']
    assert all(entry["family"] == "claude" for entry in log)


def test_malformed_json_retried_once_then_read(plain_ledger: Ledger) -> None:
    backend = FakeBackend(Family.claude, in_turn("oops", reply_of(conjecture())))
    records = generate_plain(
        PAIR,
        Family.claude,
        1,
        ledger=plain_ledger,
        budget=BUDGET,
        backends={Family.claude: backend},
    )
    assert len(records) == 1 and len(backend.prompts) == 2
    assert "could not be read" in backend.prompts[1]
    steps = records[0].stages[Stage.generator].data["steps"]
    assert isinstance(steps, list) and len(steps) == 2
    assert len(dropped_log(plain_ledger)) == 1


def test_bad_item_dropped_and_logged(plain_ledger: Ledger) -> None:
    text = reply_of(conjecture(statement="A"), {"statement": "no other fields"})
    backend = FakeBackend(Family.claude, in_turn(text))
    records = generate_plain(
        PAIR,
        Family.claude,
        2,
        ledger=plain_ledger,
        budget=BUDGET,
        backends={Family.claude: backend},
    )
    assert [r.statement for r in records] == ["A"]
    assert len(backend.prompts) == 1
    log = dropped_log(plain_ledger)
    assert len(log) == 1 and "no other fields" in str(log[0]["item"])


def test_parse_accepts_bare_list_and_prose_around_json() -> None:
    items, failures = parse_conjectures("Here you go:\n" + json.dumps([GOOD]) + "\nThanks.")
    assert failures == [] and [c.statement for c in items] == [GOOD["statement"]]
    with pytest.raises(ValueError):
        parse_conjectures(json.dumps({"conjectures": []}))
