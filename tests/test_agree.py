"""Tests for dual encodings and their agreement (build plan doc-5wh, Task 7, second half).

Two families each write an encoding from the record alone. Both run in the
sandbox on the same samples; margins and predicates must agree. Disagreement
makes the record `ambiguous` and asks the generator family for a sharper
statement, which becomes a child record. A record that is still ambiguous after
two sharpenings is `abandoned` (minilab-e47de8).

Every model here is a fake backend. The encodings really run in the sandbox.
"""

import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest

from minilab.encoding.agree import (
    Agreement,
    agree,
    encode_twice,
    run_encoding_stage,
    sharpen,
)
from minilab.encoding.prompts import encoding_prompt, extract_source, parse_statement
from minilab.encoding.sandbox import Sandboxed
from minilab.ledger import Ledger
from minilab.models.adapter import Budget, ModelReply
from minilab.records import Family, Provenance, Record, RecordStore, Stage, Status
from minilab.run import MemoryPolicy, RunManifest, save_manifest

RUN = "run-1"
PAIR = ("information-theory", "optimisation")


def toy(margin: str, *, key: str = "t", name: str = "Toy", capability: bool = True) -> str:
    """An encoding over one integer parameter `key` in 0..9 whose margin is `margin`.

    `margin` is an expression in the Fraction `t`.
    """
    exact = (
        f"""
    def exact_margin(self, x: Instance) -> Fraction:
        t = to_fraction(x.values["{key}"])
        return {margin}
"""
        if capability
        else ""
    )
    return f"""
import random
from fractions import Fraction

import mpmath

from minilab.encoding.api import Instance, to_fraction


class {name}:
    def margin(self, x: Instance) -> mpmath.mpf:
        t = to_fraction(x.values["{key}"])
        value = {margin}
        return mpmath.mpf(value.numerator) / value.denominator

    def assumptions(self, x: Instance) -> list[bool]:
        return [True]

    def in_domain(self, x: Instance) -> bool:
        t = to_fraction(x.values["{key}"])
        return 0 <= t <= 9

    def near_degenerate(self, x: Instance) -> bool:
        return False

    def sample(self, rng: random.Random) -> Instance:
        return Instance(values={{"{key}": str(rng.randint(0, 9))}}, precision_digits=30)
{exact}

ENCODING = {name}
"""


GOOD = toy("1 - t / 10")
OFF = toy("2 - t / 10", name="Off")
RENAMED = toy("1 - t / 10", key="s", name="Renamed")


def fenced(source: str) -> str:
    return f"Here is the encoding.\n\n```python\n{source}\n```\n"


SHARPER = {
    "statement": "For t in 0..9, 1 - t/10 >= 0.",
    "objects": ["t"],
    "assumptions": ["t is an integer"],
    "domain_of_validity": "0 <= t <= 9",
    "instance_family": "t: an integer in 0..9",
}


Reply = Callable[[str], str]


class FakeBackend:
    """A backend whose reply text comes from `reply(prompt)`. It records each prompt."""

    def __init__(self, family: Family, reply: Reply) -> None:
        self.family = family
        self.reply = reply
        self.prompts: list[str] = []

    def run(
        self, prompt: str, *, cwd: Path, env: Mapping[str, str], timeout_s: float
    ) -> ModelReply:
        self.prompts.append(prompt)
        return ModelReply(
            text=self.reply(prompt), tokens_in=10, tokens_out=10, estimated=False, model="fake-1"
        )


def is_encoding_prompt(prompt: str) -> bool:
    return "ENCODING" in prompt


def backends(
    claude: str, codex: str, sharper: str = json.dumps(SHARPER)
) -> dict[Family, FakeBackend]:
    """Claude and Codex encode with the given sources; any family answers a sharpen call."""

    def answer(source: str) -> Reply:
        return lambda prompt: fenced(source) if is_encoding_prompt(prompt) else sharper

    return {
        Family.claude: FakeBackend(Family.claude, answer(claude)),
        Family.codex: FakeBackend(Family.codex, answer(codex)),
        Family.gemini: FakeBackend(Family.gemini, lambda prompt: sharper),
        Family.local: FakeBackend(Family.local, answer(GOOD)),
    }


def make_record(local: str = "a", author: Family = Family.gemini) -> Record:
    return Record(
        id=f"{RUN}.{local}",
        run_id=RUN,
        pair=PAIR,
        mode="plain",
        statement="For every integer t in 0..9, 1 - t/10 >= 0.",
        objects=["t"],
        assumptions=["t is an integer"],
        domain_of_validity="0 <= t <= 9",
        instance_family="t: an integer in 0..9",
        provenance=Provenance(family=author, model="m", prompt_hash="h"),
    )


@pytest.fixture
def store(tmp_path: Path) -> RecordStore:
    return RecordStore(tmp_path)


@pytest.fixture
def ledger(store: RecordStore) -> Ledger:
    save_manifest(
        store.root,
        RunManifest(
            run_id=RUN,
            mode="plain",
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


BUDGET = Budget(tokens_per_pair=1_000_000)


# --- agree ---


def test_agreeing_encodings_pass() -> None:
    result = agree(GOOD, toy("(10 - t) / 10", name="Other"), n=20, assumptions=1)
    assert result == Agreement(agreed=True, compared=20, reason=None)


def test_disagreeing_margins_do_not_agree() -> None:
    result = agree(GOOD, OFF, n=20, assumptions=1)
    assert not result.agreed
    assert result.reason is not None and "margin" in result.reason


def test_parameter_names_must_match() -> None:
    result = agree(GOOD, RENAMED, n=20, assumptions=1)
    assert not result.agreed
    assert result.reason is not None and "parameter names" in result.reason


def test_disagreeing_predicates_do_not_agree() -> None:
    wider = GOOD.replace("0 <= t <= 9", "0 <= t <= 5").replace("class Toy", "class Narrow")
    wider = wider.replace("ENCODING = Toy", "ENCODING = Narrow")
    result = agree(GOOD, wider, n=40, assumptions=1)
    assert not result.agreed
    assert result.reason is not None and "in_domain" in result.reason


def test_assumption_count_must_match_the_record() -> None:
    result = agree(GOOD, GOOD, n=10, assumptions=2)
    assert not result.agreed
    assert result.reason is not None and "assumptions" in result.reason


def test_encoding_needs_a_certificate_capability() -> None:
    result = agree(GOOD, toy("1 - t / 10", name="Bare", capability=False), n=10, assumptions=1)
    assert not result.agreed
    assert result.reason is not None and "exact_margin" in result.reason


def test_failing_encoding_does_not_agree() -> None:
    result = agree(GOOD, "import os\n" + GOOD, n=10, assumptions=1)
    assert not result.agreed
    assert result.reason is not None and "forbidden import" in result.reason


def test_out_of_domain_samples_do_not_count_as_agreement() -> None:
    outside = GOOD.replace("rng.randint(0, 9)", "rng.randint(20, 29)")
    result = agree(outside, outside, n=10, assumptions=1)
    assert not result.agreed
    assert result.reason is not None and "domain" in result.reason


# --- prompts ---


def test_encoding_prompt_asks_for_names_and_a_capability() -> None:
    prompt = encoding_prompt(make_record())
    assert "t: an integer in 0..9" in prompt
    for capability in ("exact_margin", "margin_ball", "primal_dual"):
        assert capability in prompt
    assert "ENCODING" in prompt


def test_extract_source_takes_the_fenced_block() -> None:
    assert extract_source(fenced(GOOD)).strip() == GOOD.strip()
    assert extract_source(GOOD) == GOOD


@pytest.mark.parametrize(
    "text",
    ["not json", json.dumps({**SHARPER, "statement": ""}), json.dumps(["a"]), "{}"],
)
def test_parse_statement_refuses_bad_replies(text: str) -> None:
    with pytest.raises(ValueError):
        parse_statement(text)


def test_parse_statement_reads_a_fenced_reply() -> None:
    parsed = parse_statement(f"Sure.\n```json\n{json.dumps(SHARPER)}\n```")
    assert parsed.statement == SHARPER["statement"]


# --- encode_twice ---


@pytest.mark.parametrize("author", [Family.gemini, Family.claude, Family.codex])
def test_encoders_are_two_families_never_gemini(
    store: RecordStore, ledger: Ledger, author: Family
) -> None:
    record = make_record(author=author)
    store.create(record)
    fakes = backends(GOOD, OFF)
    a, b = encode_twice(record, ledger=ledger, budget=BUDGET, backends=fakes)
    assert {a.family, b.family} == {Family.claude, Family.codex}
    by_family = {a.family: a, b.family: b}
    assert by_family[Family.claude].source.strip() == GOOD.strip()
    assert by_family[Family.codex].source.strip() == OFF.strip()
    assert fakes[Family.gemini].prompts == []
    # Neither encoder sees the other's output: both get the same prompt, from the record.
    assert fakes[Family.claude].prompts == fakes[Family.codex].prompts == [encoding_prompt(record)]


# --- the encoding stage and the ambiguous rounds ---


def test_agreed_record_keeps_its_status_and_stores_both_encodings(
    store: RecordStore, ledger: Ledger
) -> None:
    store.create(make_record())
    result = run_encoding_stage(
        f"{RUN}.a", ledger=ledger, budget=BUDGET, backends=backends(GOOD, GOOD), n=10
    )
    assert result.agreement.agreed
    assert result.child is None
    record = store.load(f"{RUN}.a")
    assert record.status is Status.proposed
    output = record.stages[Stage.encoding]
    assert sorted(output.evidence) == [
        f"encodings/{RUN}.a.claude.py",
        f"encodings/{RUN}.a.codex.py",
    ]
    for path in output.evidence:
        assert (store.root / "runs" / RUN / path).read_text().strip() == GOOD.strip()
    encoders = [Sandboxed(source=s, family=f) for f, s in result.sources.items()]
    assert {e.family for e in encoders} == {Family.claude, Family.codex}


def test_disagreeing_encodings_give_ambiguous(store: RecordStore, ledger: Ledger) -> None:
    store.create(make_record())
    fakes = backends(GOOD, OFF)
    result = run_encoding_stage(f"{RUN}.a", ledger=ledger, budget=BUDGET, backends=fakes, n=10)
    assert not result.agreement.agreed
    assert store.load(f"{RUN}.a").status is Status.ambiguous
    # The generator family (Gemini here) was asked for a sharper statement.
    assert len(fakes[Family.gemini].prompts) == 1
    assert result.child is not None
    child = store.load(result.child)
    assert child.provenance.parents == [f"{RUN}.a"]
    assert child.provenance.family is Family.gemini
    assert child.statement == SHARPER["statement"]
    assert child.status is Status.proposed
    assert Stage.encoding not in child.stages


def test_ambiguous_rounds_end_in_abandoned(store: RecordStore, ledger: Ledger) -> None:
    store.create(make_record())
    fakes = backends(GOOD, OFF)
    record_id = f"{RUN}.a"
    statuses: list[Status] = []
    for _ in range(3):
        result = run_encoding_stage(record_id, ledger=ledger, budget=BUDGET, backends=fakes, n=10)
        statuses.append(store.load(record_id).status)
        if result.child is None:
            break
        record_id = result.child
    assert statuses == [Status.ambiguous, Status.ambiguous, Status.abandoned]
    # Two sharpen calls, none after the last round.
    assert len(fakes[Family.gemini].prompts) == 2


def test_unparseable_sharpening_leaves_the_record_ambiguous(
    store: RecordStore, ledger: Ledger
) -> None:
    store.create(make_record())
    fakes = backends(GOOD, OFF, sharper="no json here")
    result = run_encoding_stage(f"{RUN}.a", ledger=ledger, budget=BUDGET, backends=fakes, n=10)
    assert result.child is None
    assert store.load(f"{RUN}.a").status is Status.ambiguous
    # Resume: a later sharpen call with a good reply makes the child.
    good = backends(GOOD, OFF)
    child = sharpen(f"{RUN}.a", ledger=ledger, budget=BUDGET, backends=good)
    assert child is not None
    assert store.load(child).provenance.parents == [f"{RUN}.a"]
    # And sharpening again returns the same child, with no new call.
    assert sharpen(f"{RUN}.a", ledger=ledger, budget=BUDGET, backends=good) == child
    assert len(good[Family.gemini].prompts) == 1


def test_stage_refuses_a_record_that_is_not_proposed(store: RecordStore, ledger: Ledger) -> None:
    record = make_record().model_copy(update={"instance_family": None, "status": Status.untestable})
    store.create(record)
    fakes = backends(GOOD, GOOD)
    with pytest.raises(ValueError, match="untestable"):
        run_encoding_stage(f"{RUN}.a", ledger=ledger, budget=BUDGET, backends=fakes, n=10)
    assert all(f.prompts == [] for f in fakes.values())


def test_encode_charges_the_encoding_stage(store: RecordStore, ledger: Ledger) -> None:
    store.create(make_record())
    run_encoding_stage(f"{RUN}.a", ledger=ledger, budget=BUDGET, backends=backends(GOOD, GOOD), n=5)
    stages = {(c.family, c.stage) for c in ledger.charges()}
    assert stages == {(Family.claude, Stage.encoding), (Family.codex, Stage.encoding)}
