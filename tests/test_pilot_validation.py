"""Validation against the pilot (build plan doc-5wh, Task 17; design doc-gmj,
"Validation against the pilot").

Before the benchmark, the filters must reproduce the pilot's lessons:

- the three wrong pilot kills (warm-start artefact, grid gap, point outside
  the domain) are rejected, and the certified S7 and S7' counterexamples are
  accepted;
- S8's KKT hypothesis is flagged by the hypothesis audit;
- S1 and S8 are flagged by the triviality screen (live; at least 2 of 3 runs).

Each case goes in at the deepest point the pipeline uses for it. The kills go
through `check_kill` with two sandboxed encodings, which is the call the
falsifier stage makes on every candidate; the falsifier's search cannot find
these hand-built points itself (user decision, bead minilab-cf6a85). S8 and
S1 are hand-built survivor records run through `drive` with the real
formaliser and triviality stages. Offline tests use scripted model replies;
only the `live` tests call a model.

The pilot has one hand-built encoding per conjecture, so in most kill checks
both encodings are the same source under two family labels. A mixed pair (the
encoding with its domain check removed, next to the real one) checks that a
kill needs both encodings to accept. Nothing here checks how independent two
real encodings are.
"""

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import mpmath
import pytest
from fixtures.pilot.s7_y2 import Y2Encoding, rho_float
from pydantic import JsonValue

from minilab.certify import (
    CERTIFICATE,
    Certificate,
    DualityCertificate,
    IntervalCertificate,
    check_kill,
)
from minilab.encoding.api import Instance, to_fraction
from minilab.encoding.sandbox import Sandboxed
from minilab.models.backends import ModelReply
from minilab.pipeline import DriverSettings, StageContext, default_runners, drive, start_run
from minilab.records import Family, Provenance, Record, RecordStore, Stage, StageOutput, Status
from minilab.run import DEFAULT_MODELS, MemoryPolicy, RunManifest, default_models
from minilab.scheduler import Pair
from minilab.screen.triviality import hold_reasons

PILOT = Path(__file__).parent / "fixtures" / "pilot"
RUN = "pilot"
PAIR: Pair = ("information-theory", "optimisation")
FAMILIES: tuple[Family, ...] = (Family.claude, Family.codex, Family.gemini)
NOW = datetime(2026, 11, 28, 1, 0, tzinfo=UTC)

S1_LEAN = (PILOT / "s1.lean").read_text(encoding="utf-8")
S8_LEAN = (PILOT / "s8.lean").read_text(encoding="utf-8")


def source_of(module: str, cls: str, extra: str = "") -> str:
    """A pilot encoding as sandbox source: the module, then `ENCODING = cls`."""
    return (PILOT / f"{module}.py").read_text(encoding="utf-8") + extra + f"\n\nENCODING = {cls}\n"


S7 = source_of("s7", "S7Encoding")
Y2 = source_of("s7_y2", "Y2Encoding")


def load(name: str) -> Certificate:
    return CERTIFICATE.validate_json((PILOT / f"{name}.json").read_text(encoding="utf-8"))


def two(source: str) -> tuple[Sandboxed, Sandboxed]:
    """The same source as two families' encodings (see the module docstring)."""
    return Sandboxed(source, Family.claude), Sandboxed(source, Family.codex)


# --- kills ---


def test_warm_start_artefact_rejected_as_a_kill() -> None:
    cert = load("warm_start_artefact")
    assert isinstance(cert, DualityCertificate)
    # The certificate claims a large violation: trusting the solver's warm
    # start, a falsifier would have reported a kill.
    assert to_fraction(cert.violation) < -18
    result = check_kill(*two(S7), cert, assumptions=2)
    assert not result.accepted
    assert "upper bound on the margin" in result.reason


@pytest.mark.parametrize("digits", [16, 15])
def test_grid_gap_rejected_as_a_kill(digits: int) -> None:
    stored = load("grid_gap")
    assert isinstance(stored, IntervalCertificate)
    # The stored certificate asks for 16 digits, where even a plain mpf margin
    # is already positive. At 15 digits a plain mpf margin still shows the
    # float artefact, so only the ball arithmetic and the repeat at more
    # digits reject it.
    point = stored.point.model_copy(update={"precision_digits": digits})
    cert = stored.model_copy(update={"digits": digits, "point": point})
    c, a1, b = (_scalar(cert.point, k) for k in ("c", "a1", "b"))
    assert rho_float(c, a1, b) > 0.25  # the float grid shows a violation here
    if digits == 15:
        with mpmath.mp.workdps(15):
            assert Y2Encoding().margin(cert.point) < 0  # plain precision is fooled
    with mpmath.mp.workdps(100):
        assert Y2Encoding().margin(cert.point) > 0  # at 100 digits there is none
    result = check_kill(*two(Y2), cert, assumptions=2)
    assert not result.accepted
    assert "below zero" in result.reason


ANY_DOMAIN = """

class AnyDomain(S7Encoding):
    def in_domain(self, x):
        return True
"""


def test_out_of_domain_point_rejected_as_a_kill() -> None:
    cert = load("out_of_domain")
    result = check_kill(*two(S7), cert, assumptions=2)
    assert not result.accepted
    assert "outside the domain" in result.reason
    # Only the domain check stops it: with that check gone, both sandboxed
    # encodings accept the same certificate as a kill.
    loose = source_of("s7", "AnyDomain", ANY_DOMAIN)
    assert check_kill(*two(loose), cert, assumptions=2).accepted
    # A kill needs both encodings: one that accepts is not enough, in
    # either position.
    for first, second, refusing in (
        (Sandboxed(loose, Family.claude), Sandboxed(S7, Family.codex), "second"),
        (Sandboxed(S7, Family.claude), Sandboxed(loose, Family.codex), "first"),
    ):
        mixed = check_kill(first, second, cert, assumptions=2)
        assert not mixed.accepted
        assert f"the {refusing} encoding refuses" in mixed.reason
        assert "outside the domain" in mixed.reason


@pytest.mark.parametrize("name", ["s7_counterexample", "s7prime_counterexample"])
def test_s7_counterexamples_accepted_as_kills(name: str) -> None:
    result = check_kill(*two(S7), load(name), assumptions=2)
    assert result.accepted, result.reason


def _scalar(x: Instance, key: str) -> float:
    value = x.values[key]
    assert not isinstance(value, list)
    return float(to_fraction(value))


# --- survivors through the driver ---

S8_STATEMENT = (
    "Every discrete memoryless channel W has a capacity-achieving input distribution "
    "whose support has at most rank(W) letters."
)
S8_ASSUMPTIONS = [
    "W is a discrete memoryless channel: a matrix with m input rows and n output columns, "
    "each row a probability distribution",
    "p is a capacity-achieving input distribution for W",
]
S1_STATEMENT = (
    "For q in (0, 1), the Bregman divergence of the Tsallis potential "
    "phi_q(p) = (sum_i p_i^q - 1) / (q - 1) on the simplex satisfies "
    "B_q(p, r) >= (q / 2) ||p - r||_1^2."
)
S1_ASSUMPTIONS = [
    "q lies in the open interval (0, 1)",
    "p is a probability distribution on n letters",
    "r is a probability distribution on n letters",
    "every r_i is positive",
]


def kind_of(prompt: str) -> str:
    if "Write this conjecture as a Lean 4" in prompt:
        return "formalise"
    if "Translate the Lean theorem" in prompt:
        return "roundtrip"
    if "hypothesis audit" in prompt:
        return "audit"
    return "other"


class ScriptedBackend:
    """Replies by the kind of prompt from a queue that every family shares, so
    the order of replies does not depend on which family a stage picks."""

    def __init__(self, family: Family, replies: dict[str, list[str]]) -> None:
        self.family = family
        self.model = DEFAULT_MODELS[family]
        self.replies = replies
        self.prompts: list[tuple[str, str]] = []

    def run(
        self, prompt: str, *, cwd: Path, env: Mapping[str, str], timeout_s: float
    ) -> ModelReply:
        kind = kind_of(prompt)
        self.prompts.append((kind, prompt))
        queue = self.replies.get(kind)
        if not queue:
            raise AssertionError(f"{self.family.value} got an unscripted {kind} prompt")
        return ModelReply(
            text=queue.pop(0), tokens_in=10, tokens_out=10, estimated=False, model=self.model
        )


def scripted(replies: Mapping[str, Sequence[str]]) -> dict[Family, ScriptedBackend]:
    shared = {k: list(v) for k, v in replies.items()}
    return {f: ScriptedBackend(f, shared) for f in FAMILIES}


def lean_block(source: str) -> str:
    return f"Here it is.\n\n```lean\n{source}```\n"


def audit_reply(kinds: Mapping[str, tuple[str, str | None]]) -> str:
    return json.dumps(
        {
            "hypotheses": [
                {"label": label, "kind": kind, "counterpart": counterpart, "reason": "r"}
                for label, (kind, counterpart) in kinds.items()
            ],
            "hidden": [],
        }
    )


def start(store: RecordStore) -> None:
    start_run(
        store,
        RunManifest(
            run_id=RUN,
            mode="structured",
            pairs=(PAIR,),
            memory_policy=MemoryPolicy.blind,
            memory_before=None,
            literature_before=None,
            budget_tokens_per_pair=1_000_000,
            families=FAMILIES,
            models=default_models(FAMILIES),
            started=NOW,
        ),
        DriverSettings(),
    )


def no_generation(
    pair: Pair, family: Family, n: int, earlier: Sequence[str], ctx: StageContext
) -> Sequence[Record]:
    """The pilot records are built by hand, so the driver generates nothing."""
    return []


S1_WITNESS: dict[str, JsonValue] = {
    "values": {"n": "2", "q": "0.5", "p": ["0.3", "0.7"], "r": ["0.5", "0.5"]},
    "precision_digits": 30,
}
"""A point that satisfies all five of S1's hypotheses, as the falsifier's worst point."""


def survivor(
    store: RecordStore,
    local: str,
    statement: str,
    assumptions: list[str],
    witness: dict[str, JsonValue] | None = None,
) -> str:
    """A hand-built pilot record that has passed encoding, falsifier, and literature.

    `witness` is the falsifier's worst point, which the formaliser's vacuity
    check instantiates. S8 has none: its theorem takes a channel matrix,
    which a witness cannot give, so the formaliser stage notes that no
    witness was instantiated, and a proof of it is held, never trivial.
    """
    record = Record(
        id=f"{RUN}.{local}",
        run_id=RUN,
        pair=PAIR,
        mode="structured",
        statement=statement,
        objects=[],
        assumptions=assumptions,
        domain_of_validity="finite alphabets",
        instance_family="alphabets of 2 to 4 letters",
        provenance=Provenance(family=Family.gemini, model="m", prompt_hash="h"),
    )
    store.create(record)
    coverage: dict[str, JsonValue] = {
        "space": "hand-built pilot record",
        "dimensions": 0,
        "starts": 0,
        "worst_margin": None,
        "worst_point": witness,
        "uncovered": [],
    }
    store.write_stage(
        record.id,
        Stage.encoding,
        StageOutput(written_at=NOW, data={"agreed": True}, notes="hand-built pilot record"),
    )
    store.write_stage(
        record.id,
        Stage.falsifier,
        StageOutput(written_at=NOW, data={"coverage": coverage}),
        status=Status.survived,
    )
    store.write_stage(record.id, Stage.literature, StageOutput(written_at=NOW, notes="none"))
    return record.id


def formalise_through_driver(
    store: RecordStore, backends: Mapping[Family, ScriptedBackend]
) -> None:
    """Run the real formaliser stage on every survivor, through `drive`."""
    runners = {Stage.formaliser: default_runners()[Stage.formaliser]}
    outcome = drive(store, RUN, runners=runners, generator=no_generation, backends=backends)
    assert outcome.stopped is None, outcome.stopped
    assert outcome.skipped == []


S8_VERDICTS: dict[str, tuple[str, str | None]] = {
    "hW": ("matches", "assumption 1"),
    "hp": ("matches", "assumption 2"),
}
S1_VERDICTS: dict[str, tuple[str, str | None]] = {
    "hq0": ("matches", "assumption 1"),
    "hq1": ("matches", "assumption 1"),
    "hp": ("matches", "assumption 2"),
    "hr": ("matches", "assumption 3"),
    "hrpos": ("matches", "assumption 4"),
}


def test_s8_kkt_hypothesis_flagged_through_the_driver(tmp_path: Path) -> None:
    store = RecordStore(tmp_path)
    start(store)
    flagged = survivor(store, "s8", S8_STATEMENT, S8_ASSUMPTIONS)
    # The control: the same file, but the checker matches hKKT to an
    # assumption. It shows that the refusal comes from the audit's verdict on
    # hKKT and from nothing else in the stage.
    control = survivor(store, "s8control", S8_STATEMENT, S8_ASSUMPTIONS)
    assert flagged < control  # the driver runs records in id order
    backends = scripted(
        {
            "formalise": [lean_block(S8_LEAN)] * 2,
            "roundtrip": ["For every channel W ..."] * 2,
            "audit": [
                audit_reply({**S8_VERDICTS, "hKKT": ("begs_question", None)}),
                audit_reply({**S8_VERDICTS, "hKKT": ("matches", "assumption 2")}),
            ],
        }
    )
    formalise_through_driver(store, backends)

    data = store.load(flagged).stages[Stage.formaliser].data
    assert data["rejected"] is True
    reasons = data["reasons"]
    assert isinstance(reasons, list) and reasons
    assert all("hKKT" in str(r) for r in reasons), reasons
    audit = data["audit"]
    assert isinstance(audit, dict) and audit["extra"] == ["hKKT"]
    hypotheses = audit["hypotheses"]
    assert isinstance(hypotheses, list)
    kkt = [h for h in hypotheses if isinstance(h, dict) and h.get("label") == "hKKT"]
    assert kkt == [kkt[0]] and kkt[0]["kind"] == "begs_question" and kkt[0]["by"] == "checker"
    assert data["lean_statement"] is None
    assert store.load(flagged).status is Status.survived

    control_data = store.load(control).stages[Stage.formaliser].data
    assert control_data["rejected"] is False, control_data["reasons"]

    # The audit was asked by a checker, never by Gemini.
    audits = [f for f, b in backends.items() for k, _ in b.prompts if k == "audit"]
    assert len(audits) == 2
    assert Family.gemini not in audits

    # Then the triviality stage leaves the refused formalisation unscreened,
    # with no model call (a call would fail: nothing is scripted). The runner
    # is called for the flagged record alone, because the control would need
    # a live prover.
    assert Stage.triviality not in store.load(control).stages
    triviality = default_runners()[Stage.triviality]
    triviality(flagged, StageContext.for_run(store, RUN, scripted({})))
    screen = store.load(flagged).stages[Stage.triviality].data
    assert screen["screened"] is False
    assert screen["trivial"] is False
    why = screen["reasons"]
    assert isinstance(why, list) and "rejected the formalisation" in str(why[0])


def test_s1_formalisation_is_clean_through_the_driver(tmp_path: Path) -> None:
    """With the witness and a clean audit, nothing would hold a proof of S1
    back, so the live screen can set it trivial."""
    store = RecordStore(tmp_path)
    start(store)
    record_id = survivor(store, "s1", S1_STATEMENT, S1_ASSUMPTIONS, S1_WITNESS)
    backends = scripted(
        {
            "formalise": [lean_block(S1_LEAN)],
            "roundtrip": ["the theorem in English"],
            "audit": [audit_reply(S1_VERDICTS)],
        }
    )
    formalise_through_driver(store, backends)
    data = store.load(record_id).stages[Stage.formaliser].data
    assert data["rejected"] is False, data["reasons"]
    vacuity = data["vacuity"]
    assert isinstance(vacuity, dict) and vacuity["satisfied_by_witness"] is True
    assert hold_reasons(data) == []


# --- the triviality screen through the driver (live) ---

RUNS = 3
NEEDED = 2


def screened_through_driver(
    store: RecordStore,
    lean: str,
    audit_kinds: Mapping[str, tuple[str, str | None]],
    *,
    s1: bool,
) -> list[str]:
    """Formalise RUNS copies with scripted models, then screen them with live ones.

    The copies are screened one after another in one `drive` call. If the
    deterministic tactic battery proves the theorem, all three pass together,
    so "2 of 3" then means one result, not three tries of a model.
    """
    start(store)
    ids = [
        survivor(
            store,
            f"{'s1' if s1 else 's8'}{i}",
            S1_STATEMENT if s1 else S8_STATEMENT,
            S1_ASSUMPTIONS if s1 else S8_ASSUMPTIONS,
            S1_WITNESS if s1 else None,
        )
        for i in range(RUNS)
    ]
    backends = scripted(
        {
            "formalise": [lean_block(lean)] * RUNS,
            "roundtrip": ["the theorem in English"] * RUNS,
            "audit": [audit_reply(audit_kinds)] * RUNS,
        }
    )
    formalise_through_driver(store, backends)
    for record_id in ids:
        assert store.load(record_id).stages[Stage.formaliser].data["rejected"] is False

    outcome = drive(
        store,
        RUN,
        runners={Stage.triviality: default_runners()[Stage.triviality]},
        generator=no_generation,
    )
    assert outcome.stopped is None, outcome.stopped
    assert outcome.skipped == [], outcome.skipped
    for record_id in ids:
        data = store.load(record_id).stages[Stage.triviality].data
        shown = ("trivial", "held", "proved_by", "corollary_of")
        print(record_id, json.dumps({k: data.get(k) for k in shown}))
    return ids


def screen_of(store: RecordStore, record_id: str) -> dict[str, JsonValue]:
    return store.load(record_id).stages[Stage.triviality].data


@pytest.mark.live
def test_s1_flagged_trivial_through_the_driver(tmp_path: Path) -> None:
    store = RecordStore(tmp_path)
    ids = screened_through_driver(store, S1_LEAN, S1_VERDICTS, s1=True)
    trivial = [r for r in ids if screen_of(store, r)["trivial"] is True]
    assert len(trivial) >= NEEDED
    assert all(store.load(r).status is Status.trivial for r in trivial)


@pytest.mark.live
def test_s8_flagged_trivial_through_the_driver(tmp_path: Path) -> None:
    """A counterfactual: on the real path the audit refuses hKKT and S8 is never
    screened (the offline test above). Here the audit states hKKT as a
    weakening, so the screen runs as a second line of defence; a proof is
    held for review, not set trivial."""
    store = RecordStore(tmp_path)
    kinds = {**S8_VERDICTS, "hKKT": ("weakening", None)}
    ids = screened_through_driver(store, S8_LEAN, kinds, s1=False)
    screens = [screen_of(store, r) for r in ids]
    proved = [d for d in screens if d["screened"] is True and d["proved_by"] is not None]
    assert len(proved) >= NEEDED
    for data in proved:
        assert data["trivial"] is False and data["held"] is True
        holds = data["hold_reasons"]
        assert isinstance(holds, list)
        assert any("hKKT" in str(h) and "weakening" in str(h) for h in holds), holds
