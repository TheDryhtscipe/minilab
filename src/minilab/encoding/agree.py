"""Dual encodings and their agreement (build plan doc-5wh, Task 7; minilab-e47de8).

Code written by a model sits inside every verdict, so two model families each
write the encoding from the record alone, and both are run on the same samples
(design doc-gmj, "Encoding"):

- `encode_twice` asks two families, never Gemini, chosen by `pick_checker`'s
  rules. Both get the same prompt, built from the record only.
- `agree` runs both encodings in the sandbox on `n` shared samples, half drawn
  by each sampler, and compares the parameter names, `in_domain`, the
  assumption predicates, and (where both say the point is in the domain) the
  margins.
- `run_encoding_stage` does both and writes the encoding stage. Agreement
  leaves the status as it is. Disagreement sets `ambiguous` and calls
  `sharpen`, which asks the generator family for a sharper statement and stores
  it as a child record (provenance parent: the ambiguous record). A record whose
  chain already holds `MAX_ROUNDS` ambiguous ancestors is `abandoned` instead,
  with no further call (user decisions, 25 Nov 2026).

A record never goes from here to a verdict: this stage only lets it on to the
falsifier, or stops it.
"""

import hashlib
import random
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import mpmath
from pydantic import BaseModel, ConfigDict, Field

from minilab.certify.result import working_digits
from minilab.encoding.api import Instance
from minilab.encoding.prompts import (
    encoding_prompt,
    extract_source,
    parse_statement,
    sharpen_prompt,
)
from minilab.encoding.sandbox import (
    CAPABILITIES,
    DEFAULT_TIMEOUT_S,
    Call,
    EncodingOutput,
    Method,
    Sandboxed,
    SandboxFailure,
    run_encoding,
)
from minilab.ledger import Ledger
from minilab.models.adapter import Budget, Role, call, pick_checker
from minilab.models.backends import Backend
from minilab.records import (
    Family,
    Provenance,
    Record,
    RecordExists,
    RecordStore,
    Stage,
    StageOutput,
    Status,
)

# How many times an ambiguous record is sent back to be sharpened. A record
# with this many ambiguous ancestors that still disagrees is abandoned.
MAX_ROUNDS = 2

# The working precision for drawing samples, in decimal digits. Each sample
# names its own precision for evaluation.
SAMPLE_DIGITS = 50

# Margins closer than this agree whatever their size, so that two exact zeros
# computed differently are not a disagreement.
DEFAULT_ATOL = 1e-30

# The evaluation calls made on each shared sample, in this order.
_EVAL: tuple[Method, ...] = ("in_domain", "assumptions", "margin")


class Agreement(BaseModel):
    """Whether two encodings agree, on how many samples, and if not, the first reason."""

    model_config = ConfigDict(frozen=True)

    agreed: bool
    compared: int = Field(ge=0)
    """Samples compared before the answer: all `n` when they agree."""
    reason: str | None


class StageResult(BaseModel):
    """What the encoding stage did to one record."""

    model_config = ConfigDict(frozen=True)

    agreement: Agreement
    status: Status
    sources: dict[Family, str]
    """Each encoding's source, by the family that wrote it."""
    child: str | None
    """The sharper record made after a disagreement, if any."""


def _no(reason: str, compared: int = 0) -> Agreement:
    return Agreement(agreed=False, compared=compared, reason=reason)


def _sample(source: str, seeds: Sequence[int], timeout_s: float) -> tuple[list[Instance], str]:
    """Draw one sample per seed. Return the samples and a failure reason ('' if none)."""
    calls = [Call(method="sample", seed=s, digits=SAMPLE_DIGITS) for s in seeds]
    output = run_encoding(source, calls, timeout_s)
    if output.failure is not None:
        return [], output.failure
    if not any(c in output.capabilities for c in CAPABILITIES):
        return [], f"it has none of {', '.join(CAPABILITIES)}"
    samples: list[Instance] = []
    for result in output.results:
        try:
            value = result.rebuild()
        except SandboxFailure as exc:
            return [], str(exc)
        if type(value) is not Instance:
            return [], "its sampler did not give an Instance"
        samples.append(value)
    return samples, ""


def _evaluate(source: str, points: Sequence[Instance], timeout_s: float) -> EncodingOutput:
    calls = [Call(method=m, point=x, digits=x.precision_digits) for x in points for m in _EVAL]
    return run_encoding(source, calls, timeout_s)


def _margins_close(a: mpmath.mpf, b: mpmath.mpf, rtol: float, atol: float) -> bool:
    gap = abs(a - b)
    return bool(gap <= atol or gap <= rtol * max(abs(a), abs(b)))


def agree(
    a: str,
    b: str,
    n: int = 200,
    rtol: float = 1e-9,
    *,
    assumptions: int,
    atol: float = DEFAULT_ATOL,
    seed: int = 0,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> Agreement:
    """Say whether encodings `a` and `b` (sources) agree on `n` shared samples.

    `a` draws the first half of the samples and `b` the rest, from seeds made
    from `seed`. They agree when: both run; each has at least one of
    `exact_margin`, `margin_ball`, `primal_dual`; every sample from both uses
    the same parameter names; on every sample both give the same `in_domain`
    and the same list of `assumptions` (of length `assumptions`, the record's
    count); where both say the point is in the domain, their margins are within
    `rtol` of each other relative to the larger, or within `atol`; and at least
    a quarter of the samples are in the domain. An error in any call is a
    disagreement. Never raises for anything an encoding does.
    """
    if n < 2:
        raise ValueError(f"agree needs at least 2 samples, one from each sampler; got {n}")
    rng = random.Random(seed)
    seeds = [rng.getrandbits(63) for _ in range(n)]
    half = (n + 1) // 2
    points: list[Instance] = []
    names: dict[str, frozenset[str]] = {}
    for label, source, mine in (("a", a, seeds[:half]), ("b", b, seeds[half:])):
        samples, failure = _sample(source, mine, timeout_s)
        if failure:
            return _no(f"encoding {label} failed: {failure}")
        keys = {frozenset(x.values) for x in samples}
        if len(keys) != 1:
            return _no(f"encoding {label}'s samples use different parameter names from each other")
        names[label] = keys.pop()
        points.extend(samples)
    if names["a"] != names["b"]:
        return _no(
            f"the parameter names differ: a uses {sorted(names['a'])}, b uses {sorted(names['b'])}"
        )
    outputs: dict[str, EncodingOutput] = {}
    for label, source in (("a", a), ("b", b)):
        output = _evaluate(source, points, timeout_s)
        if output.failure is not None:
            return _no(f"encoding {label} failed: {output.failure}")
        outputs[label] = output
    in_domain = 0
    for i, x in enumerate(points):
        where = f"at sample {i} {dict(x.values)}"
        seen: dict[str, tuple[bool, list[bool], mpmath.mpf | None]] = {}
        for label, output in outputs.items():
            first = len(_EVAL) * i
            try:
                domain = output.results[first].rebuild()
                held = output.results[first + 1].rebuild()
                margin = output.results[first + 2].rebuild() if domain is True else None
            except SandboxFailure as exc:
                return _no(f"encoding {label} raised {where}: {exc}", i)
            if type(domain) is not bool:
                return _no(f"encoding {label}'s in_domain is not a bool {where}", i)
            if not isinstance(held, list):
                return _no(f"encoding {label}'s assumptions are not a list {where}", i)
            flags: list[bool] = []
            for flag in cast(list[object], held):
                if type(flag) is not bool:
                    return _no(f"encoding {label}'s assumptions are not all bools {where}", i)
                flags.append(flag)
            if len(flags) != assumptions:
                return _no(
                    f"encoding {label} gives {len(flags)} assumptions; "
                    f"the record has {assumptions}",
                    i,
                )
            if margin is not None and type(margin) is not mpmath.mpf:
                return _no(f"encoding {label}'s margin is not an mpf {where}", i)
            seen[label] = (domain, flags, margin)
        (dom_a, held_a, m_a), (dom_b, held_b, m_b) = seen["a"], seen["b"]
        if dom_a != dom_b:
            return _no(f"in_domain differs {where}: a says {dom_a}, b says {dom_b}", i)
        if held_a != held_b:
            return _no(f"the assumptions differ {where}: a says {held_a}, b says {held_b}", i)
        if m_a is not None and m_b is not None:
            in_domain += 1
            with working_digits(x.precision_digits):
                close = _margins_close(m_a, m_b, rtol, atol)
                shown = (mpmath.nstr(m_a, 12), mpmath.nstr(m_b, 12))
            if not close:
                return _no(f"the margin differs {where}: a gives {shown[0]}, b gives {shown[1]}", i)
    if in_domain < max(1, n // 4):
        return _no(f"only {in_domain} of {n} samples are in the domain", n)
    return Agreement(agreed=True, compared=n, reason=None)


def encoders(author: Family, available: Sequence[Family]) -> tuple[Family, Family]:
    """The two families that encode work by `author`: never Gemini, never one twice.

    The first is `pick_checker(author)`; the second is `pick_checker` of the
    first, which may be the author.
    """
    first = pick_checker(author, available)
    second = pick_checker(first, available)
    return first, second


def encode_twice(
    record: Record,
    *,
    ledger: Ledger,
    budget: Budget,
    backends: Mapping[Family, Backend] | None = None,
) -> tuple[Sandboxed, Sandboxed]:
    """Ask two families for the record's encoding, each from the record alone.

    Each comes back as a `Sandboxed` whose `family` is the family that wrote
    it, as `check_kill` needs. Model failures (`ModelUnavailable`,
    `BudgetExceeded`, ...) are raised as `call` raises them; nothing is written.
    """
    prompt = encoding_prompt(record)
    found: list[Sandboxed] = []
    for family in encoders(record.provenance.family, ledger.manifest.families):
        reply = call(
            family,
            Role.encode,
            prompt,
            record_id=record.id,
            pair=record.pair,
            ledger=ledger,
            budget=budget,
            backends=backends,
        )
        found.append(Sandboxed(source=extract_source(reply.text), family=family))
    return found[0], found[1]


def _round(store: RecordStore, record: Record) -> int:
    """How many ambiguous ancestors the record has: the sharpenings that led to it."""
    count = 0
    seen = {record.id}
    current = record
    while current.provenance.parents:
        parent_id = current.provenance.parents[0]
        if parent_id in seen:
            break
        seen.add(parent_id)
        try:
            parent = store.load(parent_id)
        except KeyError:
            break
        if parent.status is not Status.ambiguous:
            break
        count += 1
        current = parent
    return count


def _encodings_dir(store: RecordStore, run_id: str) -> Path:
    return store.root / "runs" / run_id / "encodings"


def run_encoding_stage(
    record_id: str,
    *,
    ledger: Ledger,
    budget: Budget,
    backends: Mapping[Family, Backend] | None = None,
    n: int = 200,
    rtol: float = 1e-9,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> StageResult:
    """Encode a proposed record twice, compare the encodings, and write the encoding stage.

    Each encoding is saved under the run at `encodings/<record_id>.<family>.py`
    and named in the stage's evidence. Agreement leaves the status unchanged;
    disagreement gives `ambiguous`, then a `sharpen` call, or `abandoned` once
    the record has `MAX_ROUNDS` ambiguous ancestors. Raise ValueError for a
    record that is not `proposed` or already has an encoding stage. A model
    failure while encoding leaves the record unchanged; one while sharpening
    leaves it ambiguous, and `sharpen` can be called again.
    """
    store = ledger.store
    record = store.load(record_id)
    if record.status is not Status.proposed:
        raise ValueError(f"{record_id} is {record.status}; only a proposed record is encoded")
    if Stage.encoding in record.stages:
        raise ValueError(f"{record_id} already has an encoding stage")
    a, b = encode_twice(record, ledger=ledger, budget=budget, backends=backends)
    directory = _encodings_dir(store, record.run_id)
    directory.mkdir(parents=True, exist_ok=True)
    evidence: list[str] = []
    for enc in (a, b):
        name = f"{record.id}.{enc.family.value}.py"
        (directory / name).write_text(enc.source, encoding="utf-8")
        evidence.append(f"encodings/{name}")
    agreement = agree(
        a.source,
        b.source,
        n,
        rtol,
        assumptions=len(record.assumptions),
        timeout_s=timeout_s,
    )
    rounds = _round(store, record)
    if agreement.agreed:
        status = record.status
    elif rounds >= MAX_ROUNDS:
        status = Status.abandoned
    else:
        status = Status.ambiguous
    store.write_stage(
        record.id,
        Stage.encoding,
        StageOutput(
            written_at=datetime.now(UTC),
            data={
                "families": [a.family.value, b.family.value],
                "agreed": agreement.agreed,
                "compared": agreement.compared,
                "reason": agreement.reason,
                "round": rounds,
            },
            evidence=evidence,
            notes=agreement.reason or "",
        ),
        status=status,
    )
    child = None
    if status is Status.ambiguous:
        child = sharpen(record.id, ledger=ledger, budget=budget, backends=backends)
    return StageResult(
        agreement=agreement,
        status=status,
        sources={a.family: a.source, b.family: b.source},
        child=child,
    )


def _child_id(record: Record, rounds: int) -> str:
    local = record.id.split(".", 1)[1]
    digest = hashlib.sha256(record.id.encode()).hexdigest()[:6]
    suffix = f"-r{rounds + 1}-{digest}"
    return f"{record.run_id}.{local[: 64 - len(suffix)]}{suffix}"


def _existing_child(store: RecordStore, record: Record) -> str | None:
    for other in store.list(record.run_id):
        if record.id in other.provenance.parents:
            return other.id
    return None


def sharpen(
    record_id: str,
    *,
    ledger: Ledger,
    budget: Budget,
    backends: Mapping[Family, Backend] | None = None,
) -> str | None:
    """Ask the generator family for a sharper statement of an ambiguous record.

    Store it as a new proposed record (or untestable, without an instance
    family) whose provenance names the ambiguous one as its parent, and return
    its id. If the record already has such a child, return it with no call.
    Return None if the reply cannot be read; the record stays ambiguous and
    this can be called again. Raise ValueError for a record that is not
    ambiguous, and whatever `call` raises for a model failure.
    """
    store = ledger.store
    record = store.load(record_id)
    if record.status is not Status.ambiguous or Stage.encoding not in record.stages:
        raise ValueError(f"{record_id} is {record.status}; only an ambiguous record is sharpened")
    existing = _existing_child(store, record)
    if existing is not None:
        return existing
    reason = record.stages[Stage.encoding].notes or "the encodings disagreed"
    prompt = sharpen_prompt(record, reason)
    family = record.provenance.family
    reply = call(
        family,
        Role.generate,
        prompt,
        record_id=record.id,
        pair=record.pair,
        ledger=ledger,
        budget=budget,
        backends=backends,
    )
    try:
        sharper = parse_statement(reply.text)
    except ValueError:
        return None
    child = Record(
        id=_child_id(record, _round(store, record)),
        run_id=record.run_id,
        pair=record.pair,
        mode=record.mode,
        statement=sharper.statement,
        objects=sharper.objects,
        assumptions=sharper.assumptions,
        domain_of_validity=sharper.domain_of_validity,
        instance_family=sharper.instance_family,
        provenance=Provenance(
            family=family,
            model=reply.model,
            prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
            parents=[record.id],
        ),
    )
    try:
        store.create(child)
    except RecordExists:
        pass
    return child.id
