"""Dual encodings and their agreement (build plan doc-5wh, Task 7; minilab-e47de8).

Code written by a model sits inside every verdict, so two model families each
write the encoding from the record alone, and both are run on the same samples
(design doc-gmj, "Encoding"):

- `encode_twice` asks two families, never Gemini, chosen by `pick_checker`'s
  rules. Both get the same prompt, built from the record only. The second
  encoder may be the author's own family: the design asks for two different
  families, not two non-authors, and only Claude and Codex encode (user
  decision, 25 Nov 2026).
- `agree` runs both encodings in the sandbox on `n` shared samples, half drawn
  by each sampler. It first checks each encoding on its own (it runs, its
  values have the right types, its certificate method matches its margin, its
  sampler varies), then compares the two: parameter names, `in_domain`, the
  assumption predicates, and, where both say the point is in the domain, the
  margins.
- `run_encoding_stage` does both and writes the encoding stage. An encoding
  that is broken is asked for once more from the same family. Agreement leaves
  the status as it is. Disagreement asks the generator family for a sharper
  statement, at most `MAX_SHARPEN_ATTEMPTS` times; the stage is written as
  `ambiguous` with the sharper statement in its data, and `make_child` stores
  that statement as a child record whose provenance names the ambiguous one. A
  record whose chain already holds `MAX_ROUNDS` ambiguous ancestors, or whose
  generator gives no readable statement, is `abandoned` (user decisions,
  25 Nov 2026).

Nothing here gives a verdict: this stage only lets a record on to the
falsifier, or stops it. A sandbox that cannot run at all raises
`SandboxUnavailable` and changes no record.
"""

import hashlib
import random
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from fractions import Fraction
from typing import Literal, cast

import flint
import mpmath
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from minilab.certify.result import working_digits
from minilab.encoding.api import Instance
from minilab.encoding.prompts import (
    Statement,
    encoding_prompt,
    extract_source,
    parse_statement,
    retry_prompt,
    sharpen_prompt,
)
from minilab.encoding.sandbox import (
    CAPABILITIES,
    DEFAULT_TIMEOUT_S,
    Call,
    CallResult,
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

# How many sharpen calls one disagreement may make before the record is
# abandoned for want of a readable statement.
MAX_SHARPEN_ATTEMPTS = 2

# The working precision for drawing samples, in decimal digits. Each sample
# names its own precision for evaluation.
SAMPLE_DIGITS = 50

# The most calls in one sandbox run. Each run gets the whole time limit, so an
# encoding has about `timeout_s / BATCH_CALLS` seconds per call.
BATCH_CALLS = 60

# An encoding that names itself as an encoding and does nothing else. If it
# fails too, the sandbox is broken, not the encoding.
_PROBE = "class Probe:\n    pass\n\n\nENCODING = Probe\n"

Label = Literal["a", "b"]


class SandboxUnavailable(Exception):
    """The sandbox cannot run even an empty encoding, so no encoding can be judged."""


class Agreement(BaseModel):
    """Whether two encodings agree, and if not, the first reason and whose fault it was."""

    model_config = ConfigDict(frozen=True)

    agreed: bool
    compared: int = Field(ge=0)
    """Samples compared before the answer: all `n` when they agree."""
    reason: str | None
    broken: Label | None = None
    """Set when one encoding failed on its own (it crashed, gave the wrong
    types, its certificate method contradicts its margin, or its sampler does
    not vary), not when the two merely disagree."""


class StageResult(BaseModel):
    """What the encoding stage did to one record."""

    model_config = ConfigDict(frozen=True)

    agreement: Agreement
    status: Status
    sources: dict[Family, str]
    """Each final encoding's source, by the family that wrote it."""
    child: str | None
    """The sharper record made after a disagreement, if any."""


class _Broken(Exception):
    """One encoding failed on its own."""


def _no(reason: str, compared: int = 0, broken: Label | None = None) -> Agreement:
    return Agreement(agreed=False, compared=compared, reason=reason, broken=broken)


def _check_sandbox() -> None:
    """Raise SandboxUnavailable unless the sandbox can run an empty encoding."""
    try:
        output = run_encoding(_PROBE, [], DEFAULT_TIMEOUT_S)
    except OSError as exc:
        raise SandboxUnavailable(f"the sandbox cannot start: {exc}") from exc
    if output.failure is not None:
        raise SandboxUnavailable(f"the sandbox cannot run an empty encoding: {output.failure}")


def _run(source: str, calls: Sequence[Call], timeout_s: float) -> EncodingOutput:
    """Make `calls` in runs of at most `BATCH_CALLS`, each a fresh sandbox.

    A whole-run failure is the encoding's fault only if the sandbox still runs
    an empty encoding; otherwise raise SandboxUnavailable.
    """
    results: list[CallResult] = []
    capabilities: list[str] = []
    chunks = [calls[i : i + BATCH_CALLS] for i in range(0, len(calls), BATCH_CALLS)] or [[]]
    for chunk in chunks:
        try:
            output = run_encoding(source, chunk, timeout_s)
        except OSError as exc:
            raise SandboxUnavailable(f"the sandbox cannot start: {exc}") from exc
        if output.failure is not None:
            _check_sandbox()
            return output
        results.extend(output.results)
        capabilities = output.capabilities
    return EncodingOutput(results=results, capabilities=capabilities)


def _rebuild(result: CallResult, what: str) -> object:
    try:
        return result.rebuild()
    except SandboxFailure as exc:
        raise _Broken(f"{what}: {exc}") from exc


def _sample(
    source: str, seeds: Sequence[int], timeout_s: float
) -> tuple[list[Instance], list[str]]:
    """Draw one sample per seed; return the samples and the encoding's capabilities."""
    calls = [Call(method="sample", seed=s, digits=SAMPLE_DIGITS) for s in seeds]
    output = _run(source, calls, timeout_s)
    if output.failure is not None:
        raise _Broken(output.failure)
    capabilities = [c for c in CAPABILITIES if c in output.capabilities]
    if not capabilities:
        raise _Broken(f"it has none of {', '.join(CAPABILITIES)}")
    samples: list[Instance] = []
    with working_digits(SAMPLE_DIGITS):
        for result in output.results:
            value = _rebuild(result, "its sampler raised")
            if type(value) is not Instance:
                raise _Broken("its sampler did not give an Instance")
            samples.append(value)
    if len(samples) > 1 and len({x.model_dump_json() for x in samples}) < 2:
        raise _Broken(f"its sampler gave the same point {len(samples)} times")
    if len({frozenset(x.values) for x in samples}) > 1:
        raise _Broken("its samples use different parameter names from each other")
    return samples, capabilities


def _mpf_ball(value: mpmath.mpf) -> flint.arb:
    """The mpf as an exact arb (at the current precision, which holds it).

    `man_exp` drops the sign, so read the raw (sign, man, exp, bits) tuple.
    """
    sign, man, exp, _ = cast(tuple[int, int, int, int], value._mpf_)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
    ball = flint.arb(man) * flint.arb(2) ** exp
    return -ball if sign else ball


def _widen(ball: flint.arb, rtol: float) -> flint.arb:
    """The ball grown by `rtol` of its size, on each side."""
    r = flint.arb(rtol)
    return ball.union(ball * (1 - r)).union(ball * (1 + r))


def _claim_ball(method: str, value: object) -> flint.arb:
    """What a certificate method says about the margin, as an arb. Raise _Broken if malformed."""
    if method == "exact_margin":
        if type(value) is not Fraction:
            raise _Broken("its exact_margin is not a Fraction")
        return flint.arb(flint.fmpq(value.numerator, value.denominator))
    if method == "margin_ball":
        if type(value) is not flint.arb or not value.is_finite():
            raise _Broken("its margin_ball is not a finite arb")
        return value
    if not (isinstance(value, tuple) and len(cast(tuple[object, ...], value)) == 2):
        raise _Broken("its primal_dual is not a pair of arbs")
    first, second = cast(tuple[object, object], value)
    if type(first) is not flint.arb or type(second) is not flint.arb:
        raise _Broken("its primal_dual is not a pair of arbs")
    if not (first.is_finite() and second.is_finite()):
        raise _Broken("its primal_dual bounds are not finite")
    return first.union(second)


def _same_sign(a: mpmath.mpf, b: mpmath.mpf) -> bool:
    return (a > 0) == (b > 0) and (a < 0) == (b < 0)


def _margins_close(a: mpmath.mpf, b: mpmath.mpf, rtol: float) -> bool:
    """Same sign, and within `rtol` of each other relative to the larger."""
    return _same_sign(a, b) and bool(abs(a - b) <= rtol * max(abs(a), abs(b)))


def _predicates(
    output: EncodingOutput, i: int, assumptions: int, where: str
) -> tuple[bool, list[bool]]:
    domain = _rebuild(output.results[2 * i], f"in_domain raised {where}")
    held = _rebuild(output.results[2 * i + 1], f"assumptions raised {where}")
    if type(domain) is not bool:
        raise _Broken(f"its in_domain is not a bool {where}")
    if not isinstance(held, list):
        raise _Broken(f"its assumptions are not a list {where}")
    flags: list[bool] = []
    for flag in cast(list[object], held):
        if type(flag) is not bool:
            raise _Broken(f"its assumptions are not all bools {where}")
        flags.append(flag)
    if len(flags) != assumptions:
        raise _Broken(f"it gives {len(flags)} assumptions; the record has {assumptions}")
    return domain, flags


def _margin(
    output: EncodingOutput, k: int, capabilities: Sequence[str], rtol: float, where: str
) -> mpmath.mpf:
    """The margin from call group `k`, checked against the encoding's certificate methods."""
    step = 1 + len(capabilities)
    margin = _rebuild(output.results[step * k], f"margin raised {where}")
    if type(margin) is not mpmath.mpf:
        raise _Broken(f"its margin is not a finite mpf {where}")
    widened = _widen(_mpf_ball(margin), rtol)
    for j, method in enumerate(capabilities, start=1):
        claim = _claim_ball(
            method, _rebuild(output.results[step * k + j], f"{method} raised {where}")
        )
        if not claim.overlaps(widened):
            raise _Broken(
                f"its {method} {claim.str(12)} contradicts its margin "
                f"{mpmath.nstr(margin, 12)} {where}"
            )
    return margin


def agree(
    a: str,
    b: str,
    n: int = 200,
    rtol: float = 1e-9,
    *,
    assumptions: int,
    seed: int = 0,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> Agreement:
    """Say whether encodings `a` and `b` (sources) agree on `n` shared samples.

    `a` draws the first half of the samples and `b` the rest, from seeds made
    from `seed`. Each encoding must, on its own: run; have at least one of
    `exact_margin`, `margin_ball`, `primal_dual`, each consistent with its
    margin to within `rtol`; give a bool `in_domain` and a list of
    `assumptions` bools of the record's length; and draw at least two distinct
    points. A failure of one of these sets `broken`. The two then agree when
    every sample from both uses the same parameter names; both give the same
    `in_domain` and `assumptions` on every sample; on samples both put in the
    domain, the margins have the same sign and are within `rtol` of each other
    relative to the larger; and at least a quarter of the samples are in the
    domain.

    Never raises for anything an encoding does. Raise SandboxUnavailable if
    the sandbox cannot run an empty encoding, and ValueError for `n` below 2.
    """
    if n < 2:
        raise ValueError(f"agree needs at least 2 samples, one from each sampler; got {n}")
    rng = random.Random(seed)
    seeds = [rng.getrandbits(63) for _ in range(n)]
    half = (n + 1) // 2
    sources: dict[Label, str] = {"a": a, "b": b}
    mine: dict[Label, list[int]] = {"a": seeds[:half], "b": seeds[half:]}
    points: list[Instance] = []
    names: dict[Label, frozenset[str]] = {}
    caps: dict[Label, list[str]] = {}
    for label in ("a", "b"):
        try:
            samples, caps[label] = _sample(sources[label], mine[label], timeout_s)
        except _Broken as exc:
            return _no(f"encoding {label} failed: {exc}", broken=label)
        names[label] = frozenset(samples[0].values)
        points.extend(samples)
    if names["a"] != names["b"]:
        return _no(
            f"the parameter names differ: a uses {sorted(names['a'])}, b uses {sorted(names['b'])}"
        )
    # Pass 1: the predicates on every point.
    predicate_calls = [
        Call(method=m, point=x, digits=x.precision_digits)
        for x in points
        for m in ("in_domain", "assumptions")
    ]
    seen: dict[Label, list[tuple[bool, list[bool]]]] = {}
    for label in ("a", "b"):
        output = _run(sources[label], predicate_calls, timeout_s)
        if output.failure is not None:
            return _no(f"encoding {label} failed: {output.failure}", broken=label)
        try:
            seen[label] = [
                _predicates(output, i, assumptions, f"at sample {i} {dict(x.values)}")
                for i, x in enumerate(points)
            ]
        except _Broken as exc:
            return _no(f"encoding {label} failed: {exc}", broken=label)
    inside: list[int] = []
    for i, x in enumerate(points):
        where = f"at sample {i} {dict(x.values)}"
        (dom_a, held_a), (dom_b, held_b) = seen["a"][i], seen["b"][i]
        if dom_a != dom_b:
            return _no(f"in_domain differs {where}: a says {dom_a}, b says {dom_b}", i)
        if held_a != held_b:
            return _no(f"the assumptions differ {where}: a says {held_a}, b says {held_b}", i)
        if dom_a:
            inside.append(i)
    if len(inside) < max(1, n // 4):
        return _no(f"only {len(inside)} of {n} samples are in the domain", n)
    # Pass 2: margins and certificate methods, only where both say in domain.
    margins: dict[Label, list[mpmath.mpf]] = {}
    for label in ("a", "b"):
        methods = cast(list[Method], ["margin", *caps[label]])
        calls = [
            Call(method=m, point=points[i], digits=points[i].precision_digits)
            for i in inside
            for m in methods
        ]
        output = _run(sources[label], calls, timeout_s)
        if output.failure is not None:
            return _no(f"encoding {label} failed: {output.failure}", broken=label)
        margins[label] = []
        for k, i in enumerate(inside):
            x = points[i]
            with working_digits(x.precision_digits):
                try:
                    margins[label].append(
                        _margin(output, k, caps[label], rtol, f"at sample {i} {dict(x.values)}")
                    )
                except _Broken as exc:
                    return _no(f"encoding {label} failed: {exc}", i, broken=label)
    for k, i in enumerate(inside):
        x = points[i]
        m_a, m_b = margins["a"][k], margins["b"][k]
        with working_digits(x.precision_digits):
            close = _margins_close(m_a, m_b, rtol)
            shown = (mpmath.nstr(m_a, 12), mpmath.nstr(m_b, 12))
        if not close:
            return _no(
                f"the margin differs at sample {i} {dict(x.values)}: "
                f"a gives {shown[0]}, b gives {shown[1]}",
                i,
            )
    return Agreement(agreed=True, compared=n, reason=None)


def encoders(author: Family, available: Sequence[Family]) -> tuple[Family, Family]:
    """The two families that encode work by `author`: never Gemini, never one twice.

    The first is `pick_checker(author)`; the second is `pick_checker` of the
    first, which may be the author.
    """
    first = pick_checker(author, available)
    second = pick_checker(first, available)
    return first, second


def _encode(
    record: Record,
    family: Family,
    prompt: str,
    *,
    ledger: Ledger,
    budget: Budget,
    backends: Mapping[Family, Backend] | None,
) -> Sandboxed:
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
    return Sandboxed(source=extract_source(reply.text), family=family)


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
    `BudgetExceeded`, ...) are raised as `call` raises them, and nothing is
    written; a call that ran is charged, so a retry pays for it again.
    """
    prompt = encoding_prompt(record)
    first, second = encoders(record.provenance.family, ledger.manifest.families)
    return (
        _encode(record, first, prompt, ledger=ledger, budget=budget, backends=backends),
        _encode(record, second, prompt, ledger=ledger, budget=budget, backends=backends),
    )


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


def _ask_sharper(
    record: Record,
    reason: str,
    *,
    ledger: Ledger,
    budget: Budget,
    backends: Mapping[Family, Backend] | None,
) -> tuple[Statement, str, str] | None:
    """Ask the generator family for a sharper statement, up to MAX_SHARPEN_ATTEMPTS times.

    Return the statement, the model's name, and the prompt's hash, or None if
    no reply could be read.
    """
    prompt = sharpen_prompt(record, reason)
    for _ in range(MAX_SHARPEN_ATTEMPTS):
        reply = call(
            record.provenance.family,
            Role.generate,
            prompt,
            record_id=record.id,
            pair=record.pair,
            ledger=ledger,
            budget=budget,
            backends=backends,
        )
        try:
            statement = parse_statement(reply.text)
        except ValueError:
            continue
        return statement, reply.model, hashlib.sha256(prompt.encode()).hexdigest()
    return None


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

    A broken encoding (see `Agreement.broken`) is asked for again, once per
    family, with its own failure shown to it and nothing of the other's. Each
    final encoding is saved under the run at `encodings/<record_id>.<family>.py`
    and named in the stage's evidence. Agreement leaves the status unchanged.
    Disagreement gives `abandoned` once the record has `MAX_ROUNDS` ambiguous
    ancestors; otherwise the generator family is asked for a sharper
    statement, and the record becomes `ambiguous` with a child record made by
    `make_child`, or `abandoned` if no reply could be read.

    Raise ValueError for a record that is not `proposed` or already has an
    encoding stage. A model failure or SandboxUnavailable raises before the
    stage is written, so the record is unchanged and the stage can run again
    (paying again for the calls already made).
    """
    store = ledger.store
    record = store.load(record_id)
    if record.status is not Status.proposed:
        raise ValueError(f"{record_id} is {record.status}; only a proposed record is encoded")
    if Stage.encoding in record.stages:
        raise ValueError(f"{record_id} already has an encoding stage")
    pair = list(encode_twice(record, ledger=ledger, budget=budget, backends=backends))
    reasked: list[str] = []

    def compare() -> Agreement:
        return agree(
            pair[0].source,
            pair[1].source,
            n,
            rtol,
            assumptions=len(record.assumptions),
            timeout_s=timeout_s,
        )

    agreement = compare()
    while agreement.broken is not None:
        index = 0 if agreement.broken == "a" else 1
        family = pair[index].family
        if family.value in reasked:
            break
        reasked.append(family.value)
        prompt = retry_prompt(record, agreement.reason or "it failed")
        pair[index] = _encode(
            record, family, prompt, ledger=ledger, budget=budget, backends=backends
        )
        agreement = compare()

    directory = store.root / "runs" / record.run_id / "encodings"
    directory.mkdir(parents=True, exist_ok=True)
    evidence: list[str] = []
    for enc in pair:
        name = f"{record.id}.{enc.family.value}.py"
        (directory / name).write_text(enc.source, encoding="utf-8")
        evidence.append(f"encodings/{name}")

    rounds = _round(store, record)
    data: dict[str, JsonValue] = {
        "families": [enc.family.value for enc in pair],
        "agreed": agreement.agreed,
        "compared": agreement.compared,
        "reason": agreement.reason,
        "broken": agreement.broken,
        "reasked": cast(list[JsonValue], reasked),
        "round": rounds,
    }
    notes = agreement.reason or ""
    if agreement.agreed:
        status = record.status
    elif rounds >= MAX_ROUNDS:
        status = Status.abandoned
        notes += f" Abandoned after {rounds} sharpening rounds."
    else:
        sharper = _ask_sharper(
            record, agreement.reason or "", ledger=ledger, budget=budget, backends=backends
        )
        if sharper is None:
            status = Status.abandoned
            notes += (
                " The generator gave no readable sharper statement"
                f" in {MAX_SHARPEN_ATTEMPTS} attempts."
            )
        else:
            status = Status.ambiguous
            statement, model, prompt_hash = sharper
            data["sharper"] = cast(dict[str, JsonValue], statement.model_dump(mode="json"))
            data["sharper_model"] = model
            data["sharper_prompt_hash"] = prompt_hash
    store.write_stage(
        record.id,
        Stage.encoding,
        StageOutput(
            written_at=datetime.now(UTC), data=data, evidence=evidence, notes=notes.strip()
        ),
        status=status,
    )
    child = make_child(store, record.id) if status is Status.ambiguous else None
    return StageResult(
        agreement=agreement,
        status=status,
        sources={enc.family: enc.source for enc in pair},
        child=child,
    )


def child_id(record: Record) -> str:
    """The id of the sharper record made from an ambiguous one.

    It depends on the parent's id only, so it is the same on every run.
    """
    local = record.id.split(".", 1)[1]
    digest = hashlib.sha256(record.id.encode()).hexdigest()[:6]
    suffix = f"-s{digest}"
    return f"{record.run_id}.{local[: 64 - len(suffix)]}{suffix}"


def make_child(store: RecordStore, record_id: str) -> str:
    """Store the sharper statement held in an ambiguous record's encoding stage as its child.

    The child is a new proposed record (or untestable, without an instance
    family) whose provenance names the ambiguous record as its parent. Makes
    no model call, so calling it again (for example on resume, if the process
    stopped between writing the stage and storing the child) is safe: it
    returns the child already stored. Raise ValueError for a record that is not
    ambiguous, and RecordExists if another record already holds the child's id.
    """
    record = store.load(record_id)
    output = record.stages.get(Stage.encoding)
    if record.status is not Status.ambiguous or output is None or "sharper" not in output.data:
        raise ValueError(f"{record_id} is {record.status} with no sharper statement")
    target = child_id(record)
    try:
        existing = store.load(target)
    except KeyError:
        existing = None
    if existing is not None:
        if existing.provenance.parents != [record.id]:
            raise RecordExists(f"{target} exists and is not the sharper form of {record.id}")
        return target
    statement = Statement.model_validate(output.data["sharper"])
    child = Record(
        id=target,
        run_id=record.run_id,
        pair=record.pair,
        mode=record.mode,
        statement=statement.statement,
        objects=statement.objects,
        assumptions=statement.assumptions,
        domain_of_validity=statement.domain_of_validity,
        instance_family=statement.instance_family,
        provenance=Provenance(
            family=record.provenance.family,
            model=str(output.data.get("sharper_model", "")),
            prompt_hash=str(output.data.get("sharper_prompt_hash", "")),
            parents=[record.id],
        ),
    )
    store.create(child)
    return target
