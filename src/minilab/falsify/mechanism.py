"""Mechanisms for certified kills (design doc-gmj, "Falsifier"; build plan doc-5wh, Task 8).

For a certified kill, a checker family (never the generator's, never Gemini)
proposes a mechanism: a structural reason for the failure, given as a module
run in the sandbox. Its class (bound to `ENCODING`, as the sandbox requires)
has two methods:

- `predicate(x) -> bool`: the structural condition the mechanism names;
- `sample(rng) -> Instance`: an instance generator, giving instances where
  the predicate holds and the conjecture should fail.

`check_mechanism` draws `FRESH` instances from fresh seeds. An instance counts
only if it is new (not the kill's point, not a repeat), its predicate holds,
and the conjecture's failure there is certified by the same certifier as a
kill. The mechanism passes with at least `NEED` of them. A kill with a passed
mechanism goes to the human queue.
"""

import random
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

from minilab.certify import Certificate, ExactCertificate, IntervalCertificate
from minilab.encoding.api import EncodingFactory, Instance
from minilab.encoding.prompts import extract_source
from minilab.encoding.sandbox import ALLOWED_MODULES, DEFAULT_TIMEOUT_S
from minilab.falsify.search import (
    CertificateUnavailable,
    Certifier,
    build_certificate,
    sandbox_points,
    sandbox_samples,
)
from minilab.ledger import Ledger
from minilab.models.adapter import Budget, Role, call, pick_checker
from minilab.models.backends import Backend
from minilab.records import Family, Record

# Fresh instances drawn from the mechanism's generator.
FRESH = 10

# Instances on which the predicted failure must hold, certified.
NEED = 9


class MechanismResult(BaseModel):
    """Whether a proposed mechanism's prediction held, and on how many instances."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: bool
    held: int = Field(ge=0)
    """Fresh instances where the predicate held and the failure was certified."""
    tried: int = Field(ge=0)
    family: Family
    explanation: str
    """The model's own account of the mechanism, outside its code."""
    source: str
    reasons: list[str]
    """Why each instance that did not count failed."""


def _record_text(record: Record) -> str:
    assumptions = "\n".join(f"  {i}. {a}" for i, a in enumerate(record.assumptions, start=1))
    return (
        f"Statement: {record.statement}\n"
        f"Assumptions, in order:\n{assumptions or '  (none)'}\n"
        f"Domain of validity: {record.domain_of_validity}\n"
        f"Instance family and its parameters: {record.instance_family}\n"
    )


def _margin_text(kill: Certificate) -> str:
    if isinstance(kill, ExactCertificate):
        return f"exactly {kill.margin}"
    if isinstance(kill, IntervalCertificate):
        return f"in [{kill.enclosure[0]}, {kill.enclosure[1]}]"
    return f"about {kill.violation}"


def mechanism_prompt(record: Record, kill: Certificate, original_source: str | None) -> str:
    """The prompt that asks a checker family for a mechanism behind a certified kill."""
    allowed = ", ".join(sorted(ALLOWED_MODULES))
    code = (
        f"The conjecture's encoding:\n```python\n{original_source}\n```\n"
        if original_source
        else ""
    )
    return f"""This conjecture has a certified counterexample.

{_record_text(record)}
Counterexample, as `Instance.values`: {dict(kill.point.values)}
Margin there (negative means the conjecture fails): {_margin_text(kill)}

{code}Propose a mechanism: a structural reason why the conjecture fails. First explain it in
a few sentences. Then give it as a Python module that binds the name ENCODING, at the
top level, to a class with these methods. `Instance` comes from `minilab.encoding.api`.

- predicate(self, x: Instance) -> bool: true when x has the structure your mechanism
  names.
- sample(self, rng: random.Random) -> Instance: a new instance, drawn only with `rng`,
  with that structure, where your mechanism predicts the conjecture fails. Use the
  conjecture's parameter names, and exact decimal strings or Fractions as values.

The mechanism is tested on fresh instances from your generator: the predicate must hold
and the failure must be certified on at least {NEED} of {FRESH}.

Import only these modules: {allowed}. Do not change the working precision. Do not read
files, use the network, or print.

Put the module in one ```python block."""


def _explanation(text: str, source: str) -> str:
    """The reply without its code block."""
    fenced = text.find("```")
    return (text[:fenced] if fenced >= 0 else text.replace(source, "")).strip()


def _values_key(x: Instance) -> str:
    return Instance(values=x.values, precision_digits=1).model_dump_json()


def _digits(kill: Certificate) -> int:
    if isinstance(kill, ExactCertificate):
        return kill.point.precision_digits
    return kill.digits


def check_mechanism(
    record: Record,
    kill: Certificate,
    *,
    make: EncodingFactory,
    certify: Certifier,
    ledger: Ledger,
    budget: Budget,
    backends: Mapping[Family, Backend] | None = None,
    rng: random.Random,
    original_source: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> MechanismResult:
    """Ask a checker family for a mechanism behind `kill`, and test its prediction.

    Each fresh instance is certified as a violation of the conjecture `make`
    encodes, by `certify` (in the pipeline, `check_kill` with both encodings).
    Raise NoChecker if no family may check, and model failures as `call`
    raises them; nothing is written.
    """
    family = pick_checker(record.provenance.family, ledger.manifest.families)
    reply = call(
        family,
        Role.mechanism,
        mechanism_prompt(record, kill, original_source),
        record_id=record.id,
        pair=record.pair,
        ledger=ledger,
        budget=budget,
        backends=backends,
    )
    source = extract_source(reply.text)
    seeds = [rng.getrandbits(63) for _ in range(FRESH)]
    samples, _ = sandbox_samples(source, seeds, timeout_s)
    holds, errors = sandbox_points(source, "predicate", samples, timeout_s)
    seen = {_values_key(kill.point)}
    digits = _digits(kill)
    held = 0
    reasons: list[str] = []
    for i, (x, holds_here, error) in enumerate(zip(samples, holds, errors, strict=True)):
        if x is None:
            reasons.append(f"instance {i}: the generator failed")
            continue
        key = _values_key(x)
        if key in seen:
            reasons.append(f"instance {i}: not fresh: {dict(x.values)}")
            continue
        seen.add(key)
        if error is not None:
            reasons.append(f"instance {i}: the predicate failed: {error}")
            continue
        if holds_here is not True:
            reasons.append(f"instance {i}: the predicate does not hold at {dict(x.values)}")
            continue
        try:
            cert = build_certificate(make, x, digits)
        except CertificateUnavailable as exc:
            reasons.append(f"instance {i}: no certificate: {exc}")
            continue
        result = certify(cert)
        if not result.accepted:
            reasons.append(f"instance {i}: the failure is not certified: {result.reason}")
            continue
        held += 1
    return MechanismResult(
        passed=held >= NEED,
        held=held,
        tried=FRESH,
        family=family,
        explanation=_explanation(reply.text, source),
        source=source,
        reasons=reasons,
    )
