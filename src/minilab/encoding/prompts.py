"""Prompts for the encoding stage, and readers for their replies (minilab-e47de8).

Two prompts:

- `encoding_prompt` asks a model to write a record's encoding. It shows the
  record alone, never another encoding, so two families write independently
  (design doc-gmj, "Encoding"). It names the parameters from the record's
  instance family: both encoders must use those names, and a mismatch makes
  the record ambiguous (user decision, 25 Nov 2026).
- `sharpen_prompt` asks the generator family for a sharper statement after the
  encodings disagreed. `parse_statement` reads its JSON reply. Task 15's
  generator may reuse or replace this reader.
"""

import json
import re

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from minilab.encoding.sandbox import ALLOWED_MODULES
from minilab.records import Record

_FENCE = re.compile(r"```[ \t]*(?P<lang>[A-Za-z0-9_+-]*)[ \t]*\n(?P<body>.*?)\n?```", re.DOTALL)


def _record_text(record: Record) -> str:
    assumptions = "\n".join(f"  {i}. {a}" for i, a in enumerate(record.assumptions, start=1))
    return (
        f"Statement: {record.statement}\n"
        f"Objects: {', '.join(record.objects)}\n"
        f"Assumptions, in order:\n{assumptions or '  (none)'}\n"
        f"Domain of validity: {record.domain_of_validity}\n"
        f"Instance family and its parameters: {record.instance_family}\n"
    )


def encoding_prompt(record: Record) -> str:
    """The prompt that asks for a record's encoding. It shows the record only."""
    allowed = ", ".join(sorted(ALLOWED_MODULES))
    count = len(record.assumptions)
    return f"""Write a Python module that encodes this conjecture so that code can test it.

{_record_text(record)}
The module must bind the name ENCODING, at the top level, to a class. An instance of
the class needs these methods. `Instance` comes from `minilab.encoding.api`; its
`values` maps each parameter name to a value (an exact decimal string, a
Fraction, or a list of decimal strings), and `precision_digits` is the working
precision in decimal digits.

- margin(self, x: Instance) -> mpmath.mpf: the conjecture's margin at x. Write
  the claim as an inequality LHS <= RHS (or LHS >= RHS) exactly as the
  statement gives it, and return RHS - LHS (or LHS - RHS), unscaled. A margin
  below zero means the conjecture fails at x.
- assumptions(self, x: Instance) -> list[bool]: one bool per assumption, in the
  order above ({count} in all), true when x satisfies it.
- in_domain(self, x: Instance) -> bool: true when x is in the domain of validity.
- sample(self, rng: random.Random) -> Instance: a random point of the instance
  family, drawn only with `rng`. Most samples should be in the domain.
- near_degenerate(self, x: Instance) -> bool: true near a degenerate case,
  where cancellation can spoil the margin.

Use exactly the parameter names that the instance family above gives, as the
keys of `Instance.values`. Do not rename them or add others.

The class must also have at least one of these, computed in exact or ball
arithmetic, never in floats:

- exact_margin(self, x: Instance) -> fractions.Fraction: the margin as an exact
  rational, where it is rational at rational points;
- margin_ball(self, x: Instance) -> flint.arb: an arb ball at python-flint's
  current precision that contains the true margin;
- primal_dual(self, x: Instance) -> tuple[flint.arb, flint.arb]: for a margin
  defined by an optimisation, one ball bounding it from above (a primal-feasible
  point) and one from below (a dual-feasible point), each started cold.

Import only these modules: {allowed}. `minilab.encoding.api` gives
`Instance`, `to_fraction`, `to_mpf`, and `to_arb` to read a value. Do not
change the working precision. Do not read files, use the network, or print.

Reply with the module only, in one ```python block."""


def extract_source(text: str) -> str:
    """The encoding source in a reply: its first ```python block, or the whole reply."""
    for found in _FENCE.finditer(text):
        if found.group("lang").lower() in ("python", "py", ""):
            return found.group("body")
    return text


class Statement(BaseModel):
    """A sharper statement of a conjecture, as the generator family wrote it."""

    model_config = ConfigDict(extra="ignore", frozen=True, str_strip_whitespace=True)

    statement: str = Field(min_length=1)
    objects: list[str]
    assumptions: list[str]
    domain_of_validity: str = Field(min_length=1)
    instance_family: str | None


def sharpen_prompt(record: Record, disagreement: str) -> str:
    """The prompt that asks the generator family to make a statement precise."""
    return f"""Two independent encodings of this conjecture as code disagreed, so its
statement is ambiguous.

{_record_text(record)}
What disagreed: {disagreement}

Rewrite the conjecture so that it has one reading. Keep the same claim; make
precise what was left open: the exact quantities, the direction of the
inequality, the assumptions, and the domain. The instance family must name each
parameter and its range, and each parameter name must be a plain identifier.

Reply with one JSON object and nothing else, with these keys: "statement",
"objects" (a list of strings), "assumptions" (a list of strings),
"domain_of_validity", and "instance_family"."""


def parse_statement(text: str) -> Statement:
    """Read a sharpen reply. Raise ValueError if it is not one JSON object of the right shape."""
    candidates = [m.group("body") for m in _FENCE.finditer(text)] + [text]
    problems: list[str] = []
    for candidate in candidates:
        try:
            value: object = json.loads(candidate)
        except json.JSONDecodeError as exc:
            problems.append(f"not JSON: {exc}")
            continue
        if not isinstance(value, dict):
            problems.append("not a JSON object")
            continue
        try:
            return Statement.model_validate(value)
        except ValidationError as exc:
            problems.append(f"wrong shape: {str(exc)[:300]}")
    raise ValueError("; ".join(problems))
