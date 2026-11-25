"""Read a generator reply: a JSON list of conjectures (build plan doc-5wh, Task 15).

The generator asks for one JSON object, `{"conjectures": [...]}`, each item
with `statement`, `objects`, `assumptions`, `domain_of_validity`, and
`instance_family`. Models wrap JSON in prose or code fences, so the reader
tries each fenced block, then the whole reply, then the span from the first
bracket to the last. It also takes a bare list.

A reply with no readable conjecture raises `ParseFailure`. A reply with some
readable items returns them, with a message for each item it could not read:
the caller logs those and drops them.
"""

import json
import re

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError

_FENCE = re.compile(r"```[ \t]*[A-Za-z0-9_+-]*[ \t]*\n(?P<body>.*?)\n?```", re.DOTALL)
_ITEMS: TypeAdapter[list[JsonValue]] = TypeAdapter(list[JsonValue])


class ParseFailure(ValueError):
    """A generator reply holds no conjecture that can be read."""


class Conjecture(BaseModel):
    """One conjecture as a generator wrote it.

    A missing, null, or blank `instance_family` is read as None, which makes
    the record `untestable` (design doc-gmj, "Generator").
    """

    model_config = ConfigDict(extra="ignore", frozen=True, str_strip_whitespace=True)

    statement: str = Field(min_length=1)
    objects: list[str]
    assumptions: list[str]
    domain_of_validity: str = Field(min_length=1)
    instance_family: str | None = None


class _Envelope(BaseModel):
    model_config = ConfigDict(extra="ignore")

    conjectures: list[JsonValue]


def _candidates(text: str) -> list[str]:
    found = [m.group("body") for m in _FENCE.finditer(text)]
    found.append(text)
    for opening, closing in (("{", "}"), ("[", "]")):
        start, end = text.find(opening), text.rfind(closing)
        if 0 <= start < end:
            found.append(text[start : end + 1])
    return found


def _items(value: object) -> list[JsonValue]:
    """The list of conjecture items in a parsed reply. Raise ValueError if there is none."""
    if isinstance(value, list):
        return _ITEMS.validate_python(value)
    if isinstance(value, dict):
        if "conjectures" in value:
            try:
                return _Envelope.model_validate(value).conjectures
            except ValidationError as exc:
                raise ValueError("'conjectures' is not a list") from exc
        if "statement" in value:
            return [_ITEMS.validate_python([value])[0]]
    raise ValueError("no list of conjectures")


def parse_conjectures(text: str) -> tuple[list[Conjecture], list[str]]:
    """Read the conjectures in a generator reply.

    Return the conjectures that could be read, in order, and one message for
    each item that could not (it names the item). Raise `ParseFailure` if the
    reply holds no JSON list of conjectures or none of its items can be read.
    """
    problems: list[str] = []
    for candidate in _candidates(text):
        try:
            value: object = json.loads(candidate)
        except json.JSONDecodeError as exc:
            problems.append(f"not JSON: {exc}")
            continue
        try:
            items = _items(value)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        conjectures: list[Conjecture] = []
        failures: list[str] = []
        for index, item in enumerate(items):
            try:
                conjectures.append(Conjecture.model_validate(item))
            except ValidationError as exc:
                shown = json.dumps(item)[:500]
                failures.append(f"item {index} ({shown}): {str(exc)[:300]}")
        if conjectures:
            return conjectures, failures
        problems.append("no readable conjecture" + (f": {failures[0]}" if failures else ""))
    raise ParseFailure("; ".join(dict.fromkeys(problems)))
