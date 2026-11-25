"""Mathlib search: Loogle, then `exact?` (build plan doc-5wh, Task 10).

Loogle searches mathlib by name, constant, and type pattern over its JSON API.
A query Loogle cannot parse raises `QueryRejected`: the search did not run,
but the service is up.

`exact?` runs only when a Lean statement exists. The statement's one `sorry`
becomes `exact?`, and the file goes through `check_lean`. A success gives the
term Lean suggests, and Lean has checked it.

Neither source can be date-capped. Loogle searches the latest mathlib, and
the repo's mathlib pin is later than the benchmark's cap. So under a date cap
the filter treats mathlib hits as notes only (decision on bead minilab-4e42a7).
"""

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from minilab.lean.runner import LeanUnavailable, check_lean
from minilab.literature.http import LiteratureHttp, SourceUnavailable

API = "https://loogle.lean-lang.org/json"
EXACT_TIMEOUT_S = 120.0
DEFAULT_MAX_HITS = 10

_BY_SORRY = re.compile(r"\bby\s+sorry\b")
_SORRY = re.compile(r"\bsorry\b")
_IMPORT = re.compile(r"^import\s", re.MULTILINE)
_TRY_THIS = "Try this:"
_LABEL = re.compile(r"^\[[a-z]+\]\s*")


class QueryRejected(Exception):
    """The source is up but refused the query (for Loogle, a parse error)."""


class Hit(BaseModel):
    """One mathlib declaration found for a query."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    type: str
    module: str
    doc: str | None
    via: Literal["loogle", "exact?"]


class _LoogleHit(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    type: str
    module: str
    doc: str | None = None


class _LoogleReply(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hits: list[_LoogleHit] = Field(default_factory=lambda: [])
    error: str | None = None


def search_loogle(
    query: str, *, client: LiteratureHttp | None = None, max_hits: int = DEFAULT_MAX_HITS
) -> list[Hit]:
    """Search mathlib with Loogle. Raise QueryRejected or SourceUnavailable."""
    body = (client or LiteratureHttp()).get(API, {"q": query})
    try:
        reply = _LoogleReply.model_validate_json(body)
    except ValidationError as exc:
        raise SourceUnavailable(f"Loogle: the response cannot be read: {exc}") from exc
    if reply.error is not None:
        raise QueryRejected(f"Loogle: {reply.error}")
    return [
        Hit(name=h.name, type=" ".join(h.type.split()), module=h.module, doc=h.doc, via="loogle")
        for h in reply.hits[:max_hits]
    ]


def exact_source(statement_lean: str) -> str:
    """The statement with its one `sorry` replaced by `exact?`.

    Raise ValueError unless the source holds exactly one `sorry`.
    """
    count = len(_SORRY.findall(statement_lean))
    if count != 1:
        raise ValueError(f"a Lean statement for exact? needs exactly one sorry, not {count}")
    if _BY_SORRY.search(statement_lean):
        return _BY_SORRY.sub("by exact?", statement_lean)
    return _SORRY.sub("(by exact?)", statement_lean)


def search_exact(
    statement_lean: str, *, project: Path | None = None, timeout_s: float = EXACT_TIMEOUT_S
) -> list[Hit]:
    """Run `exact?` on the statement. Return the suggested term as a hit, or no hits.

    A hit needs Lean to accept the file with no `sorry`. Raise SourceUnavailable
    if Lean cannot run or times out: then the search did not happen.
    """
    source = exact_source(statement_lean)
    if not _IMPORT.search(source):
        source = "import Mathlib\n" + source
    try:
        result = check_lean(source, timeout_s, project=project)
    except LeanUnavailable as exc:
        raise SourceUnavailable(f"exact?: {exc}") from exc
    if result.timed_out:
        raise SourceUnavailable(f"exact?: Lean timed out after {timeout_s} s")
    if not result.ok or result.has_sorry:
        return []
    hits: list[Hit] = []
    for message in result.messages:
        if message.severity == "information" and message.text.startswith(_TRY_THIS):
            term = _LABEL.sub("", message.text.removeprefix(_TRY_THIS).strip())
            hits.append(Hit(name=term, type="", module="", doc=None, via="exact?"))
    return hits


def search_mathlib(
    statement_lean: str | None,
    query: str,
    *,
    client: LiteratureHttp | None = None,
    project: Path | None = None,
) -> list[Hit]:
    """Loogle hits for `query`, then `exact?` hits for the statement if there is one."""
    hits = search_loogle(query, client=client) if query.strip() else []
    if statement_lean is not None:
        hits += search_exact(statement_lean, project=project)
    return hits
