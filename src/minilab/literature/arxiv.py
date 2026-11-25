"""arXiv search through the export API, with a date cap (build plan doc-5wh, Task 10).

The cap is applied twice. The query asks only for papers submitted up to the
cap (`submittedDate`), and every result whose first version (`published` in
the feed) is after the cap is dropped, whatever the API returned.

A paper first posted before the cap may have later versions. The feed's
abstract belongs to the latest version, so for a paper updated after the cap
the abstract is left out, and the PDF to read is the first version's. Version
1 is the only version whose date the feed gives besides the latest.
"""

import re
from datetime import UTC, date, datetime, time
from xml.etree import ElementTree

from pydantic import BaseModel, ConfigDict, Field

from minilab.literature.http import LiteratureHttp, SourceUnavailable

API = "https://export.arxiv.org/api/query"
ABS = "https://arxiv.org/abs/"
PDF = "https://arxiv.org/pdf/"
DEFAULT_MAX_RESULTS = 10

_ATOM = "{http://www.w3.org/2005/Atom}"
_ID = re.compile(r"arxiv\.org/abs/(?P<id>.+?)v(?P<version>\d+)$")
# Characters with a meaning in arXiv's query syntax.
_SYNTAX = re.compile(r'["()\[\]:]')


class Paper(BaseModel):
    """One arXiv paper as the filter may use it.

    `summary` is None when the paper was revised after the cap: the feed's
    abstract may come from a version the cap excludes. `pdf_url` is the
    version to read: the latest, or version 1 when the latest is after the cap.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    arxiv_id: str
    version: int = Field(ge=1)
    """The latest version."""
    title: str
    summary: str | None
    published: datetime
    """When version 1 was posted."""
    updated: datetime
    """When the latest version was posted."""
    revised_after_cap: bool
    abs_url: str
    pdf_url: str


def arxiv_query(phrases: list[str]) -> str:
    """An arXiv search expression that needs every phrase somewhere in a paper."""
    cleaned = [" ".join(_SYNTAX.sub(" ", p).split()) for p in phrases]
    return " AND ".join(f'all:"{p}"' for p in cleaned if p)


def cap_end(before: date) -> datetime:
    """The last moment of the cap day, in UTC."""
    return datetime.combine(before, time(23, 59, 59), tzinfo=UTC)


def _text(entry: ElementTree.Element, tag: str) -> str:
    found = entry.find(_ATOM + tag)
    if found is None or found.text is None:
        raise SourceUnavailable(f"arXiv: an entry has no {tag}")
    return " ".join(found.text.split())


def _when(entry: ElementTree.Element, tag: str) -> datetime:
    try:
        return datetime.fromisoformat(_text(entry, tag))
    except ValueError as exc:
        raise SourceUnavailable(f"arXiv: an entry's {tag} is not a date") from exc


def _paper(entry: ElementTree.Element, before: date | None) -> Paper | None:
    match = _ID.search(_text(entry, "id"))
    if match is None:
        raise SourceUnavailable(f"arXiv: an entry has an id it cannot read: {_text(entry, 'id')}")
    arxiv_id, version = match.group("id"), int(match.group("version"))
    published, updated = _when(entry, "published"), _when(entry, "updated")
    if before is not None and published > cap_end(before):
        return None
    revised = before is not None and updated > cap_end(before)
    pdf_version = 1 if revised else version
    return Paper(
        arxiv_id=arxiv_id,
        version=version,
        title=_text(entry, "title"),
        summary=None if revised else _text(entry, "summary"),
        published=published,
        updated=updated,
        revised_after_cap=revised,
        abs_url=f"{ABS}{arxiv_id}v{pdf_version}",
        pdf_url=f"{PDF}{arxiv_id}v{pdf_version}",
    )


def search_arxiv(
    query: str,
    before: date | None,
    *,
    client: LiteratureHttp | None = None,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> list[Paper]:
    """Search arXiv with `query` (an arXiv search expression), dropping papers after `before`.

    Raise SourceUnavailable if arXiv fails or answers with something that is
    not a feed of results. An empty list means arXiv answered and found nothing.
    """
    expression = query
    if before is not None:
        expression = f"({query}) AND submittedDate:[190001010000 TO {before:%Y%m%d}2359]"
    body = (client or LiteratureHttp()).get(
        API,
        {
            "search_query": expression,
            "start": "0",
            "max_results": str(max_results),
            "sortBy": "relevance",
        },
    )
    try:
        feed = ElementTree.fromstring(body)
    except ElementTree.ParseError as exc:
        raise SourceUnavailable(f"arXiv: the response is not XML: {exc}") from exc
    if feed.tag != _ATOM + "feed":
        raise SourceUnavailable(f"arXiv: the response is not an Atom feed but {feed.tag}")
    papers: list[Paper] = []
    for entry in feed.findall(_ATOM + "entry"):
        if "/api/errors" in _text(entry, "id"):
            raise SourceUnavailable(f"arXiv: the API refused the query: {_text(entry, 'summary')}")
        paper = _paper(entry, before)
        if paper is not None:
            papers.append(paper)
    return papers
