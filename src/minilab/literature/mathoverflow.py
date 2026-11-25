"""MathOverflow search through the Stack Exchange API, with a date cap (plan doc-5wh, Task 10).

The search asks for questions created up to the cap (`todate`). Then the
answers to those questions are fetched, because answers are where known
results are usually cited.

The cap is applied to every post, question or answer:

- a post created after the cap is dropped;
- a post edited after the cap loses its body, because the API gives only the
  latest text. A question edited after the cap loses its title too, and so
  do its answers, which carry that title. `withheld` says why.

Bodies come as HTML and are reduced to plain text. A reply with neither
`items` nor `error_id`, or a timestamp that cannot be a date, raises
`SourceUnavailable`: it is never read as an empty result.
"""

from datetime import UTC, date, datetime
from html.parser import HTMLParser
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from minilab.literature.arxiv import cap_end
from minilab.literature.http import LiteratureHttp, SourceUnavailable, default_client

API = "https://api.stackexchange.com/2.3"
HOST = "api.stackexchange.com"
SITE = "mathoverflow"
DEFAULT_PAGE_SIZE = 10


class Post(BaseModel):
    """One MathOverflow question or answer as the filter may use it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["question", "answer"]
    question_id: int
    answer_id: int | None
    title: str | None
    """The question's title; for an answer, the title of its question. None
    when that question was edited after the cap."""
    body: str | None
    """Plain text, or None when the post was edited after the cap."""
    withheld: str | None
    """What was withheld under the cap, and why."""
    created: datetime
    last_edited: datetime | None
    link: str


class _RawQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question_id: int
    title: str
    body: str = ""
    link: str
    creation_date: int
    last_edit_date: int | None = None


class _RawAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore")

    answer_id: int
    question_id: int
    body: str = ""
    creation_date: int
    last_edit_date: int | None = None


class _Page[T](BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: list[T] | None = None
    backoff: int | None = None
    error_id: int | None = None
    error_name: str | None = None
    error_message: str | None = None


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def html_text(html: str) -> str:
    """The text of an HTML fragment, with whitespace collapsed."""
    parser = _Text()
    parser.feed(html)
    parser.close()
    return " ".join(" ".join(parser.parts).split())


def _when(stamp: int) -> datetime:
    try:
        return datetime.fromtimestamp(stamp, tz=UTC)
    except (OverflowError, ValueError, OSError) as exc:
        raise SourceUnavailable(f"MathOverflow: {stamp} is not a usable timestamp") from exc


def _get[T](client: LiteratureHttp, path: str, params: dict[str, str], item: type[T]) -> list[T]:
    body = client.get(f"{API}{path}", {**params, "site": SITE, "filter": "withbody"})
    try:
        page = _Page[item].model_validate_json(body)
    except ValidationError as exc:
        raise SourceUnavailable(f"MathOverflow: the response cannot be read: {exc}") from exc
    if page.backoff is not None:
        client.defer(HOST, float(page.backoff))
    if page.error_id is not None:
        raise SourceUnavailable(
            f"MathOverflow: {page.error_name} ({page.error_id}): {page.error_message}"
        )
    if page.items is None:
        raise SourceUnavailable("MathOverflow: the response has neither items nor an error")
    return page.items


def _capped(created: int, edited: int | None, before: date | None) -> tuple[bool, str | None]:
    """Whether to keep a post, and, if it was edited after the cap, when."""
    created_at = _when(created)
    edited_at = None if edited is None else _when(edited)
    if before is None:
        return True, None
    end = cap_end(before)
    if created_at > end:
        return False, None
    if edited_at is not None and edited_at > end:
        return True, f"edited on {edited_at:%Y-%m-%d}, after the cap {before}"
    return True, None


def search_mathoverflow(
    query: str,
    before: date | None,
    *,
    client: LiteratureHttp | None = None,
    pagesize: int = DEFAULT_PAGE_SIZE,
) -> list[Post]:
    """Search MathOverflow questions for `query`, then read their answers, all capped at `before`.

    Return the questions in the API's order, then the answers. Raise
    SourceUnavailable if the API fails, reports an error, or sends a reply
    that cannot be read.
    """
    client = client or default_client()
    params = {"q": query, "pagesize": str(pagesize), "order": "desc", "sort": "relevance"}
    if before is not None:
        params["todate"] = str(int(cap_end(before).timestamp()))
    questions = _get(client, "/search/advanced", params, _RawQuestion)
    posts: list[Post] = []
    # Each kept question's title (None if withheld), why it was withheld, and its link.
    titles: dict[int, tuple[str | None, str | None, str]] = {}
    for q in questions:
        keep, edited = _capped(q.creation_date, q.last_edit_date, before)
        if not keep:
            continue
        title = None if edited else html_text(q.title)
        titles[q.question_id] = (title, edited, q.link)
        posts.append(
            Post(
                kind="question",
                question_id=q.question_id,
                answer_id=None,
                title=title,
                body=None if edited else html_text(q.body),
                withheld=f"title and body withheld: {edited}" if edited else None,
                created=_when(q.creation_date),
                last_edited=None if q.last_edit_date is None else _when(q.last_edit_date),
                link=q.link,
            )
        )
    if not titles:
        return posts
    ids = ";".join(str(i) for i in titles)
    answers = _get(client, f"/questions/{ids}/answers", {"pagesize": "100"}, _RawAnswer)
    for a in answers:
        keep, edited = _capped(a.creation_date, a.last_edit_date, before)
        if not keep or a.question_id not in titles:
            continue
        title, question_edited, link = titles[a.question_id]
        reasons: list[str] = []
        if question_edited:
            reasons.append(f"title withheld: the question was {question_edited}")
        if edited:
            reasons.append(f"body withheld: {edited}")
        posts.append(
            Post(
                kind="answer",
                question_id=a.question_id,
                answer_id=a.answer_id,
                title=title,
                body=None if edited else html_text(a.body),
                withheld="; ".join(reasons) or None,
                created=_when(a.creation_date),
                last_edited=None if a.last_edit_date is None else _when(a.last_edit_date),
                link=f"{link}#{a.answer_id}",
            )
        )
    return posts
