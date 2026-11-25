"""The known-result filter (design doc-gmj, stage 4; build plan doc-5wh, Task 10).

`literature_check` looks for a record's result in mathlib (Loogle, then
`exact?` when a Lean statement exists), on arXiv, and on MathOverflow, with
the run's date cap. It makes two model calls, both by a checker family (never
the record's author, never Gemini):

1. the checker writes search queries from the record;
2. the checker reads the candidates found and says which, if any, state the
   conjecture's result, quoting the passage.

A match counts only if code can confirm it: the candidate is one the search
found and showed the checker, the quote passes `quote_found`, and the checker
says the passage states the same result or a stronger one (`same` or
`implies`). Then the status is `known`. A proof by `exact?` counts on its
own, because Lean checked it.

`quote_found` reads only a candidate's `text`, which holds only what the
source said: never its title, and never the labels the prompt adds. On arXiv
and MathOverflow a quote needs at least `MIN_QUOTE` characters after
normalising; for mathlib it may be the declaration's name or any
`MIN_MATHLIB_QUOTE` characters of its text. Normalising joins words split by
a hyphen at a line break and writes >=, <=, and != for their symbols. It
cannot tell such a split word from a real hyphen at a line end.

Under a date cap nothing posted or revised after the cap is shown: a late
paper or post is dropped, and the title and text of a late revision are
withheld. Mathlib matches are notes only, and mathlib's state takes no part
in the status, since neither Loogle (latest mathlib) nor the repo's mathlib
pin can be capped (decision doc-53t, bead minilab-4e42a7).

The status, in order of precedence:

1. `known`: a match counts.
2. `unchecked`: a source that counts could not be searched, or the
   checker's verdict could not be read. An outage is never "not found".
3. `unconfirmed`: the checker named a `same` or `implies` match from a
   source that counts, but its quote fails the check.
4. `not_found`: none of these.

On `unchecked`, `run_literature_stage` writes the log but not the stage, and
raises `LiteratureUnchecked`, so the record is unchanged and the stage runs
again on resume. On `unconfirmed` it writes the stage and leaves the
record's status, and the notes name the match that could not be confirmed.

Every query, its outcome, its hits, and the hits shown to the checker are
logged in the output, and the stage writes the whole output to
`runs/<run_id>/literature/`.
"""

import json
import re
import unicodedata
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from minilab.ledger import Ledger
from minilab.literature.arxiv import Paper, arxiv_query, search_arxiv
from minilab.literature.http import LiteratureHttp, SourceUnavailable, default_client
from minilab.literature.loogle import Hit, QueryRejected, search_exact, search_loogle
from minilab.literature.mathoverflow import Post, search_mathoverflow
from minilab.literature.pdf import pdf_text
from minilab.models.adapter import Budget, Role, call, pick_checker
from minilab.models.backends import Backend
from minilab.records import Family, Record, Stage, StageOutput, Status

SourceName = Literal["mathlib", "arxiv", "mathoverflow"]
SourceState = Literal["checked", "unchecked"]
Closeness = Literal["same", "implies", "special_case", "related", "unrelated"]
Outcome = Literal["ok", "unavailable", "rejected"]
LiteratureStatus = Literal["known", "unchecked", "unconfirmed", "not_found"]

SOURCES: tuple[SourceName, ...] = ("mathlib", "arxiv", "mathoverflow")
KNOWN_CLOSENESS: frozenset[str] = frozenset({"same", "implies"})

MAX_QUERIES = 3
MAX_ARXIV = 8
"""arXiv candidates, taken in turn from each query's hits."""
MAX_ANSWERS = 6
MAX_QUESTIONS = 6
"""MathOverflow candidates: answers first, then questions."""
MAX_HITS = 10
"""Mathlib candidates: `exact?` hits first, then Loogle hits."""
PDF_PAPERS = 3
"""The top arXiv candidates whose PDFs are read."""
HEAD_CHARS = 1500
EXCERPT_CHARS = 6000
POST_CHARS = 3000
MIN_QUOTE = 40
"""The shortest quote, after normalising, that confirms an arXiv or MathOverflow match."""
MIN_MATHLIB_QUOTE = 12
"""The shortest quote that confirms a mathlib match, unless it is the declaration's name."""


class LiteratureUnchecked(Exception):
    """The literature check could not finish, so the stage was not written."""

    def __init__(self, output: "LiteratureOutput") -> None:
        super().__init__(f"literature unchecked: {output.notes}")
        self.output = output


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class QueryLog(_Model):
    source: SourceName
    query: str
    outcome: Outcome
    detail: str = ""
    hits: list[str] = Field(default_factory=lambda: [])
    """Keys of the candidates this query found, after the date cap."""
    shown: list[str] = Field(default_factory=lambda: [])
    """Keys of this query's hits that the checker was shown. Some hits are cut
    by the candidate limits; `exact?` hits are never judged, as Lean checked them."""


class Section(_Model):
    """One part of a candidate's source text, under the label the prompt gives it."""

    label: str
    text: str


class Candidate(_Model):
    """A paper, post, or declaration found by the search, under its key."""

    key: str
    source: SourceName
    title: str | None
    """None when withheld under the date cap; `note` says why."""
    sections: list[Section]
    """The source's own text. The prompt shows each under its label."""
    link: str
    posted: datetime | None
    note: str = ""

    @property
    def text(self) -> str:
        """The source text the checker was shown, without the key, title, or labels."""
        return "\n\n".join(s.text for s in self.sections)


class Match(_Model):
    key: str
    closeness: Closeness
    quote: str
    note: str
    verified: bool
    """The candidate exists and the quote is in its text."""
    counts: bool
    """The match makes the record `known`."""


class Queries(_Model):
    arxiv: list[list[str]]
    mathoverflow: list[str]
    loogle: list[str]


class LiteratureOutput(_Model):
    status: LiteratureStatus
    before: date | None
    sources: dict[SourceName, SourceState]
    queries: list[QueryLog]
    candidates: list[Candidate]
    matches: list[Match]
    checker: Family | None
    judged: bool
    """The checker's verdict was read (or there was nothing to judge)."""
    notes: str


# --- reading model replies ---

_FENCE = re.compile(r"```[ \t]*[A-Za-z0-9_+-]*[ \t]*\n(?P<body>.*?)\n?```", re.DOTALL)


def _json_object(text: str) -> dict[str, JsonValue] | None:
    """The first JSON object in a reply: in a code fence, the whole reply, or a brace span."""
    found = [m.group("body") for m in _FENCE.finditer(text)]
    found.append(text)
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        found.append(text[start : end + 1])
    for candidate in found:
        try:
            value: object = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return cast(dict[str, JsonValue], value)
    return None


class _RawQueries(BaseModel):
    model_config = ConfigDict(extra="ignore")

    arxiv: list[list[str]] = Field(default_factory=lambda: [])
    mathoverflow: list[str] = Field(default_factory=lambda: [])
    loogle: list[str] = Field(default_factory=lambda: [])


class _RawMatch(BaseModel):
    model_config = ConfigDict(extra="ignore")

    key: str
    closeness: Closeness
    quote: str = ""
    note: str = ""


class _RawVerdict(BaseModel):
    model_config = ConfigDict(extra="ignore")

    matches: list[_RawMatch]


def fallback_queries(record: Record) -> Queries:
    """Queries made from the record's objects, for when the checker's cannot be read."""
    objects = [o for o in record.objects if o.strip()][:3] or [record.statement[:80]]
    words = sorted((w for o in objects for w in re.findall(r"[A-Za-z]{4,}", o)), key=len)
    return Queries(
        arxiv=[objects],
        mathoverflow=[" ".join(objects)],
        loogle=[f'"{words[-1]}"'] if words else [],
    )


def read_queries(text: str, record: Record) -> tuple[Queries, str]:
    """The checker's queries, with any source it left empty filled from the fallback."""
    fallback = fallback_queries(record)
    value = _json_object(text)
    try:
        raw = _RawQueries.model_validate(value) if value is not None else None
    except ValidationError:
        raw = None
    if raw is None:
        return fallback, "The query reply could not be read; queries were made from the objects."
    arxiv = [[p for p in q if p.strip()] for q in raw.arxiv]
    arxiv = [q for q in arxiv if q][:MAX_QUERIES]
    mathoverflow = [q for q in raw.mathoverflow if q.strip()][:MAX_QUERIES]
    loogle = [q for q in raw.loogle if q.strip()][:MAX_QUERIES]
    return (
        Queries(
            arxiv=arxiv or fallback.arxiv,
            mathoverflow=mathoverflow or fallback.mathoverflow,
            loogle=loogle or fallback.loogle,
        ),
        "",
    )


# --- prompts ---


def _record_text(record: Record) -> str:
    assumptions = "\n".join(f"  - {a}" for a in record.assumptions) or "  (none)"
    return (
        f"Statement: {record.statement}\n"
        f"Objects: {', '.join(record.objects)}\n"
        f"Assumptions:\n{assumptions}\n"
        f"Domain of validity: {record.domain_of_validity}\n"
    )


def queries_prompt(record: Record) -> str:
    return f"""We want to know whether this conjecture is already a known result.

{_record_text(record)}
Write search queries for three sources:

- "arxiv": up to {MAX_QUERIES} queries for the arXiv API. Each query is a list of short
  phrases that a paper stating the result would all contain, such as
  ["rate distortion", "Blahut-Arimoto", "linear"].
- "mathoverflow": up to {MAX_QUERIES} plain-text queries for MathOverflow.
- "loogle": up to {MAX_QUERIES} Loogle queries for mathlib, such as
  Real.log, "entropy"   or   ?a * ?b = ?b * ?a

Answer with one JSON object with the keys "arxiv", "mathoverflow", and "loogle", and
nothing else."""


def _render(candidate: Candidate) -> str:
    title = "(withheld under the date cap)" if candidate.title is None else candidate.title
    body = "\n".join(f"{s.label}: {s.text}" for s in candidate.sections)
    return (
        f"=== {candidate.key} ===\nTitle: {title}\n{body or '(text withheld under the date cap)'}"
    )


def judge_prompt(record: Record, candidates: list[Candidate]) -> str:
    shown = "\n\n".join(_render(c) for c in candidates)
    return f"""Decide whether any of the texts below already states this conjecture's result.

{_record_text(record)}
Texts found in mathlib, on arXiv, and on MathOverflow, each under its key:

{shown}

For each text that bears on the conjecture, give:

- "key": the text's key, exactly as above;
- "closeness": "same" (it states this result), "implies" (it states a stronger result that
  gives this one directly), "special_case" (it states only a special case), "related" (same
  objects, a different claim), or "unrelated";
- "quote": the passage that states the result, copied exactly from the text above, at
  least {MIN_QUOTE} characters, and never from a title or a label such as "Abstract:".
  For a mathlib declaration, its name will do;
- "note": one or two sentences on how the passage matches the conjecture.

Leave out unrelated texts. Answer with one JSON object, {{"matches": [...]}}, and nothing
else. If nothing bears on it, answer {{"matches": []}}."""


# --- candidates ---


def _terms(record: Record) -> set[str]:
    text = " ".join([record.statement, *record.objects]).lower()
    return set(re.findall(r"[a-z]{4,}", text))


def excerpt(text: str, terms: set[str]) -> str:
    """The start of a paper, then its paragraphs that mention the most terms, in order."""
    head = text[:HEAD_CHARS]
    paragraphs = [p for p in re.split(r"\n\s*\n", text[HEAD_CHARS:]) if p.strip()]
    scored = sorted(
        range(len(paragraphs)),
        key=lambda i: -sum(term in paragraphs[i].lower() for term in terms),
    )
    chosen: list[int] = []
    room = EXCERPT_CHARS - len(head)
    for i in scored:
        if len(paragraphs[i]) <= room:
            chosen.append(i)
            room -= len(paragraphs[i])
    return "\n\n".join([head, *(paragraphs[i] for i in sorted(chosen))])


def _paper_candidate(
    paper: Paper, rank: int, record: Record, client: LiteratureHttp, cache_dir: Path | None
) -> Candidate:
    sections: list[Section] = []
    notes: list[str] = []
    if paper.withheld is not None:
        notes.append(paper.withheld)
    if paper.summary is not None:
        sections.append(Section(label="Abstract", text=paper.summary))
    if rank < PDF_PAPERS:
        try:
            text = pdf_text(paper.pdf_url, client=client, cache_dir=cache_dir)
            sections.append(Section(label="From the PDF", text=excerpt(text, _terms(record))))
            notes.append(f"PDF: {paper.pdf_url}")
        except SourceUnavailable as exc:
            notes.append(f"PDF not read: {exc}")
    return Candidate(
        key=f"arxiv:{paper.arxiv_id}",
        source="arxiv",
        title=paper.title,
        sections=sections,
        link=paper.abs_url,
        posted=paper.published,
        note="; ".join(notes),
    )


def _post_key(post: Post) -> str:
    if post.answer_id is None:
        return f"mathoverflow:q{post.question_id}"
    return f"mathoverflow:a{post.answer_id}"


def _post_candidate(post: Post) -> Candidate:
    label = "Question" if post.kind == "question" else "Answer"
    sections = [] if post.body is None else [Section(label=label, text=post.body[:POST_CHARS])]
    return Candidate(
        key=_post_key(post),
        source="mathoverflow",
        title=post.title,
        sections=sections,
        link=post.link,
        posted=post.created,
        note=post.withheld or "",
    )


def _hit_candidate(hit: Hit) -> Candidate:
    if hit.via == "exact?":
        return Candidate(
            key=f"mathlib:exact?:{hit.name}",
            source="mathlib",
            title="exact? closed the Lean statement",
            sections=[Section(label="Term", text=hit.name)],
            link="",
            posted=None,
        )
    sections = [
        Section(label="Declaration", text=f"{hit.name} : {hit.type}"),
        Section(label="Module", text=hit.module),
    ]
    if hit.doc:
        sections.append(Section(label="Docstring", text=hit.doc))
    return Candidate(
        key=f"mathlib:{hit.name}",
        source="mathlib",
        title=hit.name,
        sections=sections,
        link="",
        posted=None,
    )


# --- matching ---


_HYPHEN_BREAK = re.compile(r"-[ \t]*\r?\n\s*")
_SYMBOLS = {"≥": ">=", "≤": "<=", "≠": "!="}


def normalise(text: str) -> str:
    """`text` for comparing quotes: line-end hyphens joined, symbols spelt, spaces collapsed."""
    text = _HYPHEN_BREAK.sub("", unicodedata.normalize("NFKC", text))
    for symbol, spelt in _SYMBOLS.items():
        text = text.replace(symbol, spelt)
    return " ".join(text.casefold().split())


def quote_found(quote: str, candidate: Candidate) -> bool:
    """Whether `quote` confirms a match on `candidate`, as the module describes.

    Only `candidate.text` is read: never the title, the key, or a label.
    """
    wanted = normalise(quote)
    if not wanted:
        return False
    shown = normalise(candidate.text)
    if candidate.source == "mathlib":
        if wanted == normalise(candidate.key.removeprefix("mathlib:")):
            return wanted in shown
        return len(wanted) >= MIN_MATHLIB_QUOTE and wanted in shown
    return len(wanted) >= MIN_QUOTE and wanted in shown


def _counts(source: SourceName, before: date | None) -> bool:
    return not (source == "mathlib" and before is not None)


# --- the check ---


class _Search:
    """Runs the searches and gathers the log and the candidates."""

    def __init__(
        self, record: Record, before: date | None, client: LiteratureHttp, cache_dir: Path | None
    ) -> None:
        self.record = record
        self.before = before
        self.client = client
        self.cache_dir = cache_dir
        self.log: list[QueryLog] = []
        self.papers: dict[str, Paper] = {}
        self.posts: dict[str, Post] = {}
        self.hits: dict[str, Hit] = {}

    def _note(
        self,
        source: SourceName,
        query: str,
        outcome: Outcome,
        detail: str = "",
        hits: list[str] | None = None,
    ) -> None:
        self.log.append(
            QueryLog(source=source, query=query, outcome=outcome, detail=detail, hits=hits or [])
        )

    def arxiv(self, phrases: list[str]) -> None:
        query = arxiv_query(phrases)
        if not query:
            self._note("arxiv", json.dumps(phrases), "rejected", "nothing is left after cleaning")
            return
        try:
            papers = search_arxiv(query, self.before, client=self.client)
        except SourceUnavailable as exc:
            self._note("arxiv", query, "unavailable", str(exc))
            return
        keys: list[str] = []
        for paper in papers:
            key = f"arxiv:{paper.arxiv_id}"
            keys.append(key)
            self.papers.setdefault(key, paper)
        self._note("arxiv", query, "ok", hits=keys)

    def mathoverflow(self, query: str) -> None:
        try:
            posts = search_mathoverflow(query, self.before, client=self.client)
        except SourceUnavailable as exc:
            self._note("mathoverflow", query, "unavailable", str(exc))
            return
        keys: list[str] = []
        for post in posts:
            key = _post_key(post)
            keys.append(key)
            self.posts.setdefault(key, post)
        self._note("mathoverflow", query, "ok", hits=keys)

    def loogle(self, query: str) -> None:
        try:
            hits = search_loogle(query, client=self.client)
        except QueryRejected as exc:
            self._note("mathlib", query, "rejected", str(exc))
            return
        except SourceUnavailable as exc:
            self._note("mathlib", query, "unavailable", str(exc))
            return
        self._add_hits(query, hits)

    def exact(self, statement_lean: str, project: Path | None) -> None:
        query = "exact? on the Lean statement"
        try:
            hits = search_exact(statement_lean, project=project)
        except ValueError as exc:
            self._note("mathlib", query, "rejected", str(exc))
            return
        except SourceUnavailable as exc:
            self._note("mathlib", query, "unavailable", str(exc))
            return
        self._add_hits(query, hits)

    def _add_hits(self, query: str, hits: list[Hit]) -> None:
        keys: list[str] = []
        for hit in hits:
            key = _hit_candidate(hit).key
            keys.append(key)
            self.hits.setdefault(key, hit)
        self._note("mathlib", query, "ok", hits=keys)

    def sources(self) -> dict[SourceName, SourceState]:
        return {
            s: "checked"
            if any(q.source == s and q.outcome == "ok" for q in self.log)
            else "unchecked"
            for s in SOURCES
        }

    def _arxiv_keys(self) -> list[str]:
        """Up to MAX_ARXIV papers, taken in turn from each query's hits."""
        lists = [q.hits for q in self.log if q.source == "arxiv" and q.outcome == "ok"]
        chosen: list[str] = []
        depth = max((len(hits) for hits in lists), default=0)
        for i in range(depth):
            for hits in lists:
                if len(chosen) < MAX_ARXIV and i < len(hits) and hits[i] not in chosen:
                    chosen.append(hits[i])
        return chosen

    def candidates(self) -> list[Candidate]:
        exact = [h for h in self.hits.values() if h.via == "exact?"]
        loogle = [h for h in self.hits.values() if h.via != "exact?"]
        found = [_hit_candidate(h) for h in exact + loogle[: max(MAX_HITS - len(exact), 0)]]
        for rank, key in enumerate(self._arxiv_keys()):
            paper = self.papers[key]
            found.append(_paper_candidate(paper, rank, self.record, self.client, self.cache_dir))
        answers = [p for p in self.posts.values() if p.kind == "answer"][:MAX_ANSWERS]
        questions = [p for p in self.posts.values() if p.kind == "question"][:MAX_QUESTIONS]
        found += [_post_candidate(p) for p in answers + questions]
        return found

    def mark_shown(self, shown: set[str]) -> None:
        """Record on each query which of its hits the checker was shown."""
        self.log = [
            q.model_copy(update={"shown": [k for k in q.hits if k in shown]}) for q in self.log
        ]


def literature_check(
    record: Record,
    before: date | None,
    *,
    ledger: Ledger,
    budget: Budget,
    backends: Mapping[Family, Backend] | None = None,
    client: LiteratureHttp | None = None,
    statement_lean: str | None = None,
    project: Path | None = None,
    cache_dir: Path | None = None,
) -> LiteratureOutput:
    """Look for `record`'s result in the literature up to `before`, as the module describes.

    Nothing is written. Raise NoChecker if no family may check the record,
    and model failures as `call` raises them.
    """
    client = client or default_client()
    checker = pick_checker(record.provenance.family, ledger.manifest.families)
    reply = call(
        checker,
        Role.literature,
        queries_prompt(record),
        record_id=record.id,
        pair=record.pair,
        ledger=ledger,
        budget=budget,
        backends=backends,
    )
    queries, query_note = read_queries(reply.text, record)
    notes: list[str] = [query_note] if query_note else []

    search = _Search(record, before, client, cache_dir)
    for query in queries.loogle:
        search.loogle(query)
    if statement_lean is not None:
        search.exact(statement_lean, project)
    for phrases in queries.arxiv:
        search.arxiv(phrases)
    for query in queries.mathoverflow:
        search.mathoverflow(query)
    sources = search.sources()
    candidates = search.candidates()
    by_key = {c.key: c for c in candidates}

    matches: list[Match] = []
    for c in candidates:
        if c.key.startswith("mathlib:exact?:"):
            matches.append(
                Match(
                    key=c.key,
                    closeness="same",
                    quote=c.text,
                    note="exact? closed the Lean statement, and Lean checked the proof.",
                    verified=True,
                    counts=_counts("mathlib", before),
                )
            )

    judged = True
    judged_candidates = [c for c in candidates if not c.key.startswith("mathlib:exact?:")]
    search.mark_shown({c.key for c in judged_candidates})
    if judged_candidates:
        verdict = call(
            checker,
            Role.literature,
            judge_prompt(record, judged_candidates),
            record_id=record.id,
            pair=record.pair,
            ledger=ledger,
            budget=budget,
            backends=backends,
        )
        value = _json_object(verdict.text)
        try:
            raw = _RawVerdict.model_validate(value) if value is not None else None
        except ValidationError:
            raw = None
        if raw is None:
            judged = False
            notes.append("The checker's verdict could not be read.")
        else:
            for m in raw.matches:
                candidate = by_key.get(m.key)
                if candidate is None or candidate.key.startswith("mathlib:exact?:"):
                    notes.append(f"The checker named {m.key!r}, which was not shown to it.")
                    continue
                verified = quote_found(m.quote, candidate)
                matches.append(
                    Match(
                        key=m.key,
                        closeness=m.closeness,
                        quote=m.quote,
                        note=m.note,
                        verified=verified,
                        counts=verified
                        and m.closeness in KNOWN_CLOSENESS
                        and _counts(candidate.source, before),
                    )
                )

    unconfirmed: list[Match] = []
    for m in matches:
        if m.closeness not in KNOWN_CLOSENESS:
            continue
        source = by_key[m.key].source
        if not _counts(source, before):
            notes.append(
                f"mathlib match {m.key} ({m.closeness}) is a note only: mathlib cannot be "
                f"capped at {before}."
            )
        elif not m.verified:
            unconfirmed.append(m)
            notes.append(
                f"Could not confirm the {m.closeness} match {m.key}: its quote is not in "
                "the text the checker was shown, or is too short."
            )

    # Under a date cap mathlib can never count, so its state takes no part.
    missing = [s for s in SOURCES if _counts(s, before) and sources[s] == "unchecked"]
    if any(m.counts for m in matches):
        status: LiteratureStatus = "known"
        notes.append("Known: " + ", ".join(m.key for m in matches if m.counts) + ".")
    elif missing or not judged:
        status = "unchecked"
        if missing:
            notes.append("Unchecked sources: " + ", ".join(missing) + ".")
    elif unconfirmed:
        status = "unconfirmed"
        notes.append("Unconfirmed: " + ", ".join(m.key for m in unconfirmed) + ".")
    else:
        status = "not_found"
    return LiteratureOutput(
        status=status,
        before=before,
        sources=sources,
        queries=search.log,
        candidates=candidates,
        matches=matches,
        checker=checker,
        judged=judged,
        notes=" ".join(notes),
    )


LITERATURE_STATUSES: frozenset[Status] = frozenset({Status.proposed, Status.survived})
"""The statuses a record may have when the literature stage runs."""


def run_literature_stage(
    record_id: str,
    *,
    ledger: Ledger,
    budget: Budget,
    backends: Mapping[Family, Backend] | None = None,
    client: LiteratureHttp | None = None,
    statement_lean: str | None = None,
    project: Path | None = None,
    cache_dir: Path | None = None,
) -> LiteratureOutput:
    """Run `literature_check` with the run's date cap, log it, and write the stage.

    The whole output goes to `runs/<run_id>/literature/<id>.<time>.json` on
    every attempt. A `known` result sets the status; `unconfirmed` and
    `not_found` write the stage and leave the status. An `unchecked` result
    writes no stage and raises LiteratureUnchecked.
    Raise ValueError for a record that is not proposed or survived, or that
    already has a literature stage.
    """
    store = ledger.store
    record = store.load(record_id)
    if record.status not in LITERATURE_STATUSES:
        raise ValueError(f"{record_id} is {record.status}; the literature stage does not apply")
    if Stage.literature in record.stages:
        raise ValueError(f"{record_id} already has a literature stage")
    output = literature_check(
        record,
        ledger.manifest.literature_before,
        ledger=ledger,
        budget=budget,
        backends=backends,
        client=client,
        statement_lean=statement_lean,
        project=project,
        cache_dir=cache_dir,
    )
    run_dir = store.root / "runs" / record.run_id
    directory = run_dir / "literature"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    log = directory / f"{record.id}.{stamp}.json"
    log.write_text(output.model_dump_json(indent=2), encoding="utf-8")
    if output.status == "unchecked":
        raise LiteratureUnchecked(output)
    data: dict[str, JsonValue] = {
        "status": output.status,
        "before": None if output.before is None else output.before.isoformat(),
        "sources": cast(dict[str, JsonValue], dict(output.sources)),
        "checker": None if output.checker is None else output.checker.value,
        "matches": cast(list[JsonValue], [m.model_dump(mode="json") for m in output.matches]),
        "queries": len(output.queries),
        "candidates": len(output.candidates),
    }
    store.write_stage(
        record.id,
        Stage.literature,
        StageOutput(
            written_at=datetime.now(UTC),
            data=data,
            evidence=[log.relative_to(run_dir).as_posix()],
            notes=output.notes,
        ),
        status=Status.known if output.status == "known" else None,
    )
    return output
