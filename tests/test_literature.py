"""Tests for the known-result filter (build plan doc-5wh, Task 10).

Every HTTP response comes from `tests/fixtures/literature/`: responses
recorded from arXiv, the Stack Exchange API, and Loogle on 25 November 2026,
trimmed, with user data removed. Entries named "synthetic" were added by hand
to test the date cap. No test reaches the network.
"""

import ast
import json
import re
import shutil
import sys
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest

import minilab
import minilab.literature.filter as lit_filter
import minilab.literature.http as lit_http
import minilab.literature.pdf as lit_pdf
from minilab.ledger import Ledger
from minilab.literature.arxiv import arxiv_query, search_arxiv
from minilab.literature.filter import (
    LiteratureOutput,
    LiteratureUnchecked,
    literature_check,
    run_literature_stage,
)
from minilab.literature.http import (
    ALLOWED_HOSTS,
    HostNotAllowed,
    LiteratureHttp,
    SourceUnavailable,
)
from minilab.literature.loogle import Hit, QueryRejected, exact_source, search_loogle
from minilab.literature.mathoverflow import search_mathoverflow
from minilab.literature.pdf import PdfUnavailable, pdf_text
from minilab.models.adapter import Budget
from minilab.models.backends import ModelReply
from minilab.records import Family, Provenance, Record, RecordStore, Stage, Status
from minilab.run import MemoryPolicy, RunManifest, save_manifest

FIXTURES = Path(__file__).parent / "fixtures"
LIT = FIXTURES / "literature"
CAP = date(2026, 6, 30)
RUN = "lit"
PAIR = ("information-theory", "optimisation")


# --- a recorded network ---

Route = Callable[[httpx.Request], httpx.Response]


def fixture_bytes(name: str) -> bytes:
    return (LIT / name).read_bytes()


def ok(name: str, content_type: str = "application/json") -> Route:
    return lambda request: httpx.Response(
        200, content=fixture_bytes(name), headers={"content-type": content_type}
    )


def down(request: httpx.Request) -> httpx.Response:
    return httpx.Response(503, text="Service Unavailable")


class Network:
    """A recorded network: routes by host and path, and logs each request."""

    def __init__(self, routes: Mapping[tuple[str, str], Route]) -> None:
        self.routes = dict(routes)
        self.requests: list[httpx.Request] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        for (host, prefix), route in self.routes.items():
            if request.url.host == host and path.startswith(prefix):
                return route(request)
        return httpx.Response(404, text=f"no route for {request.url}")

    def client(self) -> LiteratureHttp:
        return LiteratureHttp(transport=httpx.MockTransport(self.handle), sleep=lambda s: None)


def healthy_routes() -> dict[tuple[str, str], Route]:
    return {
        ("export.arxiv.org", "/api/query"): ok("arxiv_feed.xml", "application/atom+xml"),
        ("api.stackexchange.com", "/2.3/search/advanced"): ok("mo_search.json"),
        ("api.stackexchange.com", "/2.3/questions/"): ok("mo_answers.json"),
        ("loogle.lean-lang.org", "/json"): ok("loogle.json"),
        ("arxiv.org", "/pdf/"): lambda r: httpx.Response(
            200, content=(FIXTURES / "sample.pdf").read_bytes()
        ),
    }


# --- arXiv ---


def test_date_cap_drops_late_papers() -> None:
    net = Network(healthy_routes())
    papers = search_arxiv(arxiv_query(["rate distortion", "linear"]), CAP, client=net.client())
    ids = [p.arxiv_id for p in papers]
    # The API returned the late paper even with the date range; it must be dropped.
    assert "2608.01234" not in ids
    assert ids == ["2411.18494", "2306.02435", "2602.04321"]
    query = net.requests[0].url.params["search_query"]
    assert "submittedDate:[190001010000 TO 202606302359]" in query
    # Revised after the cap: no abstract from the late version, and the first version's PDF.
    revised = papers[2]
    assert revised.revised_after_cap
    assert revised.summary is None
    # The title is the latest version's too, so it is withheld, with the reason.
    assert revised.title is None
    assert revised.withheld is not None and "2026-09-10" in revised.withheld
    assert papers[1].title is not None and papers[1].withheld is None
    assert revised.pdf_url == "https://arxiv.org/pdf/2602.04321v1"
    assert papers[1].summary is not None and "rate distortion" in papers[1].summary.lower()


def test_no_cap_keeps_every_paper() -> None:
    net = Network(healthy_routes())
    papers = search_arxiv(arxiv_query(["rate distortion"]), None, client=net.client())
    assert len(papers) == 4
    assert "submittedDate" not in net.requests[0].url.params["search_query"]
    assert all(not p.revised_after_cap for p in papers)


def test_arxiv_query_quotes_phrases_and_strips_syntax() -> None:
    assert (
        arxiv_query(['rate "distortion"', "a) OR (b"]) == 'all:"rate distortion" AND all:"a OR b"'
    )


def test_arxiv_outage_raises_source_unavailable() -> None:
    routes = healthy_routes()
    routes[("export.arxiv.org", "/api/query")] = down
    with pytest.raises(SourceUnavailable):
        search_arxiv(arxiv_query(["x"]), CAP, client=Network(routes).client())


def test_arxiv_garbage_is_unavailable_not_empty() -> None:
    routes = healthy_routes()
    routes[("export.arxiv.org", "/api/query")] = lambda r: httpx.Response(200, text="<html>")
    with pytest.raises(SourceUnavailable):
        search_arxiv(arxiv_query(["x"]), CAP, client=Network(routes).client())


# --- MathOverflow ---


def test_mathoverflow_date_cap() -> None:
    net = Network(healthy_routes())
    posts = search_mathoverflow("rate distortion", CAP, client=net.client())
    search = net.requests[0]
    assert search.url.params["site"] == "mathoverflow"
    # todate is the last second of the cap day, in UTC.
    assert search.url.params["todate"] == str(int(datetime(2026, 7, 1, tzinfo=UTC).timestamp()) - 1)
    questions = {p.question_id: p for p in posts if p.kind == "question"}
    assert 999001 not in questions, "a question asked after the cap was kept"
    edited = questions[999002]
    assert edited.body is None and edited.withheld is not None
    assert questions[288863].body is not None and "<p>" not in questions[288863].body
    answers = [p for p in posts if p.kind == "answer"]
    # The recorded answer was last edited in 2023; the synthetic one was posted after the cap.
    assert [a.answer_id for a in answers] == [211054]


def test_mathoverflow_without_cap_keeps_late_posts() -> None:
    posts = search_mathoverflow("rate distortion", None, client=Network(healthy_routes()).client())
    assert {p.answer_id for p in posts if p.kind == "answer"} == {211054, 999101}
    assert any(p.question_id == 999001 for p in posts)


def test_mathoverflow_api_error_is_unavailable() -> None:
    routes = healthy_routes()
    routes[("api.stackexchange.com", "/2.3/search/advanced")] = lambda r: httpx.Response(
        400, json={"error_id": 502, "error_name": "throttle_violation", "error_message": "slow"}
    )
    with pytest.raises(SourceUnavailable):
        search_mathoverflow("q", CAP, client=Network(routes).client())


# --- mathlib ---


def test_loogle_hits_and_rejected_query() -> None:
    hits = search_loogle('Real.log, "entropy"', client=Network(healthy_routes()).client())
    assert [h.name for h in hits][:1] == ["Real.binEntropy_le_log_two"]
    assert all(h.via == "loogle" for h in hits)
    routes = healthy_routes()
    routes[("loogle.lean-lang.org", "/json")] = ok("loogle_error.json")
    with pytest.raises(QueryRejected):
        search_loogle("(", client=Network(routes).client())


def test_exact_source_replaces_the_one_sorry() -> None:
    assert exact_source("theorem t : 1 + 1 = 2 := by sorry") == "theorem t : 1 + 1 = 2 := by exact?"
    assert exact_source("theorem t : 1 + 1 = 2 := sorry") == "theorem t : 1 + 1 = 2 := (by exact?)"
    with pytest.raises(ValueError):
        exact_source("theorem t : True := trivial")
    with pytest.raises(ValueError):
        exact_source("theorem a : True := sorry\ntheorem b : True := sorry")


# --- PDFs ---


def test_pdf_text_from_fixture(tmp_path: Path) -> None:
    net = Network(healthy_routes())
    text = pdf_text("https://arxiv.org/pdf/2306.02435v1", client=net.client(), cache_dir=tmp_path)
    assert "rate distortion function is linear" in text
    assert len(list(tmp_path.glob("*.pdf"))) == 1
    # A second read comes from the cache.
    again = pdf_text("https://arxiv.org/pdf/2306.02435v1", client=net.client(), cache_dir=tmp_path)
    assert again == text and len(net.requests) == 1


def test_pdf_text_refuses_non_pdf(tmp_path: Path) -> None:
    routes = healthy_routes()
    routes[("arxiv.org", "/pdf/")] = lambda r: httpx.Response(200, text="<html>captcha</html>")
    with pytest.raises(SourceUnavailable):
        pdf_text("https://arxiv.org/pdf/1v1", client=Network(routes).client(), cache_dir=tmp_path)
    assert not list(tmp_path.glob("*.pdf"))


def test_pdf_text_refuses_other_hosts(tmp_path: Path) -> None:
    with pytest.raises(HostNotAllowed):
        pdf_text("https://example.com/a.pdf", client=Network({}).client(), cache_dir=tmp_path)


# --- the client ---


def test_client_refuses_other_hosts() -> None:
    net = Network({})
    for url in (
        "https://www.google.com/search?q=x",
        "http://arxiv.org/abs/1",
        "https://arxiv.org:8443/abs/1",
        "https://user:secret@arxiv.org/abs/1",
    ):
        with pytest.raises(HostNotAllowed):
            net.client().get(url)
    assert not net.requests


def test_redirect_off_the_list_is_unavailable_not_a_caller_bug() -> None:
    """A redirect is the server's doing: it makes the source unavailable."""
    net = Network(
        {
            ("arxiv.org", "/abs/"): lambda r: httpx.Response(
                302, headers={"location": "https://www.google.com/search?q=x"}
            ),
            ("export.arxiv.org", "/api/"): lambda r: httpx.Response(
                301, headers={"location": "http://export.arxiv.org/api/query"}
            ),
        }
    )
    for url in ("https://arxiv.org/abs/1", "https://export.arxiv.org/api/query"):
        with pytest.raises(SourceUnavailable):
            net.client().get(url)
    assert [r.url.host for r in net.requests] == ["arxiv.org", "export.arxiv.org"]


def test_error_body_is_read_only_in_part() -> None:
    pulled = [0]

    def body() -> Iterator[bytes]:
        for _ in range(1000):
            pulled[0] += 1
            yield b"x" * 100

    client = LiteratureHttp(
        transport=httpx.MockTransport(lambda r: httpx.Response(500, content=body())),
        sleep=lambda s: None,
    )
    with pytest.raises(SourceUnavailable, match="HTTP 500"):
        client.get("https://export.arxiv.org/api/query")
    assert pulled[0] <= 21, f"read {pulled[0] * 100} bytes of an error body"


def test_searches_share_one_default_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """With no client passed, every search uses one shared client, so rate limits hold."""
    net = Network(healthy_routes())
    monkeypatch.setattr(lit_http, "_default", net.client(), raising=False)

    def no_new_client(self: LiteratureHttp, *args: object, **kwargs: object) -> None:
        raise AssertionError("a search made its own client instead of the shared one")

    monkeypatch.setattr(LiteratureHttp, "__init__", no_new_client)
    search_arxiv(arxiv_query(["rate distortion"]), CAP)
    search_mathoverflow("rate distortion", CAP)
    search_loogle("Real.log")
    pdf_text("https://arxiv.org/pdf/2306.02435v1", cache_dir=tmp_path)
    assert {r.url.host for r in net.requests} == {
        "export.arxiv.org",
        "api.stackexchange.com",
        "loogle.lean-lang.org",
        "arxiv.org",
    }


def test_client_waits_between_arxiv_calls() -> None:
    waits: list[float] = []
    now = [100.0]
    net = Network(healthy_routes())
    client = LiteratureHttp(
        transport=httpx.MockTransport(net.handle), sleep=waits.append, clock=lambda: now[0]
    )
    client.get("https://export.arxiv.org/api/query")
    client.get("https://arxiv.org/pdf/1v1")
    assert waits == [pytest.approx(3.0)]  # pyright: ignore[reportUnknownMemberType]


# --- the filter ---


class FakeBackend:
    """Answers the query prompt with `queries` and the judging prompt with `judge`."""

    def __init__(self, family: Family, queries: str, judge: str) -> None:
        self.family = family
        self.queries = queries
        self.judge = judge
        self.prompts: list[str] = []

    def run(
        self, prompt: str, *, cwd: Path, env: Mapping[str, str], timeout_s: float
    ) -> ModelReply:
        self.prompts.append(prompt)
        text = self.queries if "search queries" in prompt else self.judge
        return ModelReply(text=text, tokens_in=10, tokens_out=10, estimated=False, model="fake-1")


QUERIES = json.dumps(
    {
        "arxiv": [["rate distortion", "linear"]],
        "mathoverflow": ["rate distortion"],
        "loogle": ['Real.log, "entropy"'],
    }
)
NOTHING_CLOSE = json.dumps({"matches": []})
# A sentence from the recorded abstract of arXiv 2306.02435.
QUOTE = (
    "We define the complexity of a continuous-time linear system to be the minimum number of "
    "bits required to describe its forward increments"
)


def judge(key: str, closeness: str, quote: str) -> str:
    return json.dumps(
        {"matches": [{"key": key, "closeness": closeness, "quote": quote, "note": "n"}]}
    )


def backends(judge_reply: str) -> dict[Family, FakeBackend]:
    return {f: FakeBackend(f, QUERIES, judge_reply) for f in Family}


def make_record(local: str = "a", status: Status = Status.survived) -> Record:
    return Record(
        id=f"{RUN}.{local}",
        run_id=RUN,
        pair=PAIR,
        mode="plain",
        statement=(
            "The rate distortion function is linear when the optimal output law is not unique."
        ),
        objects=["rate distortion function", "output law"],
        assumptions=["finite alphabets"],
        domain_of_validity="finite alphabets",
        instance_family="2 by 3 channels",
        provenance=Provenance(family=Family.claude, model="m", prompt_hash="h"),
        status=status,
    )


def ledger_budget() -> Budget:
    return Budget(tokens_per_pair=1_000_000)


@pytest.fixture
def store(tmp_path: Path) -> RecordStore:
    return RecordStore(tmp_path)


def make_ledger(store: RecordStore, before: date | None) -> Ledger:
    save_manifest(
        store.root,
        RunManifest(
            run_id=RUN,
            mode="plain",
            pairs=(PAIR,),
            memory_policy=MemoryPolicy.blind,
            memory_before=None,
            literature_before=before,
            budget_tokens_per_pair=1_000_000,
            families=tuple(Family),
            started=datetime(2026, 11, 25, 12, 0, tzinfo=UTC),
        ),
    )
    return Ledger(store, RUN)


def check(
    store: RecordStore,
    judge_reply: str,
    *,
    routes: Mapping[tuple[str, str], Route] | None = None,
    before: date | None = CAP,
) -> tuple[Network, dict[Family, FakeBackend], Ledger]:
    ledger = make_ledger(store, before)
    store.create(make_record())
    return Network(routes or healthy_routes()), backends(judge_reply), ledger


def test_close_verified_match_is_known(store: RecordStore, tmp_path: Path) -> None:
    net, fakes, ledger = check(store, judge("arxiv:2306.02435", "same", QUOTE))
    out = literature_check(
        store.load(f"{RUN}.a"),
        CAP,
        ledger=ledger,
        budget=ledger_budget(),
        backends=fakes,
        client=net.client(),
        cache_dir=tmp_path / "pdf",
    )
    assert out.status == "known"
    assert out.matches[0].verified
    assert out.checker is Family.codex or out.checker is Family.claude
    assert out.checker is not Family.claude, "the checker must not be the author's family"
    # Every query is logged with its outcome and hits.
    sources = {q.source for q in out.queries}
    assert sources == {"arxiv", "mathoverflow", "mathlib"}
    assert all(q.outcome == "ok" for q in out.queries)
    assert any("arxiv:2306.02435" in q.hits for q in out.queries)


def test_quote_not_in_candidate_is_not_known(store: RecordStore, tmp_path: Path) -> None:
    net, fakes, ledger = check(
        store, judge("arxiv:2306.02435", "same", "A sentence the paper never contains at all.")
    )
    out = literature_check(
        store.load(f"{RUN}.a"),
        CAP,
        ledger=ledger,
        budget=ledger_budget(),
        backends=fakes,
        client=net.client(),
        cache_dir=tmp_path / "pdf",
    )
    assert out.status == "unconfirmed"
    assert not out.matches[0].verified
    assert "arxiv:2306.02435" in out.notes


def test_related_match_is_not_known(store: RecordStore, tmp_path: Path) -> None:
    net, fakes, ledger = check(store, judge("arxiv:2306.02435", "related", QUOTE))
    out = literature_check(
        store.load(f"{RUN}.a"),
        CAP,
        ledger=ledger,
        budget=ledger_budget(),
        backends=fakes,
        client=net.client(),
        cache_dir=tmp_path / "pdf",
    )
    assert out.status == "not_found"
    assert out.matches[0].verified


def test_judge_cannot_cite_a_late_paper(store: RecordStore, tmp_path: Path) -> None:
    net, fakes, ledger = check(
        store, judge("arxiv:2608.01234", "same", "Synthetic test entry. Its first version")
    )
    out = literature_check(
        store.load(f"{RUN}.a"),
        CAP,
        ledger=ledger,
        budget=ledger_budget(),
        backends=fakes,
        client=net.client(),
        cache_dir=tmp_path / "pdf",
    )
    assert out.status == "not_found"
    assert all(c.key != "arxiv:2608.01234" for c in out.candidates)
    assert not any("2608.01234" in p for f in fakes.values() for p in f.prompts)


def test_mathlib_match_under_cap_is_a_note_only(store: RecordStore, tmp_path: Path) -> None:
    mathlib_quote = "Real.binEntropy_le_log_two"
    net, fakes, ledger = check(store, judge(f"mathlib:{mathlib_quote}", "same", mathlib_quote))
    record = store.load(f"{RUN}.a")
    capped = literature_check(
        record,
        CAP,
        ledger=ledger,
        budget=ledger_budget(),
        backends=fakes,
        client=net.client(),
        cache_dir=tmp_path / "pdf",
    )
    assert capped.status == "not_found"
    assert capped.matches[0].verified
    assert "mathlib" in capped.notes
    uncapped = literature_check(
        record,
        None,
        ledger=ledger,
        budget=ledger_budget(),
        backends=fakes,
        client=net.client(),
        cache_dir=tmp_path / "pdf",
    )
    assert uncapped.status == "known"


def test_literature_outage_is_unchecked(store: RecordStore, tmp_path: Path) -> None:
    routes = healthy_routes()
    routes[("export.arxiv.org", "/api/query")] = down
    net, fakes, ledger = check(store, NOTHING_CLOSE, routes=routes)
    record_path = store.root / "runs" / RUN / "records" / f"{RUN}.a.json"
    before_bytes = record_path.read_bytes()
    out = literature_check(
        store.load(f"{RUN}.a"),
        CAP,
        ledger=ledger,
        budget=ledger_budget(),
        backends=fakes,
        client=net.client(),
        cache_dir=tmp_path / "pdf",
    )
    assert out.status == "unchecked", "an outage must never read as not found"
    assert out.sources["arxiv"] == "unchecked"
    assert out.sources["mathoverflow"] == "checked"
    assert any(q.source == "arxiv" and q.outcome == "unavailable" for q in out.queries)

    with pytest.raises(LiteratureUnchecked) as caught:
        run_literature_stage(
            f"{RUN}.a",
            ledger=ledger,
            budget=ledger_budget(),
            backends=fakes,
            client=net.client(),
            cache_dir=tmp_path / "pdf",
        )
    assert caught.value.output.status == "unchecked"
    assert record_path.read_bytes() == before_bytes
    # The attempt is still logged, outside the record.
    assert list((store.root / "runs" / RUN / "literature").glob(f"{RUN}.a.*.json"))


def test_outage_with_a_close_match_elsewhere_is_known(store: RecordStore, tmp_path: Path) -> None:
    routes = healthy_routes()
    routes[("api.stackexchange.com", "/2.3/search/advanced")] = down
    net, fakes, ledger = check(store, judge("arxiv:2306.02435", "implies", QUOTE), routes=routes)
    result = run_literature_stage(
        f"{RUN}.a",
        ledger=ledger,
        budget=ledger_budget(),
        backends=fakes,
        client=net.client(),
        cache_dir=tmp_path / "pdf",
    )
    assert result.status == "known"
    record = store.load(f"{RUN}.a")
    assert record.status is Status.known
    output = record.stages[Stage.literature]
    assert output.data["status"] == "known"
    assert output.evidence and (store.root / "runs" / RUN / output.evidence[0]).is_file()


def test_stage_not_found_leaves_status(store: RecordStore, tmp_path: Path) -> None:
    net, fakes, ledger = check(store, NOTHING_CLOSE)
    result = run_literature_stage(
        f"{RUN}.a",
        ledger=ledger,
        budget=ledger_budget(),
        backends=fakes,
        client=net.client(),
        cache_dir=tmp_path / "pdf",
    )
    assert result.status == "not_found"
    record = store.load(f"{RUN}.a")
    assert record.status is Status.survived
    assert record.stages[Stage.literature].data["status"] == "not_found"


def test_unreadable_judge_reply_is_not_not_found(store: RecordStore, tmp_path: Path) -> None:
    net, fakes, ledger = check(store, "I could not decide.")
    record_path = store.root / "runs" / RUN / "records" / f"{RUN}.a.json"
    before_bytes = record_path.read_bytes()
    with pytest.raises(LiteratureUnchecked):
        run_literature_stage(
            f"{RUN}.a",
            ledger=ledger,
            budget=ledger_budget(),
            backends=fakes,
            client=net.client(),
            cache_dir=tmp_path / "pdf",
        )
    assert record_path.read_bytes() == before_bytes


# --- no general web search ---

SRC = Path(minilab.__file__).parent
THIRD_PARTY = {"pydantic", "mpmath", "flint", "numpy", "scipy", "httpx", "typer", "minilab"}
NETWORK_STDLIB = {"urllib", "http", "socket", "ftplib", "smtplib", "webbrowser", "xmlrpc", "ssl"}
HTTPX_MODULES = {"literature/http.py", "models/backends.py"}
LOOPBACK = {"127.0.0.1", "localhost"}


def _imports(tree: ast.Module) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


def test_no_general_web_search() -> None:
    url = re.compile(r"https?://([^/:\s\"'?#]+)")
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = _imports(tree)
        foreign = imports - THIRD_PARTY - set(sys.stdlib_module_names)
        assert not foreign, f"{rel} imports {foreign}, which may be a search-engine client"
        assert not imports & NETWORK_STDLIB, (
            f"{rel} imports a network module: {imports & NETWORK_STDLIB}"
        )
        if "httpx" in imports:
            assert rel in HTTPX_MODULES, f"{rel} makes HTTP calls outside the allowed modules"
        for node in ast.walk(tree):
            # A string in braces is an XML namespace in ElementTree's tag form; it is never fetched.
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and not node.value.startswith("{")
            ):
                for host in url.findall(node.value):
                    assert host in ALLOWED_HOSTS | LOOPBACK, f"{rel} names host {host}"
    assert ALLOWED_HOSTS == {
        "arxiv.org",
        "export.arxiv.org",
        "api.stackexchange.com",
        "loogle.lean-lang.org",
    }


def test_pdftotext_is_installed() -> None:
    assert shutil.which("pdftotext") is not None


def test_pdf_text_needs_bwrap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    real_which = shutil.which

    def which(name: str) -> str | None:
        return None if name == "bwrap" else real_which(name)

    monkeypatch.setattr(lit_pdf.shutil, "which", which)
    ran: list[object] = []

    def popen(*args: object, **kwargs: object) -> None:
        ran.append(args)
        raise AssertionError("pdftotext was started without bwrap")

    monkeypatch.setattr(lit_pdf.subprocess, "Popen", popen)
    with pytest.raises(PdfUnavailable, match="bwrap"):
        pdf_text(
            "https://arxiv.org/pdf/2306.02435v1",
            client=Network(healthy_routes()).client(),
            cache_dir=tmp_path,
        )
    assert not ran


# --- review fixes: source replies ---


def atom(published: str, updated: str, title: str = "T") -> bytes:
    return (
        "<feed xmlns='http://www.w3.org/2005/Atom'><entry>"
        "<id>http://arxiv.org/abs/2601.00001v1</id>"
        f"<title>{title}</title><summary>S</summary>"
        f"<published>{published}</published><updated>{updated}</updated>"
        "</entry></feed>"
    ).encode()


def one_route(response: httpx.Response) -> LiteratureHttp:
    return LiteratureHttp(transport=httpx.MockTransport(lambda r: response), sleep=lambda s: None)


@pytest.mark.parametrize("stamp", ["2026-06-30T12:00:00", "2026-06-30"])
@pytest.mark.parametrize("before", [CAP, None])
def test_arxiv_date_without_timezone_is_unavailable(stamp: str, before: date | None) -> None:
    client = one_route(httpx.Response(200, content=atom(stamp, stamp)))
    with pytest.raises(SourceUnavailable):
        search_arxiv("x", before, client=client)


def mo_routes(
    questions: list[dict[str, object]], answers: list[dict[str, object]]
) -> dict[tuple[str, str], Route]:
    routes = healthy_routes()
    routes[("api.stackexchange.com", "/2.3/search/advanced")] = lambda r: httpx.Response(
        200, json={"items": questions}
    )
    routes[("api.stackexchange.com", "/2.3/questions/")] = lambda r: httpx.Response(
        200, json={"items": answers}
    )
    return routes


def question(qid: int, **fields: object) -> dict[str, object]:
    return {
        "question_id": qid,
        "title": f"Question {qid}",
        "body": f"<p>The body of question {qid}.</p>",
        "link": f"https://mathoverflow.net/q/{qid}",
        "creation_date": 1500000000,
        **fields,
    }


def answer(aid: int, qid: int, body: str, **fields: object) -> dict[str, object]:
    return {
        "answer_id": aid,
        "question_id": qid,
        "body": body,
        "creation_date": 1500000100,
        **fields,
    }


@pytest.mark.parametrize("before", [CAP, None])
def test_mathoverflow_bad_timestamp_is_unavailable(before: date | None) -> None:
    net = Network(mo_routes([question(1, creation_date=10**14)], []))
    with pytest.raises(SourceUnavailable):
        search_mathoverflow("x", before, client=net.client())


def test_mathoverflow_edited_question_withholds_title() -> None:
    edited = question(
        7, title="Post-cap edited title states the theorem", last_edit_date=1790000000
    )
    net = Network(mo_routes([edited], [answer(70, 7, "<p>An answer from 2017.</p>")]))
    posts = search_mathoverflow("x", CAP, client=net.client())
    by_kind = {p.kind: p for p in posts}
    q, a = by_kind["question"], by_kind["answer"]
    assert q.title is None and q.body is None
    assert q.withheld is not None and "2026-09-21" in q.withheld
    # The answer was not edited, but it carries its question's title, which is withheld.
    assert a.title is None and a.body == "An answer from 2017."
    assert a.withheld is not None and "question" in a.withheld
    assert not any("Post-cap" in str(p.model_dump()) for p in posts)


@pytest.mark.parametrize("reply", [{}, {"header": "Found 3", "count": 3, "results": []}])
def test_loogle_reply_of_unknown_shape_is_unavailable(reply: dict[str, object]) -> None:
    with pytest.raises(SourceUnavailable):
        search_loogle("x", client=one_route(httpx.Response(200, json=reply)))


@pytest.mark.parametrize("reply", [{}, {"quota_remaining": 10}])
def test_stack_exchange_reply_of_unknown_shape_is_unavailable(reply: dict[str, object]) -> None:
    with pytest.raises(SourceUnavailable):
        search_mathoverflow("x", CAP, client=one_route(httpx.Response(200, json=reply)))


# --- review fixes: the filter ---


def run_check(
    store: RecordStore,
    judge_reply: str,
    tmp_path: Path,
    *,
    routes: Mapping[tuple[str, str], Route] | None = None,
    before: date | None = CAP,
    queries: str | None = None,
    statement_lean: str | None = None,
) -> tuple[LiteratureOutput, Network, dict[Family, FakeBackend]]:
    net, fakes, ledger = check(store, judge_reply, routes=routes, before=before)
    if queries is not None:
        for fake in fakes.values():
            fake.queries = queries
    out = literature_check(
        store.load(f"{RUN}.a"),
        before,
        ledger=ledger,
        budget=ledger_budget(),
        backends=fakes,
        client=net.client(),
        statement_lean=statement_lean,
        cache_dir=tmp_path / "pdf",
    )
    return out, net, fakes


def judge_prompts(fakes: Mapping[Family, FakeBackend]) -> list[str]:
    return [p for f in fakes.values() for p in f.prompts if "search queries" not in p]


def test_revised_paper_title_is_withheld_and_cannot_be_quoted(
    store: RecordStore, tmp_path: Path
) -> None:
    late_title = "A synthetic paper revised after the cap"
    out, _, fakes = run_check(
        store, judge("arxiv:2602.04321", "same", late_title + " and more words"), tmp_path
    )
    revised = next(c for c in out.candidates if c.key == "arxiv:2602.04321")
    assert revised.title is None and "withheld" in revised.note
    assert not any(late_title in p for p in judge_prompts(fakes))
    assert out.status != "known"


def test_edited_question_title_is_withheld_from_the_checker(
    store: RecordStore, tmp_path: Path
) -> None:
    late_title = "Post-cap edited title: the rate distortion function is linear"
    edited = question(7, title=late_title, last_edit_date=1790000000)
    routes = mo_routes([edited], [answer(70, 7, "<p>An answer from 2017.</p>")])
    out, _, fakes = run_check(
        store, judge("mathoverflow:q7", "same", late_title), tmp_path, routes=routes
    )
    assert not any(late_title in p for p in judge_prompts(fakes))
    assert all(c.title is None for c in out.candidates if c.source == "mathoverflow")
    assert out.status != "known"


@pytest.mark.parametrize(
    "quote",
    [
        "From the PDF (https://arxiv.org/pdf/2306.02435v1)",
        "Abstract: We define the complexity of a continuous-time linear system",
        # The title: true text, but never what the quote check reads.
        "On the complexity of linear systems: an approach via rate distortion theory",
        # In the abstract, but under 40 characters.
        "We define the complexity",
    ],
)
def test_quote_must_be_long_source_text(store: RecordStore, tmp_path: Path, quote: str) -> None:
    out, _, fakes = run_check(store, judge("arxiv:2306.02435", "same", quote), tmp_path)
    assert out.status == "unconfirmed"
    paper = next(c for c in out.candidates if c.key == "arxiv:2306.02435")
    assert "From the PDF" not in paper.text and "Abstract:" not in paper.text
    assert "https://arxiv.org/pdf/2306.02435v1" in paper.note
    # The labels are still in what the checker reads.
    assert "Abstract:" in judge_prompts(fakes)[0]


def test_quote_survives_pdf_hyphenation_and_symbols(
    store: RecordStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = (
        "Theorem 2. The rate dis-\n      tortion function is linear when f(x) ≥ 0\n"
        "and the optimal output law is not unique."
    )

    def fake_pdf_text(url: str, **kwargs: object) -> str:
        return pdf

    monkeypatch.setattr(lit_filter, "pdf_text", fake_pdf_text)
    quote = "the rate distortion function is linear when f(x) >= 0 and the optimal output law"
    out, _, _ = run_check(store, judge("arxiv:2306.02435", "same", quote), tmp_path)
    assert out.status == "known"


@pytest.mark.parametrize(
    ("quote", "status"),
    [
        ("Real.binEntropy_le_log_two", "known"),
        ("Real.binEntropy p ≤ Real.log 2", "known"),
        ("Module: Mathlib.Analysis.SpecialFunctions.BinaryEntropy", "unconfirmed"),
        ("Real.log 2", "unconfirmed"),
    ],
)
def test_mathlib_quotes(store: RecordStore, tmp_path: Path, quote: str, status: str) -> None:
    out, _, _ = run_check(
        store, judge("mathlib:Real.binEntropy_le_log_two", "same", quote), tmp_path, before=None
    )
    assert out.status == status


def loogle_hits(n: int) -> Route:
    hits = [{"name": f"Lemma{i}", "type": "True", "module": "M", "doc": None} for i in range(n)]
    return lambda r: httpx.Response(200, json={"hits": hits})


def test_exact_hit_comes_before_loogle_hits(
    store: RecordStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exact = Hit(name="exact foo_bar", type="", module="", doc=None, via="exact?")

    def fake_exact(statement_lean: str, *, project: Path | None = None) -> list[Hit]:
        return [exact]

    monkeypatch.setattr(lit_filter, "search_exact", fake_exact)
    routes = healthy_routes()
    routes[("loogle.lean-lang.org", "/json")] = loogle_hits(12)
    out, _, _ = run_check(
        store,
        NOTHING_CLOSE,
        tmp_path,
        routes=routes,
        before=None,
        statement_lean="theorem t : True := sorry",
    )
    assert out.candidates[0].key == "mathlib:exact?:exact foo_bar"
    assert out.status == "known"


def atom_feed(prefix: str, n: int) -> bytes:
    entries = "".join(
        f"<entry><id>http://arxiv.org/abs/{prefix}.{i:05d}v1</id><title>P{i}</title>"
        f"<summary>S{i}</summary><published>2020-01-01T00:00:00Z</published>"
        "<updated>2020-01-01T00:00:00Z</updated></entry>"
        for i in range(n)
    )
    return f"<feed xmlns='http://www.w3.org/2005/Atom'>{entries}</feed>".encode()


def test_arxiv_candidates_are_taken_round_robin(store: RecordStore, tmp_path: Path) -> None:
    def arxiv(request: httpx.Request) -> httpx.Response:
        prefix = "2002" if "second" in request.url.params["search_query"] else "2001"
        return httpx.Response(200, content=atom_feed(prefix, 10))

    routes = healthy_routes()
    routes[("export.arxiv.org", "/api/query")] = arxiv
    queries = json.dumps({"arxiv": [["first"], ["second"]], "mathoverflow": ["x"], "loogle": ["y"]})
    out, _, _ = run_check(store, NOTHING_CLOSE, tmp_path, routes=routes, queries=queries)
    shown = [c.key for c in out.candidates if c.source == "arxiv"]
    assert shown[:4] == [
        "arxiv:2001.00000",
        "arxiv:2002.00000",
        "arxiv:2001.00001",
        "arxiv:2002.00001",
    ]
    assert len(shown) == lit_filter.MAX_ARXIV
    logs = [q.model_dump() for q in out.queries if q.source == "arxiv"]
    assert logs[0].get("shown") == [f"arxiv:2001.{i:05d}" for i in range(4)]
    assert logs[1].get("shown") == [f"arxiv:2002.{i:05d}" for i in range(4)]


def test_mathoverflow_answers_come_first(store: RecordStore, tmp_path: Path) -> None:
    questions = [question(100 + i) for i in range(10)]
    answers = [
        answer(5000 + i, 100 + i, f"<p>Answer {i}: this is Theorem 3 of a 1990 paper.</p>")
        for i in range(10)
    ]
    out, _, _ = run_check(store, NOTHING_CLOSE, tmp_path, routes=mo_routes(questions, answers))
    shown = [c.key for c in out.candidates if c.source == "mathoverflow"]
    assert shown == [f"mathoverflow:a{5000 + i}" for i in range(6)] + [
        f"mathoverflow:q{100 + i}" for i in range(6)
    ]
    log = next(q.model_dump() for q in out.queries if q.source == "mathoverflow")
    assert sorted(log.get("shown") or []) == sorted(shown)
    assert len(log["hits"]) == 20


def test_empty_arxiv_query_is_rejected_not_sent(store: RecordStore, tmp_path: Path) -> None:
    queries = json.dumps(
        {
            "arxiv": [['"()"', ":"], ["rate distortion", "linear"]],
            "mathoverflow": ["rate distortion"],
            "loogle": ["Real.log"],
        }
    )
    out, net, _ = run_check(store, NOTHING_CLOSE, tmp_path, queries=queries)
    arxiv_logs = [q for q in out.queries if q.source == "arxiv"]
    assert [q.outcome for q in arxiv_logs] == ["rejected", "ok"]
    assert len([r for r in net.requests if r.url.host == "export.arxiv.org"]) == 1


def test_offlist_redirect_makes_the_stage_unchecked(store: RecordStore, tmp_path: Path) -> None:
    routes = healthy_routes()
    routes[("export.arxiv.org", "/api/query")] = lambda r: httpx.Response(
        302, headers={"location": "http://export.arxiv.org/api/query"}
    )
    net, fakes, ledger = check(store, NOTHING_CLOSE, routes=routes)
    with pytest.raises(LiteratureUnchecked):
        run_literature_stage(
            f"{RUN}.a",
            ledger=ledger,
            budget=ledger_budget(),
            backends=fakes,
            client=net.client(),
            cache_dir=tmp_path / "pdf",
        )


def test_unconfirmed_stage_is_written_and_status_kept(store: RecordStore, tmp_path: Path) -> None:
    bad = "A sentence the paper never contains, long enough to be checked."
    net, fakes, ledger = check(store, judge("arxiv:2306.02435", "same", bad))
    result = run_literature_stage(
        f"{RUN}.a",
        ledger=ledger,
        budget=ledger_budget(),
        backends=fakes,
        client=net.client(),
        cache_dir=tmp_path / "pdf",
    )
    assert result.status == "unconfirmed"
    record = store.load(f"{RUN}.a")
    assert record.status is Status.survived
    stage = record.stages[Stage.literature]
    assert stage.data["status"] == "unconfirmed"
    assert "arxiv:2306.02435" in stage.notes and "confirm" in stage.notes


def test_status_precedence(store: RecordStore, tmp_path: Path) -> None:
    bad = {"key": "arxiv:2411.18494", "closeness": "same", "quote": "x" * 50, "note": "n"}
    good = {"key": "arxiv:2306.02435", "closeness": "implies", "quote": QUOTE, "note": "n"}
    # known beats unconfirmed.
    out, _, _ = run_check(store, json.dumps({"matches": [bad, good]}), tmp_path)
    assert out.status == "known"
    # unchecked beats unconfirmed.
    routes = healthy_routes()
    routes[("api.stackexchange.com", "/2.3/search/advanced")] = down
    store2 = RecordStore(tmp_path / "second")
    out, _, _ = run_check(store2, json.dumps({"matches": [bad]}), tmp_path, routes=routes)
    assert out.status == "unchecked"


def test_mathlib_state_counts_only_without_a_cap(store: RecordStore, tmp_path: Path) -> None:
    routes = healthy_routes()
    routes[("loogle.lean-lang.org", "/json")] = down
    capped, _, _ = run_check(store, NOTHING_CLOSE, tmp_path, routes=routes)
    assert capped.sources["mathlib"] == "unchecked"
    assert capped.status == "not_found", "under a cap mathlib can never count"
    store2 = RecordStore(tmp_path / "second")
    uncapped, _, _ = run_check(store2, NOTHING_CLOSE, tmp_path, routes=routes, before=None)
    assert uncapped.status == "unchecked"
    # Nor does an unconfirmed mathlib match under a cap.
    store3 = RecordStore(tmp_path / "third")
    out, _, _ = run_check(
        store3, judge("mathlib:Real.binEntropy_le_log_two", "same", "Real.log 2"), tmp_path
    )
    assert out.status == "not_found"
