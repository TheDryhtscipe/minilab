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
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest

import minilab
from minilab.ledger import Ledger
from minilab.literature.arxiv import arxiv_query, search_arxiv
from minilab.literature.filter import (
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
from minilab.literature.loogle import QueryRejected, exact_source, search_loogle
from minilab.literature.mathoverflow import search_mathoverflow
from minilab.literature.pdf import pdf_text
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


def test_client_refuses_other_hosts_and_redirects_off_list() -> None:
    net = Network(
        {
            ("arxiv.org", "/abs/"): lambda r: httpx.Response(
                302, headers={"location": "https://www.google.com/search?q=x"}
            )
        }
    )
    with pytest.raises(HostNotAllowed):
        net.client().get("https://www.google.com/search?q=x")
    with pytest.raises(HostNotAllowed):
        net.client().get("https://arxiv.org/abs/1")
    assert [r.url.host for r in net.requests] == ["arxiv.org"]


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
    assert out.status == "not_found"
    assert not out.matches[0].verified


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
