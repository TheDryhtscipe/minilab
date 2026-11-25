"""Tests for the model adapter and its backends (build plan Task 4).

Every test uses fake backends or canned tool output. None calls a live model.
"""

import json
import os
import signal
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from itertools import combinations
from pathlib import Path

import httpx
import pytest

import minilab.models.backends as backends_module
from minilab.ledger import Ledger
from minilab.models.adapter import (
    ROLE_STAGE,
    Budget,
    BudgetExceeded,
    ModelNotStarted,
    ModelReply,
    ModelUnavailable,
    NoChecker,
    Role,
    RoleNotAllowed,
    call,
    pick_checker,
    scrubbed_env,
)
from minilab.models.backends import (
    MAX_ARG_BYTES,
    ClaudeBackend,
    CodexBackend,
    GeminiBackend,
    LocalBackend,
    estimate_tokens,
    parse_agy,
    parse_claude,
    parse_codex,
    run_command,
)
from minilab.records import Family, Provenance, Record, RecordStore, Stage
from minilab.run import MemoryPolicy, RunManifest, save_manifest

RUN = "run-1"
PAIR = ("information-theory", "optimisation")
OTHER_PAIR = ("kernels", "probability")


class FakeBackend:
    """A backend that records what it saw and returns a fixed reply, or raises."""

    def __init__(
        self,
        family: Family,
        *,
        tokens_in: int = 10,
        tokens_out: int = 5,
        fail: bool = False,
        error: BaseException | None = None,
        action: Callable[[Path], None] | None = None,
    ) -> None:
        self.family = family
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.fail = fail
        self.error = error
        self.action = action
        self.calls = 0
        self.seen_env: dict[str, str] = {}
        self.seen_cwd: Path | None = None
        self.seen_cwd_entries: list[str] = []

    def run(
        self, prompt: str, *, cwd: Path, env: Mapping[str, str], timeout_s: float
    ) -> ModelReply:
        self.calls += 1
        self.seen_env = dict(env)
        self.seen_cwd = cwd
        self.seen_cwd_entries = sorted(os.listdir(cwd))
        if self.action is not None:
            self.action(cwd)
        if self.error is not None:
            raise self.error
        if self.fail:
            raise ModelUnavailable("fake: usage limit reached")
        return ModelReply(
            text=f"reply to {prompt}",
            tokens_in=self.tokens_in,
            tokens_out=self.tokens_out,
            estimated=False,
            model="fake-1",
        )


def _record(local: str = "a") -> Record:
    return Record(
        id=f"{RUN}.{local}",
        run_id=RUN,
        pair=PAIR,
        mode="structured",
        statement="R(D) is convex in D for every source P.",
        objects=["R(D)", "P"],
        assumptions=["P has full support"],
        domain_of_validity="0 <= D <= D_max",
        instance_family="n in 1..4",
        provenance=Provenance(family=Family.claude, model="m", prompt_hash="h"),
    )


@pytest.fixture
def store(tmp_path: Path) -> RecordStore:
    return RecordStore(tmp_path)


@pytest.fixture
def ledger(store: RecordStore) -> Ledger:
    save_manifest(
        store.root,
        RunManifest(
            run_id=RUN,
            mode="structured",
            pairs=(PAIR, OTHER_PAIR),
            memory_policy=MemoryPolicy.blind,
            memory_before=None,
            literature_before=None,
            budget_tokens_per_pair=100,
            families=tuple(Family),
            started=datetime(2026, 11, 24, 12, 0, tzinfo=UTC),
        ),
    )
    return Ledger(store, RUN)


def _call(
    ledger: Ledger,
    backend: FakeBackend,
    *,
    role: Role = Role.encode,
    record_id: str | None = None,
    pair: tuple[str, str] = PAIR,
    budget: int = 100,
) -> ModelReply:
    return call(
        backend.family,
        role,
        "prompt",
        record_id=record_id,
        pair=pair,
        ledger=ledger,
        budget=Budget(tokens_per_pair=budget),
        backends={backend.family: backend},
    )


# --- The four tests the plan names ---------------------------------------------------------


def test_budget_exceeded_before_call(ledger: Ledger) -> None:
    backend = FakeBackend(Family.claude, tokens_in=60, tokens_out=40)
    _call(ledger, backend)
    assert ledger.pair_spend(PAIR) == 100
    with pytest.raises(BudgetExceeded):
        _call(ledger, backend)
    assert backend.calls == 1, "the backend must not be called once the pair is at its budget"
    assert ledger.pair_spend(PAIR) == 100
    # Another pair has its own budget.
    _call(ledger, backend, pair=OTHER_PAIR)
    assert backend.calls == 2


def test_adapter_failure_leaves_record_unchanged(store: RecordStore, ledger: Ledger) -> None:
    store.create(_record())
    path = store.root / "runs" / RUN / "records" / f"{RUN}.a.json"
    before = path.read_bytes()
    backend = FakeBackend(Family.codex, fail=True)
    with pytest.raises(ModelUnavailable):
        _call(ledger, backend, record_id=f"{RUN}.a")
    assert backend.calls == 1
    assert path.read_bytes() == before
    # The failed call is still charged to its pair, as an estimate.
    [charge] = ledger.charges()
    assert charge.estimated
    assert charge.family is Family.codex
    assert charge.tokens_out == 0
    assert charge.tokens_in == estimate_tokens("prompt")


def test_pick_checker_never_author_never_gemini() -> None:
    families = list(Family)
    subsets = [
        list(subset) for size in range(len(families) + 1) for subset in combinations(families, size)
    ]
    for author in Family:
        for available in subsets:
            for short in (False, True):
                for codex_left in (False, True):
                    try:
                        checker = pick_checker(
                            author, available, short=short, codex_budget_left=codex_left
                        )
                    except NoChecker:
                        assert not ({Family.claude, Family.codex} & set(available)) - {author}
                        continue
                    assert checker is not author
                    assert checker is not Family.gemini
                    assert checker in available
                    if checker is Family.codex and Family.claude in available:
                        assert author is Family.claude or (short and codex_left)
    # Preferences, stated directly.
    both = [Family.claude, Family.codex, Family.gemini]
    assert pick_checker(Family.gemini, both) is Family.claude
    assert pick_checker(Family.gemini, both, short=True) is Family.claude
    assert pick_checker(Family.gemini, both, short=True, codex_budget_left=True) is Family.codex
    assert pick_checker(Family.claude, both) is Family.codex


def test_scrubbed_env_and_empty_cwd(ledger: Ledger, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://example.invalid")
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("GEMINI_API_KEY", "secret")
    backend = FakeBackend(Family.claude)
    _call(ledger, backend)
    assert backend.seen_env, "the backend must get an environment"
    leaked = [
        name
        for name in backend.seen_env
        if name.startswith(("ANTHROPIC_", "CLAUDE_CODE_USE_", "OPENAI_", "GEMINI_"))
    ]
    assert leaked == []
    assert backend.seen_env.get("HOME") == os.environ["HOME"]
    assert backend.seen_cwd is not None
    assert backend.seen_cwd_entries == []
    assert backend.seen_cwd != Path.cwd()
    assert not backend.seen_cwd.exists(), "the call's working directory is removed afterwards"


# --- Further adapter rules -----------------------------------------------------------------


def test_call_charges_reply_tokens_to_role_stage(store: RecordStore, ledger: Ledger) -> None:
    store.create(_record())
    backend = FakeBackend(Family.claude, tokens_in=7, tokens_out=3)
    reply = _call(ledger, backend, role=Role.formalise, record_id=f"{RUN}.a")
    assert reply.text == "reply to prompt"
    [charge] = ledger.charges()
    assert (charge.tokens_in, charge.tokens_out, charge.estimated) == (7, 3, False)
    assert charge.stage is Stage.formaliser
    assert charge.record_id == f"{RUN}.a"


def test_gemini_refused_for_every_role_but_generate(ledger: Ledger) -> None:
    backend = FakeBackend(Family.gemini)
    for role in Role:
        if role is Role.generate:
            continue
        with pytest.raises(RoleNotAllowed):
            _call(ledger, backend, role=role)
    assert backend.calls == 0
    _call(ledger, backend, role=Role.generate)
    assert backend.calls == 1


def test_record_of_other_pair_refused_before_call(store: RecordStore, ledger: Ledger) -> None:
    store.create(_record())
    backend = FakeBackend(Family.claude)
    with pytest.raises(ValueError):
        _call(ledger, backend, record_id=f"{RUN}.a", pair=OTHER_PAIR)
    assert backend.calls == 0


def test_scrubbed_env_keeps_only_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOME_TOKEN", "x")
    monkeypatch.setenv("PATH", "/usr/bin")
    env = scrubbed_env()
    assert "SOME_TOKEN" not in env
    assert env["PATH"] == "/usr/bin"


# --- Backend parsing and the subprocess helper --------------------------------------------


def test_parse_claude_event_list() -> None:
    raw = json.dumps(
        [
            {"type": "system", "subtype": "init", "model": "claude-opus-5-5", "tools": []},
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": "hello",
                "usage": {
                    "input_tokens": 10,
                    "cache_creation_input_tokens": 100,
                    "cache_read_input_tokens": 1000,
                    "output_tokens": 7,
                },
            },
        ]
    )
    reply = parse_claude(raw)
    assert reply == ModelReply(
        text="hello", tokens_in=1110, tokens_out=7, estimated=False, model="claude-opus-5-5"
    )


def test_parse_claude_error_and_usage_limit() -> None:
    error = json.dumps(
        [{"type": "result", "subtype": "success", "is_error": True, "result": "Overloaded"}]
    )
    with pytest.raises(ModelUnavailable):
        parse_claude(error)
    limit = json.dumps(
        [
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": "Claude AI usage limit reached|1764000000",
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
        ]
    )
    with pytest.raises(ModelUnavailable):
        parse_claude(limit)
    with pytest.raises(ModelUnavailable):
        parse_claude("not json")


def test_long_answer_mentioning_rate_limits_is_not_a_failure() -> None:
    text = "The rate limit of the channel is reached when " + "the input law is optimal. " * 20
    raw = json.dumps(
        [
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": text,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        ]
    )
    assert parse_claude(raw).text == text


def test_parse_codex_jsonl() -> None:
    lines = [
        {"type": "thread.started", "thread_id": "t"},
        {"type": "item.completed", "item": {"type": "reasoning", "text": "thinking"}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "answer"}},
        {
            "type": "turn.completed",
            "usage": {"input_tokens": 20, "cached_input_tokens": 5, "output_tokens": 9},
        },
    ]
    reply = parse_codex("\n".join(json.dumps(line) for line in lines), model="gpt-x")
    assert reply == ModelReply(text="answer", tokens_in=20, tokens_out=9, estimated=False,
                               model="gpt-x")


def test_parse_codex_failure_and_missing_usage() -> None:
    failed = [{"type": "turn.failed", "error": {"message": "You've hit your usage limit."}}]
    with pytest.raises(ModelUnavailable):
        parse_codex("\n".join(json.dumps(line) for line in failed), model="gpt-x")
    no_usage = [{"type": "item.completed", "item": {"type": "agent_message", "text": "abcd" * 5}}]
    reply = parse_codex("\n".join(json.dumps(line) for line in no_usage), model="gpt-x",
                        prompt="p" * 40)
    assert reply.estimated
    assert (reply.tokens_in, reply.tokens_out) == (10, 5)


def test_parse_agy() -> None:
    ok = {
        "status": "SUCCESS",
        "response": "answer",
        # As agy 1.5.16 reports it: output includes thinking; total is input plus output.
        "usage": {"input_tokens": 12072, "output_tokens": 47, "thinking_tokens": 37,
                  "cache_read_tokens": 0, "total_tokens": 12119},
    }
    reply = parse_agy(json.dumps(ok), model="gemini-x")
    assert reply == ModelReply(text="answer", tokens_in=12072, tokens_out=47, estimated=False,
                               model="gemini-x")
    # Tokens in the total that neither count covers are charged as input.
    ok["usage"] = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 20}
    assert parse_agy(json.dumps(ok), model="gemini-x").tokens_in == 15
    failed = {"status": "ERROR", "error": "quota exceeded"}
    with pytest.raises(ModelUnavailable):
        parse_agy(json.dumps(failed), model="gemini-x")


def test_estimate_tokens_four_chars() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


def test_run_command_timeout_kills_and_raises(tmp_path: Path) -> None:
    # The tool starts a grandchild that holds stdout open; killing only the tool would
    # leave the call waiting for the grandchild.
    script = "import subprocess, time; subprocess.Popen(['sleep', '30']); time.sleep(30)"
    started = time.monotonic()
    with pytest.raises(ModelUnavailable, match="timed out"):
        run_command([sys.executable, "-c", script], stdin="", cwd=tmp_path,
                    env=scrubbed_env(), timeout_s=0.5)
    assert time.monotonic() - started < 10, "the whole process group must be killed"


def test_run_command_nonzero_exit_and_missing_binary(tmp_path: Path) -> None:
    argv = [sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"]
    with pytest.raises(ModelUnavailable, match="boom"):
        run_command(argv, stdin="", cwd=tmp_path, env=scrubbed_env(), timeout_s=10)
    with pytest.raises(ModelUnavailable):
        run_command(["/nonexistent/tool"], stdin="", cwd=tmp_path, env=scrubbed_env(),
                    timeout_s=10)


def test_backend_command_lines_carry_isolation_flags() -> None:
    claude = ClaudeBackend().argv()
    for flag in ("-p", "--tools", "--setting-sources", "--strict-mcp-config",
                 "--no-session-persistence"):
        assert flag in claude
    assert "--bare" not in claude
    assert claude[claude.index("--tools") + 1] == ""
    assert claude[claude.index("--setting-sources") + 1] == ""
    codex = CodexBackend().argv()
    for flag in ("--ignore-user-config", "--ignore-rules", "--ephemeral", "--json",
                 "--skip-git-repo-check"):
        assert flag in codex
    assert codex[codex.index("--sandbox") + 1] == "read-only"
    assert 'web_search="disabled"' in codex
    for feature in ("shell_tool", "unified_exec", "browser_use", "computer_use", "apps",
                    "plugins"):
        assert feature in codex
    agy = GeminiBackend().argv("-starts with a dash")
    assert agy[-1] == "--print=-starts with a dash"
    settings = json.loads(agy[agy.index("--settings") + 1])
    assert settings["allowNonWorkspaceAccess"] is False
    assert settings["useG1Credits"] is False
    for action in ("read_file", "write_file", "read_url", "execute_url", "download_url",
                   "command", "mcp"):
        assert f"{action}(*)" in settings["permissions"]["deny"]
    assert "--dangerously-skip-permissions" not in agy
    assert LocalBackend().model == "qwen3.5:4b"


# --- Charging failed calls -----------------------------------------------------------------


def test_failed_call_charged_with_reported_usage(ledger: Ledger) -> None:
    error = ModelUnavailable("claude: error result", tokens_in=500, tokens_out=70)
    with pytest.raises(ModelUnavailable):
        _call(ledger, FakeBackend(Family.claude, error=error))
    [charge] = ledger.charges()
    assert (charge.tokens_in, charge.tokens_out, charge.estimated) == (500, 70, True)


def test_failed_call_input_is_at_least_the_prompt_estimate(ledger: Ledger) -> None:
    error = ModelUnavailable("codex: failed", tokens_in=1, tokens_out=None)
    with pytest.raises(ModelUnavailable):
        _call(ledger, FakeBackend(Family.codex, error=error))
    [charge] = ledger.charges()
    assert (charge.tokens_in, charge.tokens_out) == (estimate_tokens("prompt"), 0)


def test_not_started_call_is_not_charged(ledger: Ledger) -> None:
    with pytest.raises(ModelNotStarted):
        _call(ledger, FakeBackend(Family.claude, error=ModelNotStarted("no binary")))
    assert ledger.charges() == []


def test_any_backend_exception_is_charged_and_raised(ledger: Ledger) -> None:
    with pytest.raises(RuntimeError, match="parser bug"):
        _call(ledger, FakeBackend(Family.claude, error=RuntimeError("parser bug")))
    [charge] = ledger.charges()
    assert charge.estimated
    assert charge.tokens_in == estimate_tokens("prompt")


def test_failing_charge_does_not_hide_the_call_error(
    ledger: Ledger, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_charge(**_: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(ledger, "charge", broken_charge)
    with pytest.raises(ModelUnavailable, match="tool broke") as caught:
        _call(ledger, FakeBackend(Family.claude, error=ModelUnavailable("tool broke")))
    assert any("disk full" in note for note in caught.value.__notes__)


# --- Checks before the call ----------------------------------------------------------------


def test_family_not_in_manifest_refused_before_call(store: RecordStore) -> None:
    save_manifest(
        store.root,
        RunManifest(
            run_id="run-2",
            mode="structured",
            pairs=(PAIR,),
            memory_policy=MemoryPolicy.blind,
            memory_before=None,
            literature_before=None,
            budget_tokens_per_pair=100,
            families=(Family.claude,),
            started=datetime(2026, 11, 24, 12, 0, tzinfo=UTC),
        ),
    )
    backend = FakeBackend(Family.codex)
    with pytest.raises(ValueError, match="manifest"):
        _call(Ledger(store, "run-2"), backend)
    assert backend.calls == 0


def test_record_of_other_run_refused_before_call(ledger: Ledger) -> None:
    backend = FakeBackend(Family.claude)
    with pytest.raises(ValueError, match="not in run"):
        _call(ledger, backend, record_id="run-2.a")
    assert backend.calls == 0


def test_family_given_as_string_and_backend_family_checked(ledger: Ledger) -> None:
    backend = FakeBackend(Family.claude)
    reply = call("claude", Role.encode, "prompt", record_id=None, pair=PAIR, ledger=ledger,
                 budget=Budget(tokens_per_pair=100), backends={Family.claude: backend})
    assert reply.text == "reply to prompt"
    # A gemini backend registered under claude must not be run as claude.
    impostor = FakeBackend(Family.gemini)
    with pytest.raises(ValueError, match="backend"):
        call(Family.claude, Role.audit, "prompt", record_id=None, pair=PAIR, ledger=ledger,
             budget=Budget(tokens_per_pair=100), backends={Family.claude: impostor})
    assert impostor.calls == 0
    with pytest.raises(RoleNotAllowed):
        call("gemini", Role.audit, "prompt", record_id=None, pair=PAIR, ledger=ledger,
             budget=Budget(tokens_per_pair=100), backends={Family.gemini: impostor})


def test_budget_larger_than_manifest_refused(ledger: Ledger) -> None:
    backend = FakeBackend(Family.claude, tokens_in=30, tokens_out=0)
    with pytest.raises(ValueError, match="budget"):
        _call(ledger, backend, budget=101)
    assert backend.calls == 0
    _call(ledger, backend, budget=50)
    _call(ledger, backend, budget=50)
    with pytest.raises(BudgetExceeded):
        _call(ledger, backend, budget=50)


def test_pick_checker_coerces_strings() -> None:
    assert pick_checker("claude", ["claude", "codex"]) is Family.codex
    assert pick_checker("codex", ["claude", "codex"]) is Family.claude
    with pytest.raises(NoChecker):
        pick_checker("claude", ["claude", "gemini"])


def test_role_stage_map() -> None:
    assert ROLE_STAGE == {
        Role.generate: Stage.generator,
        Role.encode: Stage.encoding,
        Role.reformulate: Stage.falsifier,
        Role.mechanism: Stage.falsifier,
        Role.formalise: Stage.formaliser,
        Role.roundtrip: Stage.formaliser,
        Role.audit: Stage.formaliser,
        Role.literature: Stage.literature,
        Role.prove: Stage.prover,
        Role.curate: Stage.queue,
        Role.grade: Stage.queue,
    }


_LOCK_CHILD = """
import sys, time
from pathlib import Path
from minilab.ledger import Ledger
from minilab.records import Family, RecordStore
from minilab.models.adapter import Budget, BudgetExceeded, ModelReply, Role, call

class Slow:
    family = Family.claude
    def run(self, prompt, *, cwd, env, timeout_s):
        began = time.time()
        time.sleep(1.0)
        print(f"ran {began} {time.time()}")
        return ModelReply(text="x", tokens_in=60, tokens_out=0, estimated=False, model="m")

ledger = Ledger(RecordStore(Path(sys.argv[1])), "run-1")
start_at = float(sys.argv[2])
while time.time() < start_at:
    time.sleep(0.005)
try:
    call(Family.claude, Role.encode, "p", record_id=None, pair=("information-theory",
         "optimisation"), ledger=ledger,
         budget=Budget(tokens_per_pair=100, reserve_output_tokens=59),
         backends={Family.claude: Slow()})
    print("ok")
except BudgetExceeded:
    print("refused")
"""


def test_pair_lock_lets_only_the_budget_through(
    store: RecordStore, ledger: Ledger, tmp_path: Path
) -> None:
    script = tmp_path / "child.py"
    script.write_text(_LOCK_CHILD)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    start_at = time.time() + 4
    procs = [
        subprocess.Popen([sys.executable, str(script), str(store.root), str(start_at)],
                         stdout=subprocess.PIPE, text=True, env=env)
        for _ in range(4)
    ]
    lines = [proc.communicate(timeout=60)[0].split() for proc in procs]
    outcomes = sorted(line[-1] for line in lines)
    # Each call reserves 1 (the prompt) + 59 tokens, so two fit under 100 and the rest wait
    # for no one: they are refused at once.
    assert outcomes == ["ok", "ok", "refused", "refused"]
    assert ledger.pair_spend(PAIR) == 120
    assert ledger.open_reservations() == []
    spans = [(float(line[1]), float(line[2])) for line in lines if line[0] == "ran"]
    assert len(spans) == 2
    (a0, a1), (b0, b1) = spans
    assert a0 < b1 and b0 < a1, f"the two calls on one pair did not overlap: {spans}"


# --- Temporary directory -------------------------------------------------------------------


def test_read_only_tree_in_call_dir_is_removed(ledger: Ledger) -> None:
    def make_read_only_tree(cwd: Path) -> None:
        (cwd / "ro").mkdir()
        (cwd / "ro" / "f").write_text("x")
        (cwd / "ro").chmod(stat.S_IRUSR | stat.S_IXUSR)

    backend = FakeBackend(Family.claude, action=make_read_only_tree)
    _call(ledger, backend)
    assert backend.seen_cwd is not None
    assert not backend.seen_cwd.exists()


def test_call_dir_that_stays_is_warned_about(
    ledger: Ledger, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_rmtree(path: object, **_: object) -> None:
        return None

    monkeypatch.setattr("minilab.models.adapter.shutil.rmtree", no_rmtree)
    backend = FakeBackend(Family.claude)
    with pytest.warns(UserWarning, match="minilab-call-"):
        reply = _call(ledger, backend)
    assert reply.text == "reply to prompt"
    assert backend.seen_cwd is not None
    backend.seen_cwd.rmdir()


def test_scrubbed_env_drops_account_and_trust_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("CLAUDE_CONFIG_DIR", "CODEX_HOME", "SSL_CERT_FILE", "SSL_CERT_DIR"):
        monkeypatch.setenv(name, "/elsewhere")
    monkeypatch.setenv("XDG_CONFIG_HOME", "/home/x/.config")
    env = scrubbed_env()
    assert not {"CLAUDE_CONFIG_DIR", "CODEX_HOME", "SSL_CERT_FILE", "SSL_CERT_DIR"} & set(env)
    assert env["XDG_CONFIG_HOME"] == "/home/x/.config"


# --- Parsers -------------------------------------------------------------------------------


def _claude_result(text: object = "answer", **fields: object) -> str:
    result: dict[str, object] = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": text,
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }
    result.update(fields)
    return json.dumps([result])


def _codex(*events: object) -> str:
    return "\n".join(json.dumps(event) for event in events)


_MESSAGE = {"type": "item.completed", "item": {"type": "agent_message", "text": "answer"}}
_DONE = {"type": "turn.completed", "usage": {"input_tokens": 20, "output_tokens": 9}}


@pytest.mark.parametrize("answer", ["The answer is 429.", "The limit reached as n→∞ is 1/2."])
def test_short_maths_answers_are_replies(answer: str) -> None:
    assert parse_claude(_claude_result(answer)).text == answer
    message = {"type": "item.completed", "item": {"type": "agent_message", "text": answer}}
    assert parse_codex(_codex(message, _DONE), model="m").text == answer
    agy = {"status": "SUCCESS", "response": answer}
    assert parse_agy(json.dumps(agy), model="m").text == answer


def test_parse_claude_subtype_must_be_success() -> None:
    with pytest.raises(ModelUnavailable, match="error_max_turns"):
        parse_claude(_claude_result("partial", subtype="error_max_turns"))


def test_parse_claude_error_result_carries_usage() -> None:
    raw = _claude_result("", is_error=True, subtype="error_during_execution",
                         usage={"input_tokens": 50000, "output_tokens": 8000})
    with pytest.raises(ModelUnavailable) as caught:
        parse_claude(raw)
    assert (caught.value.tokens_in, caught.value.tokens_out) == (50000, 8000)


def test_parse_claude_last_result_wins() -> None:
    first = {"type": "result", "subtype": "success", "is_error": False, "result": "old",
             "usage": {"input_tokens": 1, "output_tokens": 1}}
    last = {**first, "result": "new"}
    assert parse_claude(json.dumps([first, last])).text == "new"


@pytest.mark.parametrize("text", [None, "", 42])
def test_parse_claude_missing_text_is_malformed(text: object) -> None:
    with pytest.raises(ModelUnavailable, match="no reply text"):
        parse_claude(_claude_result(text))


def test_parse_claude_model_from_model_usage() -> None:
    raw = _claude_result(modelUsage={"claude-sonnet-4-6": {"inputTokens": 3}})
    assert parse_claude(raw).model == "claude-sonnet-4-6"
    assert parse_claude(_claude_result()).model == "claude"


def test_parse_claude_counts() -> None:
    # Bools and negatives are not counts: each such count is estimated.
    reply = parse_claude(_claude_result(usage={"input_tokens": True, "output_tokens": 5}),
                         prompt="p" * 40)
    assert (reply.tokens_in, reply.tokens_out, reply.estimated) == (10, 5, True)
    reply = parse_claude(_claude_result(usage={"input_tokens": 4, "output_tokens": -1}))
    assert (reply.tokens_in, reply.tokens_out, reply.estimated) == (4, 2, True)
    reply = parse_claude(_claude_result(usage={"input_tokens": 4, "output_tokens": "9"}))
    assert reply.estimated
    reply = parse_claude(_claude_result(usage={"input_tokens": 4, "output_tokens": 2.5}))
    assert reply.estimated
    # An empty usage object is an estimate, not a charge of zero.
    reply = parse_claude(_claude_result("abcdefgh", usage={}), prompt="p" * 40)
    assert (reply.tokens_in, reply.tokens_out, reply.estimated) == (10, 2, True)
    # Whole-number floats are counts.
    reply = parse_claude(_claude_result(usage={"input_tokens": 1.2e4, "output_tokens": 3.0}))
    assert (reply.tokens_in, reply.tokens_out, reply.estimated) == (12000, 3, False)
    # An invalid cache count makes the input count an estimate.
    reply = parse_claude(_claude_result(usage={"input_tokens": 4, "output_tokens": 3,
                                               "cache_read_input_tokens": False}),
                         prompt="p" * 40)
    assert (reply.tokens_in, reply.estimated) == (10, True)


def test_parse_codex_error_then_completion_is_ok() -> None:
    reconnect = {"type": "error", "message": "Reconnecting... 1/5"}
    reply = parse_codex(_codex(reconnect, _MESSAGE, _DONE), model="m")
    assert reply.text == "answer"


def test_parse_codex_error_without_completion_is_fatal() -> None:
    error = {"type": "error", "message": "stream disconnected"}
    with pytest.raises(ModelUnavailable, match="stream disconnected"):
        parse_codex(_codex(_DONE, _MESSAGE, error), model="m")


def test_parse_codex_turn_failed_carries_usage() -> None:
    failed = {"type": "turn.failed", "error": {"message": "boom"}}
    with pytest.raises(ModelUnavailable, match="boom") as caught:
        parse_codex(_codex(_DONE, _MESSAGE, failed), model="m")
    assert (caught.value.tokens_in, caught.value.tokens_out) == (20, 9)


def test_parse_codex_skips_lines_that_are_not_objects() -> None:
    raw = "Reading prompt from stdin...\n[1, 2]\n" + _codex(_MESSAGE, _DONE)
    assert parse_codex(raw, model="m").text == "answer"
    with pytest.raises(ModelUnavailable, match="no agent message"):
        parse_codex("not json\n[]\n" + _codex(_DONE), model="m")
    empty = {"type": "item.completed", "item": {"type": "agent_message", "text": ""}}
    with pytest.raises(ModelUnavailable, match="no agent message"):
        parse_codex(_codex(empty, _DONE), model="m")


def test_parse_codex_counts() -> None:
    # Reasoning is already inside output_tokens; it is not added again.
    done = {"type": "turn.completed",
            "usage": {"input_tokens": 20, "output_tokens": 9, "reasoning_output_tokens": 4}}
    reply = parse_codex(_codex(_MESSAGE, done), model="m")
    assert (reply.tokens_in, reply.tokens_out) == (20, 9)
    # A total above input plus output is charged as input.
    done = {"type": "turn.completed",
            "usage": {"input_tokens": 20, "output_tokens": 9, "total_tokens": 35}}
    assert parse_codex(_codex(_MESSAGE, done), model="m").tokens_in == 26
    # An empty usage object is an estimate.
    no_counts: dict[str, int] = {}
    done = {"type": "turn.completed", "usage": no_counts}
    assert parse_codex(_codex(_MESSAGE, done), model="m").estimated


def test_parse_agy_empty_response_and_counts() -> None:
    with pytest.raises(ModelUnavailable, match="no reply text"):
        parse_agy(json.dumps({"status": "SUCCESS", "response": ""}), model="m")
    failed = {"status": "ERROR", "error": "quota", "usage": {"input_tokens": 7,
                                                             "output_tokens": 1}}
    with pytest.raises(ModelUnavailable) as caught:
        parse_agy(json.dumps(failed), model="m")
    assert (caught.value.tokens_in, caught.value.tokens_out) == (7, 1)
    ok = {"status": "SUCCESS", "response": "answer",
          "usage": {"input_tokens": 1.2e4, "output_tokens": False}}
    reply = parse_agy(json.dumps(ok), model="m")
    assert (reply.tokens_in, reply.tokens_out, reply.estimated) == (12000, 2, True)


# --- run_command ---------------------------------------------------------------------------


def _tool(tmp_path: Path, name: str, body: str) -> str:
    path = tmp_path / name
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(0o755)
    return str(path)


def _kill(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def test_run_command_kills_children_after_success(tmp_path: Path) -> None:
    # The child counts into a file while it lives; the count must stop once the call returns.
    beat = tmp_path / "beat"
    child = (
        "import itertools, sys, time\n"
        "for n in itertools.count():\n"
        "    open(sys.argv[1], 'w').write(str(n)); time.sleep(0.05)\n"
    )
    tool = _tool(tmp_path, "tool", f"""
import subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", {child!r}, {str(beat)!r}],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
open({str(tmp_path / "pid")!r}, "w").write(str(child.pid))
while not __import__("os").path.exists({str(beat)!r}):
    time.sleep(0.01)
print("done")
""")
    out = run_command([tool], stdin="", cwd=tmp_path, env=scrubbed_env(), timeout_s=30)
    assert out == "done\n"
    try:
        time.sleep(0.5)
        before = beat.read_text()
        time.sleep(1)
        assert beat.read_text() == before, "a child of the tool outlived the call"
    finally:
        _kill(int((tmp_path / "pid").read_text()))


def test_run_command_timeout_with_child_in_new_session(tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"
    tool = _tool(tmp_path, "tool", f"""
import subprocess, sys, time
child = subprocess.Popen(["sleep", "40"], start_new_session=True)
open({str(pid_file)!r}, "w").write(str(child.pid))
time.sleep(40)
""")
    started = time.monotonic()
    try:
        with pytest.raises(ModelUnavailable, match="timed out"):
            run_command([tool], stdin="", cwd=tmp_path, env=scrubbed_env(), timeout_s=1)
        assert time.monotonic() - started < 20, "a child holding the pipes must not block"
    finally:
        if pid_file.exists():
            _kill(int(pid_file.read_text()))


def test_run_command_refuses_before_start(tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    tool = _tool(tmp_path, "tool", f"open({str(marker)!r}, 'w').write('x')\n")
    env = scrubbed_env()
    with pytest.raises(ModelNotStarted):
        run_command([tool], stdin="bad \ud800 text", cwd=tmp_path, env=env, timeout_s=10)
    with pytest.raises(ModelNotStarted):
        run_command([tool, "bad \ud800"], stdin="", cwd=tmp_path, env=env, timeout_s=10)
    with pytest.raises(ModelNotStarted):
        run_command([tool, "a\0b"], stdin="", cwd=tmp_path, env=env, timeout_s=10)
    with pytest.raises(ModelNotStarted):
        run_command([str(tmp_path / "missing")], stdin="", cwd=tmp_path, env=env, timeout_s=10)
    assert not marker.exists()


def test_run_command_decodes_bad_utf8(tmp_path: Path) -> None:
    tool = _tool(tmp_path, "tool", "import sys; sys.stdout.buffer.write(b'\\xffok')\n")
    out = run_command([tool], stdin="é", cwd=tmp_path, env=scrubbed_env(), timeout_s=10)
    assert out == "\ufffdok"


# --- Backends end to end, against fake tools -----------------------------------------------

_CLAUDE_OK = """
import json, sys
prompt = sys.stdin.read()
print(json.dumps([
    {"type": "system", "subtype": "init", "model": "claude-fake"},
    {"type": "result", "subtype": "success", "is_error": False, "result": "got " + prompt,
     "usage": {"input_tokens": 11, "cache_read_input_tokens": 100, "output_tokens": 4}},
]))
"""

_CLAUDE_FAIL = """
import json, sys
sys.stdin.read()
print(json.dumps([{"type": "result", "subtype": "error_during_execution", "is_error": True,
                   "result": "", "usage": {"input_tokens": 500, "output_tokens": 60}}]))
sys.exit(1)
"""

_CODEX_OK = """
import json, sys
prompt = sys.stdin.read()
print("Reading prompt from stdin...")
for event in [
    {"type": "thread.started", "thread_id": "t"},
    {"type": "item.completed", "item": {"type": "agent_message", "text": "got " + prompt}},
    {"type": "turn.completed", "usage": {"input_tokens": 30, "output_tokens": 8}},
]:
    print(json.dumps(event))
"""

_CODEX_FAIL = """
import json, sys
sys.stdin.read()
print(json.dumps({"type": "turn.failed", "error": {"message": "You've hit your usage limit."}}))
sys.exit(1)
"""

_AGY_OK = """
import json, sys
[prompt] = [a[len("--print="):] for a in sys.argv if a.startswith("--print=")]
print(json.dumps({"status": "SUCCESS", "response": "got " + prompt,
                  "usage": {"input_tokens": 12, "output_tokens": 5, "total_tokens": 17}}))
"""

_AGY_FAIL = """
import json, sys
print(json.dumps({"status": "ERROR", "error": "quota exceeded"}))
sys.exit(1)
"""


def test_claude_backend_end_to_end(tmp_path: Path) -> None:
    env = scrubbed_env()
    ok = ClaudeBackend(binary=_tool(tmp_path, "claude_ok", _CLAUDE_OK))
    reply = ok.run("hi", cwd=tmp_path, env=env, timeout_s=10)
    assert reply == ModelReply(text="got hi", tokens_in=111, tokens_out=4, estimated=False,
                               model="claude-fake")
    fail = ClaudeBackend(binary=_tool(tmp_path, "claude_fail", _CLAUDE_FAIL))
    with pytest.raises(ModelUnavailable, match="exit 1") as caught:
        fail.run("hi", cwd=tmp_path, env=env, timeout_s=10)
    assert (caught.value.tokens_in, caught.value.tokens_out) == (500, 60)


def test_codex_backend_end_to_end(tmp_path: Path) -> None:
    env = scrubbed_env()
    ok = CodexBackend(model="gpt-fake", binary=_tool(tmp_path, "codex_ok", _CODEX_OK))
    reply = ok.run("hi", cwd=tmp_path, env=env, timeout_s=10)
    assert reply == ModelReply(text="got hi", tokens_in=30, tokens_out=8, estimated=False,
                               model="gpt-fake")
    fail = CodexBackend(binary=_tool(tmp_path, "codex_fail", _CODEX_FAIL))
    with pytest.raises(ModelUnavailable, match="exit 1"):
        fail.run("hi", cwd=tmp_path, env=env, timeout_s=10)


def test_gemini_backend_end_to_end(tmp_path: Path) -> None:
    env = scrubbed_env()
    ok = GeminiBackend(model="gemini-fake", binary=_tool(tmp_path, "agy_ok", _AGY_OK))
    reply = ok.run("-hi", cwd=tmp_path, env=env, timeout_s=10)
    assert reply == ModelReply(text="got -hi", tokens_in=12, tokens_out=5, estimated=False,
                               model="gemini-fake")
    fail = GeminiBackend(binary=_tool(tmp_path, "agy_fail", _AGY_FAIL))
    with pytest.raises(ModelUnavailable, match="quota exceeded"):
        fail.run("hi", cwd=tmp_path, env=env, timeout_s=10)


def test_failed_tool_charged_with_its_reported_usage(ledger: Ledger, tmp_path: Path) -> None:
    backend = ClaudeBackend(binary=_tool(tmp_path, "claude_fail", _CLAUDE_FAIL))
    with pytest.raises(ModelUnavailable):
        call(Family.claude, Role.encode, "prompt", record_id=None, pair=PAIR, ledger=ledger,
             budget=Budget(tokens_per_pair=100), backends={Family.claude: backend})
    [charge] = ledger.charges()
    assert (charge.tokens_in, charge.tokens_out, charge.estimated) == (500, 60, True)


@pytest.mark.parametrize(
    "prompt", ["x" * (MAX_ARG_BYTES + 1000), "a\0b", "bad \ud800"], ids=["size", "nul", "text"]
)
def test_agy_refusals_not_started_and_not_charged(
    ledger: Ledger, tmp_path: Path, prompt: str
) -> None:
    # MAX_ARG_BYTES + 1000 still fits in one Linux argument, so only the size check stops it.
    marker = tmp_path / "ran"
    tool = _tool(tmp_path, "agy", f"open({str(marker)!r}, 'w').write('x')\n" + _AGY_OK)
    backend = GeminiBackend(binary=tool)
    with pytest.raises(ModelNotStarted):
        call(Family.gemini, Role.generate, prompt, record_id=None, pair=PAIR, ledger=ledger,
             budget=Budget(tokens_per_pair=100), backends={Family.gemini: backend})
    assert not marker.exists()
    assert ledger.charges() == []


# --- Local backend -------------------------------------------------------------------------


def test_local_backend_parses_reply(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["prompt"] == "hi"
        return httpx.Response(200, json={"response": "hello", "model": "qwen-fake",
                                         "prompt_eval_count": 6, "eval_count": 2})

    backend = LocalBackend(transport=httpx.MockTransport(handler))
    reply = backend.run("hi", cwd=tmp_path, env={}, timeout_s=10)
    assert reply == ModelReply(text="hello", tokens_in=6, tokens_out=2, estimated=False,
                               model="qwen-fake")


class _FakeOllama(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 (the name http.server calls)
        self.rfile.read(int(self.headers["Content-Length"]))
        body = json.dumps({"response": "hello", "model": "qwen-fake"}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return None


def test_local_backend_ignores_proxy_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A fake ollama on the loopback; the proxy variables point at a closed port.
    for name in ("ALL_PROXY", "HTTP_PROXY", "all_proxy", "http_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    server = HTTPServer(("127.0.0.1", 0), _FakeOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        backend = LocalBackend(url=f"http://127.0.0.1:{server.server_port}")
        reply = backend.run("hi", cwd=tmp_path, env={}, timeout_s=10)
    finally:
        server.shutdown()
        server.server_close()
    assert reply.text == "hello"
    assert reply.estimated


def test_local_backend_timeout(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    backend = LocalBackend(transport=httpx.MockTransport(handler))
    with pytest.raises(ModelUnavailable, match="timed out"):
        backend.run("hi", cwd=tmp_path, env={}, timeout_s=0.1)


# --- Reservations and the pair lock (minilab-294360) --------------------------------------


def test_pair_lock_is_free_while_the_tool_runs(ledger: Ledger) -> None:
    taken: list[bool] = []

    def try_the_lock(cwd: Path) -> None:
        def take() -> None:
            with ledger.pair_lock(PAIR):
                taken.append(True)

        thread = threading.Thread(target=take, daemon=True)
        thread.start()
        thread.join(timeout=5)
        taken.append(not thread.is_alive())

    backend = FakeBackend(Family.claude, action=try_the_lock)
    _call(ledger, backend)
    assert taken == [True, True], "the pair lock was held during the model call"


def test_call_reserves_then_settles(ledger: Ledger) -> None:
    seen: list[int] = []
    backend = FakeBackend(
        Family.claude, tokens_in=10, tokens_out=5,
        action=lambda cwd: seen.append(ledger.pair_spend(PAIR)),
    )
    call(Family.claude, Role.encode, "x" * 40, record_id=None, pair=PAIR, ledger=ledger,
         budget=Budget(tokens_per_pair=100, reserve_output_tokens=7),
         backends={Family.claude: backend})
    assert seen == [10 + 7], "during the call the pair's spend is the reservation"
    assert ledger.pair_spend(PAIR) == 15
    assert ledger.open_reservations() == []


def test_open_reservations_count_against_the_budget(ledger: Ledger) -> None:
    ledger.reserve(pair=PAIR, stage=Stage.encoding, family=Family.claude, tokens=100)
    backend = FakeBackend(Family.claude)
    with pytest.raises(BudgetExceeded):
        _call(ledger, backend)
    assert backend.calls == 0


def test_not_started_call_releases_its_reservation(ledger: Ledger) -> None:
    backend = FakeBackend(Family.claude, error=ModelNotStarted("no binary"))
    with pytest.raises(ModelNotStarted):
        _call(ledger, backend)
    assert ledger.pair_spend(PAIR) == 0
    assert ledger.open_reservations() == []
    assert ledger.charges() == []


def test_failed_call_settles_its_reservation(ledger: Ledger) -> None:
    backend = FakeBackend(Family.claude, fail=True)
    with pytest.raises(ModelUnavailable):
        _call(ledger, backend)
    assert ledger.open_reservations() == []
    [charge] = ledger.charges()
    assert charge.estimated and charge.reservation is not None


def test_budget_default_output_allowance() -> None:
    assert Budget(tokens_per_pair=1).reserve_output_tokens == 32_000
    with pytest.raises(ValueError):
        Budget(tokens_per_pair=1, reserve_output_tokens=-1)


# --- Tool output cap (minilab-294360) ------------------------------------------------------


def test_run_command_fails_past_the_output_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(backends_module, "MAX_OUTPUT_BYTES", 1 << 20, raising=False)
    pid_file = tmp_path / "pid"
    tool = _tool(tmp_path, "tool", f"""
import os, sys, time
open({str(pid_file)!r}, "w").write(str(os.getpid()))
chunk = b"x" * 65536
for _ in range(320):
    sys.stdout.buffer.write(chunk)
sys.stdout.flush()
time.sleep(60)
""")
    started = time.monotonic()
    with pytest.raises(ModelUnavailable, match="more than 1048576 bytes"):
        run_command([tool], stdin="", cwd=tmp_path, env=scrubbed_env(), timeout_s=20)
    assert time.monotonic() - started < 15
    pid = int(pid_file.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        _kill(pid)
        pytest.fail("the tool outlived an oversized call")


def test_run_command_keeps_only_the_end_of_a_flood_of_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(backends_module, "MAX_OUTPUT_BYTES", 1 << 20, raising=False)
    tool = _tool(tmp_path, "tool", """
import sys
for _ in range(160):
    sys.stderr.write("e" * 65536)
sys.stderr.write("THE END")
sys.stdout.write("ok")
sys.exit(3)
""")
    with pytest.raises(ModelUnavailable, match="THE END") as info:
        run_command([tool], stdin="", cwd=tmp_path, env=scrubbed_env(), timeout_s=20)
    assert "exit 3" in str(info.value)


def test_run_command_large_prompt_and_reply_do_not_deadlock(tmp_path: Path) -> None:
    tool = _tool(tmp_path, "tool", """
import sys
data = sys.stdin.buffer.read()
sys.stdout.buffer.write(data[::-1])
""")
    prompt = "ab" * (1 << 20)
    out = run_command([tool], stdin=prompt, cwd=tmp_path, env=scrubbed_env(), timeout_s=20)
    assert out == prompt[::-1]
