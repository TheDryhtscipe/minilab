"""Tests for the model adapter and its backends (build plan Task 4).

Every test uses fake backends or canned tool output. None calls a live model.
"""

import json
import os
import sys
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import pytest

from minilab.ledger import Ledger
from minilab.models.adapter import (
    Budget,
    BudgetExceeded,
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
    ) -> None:
        self.family = family
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.fail = fail
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
