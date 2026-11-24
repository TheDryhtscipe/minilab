"""Backends: one per model family, each wrapping one tool's headless mode.

A backend turns a prompt into a `ModelReply`. It runs its tool in the working
directory and environment the adapter gives it (an empty temporary directory
and a scrubbed environment; see `minilab.models.adapter`). It reads the token
counts from the tool's JSON output. Where the tool reports none, it estimates
them at 4 characters per token and sets `estimated`.

Any failure raises `ModelUnavailable`: a missing binary, a non-zero exit, a
timeout, output that does not parse, an error the tool reports, or a usage
limit message. A backend never returns a reply for a failed call.

The isolation flags for each tool:

- Claude (`claude -p`): `--tools ""`, `--setting-sources ""`,
  `--strict-mcp-config`, `--no-session-persistence`, and a short system prompt
  in place of Claude Code's. Never `--bare`, which bills the API key, not the
  subscription (MuninnDB `common` memory `01M3WPMQETWRFJQPDGFY0ZVBJG`).
- Codex (`codex exec`): the user's `config.toml` and rules are not loaded (so
  no MCP servers), a read-only sandbox, web search disabled, and every tool
  feature that reaches outside the prompt turned off, including the shell.
- Gemini (Antigravity CLI, `agy -p`): a session-only `--settings` object that
  denies every file, URL, command, and MCP action, keeps the agent inside its
  empty workspace, and never spends paid credits.
- Local (ollama HTTP API): the model has no tools.

Whether each tool really is isolated is checked live by
`scripts/probe_isolation.py`, not assumed from these flags.
"""

import json
import math
import os
import re
import signal
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from minilab.records import Family

CHARS_PER_TOKEN = 4
"""Characters per token, for estimates when a tool reports no counts."""

DEFAULT_TIMEOUT_S = 900.0
"""Default time limit for one model call, in seconds."""

SYSTEM_PROMPT = (
    "You are a careful research mathematician. You have no tools: answer from your own "
    "knowledge, in plain text, and say so when you are unsure."
)
"""System prompt for the Claude backend, in place of Claude Code's own."""

_LIMIT = re.compile(
    r"usage limit|rate[ -]?limit|quota|too many requests|\b429\b|limit reached|"
    r"hit your (usage )?limit|resource[_ ]exhausted|out of credits",
    re.IGNORECASE,
)
MAX_ARG_BYTES = 120_000
"""Largest prompt agy can take: Linux caps one argument at 128 KiB."""

_SHORT_REPLY = 300
"""A reply this short that matches `_LIMIT` is a usage limit message, not an answer."""


class ModelUnavailable(Exception):
    """The tool failed, timed out, or hit a usage limit. No reply exists."""


class ModelReply(BaseModel):
    """A model's answer and what it cost."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    estimated: bool
    """True if either count was estimated because the tool did not report it."""
    model: str


class Backend(Protocol):
    """One model family's tool."""

    @property
    def family(self) -> Family: ...

    def run(
        self, prompt: str, *, cwd: Path, env: Mapping[str, str], timeout_s: float
    ) -> ModelReply: ...


def estimate_tokens(text: str) -> int:
    """Estimate a token count at `CHARS_PER_TOKEN` characters per token, rounded up."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def _limit_message(text: str) -> bool:
    return len(text) <= _SHORT_REPLY and _LIMIT.search(text) is not None


def _tail(text: str, limit: int = 500) -> str:
    text = text.strip()
    return text if len(text) <= limit else "..." + text[-limit:]


def run_command(
    argv: Sequence[str],
    *,
    stdin: str,
    cwd: Path,
    env: Mapping[str, str],
    timeout_s: float,
) -> str:
    """Run `argv` with `stdin`, and return its stdout.

    The tool runs in its own process group. On a timeout the whole group is
    killed, so a tool's child processes do not outlive the call. Raise
    `ModelUnavailable` on a missing binary, a timeout, or a non-zero exit; the
    message carries the end of stderr.
    """
    try:
        proc = subprocess.Popen(
            list(argv),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=dict(env),
            text=True,
            start_new_session=True,
        )
    except OSError as exc:
        raise ModelUnavailable(f"{argv[0]}: cannot start: {exc}") from exc
    try:
        out, err = proc.communicate(stdin, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()
        raise ModelUnavailable(f"{argv[0]}: timed out after {timeout_s} s") from exc
    if proc.returncode != 0:
        raise ModelUnavailable(
            f"{argv[0]}: exit {proc.returncode}: {_tail(err) or _tail(out)}"
        )
    return out


def _load_json(raw: str, tool: str) -> JsonValue:
    try:
        value: JsonValue = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ModelUnavailable(f"{tool}: output is not JSON: {_tail(raw)}") from exc
    return value


def _obj(value: JsonValue) -> dict[str, JsonValue]:
    return value if isinstance(value, dict) else {}


def _int(value: JsonValue) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _reply(
    text: str, tokens_in: int | None, tokens_out: int | None, model: str, prompt: str
) -> ModelReply:
    estimated = tokens_in is None or tokens_out is None
    return ModelReply(
        text=text,
        tokens_in=estimate_tokens(prompt) if tokens_in is None else tokens_in,
        tokens_out=estimate_tokens(text) if tokens_out is None else tokens_out,
        estimated=estimated,
        model=model,
    )


# --- Claude ---------------------------------------------------------------------------------


def parse_claude(raw: str, *, prompt: str = "") -> ModelReply:
    """Parse `claude -p --output-format json` output: a list of events, result last.

    Input tokens include cache writes and cache reads, because all of them
    count against the subscription.
    """
    value = _load_json(raw, "claude")
    events = value if isinstance(value, list) else [value]
    empty: dict[str, JsonValue] = {}
    init = next(
        (_obj(e) for e in events if _obj(e).get("type") == "system"
         and _obj(e).get("subtype") == "init"),
        empty,
    )
    result = next((_obj(e) for e in reversed(events) if _obj(e).get("type") == "result"), None)
    if result is None:
        raise ModelUnavailable(f"claude: no result event: {_tail(raw)}")
    text = result.get("result")
    text = text if isinstance(text, str) else ""
    if result.get("is_error") is not False or result.get("subtype") != "success":
        raise ModelUnavailable(f"claude: error result: {_tail(text or raw)}")
    if _limit_message(text):
        raise ModelUnavailable(f"claude: usage limit: {text}")
    model = init.get("model")
    usage = result.get("usage")
    tokens_in: int | None = None
    tokens_out: int | None = None
    if isinstance(usage, dict):
        tokens_in = (
            _int(usage.get("input_tokens"))
            + _int(usage.get("cache_creation_input_tokens"))
            + _int(usage.get("cache_read_input_tokens"))
        )
        tokens_out = _int(usage.get("output_tokens"))
    return _reply(text, tokens_in, tokens_out, model if isinstance(model, str) else "claude",
                  prompt)


class ClaudeBackend:
    family = Family.claude

    def __init__(self, model: str | None = None, binary: str = "claude") -> None:
        self.model = model
        self.binary = binary

    def argv(self) -> list[str]:
        argv = [
            self.binary,
            "-p",
            "--output-format", "json",
            "--system-prompt", SYSTEM_PROMPT,
            "--tools", "",
            "--setting-sources", "",
            "--strict-mcp-config",
            "--no-session-persistence",
        ]
        if self.model is not None:
            argv += ["--model", self.model]
        return argv

    def run(
        self, prompt: str, *, cwd: Path, env: Mapping[str, str], timeout_s: float
    ) -> ModelReply:
        out = run_command(self.argv(), stdin=prompt, cwd=cwd, env=env, timeout_s=timeout_s)
        return parse_claude(out, prompt=prompt)


# --- Codex ----------------------------------------------------------------------------------

CODEX_DISABLED_FEATURES: tuple[str, ...] = (
    "shell_tool",
    "unified_exec",
    "shell_snapshot",
    "apps",
    "plugins",
    "remote_plugin",
    "browser_use",
    "browser_use_external",
    "in_app_browser",
    "computer_use",
    "hooks",
    "image_generation",
    "view_image",
    "skill_search",
    "skill_mcp_dependency_install",
    "tool_suggest",
)
"""Codex features that give the model a way out of the prompt (checked on codex-cli 0.153)."""


def parse_codex(raw: str, *, model: str, prompt: str = "") -> ModelReply:
    """Parse `codex exec --json` output: one JSON event per line.

    The answer is the last `agent_message` item. Tokens come from the
    `turn.completed` events; Codex counts cached input inside `input_tokens`.
    """
    text: str | None = None
    tokens_in = 0
    tokens_out = 0
    usage_seen = False
    for line in raw.splitlines():
        if not line.strip():
            continue
        event = _obj(_load_json(line, "codex"))
        kind = event.get("type")
        if kind in ("turn.failed", "error"):
            error = _obj(event.get("error"))
            message = error.get("message", event.get("message", line))
            raise ModelUnavailable(f"codex: {message}")
        if kind == "item.completed":
            item = _obj(event.get("item"))
            if item.get("type") == "agent_message" and isinstance(item.get("text"), str):
                text = str(item["text"])
        if kind == "turn.completed":
            usage = event.get("usage")
            if isinstance(usage, dict):
                usage_seen = True
                tokens_in += _int(usage.get("input_tokens"))
                tokens_out += _int(usage.get("output_tokens")) + _int(
                    usage.get("reasoning_output_tokens")
                )
    if text is None:
        raise ModelUnavailable(f"codex: no agent message: {_tail(raw)}")
    if _limit_message(text):
        raise ModelUnavailable(f"codex: usage limit: {text}")
    if not usage_seen:
        return _reply(text, None, None, model, prompt)
    return _reply(text, tokens_in, tokens_out, model, prompt)


class CodexBackend:
    family = Family.codex

    def __init__(self, model: str | None = None, binary: str = "codex") -> None:
        self.model = model
        self.binary = binary

    def argv(self) -> list[str]:
        argv = [
            self.binary,
            "exec",
            "--json",
            "--ignore-user-config",
            "--ignore-rules",
            "--ephemeral",
            "--skip-git-repo-check",
            "--sandbox", "read-only",
            "-c", 'web_search="disabled"',
        ]
        for feature in CODEX_DISABLED_FEATURES:
            argv += ["--disable", feature]
        if self.model is not None:
            argv += ["--model", self.model]
        return argv + ["-"]

    def run(
        self, prompt: str, *, cwd: Path, env: Mapping[str, str], timeout_s: float
    ) -> ModelReply:
        out = run_command(self.argv(), stdin=prompt, cwd=cwd, env=env, timeout_s=timeout_s)
        return parse_codex(out, model=self.model or "codex-default", prompt=prompt)


# --- Gemini (Antigravity CLI) ---------------------------------------------------------------

AGY_SETTINGS: dict[str, JsonValue] = {
    "toolPermission": "strict",
    "allowNonWorkspaceAccess": False,
    "enableTerminalSandbox": True,
    "useG1Credits": False,
    "disableSlashCommands": True,
    "permissions": {
        "deny": [
            "read_file(*)",
            "write_file(*)",
            "read_url(*)",
            "execute_url(*)",
            "download_url(*)",
            "command(*)",
            "mcp(*)",
        ],
    },
}
"""Session-only agy settings: every action denied (agy 1.5.16). Never saved to disk."""


def parse_agy(raw: str, *, model: str, prompt: str = "") -> ModelReply:
    """Parse `agy -p --output-format json` output: one JSON object.

    On agy 1.5.16, `output_tokens` already includes `thinking_tokens`, and
    `total_tokens` is input plus output. Any part of the total that the
    input and output counts do not cover is charged as input, so no spend is
    lost if that changes.
    """
    value = _obj(_load_json(raw, "agy"))
    status = value.get("status")
    text = value.get("response")
    if status != "SUCCESS" or not isinstance(text, str):
        raise ModelUnavailable(f"agy: status {status}: {value.get('error', _tail(raw))}")
    if _limit_message(text):
        raise ModelUnavailable(f"agy: usage limit: {text}")
    usage = value.get("usage")
    if not isinstance(usage, dict):
        return _reply(text, None, None, model, prompt)
    tokens_in = _int(usage.get("input_tokens"))
    tokens_out = _int(usage.get("output_tokens"))
    tokens_in += max(0, _int(usage.get("total_tokens")) - tokens_in - tokens_out)
    return _reply(text, tokens_in, tokens_out, model, prompt)


class GeminiBackend:
    family = Family.gemini

    def __init__(self, model: str | None = None, binary: str | None = None) -> None:
        self.model = model
        self.binary = binary or str(Path.home() / ".local" / "bin" / "agy")

    def argv(self, prompt: str = "") -> list[str]:
        """agy reads its prompt only from the command line (`--print=<prompt>`), not stdin."""
        argv = [
            self.binary,
            "--output-format", "json",
            "--disable-slash-commands",
            "--settings", json.dumps(AGY_SETTINGS),
        ]
        if self.model is not None:
            argv += ["--model", self.model]
        return argv + [f"--print={prompt}"]

    def run(
        self, prompt: str, *, cwd: Path, env: Mapping[str, str], timeout_s: float
    ) -> ModelReply:
        if len(prompt.encode()) > MAX_ARG_BYTES:
            raise ModelUnavailable(
                f"agy: prompt is {len(prompt.encode())} bytes; the limit for one "
                f"command-line argument is {MAX_ARG_BYTES}"
            )
        out = run_command(self.argv(prompt), stdin="", cwd=cwd, env=env, timeout_s=timeout_s)
        return parse_agy(out, model=self.model or "agy-default", prompt=prompt)


# --- Local (ollama) -------------------------------------------------------------------------


class _OllamaReply(BaseModel):
    model_config = ConfigDict(extra="ignore")

    response: str
    model: str
    prompt_eval_count: int | None = Field(default=None, ge=0)
    eval_count: int | None = Field(default=None, ge=0)


class LocalBackend:
    """The local weak model through ollama's HTTP API. It has no tools to isolate."""

    family = Family.local

    def __init__(
        self, model: str = "qwen3.5:4b", url: str = "http://127.0.0.1:11434"
    ) -> None:
        self.model = model
        self.url = url

    def run(
        self, prompt: str, *, cwd: Path, env: Mapping[str, str], timeout_s: float
    ) -> ModelReply:
        try:
            response = httpx.post(
                f"{self.url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=timeout_s,
            )
            response.raise_for_status()
            reply = _OllamaReply.model_validate_json(response.content)
        except httpx.TimeoutException as exc:
            raise ModelUnavailable(f"ollama: timed out after {timeout_s} s") from exc
        except (httpx.HTTPError, ValidationError) as exc:
            raise ModelUnavailable(f"ollama: {exc}") from exc
        return _reply(reply.response, reply.prompt_eval_count, reply.eval_count, reply.model,
                      prompt)
