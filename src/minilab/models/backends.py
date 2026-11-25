"""Backends: one per model family, each wrapping one tool's headless mode.

A backend turns a prompt into a `ModelReply`. It runs its tool in the working
directory and environment the adapter gives it (an empty temporary directory
and a scrubbed environment; see `minilab.models.adapter`). It reads the token
counts from the tool's JSON output. Where the tool reports a count as missing
or invalid, it estimates that count at 4 characters per token and sets
`estimated`.

Any failure raises `ModelUnavailable`: a non-zero exit, a timeout, output that
does not parse, a reply with no text, or an error the tool reports. A failure
before the tool started (a missing binary, a prompt that cannot be passed to
the tool) raises its subclass `ModelNotStarted`. When a failed call's output
still reports the tokens it used, the exception carries them, so the adapter
can charge them. A backend never returns a reply for a failed call.

The isolation flags for each tool:

- Claude (`claude -p`): `--tools ""`, `--setting-sources ""`,
  `--strict-mcp-config`, `--no-session-persistence`, and a short system prompt
  in place of Claude Code's. Never `--bare`, which bills the API key, not the
  subscription (MuninnDB `common` memory `01M3WPMQETWRFJQPDGFY0ZVBJG`).
- Codex (`codex exec`): the user's `config.toml` and rules are not loaded (so
  no MCP servers), a read-only sandbox, web search disabled, and every tool
  feature that reaches outside the prompt turned off, including the shell.
- Gemini (Antigravity CLI, `agy -p`): a session-only `--settings` object that
  denies file, URL, command, and MCP actions, keeps the agent inside its empty
  workspace, and never spends paid credits. The deny list does not cover
  agy's built-in `search_web` tool, nor its browser or subagent tools. The
  probe of 2026-11-24 found that Gemini still searches the web, so Gemini is
  "normal runs only" and may not be used in the benchmark (see
  `src/minilab/models/isolation.json`; follow-up bead minilab-2be43f).
- Local (ollama HTTP API): the model has no tools. The HTTP client ignores
  proxy variables in the environment.

Whether each tool really is isolated is checked live by
`scripts/probe_isolation.py`, not assumed from these flags.
"""

import json
import math
import os
import re
import selectors
import signal
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from minilab.records import Family

CHARS_PER_TOKEN = 4
"""Characters per token, for estimates when a tool reports no counts."""

DEFAULT_TIMEOUT_S = 900.0
"""Default time limit for one model call, in seconds."""

KILL_WAIT_S = 5.0
"""After a timeout kills the tool, how long to wait for its output pipes to close."""

MAX_OUTPUT_BYTES = 64 * 2**20
"""Most bytes of standard output one tool call may print. Past this the call fails."""

STDERR_KEEP_BYTES = 64 * 2**10
"""Bytes kept from the end of a tool's standard error; the rest is read and dropped."""

_READ_CHUNK = 2**16

SYSTEM_PROMPT = (
    "You are a careful research mathematician. You have no tools: answer from your own "
    "knowledge, in plain text, and say so when you are unsure."
)
"""System prompt for the Claude backend, in place of Claude Code's own."""

_CLAUDE_LIMIT = re.compile(r"^\s*Claude AI usage limit reached")
"""Claude has printed this usage limit message as a result that is not marked an error."""

MAX_ARG_BYTES = 120_000
"""Largest prompt agy can take: Linux caps one argument at 128 KiB."""


class ModelUnavailable(Exception):
    """The tool failed, timed out, or hit a usage limit. No reply exists.

    `tokens_in` and `tokens_out` are the counts the tool reported for the
    failed call, or None where it reported none.
    """

    def __init__(
        self, message: str, *, tokens_in: int | None = None, tokens_out: int | None = None
    ) -> None:
        super().__init__(message)
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out


class ModelNotStarted(ModelUnavailable):
    """The call failed before the tool started, so it used no tokens."""


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


def _tail(text: str, limit: int = 500) -> str:
    text = text.strip()
    return text if len(text) <= limit else "..." + text[-limit:]


class OutputTooLarge(ModelUnavailable):
    """The tool printed more than `MAX_OUTPUT_BYTES` bytes, so the call was stopped."""


class ToolExit(ModelUnavailable):
    """The tool started and exited non-zero. `stdout` is what it printed."""

    def __init__(self, message: str, *, stdout: str) -> None:
        super().__init__(message)
        self.stdout = stdout


def _killpg(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        # The group is gone (or its id now belongs to a process we may not signal).
        pass


def run_command(
    argv: Sequence[str],
    *,
    stdin: str,
    cwd: Path,
    env: Mapping[str, str],
    timeout_s: float,
) -> str:
    """Run `argv` with `stdin` (UTF-8), and return its stdout decoded as UTF-8.

    Bytes that are not valid UTF-8 in the output become U+FFFD. Raise
    `ModelNotStarted`, before anything starts, if `stdin` or an argument
    cannot be encoded as UTF-8, if an argument holds a NUL character, or if
    the process cannot be started (a missing binary, for one).

    The tool runs in its own process group. After the call, whether it
    succeeded or not, the whole group is killed, so the tool's child processes
    do not outlive it. On a timeout the group is killed at once; if a child
    that left the group still holds the output pipes, they are closed after
    `KILL_WAIT_S` seconds rather than waited on. Raise `ModelUnavailable` on a
    timeout, and `ToolExit` (a `ModelUnavailable`) on a non-zero exit; the
    message carries the end of stderr.

    At most `MAX_OUTPUT_BYTES` of stdout are kept in memory. If the tool
    prints more, its group is killed at once and `OutputTooLarge` (a
    `ModelUnavailable`) is raised. Only the last `STDERR_KEEP_BYTES` of
    stderr are kept.
    """
    tool = argv[0] if argv else "(no command)"
    if not argv:
        raise ModelNotStarted("empty command")
    try:
        data = stdin.encode("utf-8")
        for arg in argv:
            arg.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ModelNotStarted(
            f"{tool}: the prompt or an argument is not valid text: {exc}"
        ) from exc
    if any("\0" in arg for arg in argv):
        raise ModelNotStarted(f"{tool}: an argument holds a NUL character")
    try:
        proc = subprocess.Popen(
            list(argv),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=dict(env),
            start_new_session=True,
        )
    except (OSError, ValueError) as exc:
        raise ModelNotStarted(f"{tool}: cannot start: {exc}") from exc
    max_out = MAX_OUTPUT_BYTES
    try:
        try:
            out_b, err_b = _exchange(proc, data, timeout_s=timeout_s, max_out=max_out)
        except subprocess.TimeoutExpired as exc:
            _killpg(proc.pid)
            _drain_then_close(proc)
            raise ModelUnavailable(f"{tool}: timed out after {timeout_s} s") from exc
        except _Overflow as exc:
            _killpg(proc.pid)
            _drain_then_close(proc, read=False)
            raise OutputTooLarge(
                f"{tool}: printed more than {max_out} bytes; stopped"
            ) from exc
    finally:
        _killpg(proc.pid)
    out = out_b.decode("utf-8", errors="replace")
    err = err_b.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        raise ToolExit(
            f"{tool}: exit {proc.returncode}: {_tail(err) or _tail(out)}", stdout=out
        )
    return out


class _Overflow(Exception):
    """Standard output passed its cap."""


def _exchange(
    proc: "subprocess.Popen[bytes]", data: bytes, *, timeout_s: float, max_out: int
) -> tuple[bytes, bytes]:
    """Write `data` to the tool's stdin and read its stdout and stderr until both close.

    Like `Popen.communicate`, but stdout is capped at `max_out` bytes (raise
    `_Overflow` past it) and only the end of stderr is kept. Raise
    `subprocess.TimeoutExpired` if the pipes are still open after `timeout_s`
    seconds. Wait for the tool to exit before returning.
    """
    assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
    deadline = time.monotonic() + timeout_s
    out = bytearray()
    err = bytearray()
    sent = 0
    with selectors.DefaultSelector() as sel:
        if data:
            os.set_blocking(proc.stdin.fileno(), False)
            sel.register(proc.stdin, selectors.EVENT_WRITE)
        else:
            proc.stdin.close()
        sel.register(proc.stdout, selectors.EVENT_READ)
        sel.register(proc.stderr, selectors.EVENT_READ)
        while sel.get_map():
            left = deadline - time.monotonic()
            if left <= 0:
                raise subprocess.TimeoutExpired(proc.args, timeout_s)
            for key, _ in sel.select(left):
                pipe = key.fileobj
                if pipe is proc.stdin:
                    try:
                        sent += os.write(proc.stdin.fileno(), data[sent : sent + _READ_CHUNK])
                    except BlockingIOError:
                        continue
                    except BrokenPipeError:
                        sent = len(data)
                    if sent >= len(data):
                        sel.unregister(proc.stdin)
                        proc.stdin.close()
                    continue
                fd = key.fd
                chunk = os.read(fd, _READ_CHUNK)
                if not chunk:
                    sel.unregister(pipe)
                    continue
                if pipe is proc.stdout:
                    out += chunk
                    if len(out) > max_out:
                        raise _Overflow
                else:
                    err += chunk
                    if len(err) > STDERR_KEEP_BYTES:
                        del err[: len(err) - STDERR_KEEP_BYTES]
    left = deadline - time.monotonic()
    try:
        proc.wait(timeout=max(left, 0.0))
    except subprocess.TimeoutExpired:
        raise subprocess.TimeoutExpired(proc.args, timeout_s) from None
    return bytes(out), bytes(err)


def _drain_then_close(proc: "subprocess.Popen[bytes]", *, read: bool = True) -> None:
    """After the group was killed: close the pipes and reap the tool.

    With `read`, first discard what is left on stdout and stderr for up to
    `KILL_WAIT_S` seconds, so the pipes close cleanly; a child that left the
    group and still holds them is not waited on longer than that.
    """
    if read:
        try:
            _exchange(proc, b"", timeout_s=KILL_WAIT_S, max_out=MAX_OUTPUT_BYTES)
        except (subprocess.TimeoutExpired, _Overflow, OSError, ValueError):
            pass
    for pipe in (proc.stdin, proc.stdout, proc.stderr):
        if pipe is not None:
            pipe.close()
    proc.wait()


def _run_and_parse(
    argv: Sequence[str],
    parse: Callable[[str], ModelReply],
    *,
    stdin: str,
    cwd: Path,
    env: Mapping[str, str],
    timeout_s: float,
) -> ModelReply:
    """Run the tool and parse its output. On a non-zero exit, keep any usage it reported."""
    try:
        out = run_command(argv, stdin=stdin, cwd=cwd, env=env, timeout_s=timeout_s)
    except ToolExit as exc:
        try:
            reply = parse(exc.stdout)
        except ModelUnavailable as parsed:
            tokens_in, tokens_out = parsed.tokens_in, parsed.tokens_out
        else:
            tokens_in = None if reply.estimated else reply.tokens_in
            tokens_out = None if reply.estimated else reply.tokens_out
        raise ModelUnavailable(str(exc), tokens_in=tokens_in, tokens_out=tokens_out) from exc
    return parse(out)


def _load_json(raw: str, tool: str) -> JsonValue:
    try:
        value: JsonValue = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ModelUnavailable(f"{tool}: output is not JSON: {_tail(raw)}") from exc
    return value


def _obj(value: JsonValue) -> dict[str, JsonValue]:
    return value if isinstance(value, dict) else {}


def _int(value: JsonValue) -> int | None:
    """A token count, or None if it is missing or not a non-negative whole number."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and math.isfinite(value) and value.is_integer() and value >= 0:
        return int(value)
    return None


def _sum(usage: dict[str, JsonValue], required: str, *optional: str) -> int | None:
    """`required` plus every `optional` count present; None if any of them is invalid."""
    total = _int(usage.get(required))
    for name in optional:
        if total is None:
            return None
        if name in usage:
            part = _int(usage[name])
            total = None if part is None else total + part
    return total


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

    The last `result` event counts. Input tokens include cache writes and
    cache reads, because all of them count against the subscription. The
    model name comes from the `system/init` event, else the first key of the
    result's `modelUsage`, else "claude".
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
    tokens_in: int | None = None
    tokens_out: int | None = None
    usage = result.get("usage")
    if isinstance(usage, dict):
        tokens_in = _sum(
            usage, "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"
        )
        tokens_out = _int(usage.get("output_tokens"))
    text = result.get("result")
    if result.get("is_error") is not False or result.get("subtype") != "success":
        detail = text if isinstance(text, str) and text else raw
        raise ModelUnavailable(
            f"claude: error result ({result.get('subtype')}): {_tail(detail)}",
            tokens_in=tokens_in, tokens_out=tokens_out,
        )
    if not isinstance(text, str) or not text:
        raise ModelUnavailable(
            f"claude: result has no reply text: {_tail(raw)}",
            tokens_in=tokens_in, tokens_out=tokens_out,
        )
    if _CLAUDE_LIMIT.match(text):
        raise ModelUnavailable(
            f"claude: usage limit: {_tail(text)}", tokens_in=tokens_in, tokens_out=tokens_out
        )
    model = init.get("model")
    if not isinstance(model, str):
        model_usage = result.get("modelUsage")
        model = next(iter(model_usage), None) if isinstance(model_usage, dict) else None
    return _reply(text, tokens_in, tokens_out, model or "claude", prompt)


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
        return _run_and_parse(
            self.argv(), lambda out: parse_claude(out, prompt=prompt),
            stdin=prompt, cwd=cwd, env=env, timeout_s=timeout_s,
        )


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

    The answer is the last non-empty `agent_message` item. Lines that are not
    JSON objects are skipped. A `turn.failed` event is fatal. An `error` event
    is fatal only if no `turn.completed` follows it, because Codex reports
    errors it recovers from (such as "Reconnecting...") the same way.

    Tokens come from the `turn.completed` events. Codex counts cached input
    inside `input_tokens` and reasoning inside `output_tokens`. Any part of
    `total_tokens` that the input and output counts do not cover is charged as
    input, as in `parse_agy`.
    """
    text: str | None = None
    tokens_in: int | None = 0
    tokens_out: int | None = 0
    usage_seen = False
    pending_error: str | None = None

    def reported() -> tuple[int | None, int | None]:
        return (tokens_in, tokens_out) if usage_seen else (None, None)

    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            parsed: JsonValue = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        event = parsed
        kind = event.get("type")
        if kind in ("turn.failed", "error"):
            error = _obj(event.get("error"))
            message = str(error.get("message", event.get("message", line)))
            if kind == "turn.failed":
                got_in, got_out = reported()
                raise ModelUnavailable(f"codex: {message}", tokens_in=got_in, tokens_out=got_out)
            pending_error = message
        elif kind == "item.completed":
            item = _obj(event.get("item"))
            item_text = item.get("text")
            if item.get("type") == "agent_message" and isinstance(item_text, str) and item_text:
                text = item_text
        elif kind == "turn.completed":
            pending_error = None
            usage = event.get("usage")
            if isinstance(usage, dict):
                usage_seen = True
                part_in = _int(usage.get("input_tokens"))
                part_out = _int(usage.get("output_tokens"))
                total = _int(usage.get("total_tokens"))
                if part_in is not None and part_out is not None and total is not None:
                    part_in += max(0, total - part_in - part_out)
                tokens_in = None if tokens_in is None or part_in is None else tokens_in + part_in
                tokens_out = (
                    None if tokens_out is None or part_out is None else tokens_out + part_out
                )
    got_in, got_out = reported()
    if pending_error is not None:
        raise ModelUnavailable(f"codex: {pending_error}", tokens_in=got_in, tokens_out=got_out)
    if text is None:
        raise ModelUnavailable(
            f"codex: no agent message: {_tail(raw)}", tokens_in=got_in, tokens_out=got_out
        )
    return _reply(text, got_in, got_out, model, prompt)


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
        model = self.model or "codex-default"
        return _run_and_parse(
            self.argv(), lambda out: parse_codex(out, model=model, prompt=prompt),
            stdin=prompt, cwd=cwd, env=env, timeout_s=timeout_s,
        )


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
"""Session-only agy settings (agy 1.5.16). Never saved to disk.

They deny file, URL, command, and MCP actions. They do not cover agy's
built-in `search_web` tool, nor its browser or subagent tools: the probe of
2026-11-24 found that Gemini still searches the web with these settings. So
Gemini is "normal runs only" (see `src/minilab/models/isolation.json`;
follow-up bead minilab-2be43f).
"""


def parse_agy(raw: str, *, model: str, prompt: str = "") -> ModelReply:
    """Parse `agy -p --output-format json` output: one JSON object.

    On agy 1.5.16, `output_tokens` already includes `thinking_tokens`, and
    `total_tokens` is input plus output. Any part of the total that the
    input and output counts do not cover is charged as input, so no spend is
    lost if that changes.
    """
    value = _obj(_load_json(raw, "agy"))
    tokens_in: int | None = None
    tokens_out: int | None = None
    usage = value.get("usage")
    if isinstance(usage, dict):
        tokens_in = _int(usage.get("input_tokens"))
        tokens_out = _int(usage.get("output_tokens"))
        total = _int(usage.get("total_tokens"))
        if tokens_in is not None and tokens_out is not None and total is not None:
            tokens_in += max(0, total - tokens_in - tokens_out)
    status = value.get("status")
    text = value.get("response")
    if status != "SUCCESS":
        raise ModelUnavailable(
            f"agy: status {status}: {value.get('error', _tail(raw))}",
            tokens_in=tokens_in, tokens_out=tokens_out,
        )
    if not isinstance(text, str) or not text:
        raise ModelUnavailable(
            f"agy: no reply text: {_tail(raw)}", tokens_in=tokens_in, tokens_out=tokens_out
        )
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
        """Run agy. A prompt over `MAX_ARG_BYTES` raises `ModelNotStarted`; so does a
        prompt with a NUL character or one that is not valid text (see `run_command`)."""
        size = len(prompt.encode("utf-8", errors="replace"))
        if size > MAX_ARG_BYTES:
            raise ModelNotStarted(
                f"agy: prompt is {size} bytes; the limit for one "
                f"command-line argument is {MAX_ARG_BYTES}"
            )
        model = self.model or "agy-default"
        return _run_and_parse(
            self.argv(prompt), lambda out: parse_agy(out, model=model, prompt=prompt),
            stdin="", cwd=cwd, env=env, timeout_s=timeout_s,
        )


# --- Local (ollama) -------------------------------------------------------------------------


class _OllamaReply(BaseModel):
    model_config = ConfigDict(extra="ignore")

    response: str
    model: str
    prompt_eval_count: int | None = Field(default=None, ge=0)
    eval_count: int | None = Field(default=None, ge=0)


class LocalBackend:
    """The local weak model through ollama's HTTP API. It has no tools to isolate.

    The HTTP client does not read proxy settings from the environment
    (`trust_env=False`). `transport` replaces the network, for tests.
    """

    family = Family.local

    def __init__(
        self,
        model: str = "qwen3.5:4b",
        url: str = "http://127.0.0.1:11434",
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.url = url
        self.transport = transport

    def run(
        self, prompt: str, *, cwd: Path, env: Mapping[str, str], timeout_s: float
    ) -> ModelReply:
        try:
            with httpx.Client(trust_env=False, transport=self.transport) as client:
                response = client.post(
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
