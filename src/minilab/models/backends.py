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
  no MCP servers), a read-only sandbox, web search disabled, every tool
  feature that reaches outside the prompt turned off, including the shell, no
  multi-agent tools or skill catalog, and an empty `HOME` and `CODEX_HOME` per
  call holding only a copy of the login, so the user's `AGENTS.md` and skills
  in `~/.codex` and `~/.agents` do not load, nor does a project `AGENTS.md`.
  A system-wide `/etc/codex` would still load. What codex then sends is checked by
  `minilab.models.codex_request` (bead minilab-811a75).
- Gemini (Antigravity CLI, `agy --input-format stream-json`): a session-only
  `--settings` object that denies file, URL, command, and MCP actions, keeps
  the agent inside its empty workspace, and never spends paid credits; a
  `PreToolUse` hook, written into the call's empty directory, that denies every
  tool, `search_web` and the browser, subagent, and workflow tools included; an
  empty `HOME`, so none of the user's agy rules, memories, skills, or hooks
  load; and a parser that fails any call without proof that the hook denied
  every tool step (bead minilab-2be43f).
- Local (ollama HTTP API): the model has no tools. The HTTP client ignores
  proxy variables in the environment.

Whether each tool really is isolated is checked live by
`scripts/probe_isolation.py`, not assumed from these flags.
"""

import fcntl
import json
import math
import os
import re
import secrets
import selectors
import signal
import subprocess
import time
import warnings
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

MAX_OUTPUT_BYTES = 16 * 2**20
"""Most bytes of standard output one tool call may print. Past this the call fails.

A reply of 16 MiB is about 4 million tokens, far more than any model writes in
one call. Peak memory for one call is a few times this (the bytes, then the
decoded text)."""

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


class ModelUnavailable(Exception):
    """The tool failed, timed out, or hit a usage limit. No reply exists.

    `tokens_in` and `tokens_out` are the counts the tool reported for the
    failed call, or None where it reported none.

    `output` is what the tool printed (or ollama answered) when it ended the
    call itself: it exited, or answered with an error or with output that does
    not parse. It is None when the call was cut off by a timeout, a signal, or
    the output cap, since the tool may have spent more than it printed. Where
    the tool reported no output count, the adapter charges an estimate of
    `output` in place of the call's whole output reservation.
    """

    def __init__(
        self,
        message: str,
        *,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        output: str | None = None,
    ) -> None:
        super().__init__(message)
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.output = output


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
    tools_denied: int = Field(default=0, ge=0)
    """Tool calls the tool's own guard denied during the call (agy only)."""


class Backend(Protocol):
    """One model family's tool."""

    @property
    def family(self) -> Family: ...

    @property
    def model(self) -> str:
        """The model id the backend asks its tool for (`--model`, or ollama's `model`).

        A reply's `model` is what the tool reported where it reports one (Claude,
        ollama), else this id."""
        ...

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
    """The tool printed more than `MAX_OUTPUT_BYTES` bytes, so the call was stopped.

    `tokens_out` estimates the output read so far, at `CHARS_PER_TOKEN` bytes
    per token, so the adapter charges it.
    """


class ToolExit(ModelUnavailable):
    """The tool started and exited non-zero. `stdout` is what it printed.

    `returncode` is negative when a signal killed the tool.
    """

    def __init__(self, message: str, *, stdout: str, returncode: int) -> None:
        super().__init__(message)
        self.stdout = stdout
        self.returncode = returncode


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
    `ModelUnavailable`) is raised, carrying an estimate of the output tokens
    read. Only the last `STDERR_KEEP_BYTES` of stderr are kept.
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
            err = exc.err.decode("utf-8", errors="replace")
            raise OutputTooLarge(
                f"{tool}: printed more than {max_out} bytes; stopped: {_tail(err)}",
                tokens_out=math.ceil(exc.read / CHARS_PER_TOKEN),
            ) from exc
    finally:
        _killpg(proc.pid)
        if proc.returncode is None:
            _drain_then_close(proc, read=False)
    out = out_b.decode("utf-8", errors="replace")
    err = err_b.decode("utf-8", errors="replace")
    code = proc.wait()  # Already reaped; this only reads the exit status.
    if code != 0:
        raise ToolExit(
            f"{tool}: exit {code}: {_tail(err) or _tail(out)}", stdout=out, returncode=code
        )
    return out


class _Overflow(Exception):
    """Standard output passed its cap. `read` is the bytes of it read; `err` is stderr's tail."""

    def __init__(self, read: int, err: bytes) -> None:
        super().__init__(read)
        self.read = read
        self.err = err


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
                        raise _Overflow(len(out), bytes(err))
                else:
                    err += chunk
                    if len(err) > STDERR_KEEP_BYTES:
                        del err[: len(err) - STDERR_KEEP_BYTES]
    # The pipes closed, so the tool is exiting: allow it a moment even at the deadline.
    left = deadline - time.monotonic()
    try:
        proc.wait(timeout=max(left, 0.1))
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
    """Run the tool and parse its output. On a non-zero exit, keep any usage it reported.

    When the tool ended the call itself (it exited, but not by a signal), the
    `ModelUnavailable` raised carries what it printed as `output`.
    """
    try:
        out = run_command(argv, stdin=stdin, cwd=cwd, env=env, timeout_s=timeout_s)
    except ToolExit as exc:
        try:
            reply = parse(exc.stdout)
        except ModelUnavailable as parsed:
            tokens_in, tokens_out = parsed.tokens_in, parsed.tokens_out
            message = f"{exc}; {parsed}"
        else:
            tokens_in = None if reply.estimated else reply.tokens_in
            tokens_out = None if reply.estimated else reply.tokens_out
            message = str(exc)
        raise ModelUnavailable(
            message,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            output=exc.stdout if exc.returncode > 0 else None,
        ) from exc
    try:
        return parse(out)
    except ModelUnavailable as exc:
        exc.output = out
        raise


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

    def __init__(self, *, model: str, binary: str = "claude") -> None:
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
        return argv + ["--model", self.model]

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
    "goals",
    "memories",
    "sleep_tool",
)
"""Codex features that give the model a way out of the prompt, or tools and
context it does not need (checked on codex-cli 0.172)."""

CODEX_CONFIG: tuple[str, ...] = (
    'web_search="disabled"',
    "agents.enabled=false",
    "skills.include_instructions=false",
    "skills.bundled.enabled=false",
    'model_reasoning_effort="medium"',
    "project_doc_max_bytes=0",
    "project_root_markers=[]",
)
"""`-c` overrides for every call. They turn off the multi-agent tools and
prompts and the skill catalog, which no feature flag covers (bead
minilab-811a75), and a project `AGENTS.md` found by walking up from the
working directory (as when `TMPDIR` is inside a repository).
`--ignore-user-config` leaves the effort at the model's
default, `low`, so the effort is set here (user decision 2026-11-28: medium to
start)."""

_LOGIN_LOCK = ".minilab-auth.lock"


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

    def __init__(
        self, *, model: str, binary: str = "codex", config: Sequence[str] = ()
    ) -> None:
        self.model = model
        self.binary = binary
        self.config = tuple(config)
        """Extra `-c key=value` overrides, after the isolation flags. Only the
        request sink (`minilab.models.codex_request`) passes any."""

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
        ]
        for override in CODEX_CONFIG:
            argv += ["-c", override]
        for feature in CODEX_DISABLED_FEATURES:
            argv += ["--disable", feature]
        for override in self.config:
            argv += ["-c", override]
        return argv + ["--model", self.model, "-"]

    def run(
        self, prompt: str, *, cwd: Path, env: Mapping[str, str], timeout_s: float
    ) -> ModelReply:
        """Run codex in `cwd`, the call's own empty directory, with a home of its own.

        Codex reads the user's `AGENTS.md` and skills from `CODEX_HOME` and
        `HOME` even with `--ignore-user-config`. So the backend makes empty
        folders `cwd/.home` (for `HOME` and the XDG folders) and
        `cwd/.codex-home` (for `CODEX_HOME`), and copies in only the user's
        login, `~/.codex/auth.json`. If codex refreshes the login during the
        call, the new login is written back afterwards, even if the call
        failed, unless the user's file changed in the meantime: then that
        newer login is kept (user decision 2026-11-28). A lock file next to
        the login lets only one minilab codex call run at a time. A
        write-back problem never replaces the call's result: it is a warning
        on a reply and a note on an error. Raise `ModelNotStarted` if there is
        no login or the folders cannot be made.
        """
        home = Path(env.get("HOME") or Path.home())
        login = home / ".codex" / "auth.json"
        own_home = cwd / ".home"
        codex_home = cwd / ".codex-home"
        copy = codex_home / "auth.json"
        own_env = {
            **env,
            "HOME": str(own_home),
            "CODEX_HOME": str(codex_home),
            "XDG_CONFIG_HOME": str(own_home / ".config"),
            "XDG_DATA_HOME": str(own_home / ".local" / "share"),
            "XDG_CACHE_HOME": str(own_home / ".cache"),
            "XDG_STATE_HOME": str(own_home / ".local" / "state"),
        }
        try:
            lock = open(login.parent / _LOGIN_LOCK, "a")
        except OSError as exc:
            raise ModelNotStarted(f"codex: cannot lock the login at {login}: {exc}") from exc
        with lock:
            # One minilab codex call at a time per login: two calls that refresh
            # the same rotating token could get the whole login revoked.
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                original = login.read_bytes()
            except OSError as exc:
                raise ModelNotStarted(f"codex: cannot read the login at {login}: {exc}") from exc
            try:
                own_home.mkdir(exist_ok=True)
                codex_home.mkdir(mode=0o700, exist_ok=True)
                _write_private(copy, original)
            except OSError as exc:
                raise ModelNotStarted(f"codex: cannot make the call's home: {exc}") from exc
            try:
                reply = _run_and_parse(
                    self.argv(), lambda out: parse_codex(out, model=self.model, prompt=prompt),
                    stdin=prompt, cwd=cwd, env=own_env, timeout_s=timeout_s,
                )
            except BaseException as exc:
                problem = _write_back_login(login, copy, original)
                if problem is not None:
                    exc.add_note(problem)
                raise
            problem = _write_back_login(login, copy, original)
            if problem is not None:
                warnings.warn(problem, stacklevel=2)
            return reply


def _write_private(path: Path, data: bytes) -> None:
    """Write `data` to `path`, readable only by the user, and flush it to disk."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as out:
        out.write(data)
        out.flush()
        os.fsync(out.fileno())


def _is_login(data: bytes) -> bool:
    """Whether `data` parses as a codex login: a JSON object with a `tokens` object."""
    try:
        value: JsonValue = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    return isinstance(value, dict) and isinstance(value.get("tokens"), dict)


def _write_back_login(login: Path, copy: Path, original: bytes) -> str | None:
    """Put a login codex refreshed during the call back in the user's file.

    The caller holds the login lock. Do nothing if the copy is unchanged or
    unreadable, if it is not a whole login (a call killed mid-write), or if
    the user's file no longer holds `original` (a codex outside minilab
    refreshed it, and that login is newer). The swap is atomic. Return a
    message for the caller to report when the write-back was refused or failed;
    it never raises, so the call's own result or error stands.
    """
    try:
        refreshed = copy.read_bytes()
    except OSError:
        return None
    if refreshed == original:
        return None
    if not _is_login(refreshed):
        return f"codex: the refreshed login in {copy} is not a whole login; kept {login}"
    staged = login.with_name(f".auth.json.minilab-{secrets.token_hex(8)}")
    try:
        if login.read_bytes() != original:
            return f"codex: {login} changed during the call; kept it, not the call's login"
        _write_private(staged, refreshed)
        os.replace(staged, login)
    except OSError as exc:
        staged.unlink(missing_ok=True)
        return f"codex: could not write the refreshed login back to {login}: {exc}"
    return None


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
2026-11-24 found that Gemini still searched the web with these settings
alone. The hook from `agy_hooks` blocks those.
"""

AGY_DENIED = "tool call denied by pre-tool hook: "
"""How agy 1.5.16 starts the error of a tool step that a `PreToolUse` hook denied."""


def agy_hooks(nonce: str) -> dict[str, JsonValue]:
    """agy hooks that deny every tool call, written to `.agents/hooks.json` in the call's
    empty working directory. `nonce` goes into the deny reason, so `parse_agy` can tell
    this call's denial from any other error.

    A `PreToolUse` hook with matcher `*` runs before every tool, built-in ones
    such as `search_web`, `run_workflow`, and the browser and subagent tools
    included, and its `deny` stops the tool. On agy 1.5.16, observed live: a
    hook that fails, times out, or prints something that is not JSON also
    stops the tool, and an `allow` from another hook does not override a deny.
    """
    reason = json.dumps({"decision": "deny", "reason": f"This session has no tools. {nonce}"})
    return {
        "minilab-deny-all-tools": {
            "PreToolUse": [
                {
                    "matcher": "*",
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"cat >/dev/null; printf '%s\\n' '{reason}'",
                            "timeout": 10,
                        }
                    ],
                }
            ]
        }
    }


_AGY_STEPS_WITHOUT_TOOLS = frozenset({"user_input", "agent_response", "system_message"})
"""Step types that run no tool. A `system_message` is text agy adds itself; with
the call's own empty HOME, no user hook or setting can add one."""

_AGY_REPOSITORY_MARKERS = (".git", ".jj", ".hg")
"""agy reads rules, skills, hooks, and MCP servers from every folder between the
working directory and the repository root, so the call must not sit in a repository."""


def _agy_tools_not_blocked(
    events: Sequence[dict[str, JsonValue]], *, conversation: JsonValue, nonce: str
) -> list[str]:
    """Each step that may have run a tool or reached outside the call.

    A `tool` step must end in state `ERROR` with this call's hook denial as its
    error. A `subagent` step must end in `ERROR`: agy 1.5.16 reports no error
    text for it. A step of another conversation, a step with no index, and a
    step of a type not known to run no tool are all reported.
    """
    last: dict[int, dict[str, JsonValue]] = {}
    problems: list[str] = []
    for event in events:
        step = _obj(event.get("step_update"))
        step_type = step.get("step_type")
        if step.get("conversation_id") != conversation:
            problems.append(f"a step of another conversation ({step.get('conversation_id')})")
            continue
        if step_type in _AGY_STEPS_WITHOUT_TOOLS:
            continue
        if step_type not in ("tool", "subagent"):
            problems.append(
                f"unknown step type {step_type!r} (add it to _AGY_STEPS_WITHOUT_TOOLS "
                "only if it runs no tool)"
            )
            continue
        index = _int(step.get("step_index"))
        if index is None:
            problems.append(f"{step.get('tool_name') or '<unnamed>'} (no step index)")
            continue
        last[index] = step
    for step in last.values():
        name = step.get("tool_name") or "<unnamed>"
        state = step.get("state")
        if state != "ERROR":
            problems.append(f"{name} ({state})")
            continue
        if step.get("step_type") == "tool":
            message = _obj(_obj(step.get("tool_info")).get("error")).get("message")
            if not (isinstance(message, str) and message.startswith(AGY_DENIED)
                    and nonce in message):
                problems.append(f"{name} (an error that is not this call's hook denial)")
    return problems


def parse_agy(raw: str, *, model: str, nonce: str, prompt: str = "") -> ModelReply:
    """Parse `agy --output-format stream-json` output: one JSON event per line.

    The events are one `init`, a `step_update` for each step, and one
    `result`. The parser raises `ModelUnavailable` for a call in which a step
    may have run a tool (see `_agy_tools_not_blocked`; `nonce` is the one in
    this call's hook) and for an event of an unknown kind. The hook is what
    blocks the tools; this checks every call for proof that it did.
    `tools_denied` on the reply counts the tool steps the hook denied.

    On agy 1.5.16, `output_tokens` already includes `thinking_tokens`, and
    `total_tokens` is input plus output. Any part of the total that the
    input and output counts do not cover is charged as input, so no spend is
    lost if that changes.
    """
    steps: list[dict[str, JsonValue]] = []
    init: dict[str, JsonValue] | None = None
    result: dict[str, JsonValue] | None = None
    for line in raw.splitlines():
        if not line.strip():
            continue
        event = _obj(_load_json(line, "agy"))
        kind = event.get("event")
        if kind == "step_update":
            steps.append(event)
        elif kind == "result":
            if result is not None:
                raise ModelUnavailable(f"agy: more than one result event: {_tail(raw)}")
            result = _obj(event.get("result"))
        elif kind == "init":
            if init is not None:
                raise ModelUnavailable(f"agy: more than one init event: {_tail(raw)}")
            init = event
        else:
            raise ModelUnavailable(f"agy: unexpected event {kind!r}: {_tail(line)}")
    if result is None:
        raise ModelUnavailable(f"agy: no result event: {_tail(raw)}")
    tokens_in: int | None = None
    tokens_out: int | None = None
    usage = result.get("usage")
    if isinstance(usage, dict):
        tokens_in = _int(usage.get("input_tokens"))
        tokens_out = _int(usage.get("output_tokens"))
        total = _int(usage.get("total_tokens"))
        if tokens_in is not None and tokens_out is not None and total is not None:
            tokens_in += max(0, total - tokens_in - tokens_out)
    conversation = None if init is None else init.get("conversation_id")
    if not isinstance(conversation, str) or not conversation:
        raise ModelUnavailable(
            f"agy: no init event with a conversation id: {_tail(raw)}",
            tokens_in=tokens_in, tokens_out=tokens_out,
        )
    not_blocked = _agy_tools_not_blocked(steps, conversation=conversation, nonce=nonce)
    if not_blocked:
        raise ModelUnavailable(
            f"agy: a step was not blocked: {', '.join(not_blocked)}",
            tokens_in=tokens_in, tokens_out=tokens_out,
        )
    status = result.get("status")
    text = result.get("response")
    if status != "SUCCESS":
        raise ModelUnavailable(
            f"agy: status {status}: {result.get('error', _tail(raw))}",
            tokens_in=tokens_in, tokens_out=tokens_out,
        )
    if not isinstance(text, str) or not text:
        raise ModelUnavailable(
            f"agy: no reply text: {_tail(raw)}", tokens_in=tokens_in, tokens_out=tokens_out
        )
    denied = {
        _int(_obj(event.get("step_update")).get("step_index"))
        for event in steps
        if _obj(event.get("step_update")).get("step_type") == "tool"
    }
    reply = _reply(text, tokens_in, tokens_out, model, prompt)
    return reply.model_copy(update={"tools_denied": len(denied)})


class GeminiBackend:
    family = Family.gemini

    def __init__(self, *, model: str, binary: str | None = None) -> None:
        self.model = model
        self.binary = binary or str(Path.home() / ".local" / "bin" / "agy")

    def argv(self) -> list[str]:
        """The prompt goes on stdin (`--input-format stream-json`), not on the command
        line, so it has no 128 KiB argument limit and other local users cannot read it
        in the process list."""
        argv = [
            self.binary,
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--disable-slash-commands",
            "--settings", json.dumps(AGY_SETTINGS),
        ]
        return argv + ["--model", self.model]

    def run(
        self, prompt: str, *, cwd: Path, env: Mapping[str, str], timeout_s: float
    ) -> ModelReply:
        """Run agy in `cwd`, which must be the call's own empty directory outside any
        repository.

        Before agy starts, the backend writes `agy_hooks` into `cwd/.agents/` and
        makes an empty home folder, `cwd/.home`, for `HOME` and the XDG folders. So
        agy loads none of the user's rules, memories, skills, hooks, plugins, or
        MCP servers, and keeps no record of the call in the user's own home: its
        transcript stays in `cwd/.home` until the caller removes `cwd`
        (`isolated_run` does). agy still finds the user's login in the OS keyring
        over D-Bus. The prompt goes to agy on stdin as
        one stream-json user message. It raises `ModelNotStarted` for an empty or
        blank prompt, one that is not valid text, a `cwd` inside a repository, or a
        hook or home folder that cannot be written.
        """
        if not prompt.strip():
            raise ModelNotStarted("agy: the prompt is empty")
        try:
            prompt.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ModelNotStarted(f"agy: the prompt is not valid text: {exc}") from exc
        # One user message on one line: json.dumps escapes every line break.
        message = json.dumps({"event": "user", "message": {"content": prompt}}) + "\n"
        here = cwd.resolve()
        for folder in (here, *here.parents):
            for marker in _AGY_REPOSITORY_MARKERS:
                if (folder / marker).exists():
                    raise ModelNotStarted(
                        f"agy: the working directory is inside a repository ({folder / marker})"
                    )
        nonce = secrets.token_hex(16)
        home = cwd / ".home"
        try:
            agents = cwd / ".agents"
            agents.mkdir(exist_ok=True)
            (agents / "hooks.json").write_text(json.dumps(agy_hooks(nonce)))
            home.mkdir(exist_ok=True)
        except OSError as exc:
            raise ModelNotStarted(f"agy: cannot write the deny-all hook: {exc}") from exc
        own_env = {
            **env,
            "HOME": str(home),
            "XDG_CONFIG_HOME": str(home / ".config"),
            "XDG_DATA_HOME": str(home / ".local" / "share"),
            "XDG_CACHE_HOME": str(home / ".cache"),
            "XDG_STATE_HOME": str(home / ".local" / "state"),
        }
        return _run_and_parse(
            self.argv(),
            lambda out: parse_agy(out, model=self.model, nonce=nonce, prompt=prompt),
            stdin=message, cwd=cwd, env=own_env, timeout_s=timeout_s,
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
        *,
        model: str,
        url: str = "http://127.0.0.1:11434",
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
        except httpx.TimeoutException as exc:
            raise ModelUnavailable(f"ollama: timed out after {timeout_s} s") from exc
        except httpx.HTTPError as exc:
            raise ModelUnavailable(f"ollama: {exc}") from exc
        try:
            response.raise_for_status()
            reply = _OllamaReply.model_validate_json(response.content)
        except (httpx.HTTPError, ValidationError) as exc:
            raise ModelUnavailable(f"ollama: {exc}", output=response.text) from exc
        return _reply(reply.response, reply.prompt_eval_count, reply.eval_count, reply.model,
                      prompt)
