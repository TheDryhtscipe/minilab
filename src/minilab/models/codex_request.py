"""See exactly what a Codex call sends to the model, without calling a model.

`capture_codex_request` runs `codex exec` with the backend's own flags, but
points it at a local HTTP server instead of OpenAI. The server records the
`/v1/responses` request body and answers every request with an error, so no
model runs and nothing is spent. `audit_codex_request` then checks that body:
no user rules (`AGENTS.md`), no skill catalog, no multi-agent prompts, no path
under the user's home, and no tool outside a fixed list.

The list is pinned on purpose (user decision 2026-11-28, bead minilab-811a75).
codex-cli 0.172 gives the model a code-mode `exec` tool, a JavaScript isolate
with no file system or network, and `request_user_input` tools that get no
answer in `codex exec`. The model catalog sets these, not a feature flag. They
cannot bring outside information into a call, so they are allowed. Any other
tool, including one a later codex version adds, fails the audit.
"""

import json
import re
import tempfile
import threading
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from pydantic import JsonValue

from minilab.models.backends import CodexBackend, ModelUnavailable

CODEX_ALLOWED_TOOLS: frozenset[str] = frozenset(
    {"exec", "wait", "request_user_input", "request_user_input_async",
     "send_message_to_user_async"}
)
"""Tools the model may see, all in the `functions` namespace (codex-cli 0.172)."""

CODEX_ALLOWED_NESTED_TOOLS: frozenset[str] = frozenset(
    {"apply_patch", "list_mcp_resource_templates", "list_mcp_resources",
     "read_mcp_resource", "clock__curr_time"}
)
"""Tools reachable from inside `exec`. `apply_patch` cannot write in the
read-only sandbox, and the MCP readers find no servers, because the user's
config is not loaded."""

USER_CONTEXT_MARKERS: tuple[tuple[str, str], ...] = (
    ("# AGENTS.md instructions", "AGENTS.md instructions"),
    ("<skills_instructions>", "the skills catalog"),
    # A path to one: codex's own instructions name SKILL.md in passing.
    ("/SKILL.md", "a skill file"),
    ("<multi_agent", "multi_agent prompts"),
)
"""Text that shows user context reached the model, with a name for the report."""

_NESTED = re.compile(r"### `([^`]+)`")
_SINK = "minilab_sink"


def _texts(value: JsonValue) -> list[str]:
    """Every string in a JSON value, keys excluded."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [text for item in value for text in _texts(item)]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _texts(item)]
    return []


def _name(value: JsonValue) -> str:
    if isinstance(value, dict):
        name = value.get("name")
        if isinstance(name, str):
            return name
        kind = value.get("type")
        if isinstance(kind, str):
            return kind
    return "?"


def _tools_in(value: JsonValue) -> list[JsonValue]:
    if isinstance(value, dict):
        tools = value.get("tools")
        if isinstance(tools, list):
            return tools
    return []


def audit_codex_request(body: JsonValue, *, home: Path) -> list[str]:
    """Return what is wrong with a captured request body; an empty list means it is clean."""
    if not isinstance(body, dict):
        return ["the request is not a JSON object"]
    problems: list[str] = []
    for tool in _tools_in(body):
        problems.append(f"tool not allowed: {_name(tool)}")
    items = body.get("input")
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and item.get("type") == "additional_tools":
            for space in _tools_in(item):
                if _name(space) != "functions":
                    problems.append(f"tool namespace not allowed: {_name(space)}")
                    continue
                for tool in _tools_in(space):
                    name = _name(tool)
                    if name not in CODEX_ALLOWED_TOOLS:
                        problems.append(f"tool not allowed: functions.{name}")
                    if name == "exec" and isinstance(tool, dict):
                        description = tool.get("description")
                        nested = _NESTED.findall(description if isinstance(description, str)
                                                 else "")
                        problems.extend(f"nested exec tool not allowed: {n}" for n in nested
                                        if n not in CODEX_ALLOWED_NESTED_TOOLS)
    # Tool descriptions are vendor text; user context arrives in messages and instructions.
    context: list[str] = []
    for key, value in body.items():
        if key == "input" and isinstance(value, list):
            context += [t for it in value if not (isinstance(it, dict)
                        and it.get("type") == "additional_tools") for t in _texts(it)]
        elif key == "instructions":
            context += _texts(value)
    joined = "\n".join(context)
    for marker, label in USER_CONTEXT_MARKERS:
        if marker in joined:
            problems.append(f"user context reached the model: {label}")
    if f"{home}/" in joined:
        problems.append(f"a path under the user's home reached the model: {home}")
    return problems


class _Sink(BaseHTTPRequestHandler):
    bodies: list[bytes]

    def _refuse(self) -> None:
        payload = b'{"error":{"message":"minilab request sink","type":"invalid_request_error"}}'
        self.send_response(400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        if self.path.endswith("/responses"):
            self.bodies.append(body)
        self._refuse()

    def do_GET(self) -> None:
        self._refuse()

    def log_message(self, format: str, *args: object) -> None:
        return


def capture_codex_request(
    *, model: str, env: Mapping[str, str], timeout_s: float, binary: str = "codex"
) -> JsonValue:
    """Run one Codex call against a local sink and return the request body it sent.

    The call uses `CodexBackend`'s own flags and per-call home, plus a provider
    override that points it at the sink. Raise RuntimeError if codex sent no
    request.
    """
    bodies: list[bytes] = []
    handler = type("Sink", (_Sink,), {"bodies": bodies})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    provider = (
        f'model_providers.{_SINK}={{name="{_SINK}",base_url="http://127.0.0.1:{port}/v1",'
        'wire_api="responses",request_max_retries=0,stream_max_retries=0}'
    )
    backend = CodexBackend(
        model=model, binary=binary, config=(provider, f'model_provider="{_SINK}"')
    )
    error = "codex sent no request"
    try:
        with tempfile.TemporaryDirectory(prefix="minilab-codex-sink-") as cwd:
            try:
                backend.run("Reply with OK.", cwd=Path(cwd), env=env, timeout_s=timeout_s)
            except ModelUnavailable as exc:
                error = str(exc)
    finally:
        server.shutdown()
        server.server_close()
    if not bodies:
        raise RuntimeError(f"{error}")
    value: JsonValue = json.loads(bodies[0])
    return value
