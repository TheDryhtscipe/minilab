"""Audit what a Codex call sends to the model.

`scripts/codex_sink.py` captures the request: it runs `codex exec` with the
backend's own flags against a local HTTP server instead of OpenAI, so no
model runs and nothing is spent (it lives outside `src`, which holds no
network code). `audit_codex_request` then checks that body:
no user rules (`AGENTS.md`), no skill catalog, no multi-agent prompts, no path
under the user's home, and no tool outside a fixed list.

The list is pinned on purpose (user decision 2026-11-28, bead minilab-811a75).
codex-cli 0.172 gives the model a code-mode `exec` tool, a JavaScript isolate
with no file system or network, and `request_user_input` tools that get no
answer in `codex exec`. The model catalog sets these, not a feature flag. They
cannot bring outside information into a call, so they are allowed. Any other
tool the request lists, and any change in the request's shape, fails the audit.
A nested tool that the exec description does not list is out of its sight.
"""

import re
from pathlib import Path

from pydantic import JsonValue

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
    ("<INSTRUCTIONS>", "an instructions file"),
)
"""Text that shows user context reached the model, with a name for the report."""

_NESTED = re.compile(r"### `([^`]+)`")


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


def user_markers(home: Path) -> list[tuple[str, str]]:
    """Text from the user's own codex context, to look for in a request.

    The longest lines of `~/.codex/AGENTS.md` and the name of each skill
    folder under `~/.codex/skills` and `~/.agents/skills`, so a leak is found
    by its content, whatever wrapper a later codex puts round it. Lines under
    40 characters and skill names under 8 are left out, being too common.
    """
    markers: list[tuple[str, str]] = []
    try:
        lines = (home / ".codex" / "AGENTS.md").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        lines = []
    for line in sorted({ln.strip() for ln in lines if len(ln.strip()) >= 40}, key=len)[-5:]:
        markers.append((line, "a line of ~/.codex/AGENTS.md"))
    for root in (home / ".codex" / "skills", home / ".agents" / "skills"):
        try:
            names = [p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]
        except OSError:
            names = []
        markers += [(f"{name}/SKILL.md", f"the skill {name}") for name in names
                    if len(name) >= 8]
    return markers


def audit_codex_request(body: JsonValue, *, home: Path) -> list[str]:
    """Return what is wrong with a captured request body; an empty list means it is clean.

    It fails closed: the request must have one `additional_tools` item with
    only the `functions` namespace, its tools must be exactly
    `CODEX_ALLOWED_TOOLS`, and the nested tools that `exec` lists must be
    exactly `CODEX_ALLOWED_NESTED_TOOLS`. A deferred nested tool that the exec
    description leaves out cannot be seen here. Text outside the tool
    definitions must hold none of `USER_CONTEXT_MARKERS`, no path under
    `home` or `~/`, and none of `user_markers(home)`.
    """
    if not isinstance(body, dict):
        return ["the request is not a JSON object"]
    problems: list[str] = []
    for tool in _tools_in(body):
        problems.append(f"tool not allowed: {_name(tool)}")
    items = body.get("input")
    if not isinstance(items, list) or not items:
        return [*problems, "the request has no input items"]
    tool_items = [it for it in items if isinstance(it, dict)
                  and it.get("type") == "additional_tools"]
    if len(tool_items) != 1:
        problems.append(f"expected one additional_tools item, found {len(tool_items)}")
    functions: set[str] = set()
    nested: set[str] | None = None
    for item in tool_items:
        spaces = _tools_in(item)
        for space in spaces:
            if _name(space) != "functions":
                problems.append(f"tool namespace not allowed: {_name(space)}")
                continue
            for tool in _tools_in(space):
                name = _name(tool)
                functions.add(name)
                if name not in CODEX_ALLOWED_TOOLS:
                    problems.append(f"tool not allowed: functions.{name}")
                if name == "exec" and isinstance(tool, dict):
                    description = tool.get("description")
                    nested = set(_NESTED.findall(description if isinstance(description, str)
                                                 else ""))
    missing = sorted(CODEX_ALLOWED_TOOLS - functions)
    if tool_items and missing:
        problems.append(f"expected tools missing (the request format may have changed): "
                        f"{', '.join(missing)}")
    if nested is not None:
        problems.extend(f"nested exec tool not allowed: {n}" for n in sorted(nested)
                        if n not in CODEX_ALLOWED_NESTED_TOOLS)
        gone = sorted(CODEX_ALLOWED_NESTED_TOOLS - nested)
        if gone:
            problems.append(f"expected nested exec tools missing (the exec description may "
                            f"have changed): {', '.join(gone)}")
    # Tool descriptions are vendor text; user context arrives in messages and instructions.
    context: list[str] = []
    for key, value in body.items():
        if key == "input" and isinstance(value, list):
            context += [t for it in value if not (isinstance(it, dict)
                        and it.get("type") == "additional_tools") for t in _texts(it)]
        elif key == "instructions":
            context += _texts(value)
    joined = "\n".join(context)
    for marker, label in (*USER_CONTEXT_MARKERS, *user_markers(home)):
        if marker in joined:
            problems.append(f"user context reached the model: {label}")
    if f"{home}/" in joined or "~/" in joined:
        problems.append(f"a path under the user's home reached the model: {home}")
    return problems
