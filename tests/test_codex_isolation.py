"""Codex isolation: its own home per call, the login copied in and back, and the request audit.

Bead minilab-811a75. The sink test runs the real `codex` binary against a local
HTTP server and never reaches a model; it is marked live because codex may
still contact other hosts on its own.
"""

import json
import os
import shutil
import stat
import sys
from pathlib import Path

import pytest
from pydantic import JsonValue

from minilab.models.adapter import ModelNotStarted, ModelUnavailable, scrubbed_env
from minilab.models.backends import CodexBackend
from minilab.models.codex_request import (
    CODEX_ALLOWED_NESTED_TOOLS,
    CODEX_ALLOWED_TOOLS,
    audit_codex_request,
    capture_codex_request,
)

_REPLY = "\n".join(
    json.dumps(event)
    for event in (
        {"type": "item.completed", "item": {"type": "agent_message", "text": "answer"}},
        {"type": "turn.completed", "usage": {"input_tokens": 20, "output_tokens": 9}},
    )
)


def _fake_codex(tmp_path: Path, body: str) -> Path:
    """A stand-in `codex` that runs `body` (Python) and then prints a good reply."""
    script = tmp_path / "fake-codex"
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "sys.stdin.read()\n"
        f"{body}\n"
        f"print({_REPLY!r})\n"
    )
    script.chmod(0o755)
    return script


def _real_home(tmp_path: Path, login: str = '{"tokens": "old"}') -> Path:
    home = tmp_path / "realhome"
    (home / ".codex").mkdir(parents=True)
    auth = home / ".codex" / "auth.json"
    auth.write_text(login)
    auth.chmod(0o600)
    (home / ".codex" / "AGENTS.md").write_text("user rules\n")
    return home


def _call_dir(tmp_path: Path) -> Path:
    cwd = tmp_path / "call"
    cwd.mkdir()
    return cwd


def test_codex_runs_with_its_own_home_holding_only_the_login(tmp_path: Path) -> None:
    home = _real_home(tmp_path)
    seen = tmp_path / "seen.json"
    binary = _fake_codex(
        tmp_path,
        "ch = pathlib.Path(os.environ['CODEX_HOME'])\n"
        f"pathlib.Path({str(seen)!r}).write_text(json.dumps({{\n"
        "  'home': os.environ['HOME'], 'codex_home': str(ch),\n"
        "  'xdg': os.environ.get('XDG_CONFIG_HOME'),\n"
        "  'files': sorted(p.name for p in ch.iterdir()),\n"
        "  'home_files': sorted(p.name for p in pathlib.Path(os.environ['HOME']).iterdir()),\n"
        "  'auth': (ch / 'auth.json').read_text()}))",
    )
    cwd = _call_dir(tmp_path)
    backend = CodexBackend(model="gpt-test", binary=str(binary))
    reply = backend.run("hi", cwd=cwd, env={"HOME": str(home), "PATH": os.environ["PATH"]},
                        timeout_s=30)
    assert reply.text == "answer"
    got = json.loads(seen.read_text())
    assert Path(got["codex_home"]).parent == cwd
    assert Path(got["home"]).parent == cwd
    assert Path(got["xdg"]).is_relative_to(Path(got["home"]))
    assert got["files"] == ["auth.json"]
    assert got["home_files"] == []
    assert got["auth"] == '{"tokens": "old"}'


def test_a_refreshed_login_is_written_back(tmp_path: Path) -> None:
    home = _real_home(tmp_path)
    binary = _fake_codex(
        tmp_path,
        "(pathlib.Path(os.environ['CODEX_HOME']) / 'auth.json')"
        ".write_text('{\"tokens\": \"new\"}')",
    )
    backend = CodexBackend(model="gpt-test", binary=str(binary))
    backend.run("hi", cwd=_call_dir(tmp_path), env={"HOME": str(home),
                "PATH": os.environ["PATH"]}, timeout_s=30)
    auth = home / ".codex" / "auth.json"
    assert auth.read_text() == '{"tokens": "new"}'
    assert stat.S_IMODE(auth.stat().st_mode) == 0o600


def test_a_refreshed_login_is_written_back_after_a_failed_call(tmp_path: Path) -> None:
    home = _real_home(tmp_path)
    binary = _fake_codex(
        tmp_path,
        "(pathlib.Path(os.environ['CODEX_HOME']) / 'auth.json')"
        ".write_text('{\"tokens\": \"new\"}')\n"
        "sys.exit(1)",
    )
    backend = CodexBackend(model="gpt-test", binary=str(binary))
    with pytest.raises(ModelUnavailable):
        backend.run("hi", cwd=_call_dir(tmp_path), env={"HOME": str(home),
                    "PATH": os.environ["PATH"]}, timeout_s=30)
    assert (home / ".codex" / "auth.json").read_text() == '{"tokens": "new"}'


def test_a_login_changed_elsewhere_during_the_call_is_kept(tmp_path: Path) -> None:
    home = _real_home(tmp_path)
    real = home / ".codex" / "auth.json"
    binary = _fake_codex(
        tmp_path,
        f"pathlib.Path({str(real)!r}).write_text('{{\"tokens\": \"other\"}}')\n"
        "(pathlib.Path(os.environ['CODEX_HOME']) / 'auth.json')"
        ".write_text('{\"tokens\": \"new\"}')",
    )
    backend = CodexBackend(model="gpt-test", binary=str(binary))
    backend.run("hi", cwd=_call_dir(tmp_path), env={"HOME": str(home),
                "PATH": os.environ["PATH"]}, timeout_s=30)
    assert real.read_text() == '{"tokens": "other"}'


def test_an_unchanged_login_is_not_rewritten(tmp_path: Path) -> None:
    home = _real_home(tmp_path)
    real = home / ".codex" / "auth.json"
    before = real.stat().st_ino
    binary = _fake_codex(tmp_path, "pass")
    CodexBackend(model="gpt-test", binary=str(binary)).run(
        "hi", cwd=_call_dir(tmp_path), env={"HOME": str(home), "PATH": os.environ["PATH"]},
        timeout_s=30,
    )
    assert real.stat().st_ino == before


def test_no_login_refuses_before_codex_starts(tmp_path: Path) -> None:
    home = tmp_path / "nohome"
    home.mkdir()
    ran = tmp_path / "ran"
    binary = _fake_codex(tmp_path, f"pathlib.Path({str(ran)!r}).write_text('x')")
    with pytest.raises(ModelNotStarted, match="login"):
        CodexBackend(model="gpt-test", binary=str(binary)).run(
            "hi", cwd=_call_dir(tmp_path), env={"HOME": str(home), "PATH": os.environ["PATH"]},
            timeout_s=30,
        )
    assert not ran.exists()


def test_codex_argv_turns_off_agents_skills_and_sets_the_effort() -> None:
    argv = CodexBackend(model="gpt-test").argv()
    configs = {argv[i + 1] for i, a in enumerate(argv) if a == "-c"}
    assert {"agents.enabled=false", "skills.include_instructions=false",
            "skills.bundled.enabled=false", 'model_reasoning_effort="medium"'} <= configs
    disabled = {argv[i + 1] for i, a in enumerate(argv) if a == "--disable"}
    assert {"goals", "memories", "sleep_tool", "shell_tool"} <= disabled
    assert argv[-3:] == ["--model", "gpt-test", "-"]


# --- the request audit -------------------------------------------------------------------

def _body(*texts: str, tools: list[str] | None = None, nested: list[str] | None = None,
          namespaces: list[str] | None = None) -> JsonValue:
    description = "".join(f"### `{name}`\nsomething\n" for name in (nested or []))
    functions: list[JsonValue] = [
        {"type": "function", "name": name,
         "description": description if name == "exec" else ""}
        for name in (tools if tools is not None else ["exec", "wait"])
    ]
    spaces: list[JsonValue] = [{"type": "namespace", "name": "functions", "tools": functions}]
    for extra in namespaces or []:
        spaces.append({"type": "namespace", "name": extra, "tools": []})
    items: list[JsonValue] = [{"type": "additional_tools", "role": "developer", "tools": spaces}]
    for text in ("You are Codex.", *texts, "the prompt"):
        items.append({"type": "message", "role": "user",
                      "content": [{"type": "input_text", "text": text}]})
    return {"model": "m", "input": items}


HOME = Path("/home/someone")


def test_a_clean_request_passes() -> None:
    body = _body("<permissions instructions>read-only</permissions instructions>",
                 "explain where it came from (for example, a SKILL.md or AGENTS.md block)",
                 tools=sorted(CODEX_ALLOWED_TOOLS), nested=sorted(CODEX_ALLOWED_NESTED_TOOLS))
    assert audit_codex_request(body, home=HOME) == []


@pytest.mark.parametrize(("text", "needle"), [
    ("# AGENTS.md instructions\n\nrules", "AGENTS.md"),
    ("<skills_instructions>\n## Skills\n</skills_instructions>", "skills"),
    ("<multi_agent_role>You are /root</multi_agent_role>", "multi_agent"),
    ("see /home/someone/.agents/skills/canvas", "home"),
    ("load /opt/skills/canvas/SKILL.md", "skill file"),
])
def test_user_context_in_the_request_fails(text: str, needle: str) -> None:
    problems = audit_codex_request(_body(text), home=HOME)
    assert len(problems) == 1
    assert needle in problems[0]


def test_a_tool_outside_the_list_fails() -> None:
    problems = audit_codex_request(_body(tools=["exec", "exec_command"]), home=HOME)
    assert problems == ["tool not allowed: functions.exec_command"]


def test_a_nested_tool_outside_the_list_fails() -> None:
    problems = audit_codex_request(_body(nested=["apply_patch", "read_file"]), home=HOME)
    assert problems == ["nested exec tool not allowed: read_file"]


def test_another_tool_namespace_fails() -> None:
    problems = audit_codex_request(_body(namespaces=["collaboration"]), home=HOME)
    assert problems == ["tool namespace not allowed: collaboration"]


def test_top_level_tools_and_instructions_are_audited() -> None:
    body = _body()
    assert isinstance(body, dict)
    body["tools"] = [{"type": "function", "name": "web_search"}]
    body["instructions"] = "# AGENTS.md instructions\nrules"
    problems = audit_codex_request(body, home=HOME)
    assert "tool not allowed: web_search" in problems
    assert any("AGENTS.md" in p for p in problems)


def test_a_request_that_is_not_an_object_fails() -> None:
    assert audit_codex_request([], home=HOME) == ["the request is not a JSON object"]


@pytest.mark.live
@pytest.mark.skipif(shutil.which("codex") is None, reason="codex is not installed")
def test_the_real_codex_request_carries_no_user_context() -> None:
    """Runs the installed codex against a local sink with the user's real HOME.

    No model is called. Before the fix this request held ~/.codex/AGENTS.md,
    the skill catalog, and the collaboration tools.
    """
    body = capture_codex_request(model="gpt-6.2-astra", env=scrubbed_env(), timeout_s=60)
    assert audit_codex_request(body, home=Path.home()) == []
