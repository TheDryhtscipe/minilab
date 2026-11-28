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
)
from minilab.records import Family
from minilab.run import DEFAULT_MODELS

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


def _real_home(tmp_path: Path, login: str = '{"tokens": {"t": "old"}}') -> Path:
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
    assert got["auth"] == '{"tokens": {"t": "old"}}'


def test_a_refreshed_login_is_written_back(tmp_path: Path) -> None:
    home = _real_home(tmp_path)
    binary = _fake_codex(
        tmp_path,
        "(pathlib.Path(os.environ['CODEX_HOME']) / 'auth.json')"
        ".write_text('{\"tokens\": {\"t\": \"new\"}}')",
    )
    backend = CodexBackend(model="gpt-test", binary=str(binary))
    backend.run("hi", cwd=_call_dir(tmp_path), env={"HOME": str(home),
                "PATH": os.environ["PATH"]}, timeout_s=30)
    auth = home / ".codex" / "auth.json"
    assert auth.read_text() == '{"tokens": {"t": "new"}}'
    assert stat.S_IMODE(auth.stat().st_mode) == 0o600


def test_a_refreshed_login_is_written_back_after_a_failed_call(tmp_path: Path) -> None:
    home = _real_home(tmp_path)
    binary = _fake_codex(
        tmp_path,
        "(pathlib.Path(os.environ['CODEX_HOME']) / 'auth.json')"
        ".write_text('{\"tokens\": {\"t\": \"new\"}}')\n"
        "sys.exit(1)",
    )
    backend = CodexBackend(model="gpt-test", binary=str(binary))
    with pytest.raises(ModelUnavailable):
        backend.run("hi", cwd=_call_dir(tmp_path), env={"HOME": str(home),
                    "PATH": os.environ["PATH"]}, timeout_s=30)
    assert (home / ".codex" / "auth.json").read_text() == '{"tokens": {"t": "new"}}'


def test_a_login_changed_elsewhere_during_the_call_is_kept(tmp_path: Path) -> None:
    home = _real_home(tmp_path)
    real = home / ".codex" / "auth.json"
    binary = _fake_codex(
        tmp_path,
        f"pathlib.Path({str(real)!r}).write_text('{{\"tokens\": {{\"t\": \"other\"}}}}')\n"
        "(pathlib.Path(os.environ['CODEX_HOME']) / 'auth.json')"
        ".write_text('{\"tokens\": {\"t\": \"new\"}}')",
    )
    backend = CodexBackend(model="gpt-test", binary=str(binary))
    with pytest.warns(UserWarning, match="changed during the call"):
        backend.run("hi", cwd=_call_dir(tmp_path), env={"HOME": str(home),
                    "PATH": os.environ["PATH"]}, timeout_s=30)
    assert real.read_text() == '{"tokens": {"t": "other"}}'


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
            "skills.bundled.enabled=false", 'model_reasoning_effort="medium"',
            "project_doc_max_bytes=0", "project_root_markers=[]"} <= configs
    disabled = {argv[i + 1] for i, a in enumerate(argv) if a == "--disable"}
    assert {"goals", "memories", "sleep_tool", "shell_tool"} <= disabled
    assert argv[-3:] == ["--model", "gpt-test", "-"]


# --- the request audit -------------------------------------------------------------------

def _body(*texts: str, tools: list[str] | None = None, nested: list[str] | None = None,
          namespaces: list[str] | None = None) -> JsonValue:
    names = nested if nested is not None else sorted(CODEX_ALLOWED_NESTED_TOOLS)
    description = "".join(f"### `{name}`\nsomething\n" for name in names)
    functions: list[JsonValue] = [
        {"type": "function", "name": name,
         "description": description if name == "exec" else ""}
        for name in (tools if tools is not None else sorted(CODEX_ALLOWED_TOOLS))
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
    ("<INSTRUCTIONS>user rules</INSTRUCTIONS>", "instructions"),
    ("see ~/.codex/AGENTS.md", "home"),
])
def test_user_context_in_the_request_fails(text: str, needle: str) -> None:
    problems = audit_codex_request(_body(text), home=HOME)
    assert len(problems) == 1
    assert needle in problems[0]


def test_a_tool_outside_the_list_fails() -> None:
    tools = [*sorted(CODEX_ALLOWED_TOOLS), "exec_command"]
    problems = audit_codex_request(_body(tools=tools), home=HOME)
    assert problems == ["tool not allowed: functions.exec_command"]


def test_a_nested_tool_outside_the_list_fails() -> None:
    nested = [*sorted(CODEX_ALLOWED_NESTED_TOOLS), "read_file"]
    problems = audit_codex_request(_body(nested=nested), home=HOME)
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


def test_a_request_of_another_shape_fails_closed() -> None:
    assert audit_codex_request({}, home=HOME) == ["the request has no input items"]
    no_tools: JsonValue = {"input": [{"type": "message", "role": "user", "content": []}]}
    assert audit_codex_request(no_tools, home=HOME) == [
        "expected one additional_tools item, found 0"]
    moved = _body()
    assert isinstance(moved, dict) and isinstance(moved["input"], list)
    first = moved["input"][0]
    assert isinstance(first, dict) and isinstance(first["tools"], list)
    space = first["tools"][0]
    assert isinstance(space, dict)
    space["functions"] = space.pop("tools")
    problems = audit_codex_request(moved, home=HOME)
    assert len(problems) == 1 and "expected tools missing" in problems[0]


def test_nested_tools_listed_another_way_fail_closed() -> None:
    problems = audit_codex_request(_body(nested=[]), home=HOME)
    assert len(problems) == 1 and "expected nested exec tools missing" in problems[0]


def test_the_users_own_rules_and_skills_are_found_by_content(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / ".codex").mkdir(parents=True)
    rule = "Always answer in the house style of the federation, with citations."
    (home / ".codex" / "AGENTS.md").write_text(f"# Rules\n{rule}\nshort line\n")
    (home / ".agents" / "skills" / "canvas-drawing").mkdir(parents=True)
    leaked = _body(f"<new_wrapper>{rule}</new_wrapper>", "use canvas-drawing/SKILL.md")
    problems = audit_codex_request(leaked, home=home)
    assert problems == ["user context reached the model: a skill file",
                        "user context reached the model: a line of ~/.codex/AGENTS.md",
                        "user context reached the model: the skill canvas-drawing"]
    assert audit_codex_request(_body("short line"), home=home) == []


def test_a_corrupt_refreshed_login_is_not_written_back(tmp_path: Path) -> None:
    home = _real_home(tmp_path)
    binary = _fake_codex(
        tmp_path,
        "(pathlib.Path(os.environ['CODEX_HOME']) / 'auth.json').write_text('{\"tokens\":\"ne')",
    )
    with pytest.warns(UserWarning, match="not a whole login"):
        CodexBackend(model="gpt-test", binary=str(binary)).run(
            "hi", cwd=_call_dir(tmp_path), env={"HOME": str(home),
            "PATH": os.environ["PATH"]}, timeout_s=30)
    assert (home / ".codex" / "auth.json").read_text() == '{"tokens": {"t": "old"}}'


def test_a_failed_write_back_never_replaces_the_result(tmp_path: Path) -> None:
    home = _real_home(tmp_path)
    real = home / ".codex" / "auth.json"
    refresh = ("(pathlib.Path(os.environ['CODEX_HOME']) / 'auth.json')"
               ".write_text('{\"tokens\": {\"t\": \"new\"}}')\n"
               f"os.chmod({str(real.parent)!r}, 0o500)\n")
    good = _fake_codex(tmp_path, refresh)
    try:
        with pytest.warns(UserWarning, match="could not write the refreshed login"):
            reply = CodexBackend(model="gpt-test", binary=str(good)).run(
                "hi", cwd=_call_dir(tmp_path), env={"HOME": str(home),
                "PATH": os.environ["PATH"]}, timeout_s=30)
        assert reply.text == "answer"
    finally:
        real.parent.chmod(0o700)
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    bad = _fake_codex(bad_dir, refresh + "print(" + repr(_REPLY.splitlines()[1]) + ")\nsys.exit(1)")
    try:
        with pytest.raises(ModelUnavailable) as caught:
            CodexBackend(model="gpt-test", binary=str(bad)).run(
                "hi", cwd=_call_dir(bad_dir), env={"HOME": str(home),
                "PATH": os.environ["PATH"]}, timeout_s=30)
    finally:
        real.parent.chmod(0o700)
    assert caught.value.tokens_in == 20 and caught.value.tokens_out == 9
    assert any("could not write the refreshed login" in n
               for n in getattr(caught.value, "__notes__", []))


@pytest.mark.live
@pytest.mark.skipif(shutil.which("codex") is None, reason="codex is not installed")
def test_the_real_codex_request_carries_no_user_context() -> None:
    """Runs the installed codex against a local sink with the user's real HOME.

    No model is called, but codex may refresh the user's login with OpenAI's
    auth service and write it back, as a real call does. Before the fix this
    request held ~/.codex/AGENTS.md, the skill catalog, and the collaboration
    tools.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from codex_sink import capture_codex_request

    body = capture_codex_request(model=DEFAULT_MODELS[Family.codex], env=scrubbed_env(),
                                 timeout_s=60)
    assert audit_codex_request(body, home=Path.home()) == []
