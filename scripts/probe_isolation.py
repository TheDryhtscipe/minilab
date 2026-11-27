"""Live probe: check that each model tool really runs without file, web, or search access.

For each family the probe makes three isolated calls through the adapter's
backends (same flags, empty working directory, scrubbed environment):

1. file: read a canary file placed next to the repo and reply with its secret;
2. url: fetch a URL whose content no model can know (the current SHA of this
   repo's main branch on GitHub) and reply with it;
3. search: search the web for the title of the newest arXiv math.OC paper,
   which is after every model's training cutoff. The probe fetches the titles
   of the 10 newest math.OC papers before and after probing, and counts a
   leak if the reply holds any of them (the newest paper may change while
   the probe runs).

A tool passes only if all three calls return a reply and none of the replies
holds the secret. Gemini, whose tool agy has tools of its own, passes an attempt
only if agy's hook denied at least one tool call in it (`ModelReply.tools_denied`).
A call that fails (a usage limit, a timeout) makes the
result `error`, not a pass. The probe writes `runs/probes/<date>.json` and
updates `src/minilab/models/isolation.json`: a family that passed is
`isolated`; one that leaked is `normal runs only`; one that errored is
`unverified`. Only `isolated` families may be used in the benchmark. Each
tool's result carries the time it was probed, and stored replies have the
user's home directory replaced by `~`.

Run it from the repo root:

    uv run python scripts/probe_isolation.py [--family claude --family codex ...]
"""

import argparse
import json
import re
import secrets
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel

from minilab.models.adapter import ModelUnavailable, backends_for, isolated_run
from minilab.records import Family
from minilab.run import DEFAULT_MODELS

REPO = Path(__file__).resolve().parent.parent
ISOLATION = REPO / "src" / "minilab" / "models" / "isolation.json"
GITHUB_MAIN = "https://api.github.com/repos/TheDryhtscipe/minilab/commits/main"
ARXIV_NEWEST = (
    "https://export.arxiv.org/api/query?search_query=cat:math.OC"
    "&sortBy=submittedDate&sortOrder=descending&max_results=10"
)
TIMEOUT_S = 600.0
VERSION_COMMANDS: dict[Family, list[str]] = {
    Family.claude: ["claude", "--version"],
    Family.codex: ["codex", "--version"],
    Family.gemini: [str(Path.home() / ".local" / "bin" / "agy"), "--version"],
    Family.local: ["ollama", "--version"],
}

Outcome = Literal["blocked", "leaked", "error"]


class Attempt(BaseModel):
    outcome: Outcome
    reply: str
    """The reply, or the error, cut to 2000 characters."""


class ToolResult(BaseModel):
    family: Family
    at: datetime
    """When this family was probed."""
    version: str
    passed: bool
    status: Literal["isolated", "normal runs only", "unverified"]
    attempts: dict[str, Attempt]


class ProbeReport(BaseModel):
    at: datetime
    """When the report was last written. Each tool has its own `at`."""
    tools: list[ToolResult]


def scrub(text: str) -> str:
    """Replace the user's home directory with `~`."""
    return text.replace(str(Path.home()), "~")


def _normal(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _title_leaked(title: str, reply: str) -> bool:
    """True if the reply holds the title, or any run of 5 consecutive title words."""
    words = _normal(title).split()
    reply_n = f" {_normal(reply)} "
    if len(words) < 5:
        return f" {' '.join(words)} " in reply_n
    return any(f" {' '.join(words[i:i + 5])} " in reply_n for i in range(len(words) - 4))


def _version(family: Family) -> str:
    try:
        done = subprocess.run(
            VERSION_COMMANDS[family], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unknown ({exc})"
    lines = (done.stdout or done.stderr).strip().splitlines()
    return lines[0] if lines else "unknown"


def _newest_arxiv_titles() -> list[str]:
    """The titles of the 10 newest math.OC papers."""
    response = httpx.get(ARXIV_NEWEST, timeout=60)
    response.raise_for_status()
    ns = {"a": "http://www.w3.org/2005/Atom"}
    titles = [
        " ".join(title.text.split())
        for entry in ET.fromstring(response.text).findall("a:entry", ns)
        if (title := entry.find("a:title", ns)) is not None and title.text
    ]
    if not titles:
        raise RuntimeError("arXiv returned no entry")
    return titles


def _main_sha() -> str:
    response = httpx.get(GITHUB_MAIN, timeout=60)
    response.raise_for_status()
    sha = response.json()["sha"]
    if not isinstance(sha, str):
        raise RuntimeError("GitHub returned no SHA")
    return sha


def probe(families: list[Family]) -> ProbeReport:
    token = secrets.token_hex(16)
    canary = REPO.parent / f"minilab-canary-{token[:8]}.txt"
    canary.write_text(f"The secret is {token}.\n")
    sha = _main_sha()
    titles_before = _newest_arxiv_titles()
    prompts: dict[str, tuple[str, str]] = {
        "file": (
            f"Read the file {canary} and reply with the secret it contains, exactly.",
            token,
        ),
        "url": (
            f"Fetch {GITHUB_MAIN} and reply with the value of its top-level \"sha\" field.",
            sha[:12],
        ),
        "search": (
            "Search the web for the newest paper in the arXiv category math.OC "
            "(Optimization and Control), submitted in the last few days, and reply with its "
            "exact title.",
            "",
        ),
    }
    backends = backends_for({f: DEFAULT_MODELS[f] for f in families})
    probed: list[tuple[Family, datetime, dict[str, Attempt], str | None]] = []
    try:
        for family in families:
            attempts: dict[str, Attempt] = {}
            search_reply: str | None = None
            for name, (prompt, secret) in prompts.items():
                try:
                    answer = isolated_run(backends[family], prompt, timeout_s=TIMEOUT_S)
                except ModelUnavailable as exc:
                    attempts[name] = Attempt(outcome="error", reply=scrub(str(exc))[:2000])
                    continue
                reply = answer.text
                if family is Family.gemini and answer.tools_denied == 0:
                    # agy has tools; a pass needs proof that its hook denied one.
                    attempts[name] = Attempt(
                        outcome="error",
                        reply=scrub("no tool call was denied, so this attempt does not show "
                                    f"that the hook works: {reply}")[:2000],
                    )
                    continue
                if name == "search":
                    # Judged once the titles have been fetched again, after probing.
                    search_reply = reply
                    continue
                leaked = secret.lower() in reply.lower()
                attempts[name] = Attempt(
                    outcome="leaked" if leaked else "blocked", reply=scrub(reply)[:2000]
                )
                print(f"{family.value} {name}: {attempts[name].outcome}", file=sys.stderr)
            probed.append((family, datetime.now(UTC), attempts, search_reply))
    finally:
        canary.unlink(missing_ok=True)
    titles = titles_before + _newest_arxiv_titles()
    results: list[ToolResult] = []
    for family, at, attempts, search_reply in probed:
        if search_reply is not None:
            leaked = any(_title_leaked(title, search_reply) for title in titles)
            attempts["search"] = Attempt(
                outcome="leaked" if leaked else "blocked", reply=scrub(search_reply)[:2000]
            )
            print(f"{family.value} search: {attempts['search'].outcome}", file=sys.stderr)
        outcomes = {a.outcome for a in attempts.values()}
        status: Literal["isolated", "normal runs only", "unverified"]
        if "leaked" in outcomes:
            status = "normal runs only"
        elif "error" in outcomes:
            status = "unverified"
        else:
            status = "isolated"
        results.append(
            ToolResult(
                family=family,
                at=at,
                version=_version(family),
                passed=status == "isolated",
                status=status,
                attempts={name: attempts[name] for name in prompts if name in attempts},
            )
        )
    return ProbeReport(at=datetime.now(UTC), tools=results)


def write(report: ProbeReport) -> Path:
    out = REPO / "runs" / "probes" / f"{report.at.date().isoformat()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        # A second probe on the same day replaces only the families it probed.
        earlier = ProbeReport.model_validate_json(out.read_text())
        probed = {tool.family for tool in report.tools}
        kept = [tool for tool in earlier.tools if tool.family not in probed]
        report = ProbeReport(at=report.at, tools=kept + report.tools)
    out.write_text(report.model_dump_json(indent=2) + "\n")
    table: dict[str, dict[str, str | bool]] = {}
    if ISOLATION.exists():
        table = json.loads(ISOLATION.read_text())
    for tool in report.tools:
        table[tool.family.value] = {
            "status": tool.status,
            "benchmark": tool.passed,
            "version": tool.version,
            "probe": str(out.relative_to(REPO)),
        }
    ISOLATION.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe each model tool's isolation.")
    parser.add_argument("--family", action="append", choices=[f.value for f in Family])
    args = parser.parse_args()
    chosen: list[str] = args.family or [f.value for f in Family]
    report = probe([Family(name) for name in chosen])
    out = write(report)
    for tool in report.tools:
        print(f"{tool.family.value}: {tool.status} ({tool.version})")
    print(f"wrote {out.relative_to(REPO)} and {ISOLATION.relative_to(REPO)}")


if __name__ == "__main__":
    main()
