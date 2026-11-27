"""The run's MuninnDB memory: the read policy, and one memory per finished record.

Design doc-gmj, "Memory read policy"; build plan doc-5wh, Task 16. Each run's
manifest says how it may read the `minilab` vault:

- `blind`: no reads. `MemoryClient.recall` raises `MemoryReadRefused` before
  it sends anything.
- `pair`: only memories of records for the same domain pair.
- `all`: every record memory.

`memory_before` becomes MuninnDB's `before` filter. The client also checks
each answer itself, so a server that ignored a filter cannot leak a memory:
it drops a memory of another pair (under `pair`), a memory of this run, and,
with a date cap, a memory created at or after the cap or with no creation
time.

Every call names the vault `VAULT` (`minilab`). There is no way to name
another, so the pilot's vault `sci` is never read.

Writes do not depend on the policy: when a record finishes, the driver writes
one memory for it (`remember`), upserted on the op id
`minilab-record:<record id>` so a resumed run never writes it twice.

The transport is MuninnDB's MCP endpoint, called with bare JSON-RPC
`tools/call` posts (as ORC's handoff client does). The bearer token is read
from `MUNINN_MCP_TOKEN` or `~/.muninn/mcp.token`; it is never logged or put
in an error message.
"""

import json
import os
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Protocol

import httpx
from pydantic import BaseModel, ConfigDict, JsonValue

from minilab.records import Record
from minilab.run import MemoryPolicy, RunManifest
from minilab.scheduler import Pair, canonical_pair

__all__ = [
    "DEFAULT_MCP_URL",
    "RECORD_TAG",
    "VAULT",
    "EarlierRecord",
    "McpTransport",
    "MemoryClient",
    "MemoryReadRefused",
    "MemoryTransport",
    "MemoryUnavailable",
    "memory_content",
    "pair_tag",
]

VAULT = "minilab"
"""The only vault a run reads or writes."""

RECORD_TAG = "minilab-record"
"""The tag on every record memory the driver writes."""

DEFAULT_MCP_URL = "http://127.0.0.1:8750/mcp"

_TOKEN_FILE = Path("~/.muninn/mcp.token")
_TIMEOUT_S = 10.0
_STATEMENT = "Statement: "


class MemoryUnavailable(Exception):
    """MuninnDB could not be reached or did not answer as expected."""


class MemoryReadRefused(Exception):
    """The run's memory policy forbids this read."""


class MemoryTransport(Protocol):
    """Calls one MuninnDB tool and returns the JSON object it answered with."""

    def call_tool(self, name: str, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]: ...


def _token() -> str | None:
    from_env = os.environ.get("MUNINN_MCP_TOKEN", "").strip()
    if from_env:
        return from_env
    try:
        token = _TOKEN_FILE.expanduser().read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return token or None


class McpTransport:
    """MuninnDB's MCP endpoint over HTTP. Every failure becomes `MemoryUnavailable`."""

    def __init__(
        self,
        url: str | None = None,
        *,
        http: httpx.BaseTransport | None = None,
        timeout_s: float = _TIMEOUT_S,
    ) -> None:
        self.url = url or os.environ.get("MUNINN_MCP_URL", "").strip() or DEFAULT_MCP_URL
        self._http = http
        self._timeout_s = timeout_s

    def call_tool(self, name: str, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        token = _token()
        if token is None:
            raise MemoryUnavailable(
                "no MuninnDB token: set MUNINN_MCP_TOKEN or create ~/.muninn/mcp.token"
            )
        body: dict[str, JsonValue] = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        try:
            with httpx.Client(transport=self._http, timeout=self._timeout_s) as client:
                response = client.post(self.url, json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise MemoryUnavailable(f"MuninnDB request failed: {type(exc).__name__}") from exc
        if response.status_code != 200:
            raise MemoryUnavailable(f"MuninnDB answered HTTP {response.status_code}")
        try:
            envelope: JsonValue = response.json()
        except ValueError as exc:
            raise MemoryUnavailable("MuninnDB answered with a body that is not JSON") from exc
        return tool_payload(name, envelope)


def tool_payload(name: str, envelope: JsonValue) -> dict[str, JsonValue]:
    """The JSON object a tool wrote as its text result, from a JSON-RPC envelope."""
    if not isinstance(envelope, dict):
        raise MemoryUnavailable(f"MuninnDB {name}: the reply is not a JSON object")
    error = envelope.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        raise MemoryUnavailable(
            f"MuninnDB {name} failed: {message if isinstance(message, str) else 'unknown error'}"
        )
    result = envelope.get("result")
    content = result.get("content") if isinstance(result, dict) else None
    first = content[0] if isinstance(content, list) and content else None
    text = first.get("text") if isinstance(first, dict) else None
    if not isinstance(text, str):
        raise MemoryUnavailable(f"MuninnDB {name} answered without a text result")
    try:
        payload: JsonValue = json.loads(text)
    except ValueError as exc:
        raise MemoryUnavailable(f"MuninnDB {name} answered with a result that is not JSON") from exc
    if not isinstance(payload, dict):
        raise MemoryUnavailable(f"MuninnDB {name} answered with a result that is not an object")
    return payload


def pair_tag(pair: Pair) -> str:
    """The tag that marks a record memory's domain pair."""
    a, b = canonical_pair(pair)
    return f"pair:{a}/{b}"


def memory_content(record: Record) -> str:
    """The text of a record's memory. The header comes first, so a recall
    preview cut short still shows which record, run, and pair it is."""
    a, b = canonical_pair(record.pair)
    return (
        f"Minilab record {record.id}\n"
        f"Run: {record.run_id}\n"
        f"Pair: {a}/{b}\n"
        f"Mode: {record.mode}\n"
        f"Status: {record.status.value}\n"
        f"Record file: runs/{record.run_id}/records/{record.id}.json\n"
        f"{_STATEMENT}{record.statement}"
    )


class EarlierRecord(BaseModel):
    """A record from an earlier run, as its memory gives it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str
    run_id: str
    pair: str
    status: str
    statement: str


def _parse(content: str) -> EarlierRecord | None:
    """Read a memory written by `memory_content`, or None if it is not one."""
    head, sep, statement = content.partition("\n" + _STATEMENT)
    if not sep:
        return None
    lines = head.split("\n")
    if not lines[0].startswith("Minilab record "):
        return None
    fields: dict[str, str] = {}
    for line in lines[1:]:
        key, colon, value = line.partition(": ")
        if colon:
            fields[key] = value
    try:
        return EarlierRecord(
            record_id=lines[0].removeprefix("Minilab record ").strip(),
            run_id=fields["Run"],
            pair=fields["Pair"],
            status=fields["Status"],
            statement=statement.strip(),
        )
    except KeyError:
        return None


def _created(item: Mapping[str, JsonValue]) -> datetime | None:
    raw = item.get("created_at")
    if not isinstance(raw, str):
        return None
    try:
        when = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return when if when.tzinfo is not None else None


class MemoryClient:
    """Reads and writes one run's record memories in vault `minilab`, under its policy."""

    def __init__(self, manifest: RunManifest, transport: MemoryTransport) -> None:
        self.manifest = manifest
        self.transport = transport

    def recall(self, pair: Pair, limit: int = 20) -> list[EarlierRecord]:
        """Return earlier runs' records the policy allows for `pair`, best match first.

        Raise `MemoryReadRefused` under `blind` (nothing is sent), ValueError
        for a pair the manifest does not list, and `MemoryUnavailable` if
        MuninnDB fails.
        """
        policy = self.manifest.memory_policy
        if policy is MemoryPolicy.blind:
            raise MemoryReadRefused(f"run {self.manifest.run_id} is blind: it reads no memory")
        pair = canonical_pair(pair)
        if pair not in self.manifest.pairs:
            raise ValueError(f"pair {pair} is not in run {self.manifest.run_id}'s manifest")
        a, b = pair
        tags: list[JsonValue] = [RECORD_TAG]
        if policy is MemoryPolicy.pair:
            tags.append(pair_tag(pair))
        arguments: dict[str, JsonValue] = {
            "vault": VAULT,
            "context": [f"conjectures linking {a} and {b}"],
            "tags_all": tags,
            "limit": limit,
            "read_only": True,
        }
        before = self.manifest.memory_before
        if before is not None:
            arguments["before"] = before.isoformat().replace("+00:00", "Z")
        answer = self.transport.call_tool("muninn_recall", arguments)
        memories = answer.get("memories")
        if not isinstance(memories, list):
            raise MemoryUnavailable("MuninnDB muninn_recall answered without a memory list")
        found: list[EarlierRecord] = []
        for item in memories:
            if not isinstance(item, dict):
                continue
            annotations = item.get("annotations")
            if isinstance(annotations, dict) and annotations.get("superseded_by"):
                continue
            content = item.get("content")
            earlier = _parse(content) if isinstance(content, str) else None
            if earlier is None or earlier.run_id == self.manifest.run_id:
                continue
            if policy is MemoryPolicy.pair and earlier.pair != f"{a}/{b}":
                continue
            if before is not None:
                created = _created(item)
                if created is None or created >= before:
                    continue
            found.append(earlier)
        return found

    def remember(self, record: Record) -> str:
        """Write (or update) `record`'s memory and return its id.

        Raise `MemoryUnavailable` if MuninnDB fails, and ValueError for a record
        of another run.
        """
        if record.run_id != self.manifest.run_id:
            raise ValueError(f"record {record.id} is not in run {self.manifest.run_id}")
        statement = " ".join(record.statement.split())
        answer = self.transport.call_tool(
            "muninn_remember",
            {
                "vault": VAULT,
                "concept": f"Minilab record {record.id}",
                "summary": f"{record.status.value}: {statement[:160]}",
                "content": memory_content(record),
                "type": "fact",
                "tags": [
                    RECORD_TAG,
                    f"run:{record.run_id}",
                    pair_tag(record.pair),
                    f"status:{record.status.value}",
                    f"mode:{record.mode}",
                ],
                "op_id": f"minilab-record:{record.id}",
                "upsert_mode": True,
            },
        )
        memory_id = answer.get("id")
        if not isinstance(memory_id, str) or not memory_id:
            raise MemoryUnavailable("MuninnDB muninn_remember answered without an id")
        return memory_id
