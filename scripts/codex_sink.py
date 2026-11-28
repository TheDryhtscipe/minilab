"""Capture the request a Codex call sends, against a local sink. No model runs.

`capture_codex_request` runs `codex exec` with `CodexBackend`'s own flags and
per-call home, plus a provider override that points it at an HTTP server on
127.0.0.1. The server serves `cached_catalog` for `/models`, records the
`/v1/responses` body, and answers every other request with an error, so no
model runs and nothing is spent. `minilab.models.codex_request` audits the
body. This file is outside `src` because `src` holds no network code
(tests/test_literature.py, test_no_general_web_search). Used by
`scripts/probe_isolation.py` and tests/test_codex_isolation.py.
"""

import json
import tempfile
import threading
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from pydantic import JsonValue

from minilab.models.backends import CodexBackend, ModelUnavailable

_SINK = "minilab_sink"


class _Sink(BaseHTTPRequestHandler):
    bodies: list[bytes]
    catalog: bytes | None

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
        if self.catalog is not None and self.path.split("?")[0].endswith("/models"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(self.catalog)))
            self.end_headers()
            self.wfile.write(self.catalog)
            return
        self._refuse()

    def log_message(self, format: str, *args: object) -> None:
        return


def cached_catalog(home: Path) -> bytes | None:
    """The model catalog codex last fetched (`~/.codex/models_cache.json`), as `/models` serves it.

    A real call has an empty `CODEX_HOME`, so it fetches the server's catalog,
    which sets the model's tools and instructions. The sink serves this copy so
    the capture sees the same entry; with no copy codex falls back to the
    catalog built into it, which can differ.
    """
    try:
        cache: JsonValue = json.loads((home / ".codex" / "models_cache.json").read_bytes())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(cache, dict) or not isinstance(cache.get("models"), list):
        return None
    return json.dumps({"models": cache["models"]}).encode()


def capture_codex_request(
    *, model: str, env: Mapping[str, str], timeout_s: float, binary: str = "codex"
) -> JsonValue:
    """Run one Codex call against a local sink and return the request body it sent.

    The call uses `CodexBackend`'s own flags and per-call home, plus a provider
    override that points it at the sink, which serves `cached_catalog`. The
    call takes the login lock and may refresh the user's login, writing it
    back, as any real call does; it reaches no model. Raise RuntimeError if
    codex sent no request.
    """
    bodies: list[bytes] = []
    handler = type("Sink", (_Sink,), {"bodies": bodies,
                                      "catalog": cached_catalog(Path(env.get("HOME") or
                                                                     Path.home()))})
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
