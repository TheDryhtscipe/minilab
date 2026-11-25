"""The one HTTP client the known-result filter uses (build plan doc-5wh, Task 10).

The filter never uses a general web search engine (design doc-gmj, "Known-result
filter"). This client enforces that at run time: it reaches only the hosts in
`ALLOWED_HOSTS`, over HTTPS, and it follows a redirect only to another allowed
host. It does not read proxy settings from the environment, and it sends a
generic User-Agent.

Any failure of the service (a network error, a time-out, a status other than
200, or a body over the size limit) raises `SourceUnavailable`, which the
filter records as `unchecked`, never as "not found". A host off the list
raises `HostNotAllowed`, which is a bug in the caller, not an outage.

arXiv asks API users for one request every three seconds, so calls to
arXiv's hosts wait for that gap. A host may also ask for a pause (the Stack
Exchange API's `backoff`); `defer` records it, and the next call waits.
"""

import time
from collections.abc import Callable, Mapping

import httpx

ALLOWED_HOSTS: frozenset[str] = frozenset(
    {"arxiv.org", "export.arxiv.org", "api.stackexchange.com", "loogle.lean-lang.org"}
)

USER_AGENT = "minilab-literature/0.1"

# Hosts that share one rate limit, and the gap between their calls, in seconds.
_LIMIT_GROUP: dict[str, str] = {"arxiv.org": "arxiv", "export.arxiv.org": "arxiv"}
DEFAULT_INTERVALS: dict[str, float] = {"arxiv": 3.0}

DEFAULT_TIMEOUT_S = 30.0

# The largest body read, in bytes. PDFs pass their own, larger limit.
DEFAULT_MAX_BYTES = 8 * 1024**2

MAX_REDIRECTS = 5


class SourceUnavailable(Exception):
    """A literature source did not answer usably: the search did not happen."""


class HostNotAllowed(ValueError):
    """A request named a host the filter may not reach."""


def check_url(url: str) -> httpx.URL:
    """Return `url` parsed. Raise HostNotAllowed unless it is HTTPS to an allowed host."""
    parsed = httpx.URL(url)
    if parsed.scheme != "https" or parsed.host not in ALLOWED_HOSTS:
        raise HostNotAllowed(f"the literature filter may not fetch {url}")
    return parsed


class LiteratureHttp:
    """GET requests to the allowed hosts. `transport` replaces the network, for tests."""

    def __init__(
        self,
        transport: httpx.BaseTransport | None = None,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        intervals: Mapping[str, float] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.transport = transport
        self.timeout_s = timeout_s
        self.intervals = dict(DEFAULT_INTERVALS if intervals is None else intervals)
        self.sleep = sleep
        self.clock = clock
        self._next_allowed: dict[str, float] = {}

    def _group(self, host: str) -> str:
        return _LIMIT_GROUP.get(host, host)

    def _wait(self, host: str) -> None:
        group = self._group(host)
        ready = self._next_allowed.get(group)
        if ready is not None:
            delay = ready - self.clock()
            if delay > 0:
                self.sleep(delay)
        self._next_allowed[group] = self.clock() + self.intervals.get(group, 0.0)

    def defer(self, host: str, seconds: float) -> None:
        """Make the next call to `host` wait at least `seconds` from now."""
        group = self._group(host)
        ready = self.clock() + max(seconds, 0.0)
        self._next_allowed[group] = max(self._next_allowed.get(group, ready), ready)

    def get(
        self,
        url: str,
        params: Mapping[str, str] | None = None,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> bytes:
        """Fetch `url` and return the body of a 200 response.

        Raise HostNotAllowed for a host off the list, at the start or after a
        redirect, and SourceUnavailable for any failure of the service.
        """
        target = check_url(url).copy_merge_params(dict(params or {}))
        with httpx.Client(
            trust_env=False,
            transport=self.transport,
            timeout=self.timeout_s,
            follow_redirects=False,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            for _ in range(MAX_REDIRECTS + 1):
                self._wait(target.host)
                try:
                    with client.stream("GET", target) as response:
                        if response.is_redirect:
                            location = response.headers.get("location", "")
                            target = check_url(str(target.join(location)))
                            continue
                        if response.status_code != 200:
                            response.read()
                            raise SourceUnavailable(
                                f"{target.host}: HTTP {response.status_code}: {response.text[:300]}"
                            )
                        body = bytearray()
                        for chunk in response.iter_bytes():
                            body.extend(chunk)
                            if len(body) > max_bytes:
                                raise SourceUnavailable(
                                    f"{target.host}: the response is over {max_bytes} bytes"
                                )
                        return bytes(body)
                except httpx.TimeoutException as exc:
                    raise SourceUnavailable(
                        f"{target.host}: timed out after {self.timeout_s} s"
                    ) from exc
                except httpx.HTTPError as exc:
                    raise SourceUnavailable(f"{target.host}: {exc}") from exc
        raise SourceUnavailable(f"{url}: more than {MAX_REDIRECTS} redirects")
