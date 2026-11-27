"""A cache of Lean results for one test run (bead minilab-81c4fe).

Most of the suite's time is spent compiling the same few Lean files again: the
triviality tests, for example, try the same five tactics on the same pilot
fixtures in test after test. Lean gives the same result for the same input, so
`conftest.py` puts a `CachedLean` in place of `check_lean` where the
formaliser, the vacuity check, and Loogle call it. The runner's own tests
import `check_lean` directly and always run Lean.

An entry is keyed by everything `check_lean` takes: the source, the timeout,
the project, and the options. A result that timed out is never stored. Entries
are files in one directory that every test worker of the run shares, and each
entry repeats its key, which is compared before the entry is used.
"""

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, JsonValue

from minilab.lean import runner
from minilab.lean.runner import DEFAULT_TIMEOUT_S, LeanResult


class LeanCheck(Protocol):
    """The signature of `check_lean`."""

    def __call__(
        self,
        source: str,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        *,
        project: Path | None = None,
        options: Mapping[str, str] | None = None,
    ) -> LeanResult: ...


class CachedLean:
    """`check_lean` with a per-run cache in `directory`."""

    def __init__(self, inner: LeanCheck, directory: Path) -> None:
        self.inner = inner
        self.directory = directory
        self.misses = 0

    def __call__(
        self,
        source: str,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        *,
        project: Path | None = None,
        options: Mapping[str, str] | None = None,
    ) -> LeanResult:
        resolved = runner.lean_project() if project is None else project
        key: dict[str, JsonValue] = {
            "source": source,
            "timeout_s": timeout_s,
            "project": str(resolved.resolve()),
            "options": [[name, value] for name, value in sorted((options or {}).items())],
        }
        key_text = json.dumps(key, sort_keys=True)
        entry = self.directory / f"{hashlib.sha256(key_text.encode()).hexdigest()}.json"
        try:
            stored = _Entry.model_validate_json(entry.read_text(encoding="utf-8"))
        except FileNotFoundError:
            pass
        else:
            if stored.key == key_text:
                return stored.result
        self.misses += 1
        result = self.inner(source, timeout_s, project=project, options=options)
        if not result.timed_out:
            # Write then rename, so a worker reading at the same time never
            # sees half an entry.
            partial = entry.with_name(f"{entry.name}.{os.getpid()}.partial")
            partial.write_text(
                _Entry(key=key_text, result=result).model_dump_json(), encoding="utf-8"
            )
            partial.replace(entry)
        return result


class _Entry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    result: LeanResult


class NonceSequence:
    """Stands in for `new_nonce` while the cache is on.

    The formaliser tags each Lean run's markers with a random nonce, so no two
    runs would ever send Lean the same text. A fresh sequence for each test
    gives the same nonces in the same order, so tests that make the same checks
    in the same order hit the cache. The values are hashes, so no fixture can
    match them by chance.
    """

    def __init__(self) -> None:
        self.count = 0

    def __call__(self) -> str:
        self.count += 1
        return hashlib.sha256(f"minilab test nonce {self.count}".encode()).hexdigest()[:16]
