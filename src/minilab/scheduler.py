"""The pair scheduler.

Picks which pair of domains a run works on (design doc-gmj, stage 1). Visit
counts live in a state file in the repo, outside any model, and a pair's weight
falls as its visit count grows. The scheduler, not a model, picks the pair.
"""

import fcntl
import itertools
import json
import os
import random
import tempfile
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

DOMAINS: tuple[str, ...] = (
    "optimisation",
    "information-theory",
    "probability",
    "hd-statistics",
    "random-matrices",
    "optimal-transport",
    "kernels",
    "gradient-flows",
    "information-geometry",
    "graphs-spectral",
)
"""The ten domains, in the order of doc-gmj."""

Pair = tuple[str, str]

MAX_VISITS = 10**9
"""Visit counts above this are refused as corrupt."""


class StateError(Exception):
    """The state file cannot be read, or holds something other than valid visits."""


def canonical_pair(pair: Pair) -> Pair:
    """Return the pair in sorted order.

    Raise `ValueError` unless `pair` is a tuple or list of two different known
    domain slugs.
    """
    if not isinstance(pair, tuple | list) or len(pair) != 2:  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ValueError(f"a pair is two domain slugs, got {pair!r}")
    a, b = pair
    for domain in (a, b):
        if type(domain) is not str or domain not in DOMAINS:
            raise ValueError(f"unknown domain in pair: {domain!r}")
    if a == b:
        raise ValueError(f"a pair needs two different domains, got {a!r} twice")
    return (a, b) if a < b else (b, a)


def all_pairs() -> list[Pair]:
    """Return the 45 unordered pairs, each in sorted order, as a sorted list."""
    return sorted(itertools.combinations(sorted(DOMAINS), 2))


class _Visit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    pair: Pair
    visits: int = Field(ge=0, le=MAX_VISITS)

    @model_validator(mode="after")
    def _known_pair(self) -> "_Visit":
        if canonical_pair(self.pair) != self.pair:
            raise ValueError(f"pair not in sorted order: {self.pair}")
        return self


class _State(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    visits: list[_Visit]

    @model_validator(mode="after")
    def _no_duplicates(self) -> "_State":
        pairs = [v.pair for v in self.visits]
        if len(set(pairs)) != len(pairs):
            raise ValueError("a pair appears more than once")
        return self


type _Json = dict[str, _Json] | list[_Json] | str | int | float | bool | None


def _no_repeated_keys(items: list[tuple[str, _Json]]) -> dict[str, _Json]:
    keys = [key for key, _ in items]
    if len(set(keys)) != len(keys):
        raise ValueError(f"repeated key in state file: {keys}")
    return dict(items)


class Scheduler:
    """Weighted choice of domain pairs, with visit counts kept in `state_path`.

    A pair's weight is `1 / (1 + visits)`. The state file is read on every
    call, so separate schedulers on one file agree. It is created by the first
    `record_visit`. Each visit takes an exclusive lock on a sibling
    `.<name>.lock` file for the read and the write, and the write replaces the
    state file in one rename. The lock is advisory and works only between
    processes that use this class on a local file system.
    """

    def __init__(self, state_path: Path, rng: random.Random) -> None:
        self.state_path = state_path
        self.rng = rng

    @property
    def _lock_path(self) -> Path:
        return self.state_path.with_name(f".{self.state_path.name}.lock")

    @contextmanager
    def _locked(self) -> Generator[None]:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._lock_path, "a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def visits(self) -> dict[Pair, int]:
        """Return the visit count of every pair (0 for pairs never visited).

        Raise `StateError` if the file cannot be read or is not a valid state.
        """
        counts = dict.fromkeys(all_pairs(), 0)
        if not self.state_path.exists():
            return counts
        try:
            data = self.state_path.read_bytes()
            # The json module checks for repeated keys, which pydantic would
            # resolve silently; pydantic then validates strictly.
            json.loads(data, object_pairs_hook=_no_repeated_keys)
            state = _State.model_validate_json(data)
        except (OSError, ValueError) as exc:  # ValidationError is a ValueError
            raise StateError(f"{self.state_path}: {exc}") from exc
        for visit in state.visits:
            counts[visit.pair] = visit.visits
        return counts

    def pick(self, pairs: Sequence[Pair] | None = None) -> Pair:
        """Pick a pair at random, with weight `1 / (1 + visits)`. Does not record a visit.

        With `pairs`, pick only from those (the benchmark's explicit list). Raise
        `ValueError` if that list is empty, holds a bad pair, or repeats one.
        """
        if pairs is None:
            choices = all_pairs()
        else:
            choices = [canonical_pair(p) for p in pairs]
            if not choices:
                raise ValueError("the pair list is empty")
            if len(set(choices)) != len(choices):
                raise ValueError(f"the pair list repeats a pair: {choices}")
        counts = self.visits()
        weights = [1 / (1 + counts[p]) for p in choices]
        return self.rng.choices(choices, weights=weights, k=1)[0]

    def record_visit(self, pair: Pair) -> None:
        """Add one visit to `pair`, given in either order.

        Raise `ValueError` for a bad pair and `StateError` for a bad state file;
        the file is unchanged on either.
        """
        key = canonical_pair(pair)
        with self._locked():
            counts = self.visits()
            counts[key] += 1
            state = _State(
                visits=[_Visit(pair=p, visits=n) for p, n in sorted(counts.items()) if n > 0]
            )
            directory = self.state_path.parent
            fd, name = tempfile.mkstemp(
                dir=directory, prefix=f".{self.state_path.name}.", suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(state.model_dump_json(indent=2).encode())
                    handle.flush()
                    os.fchmod(handle.fileno(), 0o644)
                    os.fsync(handle.fileno())
                os.replace(name, self.state_path)
            except BaseException:
                Path(name).unlink(missing_ok=True)
                raise
            dir_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
