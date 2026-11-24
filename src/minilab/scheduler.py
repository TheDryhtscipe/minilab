"""The pair scheduler.

Picks which pair of domains a run works on (design doc-gmj, stage 1). Visit
counts live in a state file in the repo, outside any model, and a pair's weight
falls as its visit count grows. A model never picks or changes the pair.
"""

import itertools
import os
import random
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

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


class StateError(Exception):
    """The state file does not parse or holds something other than valid visits."""


def canonical_pair(pair: Pair) -> Pair:
    """Return the pair in sorted order. Raise `ValueError` for an unknown or self pair."""
    a, b = pair
    for domain in (a, b):
        if domain not in DOMAINS:
            raise ValueError(f"unknown domain: {domain!r}")
    if a == b:
        raise ValueError(f"a pair needs two different domains, got {a!r} twice")
    return (a, b) if a < b else (b, a)


def all_pairs() -> list[Pair]:
    """Return the 45 unordered pairs, each in sorted order, as a sorted list."""
    return sorted(itertools.combinations(sorted(DOMAINS), 2))


class _Visit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pair: Pair
    visits: int = Field(ge=0)

    @model_validator(mode="after")
    def _known_pair(self) -> "_Visit":
        if canonical_pair(self.pair) != self.pair:
            raise ValueError(f"pair not in sorted order: {self.pair}")
        return self


class _State(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    visits: list[_Visit]

    @model_validator(mode="after")
    def _no_duplicates(self) -> "_State":
        pairs = [v.pair for v in self.visits]
        if len(set(pairs)) != len(pairs):
            raise ValueError("a pair appears more than once")
        return self


class Scheduler:
    """Weighted choice of domain pairs, with visit counts kept in `state_path`.

    A pair's weight is `1 / (1 + visits)`. The state file is read on every call,
    so separate schedulers on one file agree. It is created by the first
    `record_visit`, and each write replaces it in one rename. Two processes
    recording visits at the same moment can lose one count.
    """

    def __init__(self, state_path: Path, rng: random.Random) -> None:
        self.state_path = state_path
        self.rng = rng

    def visits(self) -> dict[Pair, int]:
        """Return the visit count of every pair (0 for pairs never visited)."""
        counts = dict.fromkeys(all_pairs(), 0)
        if not self.state_path.exists():
            return counts
        try:
            state = _State.model_validate_json(self.state_path.read_bytes())
        except ValidationError as exc:
            raise StateError(f"{self.state_path}: {exc}") from exc
        for visit in state.visits:
            counts[visit.pair] = visit.visits
        return counts

    def pick(self) -> Pair:
        """Pick a pair at random, with weight `1 / (1 + visits)`. Does not record a visit."""
        counts = self.visits()
        pairs = all_pairs()
        weights = [1 / (1 + counts[p]) for p in pairs]
        return self.rng.choices(pairs, weights=weights, k=1)[0]

    def record_visit(self, pair: Pair) -> None:
        """Add one visit to `pair`, in either order. Raise `ValueError` for a bad pair."""
        key = canonical_pair(pair)
        counts = self.visits()
        counts[key] += 1
        state = _State(
            visits=[_Visit(pair=p, visits=n) for p, n in sorted(counts.items()) if n > 0]
        )
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(
            dir=self.state_path.parent, prefix=f".{self.state_path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(state.model_dump_json(indent=2).encode())
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, self.state_path)
        except BaseException:
            Path(name).unlink(missing_ok=True)
            raise
