"""The run manifest.

Each run writes one manifest when it starts, at `runs/<run_id>/manifest.json`.
It fixes what the run may do: its mode, its domain pairs, how it may read
memory (design doc-gmj, "Memory read policy"), its literature date cap, its
token budget per pair, and the model families it uses. A manifest is written
once and never changed.
"""

import os
import tempfile
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from minilab.records import Family, RunId, check_run_id
from minilab.scheduler import Pair, canonical_pair


class MemoryPolicy(StrEnum):
    """How a run may read the `minilab` memory vault."""

    blind = "blind"
    """No reads."""
    pair = "pair"
    """Only records for the same domain pair."""
    all = "all"
    """Every record."""


class ManifestExists(Exception):
    """The run already has a manifest."""


class RunManifest(BaseModel):
    """What a run is allowed to do, fixed when it starts.

    Pairs are stored in sorted order, so a pair has one form, and each pair may
    appear once. Times and dates must be given as such (or as ISO strings in
    JSON) and times must carry a time zone. `memory_before` needs the `pair`
    or `all` policy: a `blind` run reads nothing, so a date cap there would be
    a mistake. The model and its collections cannot be changed after creation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: RunId
    mode: Literal["plain", "structured"]
    pairs: Annotated[tuple[Pair, ...], Field(min_length=1)]
    memory_policy: MemoryPolicy
    memory_before: Annotated[AwareDatetime, Field(strict=True)] | None
    """Read only memories created before this time (MuninnDB's `before` filter)."""
    literature_before: Annotated[date, Field(strict=True)] | None
    """Literature search ignores anything first posted after this date."""
    budget_tokens_per_pair: Annotated[int, Field(strict=True, gt=0)]
    """Tokens in plus tokens out, over every model call for one pair."""
    families: Annotated[tuple[Family, ...], Field(min_length=1)]
    started: Annotated[AwareDatetime, Field(strict=True)]

    @field_validator("pairs")
    @classmethod
    def _canonical_unique_pairs(cls, value: tuple[Pair, ...]) -> tuple[Pair, ...]:
        pairs = tuple(canonical_pair(pair) for pair in value)
        seen: set[Pair] = set()
        for pair in pairs:
            if pair in seen:
                raise ValueError(f"pair {pair} appears more than once")
            seen.add(pair)
        return pairs

    @field_validator("families")
    @classmethod
    def _unique_families(cls, value: tuple[Family, ...]) -> tuple[Family, ...]:
        if len(set(value)) != len(value):
            raise ValueError(f"a family appears more than once: {value}")
        return value

    @model_validator(mode="after")
    def _no_cap_on_blind(self) -> "RunManifest":
        if self.memory_policy is MemoryPolicy.blind and self.memory_before is not None:
            raise ValueError("memory_before needs the pair or all policy; blind reads nothing")
        return self


def manifest_path(root: Path, run_id: str) -> Path:
    """Return the path of a run's manifest under the repo root `root`."""
    return root / "runs" / check_run_id(run_id) / "manifest.json"


def save_manifest(root: Path, manifest: RunManifest) -> Path:
    """Write a run's manifest and return its path.

    Raise `ManifestExists` if the run already has one; the file on disk is left
    unchanged. The file is written to a temporary name and then linked into
    place, so a reader never sees a partial manifest, and of two saves for one
    run exactly one succeeds. The manifest is validated again first, so one
    built with `model_construct` cannot bypass the checks.
    """
    manifest = RunManifest.model_validate(manifest.model_dump())
    target = manifest_path(root, manifest.run_id)
    if target.exists():
        raise ManifestExists(manifest.run_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=target.parent, prefix=".manifest.", suffix=".tmp")
    temp = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(manifest.model_dump_json(indent=2).encode())
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp, target)
        except FileExistsError as exc:
            raise ManifestExists(manifest.run_id) from exc
    finally:
        temp.unlink(missing_ok=True)
    dir_fd = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    return target


def load_manifest(root: Path, run_id: str) -> RunManifest:
    """Read a run's manifest.

    Raise `FileNotFoundError` if the run has none, and `ValueError` if the file
    holds another run's manifest.
    """
    path = manifest_path(root, run_id)
    manifest = RunManifest.model_validate_json(path.read_bytes())
    if manifest.run_id != run_id:
        raise ValueError(f"{path}: holds the manifest of run {manifest.run_id}")
    return manifest
