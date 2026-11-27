"""Tests for the test suite's Lean result cache (bead minilab-81c4fe).

Most use a fake `check_lean` that echoes its input back in `errors`, so a
result served for the wrong input shows. One compiles real Lean, to check that
a real result survives the trip through the cache unchanged.
"""

import os
from collections.abc import Mapping
from pathlib import Path

import pytest
from lean_cache import CachedLean, NonceSequence, install_lean_cache, parallel_live_refusal

from minilab.formalise import formaliser, vacuity
from minilab.lean import runner
from minilab.lean.runner import DEFAULT_TIMEOUT_S, LeanResult
from minilab.literature import loogle

_REAL_NEW_NONCE = formaliser.new_nonce

SMOKE = """import Mathlib.Data.Nat.Basic

theorem minilab_cache_check : (1 : ℕ) + 1 = 2 := rfl
"""


class Echo:
    """A fake `check_lean` that counts its calls and echoes what it was given."""

    def __init__(self, *, timed_out: bool = False) -> None:
        self.calls = 0
        self.timed_out = timed_out

    def __call__(
        self,
        source: str,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        *,
        project: Path | None = None,
        options: Mapping[str, str] | None = None,
    ) -> LeanResult:
        self.calls += 1
        given = f"{source}|{timeout_s}|{project}|{sorted((options or {}).items())}"
        return LeanResult(
            ok=False,
            errors=[given],
            has_sorry=False,
            axioms=[],
            timed_out=self.timed_out,
            axioms_by_theorem={},
            messages=[],
        )


def test_each_source_gets_its_own_result(tmp_path: Path) -> None:
    echo = Echo()
    cached = CachedLean(echo, tmp_path)
    first_a = cached("theorem a : True := trivial", 60)
    first_b = cached("theorem b : True := trivial", 60)
    again_a = cached("theorem a : True := trivial", 60)
    again_b = cached("theorem b : True := trivial", 60)
    assert again_a == first_a
    assert again_b == first_b
    assert first_a.errors[0].startswith("theorem a")
    assert first_b.errors[0].startswith("theorem b")
    assert echo.calls == 2


def test_workers_share_the_cache_directory(tmp_path: Path) -> None:
    echo = Echo()
    first = CachedLean(echo, tmp_path)("theorem a : True := trivial", 60)
    second = CachedLean(echo, tmp_path)("theorem a : True := trivial", 60)
    assert second == first
    assert echo.calls == 1


def test_timeout_project_and_options_are_part_of_the_key(tmp_path: Path) -> None:
    echo = Echo()
    cached = CachedLean(echo, tmp_path)
    source = "theorem a : True := trivial"
    results = [
        cached(source, 60),
        cached(source, 30),
        cached(source, 60, project=tmp_path),
        cached(source, 60, options={"maxHeartbeats": "400000"}),
        cached(source, 60, options={"maxHeartbeats": "200000"}),
    ]
    assert echo.calls == 5
    assert len({r.errors[0] for r in results}) == 5


def test_option_order_does_not_matter(tmp_path: Path) -> None:
    echo = Echo()
    cached = CachedLean(echo, tmp_path)
    cached("x", 60, options={"a": "1", "b": "2"})
    cached("x", 60, options={"b": "2", "a": "1"})
    assert echo.calls == 1


def test_a_timeout_is_never_stored(tmp_path: Path) -> None:
    echo = Echo(timed_out=True)
    cached = CachedLean(echo, tmp_path)
    assert cached("x", 1).timed_out
    assert cached("x", 1).timed_out
    assert echo.calls == 2
    assert list(tmp_path.iterdir()) == []


def test_an_entry_for_another_key_is_not_used(tmp_path: Path) -> None:
    # As if two keys had the same file name: the entry's own key must match.
    echo = Echo()
    cached = CachedLean(echo, tmp_path)
    cached("theorem a : True := trivial", 60)
    (entry,) = list(tmp_path.iterdir())
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other = CachedLean(echo, other_dir)
    other("theorem b : True := trivial", 60)
    (other_entry,) = [p for p in other_dir.iterdir() if p.is_file()]
    entry.write_text(other_entry.read_text())
    again = cached("theorem a : True := trivial", 60)
    assert again.errors[0].startswith("theorem a")
    assert echo.calls == 3


def test_a_real_result_comes_back_unchanged(tmp_path: Path) -> None:
    cached = CachedLean(runner.check_lean, tmp_path)
    first = cached(SMOKE, 120)
    second = cached(SMOKE, 120)
    assert first.ok, first.errors
    assert first.axioms_by_theorem == {"minilab_cache_check": []}
    assert second == first
    assert cached.misses == 1


def _skip_if_cache_off(request: pytest.FixtureRequest) -> None:
    if request.config.getoption("--no-lean-cache"):
        pytest.skip("this run has --no-lean-cache")


def test_lean_callers_use_the_cache_by_default(request: pytest.FixtureRequest) -> None:
    _skip_if_cache_off(request)
    for module in (formaliser, vacuity, loogle):
        assert isinstance(module.check_lean, CachedLean), module.__name__
    for module in (formaliser, vacuity):
        assert isinstance(module.new_nonce, NonceSequence), module.__name__


@pytest.mark.lean_uncached
def test_a_test_can_opt_out_of_the_cache() -> None:
    for module in (formaliser, vacuity, loogle):
        assert module.check_lean is runner.check_lean, module.__name__


def test_each_install_starts_the_nonces_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The fixture installs the cache for each test. Two tests that check the
    # same files in the same order must send Lean the same text, markers
    # included, so each install starts a fresh nonce sequence.
    cache = CachedLean(Echo(), tmp_path)
    install_lean_cache(monkeypatch, cache)
    first = [formaliser.new_nonce(), formaliser.new_nonce(), vacuity.new_nonce()]
    install_lean_cache(monkeypatch, cache)
    again = [formaliser.new_nonce(), formaliser.new_nonce(), vacuity.new_nonce()]
    assert again == first
    assert len(set(first)) == 3
    assert all(len(n) == 16 and int(n, 16) >= 0 for n in first)


def test_workers_share_one_cache_directory(
    lean_cache: CachedLean, tmp_path_factory: pytest.TempPathFactory
) -> None:
    base = tmp_path_factory.getbasetemp()
    if os.environ.get("PYTEST_XDIST_WORKER"):
        # Each worker's base temp is inside the run's; the cache is beside them.
        assert lean_cache.directory == base.parent / "lean-cache"
    else:
        assert lean_cache.directory == base / "lean-cache"


@pytest.mark.parametrize(
    ("markexpr", "workers", "refused"),
    [
        ("not live", 4, False),
        ("live", 4, True),
        ("live", 0, False),
        ("live or not live", 4, True),
        ("", 4, False),
    ],
)
def test_live_tests_run_in_one_process(markexpr: str, workers: int, refused: bool) -> None:
    assert (parallel_live_refusal(markexpr, workers) is not None) == refused


@pytest.mark.lean_uncached
def test_opting_out_keeps_nonces_random() -> None:
    assert formaliser.new_nonce is _REAL_NEW_NONCE
    assert vacuity.new_nonce is _REAL_NEW_NONCE
    assert formaliser.new_nonce() != formaliser.new_nonce()


def test_the_command_check_marker_is_a_nonce(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake(source: str, timeout_s: float = DEFAULT_TIMEOUT_S, **_: object) -> LeanResult:
        seen.append(source)
        return Echo()(source, timeout_s)

    monkeypatch.setattr(formaliser, "check_lean", fake)
    monkeypatch.setattr(formaliser, "new_nonce", lambda: "feedfacecafebeef")
    formaliser.command_check("theorem conjecture : True := trivial\n", timeout_s=60)
    assert "MINILAB_CMD_feedfacecafebeef_" in seen[0]
