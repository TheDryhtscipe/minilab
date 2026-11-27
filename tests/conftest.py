"""Shared pytest configuration for minilab."""

import os
from pathlib import Path

import pytest
from lean_cache import CachedLean, NonceSequence

from minilab.formalise import formaliser, vacuity
from minilab.lean import runner
from minilab.literature import loogle

_TESTS_DIR = Path(__file__).parent


def _has_test_modules() -> bool:
    return any(_TESTS_DIR.rglob("test_*.py")) or any(_TESTS_DIR.rglob("*_test.py"))


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Treat an empty suite as success only while no test modules exist.

    Pytest exits 5 when it collects nothing. Before the first test module lands
    that is the expected state, so it exits 0. Once any test module exists, an
    empty collection means something is wrong, and the exit code stays 5.
    """
    if exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED and not _has_test_modules():
        session.exitstatus = pytest.ExitCode.OK


# --- The Lean result cache (bead minilab-81c4fe; see lean_cache.py) ---------------------------

# The modules that call `check_lean` outside the runner. The runner's own tests
# import `check_lean` directly, so they always run Lean.
_LEAN_CALLERS = (formaliser, vacuity, loogle)
# The modules that tag Lean runs with `new_nonce`; see `NonceSequence`.
_NONCE_USERS = (formaliser, vacuity)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--no-lean-cache",
        action="store_true",
        help="run Lean for every check, even one already made in this run",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "lean_uncached: run Lean for every check this test makes, never the cache"
    )


@pytest.fixture(scope="session")
def lean_cache(tmp_path_factory: pytest.TempPathFactory) -> CachedLean:
    """The run's Lean cache, in a directory every test worker of the run shares."""
    base = tmp_path_factory.getbasetemp()
    # Under pytest-xdist each worker has its own base temp inside the run's.
    shared = base.parent if os.environ.get("PYTEST_XDIST_WORKER") else base
    directory = shared / "lean-cache"
    directory.mkdir(exist_ok=True)
    return CachedLean(runner.check_lean, directory)


@pytest.fixture(autouse=True)
def _cache_lean(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, lean_cache: CachedLean
) -> None:
    if request.config.getoption("--no-lean-cache"):
        return
    if request.node.get_closest_marker("lean_uncached") is not None:
        return
    for module in _LEAN_CALLERS:
        monkeypatch.setattr(module, "check_lean", lean_cache)
    nonces = NonceSequence()
    for module in _NONCE_USERS:
        monkeypatch.setattr(module, "new_nonce", nonces)
