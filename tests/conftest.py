"""Shared pytest configuration for minilab."""

from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).parent


def _has_test_modules() -> bool:
    return any(_TESTS_DIR.rglob("test_*.py"))


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Treat an empty suite as success only while no test modules exist.

    Pytest exits 5 when it collects nothing. Before the first test module lands
    that is the expected state, so it exits 0. Once any test module exists, an
    empty collection means something is wrong, and the exit code stays 5.
    """
    if exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED and not _has_test_modules():
        session.exitstatus = pytest.ExitCode.OK
