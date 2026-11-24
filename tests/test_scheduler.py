"""Tests for the pair scheduler (build plan Task 5)."""

import json
import random
from collections import Counter
from pathlib import Path

import pytest

from minilab.scheduler import DOMAINS, Scheduler, StateError, all_pairs


def test_domains_in_design_order() -> None:
    assert DOMAINS == (
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


def test_45_pairs_no_self_pairs() -> None:
    pairs = all_pairs()
    assert len(pairs) == 45
    assert len(set(pairs)) == 45
    assert all(a < b for a, b in pairs)  # each pair in canonical (sorted) order
    assert pairs == sorted(pairs)
    assert {d for pair in pairs for d in pair} == set(DOMAINS)
    assert all(pairs.count(p) == 1 and (p[1], p[0]) not in pairs for p in pairs)


def test_weight_falls_with_visits(tmp_path: Path) -> None:
    state = tmp_path / "state" / "pairs.json"
    visited = ("information-theory", "optimisation")
    fresh = ("kernels", "probability")
    setup = Scheduler(state, random.Random(0))
    for _ in range(9):
        setup.record_visit(visited)

    scheduler = Scheduler(state, random.Random(1234))
    counts = Counter(scheduler.pick() for _ in range(10_000))
    # Weight 1/10 against 1: expect about 23 against 227 over 10,000 picks.
    assert counts[visited] < counts[fresh]
    assert counts[visited] < 0.3 * counts[fresh]
    assert set(counts) <= set(all_pairs())


def test_seeded_pick_is_deterministic(tmp_path: Path) -> None:
    state = tmp_path / "pairs.json"
    Scheduler(state, random.Random(0)).record_visit(("kernels", "probability"))
    first = [Scheduler(state, random.Random(7)).pick() for _ in range(1)]
    a = Scheduler(state, random.Random(7))
    b = Scheduler(state, random.Random(7))
    seq_a = [a.pick() for _ in range(50)]
    seq_b = [b.pick() for _ in range(50)]
    assert seq_a == seq_b
    assert seq_a[0] == first[0]
    assert len(set(seq_a)) > 1  # the rng is actually used
    c = Scheduler(state, random.Random(8))
    assert [c.pick() for _ in range(50)] != seq_a


def test_unknown_domain_rejected(tmp_path: Path) -> None:
    state = tmp_path / "pairs.json"
    scheduler = Scheduler(state, random.Random(0))
    with pytest.raises(ValueError):
        scheduler.record_visit(("kernels", "alchemy"))
    with pytest.raises(ValueError):
        scheduler.record_visit(("kernels", "kernels"))
    assert not state.exists()


def test_record_visit_persists_and_ignores_order(tmp_path: Path) -> None:
    state = tmp_path / "state" / "pairs.json"
    scheduler = Scheduler(state, random.Random(0))
    assert not state.exists()
    scheduler.pick()
    assert not state.exists()  # picking alone does not create the state file
    scheduler.record_visit(("optimisation", "information-theory"))
    scheduler.record_visit(("information-theory", "optimisation"))
    assert state.is_file()
    assert Scheduler(state, random.Random(0)).visits()[("information-theory", "optimisation")] == 2
    assert sum(Scheduler(state, random.Random(0)).visits().values()) == 2
    assert sorted(p.name for p in state.parent.iterdir()) == ["pairs.json"]


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        json.dumps({"visits": [{"pair": ["kernels", "alchemy"], "visits": 1}]}),
        json.dumps({"visits": [{"pair": ["kernels", "probability"], "visits": -1}]}),
        json.dumps(
            {
                "visits": [
                    {"pair": ["kernels", "probability"], "visits": 1},
                    {"pair": ["probability", "kernels"], "visits": 2},
                ]
            }
        ),
    ],
)
def test_bad_state_file_refused(tmp_path: Path, content: str) -> None:
    state = tmp_path / "pairs.json"
    state.write_text(content)
    scheduler = Scheduler(state, random.Random(0))
    with pytest.raises(StateError):
        scheduler.pick()
    with pytest.raises(StateError):
        scheduler.record_visit(("kernels", "probability"))
    assert state.read_text() == content
