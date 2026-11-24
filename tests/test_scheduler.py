"""Tests for the pair scheduler (build plan Task 5)."""

import json
import multiprocessing
import random
import stat
from collections import Counter
from pathlib import Path

import pytest

from minilab.scheduler import DOMAINS, Pair, Scheduler, StateError, all_pairs, canonical_pair


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
    assert 0.06 < counts[visited] / counts[fresh] < 0.15
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
    assert sorted(p.name for p in state.parent.iterdir()) == [".pairs.json.lock", "pairs.json"]


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


def _state(*entries: tuple[str, str, object]) -> str:
    return json.dumps({"visits": [{"pair": [a, b], "visits": n} for a, b, n in entries]})


@pytest.mark.parametrize(
    "content",
    [
        # The same sorted pair twice.
        _state(("kernels", "probability", 1), ("kernels", "probability", 2)),
        # Repeated JSON keys, which a plain parser resolves silently.
        '{"visits": [{"pair": ["kernels", "probability"], "visits": 7, "visits": 2}]}',
        '{"visits": [{"pair": ["kernels", "probability"], "visits": 7}], "visits": []}',
        # Values that lax parsing would coerce.
        _state(("kernels", "probability", "3")),
        _state(("kernels", "probability", True)),
        _state(("kernels", "probability", 1e3)),
        _state(("kernels", "probability", 10**30)),
    ],
)
def test_more_bad_state_files_refused(tmp_path: Path, content: str) -> None:
    test_bad_state_file_refused(tmp_path, content)


def test_weights_match_formula(tmp_path: Path) -> None:
    state = tmp_path / "pairs.json"
    setup = Scheduler(state, random.Random(0))
    visited = ("information-theory", "optimisation")
    for _ in range(3):
        setup.record_visit(visited)
    scheduler = Scheduler(state, random.Random(99))
    counts = Counter(scheduler.pick() for _ in range(100_000))
    assert len(counts) == 45
    fresh = [counts[p] for p in all_pairs() if p != visited]
    mean_fresh = sum(fresh) / len(fresh)
    # Weight 1/4 against 1.
    assert 0.22 < counts[visited] / mean_fresh < 0.28


def test_pick_from_explicit_pairs(tmp_path: Path) -> None:
    state = tmp_path / "pairs.json"
    scheduler = Scheduler(state, random.Random(3))
    chosen: list[Pair] = [("optimisation", "information-theory"), ("kernels", "probability")]
    picks = {scheduler.pick(chosen) for _ in range(200)}
    assert picks == {("information-theory", "optimisation"), ("kernels", "probability")}
    for bad in (
        [],
        [("kernels", "probability"), ("probability", "kernels")],
        [("kernels", "alchemy")],
    ):
        with pytest.raises(ValueError):
            scheduler.pick(bad)


@pytest.mark.parametrize(
    "bad",
    [
        ("kernels",),
        ("kernels", "probability", "optimisation"),
        {"kernels": 1, "probability": 2},
        ("kernels", None),
        (b"kernels", "probability"),
    ],
)
def test_canonical_pair_rejects_odd_inputs(bad: object) -> None:
    with pytest.raises(ValueError, match="pair"):
        canonical_pair(bad)  # pyright: ignore[reportArgumentType]


def test_unreadable_state_is_state_error(tmp_path: Path) -> None:
    state = tmp_path / "pairs.json"
    state.mkdir()
    with pytest.raises(StateError):
        Scheduler(state, random.Random(0)).pick()


def test_state_file_mode_and_no_leftovers(tmp_path: Path) -> None:
    state = tmp_path / "state" / "pairs.json"
    Scheduler(state, random.Random(0)).record_visit(("kernels", "probability"))
    assert stat.S_IMODE(state.stat().st_mode) == 0o644
    assert not [p for p in state.parent.iterdir() if p.name.endswith(".tmp")]


def _visit_many(state: str, n: int) -> None:
    scheduler = Scheduler(Path(state), random.Random(0))
    for _ in range(n):
        scheduler.record_visit(("kernels", "probability"))


def test_concurrent_visits_all_counted(tmp_path: Path) -> None:
    state = tmp_path / "pairs.json"
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_visit_many, args=(str(state), 10)) for _ in range(4)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=60)
        assert proc.exitcode == 0
    assert Scheduler(state, random.Random(0)).visits()[("kernels", "probability")] == 40
