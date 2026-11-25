"""Tests for the Lean runner (build plan doc-5wh, Task 9).

These compile real Lean against the repo's mathlib in `lean/.lake`, so each
takes a few seconds. Source that a model wrote may run `#eval` with IO, so the
runner compiles it with no network and a read-only view of the files it needs.
"""

import time
from pathlib import Path

from minilab.lean.runner import check_lean, lean_project

SMOKE = """import Mathlib.Data.Nat.Basic

theorem minilab_smoke_check : (1 : ℕ) + 1 = 2 := rfl
"""

SORRY = """import Mathlib.Data.Nat.Basic

theorem one_eq_two : (1 : ℕ) = 2 := by sorry
"""

CHOICE = """import Mathlib.Data.Nat.Basic

namespace Probe

theorem by_choice (α : Type) [h : Nonempty α] : ∃ _x : α, True :=
  ⟨Classical.choice h, trivial⟩

theorem by_rfl : (2 : ℕ) = 2 := rfl

private theorem hidden : True := trivial

end Probe
"""

WRONG = """import Mathlib.Data.Nat.Basic

theorem wrong : (1 : ℕ) + 1 = 3 := rfl
"""

LOOP = """partial def spin (n : Nat) : Nat := spin (n + 1)

theorem fine : True := trivial

#eval spin 0
"""


def _checks_left() -> set[Path]:
    return set(lean_project().glob("MinilabCheck_*.lean"))


def test_smoke_theorem_ok() -> None:
    before = _checks_left()
    result = check_lean(SMOKE, timeout_s=120)
    assert result.ok, result.errors
    assert result.errors == []
    assert not result.has_sorry
    assert not result.timed_out
    assert result.axioms_by_theorem == {"minilab_smoke_check": []}
    assert result.axioms == []
    assert _checks_left() == before


def test_sorry_detected() -> None:
    result = check_lean(SORRY, timeout_s=120)
    assert result.has_sorry
    assert "sorryAx" in result.axioms
    assert result.errors == []
    assert not result.timed_out


def test_axioms_listed() -> None:
    result = check_lean(CHOICE, timeout_s=120)
    assert result.ok, result.errors
    assert "Classical.choice" in result.axioms_by_theorem["Probe.by_choice"]
    assert result.axioms_by_theorem["Probe.by_rfl"] == []
    assert result.axioms_by_theorem["Probe.hidden"] == []
    assert "Classical.choice" in result.axioms
    assert not result.has_sorry


def test_errors_reported_with_line() -> None:
    result = check_lean(WRONG, timeout_s=120)
    assert not result.ok
    assert result.errors
    assert result.errors[0].startswith("3:")
    assert not result.timed_out


def test_lean_timeout_is_not_a_verdict() -> None:
    before = _checks_left()
    start = time.monotonic()
    result = check_lean(LOOP, timeout_s=5)
    elapsed = time.monotonic() - start
    assert result.timed_out
    assert not result.ok
    assert not result.has_sorry
    assert result.axioms == []
    assert result.axioms_by_theorem == {}
    assert elapsed < 20
    assert _checks_left() == before


def test_eval_cannot_write_files(tmp_path: Path) -> None:
    target = tmp_path / "written.txt"
    source = f'#eval (IO.FS.writeFile "{target}" "x" : IO Unit)\n'
    result = check_lean(source, timeout_s=60)
    assert not target.exists()
    assert not result.ok
    # Lean ran the #eval and the write failed, naming the file.
    assert any(str(target) in error for error in result.errors), result.errors


# A false statement proved from a custom axiom. Each must show the axiom.
CUSTOM_AXIOM = """axiom cheat : False

theorem bad : (1 : Nat) = 2 := cheat.elim
"""

# The regex that finds theorem lines misses `open X in theorem`.
CUSTOM_AXIOM_OPEN_IN = """import Mathlib.Data.Nat.Basic

axiom cheat : False

open Nat in theorem bad2 : (1 : ℕ) = 2 := cheat.elim
"""

# No theorem at all, only the axiom.
AXIOM_ONLY = """axiom cheat : False
"""

# `#exit` stops Lean before anything appended to the source can run.
EXIT_EARLY = """axiom cheat : False

theorem bad : (1 : Nat) = 2 := cheat.elim

#exit
"""


def test_custom_axiom_listed() -> None:
    result = check_lean(CUSTOM_AXIOM, timeout_s=120)
    assert result.ok, result.errors
    assert "cheat" in result.axioms
    assert result.axioms_by_theorem["bad"] == ["cheat"]


def test_custom_axiom_behind_open_in_listed() -> None:
    result = check_lean(CUSTOM_AXIOM_OPEN_IN, timeout_s=120)
    assert result.ok, result.errors
    assert "cheat" in result.axioms
    assert result.axioms_by_theorem["bad2"] == ["cheat"]


def test_declared_axiom_listed_without_theorem() -> None:
    result = check_lean(AXIOM_ONLY, timeout_s=120)
    assert "cheat" in result.axioms


def test_axiom_check_that_never_ran_fails_closed() -> None:
    result = check_lean(EXIT_EARLY, timeout_s=120)
    assert not result.ok
    assert any("axiom check" in error for error in result.errors), result.errors
