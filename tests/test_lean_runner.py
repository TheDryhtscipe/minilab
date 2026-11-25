"""Tests for the Lean runner (build plan doc-5wh, Task 9).

These compile real Lean against the repo's mathlib in `lean/.lake`, so each
takes a few seconds. Source that a model wrote may run `#eval` with IO, so the
runner compiles it with no network and a read-only view of the files it needs.
"""

import time
from pathlib import Path

from minilab.lean.runner import check_lean, lean_project, theorem_names

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


def test_theorem_names_follow_namespaces_and_skip_comments() -> None:
    source = """
-- theorem in_line_comment : True := trivial
/- theorem in_block_comment : True := trivial -/
theorem top : True := trivial
namespace A.B
@[simp] private theorem inner (n : ℕ) : n = n := rfl
section S
lemma in_section : True := trivial
end S
end A.B
noncomputable section
protected theorem C.qualified : True := trivial
end
theorem _root_.rooted : True := trivial
"""
    assert theorem_names(source) == [
        "top",
        "A.B.inner",
        "A.B.in_section",
        "C.qualified",
        "rooted",
    ]
