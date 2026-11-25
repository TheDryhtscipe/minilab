import Mathlib

open Finset

/-! S8 from the crossconj-ml pilot (information theory x convex optimisation):
every discrete memoryless channel has a capacity-achieving input whose support
has at most rank(W) letters.

This is a reconstruction, not the pilot's Lean file. It is written from the
pilot's English statement (crossconj-ml/structured.py) and its run notes, which
say the pilot's Lean proof took the KKT conditions as a hypothesis. Here that
hypothesis is `hKKT`: every input letter's divergence D(W(.|x) || q) is at most
a constant C, with equality on the support of p. The record's assumptions do
not state it, so the hypothesis audit must list it. -/

/-- `W` is a channel: each row is a probability distribution on the outputs. -/
def IsChannel {m n : ℕ} (W : Matrix (Fin m) (Fin n) ℝ) : Prop :=
  (∀ i j, 0 ≤ W i j) ∧ ∀ i, ∑ j, W i j = 1

/-- `p` is a probability distribution on the inputs. -/
def IsDist {m : ℕ} (p : Fin m → ℝ) : Prop :=
  (∀ i, 0 ≤ p i) ∧ ∑ i, p i = 1

/-- The output distribution of `W` under the input distribution `p`. -/
noncomputable def outDist {m n : ℕ} (W : Matrix (Fin m) (Fin n) ℝ) (p : Fin m → ℝ) :
    Fin n → ℝ :=
  fun j => ∑ i, p i * W i j

/-- The divergence D(W(.|i) || q) of row `i` from `q`. -/
noncomputable def rowKL {m n : ℕ} (W : Matrix (Fin m) (Fin n) ℝ) (q : Fin n → ℝ)
    (i : Fin m) : ℝ :=
  ∑ j, W i j * Real.log (W i j / q j)

/-- The mutual information I(p, W). -/
noncomputable def mutualInfo {m n : ℕ} (W : Matrix (Fin m) (Fin n) ℝ) (p : Fin m → ℝ) : ℝ :=
  ∑ i, p i * rowKL W (outDist W p) i

/-- `p` achieves the capacity of `W`. -/
def IsCapacityAchieving {m n : ℕ} (W : Matrix (Fin m) (Fin n) ℝ) (p : Fin m → ℝ) : Prop :=
  IsDist p ∧ ∀ p', IsDist p' → mutualInfo W p' ≤ mutualInfo W p

theorem conjecture {m n : ℕ} (W : Matrix (Fin m) (Fin n) ℝ) (hW : IsChannel W)
    (p : Fin m → ℝ) (hp : IsCapacityAchieving W p)
    (hKKT : ∃ C : ℝ, ∀ i, rowKL W (outDist W p) i ≤ C ∧ (0 < p i → rowKL W (outDist W p) i = C)) :
    ∃ p' : Fin m → ℝ, IsCapacityAchieving W p' ∧ (univ.filter (fun i => p' i ≠ 0)).card ≤ W.rank := by
  sorry
