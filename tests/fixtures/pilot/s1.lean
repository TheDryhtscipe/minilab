import Mathlib

section

open Finset

/-! S1 from the crossconj-ml pilot (information theory x convex optimisation), the
Tsallis-Pinsker inequality: for q in (0, 1), the Bregman divergence of
phi_q(p) = (sum_i p_i^q - 1) / (q - 1) on the simplex satisfies
B_q(p, r) >= (q / 2) ||p - r||_1^2.

This is a reconstruction, not the pilot's Lean file. It is written from the
pilot's English statement (crossconj-ml/structured.py, S1). The pilot states
q in (0, 1], with phi_1 = sum p log p; this file leaves out q = 1, which needs
the separate formula. The pilot's Lean proof counted as real but trivial, so
the triviality screen should flag S1. -/

/-- `p` is a probability distribution. -/
def IsDist {n : ℕ} (p : Fin n → ℝ) : Prop :=
  (∀ i, 0 ≤ p i) ∧ ∑ i, p i = 1

/-- The Tsallis potential phi_q(p) = (sum_i p_i^q - 1) / (q - 1). -/
noncomputable def tsallisPhi {n : ℕ} (q : ℝ) (p : Fin n → ℝ) : ℝ :=
  (∑ i, p i ^ q - 1) / (q - 1)

/-- The Bregman divergence of phi_q between `p` and `r`. -/
noncomputable def bregman {n : ℕ} (q : ℝ) (p r : Fin n → ℝ) : ℝ :=
  tsallisPhi q p - tsallisPhi q r - ∑ i, q * r i ^ (q - 1) / (q - 1) * (p i - r i)

theorem conjecture {n : ℕ} (q : ℝ) (hq0 : 0 < q) (hq1 : q < 1) (p r : Fin n → ℝ)
    (hp : IsDist p) (hr : IsDist r) (hrpos : ∀ i, 0 < r i) :
    q / 2 * (∑ i, |p i - r i|) ^ 2 ≤ bregman q p r := by
  sorry

end
