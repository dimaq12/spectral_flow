#!/usr/bin/env python3
"""
sft/demo.py — SFT showcase, analogous to numpy.fft.

Demonstrates the full SFT pipeline:
  1. Build operator family
  2. Compute kernel
  3. Predict spectrum
  4. Inverse design
  5. Structural complexity
  6. Algebra (⊕, ⊗)
  7. Topology (monodromy, holonomy)
  8. Hessian (2nd order)
  9. Expectation (Jensen gap)
"""
from __future__ import annotations
import sys
import numpy as np

# Add parent directory for local import
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import sft

# ═══════════════════════════════════════════════════════════
# 1. BUILD AN OPERATOR FAMILY
# ═══════════════════════════════════════════════════════════

print("═══ 1. BUILD ═══\n")

# Random family: 50×50 matrices, 20 parameters
fam = sft.families.random(N=50, M=20, seed=42)
print(f"   N={fam.N}, M={fam.M}")
print(f"   Ref spectrum λ₀: [{', '.join(f'{x:.3f}' for x in fam.lam0[:5])}...]")

# ═══════════════════════════════════════════════════════════
# 2. SFT KERNEL
# ═══════════════════════════════════════════════════════════

print("\n═══ 2. KERNEL ═══\n")

W = fam.W
print(f"   W shape: {W.shape}   ({fam.N} eigenvalues × {fam.M} parameters)")
print(f"   rank(W) = {fam.W_rank}")
print(f"   complexity = rank/N = {fam.complexity:.3f}")
print(f"   κ(W) = {fam.condition_number():.1f}")
print(f"   dim(ker(W)) = {fam.isospectral_dimension()}")

# ═══════════════════════════════════════════════════════════
# 3. PREDICT
# ═══════════════════════════════════════════════════════════

print("\n═══ 3. PREDICT ═══\n")

rng = np.random.default_rng(1)
dk = rng.standard_normal(fam.M) * 0.01
lam_exact = fam.spectrum(dk)
lam_pred = fam.predict(dk)
err_pred = float(np.max(np.abs(lam_exact - lam_pred)))
print(f"   Perturbation: ||dk|| = {np.linalg.norm(dk):.4f}")
print(f"   max|λ_exact − λ_pred| = {err_pred:.2e}")

# ═══════════════════════════════════════════════════════════
# 4. INVERSE DESIGN
# ═══════════════════════════════════════════════════════════

print("\n═══ 4. INVERSE ═══\n")

target = np.sort(fam.lam0 + np.linspace(-0.15, 0.15, fam.N))
k_design, err_design, ok = fam.inverse(target, steps=40, alpha=0.5, refresh_every=5)
print(f"   Converged: {ok}")
print(f"   Final error: {err_design:.2e}")

# ═══════════════════════════════════════════════════════════
# 5. NULLSPACE
# ═══════════════════════════════════════════════════════════

print("\n═══ 5. NULLSPACE ═══\n")

ker = sft.nullspace(fam)  # free function — one of the few kept at top level
print(f"   ker(W) basis shape: {ker.shape}  "
      f"({fam.isospectral_dimension()} silent directions)")
if ker.shape[1] > 0:
    dk_ker = ker[:, 0] * 0.2
    lam_k0 = fam.spectrum(np.zeros(fam.M))
    lam_ker = fam.spectrum(dk_ker)
    shift = float(np.max(np.abs(lam_ker - lam_k0)))
    print(f"   Moving k by 0.2 along ker(W):  max|Δλ| = {shift:.2e}")
    print(f"   The operator does NOT hear this direction.")

# ═══════════════════════════════════════════════════════════
# 6. ALGEBRA
# ═══════════════════════════════════════════════════════════

print("\n═══ 6. ALGEBRA ═══\n")

fam_a = sft.families.avoided_crossing_2x2(Delta=0.3)
fam_b = sft.families.random(N=3, M=2, seed=7)

# Direct sum
fam_sum = sft.algebra.direct_sum(fam_a, fam_b)
print(f"   A ⊕ B:  N={fam_sum.N}, M={fam_sum.M}")
print(f"   λ(A ⊕ B) = λ(A) ∪ λ(B): {[f'{x:.2f}' for x in fam_sum.lam0]}")

# Tensor sum
fam_ten = sft.algebra.tensor_sum(fam_a, fam_b)
print(f"   A ⊗ B:  N={fam_ten.N}, M={fam_ten.M}  (λ = λ_A + λ_B)")
lam_a = fam_a.lam0
lam_b = fam_b.lam0
lam_ten_true = np.sort(np.add.outer(lam_a, lam_b).ravel())
lam_ten = fam_ten.lam0
print(f"   λ(A⊗B) match: {float(np.max(np.abs(np.sort(lam_ten) - np.sort(lam_ten_true)))):.2e}")

# ═══════════════════════════════════════════════════════════
# 7. TOPOLOGY
# ═══════════════════════════════════════════════════════════

print("\n═══ 7. TOPOLOGY ═══\n")

# Berry holonomy on avoided crossing
def make_loop_2d(radius=0.4, n_pts=60):
    theta = np.linspace(0, 2 * np.pi, n_pts)
    return [np.array([radius * np.cos(t), radius * np.sin(t)]) for t in theta]

loop = make_loop_2d(0.4)
hol = sft.topology.berry_holonomy(fam_a, loop, level=1)
print(f"   Berry holonomy (2×2 avoided crossing): {hol:+d}")
print(f"     −1 = Möbius topology, eigenvector flips sign over 2π loop")

# Monodromy — Hermitian avoided crossing (no EP inside → no braiding)
tracked, swapped = sft.topology.monodromy(fam_a, loop)
print(f"   Monodromy: eigenvalues braided? {len(swapped) > 0}  (swapped pairs: {swapped})")
if len(swapped) == 0:
    print(f"     (Hermitian: no true EP → levels repel, never cross)")

# Non-Hermitian EP demo (separate from sft.families)
print(f"\n   ── Non-Hermitian EP (direct) ──")
n_phi = 60
r_ep = 0.5
phi_vals = np.linspace(0, 2*np.pi, n_phi)
lams_nh = np.zeros((n_phi, 2), dtype=complex)
for idx, phi in enumerate(phi_vals):
    eps = r_ep * np.cos(phi)
    delta = r_ep * np.sin(phi)
    H_nh = np.array([[0, 1], [eps + 1j*delta, 0]])
    lams_nh[idx] = np.linalg.eigvals(H_nh)  # not tracking by continuity here, just raw
print(f"   H = [[0,1],[ε+iδ,0]], loop around (0,0)")
print(f"   λ(φ=0)  = ({lams_nh[0,0]:.3f}, {lams_nh[0,1]:.3f})")
print(f"   λ(φ=2π) = ({lams_nh[-1,0]:.3f}, {lams_nh[-1,1]:.3f})")
print(f"   ★ Eigenvalues SWAP after 2π loop around the EP")

# ═══════════════════════════════════════════════════════════
# 8. HESSIAN
# ═══════════════════════════════════════════════════════════

print("\n═══ 8. HESSIAN ═══\n")

H = sft.hessian.hessian(fam_a)
sp = sft.hessian.hessian_sparsity(H)
print(f"   Hessian shape: {H.shape}")
print(f"   Sparsity: {sp:.3f}")
print(f"   ∂²λ₁/∂k₀² = {H[1, 0, 0]:.3f}  ∂²λ₁/∂k₁² = {H[1, 1, 1]:.3f}")

# ═══════════════════════════════════════════════════════════
# 9. EXPECTATION (Jensen gap)
# ═══════════════════════════════════════════════════════════

print("\n═══ 9. EXPECTATION ═══\n")

for sigma in [0.01, 0.05, 0.10]:
    res = sft.algebra.expectation(fam, np.zeros(fam.M), sigma, n_samples=200)
    print(f"   σ={sigma:.2f}: Jensen gap = {res['gap']:.4f}")

# ═══════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════

print("\n═══ SFT — Spectral Flow Transform ═══\n")
print(f"   import sft")
print(f"   fam = sft.families.random(100, 30)")
print(f"   lam = fam.predict(perturbation)")
print(f"   k, err, ok = fam.inverse(target_spectrum)")
print(f"   R = fam.W_rank")
print(f"   H = sft.hessian.hessian(fam)")
print(f"   Z₂ = sft.topology.berry_holonomy(fam, loop)")
print(f"   A⊕B = sft.algebra.direct_sum(A, B)")
print()
print(f"   One formula.  W(i,j) = v_i^T · B_j · v_i.")
print(f"   One kernel.  All operator families.")
