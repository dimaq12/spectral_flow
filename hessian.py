"""
sft.hessian — 2nd-order perturbation: H = ∂²λ/∂k².

╔══════════════════════════════════════════════════════════════════════╗
║  INTENT                                                             ║
║  Second-order sensitivity of eigenvalues to parameters.             ║
║  H(i, j, l) = ∂²λᵢ/∂kⱼ∂kₗ — a (N, M, M) tensor.                  ║
║  Two methods: finite-difference (memoized) and analytic formula.    ║
║  Also: sparsity analysis, directional curvature.                    ║
╚══════════════════════════════════════════════════════════════════════╝
║  CONCEPT                                                            ║
║  • hessian (FD):   central 2nd-order finite difference.            ║
║    Caches eigh calls — from 4M² down to ~2M unique ones.           ║
║  • hessian_analytic (perturbation theory):                          ║
║    ∂²λᵢ/∂kⱼ∂kₗ = 2 Σ_{p≠i} (vᵢᵀBⱼv_p)(v_pᵀBₗvᵢ)/(λᵢ−λ_p)        ║
║    Requires non-degenerate spectrum.  Fully vectorized.             ║
╚══════════════════════════════════════════════════════════════════════╝
║  DEPENDENCIES                                                       ║
║  ┌── sft.core.OperatorFamily                                       ║
║  └── scipy.linalg (eigh)                                           ║
╚══════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations
import numpy as np
from scipy import linalg
from .core import OperatorFamily


def hessian(family: OperatorFamily, k0: np.ndarray | None = None,
            eps: float = 1e-4) -> np.ndarray:
    """H(i,j,l) via 2nd-order FD with eigh memoization.  Returns (N,M,M)."""
    if k0 is None:
        k0 = np.zeros(family.M)
    M = family.M; e = np.eye(M) * eps; cache = {}
    def _spectrum(kv):
        key = tuple(np.round(kv / eps).astype(int))
        if key not in cache:
            cache[key] = np.sort(linalg.eigh(family.build(kv))[0])
        return cache[key]
    H = np.zeros((family.N, M, M))
    for j in range(M):
        d1 = e[j]
        for l in range(M):
            d2 = e[l]
            H[:, j, l] = (_spectrum(k0 + d1 + d2) - _spectrum(k0 + d1 - d2)
                          - _spectrum(k0 - d1 + d2) + _spectrum(k0 - d1 - d2)) / (4.0 * eps * eps)
    return H


def hessian_analytic(family: OperatorFamily, k0: np.ndarray | None = None) -> np.ndarray:
    """Analytic 2nd-order perturbation, fully vectorized.  Returns (N,M,M)."""
    if k0 is None:
        k0 = np.zeros(family.M)
    A = family.build(k0); lam, vecs = linalg.eigh(A)
    N, M = family.N, family.M
    if M == 0:
        return np.zeros((N, 0, 0))
    Bstack = family._basis_stack
    VBV = vecs.T @ Bstack @ vecs
    denom = lam[:, None] - lam[None, :]
    denom[np.abs(denom) < 1e-12] = np.inf
    gap_inv = 1.0 / denom; np.fill_diagonal(gap_inv, 0.0)
    H = np.empty((N, M, M))
    for i in range(N):
        H[i] = 2.0 * (VBV[:, i, :] * gap_inv[i]) @ VBV[:, :, i].T
    return H


def hessian_sparsity(H: np.ndarray, tol: float = 1e-6) -> float:
    """Fraction of near-zero entries.  Higher → sparser curvature."""
    total = H.size
    nonzero = int(np.sum(np.abs(H) > tol))
    return 1.0 - nonzero / total if total > 0 else 1.0


def spectral_curvature(family: OperatorFamily, direction: np.ndarray,
                       k0: np.ndarray | None = None, eps: float = 1e-4) -> np.ndarray:
    """d²λ/dα² along direction (normalised).  Returns (N,) values."""
    d = direction / max(np.linalg.norm(direction), 1e-15)
    if k0 is None:
        k0 = np.zeros(family.M)
    lam0 = np.sort(linalg.eigh(family.build(k0))[0])
    lam_p = np.sort(linalg.eigh(family.build(k0 + eps * d))[0])
    lam_m = np.sort(linalg.eigh(family.build(k0 - eps * d))[0])
    return (lam_p - 2.0 * lam0 + lam_m) / (eps * eps)
