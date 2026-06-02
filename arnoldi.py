"""
sft.arnoldi — Arnoldi iteration: Krylov subspace, Hessenberg, Ritz values.

╔══════════════════════════════════════════════════════════════════════╗
║  INTENT                                                             ║
║  arnoldi_iteration:   A·V_k ≈ V_{k+1}·H — orthonormal Krylov basis.║
║  ritz_eigenvalues:    eigenvalues of H — approximate eigenvalues of A.║
║  krylov_solve:        A·x = b via Krylov-GMRES.                    ║
╚══════════════════════════════════════════════════════════════════════╝
"""
import numpy as np
from scipy import linalg


def arnoldi_iteration(A_fn, v0: np.ndarray, m: int, eps: float = 1e-12):
    """Krylov subspace: V (n×m), H (m×m) Hessenberg.  A·V ≈ V·H + v×e_m."""
    n = len(v0); v0 = np.asarray(v0, np.float64).copy()
    beta = np.linalg.norm(v0)
    if beta < eps: return np.zeros((n, 0)), np.zeros((0, 0))
    V = np.zeros((n, m)); H = np.zeros((m, m)); V[:, 0] = v0 / beta
    for j in range(m):
        w = A_fn(V[:, j])
        for i in range(j + 1): H[i, j] = float(np.dot(V[:, i], w)); w = w - H[i, j] * V[:, i]
        h_next = np.linalg.norm(w)
        if j + 1 < m and h_next > eps: H[j, j + 1] = h_next; V[:, j + 1] = w / h_next
        elif h_next < eps and j < m - 1: return V[:, :j + 1], H[:j + 1, :j + 1]
    return V, H


def ritz_eigenvalues(H: np.ndarray) -> np.ndarray:
    """Ritz eigenvalues from Hessenberg H (symmetrised)."""
    return linalg.eigh((H + H.T) / 2)[0]


def krylov_solve(A_fn, b: np.ndarray, m: int) -> np.ndarray:
    """Solve A·x = b via Krylov: project, solve small system, lift."""
    V, H = arnoldi_iteration(A_fn, b, m)
    if V.shape[1] == 0: return np.zeros_like(b)
    beta = np.linalg.norm(b); e1 = np.zeros(m); e1[0] = beta
    k = V.shape[1]; y = linalg.lstsq(H[:k, :k], e1[:k])[0]
    return V[:, :k] @ y
