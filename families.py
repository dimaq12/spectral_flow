"""
sft.families — Pre-built operator families: random, graph, Toeplitz, etc.

╔══════════════════════════════════════════════════════════════════════╗
║  INTENT                                                             ║
║  Convenience constructors for common operator families.             ║
║  Each function returns an OperatorFamily ready for analysis.        ║
╚══════════════════════════════════════════════════════════════════════╝
║  DEPENDENCIES                                                       ║
║  ┌── sft.core.OperatorFamily                                       ║
║  ├── numpy (rng, zeros, eye, diag)                                  ║
║  └── scipy.linalg (circulant — cycle graph reference)               ║
╚══════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations
import numpy as np
from scipy import linalg
from .core import OperatorFamily

__all__ = ["random", "graph_laplacian", "toeplitz", "diagonal", "avoided_crossing_2x2"]


def random(N: int, M: int, seed: int = 0, sparsity: float = 1.0) -> OperatorFamily:
    """Random symmetric family.  Basis: M random symmetric matrices.  A0: average of first 4."""
    if N <= 0: raise ValueError(f"N must be positive, got {N}")
    if M <= 0: raise ValueError(f"M must be positive, got {M}")
    rng = np.random.default_rng(seed)
    basis_t = rng.standard_normal((M, N, N)) * sparsity
    basis_t = (basis_t + np.swapaxes(basis_t, 1, 2)) / 2
    A0 = basis_t[:min(M, 4)].sum(axis=0) / min(M, 4)
    return OperatorFamily(A0, list(basis_t))


def graph_laplacian(adjacency: np.ndarray) -> OperatorFamily:
    """Graph Laplacian family.  Parameters: edge weights.  A0 = D − A (unweighted Laplacian)."""
    N = adjacency.shape[0]
    row, col = np.triu(adjacency, 1).nonzero()
    edges = list(zip(row.tolist(), col.tolist()))
    M = len(edges)
    basis_arr = np.zeros((M, N, N))
    for k, (u, v) in enumerate(edges):
        basis_arr[k, u, u] = basis_arr[k, v, v] = 1.0
        basis_arr[k, u, v] = basis_arr[k, v, u] = -1.0
    D = np.diag(np.sum(adjacency, axis=1))
    return OperatorFamily(D - adjacency, list(basis_arr) if M > 0 else [])


def toeplitz(N: int, diagonals: int = 5, seed: int = 0) -> OperatorFamily:
    """Toeplitz/filter family.  One basis per off-diagonal."""
    if N <= 1: raise ValueError(f"N must be >= 2")
    M = min(diagonals, N - 1)
    basis_arr = np.zeros((M, N, N))
    diag_range = np.arange(N)
    for d in range(M):
        idx = diag_range[:N - d - 1]
        basis_arr[d, idx, idx + d + 1] = basis_arr[d, idx + d + 1, idx] = 1.0
    A0 = np.zeros((N, N))  # zero operator — signal from k·B only
    return OperatorFamily(A0, list(basis_arr))


def diagonal(N: int) -> OperatorFamily:
    """Diagonal family: B_j = I → rank(W) = 1 (ORDER regime)."""
    return OperatorFamily(np.eye(N), [np.eye(N) for _ in range(N)])


def avoided_crossing_2x2(Delta: float = 0.3) -> OperatorFamily:
    """Classic 2×2: H = [[k₀, Δ], [Δ, −k₀]]."""
    return OperatorFamily(np.array([[0, Delta], [Delta, 0]]),
                          [np.array([[1, 0], [0, -1]]), np.array([[0, 1], [1, 0]])])
