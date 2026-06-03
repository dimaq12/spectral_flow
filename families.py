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
from scipy import sparse
from .core import OperatorFamily, edge_laplacian_basis, repeated_identity_basis

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
    is_sp = sparse.issparse(adjacency)
    adjacency = adjacency.tocsr().astype(np.float64) if is_sp else np.asarray(adjacency, dtype=np.float64)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"adjacency must be square, got {adjacency.shape}")
    N = adjacency.shape[0]
    if is_sp:
        row, col = sparse.triu(adjacency, 1).nonzero()
    else:
        row, col = np.triu(adjacency, 1).nonzero()
    edges = list(zip(row.tolist(), col.tolist()))
    degree = np.asarray(adjacency.sum(axis=1)).ravel() if is_sp else np.sum(adjacency, axis=1)
    if is_sp or N > 2048:
        A_sp = adjacency if is_sp else sparse.csr_matrix(adjacency)
        L0 = sparse.diags(degree) - A_sp
    else:
        D = np.diag(degree)
        L0 = D - adjacency
    return OperatorFamily(L0, edge_laplacian_basis(N, edges))


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
    return OperatorFamily(np.eye(N), repeated_identity_basis(N))


def avoided_crossing_2x2(Delta: float = 0.3) -> OperatorFamily:
    """Classic 2×2: H = [[k₀, Δ], [Δ, −k₀]]."""
    return OperatorFamily(np.array([[0, Delta], [Delta, 0]]),
                          [np.array([[1, 0], [0, -1]]), np.array([[0, 1], [1, 0]])])
