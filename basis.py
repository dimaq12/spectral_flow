"""
sft.basis — Auto-correlation Toeplitz, Gaussian affinity, graph builders.

╔══════════════════════════════════════════════════════════════════════╗
║  INTENT                                                             ║
║  toeplitz_from_signal:  autocorrelation Toeplitz (Yule-Walker).     ║
║  gaussian_affinity:     Gram matrix W = exp(−||x−y||²/2σ²).        ║
║  build_laplacian:       Laplacian from edge list + weights.         ║
║  Graph generators:       path, grid_2d, random, small_world, star.  ║
╚══════════════════════════════════════════════════════════════════════╝
"""
import numpy as np
from scipy import linalg


def toeplitz_from_signal(signal: np.ndarray, n: int | None = None) -> np.ndarray:
    """Autocorrelation Toeplitz (Yule-Walker).  A_ij = E[x[k]·x[k+|i−j|]]."""
    x = np.asarray(signal, np.float64).ravel(); N = len(x)
    n = n or max(1, min(N // 4, 64)); n = min(n, N - 1)
    r = np.array([np.mean(x[:N - lag] * x[lag:]) if N > lag else 0.0 for lag in range(n)])
    return linalg.toeplitz(r)


def gaussian_affinity(x: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """Gaussian Gram matrix: W_ij = exp(−||x_i−x_j||²/(2σ²))."""
    x = np.asarray(x, np.float64)
    if x.ndim == 1: x = x.reshape(-1, 1)
    N = x.shape[0]; diff = x[:, None] - x[None, :]; dist2 = np.sum(diff**2, axis=-1)
    if sigma is None:
        nd = dist2[np.triu_indices(N, 1)]; sigma = np.median(np.sqrt(nd + 1e-15)) * 0.75 or 1.0
    W = np.exp(-dist2 / (2.0 * sigma**2)); np.fill_diagonal(W, 0.0)
    return W


def build_laplacian(n: int, edges: list, weights: list | None = None) -> np.ndarray:
    """L = D − W from edge list + optional weights."""
    w = weights or [1.0] * len(edges); L = np.zeros((n, n))
    for (u, v), wt in zip(edges, w): L[u, v] = L[v, u] = -wt; L[u, u] += wt; L[v, v] += wt
    return L


def path_graph(n: int) -> np.ndarray:
    adj = np.zeros((n, n)); adj[np.arange(n - 1), np.arange(1, n)] = 1.0; return adj + adj.T

def grid_graph_2d(w: int, h: int) -> np.ndarray:
    N = w * h; adj = np.zeros((N, N))
    idx = lambda i, j: i * w + j
    for i in range(h):
        for j in range(w):
            u = idx(i, j)
            if j + 1 < w: adj[u, idx(i, j + 1)] = adj[idx(i, j + 1), u] = 1.0
            if i + 1 < h: adj[u, idx(i + 1, j)] = adj[idx(i + 1, j), u] = 1.0
    return adj

def random_graph(n: int, p: float = 0.3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.triu(rng.random((n, n)) < p, 1).astype(np.float64); return t + t.T

def small_world_graph(n: int, k: int = 4, p: float = 0.1, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed); adj = np.zeros((n, n))
    for i in range(n):
        for di in range(1, k // 2 + 1):
            j = (i + di) % n
            if rng.random() < p:
                j = rng.integers(0, n)
                while j == i or adj[i, j]: j = rng.integers(0, n)
            adj[i, j] = adj[j, i] = 1.0
    return adj

def star_graph(n: int) -> np.ndarray:
    adj = np.zeros((n, n)); adj[0, 1:] = 1.0; adj[1:, 0] = 1.0; return adj


def covariance_operator(data: np.ndarray, n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Build covariance matrix from (samples × features) or (signal,) data.

    Returns (Cov, Corr) — covariance and correlation matrices.
    For 1D signal, uses lag-embedding (Toeplitz covariance).

    Parameters
    ----------
    data : (T,) signal or (T, N) samples × features
    n : int or None — embedding dimension for 1D signal (default: min(T//4, 64))

    Returns (Cov, Corr).
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        N = len(data)
        n = n or max(1, min(N // 4, 64))
        n = min(n, N - 1)
        r = np.array([np.mean(data[:N - lag] * data[lag:]) if N > lag else 0.0
                       for lag in range(n)])
        Cov = linalg.toeplitz(r)
    else:
        T = data.shape[0]
        data_centered = data - data.mean(axis=0, keepdims=True)
        Cov = (data_centered.T @ data_centered) / max(T - 1, 1)
        Cov = (Cov + Cov.T) / 2

    std = np.sqrt(np.diag(Cov) + 1e-15)
    Corr = Cov / (std[:, None] * std[None, :])
    return Cov, Corr
