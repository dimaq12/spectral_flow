"""
sft.graph_gen — Graph generators and Laplacian builder.
"""
import numpy as np


def build_laplacian(n: int, edges: list, weights: list | None = None) -> np.ndarray:
    """L = D - W from edge list + optional weights."""
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
