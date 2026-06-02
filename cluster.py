"""
sft.cluster — k-way spectral clustering, kNN clustering, auto-basis selection.

╔══════════════════════════════════════════════════════════════════════╗
║  INTENT                                                             ║
║  cluster_spectral_k:  Gaussian affinity → Laplacian → k-means.      ║
║  cluster_knn_spectral: kNN graph → spectral embedding → k-means.    ║
║  choose_basis_auto:   auto-select 'dct'/'fourier'/'identity'.       ║
╚══════════════════════════════════════════════════════════════════════╝
"""
import numpy as np
from scipy import linalg


def cluster_spectral_k(x: np.ndarray, k: int, sigma: float | None = None) -> np.ndarray:
    """k-way spectral clustering: Gaussian affinity → Laplacian → k-means in embedding."""
    x = np.asarray(x, np.float64)
    if x.ndim == 1: x = x.reshape(-1, 1)
    N = x.shape[0]; sigma = sigma or np.median(np.linalg.norm(x[:, None] - x[None, :], axis=-1) + 1e-10) * 0.75
    diff = x[:, None] - x[None, :]; dist2 = np.sum(diff**2, axis=-1)
    W = np.exp(-dist2 / (2.0 * sigma**2)); np.fill_diagonal(W, 0.0)
    D_is = np.diag(1.0 / np.sqrt(np.sum(W, axis=1) + 1e-15)); Ls = np.eye(N) - D_is @ W @ D_is
    _, V = linalg.eigh(Ls); emb = V[:, :k] / (np.linalg.norm(V[:, :k], axis=1, keepdims=True) + 1e-15)
    from scipy.cluster.vq import kmeans2; _, labels = kmeans2(emb, k, minit='points', missing='warn')
    return labels.astype(int)


def cluster_knn_spectral(X: np.ndarray, k_clusters: int, k_neighbors: int = 10,
                         sigma: float | None = None) -> np.ndarray:
    """kNN graph → spectral clustering."""
    X = np.asarray(X, np.float64); N = X.shape[0]
    dist2 = np.sum((X[:, None] - X[None, :])**2, axis=-1); np.fill_diagonal(dist2, np.inf)
    nn = np.argsort(dist2, axis=1)[:, :min(k_neighbors, N - 1)]
    adj = np.zeros((N, N))
    for i in range(N): adj[i, nn[i]] = 1.0
    adj = np.maximum(adj, adj.T)
    return cluster_spectral_k(X, k_clusters, sigma)


def choose_basis_auto(signal: np.ndarray) -> str:
    """Auto-select basis: 'dct' if mid-band energy >30%, 'fourier' if HF >10%, else 'identity'."""
    x = np.asarray(signal, np.float64).ravel(); n = len(x)
    if n < 8: return "identity"
    fft = np.abs(np.fft.rfft(x)); total = np.sum(fft) + 1e-15; nf = len(fft)
    mid_frac = np.sum(fft[nf // 4:3 * nf // 4]) / total
    high_frac = np.sum(fft[3 * nf // 4:]) / total
    if mid_frac > 0.30: return "dct"
    if high_frac > 0.10: return "fourier"
    return "identity"
