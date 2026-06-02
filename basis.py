"""
sft.basis — Signal operators: Toeplitz autocorrelation, Gaussian affinity, covariance.
"""
import numpy as np
from scipy import linalg


def toeplitz_from_signal(signal: np.ndarray, n: int | None = None) -> np.ndarray:
    """Autocorrelation Toeplitz (Yule-Walker).  A_ij = E[x[k]·x[k+|i-j|]]."""
    x = np.asarray(signal, np.float64).ravel(); N = len(x)
    n = n or max(1, min(N // 4, 64)); n = min(n, N - 1)
    r = np.array([np.mean(x[:N - lag] * x[lag:]) if N > lag else 0.0 for lag in range(n)])
    return linalg.toeplitz(r)


def gaussian_affinity(x: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """Gaussian Gram matrix: W_ij = exp(-||x_i-x_j||^2/(2*sigma^2))."""
    x = np.asarray(x, np.float64)
    if x.ndim == 1: x = x.reshape(-1, 1)
    N = x.shape[0]; diff = x[:, None] - x[None, :]; dist2 = np.sum(diff**2, axis=-1)
    if sigma is None:
        nd = dist2[np.triu_indices(N, 1)]; sigma = np.median(np.sqrt(nd + 1e-15)) * 0.75 or 1.0
    W = np.exp(-dist2 / (2.0 * sigma**2)); np.fill_diagonal(W, 0.0)
    return W


def covariance_operator(data: np.ndarray, n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Build covariance matrix from (samples x features) or (signal,) data.

    Returns (Cov, Corr) — covariance and correlation matrices.
    For 1D signal, uses lag-embedding (Toeplitz covariance).

    Parameters
    ----------
    data : (T,) signal or (T, N) samples x features
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
