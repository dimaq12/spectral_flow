"""
sft.compress — Spectral compression via DCT, Hankel-SVD.

╔══════════════════════════════════════════════════════════════════════╗
║  INTENT                                                             ║
║  compress_spectral:  Hankel → SVD → keep top K → reconstruct.       ║
║  dct_codec:          DCT → keep top fraction → reconstruct.         ║
╚══════════════════════════════════════════════════════════════════════╝
║  DEPENDENCIES: numpy, scipy.linalg.svd, sft.tasks.dct_matrix       ║
╚══════════════════════════════════════════════════════════════════════╝
"""
import numpy as np
from scipy import linalg


def compress_spectral(signal: np.ndarray, K: int | None = None,
                      energy_frac: float | None = None) -> np.ndarray:
    """Hankel-SVD compression: reconstruct from top-K singular components."""
    x = np.asarray(signal, np.float64).ravel(); n = len(x)
    L = min(n // 3, 50); K2 = n - L + 1
    H = np.array([[x[i + j] for j in range(L)] for i in range(K2)])
    U, s, Vt = linalg.svd(H, full_matrices=False)
    if energy_frac is not None:
        cumsum = np.cumsum(s**2); total = cumsum[-1]
        keep = int(np.searchsorted(cumsum / total, energy_frac)) + 1; keep = np.clip(keep, 1, len(s))
    else: keep = np.clip(K or len(s), 1, len(s))
    s[keep:] = 0.0; H_comp = (U * s[None, :]) @ Vt
    out = np.zeros(n); counts = np.zeros(n)
    for i in range(K2): out[i:i + L] += H_comp[i]; counts[i:i + L] += 1
    counts[counts == 0] = 1.0
    return out / counts


def dct_codec(signal: np.ndarray, keep_frac: float = 0.5) -> np.ndarray:
    """Instant DCT codec: keep top keep_frac coefficients → reconstruct."""
    from .tasks import dct_matrix
    x = np.asarray(signal, np.float64).ravel(); n = len(x)
    C = dct_matrix(n); coeffs = C @ x
    k = max(1, int(n * keep_frac)); top = np.argsort(np.abs(coeffs))[-k:]
    coeffs_s = np.zeros(n); coeffs_s[top] = coeffs[top]
    return C.T @ coeffs_s
