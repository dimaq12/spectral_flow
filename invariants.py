"""
sft.invariants — Global spectral invariants for operator classification.

╔══════════════════════════════════════════════════════════════════════╗
║  INTENT                                                             ║
║  5 invariants that characterise an operator family without looking  ║
║  at individual eigenvalues: SVD kurtosis, Hessian sparsity,         ║
║  Poisson preimage test, W-coherence, zeta fingerprint.              ║
║  Used to distinguish ORDER, GRAPH, and degenerate regimes.          ║
╚══════════════════════════════════════════════════════════════════════╝
║  CONCEPT                                                            ║
║  svd_kurtosis(W)   — >3 sparse-rank, ≈1.8 ORDER regime            ║
║  hessian_sparsity  — 0=flat, 1=rich curvature structure            ║
║  poisson_preimage  — lower p-value ⇒ more separable parameter space║
║  w_coherence       — 0=orthogonal params, 1=fully coupled          ║
║  zeta_fingerprint  — 10-pt quantile curve of log₁₀(σ)             ║
║  all_invariants    — compute all 5 at once                         ║
╚══════════════════════════════════════════════════════════════════════╝
║  DEPENDENCIES                                                       ║
║  ┌── sft.core.OperatorFamily (W, hessian_analytic)                 ║
║  ├── numpy (quantile, mean, corr)                                   ║
║  └── scipy.linalg.svd, stats.kstest                                ║
╚══════════════════════════════════════════════════════════════════════╝
"""
import numpy as np
from scipy import linalg, stats
from .core import OperatorFamily


def svd_kurtosis(W: np.ndarray) -> float:
    """Kurtosis of SVD spectrum.  >3 → sparse-rank; ≈1.8 → ORDER."""
    s = linalg.svd(W, full_matrices=False, compute_uv=False)
    if len(s) <= 1: return 0.0
    c = s - s.mean(); m4 = np.mean(c**4); m2 = np.mean(c**2)
    return float(m4 / (m2**2)) if m2 > 1e-15 else 0.0


def hessian_sparsity(family: OperatorFamily) -> float:
    """Fraction of NON-zero structured Hessian entries.  Higher → more coupling."""
    from .hessian import hessian_analytic
    try: H = hessian_analytic(family)
    except Exception: return 0.0
    return 1.0 - float(np.mean(np.abs(H) < 1e-6 * (np.max(np.abs(H)) + 1e-15)))


def poisson_preimage(W: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic for Poisson-test on W·k. Lower ⇒ more separable."""
    rng = np.random.default_rng(42); k = rng.standard_normal(W.shape[1])
    x = W @ k; x_s = x / (np.std(x) + 1e-15); x_s = x_s - x_s.min() + 0.1
    return float(stats.kstest(x_s, 'expon')[0])


def w_coherence(W: np.ndarray) -> float:
    """Mean absolute column cross-correlation of W. 0=decoupled, 1=coupled."""
    if W.shape[1] <= 1: return 0.0
    Wn = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-15)
    corr = Wn.T @ Wn; np.fill_diagonal(corr, 0.0)
    return float(np.mean(np.abs(corr)))


def zeta_fingerprint(W: np.ndarray) -> np.ndarray:
    """10-point quantile curve of log₁₀(σ) — distribution ID."""
    s = linalg.svd(W, full_matrices=False, compute_uv=False)
    if len(s) <= 1: return np.zeros(10)
    return np.quantile(np.log10(np.maximum(s, 1e-15)), np.linspace(0, 1, 10))


def all_invariants(family: OperatorFamily) -> dict:
    """Compute all 5 invariants at once."""
    W = family.W
    return {"svd_kurtosis": svd_kurtosis(W),
            "hessian_sparsity": hessian_sparsity(family),
            "poisson_preimage": poisson_preimage(W),
            "w_coherence": w_coherence(W),
            "zeta_fingerprint": zeta_fingerprint(W)}
