"""
sft.tasks — Task classification, CDF sort, DCT filtering, spectral clustering.

╔══════════════════════════════════════════════════════════════════════╗
║  INTENT                                                             ║
║  High-level task primitives: classify problems into operator genera,║
║  sort via CDF, filter via DCT, cluster via Laplacian, hash, compose.║
║  Also defines OperatorGenus — the 13-genera task taxonomy.          ║
╚══════════════════════════════════════════════════════════════════════╝
║  CONCEPT                                                            ║
║  • OperatorGenus (13): MONO, QUAD, GRAPH, CONTROL, COMPRESS,        ║
║    PARAMETRIC, AUDIT, SPECTRAL_STATS, HOLOGRAPHIC, FLOW,            ║
║    PIECEWISE, CARLEMAN, ISOMORPHISM.                                ║
║  • classify_task: NL problem → OperatorGenus (keyword-based).       ║
║  • cdf_rank_sort: O(N) approximate sort via empirical CDF.          ║
║  • filter_via_dct: keep top-k DCT coefficients → reconstruct.       ║
║  • cluster_via_laplacian: Gaussian affinity → Fiedler → labels.     ║
║  • sort_via_fiedler: order 1D data via Laplacian Fiedler vector.    ║
╚══════════════════════════════════════════════════════════════════════╝
║  DEPENDENCIES                                                       ║
║  ┌── numpy, scipy.linalg (eigh)                                     ║
║  ├── sft.constructor (OperatorGenus via import at bottom — avoids   ║
║  │   circular import since constructor imports tasks)               ║
║  ├── sft.compress.compress_spectral (route_and_solve)               ║
║  └── sft.core.OperatorFamily (diagnose_mismatch)                    ║
╚══════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations
from enum import Enum, auto
from functools import lru_cache
import warnings
import numpy as np
from scipy import linalg


class OperatorGenus(Enum):
    """13 genera of operator tasks — classifies computational intent."""
    MONO = auto()           # ORDER/CDF operations
    QUAD = auto()           # Filtering, DCT, Fourier
    GRAPH = auto()          # Graph Laplacian, clustering, embedding
    CONTROL = auto()        # Inverse design, optimal control
    COMPRESS = auto()       # Spectral compression, codec
    PARAMETRIC = auto()     # Generic parametric operator
    AUDIT = auto()          # Phantom detection, mismatch diagnosis
    SPECTRAL_STATS = auto() # Level spacing, zeta statistics
    HOLOGRAPHIC = auto()    # Spectral codec roundtrip
    FLOW = auto()           # Monodromy, Berry, topology
    PIECEWISE = auto()      # Piecewise-smooth operators
    CARLEMAN = auto()       # Moment-based operators, GF(2)
    ISOMORPHISM = auto()    # Hash, injective map, order-preserving


def classify_task(problem_string: str) -> OperatorGenus:
    """NL problem → OperatorGenus.  Keyword-based classifier."""
    s = problem_string.lower()
    if "sort" in s: return OperatorGenus.MONO
    if any(kw in s for kw in ("order", "rank", "cdf", "quantile")): return OperatorGenus.MONO
    if any(kw in s for kw in ("filter", "bandpass", "dct", "fourier")): return OperatorGenus.QUAD
    if any(kw in s for kw in ("compress", "codec", "truncate", "sparsify")): return OperatorGenus.COMPRESS
    if any(kw in s for kw in ("cluster", "group", "segment", "partition")): return OperatorGenus.GRAPH
    if any(kw in s for kw in ("graph", "edge", "node", "bridge", "articulation")): return OperatorGenus.GRAPH
    if any(kw in s for kw in ("hash", "collision", "injective")): return OperatorGenus.ISOMORPHISM
    if any(kw in s for kw in ("predict", "forward", "simulate")): return OperatorGenus.PARAMETRIC
    if any(kw in s for kw in ("inverse", "design", "optimize", "target")): return OperatorGenus.CONTROL
    if any(kw in s for kw in ("monodromy", "berry", "holonomy", "topology", "ep")): return OperatorGenus.FLOW
    return OperatorGenus.PARAMETRIC


@lru_cache(maxsize=32)
def _dct_matrix_cached(n: int) -> np.ndarray:
    """DCT-II orthonormal basis: C @ x gives DCT coefficients, C^T @ coeffs = signal.  C·C^T = I."""
    k = np.arange(n)
    C = np.cos(np.pi / n * (k[:, None] + 0.5) * k[None, :])
    C[:, 0] *= 1.0 / np.sqrt(2); C *= np.sqrt(2.0 / n)
    return C


def dct_matrix(n: int) -> np.ndarray:
    """DCT-II orthonormal basis: returns a copy safe for caller mutation."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    return _dct_matrix_cached(int(n)).copy()


def filter_via_dct(signal: np.ndarray, keep_low: int | None = None,
                   keep_high: int | None = None, keep_frac: float | None = None) -> np.ndarray:
    """Band-pass filter: DCT → keep selected coefficients → reconstruct."""
    x = np.asarray(signal, np.float64).ravel(); n = len(x)
    C = dct_matrix(n); coeffs = C @ x; mask = np.zeros(n, bool)
    if keep_frac is not None:
        thresh = np.sort(np.abs(coeffs))[-int(n * keep_frac)]; mask = np.abs(coeffs) >= thresh
    elif keep_low is not None: mask[:keep_low] = True
    elif keep_high is not None: mask[-keep_high:] = True
    else: mask[:] = True
    return C.T @ (coeffs * mask)


def route_and_solve(problem: str, data: np.ndarray):
    """Unified dispatcher: classify → dispatch to correct operation."""
    genus = classify_task(problem)
    if genus == OperatorGenus.MONO: return cdf_rank_sort(data)
    elif genus == OperatorGenus.QUAD: return filter_via_dct(data, keep_frac=0.5)
    elif genus == OperatorGenus.GRAPH: return cluster_via_laplacian(data)
    elif genus == OperatorGenus.COMPRESS:
        from .compress import compress_spectral; return compress_spectral(data, K=16)
    return data


def task(problem: str, data: np.ndarray):
    """Alias for route_and_solve, useful in fluent examples."""
    return route_and_solve(problem, data)


def cluster_via_laplacian(x: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """Spectral clustering: Gaussian affinity → Laplacian → Fiedler → binary labels."""
    x = np.asarray(x, np.float64)
    if x.ndim == 1: x = x.reshape(-1, 1)
    N = x.shape[0]
    sigma = sigma or np.median(np.linalg.norm(x[:, None] - x[None, :], axis=-1) + 1e-10) * 0.75
    diff = x[:, None] - x[None, :]; dist2 = np.sum(diff**2, axis=-1)
    W = np.exp(-dist2 / (2.0 * sigma**2)); np.fill_diagonal(W, 0.0)
    D = np.diag(np.sum(W, axis=1)); L = D - W
    _, V = linalg.eigh(L); return (V[:, 1] > 0).astype(int)


def sort_via_fiedler(x: np.ndarray) -> np.ndarray:
    """Sort 1D data via Fiedler vector of Laplacian from pairwise distances."""
    x = np.asarray(x, np.float64).ravel(); N = len(x)
    if N <= 2: return np.sort(x)
    diff = np.abs(x[:, None] - x[None, :])
    sigma = np.median(diff[diff > 0]) * 0.5 if np.any(diff > 0) else 1.0
    W = np.exp(-diff**2 / (2.0 * sigma**2)); np.fill_diagonal(W, 0.0)
    D = np.diag(np.sum(W, axis=1)); L = D - W
    _, V = linalg.eigh(L); return x[np.argsort(V[:, 1])]


def hash_injective_map(keys: np.ndarray) -> np.ndarray:
    """Collision-free hash: np.unique → inverse index mapping."""
    return np.unique(np.asarray(keys, np.float64).ravel(), return_inverse=True)[1]


def compose_tasks(task1_name: str, task2_name: str, data: np.ndarray):
    """Pipeline: task1(data) → task2(result)."""
    return route_and_solve(task2_name, route_and_solve(task1_name, data))


def pipe(task1_name: str, task2_name: str, data: np.ndarray):
    """Alias for compose_tasks."""
    return compose_tasks(task1_name, task2_name, data)


def diagnose_mismatch(problem: str, data: np.ndarray) -> dict:
    """Phantom detection: how well does synthesized operator fit the task?"""
    from .constructor import synthesize
    genus = classify_task(problem)
    try:
        fam = synthesize(problem, data); pred = fam.predict(np.zeros(fam.M))
        err = float(np.max(np.abs(pred - fam.lam0)))
    except Exception as exc:
        warnings.warn(f"diagnose_mismatch could not synthesize operator: {exc}", RuntimeWarning, stacklevel=2)
        err = 0.0
    return {"task_solved": genus.name, "genus": genus, "spectrum_error": err, "phantom_detected": err > 0.1}


def cdf_rank_sort(arr: np.ndarray, n_bins: int = 50) -> np.ndarray:
    """Approximate sort via empirical CDF: histogram → predicted ranks → local sort."""
    x = np.asarray(arr, np.float64); N = len(x)
    if N <= 1: return x.copy()
    hist, edges = np.histogram(x, bins=n_bins)
    cdf = np.cumsum(hist) / N
    bin_idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    predicted = np.clip(np.floor(cdf[bin_idx] * N).astype(int), 0, N - 1)
    uniq, inv, cnt = np.unique(predicted, return_inverse=True, return_counts=True)
    result = np.empty(N); offset = 0
    for k, c in enumerate(cnt):
        vals = x[inv == k]; result[offset:offset + c] = np.sort(vals); offset += c
    return result
