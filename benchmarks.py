"""Reusable benchmark harness for computational breakthrough reports.

The goal is fairness: compare SFT against strong sparse baselines, report
setup/query split, eigensolve counts, error, memory estimates, and machine
metadata.  Large demos may extrapolate baseline totals from a measured sample,
but the report must say so explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import os
import platform
import time
import warnings
from typing import Callable, Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, lobpcg, ArpackError


Array = np.ndarray
MatrixBuilder = Callable[[Array], Any]


@dataclass
class TimerResult:
    name: str
    setup_ms: float
    query_ms: float
    total_ms: float
    eigensolves: int
    queries: int
    memory_mb: float = 0.0
    median_error: float = 0.0
    max_error: float = 0.0
    extrapolated: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def machine_metadata() -> dict[str, str | int]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count() or 1,
        "numpy": np.__version__,
    }


def sparse_memory_mb(A) -> float:
    if sparse.issparse(A):
        return float((A.data.nbytes + A.indices.nbytes + A.indptr.nbytes) / (1024 ** 2))
    arr = np.asarray(A)
    return float(arr.nbytes / (1024 ** 2))


def spectral_errors(reference: Array, candidate: Array) -> tuple[float, float]:
    ref = np.asarray(reference)
    cand = np.asarray(candidate)
    n = min(ref.shape[0], cand.shape[0])
    if n == 0:
        return 0.0, 0.0
    m = min(ref.shape[1], cand.shape[1]) if ref.ndim > 1 and cand.ndim > 1 else 1
    if ref.ndim > 1 and cand.ndim > 1:
        err = np.max(np.abs(ref[:n, :m] - cand[:n, :m]), axis=1)
    else:
        err = np.abs(ref[:n] - cand[:n])
    return float(np.median(err)), float(np.max(err))


def summarize_speedup(sft: TimerResult, baseline: TimerResult) -> dict[str, float]:
    return {
        "setup_speedup": baseline.setup_ms / max(sft.setup_ms, 1e-12),
        "query_speedup": baseline.query_ms / max(sft.query_ms, 1e-12),
        "total_speedup": baseline.total_ms / max(sft.total_ms, 1e-12),
        "eigensolve_reduction": baseline.eigensolves / max(sft.eigensolves, 1),
    }


def _smallest_eigs(A, k: int, which: str = "SM") -> Array:
    n = A.shape[0]
    if k >= n:
        return np.linalg.eigvalsh(A.toarray() if sparse.issparse(A) else A)[:k]
    vals = eigsh(A, k=k, which=which, return_eigenvectors=False)
    return np.sort(vals)


def run_repeated_eigsh(builder: MatrixBuilder, queries: Array, k_eigs: int,
                       max_measured: int | None = None,
                       which: str = "SM", name: str = "eigsh_loop") -> tuple[TimerResult, Array]:
    measured = queries if max_measured is None else queries[:max_measured]
    t0 = time.perf_counter()
    spectra = []
    mem = 0.0
    for q in measured:
        A = builder(q)
        mem = max(mem, sparse_memory_mb(A))
        spectra.append(_smallest_eigs(A, k_eigs, which=which))
    measured_ms = (time.perf_counter() - t0) * 1000
    scale = len(queries) / max(len(measured), 1)
    query_ms = measured_ms * scale
    result = TimerResult(
        name=name,
        setup_ms=0.0,
        query_ms=query_ms,
        total_ms=query_ms,
        eigensolves=len(queries),
        queries=len(queries),
        memory_mb=mem,
        extrapolated=max_measured is not None and len(measured) < len(queries),
        notes=f"measured {len(measured)} of {len(queries)} queries",
    )
    return result, np.asarray(spectra)


def run_lobpcg_warm_start(builder: MatrixBuilder, queries: Array, k_eigs: int,
                          max_measured: int | None = None,
                          tol: float = 1e-5, maxiter: int = 40,
                          seed: int = 0) -> tuple[TimerResult, Array]:
    measured = queries if max_measured is None else queries[:max_measured]
    rng = np.random.default_rng(seed)
    X = None
    spectra = []
    mem = 0.0
    fallback = 0
    t0 = time.perf_counter()
    for q in measured:
        A = builder(q)
        mem = max(mem, sparse_memory_mb(A))
        if X is None:
            X = rng.standard_normal((A.shape[0], k_eigs))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                vals, vecs = lobpcg(A, X, largest=False, tol=tol, maxiter=maxiter)
            idx = np.argsort(vals)[:k_eigs]
            spectra.append(vals[idx])
            X = vecs[:, idx]
        except Exception:
            fallback += 1
            vals, vecs = eigsh(A, k=k_eigs, which="SM")
            idx = np.argsort(vals)
            spectra.append(vals[idx])
            X = vecs[:, idx]
    measured_ms = (time.perf_counter() - t0) * 1000
    scale = len(queries) / max(len(measured), 1)
    query_ms = measured_ms * scale
    result = TimerResult(
        name="lobpcg_warm_start",
        setup_ms=0.0,
        query_ms=query_ms,
        total_ms=query_ms,
        eigensolves=len(queries),
        queries=len(queries),
        memory_mb=mem,
        extrapolated=max_measured is not None and len(measured) < len(queries),
        notes=f"measured {len(measured)} of {len(queries)} queries; eigsh fallbacks={fallback}",
    )
    return result, np.asarray(spectra)


def run_sft_predict_many(family, queries: Array, reference_spectra: Array | None = None,
                         name: str = "sft_predict_many") -> tuple[TimerResult, Array]:
    t0 = time.perf_counter()
    family.refresh(np.zeros(family.M))
    setup_ms = (time.perf_counter() - t0) * 1000
    eigensolves_before = getattr(family, "_eigh_count", 0)
    t0 = time.perf_counter()
    pred = family.predict_many(queries)
    query_ms = (time.perf_counter() - t0) * 1000
    eigensolves_after = getattr(family, "_eigh_count", eigensolves_before)
    med = mx = 0.0
    if reference_spectra is not None and len(reference_spectra) > 0:
        med, mx = spectral_errors(reference_spectra, pred[:len(reference_spectra)])
    result = TimerResult(
        name=name,
        setup_ms=setup_ms,
        query_ms=query_ms,
        total_ms=setup_ms + query_ms,
        eigensolves=max(eigensolves_after - eigensolves_before, 1),
        queries=len(queries),
        memory_mb=float(getattr(family._basis_backend, "materialized_elements", 0) * 8 / (1024 ** 2)),
        median_error=med,
        max_error=mx,
        notes=f"basis={family.basis_kind}, rank={family.W_rank}",
    )
    return result, pred


def markdown_result_table(results: list[TimerResult]) -> str:
    rows = []
    for r in results:
        rows.append(
            f"| {r.name} | {r.setup_ms:.1f} | {r.query_ms:.1f} | {r.total_ms:.1f} | "
            f"{r.eigensolves} | {r.memory_mb:.2f} | {r.median_error:.2e} | {r.max_error:.2e} | "
            f"{'yes' if r.extrapolated else 'no'} |"
        )
    return "\n".join(rows)
