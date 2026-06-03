"""Breakthrough benchmark harness smoke tests."""
import numpy as np
from scipy import sparse

import sft
from sft.benchmarks import TimerResult, spectral_errors, summarize_speedup


def test_timer_result_and_speedup_math():
    sft_result = TimerResult("sft", 1.0, 2.0, 3.0, 1, 100)
    base = TimerResult("base", 10.0, 200.0, 210.0, 100, 100)
    speed = summarize_speedup(sft_result, base)
    assert speed["query_speedup"] == 100.0
    assert speed["total_speedup"] == 70.0
    assert speed["eigensolve_reduction"] == 100.0


def test_spectral_errors_slices_common_modes():
    ref = np.zeros((3, 2))
    cand = np.ones((3, 5)) * 0.1
    med, mx = spectral_errors(ref, cand)
    assert med == mx == 0.1


def test_small_sft_vs_eigsh_harness_agree():
    model = sft.physics.schrodinger_1d(
        np.linspace(0.0, 1.0, 34)[1:-1],
        np.zeros(32),
        max_potential_params=4,
    )
    fam = model.family()
    queries = np.zeros((3, fam.M))

    eigsh_result, eigsh_ref = sft.benchmarks.run_repeated_eigsh(
        fam.build_sparse, queries, k_eigs=4, max_measured=None
    )
    sft_result, _ = sft.benchmarks.run_sft_predict_many(fam, queries, eigsh_ref)

    assert eigsh_result.eigensolves == 3
    assert sft_result.eigensolves >= 1
    assert sft_result.max_error < 1e-8


def test_sparse_memory_estimate_positive():
    A = sparse.eye(10, format="csc")
    assert sft.benchmarks.sparse_memory_mb(A) > 0.0
