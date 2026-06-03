"""Small industrial-baseline agreement tests."""
import numpy as np

import sft


def test_lobpcg_warm_start_returns_low_modes():
    fam = sft.physics.laplacian_grid(24, max_potential_params=4).family()
    queries = np.zeros((2, fam.M))
    result, spectra = sft.benchmarks.run_lobpcg_warm_start(
        fam.build_sparse, queries, k_eigs=3, max_measured=None, maxiter=20
    )
    assert result.queries == 2
    assert spectra.shape == (2, 3)
    assert np.all(np.isfinite(spectra))


def test_breakthrough_graph_baseline_residual_is_finite():
    adj = sft.graph_gen.path_graph(20)
    fam = sft.families.graph_laplacian(adj)
    target = fam.spectrum(np.zeros(fam.M), n_eigs=4)
    result = fam.inverse(target, n_eigs=4, steps=1, method="trust")
    assert result.error < 1e-8
    assert result.k.shape == (fam.M,)
