"""Degenerate-spectrum contracts.

SFT does not promise a unique eigenvector basis inside exactly degenerate
subspaces, but eigenvalue shapes, finite kernels, and solver diagnostics should
remain stable and explicit.
"""
import numpy as np

import sft
from sft.core import OperatorFamily


def test_repeated_eigenvalues_produce_finite_kernel():
    fam = OperatorFamily(
        np.eye(4),
        [
            np.diag([1.0, -1.0, 0.0, 0.0]),
            np.diag([0.0, 0.0, 1.0, -1.0]),
        ],
    )
    assert fam.lam0.shape == (4,)
    assert fam.W.shape == (4, 2)
    assert np.all(np.isfinite(fam.W))


def test_rank_zero_kernel_has_empty_isospectral_directions_for_no_params():
    fam = OperatorFamily(np.eye(3), [])
    assert fam.W_rank == 0
    assert fam.isospectral_dimension() == 0
    assert fam.condition_number() == np.inf


def test_rank_one_kernel_reports_expected_isospectral_dimension():
    fam = sft.families.diagonal(5)
    assert fam.W_rank == 1
    assert fam.isospectral_dimension() == fam.M - 1
    assert np.isfinite(fam.W).all()


def test_near_degenerate_avoided_crossing_is_finite():
    fam = sft.families.avoided_crossing_2x2(Delta=1e-9)
    k = np.array([1e-9, 0.0])
    lam = fam.spectrum(k)
    pred = fam.predict(k)
    H = sft.hessian.hessian_analytic(fam)
    assert lam.shape == (2,)
    assert pred.shape == (2,)
    assert H.shape == (2, 2, 2)
    assert np.all(np.isfinite(lam))
    assert np.all(np.isfinite(pred))
    assert np.all(np.isfinite(H))
