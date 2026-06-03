"""Numerical tail cases for inverse helpers."""
import numpy as np

import sft
from sft.core import OperatorFamily


def test_bottleneck_dim_larger_than_rank_is_safe():
    fam = sft.families.random(10, 3, seed=7)
    target = np.sort(fam.lam0 + np.linspace(-0.01, 0.01, fam.N))
    result = sft.inversion.bottleneck_inverse(
        fam, target, bottleneck_dim=100, steps=2, return_result=True
    )
    assert result.k.shape == (fam.M,)
    assert np.isfinite(result.error)


def test_fixed_point_empty_linear_indices_no_crash():
    fam = sft.families.random(10, 3, seed=8)
    target = np.sort(fam.lam0)
    result = sft.inversion.fixed_point_inverse(
        fam, target, linear_basis_indices=[], outer_steps=2, inner_steps=1,
        return_result=True,
    )
    np.testing.assert_allclose(result.k, np.zeros(fam.M))


def test_inverse_with_rank_zero_family_reports_inf_condition():
    fam = OperatorFamily(np.eye(5), [])
    result = fam.inverse(fam.lam0, steps=2)
    assert result.converged
    assert result.condition_number == np.inf


def test_monodromy_inverse_constant_function_has_zero_drift():
    def target_func(_k):
        return np.array([1.0, 1.0])

    lams, drift = sft.inversion.monodromy_inverse(target_func, n_pts_circle=8)
    assert lams.shape == (8,)
    assert drift == 0.0
