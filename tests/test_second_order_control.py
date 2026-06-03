"""Second-order inverse/control contracts."""
import numpy as np
import pytest

import sft


def test_inverse_method_selector_rejects_unknown_method():
    fam = sft.families.random(8, 3, seed=12)
    with pytest.raises(ValueError, match="method must be"):
        fam.inverse(fam.lam0, method="magic")


def test_hessian_inverse_reports_curvature_diagnostics():
    fam = sft.families.random(12, 4, seed=13)
    target = np.sort(fam.lam0 + np.linspace(-0.04, 0.04, fam.N))
    linear = fam.inverse(target, steps=3, alpha=0.25, method="linear")
    hess = fam.inverse(target, steps=3, alpha=0.25, method="hessian")
    assert hess.method == "hessian"
    assert hess.hessian_count > 0
    assert hess.trajectory.shape[1] == fam.M
    assert hess.residual_history.shape[0] >= 1
    assert hess.error <= linear.error * 1.5


def test_fluent_flow_builder_uses_second_order_method():
    fam = sft.families.random(10, 3, seed=14)
    target = np.sort(fam.lam0 + np.linspace(-0.02, 0.02, fam.N))
    result = fam.flow(target).via("linear").second_order().options(steps=2).solve(alpha=0.2)
    assert result.method == "hessian"
    assert result.hessian_count > 0


def test_homotopy_method_routes_to_homotopy_solver():
    fam = sft.families.random(8, 3, seed=15)
    target = np.sort(fam.lam0)
    result = fam.inverse(target, method="homotopy", steps=3)
    assert result.method == "homotopy"
    assert result.k.shape == (fam.M,)
