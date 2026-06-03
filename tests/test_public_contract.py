"""Public compatibility and quick-start contract tests."""
import numpy as np

import sft


def test_top_level_contract_exports_remain_available():
    for name in sft.__all__:
        assert hasattr(sft, name), name
    assert sft.OperatorFamily is not None
    assert sft.audio is not None
    assert sft.graph is not None
    assert sft.from_task is not None


def test_readme_quick_start_core_flow():
    fam = sft.families.random(N=20, M=6, seed=42)
    dk = np.ones(fam.M) * 0.01
    lam_pred = fam.predict(dk)
    lam_exact = fam.spectrum(dk)
    assert lam_pred.shape == lam_exact.shape
    target = np.sort(fam.lam0 + np.linspace(-0.03, 0.03, fam.N))
    result = fam.inverse(target, steps=10, alpha=0.2)
    k, err, ok = result
    assert k.shape == (fam.M,)
    assert err >= 0.0
    assert isinstance(ok, bool)
    assert result.steps > 0
    assert result.n_refresh > 0


def test_predict_at_uses_current_reference_k0_after_refresh():
    fam = sft.families.random(N=12, M=4, seed=7)
    k0 = np.array([0.03, -0.01, 0.02, 0.04])
    k1 = k0 + np.array([0.001, -0.002, 0.001, 0.0])
    fam.set_reference(fam.build(k0), k0=k0)
    np.testing.assert_allclose(fam.predict_at(k0), fam.lam0)
    err = np.max(np.abs(fam.predict_at(k1) - fam.spectrum(k1)))
    assert err < 0.1


def test_graph_family_uses_structured_backend_without_changing_kernel():
    adj = sft.graph_gen.path_graph(8)
    fam = sft.families.graph_laplacian(adj)
    lam, _, W_expected = sft.graph_response_kernel(adj)
    assert fam.basis_kind == "edge_laplacian"
    np.testing.assert_allclose(fam.lam0, lam)
    np.testing.assert_allclose(fam.W, W_expected)
    H = sft.hessian.hessian_analytic(fam)
    assert H.shape == (fam.N, fam.M, fam.M)


def test_simple_facade_helpers_and_molecular_aliases():
    x = np.array([3.0, 1.0, 2.0])
    np.testing.assert_array_equal(sft.sort(x), np.array([1.0, 2.0, 3.0]))
    assert sft.filter(np.sin(np.linspace(0, 1, 16)), keep_low=3).shape == (16,)
    labels = sft.cluster_data(np.array([[0.0], [0.1], [2.0], [2.1]]))
    assert labels.shape == (4,)

    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    mol = sft.molecular(positions, atom_types=["H", "H"], bonds=[(0, 1)])
    assert mol.kernel.shape == (2, 1)


def test_shape_guards_raise_clear_value_errors():
    fam = sft.families.random(N=8, M=3, seed=1)
    for call in (fam.predict, fam.predict_at, fam.spectrum):
        try:
            call(np.zeros(2))
        except ValueError as exc:
            assert "shape (3,)" in str(exc)
        else:
            raise AssertionError("expected ValueError")
    try:
        fam.inverse(np.zeros(7))
    except ValueError as exc:
        assert "at least 8" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_inverse_helpers_can_return_diagnostics_without_breaking_default():
    fam = sft.families.random(N=8, M=3, seed=2)
    target = np.sort(fam.lam0 + np.linspace(-0.01, 0.01, fam.N))
    k_default = sft.inversion.bottleneck_inverse(fam, target, bottleneck_dim=2, steps=2)
    assert k_default.shape == (fam.M,)
    result = sft.inversion.bottleneck_inverse(
        fam, target, bottleneck_dim=2, steps=2, return_result=True
    )
    k, err, ok = result
    assert k.shape == (fam.M,)
    assert err >= 0.0
    assert isinstance(ok, bool)
    assert result.n_refresh >= 1
