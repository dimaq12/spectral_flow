"""Tests for sft.core — OperatorFamily, kernel, predict, inverse, rank, nullspace."""
import numpy as np
import pytest
from sft.core import OperatorFamily, kernel, predict, inverse, rank, nullspace, svd_spectrum


def make_simple_family(N=10, M=3):
    """Create a simple random operator family for testing."""
    rng = np.random.default_rng(0)
    A0 = rng.standard_normal((N, N))
    A0 = (A0 + A0.T) / 2
    basis = []
    for _ in range(M):
        B = rng.standard_normal((N, N))
        B = (B + B.T) / 2
        basis.append(B)
    return OperatorFamily(A0, basis)


class TestOperatorFamily:
    def test_construction(self):
        fam = make_simple_family(10, 3)
        assert fam.N == 10
        assert fam.M == 3
        assert fam.lam0.shape == (10,)
        assert fam.W.shape == (10, 3)
        assert fam.W_rank >= 1
        assert fam.complexity == fam.W_rank / 10
        assert fam.isospectral_dimension() == fam.M - fam.W_rank

    def test_build(self):
        fam = make_simple_family(5, 2)
        k = np.array([1.0, 0.5])
        A = fam.build(k)
        assert A.shape == (5, 5)
        A_expected = fam.A0 + k[0] * fam.basis[0] + k[1] * fam.basis[1]
        np.testing.assert_array_almost_equal(A, A_expected)

    def test_build_modifies_no_reference(self):
        fam = make_simple_family(5, 2)
        A0_before = fam.A0.copy()
        k = np.array([1.0, 0.5])
        _ = fam.build(k)
        np.testing.assert_array_equal(fam.A0, A0_before)

    def test_spectrum(self):
        fam = make_simple_family(5, 2)
        k = np.array([0.0, 0.0])
        lam = fam.spectrum(k)
        assert lam.shape == (5,)
        assert np.all(np.diff(lam) >= 0)  # sorted

    def test_predict_accuracy(self):
        fam = make_simple_family(10, 3)
        rng = np.random.default_rng(42)
        dk = rng.standard_normal(3) * 0.01
        lam_exact = fam.spectrum(dk)
        lam_pred = fam.predict(dk)
        err = np.max(np.abs(lam_exact - lam_pred))
        assert err < 0.1

    def test_predict_at(self):
        fam = make_simple_family(10, 3)
        k = np.array([0.01, -0.02, 0.03])
        pred1 = fam.predict(k)
        pred2 = fam.predict_at(k)
        np.testing.assert_array_almost_equal(pred1, pred2)

    def test_condition_number(self):
        fam = make_simple_family(20, 10)
        kappa = fam.condition_number()
        assert kappa >= 1.0

    def test_complexity_is_property(self):
        fam = make_simple_family(10, 3)
        c = fam.complexity
        assert isinstance(c, float)
        assert 0 <= c <= 1

    def test_W_pinv_reconstructs_identity(self):
        fam = make_simple_family(10, 3)
        W = fam.W
        Wp = fam.W_pinv
        # W W⁺ W ≈ W
        reconstructed = W @ Wp @ W
        np.testing.assert_array_almost_equal(reconstructed, W, decimal=6)

    def test_nullspace_dimension(self):
        fam = make_simple_family(5, 10)
        ker = nullspace(fam)
        assert ker.shape[1] == fam.isospectral_dimension()
        # W @ ker ≈ 0
        if ker.shape[1] > 0:
            result = fam.W @ ker
            assert np.all(np.abs(result) < 1e-8)

    def test_svd_spectrum(self):
        fam = make_simple_family(10, 3)
        s = svd_spectrum(fam)
        assert len(s) == min(fam.N, fam.M)
        assert np.all(np.diff(s[::-1]) >= 0)

    def test_inverse_convergence(self):
        fam = make_simple_family(10, 5)
        target = np.sort(fam.lam0 + np.linspace(-0.05, 0.05, fam.N))
        k, err, ok = fam.inverse(target, steps=50, alpha=0.3)
        assert ok or err < 0.2

    def test_configurable_tolerances(self):
        rng = np.random.default_rng(1)
        A0 = rng.standard_normal((5, 5))
        A0 = (A0 + A0.T) / 2
        basis = [np.eye(5)]
        fam = OperatorFamily(A0, basis, svd_tol=1e-6, convergence_tol=1e-3)
        assert fam.svd_tol == 1e-6
        assert fam.convergence_tol == 1e-3
        assert fam.W_rank >= 1

    def test_empty_basis(self):
        rng = np.random.default_rng(2)
        A0 = rng.standard_normal((5, 5))
        A0 = (A0 + A0.T) / 2
        fam = OperatorFamily(A0, [])
        assert fam.M == 0
        assert fam.W.shape == (5, 0)
        assert fam.W_rank == 0
        assert fam.isospectral_dimension() == 0
        assert fam.complexity == 0.0
        assert fam.condition_number() == np.inf

    def test_reference_update(self):
        fam = make_simple_family(10, 3)
        lam0_first = fam.lam0.copy()
        W_first = fam.W.copy()
        A_k = fam.build(np.array([0.5, -0.3, 0.1]))
        fam.set_reference(A_k)
        # Reference changed, so cached values should differ
        assert not np.allclose(fam.lam0, lam0_first)
        assert not np.allclose(fam.W, W_first)

    def test_singular_values(self):
        fam = make_simple_family(20, 10)
        s = fam.W_singular
        assert np.all(s >= 0)
        assert np.all(np.diff(s[::-1]) >= 0)


class TestModuleFunctions:
    def test_kernel(self):
        fam = make_simple_family(10, 3)
        W = kernel(fam)
        assert W.shape == (10, 3)

    def test_predict(self):
        fam = make_simple_family(10, 3)
        dk = np.array([0.01, 0.0, 0.0])
        p1 = predict(fam, dk)
        p2 = fam.predict(dk)
        np.testing.assert_array_equal(p1, p2)

    def test_inverse(self):
        fam = make_simple_family(10, 3)
        target = np.sort(fam.lam0)
        k, err, ok = inverse(fam, target)
        assert err < 0.1

    def test_rank(self):
        fam = make_simple_family(10, 3)
        r = rank(fam)
        assert r == fam.W_rank
        assert 1 <= r <= min(10, 3)
