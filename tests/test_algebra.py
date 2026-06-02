"""Tests for sft.algebra — direct_sum, compose_linear, tensor_sum, expectation."""
import numpy as np
import pytest
from sft.core import OperatorFamily
from sft.algebra import direct_sum, compose_linear, tensor_sum, expectation


def make_family(N, M, seed=0):
    rng = np.random.default_rng(seed)
    A0 = rng.standard_normal((N, N))
    A0 = (A0 + A0.T) / 2
    basis = []
    for _ in range(M):
        B = rng.standard_normal((N, N))
        B = (B + B.T) / 2
        basis.append(B)
    return OperatorFamily(A0, basis)


class TestDirectSum:
    def test_shapes(self):
        a = make_family(5, 3)
        b = make_family(7, 4)
        ds = direct_sum(a, b)
        assert ds.N == 12
        assert ds.M == 7

    def test_eigenvalues_are_union(self):
        a = make_family(5, 2)
        b = make_family(3, 2)
        ds = direct_sum(a, b)
        lam_union = np.sort(np.concatenate([a.lam0, b.lam0]))
        np.testing.assert_array_almost_equal(np.sort(ds.lam0), lam_union)

    def test_W_is_block_diagonal(self):
        a = make_family(5, 3)
        b = make_family(3, 2)
        ds = direct_sum(a, b)
        # Cross-block terms in W should be zero: params from block A
        # cannot affect eigenvalues from block B, and vice versa.
        # Identify which rows come from which block by checking
        # if the W row has non-zero entries in the first 3 or last 2 columns.
        for i in range(ds.N):
            row_a = ds.W[i, :3]
            row_b = ds.W[i, 3:]
            # Each row should be active in only one block
            assert np.all(row_a == 0.0) or np.all(row_b == 0.0)


class TestComposeLinear:
    def test_shapes(self):
        a = make_family(5, 4)
        C = np.random.default_rng(1).standard_normal((4, 3))
        cl = compose_linear(a, C)
        assert cl.N == 5
        assert cl.M == 3

    def test_W_composes(self):
        a = make_family(8, 4)
        C = np.eye(4, 3)  # pick first 3 params
        cl = compose_linear(a, C)
        np.testing.assert_array_almost_equal(cl.W, a.W[:, :3])

    def test_W_composes_linear_transform(self):
        a = make_family(6, 4)
        C = np.random.default_rng(2).standard_normal((4, 2))
        cl = compose_linear(a, C)
        W_expected = a.W @ C
        np.testing.assert_array_almost_equal(cl.W, W_expected)


class TestTensorSum:
    def test_shapes(self):
        a = make_family(3, 2)
        b = make_family(4, 3)
        ts = tensor_sum(a, b)
        assert ts.N == 12
        assert ts.M == 5

    def test_eigenvalues_are_pairwise_sums(self):
        a = make_family(3, 2)
        b = make_family(2, 1)
        ts = tensor_sum(a, b)
        lam_expected = np.sort(np.add.outer(a.lam0, b.lam0).ravel())
        np.testing.assert_array_almost_equal(np.sort(ts.lam0), lam_expected)


class TestExpectation:
    def test_shape(self):
        a = make_family(5, 2)
        res = expectation(a, np.zeros(2), sigma=0.1, n_samples=50, seed=7)
        assert res["mean_lam"].shape == (5,)
        assert res["lam_at_mean"].shape == (5,)
        assert res["gap"] >= 0

    def test_zero_sigma(self):
        a = make_family(5, 2)
        res = expectation(a, np.array([0.1, 0.2]), sigma=0.0, n_samples=10, seed=7)
        np.testing.assert_array_almost_equal(res["mean_lam"], res["lam_at_mean"])
        assert res["gap"] == pytest.approx(0.0, abs=1e-10)
