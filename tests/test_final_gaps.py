"""Tests for sft.basis, arnoldi, and predict_at fix."""
import numpy as np
import pytest
from sft.basis import (toeplitz_from_signal, gaussian_affinity, build_laplacian,
                        path_graph, grid_graph_2d, random_graph, small_world_graph, star_graph)
from sft.arnoldi import arnoldi_iteration, ritz_eigenvalues, krylov_solve
from sft.families import random as randfam


class TestBasis:
    def test_toeplitz_from_signal(self):
        t = np.linspace(0, 1, 200)
        sig = np.sin(2 * np.pi * 10 * t)
        T = toeplitz_from_signal(sig, n=16)
        assert T.shape == (16, 16)
        for i in range(16):
            for j in range(16):
                assert T[i, j] == pytest.approx(T[j, i], abs=1e-10)

    def test_toeplitz_from_signal_toeplitzness(self):
        sig = np.random.default_rng(0).standard_normal(100)
        T = toeplitz_from_signal(sig, n=8)
        for d in range(7):
            diag = np.diag(T, d)
            assert np.allclose(diag, diag[0] * np.ones_like(diag), atol=1e-10)

    def test_gaussian_affinity(self):
        x = np.random.default_rng(1).standard_normal((20, 3))
        W = gaussian_affinity(x, sigma=1.0)
        assert W.shape == (20, 20)
        assert np.all(np.diag(W) == 0)
        assert np.all(W >= 0)
        assert np.all(W <= 1.0)

    def test_gaussian_affinity_auto_sigma(self):
        x = np.random.default_rng(2).standard_normal((10, 2))
        W = gaussian_affinity(x)
        assert W.shape == (10, 10)

    def test_build_laplacian(self):
        edges = [(0, 1), (1, 2), (2, 3)]
        weights = [1.0, 2.0, 1.0]
        L = build_laplacian(4, edges, weights)
        assert L.shape == (4, 4)
        assert L[0, 1] == -1.0
        assert L[1, 2] == -2.0
        assert np.allclose(L.sum(axis=1), 0)

    def test_path_graph(self):
        adj = path_graph(5)
        assert adj[0, 1] == 1.0
        assert adj[4, 3] == 1.0
        assert adj[0, 4] == 0.0  # not connected
        np.testing.assert_array_equal(adj, adj.T)

    def test_grid_graph_2d(self):
        adj = grid_graph_2d(3, 2)
        assert adj.shape == (6, 6)
        np.testing.assert_array_equal(adj, adj.T)

    def test_random_graph(self):
        adj = random_graph(10, p=0.5, seed=42)
        assert adj.shape == (10, 10)
        assert np.sum(adj) > 0

    def test_small_world_graph(self):
        adj = small_world_graph(20, k=4, p=0.1, seed=42)
        assert adj.shape == (20, 20)
        assert np.sum(adj) > 0

    def test_star_graph(self):
        adj = star_graph(5)
        assert adj[0, 1] == 1.0
        assert adj[0, 4] == 1.0
        assert adj[1, 2] == 0.0


class TestArnoldi:
    def test_arnoldi_iteration(self):
        A = np.diag([1.0, 2.0, 3.0, 4.0, 5.0])
        A_fn = lambda v: A @ v
        v0 = np.ones(5)
        V, H = arnoldi_iteration(A_fn, v0, m=3)
        assert V.shape == (5, 3)
        assert H.shape == (3, 3)

    def test_ritz_eigenvalues(self):
        A = np.diag([1.0, 2.0, 3.0])
        A_fn = lambda v: A @ v
        v0 = np.array([1.0, 1.0, 1.0])
        V, H = arnoldi_iteration(A_fn, v0, m=3)
        ritz = ritz_eigenvalues(H)
        assert len(ritz) == 3
        assert np.all(ritz >= 0.5)
        assert np.all(ritz <= 3.5)

    def test_krylov_solve(self):
        A = np.diag([2.0, 3.0, 4.0])
        A_fn = lambda v: A @ v
        b = np.array([1.0, 1.0, 1.0])
        x = krylov_solve(A_fn, b, m=3)
        assert x.shape == (3,)

    def test_arnoldi_breakdown_tolerant(self):
        """Arnoldi should not crash on repeated eigenvalues."""
        A = np.eye(5) * 3.0
        A_fn = lambda v: A @ v
        v0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        V, H = arnoldi_iteration(A_fn, v0, m=5)
        assert V.shape[1] >= 1  # may stop early


class TestPredictAt:
    def test_predict_at_k0_correction(self):
        fam = randfam(10, 3, seed=42)
        k1 = np.array([0.1, 0.2, 0.3])
        lam_k0 = fam.spectrum(np.zeros(3))
        lam_k1_exact = fam.spectrum(k1)
        lam_k1_pred = fam.predict_at(k1)
        err = float(np.max(np.abs(lam_k1_exact - lam_k1_pred)))
        assert err < 0.5  # linear approximation

    def test_predict_at_zero_k(self):
        fam = randfam(10, 3, seed=42)
        pred = fam.predict_at(np.zeros(3))
        np.testing.assert_array_almost_equal(pred, fam.lam0)
