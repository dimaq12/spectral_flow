"""Tests for sft.families — pre-built operator families."""
import numpy as np
import pytest
from sft.families import random, graph_laplacian, toeplitz, diagonal, avoided_crossing_2x2


class TestRandom:
    def test_basic(self):
        fam = random(10, 5, seed=42)
        assert fam.N == 10
        assert fam.M == 5
        assert fam.W.shape == (10, 5)

    def test_reproducible(self):
        fam1 = random(10, 5, seed=42)
        fam2 = random(10, 5, seed=42)
        np.testing.assert_array_equal(fam1.lam0, fam2.lam0)
        np.testing.assert_array_equal(fam1.W, fam2.W)

    def test_different_seeds(self):
        fam1 = random(10, 5, seed=42)
        fam2 = random(10, 5, seed=43)
        assert not np.allclose(fam1.lam0, fam2.lam0)

    def test_sparsity(self):
        fam_sparse = random(20, 3, seed=1, sparsity=0.3)
        fam_dense = random(20, 3, seed=1, sparsity=1.0)
        assert not np.allclose(fam_sparse.W, fam_dense.W)

    def test_negative_N_raises(self):
        with pytest.raises(ValueError):
            random(0, 5)

    def test_negative_M_raises(self):
        with pytest.raises(ValueError):
            random(10, 0)


class TestGraphLaplacian:
    def test_path_graph(self):
        N = 5
        adj = np.zeros((N, N))
        for i in range(N - 1):
            adj[i, i + 1] = adj[i + 1, i] = 1.0
        fam = graph_laplacian(adj)
        assert fam.N == N
        assert fam.M == N - 1

    def test_complete_graph(self):
        N = 4
        adj = np.ones((N, N)) - np.eye(N)
        fam = graph_laplacian(adj)
        assert fam.M == N * (N - 1) // 2

    def test_empty_graph(self):
        adj = np.zeros((5, 5))
        fam = graph_laplacian(adj)
        assert fam.M == 0


class TestToeplitz:
    def test_basic(self):
        fam = toeplitz(10, diagonals=3)
        assert fam.N == 10
        assert fam.M == 3

    def test_max_diagonals(self):
        fam = toeplitz(5, diagonals=10)
        assert fam.M == 4

    def test_N1_raises(self):
        with pytest.raises(ValueError):
            toeplitz(1)


class TestDiagonal:
    def test_rank_one(self):
        fam = diagonal(10)
        assert fam.W_rank == 1
        np.testing.assert_array_almost_equal(fam.W, np.ones((10, 10)))

    def test_complexity(self):
        fam = diagonal(10)
        assert fam.complexity == pytest.approx(0.1)


class TestAvoidedCrossing:
    def test_basic(self):
        fam = avoided_crossing_2x2(Delta=0.3)
        assert fam.N == 2
        assert fam.M == 2

    def test_spectrum_at_origin(self):
        fam = avoided_crossing_2x2(Delta=0.3)
        lam = fam.lam0
        assert abs(abs(lam[0]) - 0.3) < 1e-12
        assert abs(abs(lam[1]) - 0.3) < 1e-12

    def test_kernel_at_origin(self):
        fam = avoided_crossing_2x2(Delta=0.3)
        W = fam.W
        assert W.shape == (2, 2)
