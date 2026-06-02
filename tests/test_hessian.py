"""Tests for sft.hessian — hessian, hessian_analytic, hessian_sparsity, spectral_curvature."""
import numpy as np
import pytest
from sft.hessian import hessian, hessian_analytic, hessian_sparsity, spectral_curvature
from sft.families import avoided_crossing_2x2, random as random_family


class TestHessian:
    def test_shape(self):
        fam = random_family(10, 3)
        H = hessian(fam)
        assert H.shape == (10, 3, 3)

    def test_finite_diff_vs_analytic_non_degenerate(self):
        """For a non-degenerate system, finite-difference and analytic Hessians
        should agree approximately."""
        fam = random_family(5, 2, seed=42)
        H_fd = hessian(fam, eps=1e-4)
        H_an = hessian_analytic(fam)
        # Check per-level agreement
        for i in range(fam.N):
            np.testing.assert_allclose(H_fd[i], H_an[i], atol=1e-3, rtol=1e-2)

    def test_hessian_at_nonzero_k0(self):
        fam = random_family(5, 2, seed=42)
        k0 = np.array([0.1, -0.2])
        H = hessian(fam, k0=k0)
        assert H.shape == (5, 2, 2)

    def test_memoization_works(self):
        """Ensure hessian doesn't crash with memoization."""
        fam = random_family(8, 3, seed=1)
        H = hessian(fam)
        assert np.all(np.isfinite(H))


class TestHessianAnalytic:
    def test_shape(self):
        fam = random_family(5, 2, seed=42)
        H = hessian_analytic(fam)
        assert H.shape == (5, 2, 2)

    def test_symmetry(self):
        fam = random_family(5, 2, seed=42)
        H = hessian_analytic(fam)
        for i in range(fam.N):
            np.testing.assert_array_almost_equal(H[i], H[i].T)

    def test_avoided_crossing_curvature(self):
        fam = avoided_crossing_2x2(Delta=0.3)
        H = hessian_analytic(fam)
        # For avoided crossing: ground state has negative curvature
        # (concave, local maximum), excited state has positive curvature
        # (convex, local minimum)
        assert H[0, 0, 0] < 0  # ground state: concave
        assert H[1, 0, 0] > 0  # excited state: convex


class TestHessianSparsity:
    def test_all_nonzero(self):
        H = np.ones((5, 3, 3))
        sp = hessian_sparsity(H, tol=0.5)
        assert sp == 0.0

    def test_all_zero(self):
        H = np.zeros((5, 3, 3))
        sp = hessian_sparsity(H, tol=1e-10)
        assert sp == 1.0

    def test_empty(self):
        H = np.zeros((0, 3, 3))
        sp = hessian_sparsity(H)
        assert sp == 1.0


class TestSpectralCurvature:
    def test_shape(self):
        fam = random_family(5, 2, seed=42)
        direction = np.array([1.0, 0.0])
        curv = spectral_curvature(fam, direction)
        assert curv.shape == (5,)

    def test_direction_normalised(self):
        fam = random_family(5, 2, seed=42)
        curv1 = spectral_curvature(fam, np.array([2.0, 0.0]))
        curv2 = spectral_curvature(fam, np.array([1.0, 0.0]))
        np.testing.assert_array_almost_equal(curv1, curv2)

    def test_for_avoided_crossing(self):
        fam = avoided_crossing_2x2(Delta=0.3)
        curv = spectral_curvature(fam, np.array([1.0, 0.0]))
        assert curv[0] < 0  # ground: concave
        assert curv[1] > 0  # excited: convex
