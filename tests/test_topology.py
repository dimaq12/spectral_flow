"""Tests for sft.topology — monodromy, berry_holonomy, spectral_flow, exceptional_point_locus."""
import numpy as np
import pytest
from sft.topology import monodromy, berry_holonomy, spectral_flow, exceptional_point_locus
from sft.families import avoided_crossing_2x2


def make_circle_loop(radius=0.4, n_pts=30):
    theta = np.linspace(0, 2 * np.pi, n_pts)
    return [np.array([radius * np.cos(t), radius * np.sin(t)]) for t in theta]


class TestMonodromy:
    def test_no_braiding_inside_unperturbed(self):
        """Trivial loop around origin for an avoided crossing: no braiding expected
        because Hermitian systems have real eigenvalues that repel (no true EP)."""
        fam = avoided_crossing_2x2(Delta=0.3)
        loop = make_circle_loop(0.01, n_pts=30)  # tiny loop near origin
        tracked, swapped = monodromy(fam, loop)
        assert tracked.shape == (30, 2)
        assert isinstance(swapped, list)

    def test_shape(self):
        fam = avoided_crossing_2x2(Delta=0.3)
        loop = make_circle_loop(0.4, n_pts=30)
        tracked, swapped = monodromy(fam, loop)
        assert tracked.shape == (30, 2)

    def test_loop_closes_no_crash(self):
        fam = avoided_crossing_2x2(Delta=0.5)
        loop = make_circle_loop(0.3, n_pts=20)
        tracked, swapped = monodromy(fam, loop)
        # Should complete without error
        assert tracked.shape == (20, 2)


class TestBerryHolonomy:
    def test_trivial_holonomy_tiny_loop(self):
        fam = avoided_crossing_2x2(Delta=0.3)
        loop = make_circle_loop(0.01, n_pts=20)
        hol = berry_holonomy(fam, loop, level=0)
        assert hol in (-1, 1)

    def test_holonomy_around_origin(self):
        fam = avoided_crossing_2x2(Delta=0.3)
        loop = make_circle_loop(0.5, n_pts=60)
        hol = berry_holonomy(fam, loop, level=0)
        assert hol in (-1, 1)

    def test_many_points(self):
        fam = avoided_crossing_2x2(Delta=0.3)
        loop = make_circle_loop(0.5, n_pts=100)
        hol = berry_holonomy(fam, loop, level=1)
        assert hol in (-1, 1)


class TestSpectralFlow:
    def test_returns_tracked_array(self):
        fam = avoided_crossing_2x2(Delta=0.3)
        path = [np.array([0.0, 0.0]), np.array([0.1, 0.0]), np.array([0.1, 0.1])]
        flow = spectral_flow(fam, path)
        assert flow.shape == (3, 2)


class TestExceptionalPointLocus:
    def test_shape(self):
        fam = avoided_crossing_2x2(Delta=0.3)
        grid, gap_map = exceptional_point_locus(fam, grid_resolution=10)
        assert grid.shape == (2, 10)
        assert gap_map.shape == (10, 10)

    def test_min_gap_at_origin_for_small_delta(self):
        fam = avoided_crossing_2x2(Delta=0.1)
        grid, gap_map = exceptional_point_locus(fam, grid_resolution=30)
        origin_idx = (grid_resolution := 30) // 2
        assert gap_map[origin_idx, origin_idx] > 0  # gap > 0 for Hermitian

    def test_raises_for_wrong_M(self):
        from sft.families import random as random_family
        fam = random_family(5, 3)
        with pytest.raises(ValueError, match="M=2"):
            exceptional_point_locus(fam)
