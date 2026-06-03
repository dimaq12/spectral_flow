"""Exceptional point topology contracts."""
import numpy as np

import sft
from sft.core import OperatorFamily


def ep_family() -> OperatorFamily:
    # A(z) = [[0, 1], [z, 0]], z=x+i y.  z=0 is an exceptional point.
    A0 = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    Bx = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.complex128)
    By = np.array([[0.0, 0.0], [1.0j, 0.0]], dtype=np.complex128)
    return OperatorFamily(A0, [Bx, By], hermitian=False)


def test_complex_monodromy_detects_ep_half_winding():
    fam = ep_family()
    loop = [
        np.array([0.25 * np.cos(t), 0.25 * np.sin(t)])
        for t in np.linspace(0, 2 * np.pi, 80)
    ]
    summary = sft.topology.complex_monodromy(fam, loop)
    winding = summary["windings"][(0, 1)]
    assert summary["tracked"].shape == (80, 2)
    assert abs(abs(winding) - 0.5) < 0.1


def test_exceptional_point_atlas_finds_origin_candidate():
    fam = ep_family()
    atlas = sft.topology.exceptional_point_atlas(
        fam, grid_resolution=21, k_range=(-0.5, 0.5), threshold=1e-8
    )
    best = atlas["candidates"][0]
    assert atlas["min_gap"] < 1e-8
    np.testing.assert_allclose(best["k"], np.zeros(2), atol=1e-12)
    assert best["coalescence"] > 0.9
