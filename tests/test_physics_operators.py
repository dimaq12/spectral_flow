"""Physics/PDE operator factory contracts."""
import numpy as np

import sft
from sft.core import OperatorFamily


def test_schrodinger_1d_builds_sparse_family():
    x = np.linspace(-2.0, 2.0, 40)
    model = sft.physics.schrodinger_1d(x, lambda z: z ** 2, max_potential_params=10)
    fam = model.family()
    assert isinstance(fam, OperatorFamily)
    assert fam._sparse_mode
    assert fam.M == 10
    assert fam.spectrum(np.zeros(fam.M), n_eigs=4).shape == (4,)


def test_laplacian_grid_2d_shape_and_partial_spectrum():
    model = sft.physics.laplacian_grid((5, 4), max_potential_params=6)
    fam = model.family()
    assert fam.N == 20
    assert fam.M == 6
    assert fam.spectrum(np.zeros(fam.M), n_eigs=5).shape == (5,)


def test_exceptional_point_factory_is_nonhermitian():
    fam = sft.physics.exceptional_point_2x2().family()
    assert not fam.hermitian
    assert np.iscomplexobj(fam.W)
    atlas = sft.topology.exceptional_point_atlas(fam, grid_resolution=11, k_range=(-0.2, 0.2), threshold=1e-8)
    assert atlas["min_gap"] < 1e-8


def test_pt_symmetric_factory_has_complex_spectrum_past_ep():
    fam = sft.physics.pt_symmetric_2x2(gamma=1.2).family()
    lam = fam.lam0
    assert np.max(np.abs(np.imag(lam))) > 0.0
