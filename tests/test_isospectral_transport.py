"""Isospectral flow and spectral transport contracts."""
import numpy as np

import sft


def test_isospectral_flow_uses_kernel_nullspace():
    fam = sft.families.diagonal(6)
    flow = fam.isospectral_flow(steps=5, step_size=0.2)
    assert flow["path"].shape == (5, fam.M)
    assert flow["spectra"].shape == (5, fam.N)
    assert flow["nullity"] == fam.M - fam.W_rank
    assert flow["drift"] < 1e-8


def test_isospectral_flow_handles_no_nullspace():
    fam = sft.families.random(8, 3, seed=18)
    flow = fam.isospectral_flow(steps=4)
    assert flow["path"].shape == (4, fam.M)
    assert np.all(np.isfinite(flow["spectra"]))


def test_spectral_transport_path_shapes_and_endpoints():
    current = np.array([0.0, 1.0, 2.0])
    target = np.array([1.0, 3.0, 6.0])
    result = sft.transport.spectral_transport(current, target, n_steps=7)
    path = result["path"]
    assert path.shape == (7, 3)
    np.testing.assert_allclose(path[0], current)
    np.testing.assert_allclose(path[-1], target)
    assert result["w2"] >= 0.0
