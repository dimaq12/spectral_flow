"""Adapter edge cases for SFT 0.2 stability."""
import numpy as np
import pytest

import sft


def test_audio_empty_signal_is_finite():
    audio = sft.audio(np.array([]), frame_size=32, n_bands=4)
    assert audio.kernel.shape[1] == 4
    assert np.all(np.isfinite(audio.reference_spectrum))


def test_timeseries_short_window_errors_are_clear():
    with pytest.raises(ValueError, match="window_len"):
        sft.timeseries(np.arange(4.0), window_len=4)


def test_tabular_all_nan_column_stays_finite():
    data = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
    tab = sft.tabular(data)
    assert np.all(np.isfinite(tab.kernel))


def test_financial_zero_variance_data_stays_finite():
    fin = sft.financial(np.ones((10, 3)))
    assert np.all(np.isfinite(fin.kernel))


def test_pointcloud_single_point_has_empty_edge_kernel():
    pc = sft.pointcloud(np.array([[0.0, 0.0, 0.0]]), k=3)
    assert pc.family.N == 1
    assert pc.family.M == 0
    assert pc.kernel.shape == (1, 0)


def test_molecular_alias_conflicts_raise():
    positions = np.zeros((2, 3))
    with pytest.raises(TypeError, match="either positional atom types"):
        sft.molecular(positions, ["H", "H"], bonds=[(0, 1)], atom_types=["H", "H"])
    with pytest.raises(TypeError, match="either positional bonds"):
        sft.molecular(positions, ["H", "H"], [(0, 1)], bonds=[(0, 1)])
