"""Core validation, batching, and diagnostics contracts for SFT 0.2."""
import numpy as np
import pytest

import sft
from sft.core import OperatorFamily


def test_rejects_nan_and_inf_inputs():
    with pytest.raises(ValueError, match="A0 contains"):
        OperatorFamily(np.array([[1.0, np.nan], [np.nan, 2.0]]), [])

    fam = sft.families.random(8, 3, seed=0)
    with pytest.raises(ValueError, match="dk contains"):
        fam.predict(np.array([0.0, np.inf, 0.0]))
    with pytest.raises(ValueError, match="target_lam contains"):
        fam.inverse(np.full(fam.N, np.nan))


def test_target_longer_than_active_spectrum_warns_and_truncates():
    fam = sft.families.random(8, 3, seed=1)
    target = np.r_[fam.lam0, 99.0]
    with pytest.warns(UserWarning, match="using the first 8"):
        result = fam.inverse(target, steps=1)
    assert result.k.shape == (fam.M,)


def test_predict_many_matches_loop():
    fam = sft.families.random(12, 4, seed=2)
    rng = np.random.default_rng(2)
    DK = rng.standard_normal((10, fam.M)) * 0.01
    batched = fam.predict_many(DK)
    looped = np.array([fam.predict(dk) for dk in DK])
    np.testing.assert_allclose(batched, looped)


def test_inverse_result_reports_eigh_count_and_time():
    fam = sft.families.random(10, 4, seed=3)
    target = np.sort(fam.lam0 + np.linspace(-0.02, 0.02, fam.N))
    result = fam.inverse(target, steps=3)
    assert result.eigh_count >= 1
    assert result.time_ms >= 0.0
    assert result.regularized(1e-4).reg == pytest.approx(1e-4)


def test_empty_basis_inverse_remains_well_defined():
    fam = OperatorFamily(np.eye(4), [])
    result = fam.inverse(fam.lam0, steps=2)
    k, err, ok = result
    assert k.shape == (0,)
    assert err == pytest.approx(0.0)
    assert ok
