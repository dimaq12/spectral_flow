"""Tests for constructor audit: covariance_operator, gf3, BDSAnalyzer, codec, S1-S5."""
import numpy as np
import pytest
from sft.basis import covariance_operator
from sft.carleman import operator_family_gf3
from sft.order import BDSAnalyzer
from sft.codec import InstantSpectralCodec
from sft.verify import verify_s1_complexity, verify_s2_perturbation_theory, verify_s3_drum_shape, verify_s4_information_bound, verify_s5_rmt_vs_defect
from sft.families import random as randfam


class TestCovarianceOperator:
    def test_2d_data(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((100, 10))
        Cov, Corr = covariance_operator(data)
        assert Cov.shape == (10, 10)
        assert Corr.shape == (10, 10)
        assert np.allclose(np.diag(Corr), 1.0)

    def test_1d_signal(self):
        t = np.linspace(0, 1, 200)
        sig = np.sin(2 * np.pi * 10 * t)
        Cov, Corr = covariance_operator(sig, n=16)
        assert Cov.shape == (16, 16)
        assert np.allclose(Cov, Cov.T)

    def test_corr_bounds(self):
        data = np.random.default_rng(1).standard_normal((50, 8))
        _, Corr = covariance_operator(data)
        assert np.all(Corr >= -1.01)
        assert np.all(Corr <= 1.01)


class TestGF3:
    def test_gf3_operator(self):
        fam = operator_family_gf3(6, 4, seed=42)
        assert fam.N == 6
        assert fam.M == 4
        assert np.all(np.isfinite(fam.W))
        # Values should be from {0,1,2} based matrices
        assert np.max(fam.A0) <= 2.01


class TestBDSAnalyzer:
    def test_register_and_identify(self):
        rng = np.random.default_rng(0)
        bds = BDSAnalyzer()
        bds.register("gauss", rng.standard_normal(300), [10, 30, 90])
        bds.register("uniform", rng.uniform(-3, 3, 300), [10, 30, 90])

        best = bds.identify(rng.standard_normal(300))
        assert best in ("gauss", "uniform", "unknown")

    def test_fingerprints(self):
        rng = np.random.default_rng(1)
        bds = BDSAnalyzer()
        bds.register("test", rng.standard_normal(200))
        fps = bds.fingerprints
        assert "test" in fps
        assert len(fps["test"]) > 0


class TestInstantSpectralCodec:
    def test_roundtrip(self):
        fam = randfam(20, 6, seed=42)
        codec = InstantSpectralCodec(fam)
        dk = np.array([0.1, -0.2, 0.05, 0.0, 0.3, -0.1])
        dk_rec, err = codec.roundtrip(dk)
        assert err < 1.0

    def test_capacity(self):
        fam = randfam(15, 5, seed=42)
        codec = InstantSpectralCodec(fam)
        cap = codec.capacity()
        assert cap >= 0

    def test_optimal_scale(self):
        fam = randfam(10, 4, seed=42)
        codec = InstantSpectralCodec(fam)
        s = codec.optimal_scale(target_snr=100.0)
        assert s > 0

    def test_rank(self):
        fam = randfam(10, 4, seed=42)
        codec = InstantSpectralCodec(fam)
        assert codec.rank == fam.W_rank


class TestS1S5Slammers:
    def test_s1_complexity(self):
        assert verify_s1_complexity() is True

    def test_s2_perturbation(self):
        assert verify_s2_perturbation_theory() is True

    def test_s3_drum_shape(self):
        assert verify_s3_drum_shape() is True

    def test_s4_information(self):
        v = verify_s4_information_bound()
        assert v >= -1e-10

    def test_s5_rmt(self):
        d = verify_s5_rmt_vs_defect()
        assert 0.0 <= d <= 1.0
