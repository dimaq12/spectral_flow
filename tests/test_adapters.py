"""Tests for sft.adapters — domain adapters."""
import numpy as np
import pytest
from sft.adapters import (
    BaseAdapter, AudioAdapter, ImageAdapter, GraphAdapter,
    TextAdapter, TimeseriesAdapter,
)


class TestBaseAdapter:
    def test_properties_exist(self):
        """Verify BaseAdapter defines all expected attributes."""
        rng = np.random.default_rng(0)
        A0 = rng.standard_normal((5, 5))
        A0 = (A0 + A0.T) / 2
        from sft.core import OperatorFamily

        class MinimalAdapter(BaseAdapter):
            def __init__(self):
                self._family = OperatorFamily(A0, [np.eye(5)])

        adapter = MinimalAdapter()
        assert adapter.family is not None
        assert adapter.kernel.shape == (5, 1)
        assert adapter.rank == 1
        assert isinstance(adapter.complexity, float)
        assert adapter.spectrum.shape == (5,)


class TestAudioAdapter:
    def test_basic(self):
        rng = np.random.default_rng(0)
        sr = 44100
        t = np.arange(sr // 10) / sr
        signal = np.sin(2 * np.pi * 440 * t) + 0.01 * rng.standard_normal(len(t))
        adapter = AudioAdapter(signal, sample_rate=sr, frame_size=1024, n_bands=4)
        assert adapter.kernel.shape[1] == 4
        assert adapter.rank >= 1
        assert adapter.complexity <= 1.0

    def test_predict(self):
        rng = np.random.default_rng(1)
        signal = rng.standard_normal(44100)
        adapter = AudioAdapter(signal, n_bands=4)
        dk = np.zeros(4)
        dk[0] = 0.1
        lam_pred = adapter.predict(dk)
        assert lam_pred.shape[0] == adapter.spectrum.shape[0]

    def test_freqs(self):
        rng = np.random.default_rng(2)
        signal = rng.standard_normal(44100)
        adapter = AudioAdapter(signal, n_bands=4)
        assert len(adapter.freqs) == len(adapter.spectrum)
        assert np.all(adapter.freqs >= 0)

    def test_band_freqs(self):
        rng = np.random.default_rng(3)
        signal = rng.standard_normal(44100)
        adapter = AudioAdapter(signal, n_bands=4)
        assert len(adapter.band_freqs) == 4


class TestImageAdapter:
    def test_basic(self):
        rng = np.random.default_rng(0)
        img = rng.standard_normal((32, 32))
        adapter = ImageAdapter(img, patch_size=8, n_regions=4)
        assert adapter.kernel.shape[1] == 4
        assert adapter.n_patches > 0

    def test_rgb_to_gray(self):
        rng = np.random.default_rng(1)
        img = rng.standard_normal((32, 32, 3))
        adapter = ImageAdapter(img, patch_size=8, n_regions=4)
        assert adapter.image.ndim == 2

    def test_too_small_raises(self):
        rng = np.random.default_rng(2)
        img = rng.standard_normal((4, 4))
        with pytest.raises(ValueError):
            ImageAdapter(img, patch_size=8)

    def test_complexity(self):
        rng = np.random.default_rng(3)
        img = rng.standard_normal((32, 32))
        adapter = ImageAdapter(img, patch_size=8, n_regions=4)
        assert 0 <= adapter.complexity <= 1


class TestGraphAdapter:
    def test_basic(self):
        N = 10
        adj = np.zeros((N, N))
        for i in range(N - 1):
            adj[i, i + 1] = adj[i + 1, i] = 1.0
        adapter = GraphAdapter(adj)
        assert adapter.n_edges == N - 1
        assert adapter.kernel.shape == (N, N - 1)

    def test_isospectral_dim(self):
        N = 5
        adj = np.ones((N, N)) - np.eye(N)
        adapter = GraphAdapter(adj)
        assert adapter.isospectral_dim >= 0

    def test_non_square_raises(self):
        adj = np.zeros((5, 3))
        with pytest.raises(ValueError, match="square"):
            GraphAdapter(adj)


class TestTextAdapter:
    def test_basic(self):
        texts = ["hello world", "hello there", "world of code"]
        adapter = TextAdapter(texts, max_words=20, window=3)
        assert adapter.n_words > 0
        assert adapter.kernel.shape[1] >= 1

    def test_single_string(self):
        adapter = TextAdapter("hello world hello", max_words=10)
        assert adapter.n_words > 0

    def test_empty_texts(self):
        adapter = TextAdapter(["", ""], max_words=10)
        assert adapter.n_words == 0
        assert adapter.kernel.shape[1] >= 1

    def test_complexity(self):
        texts = ["hello world", "hello there"]
        adapter = TextAdapter(texts, max_words=10, window=2)
        assert 0 <= adapter.complexity <= 1

    def test_spectrum(self):
        texts = ["the cat sat on the mat", "the dog sat on the log"]
        adapter = TextAdapter(texts, max_words=20)
        assert adapter.spectrum.shape[0] == adapter.n_words


class TestTimeseriesAdapter:
    def test_basic(self):
        rng = np.random.default_rng(0)
        ts = rng.standard_normal(200)
        adapter = TimeseriesAdapter(ts, window_len=20)
        assert adapter.kernel.shape[1] == adapter.L
        assert adapter.L <= 20

    def test_predict(self):
        rng = np.random.default_rng(1)
        ts = np.sin(np.linspace(0, 4 * np.pi, 300)) + 0.01 * rng.standard_normal(300)
        adapter = TimeseriesAdapter(ts, window_len=20)
        dk = np.zeros(adapter.L)
        dk[0] = 0.1
        lam_pred = adapter.predict(dk)
        assert len(lam_pred) == adapter.L

    def test_window_too_small_raises(self):
        rng = np.random.default_rng(2)
        ts = rng.standard_normal(100)
        with pytest.raises(ValueError, match=">= 2"):
            TimeseriesAdapter(ts, window_len=1)

    def test_window_too_large_raises(self):
        rng = np.random.default_rng(3)
        ts = rng.standard_normal(50)
        with pytest.raises(ValueError, match="must be < series length"):
            TimeseriesAdapter(ts, window_len=100)

    def test_complexity(self):
        rng = np.random.default_rng(4)
        ts = rng.standard_normal(200)
        adapter = TimeseriesAdapter(ts, window_len=20)
        assert 0 <= adapter.complexity <= 1

    def test_spectrum(self):
        rng = np.random.default_rng(5)
        ts = rng.standard_normal(200)
        adapter = TimeseriesAdapter(ts, window_len=20)
        assert adapter.spectrum.shape[0] == adapter.L

    def test_short_series_valid(self):
        rng = np.random.default_rng(6)
        ts = rng.standard_normal(10)
        # window_len=2 is valid for series of length 10
        adapter = TimeseriesAdapter(ts, window_len=5)
        assert adapter.L >= 2
