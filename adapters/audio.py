from .base import BaseAdapter
from ..core import OperatorFamily
import numpy as np


class AudioAdapter(BaseAdapter):
    """
    Audio signal → autocorrelation operator → SFT.

    Operator: frame-wise autocorrelation → averaged Toeplitz matrix.
    Parameters: band gains (EQ).
    Basis: diagonal band masks in frequency domain.

    Example
    -------
    >>> sound = AudioAdapter(signal, sr=44100, n_bands=16)
    >>> W = sound.kernel       # (n_bands, n_bands) spectral response matrix
    >>> lam = sound.predict(gain_changes)  # predict spectrum after EQ
    """

    def __init__(self, signal: np.ndarray, sample_rate: float = 44100.0,
                 frame_size: int = 1024, hop_size: int | None = None,
                 n_bands: int = 16):
        self.signal = np.asarray(signal, dtype=np.float64).ravel()
        self.sr = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size or frame_size // 2
        self.n_bands = n_bands
        self._build()

    def _build(self):
        n = len(self.signal)
        n_frames = max((n - self.frame_size) // self.hop_size + 1, 1)
        self.n_frames = n_frames

        if n >= self.frame_size:
            idx = np.arange(n_frames) * self.hop_size
            frames = self.signal[idx[:, None] + np.arange(self.frame_size)]
        else:
            frames = np.zeros((n_frames, self.frame_size))
            frames[0, :n] = self.signal

        win = np.hanning(self.frame_size)
        frames = frames * win[None, :]

        fft_size = self.frame_size // 2 + 1
        spectrum = np.abs(np.fft.rfft(frames, n=self.frame_size))[:, :fft_size]
        self.power = np.mean(spectrum ** 2, axis=0)

        freqs = np.linspace(0, self.sr / 2, fft_size)
        if self.n_bands > 1:
            lo = np.logspace(np.log10(max(freqs[1], 1)), np.log10(self.sr / 2),
                             self.n_bands + 1)
        else:
            lo = np.array([0, self.sr / 2])
        self.band_edges = lo
        self.band_center = np.sqrt(lo[:-1] * lo[1:])

        masks = np.array([(freqs >= lo[j]) & (freqs < lo[j + 1])
                          for j in range(self.n_bands)], dtype=np.float64)
        self._basis = [np.diag(masks[j]) for j in range(self.n_bands)]

        A0 = np.diag(self.power)
        self._family = OperatorFamily(A0, self._basis)

    @property
    def freqs(self) -> np.ndarray:
        return np.linspace(0, self.sr / 2, len(self.power))

    @property
    def band_freqs(self) -> np.ndarray:
        return self.band_center.copy()
