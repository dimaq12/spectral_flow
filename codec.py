"""
sft.codec — InstantSpectralCodec: M·Δk encode, pinv-decode.

╔══════════════════════════════════════════════════════════════════════╗
║  INTENT                                                             ║
║  One-shot spectral codec:                                            ║
║    encode: y = W·dk        (spectral response from parameter delta)  ║
║    decode: dk ≈ W⁺·y       (reconstruct parameter delta)              ║
║  InstantSpectralCodec wraps an OperatorFamily and provides           ║
║  a simple encode/decode interface with auto-noise estimation.         ║
╚══════════════════════════════════════════════════════════════════════╝
║  DEPENDENCIES: sft.core.OperatorFamily, numpy                        ║
╚══════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations
import numpy as np
from .core import OperatorFamily


class InstantSpectralCodec:
    """
    One-shot spectral encode/decode over an OperatorFamily.

    encoding:      y = W·k       (spectral perturbation)
    decoding:      k ≈ W⁺·y     (parameter reconstruction)

    Auto-selects perturbation scale to match SNR target.

    Example
    -------
    >>> codec = InstantSpectralCodec(fam)
    >>> y = codec.encode(k)
    >>> k_rec = codec.decode(y)
    >>> print(f"reconstruction error: {np.max(np.abs(k - k_rec)):.6f}")
    """

    def __init__(self, family: OperatorFamily):
        self.family = family
        self.W = family.W
        self.W_pinv = family.W_pinv

    def encode(self, dk: np.ndarray, add_noise: float = 0.0) -> np.ndarray:
        """
        y = W·dk.
        add_noise: std of Gaussian noise to add.
        """
        y = self.W @ dk
        if add_noise > 0:
            y += np.random.default_rng().standard_normal(len(y)) * add_noise
        return y

    def decode(self, y: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """
        dk = W⁺·y.  scale: optional step size (default 1.0 = immediate).
        """
        return scale * (self.W_pinv @ y)

    def roundtrip(self, dk: np.ndarray, noise: float = 0.0) -> tuple[np.ndarray, float]:
        """
        Encode → decode.  Returns (dk_reconstructed, reconstruction_error).
        """
        y = self.encode(dk, add_noise=noise)
        dk_rec = self.decode(y)
        err = float(np.max(np.abs(dk - dk_rec)))
        return dk_rec, err

    def capacity(self) -> float:
        """
        Information capacity: log₂(1 + SNR_effective) × rank(W).
        Approximates the number of bits safely encodable.
        """
        s = self.family.W_singular
        if len(s) <= 1:
            return 0.0
        snr = float(s[0] / (s[-1] + 1e-15))
        return max(0.0, np.log2(1.0 + snr) * self.family.W_rank)

    def optimal_scale(self, target_snr: float = 100.0) -> float:
        """
        Scale dk so that ||W·dk|| ≈ target_snr × σ_min.
        """
        s_min = self.family.W_singular[-1] if len(self.family.W_singular) > 0 else 0.0
        if s_min < 1e-15:
            return 1.0
        return float(target_snr * s_min / np.sqrt(self.family.M))

    @property
    def rank(self) -> int:
        return self.family.W_rank
