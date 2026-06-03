from ..core import OperatorFamily
import numpy as np


class BaseAdapter:
    """Base class for all domain adapters. Provides uniform SFT interface."""

    _family: OperatorFamily

    @property
    def family(self) -> OperatorFamily:
        return self._family

    @property
    def kernel(self) -> np.ndarray:
        return self._family.W

    def predict(self, changes: np.ndarray) -> np.ndarray:
        return self._family.predict(changes)

    def predict_at(self, k: np.ndarray) -> np.ndarray:
        return self._family.predict_at(k)

    def at(self, k: np.ndarray) -> np.ndarray:
        return self._family.at(k)

    def shift(self, changes: np.ndarray) -> np.ndarray:
        return self._family.shift(changes)

    def inverse(self, target_spectrum: np.ndarray, **kw):
        return self._family.inverse(target_spectrum, **kw)

    def toward(self, target_spectrum: np.ndarray, **kw):
        return self._family.toward(target_spectrum, **kw)

    def solve(self, target_spectrum: np.ndarray, **kw):
        return self._family.solve(target_spectrum, **kw)

    def refresh(self, k: np.ndarray):
        self._family.refresh(k)
        return self

    def build(self, k: np.ndarray):
        return self._family.build(k)

    def spectrum_at(self, k: np.ndarray, **kw) -> np.ndarray:
        return self._family.spectrum(k, **kw)

    @property
    def W_pinv(self) -> np.ndarray:
        return self._family.W_pinv

    @property
    def lam0(self) -> np.ndarray:
        return self._family.lam0.copy()

    @property
    def rank(self) -> int:
        return self._family.W_rank

    @property
    def complexity(self) -> float:
        return self._family.complexity

    @property
    def reference_spectrum(self) -> np.ndarray:
        """Eigenvalues at the adapter family's current reference operator."""
        return self._family.lam0.copy()

    @property
    def spectrum(self) -> np.ndarray:
        """Backward-compatible alias for reference_spectrum."""
        return self.reference_spectrum
