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

    def inverse(self, target_spectrum: np.ndarray, **kw):
        return self._family.inverse(target_spectrum, **kw)

    @property
    def rank(self) -> int:
        return self._family.W_rank

    @property
    def complexity(self) -> float:
        return self._family.complexity

    @property
    def spectrum(self) -> np.ndarray:
        return self._family.lam0.copy()
