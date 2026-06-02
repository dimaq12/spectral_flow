from .base import BaseAdapter
from ..core import OperatorFamily
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class TimeseriesAdapter(BaseAdapter):
    """
    Timeseries → Hankel/trajectory matrix → SFT (singular spectrum).

    Operator: trajectory (Hankel) matrix H_{i,j} = x[i+j-1].
    Parameters: window coefficients — per-lag weighting.
    Basis: diagonal perturbation per lag.

    Example
    -------
    >>> ts = TimeseriesAdapter(series, window_len=50)
    >>> W = ts.kernel
    >>> lam = ts.predict(lag_weight_changes)
    """

    def __init__(self, series: np.ndarray, window_len: int = 50):
        self.series = np.asarray(series, dtype=np.float64).ravel()
        self.window_len = window_len
        n = len(self.series)
        if self.window_len < 2:
            raise ValueError(f"window_len must be >= 2, got {window_len}")
        if self.window_len >= n:
            raise ValueError(
                f"window_len ({window_len}) must be < series length ({n})"
            )
        self._build()

    def _build(self):
        n = len(self.series)
        L = min(self.window_len, n - self.window_len + 1)
        K = n - L + 1

        H = sliding_window_view(self.series, L)
        H = H.reshape(K, L)
        Cov = (H.T @ H) / K
        Cov = (Cov + Cov.T) / 2
        self.Cov = Cov
        self.L = L

        self._basis = [np.diag(np.eye(L)[lag]) for lag in range(L)]
        self._family = OperatorFamily(Cov, self._basis)
