from .base import BaseAdapter
from ..core import OperatorFamily
import numpy as np


class FinancialAdapter(BaseAdapter):
    """
    Multi-asset financial data → correlation/covariance operator → SFT.

    Operator: (T, N) returns → N×N covariance matrix.
    Parameters: per-sector or per-asset volatility/weight.
    Basis: block-diagonal sector masks.

    Performs eigenportfolio decomposition: the kernel W tells how
    much each asset's volatility parameter affects each principal
    component (market mode, sector mode, etc.).

    Example
    -------
    >>> fin = FinancialAdapter(returns, sectors=['tech','tech','fin','fin'])
    >>> W = fin.kernel
    >>> lam = fin.predict(volatility_changes)
    """

    def __init__(self, returns: np.ndarray,
                 sectors: list[str] | None = None,
                 asset_names: list[str] | None = None):
        self.returns = np.asarray(returns, dtype=np.float64)
        if self.returns.ndim == 1:
            self.returns = self.returns.reshape(-1, 1)
        self.T, self.N = self.returns.shape
        self.sectors = sectors or ["asset"] * self.N
        self.asset_names = asset_names or [f"A{i}" for i in range(self.N)]
        self._build()

    def _build(self):
        N = self.N
        returns_centered = self.returns - self.returns.mean(axis=0)[None, :]
        Cov = (returns_centered.T @ returns_centered) / max(self.T - 1, 1)
        Cov = (Cov + Cov.T) / 2
        self.Cov = Cov
        self.Corr = Cov / (np.sqrt(np.diag(Cov))[:, None] *
                            np.sqrt(np.diag(Cov))[None, :] + 1e-15)

        unique_sectors = sorted(set(self.sectors))
        sector_to_idx = {s: [i for i, sec in enumerate(self.sectors) if sec == s]
                         for s in unique_sectors}
        M = len(unique_sectors) + N
        basis_arr = np.zeros((M, N, N))

        for k, (sec, indices) in enumerate(sector_to_idx.items()):
            for idx in indices:
                basis_arr[k, idx, idx] = 1.0 / max(len(indices), 1)

        for offset, i in enumerate(range(N)):
            k = len(unique_sectors) + offset
            basis_arr[k, i, i] = 1.0

        self._family = OperatorFamily(self.Corr, list(basis_arr))
