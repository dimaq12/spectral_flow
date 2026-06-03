from .base import BaseAdapter
from ..core import OperatorFamily
import numpy as np


class TabularAdapter(BaseAdapter):
    """
    Generic tabular data → feature covariance → SFT.

    Operator: (samples, features) → feature-feature covariance matrix.
    Parameters: per-feature-group and per-feature weights.
    Basis: block-diagonal group masks + per-feature diagonal.

    Works with any rectangular data (numeric). Handles missing values
    via pairwise covariance. Categorical columns should be one-hot encoded.

    Example
    -------
    >>> tab = TabularAdapter(data, feature_groups=['demog','demog','lab','lab','lab'])
    >>> W = tab.kernel
    >>> lam = tab.predict(feature_weight_changes)
    """

    def __init__(self, data: np.ndarray,
                 feature_names: list[str] | None = None,
                 feature_groups: list[str] | None = None):
        self.data = np.asarray(data, dtype=np.float64)
        if self.data.ndim == 1:
            self.data = self.data.reshape(-1, 1)
        self.n_samples, self.n_features = self.data.shape
        self.feature_names = feature_names or [f"f{i}" for i in range(self.n_features)]
        self.feature_groups = feature_groups or ["all"] * self.n_features
        self._build()

    def _build(self):
        D = self.n_features
        valid_counts = np.sum(~np.isnan(self.data), axis=0)
        sums = np.nansum(self.data, axis=0)
        means = np.divide(sums, valid_counts, out=np.zeros(D), where=valid_counts > 0)
        data_centered = self.data - means[None, :]
        mask = ~np.isnan(data_centered)

        Cov = np.zeros((D, D))
        for i in range(D):
            for j in range(D):
                valid = mask[:, i] & mask[:, j]
                if valid.sum() > 1:
                    Cov[i, j] = (data_centered[valid, i] *
                                  data_centered[valid, j]).sum() / (valid.sum() - 1)
        Cov = (Cov + Cov.T) / 2
        self.Cov = Cov

        groups = sorted(set(self.feature_groups))
        group_indices = {g: [i for i, fg in enumerate(self.feature_groups) if fg == g]
                          for g in groups}
        M = len(groups) + D
        basis_arr = np.zeros((M, D, D))

        for k, (g, indices) in enumerate(group_indices.items()):
            for idx in indices:
                basis_arr[k, idx, idx] = 1.0 / max(len(indices), 1)

        for offset, i in enumerate(range(D)):
            basis_arr[len(groups) + offset, i, i] = 1.0

        self._family = OperatorFamily(Cov, list(basis_arr))
