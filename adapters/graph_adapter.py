from .base import BaseAdapter
from ..core import OperatorFamily
import numpy as np


class GraphAdapter(BaseAdapter):
    """
    Graph → Laplacian operator → GraphSFT.

    Operator: graph Laplacian L = D − A.
    Parameters: edge weights.
    Basis: edge Laplacian B_e = (e_u − e_v)(e_u − e_v)^T.

    kernel:  W(i,e) = (v_i(u) − v_i(v))²

    Example
    -------
    >>> net = GraphAdapter(adjacency)
    >>> W = net.kernel
    >>> lam = net.predict(edge_weight_changes)
    """

    def __init__(self, adjacency: np.ndarray):
        self.adjacency = np.asarray(adjacency, dtype=np.float64)
        if self.adjacency.shape[0] != self.adjacency.shape[1]:
            raise ValueError("Adjacency must be square")
        self.n_nodes = self.adjacency.shape[0]
        self._build()

    def _build(self):
        N = self.n_nodes
        row, col = np.triu(self.adjacency != 0, 1).nonzero()
        edges = list(zip(row.tolist(), col.tolist()))
        self.n_edges = len(edges)

        M = len(edges)
        basis_arr = np.zeros((M, N, N))
        for k, (u, v) in enumerate(edges):
            basis_arr[k, u, u] = 1.0
            basis_arr[k, v, v] = 1.0
            basis_arr[k, u, v] = -1.0
            basis_arr[k, v, u] = -1.0

        D = np.diag(np.sum(self.adjacency, axis=1))
        A0 = D - self.adjacency
        self._family = OperatorFamily(A0, list(basis_arr) if M > 0 else [])

    @property
    def isospectral_dim(self) -> int:
        """Number of edge directions that don't change the spectrum."""
        return self._family.isospectral_dimension()
