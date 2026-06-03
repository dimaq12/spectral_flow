from .base import BaseAdapter
from ..core import OperatorFamily, edge_laplacian_basis
import numpy as np
from scipy import sparse


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
        self._adjacency_sparse = sparse.issparse(adjacency)
        self.adjacency = adjacency.tocsr().astype(np.float64) if self._adjacency_sparse else np.asarray(adjacency, dtype=np.float64)
        if self.adjacency.shape[0] != self.adjacency.shape[1]:
            raise ValueError("Adjacency must be square")
        self.n_nodes = self.adjacency.shape[0]
        self._build()

    def _build(self):
        N = self.n_nodes
        if self._adjacency_sparse:
            row, col = sparse.triu(self.adjacency != 0, 1).nonzero()
        else:
            row, col = np.triu(self.adjacency != 0, 1).nonzero()
        edges = list(zip(row.tolist(), col.tolist()))
        self.n_edges = len(edges)

        degree = np.asarray(self.adjacency.sum(axis=1)).ravel() if self._adjacency_sparse else np.sum(self.adjacency, axis=1)
        if self._adjacency_sparse or N > 2048:
            A0 = sparse.diags(degree) - (self.adjacency if self._adjacency_sparse else sparse.csr_matrix(self.adjacency))
        else:
            D = np.diag(degree)
            A0 = D - self.adjacency
        self._family = OperatorFamily(A0, edge_laplacian_basis(N, edges))

    @property
    def isospectral_dim(self) -> int:
        """Number of edge directions that don't change the spectrum."""
        return self._family.isospectral_dimension()
