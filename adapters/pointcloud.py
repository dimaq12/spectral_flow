from .base import BaseAdapter
from ..core import OperatorFamily
import numpy as np
from scipy.spatial import cKDTree


class PointCloudAdapter(BaseAdapter):
    """
    3D point cloud → graph Laplacian + density operator → SFT.

    Operator: weighted k-NN graph Laplacian built from pairwise Euclidean
    distances, weighted by local density estimates.
    Parameters: per-point density/scale.
    Basis: per-point diagonal perturbations.

    Example
    -------
    >>> pts = np.random.randn(500, 3)
    >>> pc = PointCloudAdapter(pts, k=15)
    >>> W = pc.kernel
    >>> lam = pc.predict(point_weight_changes)
    """

    def __init__(self, points: np.ndarray, k: int = 15,
                 sigma: float | None = None):
        self.points = np.asarray(points, dtype=np.float64)
        if self.points.ndim != 2:
            raise ValueError("Points must be (n_points, d)")
        self.n_points = self.points.shape[0]
        self.k = max(min(k, self.n_points - 1), 1)
        self.sigma = sigma or np.max(np.std(self.points, axis=0))
        self._build()

    def _build(self):
        N = self.n_points
        X = self.points

        from scipy.spatial import cKDTree
        from scipy import sparse as sp

        tree = cKDTree(X)
        k_query = min(self.k + 1, N)
        dist, idx = tree.query(X, k=k_query)
        if k_query == 1:
            dist = dist.reshape(-1, 1)
            idx = idx.reshape(-1, 1)

        weight = np.exp(-dist**2 / (2.0 * self.sigma ** 2))

        i_idx = np.repeat(np.arange(N), k_query)
        j_idx = idx.ravel()
        mask = i_idx != j_idx
        ii, jj = i_idx[mask], j_idx[mask]
        ww = weight.ravel()[mask]

        adj = sp.coo_matrix((np.ones(len(ii)), (ii, jj)), shape=(N, N)).tocsr()
        adj = adj.maximum(adj.T).tolil()
        adj.setdiag(0)
        self.adjacency = adj.toarray()

        W_sp = sp.coo_matrix((ww, (ii, jj)), shape=(N, N)).tocsr()
        W_sp = W_sp.maximum(W_sp.T)

        row, col = adj.nonzero()
        mask_tri = row < col
        edges = list(zip(row[mask_tri].tolist(), col[mask_tri].tolist()))

        M = len(edges)
        max_params = min(M, max(10, 20000 // N) if N > 0 else 10, 500)
        if M > max_params:
            rng_perm = np.random.default_rng(M)
            edges = [edges[p] for p in rng_perm.permutation(M)[:max_params]]
        else:
            max_params = M

        basis_arr = np.zeros((max_params, N, N))
        edge_arr = np.array(edges, dtype=int) if max_params > 0 else np.empty((0, 2), dtype=int)
        if max_params > 0:
            u_idx = edge_arr[:, 0]
            v_idx = edge_arr[:, 1]
            basis_arr[np.arange(max_params), u_idx, u_idx] = 1.0
            basis_arr[np.arange(max_params), v_idx, v_idx] = 1.0
            basis_arr[np.arange(max_params), u_idx, v_idx] = -1.0
            basis_arr[np.arange(max_params), v_idx, u_idx] = -1.0

        W_dense = W_sp.toarray()
        D = np.diag(np.asarray(W_sp.sum(axis=1)).ravel())
        A0 = D - W_dense
        self._family = OperatorFamily(A0, list(basis_arr) if max_params > 0 else [])
