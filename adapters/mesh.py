from .base import BaseAdapter
from ..core import OperatorFamily
import numpy as np


class MeshAdapter(BaseAdapter):
    """
    3D triangular mesh → cotangent-weight Laplacian → SFT.

    Operator: cotangent-weight Laplacian derived from mesh geometry.
    Parameters: per-face or per-vertex stiffness.
    Basis: per-edge Laplacian (each edge weight is a parameter).

    The cotangent Laplacian is the standard discrete Laplace-Beltrami
    operator used across geometry processing, shape analysis, and PDE
    solvers on surfaces.

    Example
    -------
    >>> mesh = MeshAdapter(vertices, faces)
    >>> W = mesh.kernel
    >>> lam = mesh.predict(edge_weight_changes)
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=int)
        if self.faces.shape[1] != 3:
            raise ValueError("Faces must be triangles (N_faces, 3)")
        self.n_vertices = self.vertices.shape[0]
        self.n_faces = self.faces.shape[0]
        self._build()

    def _build(self):
        V = self.vertices
        F = self.faces
        Nv = self.n_vertices
        Nf = self.n_faces

        v0 = V[F[:, 0]]
        v1 = V[F[:, 1]]
        v2 = V[F[:, 2]]
        e01 = v1 - v0
        e12 = v2 - v1
        e20 = v0 - v2

        def _cot(a, b):
            dot_val = np.sum(a * b, axis=1)
            cross = np.cross(a, b)
            cross_norm = np.sqrt(np.sum(cross ** 2, axis=1))
            return dot_val / (cross_norm + 1e-15)

        cot0 = _cot(e20, -e12)
        cot1 = _cot(e01, -e20)
        cot2 = _cot(e12, -e01)

        L = np.zeros((Nv, Nv))
        for f_idx in range(Nf):
            i0, i1, i2 = F[f_idx]
            L[i0, i1] -= cot2[f_idx]
            L[i1, i0] -= cot2[f_idx]
            L[i1, i2] -= cot0[f_idx]
            L[i2, i1] -= cot0[f_idx]
            L[i2, i0] -= cot1[f_idx]
            L[i0, i2] -= cot1[f_idx]

        row_sum = np.sum(L, axis=1) + 1e-15
        for i in range(Nv):
            L[i, i] -= row_sum[i]

        L = (L + L.T) / 2
        self.Laplacian = L

        row, col = np.triu(np.abs(L) > 1e-15, 1).nonzero()
        edges = list(zip(row.tolist(), col.tolist()))
        self.n_edges = len(edges)

        M = len(edges)
        basis_arr = np.zeros((M, Nv, Nv))
        for k, (u, v) in enumerate(edges):
            basis_arr[k, u, u] = 1.0
            basis_arr[k, v, v] = 1.0
            basis_arr[k, u, v] = -1.0
            basis_arr[k, v, u] = -1.0

        self._family = OperatorFamily(L, list(basis_arr) if M > 0 else [])
