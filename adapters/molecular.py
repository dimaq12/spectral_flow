from .base import BaseAdapter
from ..core import OperatorFamily, edge_laplacian_basis
import numpy as np


class MolecularAdapter(BaseAdapter):
    """
    Molecule → force-field Hessian / bond operator → SFT.

    Operator: weighted adjacency (bond connectivity) + atom-type on diagonal.
    Parameters: per-bond stiffness coefficients.
    Basis: edge Laplacian for each bond.

    Uses Coulomb-like interatomic weighting: w_{ij} ∝ Z_i Z_j / r_{ij}
    for disconnected atoms, standard connectivity for bonded pairs.

    Example
    -------
    >>> mol = MolecularAdapter(positions, atom_types, bonds)
    >>> W = mol.kernel
    >>> lam = mol.predict(bond_parameter_changes)
    """

    def __init__(self, positions: np.ndarray,
                 atom_types: list[str],
                 bonds: list[tuple[int, int]],
                 atom_charges: dict[str, float] | None = None):
        self.positions = np.asarray(positions, dtype=np.float64)
        self.atom_types = atom_types
        self.n_atoms = len(atom_types)
        self.bonds = bonds

        default_charges = {"H": 1.0, "C": 6.0, "N": 7.0, "O": 8.0, "F": 9.0,
                          "Na": 11.0, "Mg": 12.0, "P": 15.0, "S": 16.0,
                          "Cl": 17.0, "K": 19.0, "Ca": 20.0, "Fe": 26.0, "Zn": 30.0}
        self.atom_charges = atom_charges or default_charges
        self._build()

    def _build(self):
        N = self.n_atoms
        X = self.positions
        charges = np.array([self.atom_charges.get(a, 6.0) for a in self.atom_types])

        diff = X[:, None, :] - X[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-10)
        bond_set = set((min(u, v), max(u, v)) for u, v in self.bonds)

        A0 = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                pair = (i, j)
                if pair in bond_set or (j, i) in bond_set:
                    A0[i, j] = A0[j, i] = 1.0
                else:
                    w = charges[i] * charges[j] / dist[i, j]
                    A0[i, j] = A0[j, i] = np.tanh(w * 0.01) * 0.1

        A0[np.diag_indices(N)] = -np.sum(A0, axis=1)

        self._family = OperatorFamily(A0, edge_laplacian_basis(N, self.bonds))
