"""Physics and PDE operator factories for SFT.

These helpers turn familiar model operators into ``OperatorFamily`` instances:
Schrodinger Hamiltonians, grid Laplacians, and PT-symmetric exceptional-point
toys.  The factories keep the old core API intact while giving physicists a
more direct vocabulary.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import sparse

from .core import OperatorFamily, coordinate_diagonal_basis


@dataclass(frozen=True)
class PhysicsOperator:
    """Small wrapper so physics factories read like model builders."""

    name: str
    operator: OperatorFamily
    metadata: dict

    def family(self) -> OperatorFamily:
        return self.operator


def pt_symmetric_2x2(gamma: float = 0.5) -> PhysicsOperator:
    """PT-symmetric Hamiltonian H=[[iγ, 1], [1, -iγ]] with an EP at |γ|=1."""
    A0 = np.array([[1j * gamma, 1.0], [1.0, -1j * gamma]], dtype=np.complex128)
    basis = [
        np.array([[1j, 0.0], [0.0, -1j]], dtype=np.complex128),
        np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128),
    ]
    fam = OperatorFamily(A0, basis, hermitian=False)
    return PhysicsOperator("pt_symmetric_2x2", fam, {"gamma": float(gamma), "ep_gamma": 1.0})


def exceptional_point_2x2() -> PhysicsOperator:
    """Square-root EP model A(z)=[[0,1],[z,0]], z=x+i y."""
    A0 = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    basis = [
        np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.complex128),
        np.array([[0.0, 0.0], [1.0j, 0.0]], dtype=np.complex128),
    ]
    fam = OperatorFamily(A0, basis, hermitian=False)
    return PhysicsOperator("exceptional_point_2x2", fam, {"z": "k0 + i*k1"})


def schrodinger_1d(x: np.ndarray, potential: np.ndarray | callable,
                   mass: float = 1.0, hbar: float = 1.0,
                   max_potential_params: int | None = None) -> PhysicsOperator:
    """Finite-difference 1D Schrodinger operator with diagonal potential controls."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 3:
        raise ValueError("x must contain at least 3 grid points")
    dx = float(np.mean(np.diff(x)))
    if dx <= 0:
        raise ValueError("x must be strictly increasing")
    V = potential(x) if callable(potential) else np.asarray(potential, dtype=np.float64).ravel()
    if V.shape != x.shape:
        raise ValueError(f"potential must have shape {x.shape}, got {V.shape}")
    coeff = hbar ** 2 / (2.0 * mass * dx * dx)
    main = np.full(x.size, 2.0 * coeff) + V
    off = np.full(x.size - 1, -coeff)
    H = sparse.diags([off, main, off], offsets=[-1, 0, 1], format="csc")
    M = x.size if max_potential_params is None else min(int(max_potential_params), x.size)
    fam = OperatorFamily(H, coordinate_diagonal_basis(x.size, M))
    return PhysicsOperator("schrodinger_1d", fam, {"N": x.size, "dx": dx, "mass": mass, "hbar": hbar})


def laplacian_grid(shape: int | tuple[int, int],
                   bc: str = "dirichlet",
                   max_potential_params: int | None = None) -> PhysicsOperator:
    """Sparse finite-difference Laplacian on a 1D or 2D grid."""
    if bc != "dirichlet":
        raise ValueError("only dirichlet boundary conditions are currently supported")
    if isinstance(shape, int):
        n = int(shape)
        if n < 2:
            raise ValueError("1D grid size must be >= 2")
        main = np.full(n, 2.0)
        off = np.full(n - 1, -1.0)
        L = sparse.diags([off, main, off], [-1, 0, 1], format="csc")
        N = n
    else:
        nx, ny = map(int, shape)
        if nx < 2 or ny < 2:
            raise ValueError("2D grid dimensions must be >= 2")
        Lx = laplacian_grid(nx).operator.A0_sparse
        Ly = laplacian_grid(ny).operator.A0_sparse
        L = sparse.kron(Lx, sparse.eye(ny), format="csc") + sparse.kron(sparse.eye(nx), Ly, format="csc")
        N = nx * ny
    M = N if max_potential_params is None else min(int(max_potential_params), N)
    fam = OperatorFamily(L, coordinate_diagonal_basis(N, M))
    return PhysicsOperator("laplacian_grid", fam, {"shape": shape, "bc": bc})
