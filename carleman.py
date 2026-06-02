"""
sft.carleman — GF(2) operators, complex Hermitian check.

╔══════════════════════════════════════════════════════════════════════╗
║  INTENT                                                             ║
║  operator_family_gf2:     XOR-based W over random binary matrices.  ║
║  complex_hf_check:        Complex Hermitian family — verify W real. ║
╚══════════════════════════════════════════════════════════════════════╝
║  DEPENDENCIES: sft.core.OperatorFamily, numpy, scipy.linalg.eigh   ║
╚══════════════════════════════════════════════════════════════════════╝
"""
import numpy as np
from scipy import linalg
from .core import OperatorFamily


def operator_family_gf2(n: int, m: int, seed: int = 0) -> OperatorFamily:
    """GF(2)-inspired: random binary symmetric matrices as basis.  Eigenvectors over R."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(m, n, n))
    basis = [np.triu(b.astype(np.float64)) + np.triu(b, 1).T.astype(np.float64) for b in bits]
    A0_bits = rng.integers(0, 2, size=(n, n)).astype(np.float64)
    return OperatorFamily(np.triu(A0_bits) + np.triu(A0_bits, 1).T, basis)


def complex_hf_check(n: int, m: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Complex Hermitian: verify Im(v^H·B·v) ≈ 0 or compute W real+imag."""
    rng = np.random.default_rng(seed)
    Ar = (rng.standard_normal((n, n)) + rng.standard_normal((n, n)).T) / 2
    Ai = (rng.standard_normal((n, n)) - rng.standard_normal((n, n)).T) / 2
    A0 = Ar + 1j * Ai
    basis = []
    for _ in range(m):
        Br = (rng.standard_normal((n, n)) + rng.standard_normal((n, n)).T) / 2
        Bi = (rng.standard_normal((n, n)) - rng.standard_normal((n, n)).T) / 2
        basis.append(Br + 1j * Bi)
    lam, V = linalg.eigh(A0); W, Wi = np.zeros((n, m)), np.zeros((n, m))
    for j in range(m):
        Bj = basis[j]
        for i in range(n):
            v = V[:, i].conj().T @ Bj @ V[:, i]; W[i, j] = float(v.real); Wi[i, j] = float(v.imag)
    return W, Wi


def operator_family_gf3(n: int, m: int, seed: int = 0) -> OperatorFamily:
    """
    GF(3) operator family: ternary {0, 1, 2} basis matrices.

    Random basis values from {0, 1, 2}, symmetrised.
    A₀ is also random GF(3) symmetric.
    """
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 3, size=(m, n, n)).astype(np.float64)
    basis = [np.triu(b) + np.triu(b, 1).T for b in bits]
    A0_bits = rng.integers(0, 3, size=(n, n)).astype(np.float64)
    A0 = np.triu(A0_bits) + np.triu(A0_bits, 1).T
    return OperatorFamily(A0, basis)
