"""
sft.core — Spectral Flow Transform: kernel, prediction, inverse design.

╔══════════════════════════════════════════════════════════════════════╗
║  INTENT                                                             ║
║  Central engine of SFT. Encodes a parametric operator family        ║
║  A(k) = A₀ + Σ kⱼ·Bⱼ and computes the Hellmann-Feynman kernel      ║
║  W(i,j) = v_iᵀ·Bⱼ·v_i = ∂λᵢ/∂kⱼ — the first-order sensitivity     ║
║  of every eigenvalue to every parameter.  From W, we get:           ║
║    • prediction  — λ(k₀+dk) ≈ λ₀ + W·dk                            ║
║    • inverse     — find k such that λ(k) ≈ target                   ║
║    • complexity  — rank(W)/N  (structural complexity ratio)         ║
║    • nullspace   — ker(W)  (silent parameter directions)            ║
╚══════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════╗
║  CONCEPT                                                            ║
║  Hellmann-Feynman theorem for linear operator families:             ║
║    ∂λᵢ/∂kⱼ = ⟨vᵢ| ∂A/∂kⱼ |vᵢ⟩ = vᵢᵀ·Bⱼ·vᵢ                        ║
║  This is EXACT for non-degenerate eigenvalues at k=0.               ║
║  W is an (N×M) matrix — one row per eigenvalue, one column per      ║
║  parameter.  Its SVD reveals:                                       ║
║    • rank(W)     — effective number of controllable spectral modes  ║
║    • W⁺          — Moore-Penrose pseudoinverse for inverse design   ║
║    • condition   — σ_max/σ_min  (navigability of parameter space)   ║
╚══════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════╗
║  DEPENDENCIES                                                       ║
║  ┌── numpy (ndarray, einsum, tensordot)                             ║
║  ├── scipy.linalg (eigh, svd, null_space)                           ║
║  └── scipy.sparse (issparse, eigsh — for large sparse operators)    ║
║                                                                     ║
║  CALLED BY:                                                         ║
║  ┌── sft.families.* (constructors)                                  ║
║  ├── sft.algebra.* (⊕, ∘, ⊗ compose OperatorFamily instances)       ║
║  ├── sft.adapters.* (12 domain adapters build OperatorFamilies)     ║
║  ├── sft.topology.* (monodromy, Berry — call build/spectrum)        ║
║  ├── sft.homotopy.track_homotopy (calls set_reference/spectrum)     ║
║  ├── sft.invariants.* (svd_kurtosis uses W matrix)                  ║
║  ├── sft.hacks → sft.inversion.* (bottleneck, fixed_point)          ║
║  └── sft.constructor.synthesize (builds OperatorFamily)             ║
╚══════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════╗
║  GRAPH (simplified call graph)                                      ║
║                                                                     ║
║  OperatorFamily.__init__                                             ║
║  └── set_reference(A0, k0=np.zeros(M))                              ║
║      ├── _eigh(A) → scipy.linalg.eigh  or  sparse.linalg.eigsh     ║
║      ├── BsV = _basis_stack @ V  (batched matmul)                   ║
║      ├── W = einsum('ni,mni->im', V, BsV)  (batched)               ║
║      └── scipy.linalg.svd(W) → W⁺, rank, singular_vals             ║
║                                                                     ║
║  .build(k) → tensordot(k, basis_stack) + A0  (cached)              ║
║  .spectrum(k) → build(k) → eigh → sort                             ║
║  .predict(dk) → lam0 + W @ dk                                      ║
║  .predict_at(k) → lam0 + W @ (k - k0_ref)                          ║
║  .inverse(target) → Newton loop: spectrum→error→dk→k+=dk           ║
║      └── adaptive refresh on stagnation                             ║
║  .nullspace() → linalg.null_space(W)                                ║
╚══════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import time
import warnings
import numpy as np
from scipy import linalg, sparse

_is_sparse = lambda x: sparse.issparse(x) if hasattr(sparse, 'issparse') else False


@dataclass(frozen=True)
class ReferenceState:
    k_ref: np.ndarray
    lam: np.ndarray
    vecs: np.ndarray


@dataclass(frozen=True)
class KernelState:
    W: np.ndarray
    W_pinv: np.ndarray
    rank: int
    singular: np.ndarray


@dataclass(frozen=True)
class BiorthogonalState:
    """Non-Hermitian spectral reference with <L_i|R_j> = delta_ij."""
    k_ref: np.ndarray
    lam: np.ndarray
    left_vecs: np.ndarray
    right_vecs: np.ndarray


class InverseResult(tuple):
    """Backward-compatible inverse result.

    Behaves like ``(k, error, converged)`` for tuple unpacking while exposing
    diagnostics needed by heavy inverse tasks.
    """

    def __new__(cls, k: np.ndarray, error: float, converged: bool,
                steps: int = 0, n_refresh: int = 0,
                condition_number: float = np.inf,
                eigh_count: int = 0, time_ms: float = 0.0,
                method: str = "linear", trajectory: np.ndarray | None = None,
                residual_history: np.ndarray | None = None,
                hessian_count: int = 0):
        obj = super().__new__(cls, (k, error, converged))
        obj.steps = int(steps)
        obj.n_refresh = int(n_refresh)
        obj.condition_number = float(condition_number)
        obj.eigh_count = int(eigh_count)
        obj.time_ms = float(time_ms)
        obj.method = str(method)
        obj.trajectory = trajectory
        obj.residual_history = residual_history
        obj.hessian_count = int(hessian_count)
        return obj

    @property
    def k(self) -> np.ndarray:
        return self[0]

    @property
    def error(self) -> float:
        return self[1]

    @property
    def converged(self) -> bool:
        return self[2]

    def regularized(self, reg: float = 1e-6) -> "InverseResult":
        """Fluent no-op annotation for APIs that chain solver options."""
        obj = InverseResult(self.k, self.error, self.converged,
                            steps=self.steps, n_refresh=self.n_refresh,
                            condition_number=self.condition_number,
                            eigh_count=self.eigh_count, time_ms=self.time_ms,
                            method=self.method, trajectory=self.trajectory,
                            residual_history=self.residual_history,
                            hessian_count=self.hessian_count)
        obj.reg = float(reg)
        return obj


class _DenseBasisBackend:
    kind = "dense"

    def __init__(self, basis: list[np.ndarray], N: int):
        self.N = N
        self.M = len(basis)
        self._basis_list = [
            B.toarray() if _is_sparse(B) else np.asarray(B).copy()
            for B in basis
        ]
        for idx, B in enumerate(self._basis_list):
            if B.shape != (self.N, self.N):
                raise ValueError(f"basis[{idx}] must have shape ({self.N}, {self.N}), got {B.shape}")
            if not np.all(np.isfinite(B)):
                raise ValueError(f"basis[{idx}] contains NaN or Inf")
        self.stack = (
            np.stack(self._basis_list)
            if self.M > 0
            else np.empty((0, self.N, self.N))
        )

    @property
    def basis(self) -> list[np.ndarray]:
        return self._basis_list

    @property
    def materialized_elements(self) -> int:
        return self.M * self.N * self.N

    def build_delta(self, k: np.ndarray) -> np.ndarray:
        if self.M == 0:
            return np.zeros((self.N, self.N), dtype=self.stack.dtype)
        return np.tensordot(k, self.stack, axes=((0,), (0,)))

    def build_delta_sparse(self, k: np.ndarray):
        return sparse.csc_matrix(self.build_delta(k))

    def kernel(self, V: np.ndarray, batch_limit: int) -> np.ndarray:
        if self.M == 0:
            return np.empty((V.shape[1], 0))
        elements_per_basis = self.N * self.N
        batch_M = max(1, batch_limit // max(elements_per_basis, 1))
        if self.M <= batch_M:
            BsV = self.stack @ V
            return np.einsum('ni,mni->im', np.conjugate(V), BsV)
        W = np.empty((V.shape[1], self.M), dtype=np.result_type(V, self.stack))
        for b in range(0, self.M, batch_M):
            be = min(b + batch_M, self.M)
            BsV_b = self.stack[b:be] @ V
            W[:, b:be] = np.einsum('ni,mni->im', np.conjugate(V), BsV_b)
        return W

    def eigen_tensor(self, V: np.ndarray) -> np.ndarray:
        return V.conj().T @ self.stack @ V


class _SparseBasisBackend:
    kind = "sparse"

    def __init__(self, basis: list, N: int):
        self.N = N
        self.M = len(basis)
        self._basis_sparse = [B.tocsc() for B in basis]
        for idx, B in enumerate(self._basis_sparse):
            if B.shape != (self.N, self.N):
                raise ValueError(f"basis[{idx}] must have shape ({self.N}, {self.N}), got {B.shape}")
            if not np.all(np.isfinite(B.data)):
                raise ValueError(f"basis[{idx}] contains NaN or Inf")
        self._basis_list: list[np.ndarray] | None = None

    @property
    def basis(self) -> list[np.ndarray]:
        if self._basis_list is None:
            self._basis_list = [B.toarray() for B in self._basis_sparse]
        return self._basis_list

    @property
    def materialized_elements(self) -> int:
        return sum(B.nnz for B in self._basis_sparse)

    def build_delta(self, k: np.ndarray):
        return self.build_delta_sparse(k).toarray()

    def build_delta_sparse(self, k: np.ndarray):
        result = sparse.csc_matrix((self.N, self.N))
        for weight, B in zip(k, self._basis_sparse):
            if weight != 0:
                result = result + weight * B
        return result

    def kernel(self, V: np.ndarray, batch_limit: int) -> np.ndarray:
        if self.M == 0:
            return np.empty((V.shape[1], 0))
        W = np.empty((V.shape[1], self.M), dtype=np.result_type(V, complex if np.iscomplexobj(V) else float))
        for j, B in enumerate(self._basis_sparse):
            BV = B @ V
            W[:, j] = np.einsum('ni,ni->i', np.conjugate(V), BV)
        return W

    def eigen_tensor(self, V: np.ndarray) -> np.ndarray:
        out = np.empty((self.M, V.shape[1], V.shape[1]), dtype=np.result_type(V, complex))
        for j, B in enumerate(self._basis_sparse):
            out[j] = V.conj().T @ (B @ V)
        return out


class _EdgeLaplacianBasisBackend:
    """Implicit edge-Laplacian basis B_e=(e_u-e_v)(e_u-e_v)^T."""

    kind = "edge_laplacian"

    def __init__(self, N: int, edges: list[tuple[int, int]]):
        self.N = N
        self.edges = [(int(u), int(v)) for u, v in edges]
        for u, v in self.edges:
            if not (0 <= u < N and 0 <= v < N):
                raise ValueError(f"edge ({u}, {v}) is outside node range 0..{N - 1}")
        self.M = len(self.edges)
        self._u = np.array([u for u, _ in self.edges], dtype=np.int64)
        self._v = np.array([v for _, v in self.edges], dtype=np.int64)
        self._basis_list: list[np.ndarray] | None = None

    @property
    def basis(self) -> list[np.ndarray]:
        if self._basis_list is None:
            basis = []
            for u, v in self.edges:
                B = np.zeros((self.N, self.N))
                B[u, u] = B[v, v] = 1.0
                B[u, v] = B[v, u] = -1.0
                basis.append(B)
            self._basis_list = basis
        return self._basis_list

    @property
    def materialized_elements(self) -> int:
        return self.M * self.N * self.N

    def build_delta(self, k: np.ndarray) -> np.ndarray:
        L = np.zeros((self.N, self.N))
        for weight, u, v in zip(k, self._u, self._v):
            L[u, u] += weight
            L[v, v] += weight
            L[u, v] -= weight
            L[v, u] -= weight
        return L

    def build_delta_sparse(self, k: np.ndarray):
        if self.M == 0:
            return sparse.csc_matrix((self.N, self.N))
        row = np.concatenate([self._u, self._v, self._u, self._v])
        col = np.concatenate([self._u, self._v, self._v, self._u])
        data = np.concatenate([k, k, -k, -k])
        return sparse.coo_matrix((data, (row, col)), shape=(self.N, self.N)).tocsc()

    def kernel(self, V: np.ndarray, batch_limit: int) -> np.ndarray:
        if self.M == 0:
            return np.empty((V.shape[1], 0))
        diff = V[self._u, :] - V[self._v, :]
        return diff.T ** 2

    def eigen_tensor(self, V: np.ndarray) -> np.ndarray:
        if self.M == 0:
            return np.empty((0, V.shape[1], V.shape[1]))
        diff = V[self._u, :] - V[self._v, :]
        return diff[:, :, None] * diff[:, None, :]


def edge_laplacian_basis(N: int, edges: list[tuple[int, int]]) -> _EdgeLaplacianBasisBackend:
    """Create an implicit edge-Laplacian basis backend for OperatorFamily."""
    return _EdgeLaplacianBasisBackend(N, edges)


class _CoordinateDiagonalBasisBackend:
    """Implicit basis B_j = diag(e_j), used by task constructors."""

    kind = "coordinate_diagonal"

    def __init__(self, N: int, M: int | None = None):
        self.N = N
        self.M = N if M is None else min(int(M), N)
        self._basis_list: list[np.ndarray] | None = None

    @property
    def basis(self) -> list[np.ndarray]:
        if self._basis_list is None:
            self._basis_list = [np.diag(np.eye(self.N)[i]) for i in range(self.M)]
        return self._basis_list

    @property
    def materialized_elements(self) -> int:
        return self.M * self.N * self.N

    def build_delta(self, k: np.ndarray) -> np.ndarray:
        D = np.zeros((self.N, self.N))
        idx = np.arange(self.M)
        D[idx, idx] = k
        return D

    def build_delta_sparse(self, k: np.ndarray):
        idx = np.arange(self.M)
        return sparse.coo_matrix((k, (idx, idx)), shape=(self.N, self.N)).tocsc()

    def kernel(self, V: np.ndarray, batch_limit: int) -> np.ndarray:
        return V[:self.M, :].T ** 2

    def eigen_tensor(self, V: np.ndarray) -> np.ndarray:
        rows = V[:self.M, :]
        return rows[:, :, None] * rows[:, None, :]


def coordinate_diagonal_basis(N: int, M: int | None = None) -> _CoordinateDiagonalBasisBackend:
    """Create an implicit coordinate-diagonal basis backend."""
    return _CoordinateDiagonalBasisBackend(N, M)


class _RepeatedIdentityBasisBackend:
    """Implicit basis B_j = I for all j, preserving families.diagonal()."""

    kind = "repeated_identity"

    def __init__(self, N: int, M: int | None = None):
        self.N = N
        self.M = N if M is None else int(M)
        self._basis_list: list[np.ndarray] | None = None

    @property
    def basis(self) -> list[np.ndarray]:
        if self._basis_list is None:
            eye = np.eye(self.N)
            self._basis_list = [eye.copy() for _ in range(self.M)]
        return self._basis_list

    @property
    def materialized_elements(self) -> int:
        return self.M * self.N * self.N

    def build_delta(self, k: np.ndarray) -> np.ndarray:
        return np.eye(self.N) * float(np.sum(k))

    def build_delta_sparse(self, k: np.ndarray):
        return sparse.eye(self.N, format="csc") * float(np.sum(k))

    def kernel(self, V: np.ndarray, batch_limit: int) -> np.ndarray:
        return np.ones((V.shape[1], self.M))

    def eigen_tensor(self, V: np.ndarray) -> np.ndarray:
        eye = np.eye(V.shape[1])
        return np.repeat(eye[None, :, :], self.M, axis=0)


def repeated_identity_basis(N: int, M: int | None = None) -> _RepeatedIdentityBasisBackend:
    """Create an implicit repeated-identity basis backend."""
    return _RepeatedIdentityBasisBackend(N, M)


# ────────────────────────────────────────────────────
#  OperatorFamily
# ────────────────────────────────────────────────────
#  CONTRACT:
#    PRE:  A0 is (N,N) symmetric, basis is list of (N,N) matrices.
#    INV:  W is (N,M).  W(i,j) = v_i^T·B_j·v_i at current reference.
#    INV:  lam0, vecs match the eigh(A_ref) sorted ascending.
#    INV:  W_pinv ≈ W⁺  (Moore-Penrose within svd_tol).
#    POST (set_reference): all cached data refreshed from A_ref.
#
#  WHY:  This is the central abstraction of SFT.  Everything else —
#    prediction, inverse design, topology, Hessian, adapters — builds
#    on OperatorFamily.  The kernel W compresses the entire spectral
#    response into one matrix.
#
#  EXAMPLE:
#    >>> fam = OperatorFamily(A0, [B1, B2, B3])
#    >>> W = fam.W                   # (N, 3) kernel
#    >>> lam = fam.predict([0.1, 0, 0])  # 1st-order prediction
#    >>> k, err, ok = fam.inverse(target_lam)  # inverse design
#    >>> print(fam.complexity)       # rank(W) / N
# ────────────────────────────────────────────────────

class OperatorFamily:
    def __init__(self, A0: np.ndarray, basis: list[np.ndarray],
                 svd_tol: float = 1e-8, convergence_tol: float = 1e-2,
                 k_eigs: int | None = None, hermitian: bool = True):
        """
        CONTRACT:
            PRE:  A0.shape == (N,N), all B in basis have shape (N,N)
            PRE:  A0 and all B_j are symmetric (not enforced, but assumed)
            POST: self.W is computed via eigh+SVD (cached)
            POST: self._k0_ref = zeros(M)  — reference parameter vector
        """
        self.hermitian = bool(hermitian)
        self._sparse_mode = _is_sparse(A0)
        if self._sparse_mode:
            if not np.all(np.isfinite(A0.data)):
                raise ValueError("A0 contains NaN or Inf")
        elif not np.all(np.isfinite(np.asarray(A0))):
            raise ValueError("A0 contains NaN or Inf")
        self.N = A0.shape[0]
        if A0.shape[0] != A0.shape[1]:
            raise ValueError(f"A0 must be a square matrix, got {A0.shape}")
        if hasattr(basis, "build_delta") and hasattr(basis, "kernel"):
            self._basis_backend = basis
        elif self._sparse_mode and isinstance(basis, list) and all(_is_sparse(B) for B in basis):
            self._basis_backend = _SparseBasisBackend(basis, self.N)
        else:
            self._basis_backend = _DenseBasisBackend(basis, self.N)
        self.M = self._basis_backend.M
        self.svd_tol = svd_tol
        self.convergence_tol = convergence_tol
        self.k_eigs = k_eigs or min(self.N, 128)
        self._stale = False
        self._k0_ref = np.zeros(self.M)
        self._eigh_count = 0

        if self._sparse_mode:
            self.A0_sparse = A0.tocsc()
            self.A0 = self.A0_sparse if self.N > 2048 else self.A0_sparse.toarray()
            self._basis_sparse = [B.tocsc() for B in basis] if isinstance(basis, list) and self.M > 0 else []
        else:
            dtype = np.float64 if self.hermitian and not np.iscomplexobj(A0) else np.complex128
            self.A0 = np.asarray(A0, dtype=dtype).copy()
            self.A0_sparse = None
            self._basis_sparse = []
        self._basis_list = self._basis_backend.basis if self._basis_backend.kind == "dense" else []
        self._basis_stack = getattr(self._basis_backend, "stack", np.empty((0, self.N, self.N)))
        if getattr(self._basis_backend, "materialized_elements", 0) > 50_000_000 and self._basis_backend.kind == "dense":
            warnings.warn(
                "Dense basis requires more than 50M elements; use a structured basis backend for heavy tasks.",
                ResourceWarning,
                stacklevel=2,
            )

        self._lam0: Optional[np.ndarray] = None
        self._vecs: Optional[np.ndarray] = None
        self._W: Optional[np.ndarray] = None
        self._Wp: Optional[np.ndarray] = None
        self._W_rank: Optional[int] = None
        self._W_singular: Optional[np.ndarray] = None
        self._left_vecs: Optional[np.ndarray] = None
        self._reference_state: Optional[ReferenceState] = None
        self._biorthogonal_state: Optional[BiorthogonalState] = None
        self._kernel_state: Optional[KernelState] = None
        self.set_reference(self.A0_sparse if self._sparse_mode else self.A0, k0=np.zeros(self.M))

    @property
    def basis(self) -> list[np.ndarray]:
        """CONTRACT: returns the basis matrices B_j as a list. Immutable copy."""
        return self._basis_backend.basis

    @property
    def basis_kind(self) -> str:
        """Internal basis representation: dense, edge_laplacian, ..."""
        return self._basis_backend.kind

    @property
    def reference_state(self) -> ReferenceState:
        """Current spectral reference: k_ref, eigenvalues, eigenvectors."""
        if self._reference_state is None or self._stale:
            self.set_reference(self.A0, k0=np.zeros(self.M))
        return self._reference_state  # type: ignore[return-value]

    @property
    def kernel_state(self) -> KernelState:
        """Current kernel cache: W, W_pinv, rank, singular values."""
        if self._kernel_state is None or self._stale:
            self.set_reference(self.A0, k0=np.zeros(self.M))
        return self._kernel_state  # type: ignore[return-value]

    def _check_k(self, k: np.ndarray, name: str = "k") -> np.ndarray:
        arr = np.asarray(k, dtype=np.float64).ravel()
        if arr.shape != (self.M,):
            raise ValueError(f"{name} must have shape ({self.M},), got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN or Inf")
        return arr

    def _check_target(self, target: np.ndarray) -> np.ndarray:
        dtype = np.float64 if self.hermitian and not np.iscomplexobj(target) else np.complex128
        arr = np.asarray(target, dtype=dtype).ravel()
        active_n = len(self._lam0) if self._lam0 is not None else self.N
        if not np.all(np.isfinite(arr)):
            raise ValueError("target_lam contains NaN or Inf")
        if arr.size < active_n:
            raise ValueError(f"target_lam must contain at least {active_n} values, got {arr.size}")
        if arr.size > active_n:
            warnings.warn(
                f"target_lam has {arr.size} values; using the first {active_n}.",
                UserWarning,
                stacklevel=2,
            )
            arr = arr[:active_n]
        return arr

    # ──────────────────────────────────────────
    #  build(k)
    #  CONTRACT:
    #    PRE:  len(k) == self.M
    #    POST: returns A₀ + Σ kⱼ·Bⱼ  (N,N) symmetric
    #    PERF: O(M·N²) via tensordot (BLAS).  Cached on repeated k.
    # ──────────────────────────────────────────
    def build(self, k: np.ndarray) -> np.ndarray:
        k = self._check_k(k)
        k_tuple = tuple(np.round(k, 8))
        if (hasattr(self, '_last_build_key') and
            self._last_build_key is not None and
            self._last_build_key == k_tuple):
            return self._last_build_result
        if self.M == 0:
            result = self.A0.copy() if hasattr(self.A0, "copy") else self.A0
        else:
            result = self.A0 + self._basis_backend.build_delta(k)
        self._last_build_key = k_tuple
        self._last_build_result = result
        return result

    def build_sparse(self, k: np.ndarray):
        """Build A(k) as a sparse CSC matrix when possible."""
        k = self._check_k(k)
        if self._sparse_mode:
            base = self.A0_sparse
        else:
            base = sparse.csc_matrix(self.A0)
        if self.M == 0:
            return base.copy()
        if hasattr(self._basis_backend, "build_delta_sparse"):
            return base + self._basis_backend.build_delta_sparse(k)
        return base + sparse.csc_matrix(self._basis_backend.build_delta(k))

    # ──────────────────────────────────────────
    #  _eigh(A)
    #  CONTRACT:
    #    PRE:  A is (N,N) symmetric
    #    POST: returns (lam, vecs) sorted ascending
    #    PERF: uses sparse eigsh when input was sparse + N>k_eigs
    # ──────────────────────────────────────────
    def _eigh(self, A: np.ndarray, n_eigs: int | None = None,
              which: str = "SM") -> tuple[np.ndarray, np.ndarray]:
        self._eigh_count += 1
        sparse_input = _is_sparse(A)
        if sparse_input:
            N = A.shape[0]
        else:
            A = np.asarray(A)
            N = A.shape[0]
            if not np.all(np.isfinite(A)):
                raise ValueError("operator matrix contains NaN or Inf")
        k = n_eigs if n_eigs is not None else (self.k_eigs if sparse_input and N > self.k_eigs else None)
        if sparse_input and k is not None and 0 < k < N:
            try:
                lam, vecs = sparse.linalg.eigsh(A, k=k, which=which)
                idx = np.argsort(lam)
                return lam[idx], vecs[:, idx]
            except (sparse.linalg.ArpackError, Exception):
                warnings.warn("sparse eigsh failed; falling back to dense eigh", RuntimeWarning, stacklevel=2)
        if sparse_input:
            A = A.toarray()
        if k is not None and 0 < k < N:
            lam, vecs = linalg.eigh(A, subset_by_index=(0, k - 1))
            return lam, vecs
        return linalg.eigh(A)

    @staticmethod
    def _sort_complex(lam: np.ndarray, *vecs: np.ndarray):
        idx = np.lexsort((np.imag(lam), np.real(lam)))
        sorted_vecs = tuple(v[:, idx] for v in vecs)
        return (lam[idx], *sorted_vecs)

    def _eig_biorthogonal(self, A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._eigh_count += 1
        if _is_sparse(A):
            A = A.toarray()
        A = np.asarray(A, dtype=np.complex128)
        if not np.all(np.isfinite(A)):
            raise ValueError("operator matrix contains NaN or Inf")
        lam, VL, VR = linalg.eig(A, left=True, right=True)
        lam, VL, VR = self._sort_complex(lam, VL, VR)
        for i in range(VR.shape[1]):
            overlap = np.vdot(VL[:, i], VR[:, i])
            if abs(overlap) > 1e-14:
                VL[:, i] = VL[:, i] / np.conj(overlap)
        return lam, VL, VR

    def _biorthogonal_kernel(self, VL: np.ndarray, VR: np.ndarray) -> np.ndarray:
        if self.M == 0:
            return np.empty((VR.shape[1], 0), dtype=np.complex128)
        W = np.empty((VR.shape[1], self.M), dtype=np.complex128)
        for j, B in enumerate(self.basis):
            BR = B @ VR
            W[:, j] = np.einsum("ni,ni->i", np.conjugate(VL), BR)
        return W

    def eigensystem(self, k: np.ndarray | None = None, n_eigs: int | None = None,
                    which: str = "SM") -> tuple[np.ndarray, np.ndarray]:
        """Exact eigenvalues/eigenvectors for A(k), optionally partial."""
        k_vec = np.zeros(self.M) if k is None else self._check_k(k)
        A = self.build_sparse(k_vec) if (self._sparse_mode or n_eigs is not None) else self.build(k_vec)
        if not self.hermitian:
            lam, _, VR = self._eig_biorthogonal(A)
            if n_eigs is not None:
                return lam[:n_eigs], VR[:, :n_eigs]
            return lam, VR
        return self._eigh(A, n_eigs=n_eigs, which=which)

    # ──────────────────────────────────────────
    #  invalidate_cache()
    #  CONTRACT:
    #    POST: self._stale = True.  Next property access triggers recompute.
    #    WHY:  Use when modifying A0 or basis externally and want fresh data.
    # ──────────────────────────────────────────
    def invalidate_cache(self) -> 'OperatorFamily':
        self._stale = True
        return self

    # ──────────────────────────────────────────
    #  set_reference(A_ref, k0=None)
    #  CONTRACT:
    #    PRE:  A_ref is (N,N) symmetric
    #    POST: lam0, vecs ← eigh(A_ref)
    #    POST: W ← einsum, W⁺ ← SVD, rank ← σ > tol
    #    POST: _k0_ref ← k0 if provided
    #    PERF: batched einsum when M > BATCH_LIMIT/N²
    # ──────────────────────────────────────────
    def set_reference(self, A_ref: np.ndarray, k0: np.ndarray | None = None,
                      n_eigs: int | None = None, which: str = "SM") -> 'OperatorFamily':
        if self.hermitian:
            self._lam0, self._vecs = self._eigh(A_ref, n_eigs=n_eigs, which=which)
            self._left_vecs = None
        else:
            self._lam0, self._left_vecs, self._vecs = self._eig_biorthogonal(A_ref)
            if n_eigs is not None:
                self._lam0 = self._lam0[:n_eigs]
                self._left_vecs = self._left_vecs[:, :n_eigs]
                self._vecs = self._vecs[:, :n_eigs]
        self._last_build_key = None
        self._last_build_result = None
        V = self._vecs
        if self.M == 0:
            dtype = np.float64 if self.hermitian else np.complex128
            self._W = np.empty((len(self._lam0), 0), dtype=dtype)
        elif self.hermitian:
            BATCH_LIMIT = 5_000_000
            self._W = self._basis_backend.kernel(V, BATCH_LIMIT)
        else:
            self._W = self._biorthogonal_kernel(self._left_vecs, V)
        if k0 is not None:
            self._k0_ref = self._check_k(k0, "k0").copy()
        U, s, Vt = linalg.svd(self._W, full_matrices=False)
        tol = self.svd_tol * max(s[0], 1e-12) if len(s) > 0 else 1e-12
        self._W_rank = int(np.sum(s > tol))
        r = max(self._W_rank, 1)
        if len(s) >= r and s[r - 1] > tol:
            self._Wp = Vt[:r].conj().T @ np.diag(1.0 / s[:r]) @ U[:, :r].conj().T
        else:
            self._Wp = np.zeros((self.M, self._W.shape[0]), dtype=self._W.dtype)
        self._W_singular = s
        self._reference_state = ReferenceState(self._k0_ref.copy(), self._lam0.copy(), self._vecs.copy())
        self._biorthogonal_state = (
            BiorthogonalState(self._k0_ref.copy(), self._lam0.copy(),
                              self._left_vecs.copy(), self._vecs.copy())
            if not self.hermitian and self._left_vecs is not None
            else None
        )
        self._kernel_state = KernelState(self._W.copy(), self._Wp.copy(), self._W_rank, self._W_singular.copy())
        return self

    @property
    def W(self) -> np.ndarray:
        """SFT kernel: W(i,j) = ∂λ_i/∂k_j.  (N, M) matrix."""
        if self._W is None or self._stale:
            self.set_reference(self.A0)
        self._stale = False
        return self._W  # type: ignore[return-value]

    @property
    def W_pinv(self) -> np.ndarray:
        """Moore-Penrose pseudoinverse W⁺.  For inverse design: dk = W⁺·(target − λ)."""
        if self._Wp is None:
            self.set_reference(self.A0)
        return self._Wp  # type: ignore[return-value]

    @property
    def W_rank(self) -> int:
        """Structural rank: number of controllable spectral degrees of freedom."""
        if self._W_rank is None:
            self.set_reference(self.A0)
        return self._W_rank  # type: ignore[return-value]

    @property
    def W_singular(self) -> np.ndarray:
        """Singular value spectrum of W (sorted descending)."""
        if self._W_singular is None:
            self.set_reference(self.A0)
        return self._W_singular  # type: ignore[return-value]

    @property
    def lam0(self) -> np.ndarray:
        """Eigenvalues at reference operator (sorted ascending)."""
        if self._lam0 is None:
            self.set_reference(self.A0)
        return self._lam0  # type: ignore[return-value]

    @property
    def vecs(self) -> np.ndarray:
        """Eigenvectors at reference operator (columns of V)."""
        if self._vecs is None:
            self.set_reference(self.A0)
        return self._vecs  # type: ignore[return-value]

    @property
    def left_vecs(self) -> np.ndarray:
        """Left eigenvectors for non-Hermitian families; equals vecs for Hermitian ones."""
        if self.hermitian:
            return self.vecs
        if self._left_vecs is None:
            self.set_reference(self.A0)
        return self._left_vecs  # type: ignore[return-value]

    @property
    def biorthogonal_state(self) -> BiorthogonalState:
        """Current non-Hermitian reference with normalized left/right eigenvectors."""
        if self.hermitian:
            raise AttributeError("biorthogonal_state is only available for hermitian=False families")
        if self._biorthogonal_state is None or self._stale:
            self.set_reference(self.A0, k0=np.zeros(self.M))
        return self._biorthogonal_state  # type: ignore[return-value]

    # ──────────────────────────────────────────
    #  spectrum(k)
    #  CONTRACT:
    #    PRE:  len(k) == M
    #    POST: returns exact λ(k) via eigh (sorted)
    #    O(N³) — expensive, use .predict() when approximation is OK
    # ──────────────────────────────────────────
    def spectrum(self, k: np.ndarray, n_eigs: int | None = None,
                 which: str = "SM") -> np.ndarray:
        k = self._check_k(k)
        if not self.hermitian:
            return self.eigensystem(k, n_eigs=n_eigs, which=which)[0]
        if n_eigs is not None:
            return self.eigensystem(k, n_eigs=n_eigs, which=which)[0]
        return np.sort(self._eigh(self.build(k))[0])

    # ──────────────────────────────────────────
    #  predict(dk)
    #  CONTRACT:
    #    PRE:  len(dk) == M
    #    POST: returns λ₀ + W·dk  (1st-order approximation)
    #    ERROR: O(||dk||²) for smooth families
    #    O(N·M) — cheap
    # ──────────────────────────────────────────
    def predict(self, dk: np.ndarray) -> np.ndarray:
        dk = self._check_k(dk, "dk")
        return self.lam0 + self.W @ dk

    def predict_many(self, DK: np.ndarray) -> np.ndarray:
        """Vectorized first-order predictions for many parameter deltas."""
        arr = np.asarray(DK, dtype=np.float64)
        if arr.ndim == 1:
            return self.predict(arr)
        if arr.ndim != 2 or arr.shape[1] != self.M:
            raise ValueError(f"DK must have shape (n, {self.M}), got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("DK contains NaN or Inf")
        return self.lam0[None, :] + arr @ self.W.T

    def _linear_inverse_step(self, residual: np.ndarray) -> np.ndarray:
        if np.iscomplexobj(self.W) or np.iscomplexobj(residual):
            A = np.vstack([self.W.real, self.W.imag])
            b = np.concatenate([residual.real, residual.imag])
            return linalg.lstsq(A, b)[0]
        return self.W_pinv @ residual

    # ──────────────────────────────────────────
    #  predict_at(k)
    #  CONTRACT:
    #    PRE:  len(k) == M
    #    POST: λ(k) ≈ λ(k₀) + W·(k−k₀) from CURRENT reference k₀
    #    NOTE: differs from predict() which always uses k₀=0.
    # ──────────────────────────────────────────
    def predict_at(self, k: np.ndarray) -> np.ndarray:
        k = self._check_k(k)
        return self.lam0 + self.W @ (k - self._k0_ref)

    def refresh(self, k: np.ndarray) -> 'OperatorFamily':
        """Move the spectral reference to A(k) and return self for chaining."""
        k = self._check_k(k)
        return self.set_reference(self.build_sparse(k) if self._sparse_mode else self.build(k), k0=k)

    def at(self, k: np.ndarray) -> np.ndarray:
        """Fluent alias for predict_at(k)."""
        return self.predict_at(k)

    def shift(self, dk: np.ndarray) -> np.ndarray:
        """Fluent alias for predict(dk)."""
        return self.predict(dk)

    def toward(self, target_lam: np.ndarray, **kw) -> InverseResult:
        """Fluent alias for inverse(target_lam, **kw)."""
        return self.inverse(target_lam, **kw)

    def flow(self, target_lam: np.ndarray) -> '_SpectralFlowBuilder':
        """Create a fluent inverse/control builder."""
        return _SpectralFlowBuilder(self, target_lam)

    def solve(self, target_lam: np.ndarray, **kw) -> InverseResult:
        """Fluent alias for inverse(target_lam, **kw)."""
        return self.inverse(target_lam, **kw)

    def cost(self, operation: str | None = None):
        """Estimate memory/eigensolve cost without executing a new operation."""
        from .operator_algebra import OperatorCost
        return OperatorCost.estimate(self, operation=operation)

    def laws(self):
        """Return core algebra laws attached to SFT families."""
        from .operator_algebra import LawSet
        return LawSet(self)

    def verify(self) -> dict:
        """Run independent core verification gates."""
        return self.laws().verify().to_dict()

    def oplus(self, other: 'OperatorFamily') -> 'OperatorFamily':
        """Fluent direct sum alias."""
        from . import algebra
        return algebra.direct_sum(self, other)

    def then(self, transform):
        """Apply a deferred algebra transform."""
        if hasattr(transform, "apply"):
            return transform.apply(self)
        return self @ transform

    def __add__(self, other):
        if not isinstance(other, OperatorFamily):
            return NotImplemented
        from . import algebra
        return algebra.direct_sum(self, other)

    def __matmul__(self, other):
        if hasattr(other, "apply"):
            return other.apply(self)
        arr = np.asarray(other, dtype=np.float64)
        if arr.ndim == 1:
            return self.predict(arr)
        if arr.ndim == 2:
            from . import algebra
            return algebra.compose_linear(self, arr)
        return NotImplemented

    # ──────────────────────────────────────────
    #  inverse(target_lam, steps, alpha, refresh_every)
    #  CONTRACT:
    #    PRE:  len(target_lam) == N (or ≥ N)
    #    POST: returns (k, max_error, success) — k ≈ argmin ||λ(k)−target||
    #    ALGO: damped Newton with periodic W refresh + adaptive stagnation
    #    PERF: ~steps eigh calls.  Adaptive reduces this by up to 2×.
    #  EXAMPLE:
    #    >>> k, err, ok = fam.inverse(target, steps=30, alpha=0.3)
    #    >>> print(f"Found k={k} with error={err:.4f}")
    # ──────────────────────────────────────────
    def inverse(self, target_lam: np.ndarray,
                steps: int = 20, alpha: float = 0.3,
                refresh_every: int = 5,
                n_eigs: int | None = None,
                which: str = "SM",
                method: str = "linear") -> InverseResult:
        if method not in {"linear", "hessian", "trust", "homotopy"}:
            raise ValueError("method must be one of: linear, hessian, trust, homotopy")
        if method == "homotopy":
            from .homotopy import track_homotopy
            result = track_homotopy(target_lam, n_steps=steps, family=self)
            result.method = "homotopy"
            return result
        t0 = time.perf_counter()
        eigh_start = self._eigh_count
        k = np.zeros(self.M)
        def _operator_for_solve(kvec: np.ndarray):
            return self.build_sparse(kvec) if (self._sparse_mode or n_eigs is not None) else self.build(kvec)

        self.set_reference(_operator_for_solve(k), k0=k, n_eigs=n_eigs, which=which)
        n_refresh = 1
        target_lam = self._check_target(target_lam)
        best_err = np.inf
        prev_err = np.inf
        stagnation_count = 0
        trajectory = [k.copy()]
        residual_history = []
        hessian_count = 0
        for step in range(steps):
            lam = self.spectrum(k, n_eigs=n_eigs, which=which)
            residual = target_lam - lam
            err = float(np.max(np.abs(residual)))
            residual_history.append(err)
            best_err = min(best_err, err)
            if err < self.convergence_tol:
                return InverseResult(k, err, True, steps=step + 1,
                                     n_refresh=n_refresh,
                                     condition_number=self.condition_number(),
                                     eigh_count=self._eigh_count - eigh_start,
                                     time_ms=(time.perf_counter() - t0) * 1000,
                                     method=method,
                                     trajectory=np.array(trajectory),
                                     residual_history=np.array(residual_history),
                                     hessian_count=hessian_count)
            if step > 0 and (step % refresh_every == 0 or
                              (err > prev_err * 0.9 and stagnation_count >= 2)):
                self.set_reference(_operator_for_solve(k), k0=k, n_eigs=n_eigs, which=which)
                n_refresh += 1
                stagnation_count = 0
            if err >= prev_err * 0.95:
                stagnation_count += 1
            else:
                stagnation_count = 0
            prev_err = err
            dk0 = self._linear_inverse_step(residual)
            if method in {"hessian", "trust"} and self.hermitian and self.M > 0:
                from .hessian import hessian_analytic
                H = hessian_analytic(self, k)
                H = H[:len(residual)]
                hessian_count += 1
                if method == "trust":
                    radius = max(0.25, 2.0 / np.sqrt(max(self.M, 1)))
                    norm = np.linalg.norm(dk0)
                    if norm > radius:
                        dk0 = dk0 * (radius / norm)
                best_err_step = np.inf
                dk = alpha * dk0
                for scale in (alpha, alpha * 0.5, alpha * 0.25, -alpha * 0.25):
                    cand = scale * dk0
                    quad = self.W @ cand + 0.5 * np.einsum("ijk,j,k->i", H, cand, cand)
                    predicted_err = float(np.max(np.abs(residual - quad)))
                    exact_err = float(np.max(np.abs(target_lam - self.spectrum(k + cand, n_eigs=n_eigs, which=which))))
                    score = min(exact_err, predicted_err)
                    if score < best_err_step:
                        best_err_step = score
                        dk = cand
            else:
                dk = alpha * dk0
            k += dk
            trajectory.append(k.copy())
        lam = self.spectrum(k, n_eigs=n_eigs, which=which)
        final = float(np.max(np.abs(lam - target_lam)))
        return InverseResult(k, final, best_err < self.convergence_tol,
                             steps=steps, n_refresh=n_refresh,
                             condition_number=self.condition_number(),
                             eigh_count=self._eigh_count - eigh_start,
                             time_ms=(time.perf_counter() - t0) * 1000,
                             method=method,
                             trajectory=np.array(trajectory),
                             residual_history=np.array(residual_history),
                             hessian_count=hessian_count)

    # ──────────────────────────────────────────
    #  condition_number()
    #  CONTRACT:
    #    POST: κ(W) = σ_max / σ_min.  ∞ if rank=0 or singular.
    #    INTERPRET: κ<10 → easy inverse; κ>1000 → ill-conditioned, needs regularisation.
    # ──────────────────────────────────────────
    def condition_number(self) -> float:
        s = self.W_singular
        if len(s) == 0 or s[-1] < 1e-15:
            return np.inf
        return float(s[0] / s[-1])

    @property
    def complexity(self) -> float:
        """rank(W) / N — structural complexity ratio.  0=ORDER regime, ~0.5=GRAPH."""
        return self.W_rank / self.N if self.N > 0 else 0.0

    # ──────────────────────────────────────────
    #  isospectral_dimension()
    #  CONTRACT:
    #    POST: dim(ker(W)) = M − rank(W).  Directions in k-space that do NOT change λ.
    #    INTERPRET: ghost parameters — the operator family is over-parameterised.
    # ──────────────────────────────────────────
    def isospectral_dimension(self) -> int:
        return max(self.M - self.W_rank, 0)

    def isospectral_flow(self, direction: np.ndarray | None = None,
                         steps: int = 16, step_size: float = 0.1) -> dict[str, np.ndarray | float]:
        """Walk along an approximate ker(W) direction and monitor spectral drift."""
        Z = nullspace(self)
        if Z.shape[1] == 0:
            path = np.zeros((steps, self.M))
        else:
            if direction is None:
                d = Z[:, 0]
            else:
                coeff = np.asarray(direction, dtype=np.float64).ravel()
                if coeff.shape == (self.M,):
                    d = Z @ (Z.T @ coeff)
                elif coeff.shape == (Z.shape[1],):
                    d = Z @ coeff
                else:
                    raise ValueError(f"direction must have shape ({self.M},) or ({Z.shape[1]},)")
            d = d / max(np.linalg.norm(d), 1e-15)
            alphas = np.linspace(-step_size, step_size, steps)
            path = alphas[:, None] * d[None, :]
        spectra = np.array([self.spectrum(k) for k in path])
        drift = float(np.max(np.abs(spectra - spectra[0]))) if spectra.size else 0.0
        return {"path": path, "spectra": spectra, "drift": drift, "nullity": float(Z.shape[1])}


class _SpectralFlowBuilder:
    def __init__(self, family: OperatorFamily, target_lam: np.ndarray):
        self.family = family
        self.target_lam = target_lam
        self._method = "linear"
        self._kw: dict[str, object] = {}

    def via(self, method: str) -> '_SpectralFlowBuilder':
        self._method = method
        return self

    def second_order(self) -> '_SpectralFlowBuilder':
        if self._method == "linear":
            self._method = "hessian"
        return self

    def trust(self) -> '_SpectralFlowBuilder':
        self._method = "trust"
        return self

    def options(self, **kw) -> '_SpectralFlowBuilder':
        self._kw.update(kw)
        return self

    def solve(self, **kw) -> InverseResult:
        opts = {**self._kw, **kw}
        return self.family.inverse(self.target_lam, method=self._method, **opts)


# ────────────────────────────────────────────────────
#  nullspace(family)
#  CONTRACT:
#    POST: basis of ker(W) via scipy null_space.
#    POST: if rank >= M → zero-shaped array.
# ────────────────────────────────────────────────────
def nullspace(family: OperatorFamily) -> np.ndarray:
    W = family.W
    if family.W_rank >= family.M:
        return np.zeros((family.M, 0))
    return linalg.null_space(W)


# ────────────────────────────────────────────────────
#  graph_response_kernel(adjacency)
#  CONTRACT:
#    PRE:  adjacency is (N,N) binary/weighted.
#    POST: returns (lam, V, W) where W(i,e) = (v_i(u) − v_i(v))².
#    WHY:  Edge-RESPONSE kernel — fundamentally different from edge-WEIGHT.
#          ∂λᵢ/∂wₑ = (v_i(u) − v_i(v))²  for weight changes on edge e=(u,v).
#    EXAMPLE:
#      >>> lam, V, W = graph_response_kernel(adj)
#      >>> print(W.shape)  # (N, num_edges)
# ────────────────────────────────────────────────────
def graph_response_kernel(adjacency: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = adjacency.shape[0]
    D = np.diag(np.sum(adjacency, axis=1))
    L0 = D - adjacency
    lam, V = linalg.eigh(L0)
    row, col = np.triu(adjacency != 0, 1).nonzero()
    edges = list(zip(row.tolist(), col.tolist()))
    M = len(edges)
    W = np.zeros((N, M))
    for k, (u, v) in enumerate(edges):
        W[:, k] = (V[u, :] - V[v, :]) ** 2
    return lam, V, W


def kernel(family: OperatorFamily) -> np.ndarray:
    return family.W

def predict(family: OperatorFamily, dk: np.ndarray) -> np.ndarray:
    return family.predict(dk)

def inverse(family: OperatorFamily, target: np.ndarray, **kw):
    return family.inverse(target, **kw)

def rank(family: OperatorFamily) -> int:
    return family.W_rank

def svd_spectrum(family: OperatorFamily) -> np.ndarray:
    return family.W_singular
