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
from typing import Optional, Tuple
import numpy as np
from scipy import linalg, sparse

_is_sparse = lambda x: sparse.issparse(x) if hasattr(sparse, 'issparse') else False


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
                 k_eigs: int | None = None):
        """
        CONTRACT:
            PRE:  A0.shape == (N,N), all B in basis have shape (N,N)
            PRE:  A0 and all B_j are symmetric (not enforced, but assumed)
            POST: self.W is computed via eigh+SVD (cached)
            POST: self._k0_ref = zeros(M)  — reference parameter vector
        """
        self._sparse_mode = _is_sparse(A0)
        self.N = A0.shape[0]
        self.M = len(basis)
        self.svd_tol = svd_tol
        self.convergence_tol = convergence_tol
        self.k_eigs = k_eigs or min(self.N, 128)
        self._stale = False
        self._k0_ref = np.zeros(self.M)

        if self._sparse_mode:
            self.A0_sparse = A0.tocsc()
            self.A0 = self.A0_sparse.toarray()
            self._basis_sparse = [B.tocsc() for B in basis] if self.M > 0 else []
            self._basis_list = [B.toarray() for B in basis] if self.M > 0 else []
            self._basis_stack = np.stack(self._basis_list) if self.M > 0 else np.empty((0, self.N, self.N))
        else:
            self.A0 = A0.copy()
            self._basis_list = [B.copy() for B in basis]
            self._basis_stack = np.stack(self._basis_list) if self.M > 0 else np.empty((0, self.N, self.N))
            self.A0_sparse = None
            self._basis_sparse = []

        self._lam0: Optional[np.ndarray] = None
        self._vecs: Optional[np.ndarray] = None
        self._W: Optional[np.ndarray] = None
        self._Wp: Optional[np.ndarray] = None
        self._W_rank: Optional[int] = None
        self._W_singular: Optional[np.ndarray] = None
        self.set_reference(self.A0, k0=np.zeros(self.M))

    @property
    def basis(self) -> list[np.ndarray]:
        """CONTRACT: returns the basis matrices B_j as a list. Immutable copy."""
        return self._basis_list

    # ──────────────────────────────────────────
    #  build(k)
    #  CONTRACT:
    #    PRE:  len(k) == self.M
    #    POST: returns A₀ + Σ kⱼ·Bⱼ  (N,N) symmetric
    #    PERF: O(M·N²) via tensordot (BLAS).  Cached on repeated k.
    # ──────────────────────────────────────────
    def build(self, k: np.ndarray) -> np.ndarray:
        k_tuple = tuple(np.round(k, 8))
        if (hasattr(self, '_last_build_key') and
            self._last_build_key is not None and
            self._last_build_key == k_tuple):
            return self._last_build_result
        if self.M == 0:
            result = self.A0.copy()
        else:
            result = self.A0 + np.tensordot(k, self._basis_stack, axes=((0,), (0,)))
        self._last_build_key = k_tuple
        self._last_build_result = result
        return result

    # ──────────────────────────────────────────
    #  _eigh(A)
    #  CONTRACT:
    #    PRE:  A is (N,N) symmetric
    #    POST: returns (lam, vecs) sorted ascending
    #    PERF: uses sparse eigsh when input was sparse + N>k_eigs
    # ──────────────────────────────────────────
    def _eigh(self, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._sparse_mode and self.N > self.k_eigs:
            try:
                return sparse.linalg.eigsh(A, k=self.k_eigs, which='SM')
            except (sparse.linalg.ArpackError, Exception):
                pass
        return linalg.eigh(A)

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
    def set_reference(self, A_ref: np.ndarray, k0: np.ndarray | None = None) -> 'OperatorFamily':
        self._lam0, self._vecs = self._eigh(A_ref)
        self._last_build_key = None
        self._last_build_result = None
        V = self._vecs
        if self.M == 0:
            self._W = np.empty((self.N, 0))
        else:
            BATCH_LIMIT = 5_000_000
            elements_per_basis = self.N * self.N
            batch_M = max(1, BATCH_LIMIT // max(elements_per_basis, 1))
            if self.M <= batch_M:
                BsV = self._basis_stack @ V
                self._W = np.einsum('ni,mni->im', V, BsV)
            else:
                self._W = np.empty((self.N, self.M))
                for b in range(0, self.M, batch_M):
                    be = min(b + batch_M, self.M)
                    BsV_b = self._basis_stack[b:be] @ V
                    self._W[:, b:be] = np.einsum('ni,mni->im', V, BsV_b)
        if k0 is not None:
            self._k0_ref = np.asarray(k0).copy()
        U, s, Vt = linalg.svd(self._W, full_matrices=False)
        tol = self.svd_tol * max(s[0], 1e-12) if len(s) > 0 else 1e-12
        self._W_rank = int(np.sum(s > tol))
        r = max(self._W_rank, 1)
        if len(s) >= r and s[r - 1] > tol:
            self._Wp = Vt[:r].T @ np.diag(1.0 / s[:r]) @ U[:, :r].T
        else:
            self._Wp = np.zeros((self.M, self.N))
        self._W_singular = s
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

    # ──────────────────────────────────────────
    #  spectrum(k)
    #  CONTRACT:
    #    PRE:  len(k) == M
    #    POST: returns exact λ(k) via eigh (sorted)
    #    O(N³) — expensive, use .predict() when approximation is OK
    # ──────────────────────────────────────────
    def spectrum(self, k: np.ndarray) -> np.ndarray:
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
        return self.lam0 + self.W @ dk

    # ──────────────────────────────────────────
    #  predict_at(k)
    #  CONTRACT:
    #    PRE:  len(k) == M
    #    POST: λ(k) ≈ λ(k₀) + W·(k−k₀) from CURRENT reference k₀
    #    NOTE: differs from predict() which always uses k₀=0.
    # ──────────────────────────────────────────
    def predict_at(self, k: np.ndarray) -> np.ndarray:
        return self.lam0 + self.W @ (k - self._k0_ref)

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
                refresh_every: int = 5) -> Tuple[np.ndarray, float, bool]:
        k = np.zeros(self.M)
        self.set_reference(self.build(k))
        best_err = np.inf
        prev_err = np.inf
        stagnation_count = 0
        for step in range(steps):
            lam = self.spectrum(k)
            err = float(np.max(np.abs(lam - target_lam)))
            best_err = min(best_err, err)
            if err < self.convergence_tol:
                return k, err, True
            if step > 0 and (step % refresh_every == 0 or
                              (err > prev_err * 0.9 and stagnation_count >= 2)):
                self.set_reference(self.build(k))
                stagnation_count = 0
            if err >= prev_err * 0.95:
                stagnation_count += 1
            else:
                stagnation_count = 0
            prev_err = err
            dk = alpha * (self.W_pinv @ (target_lam - lam))
            k += dk
        lam = self.spectrum(k)
        final = float(np.max(np.abs(lam - target_lam)))
        return k, final, best_err < self.convergence_tol

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
