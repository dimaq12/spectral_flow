"""
sft.algebra — Operator calculus: ⊕, ∘, ⊗, ∫ expectation.

╔══════════════════════════════════════════════════════════════════════╗
║  INTENT                                                             ║
║  Algebraic laws for composing OperatorFamily instances.             ║
║  Direct sum (A⊕B), composition (A∘C), tensor sum (A⊗B), and        ║
║  expectation under Gaussian parameter distribution.                 ║
║  Each law preserves the OperatorFamily interface — the result is    ║
║  itself an OperatorFamily with well-defined W, spectrum, etc.       ║
╚══════════════════════════════════════════════════════════════════════╝
║  CONCEPT                                                            ║
║  • direct_sum(A,B):  block-diag(A,B).  λ(A⊕B) = λ(A) ∪ λ(B).      ║
║  • compose_linear(A,C):  (A∘C)(k) = A(C·k).  W_new = W_A·C.       ║
║  • tensor_sum(A,B):  A⊗I + I⊗B.  λ(A⊗B) = λ(A) + λ(B).           ║
║  • expectation(fam,μ,σ):  Monte Carlo ∫λ(k)dμ.                     ║
╚══════════════════════════════════════════════════════════════════════╝
║  DEPENDENCIES                                                       ║
║  ┌── sft.core.OperatorFamily                                       ║
║  ├── numpy (np.kron, tensordot, zeros)                              ║
║  └── scipy.linalg (unused directly — kept for symmetry)             ║
╚══════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import warnings
import numpy as np
from scipy import linalg  # noqa: F401
from .core import OperatorFamily


@dataclass(frozen=True)
class AlgebraTransform:
    """Deferred algebra operation used by fluent APIs."""

    kind: str
    payload: np.ndarray

    def apply(self, family: OperatorFamily) -> OperatorFamily:
        if self.kind in {"compose", "pullback"}:
            return compose_linear(family, self.payload)
        raise ValueError(f"unknown algebra transform: {self.kind}")


@dataclass(frozen=True)
class JordanFingerprint:
    """Finite-dimensional Jordan-chain certificate from ranks of powers."""

    order: int
    rank_sequence: tuple[int, ...]
    nullity_sequence: tuple[int, ...]
    nilpotent_index: int | None
    geometric_multiplicity: int
    norms: tuple[float, ...]
    tol: float

    @property
    def is_nilpotent(self) -> bool:
        return self.nilpotent_index is not None

    @property
    def is_single_chain(self) -> bool:
        return (
            self.geometric_multiplicity == 1
            and self.nilpotent_index == self.order
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "rank_sequence": list(self.rank_sequence),
            "nullity_sequence": list(self.nullity_sequence),
            "nilpotent_index": self.nilpotent_index,
            "geometric_multiplicity": self.geometric_multiplicity,
            "norms": list(self.norms),
            "tol": self.tol,
            "is_nilpotent": self.is_nilpotent,
            "is_single_chain": self.is_single_chain,
        }


def _attach_algebra_metadata(family: OperatorFamily, operation: str,
                             before: tuple[Any, ...]) -> OperatorFamily:
    from .operator_algebra import OperatorCost

    family.algebra_operation = operation
    family.cost_after = OperatorCost.estimate(family, operation=operation)
    family.cost_before = tuple(OperatorCost.estimate(obj, operation=operation) for obj in before)
    return family


# ────────────────────────────────────────────────────
#  direct_sum(A, B)  →  OperatorFamily
#  CONTRACT:
#    PRE:  A.N, B.N > 0
#    POST: new family with N = A.N + B.N, M = A.M + B.M
#    POST: W is block-diagonal: W[:A.N, :A.M] = A.W, W[A.N:, A.M:] = B.W
#    WHY:  Combine two independent operator families into one.
#  EXAMPLE:
#    >>> fam = direct_sum(fam_a, fam_b)
#    >>> lam = fam.lam0  # sorted union of λ(A) ∪ λ(B)
# ────────────────────────────────────────────────────
def direct_sum(fam_a: OperatorFamily, fam_b: OperatorFamily) -> OperatorFamily:
    na, nb = fam_a.N, fam_b.N
    ma, mb = fam_a.M, fam_b.M
    nd = na + nb
    A0_ab = np.zeros((nd, nd))
    A0_ab[:na, :na] = fam_a.A0
    A0_ab[na:, na:] = fam_b.A0
    basis_ab = []
    for j in range(ma + mb):
        B = np.zeros((nd, nd))
        if j < ma:
            B[:na, :na] = fam_a.basis[j]
        else:
            B[na:, na:] = fam_b.basis[j - ma]
        basis_ab.append(B)
    return _attach_algebra_metadata(OperatorFamily(A0_ab, basis_ab), "direct_sum", (fam_a, fam_b))


# ────────────────────────────────────────────────────
#  compose_linear(outer, C)  →  OperatorFamily
#  CONTRACT:
#    PRE:  C.shape == (outer.M, m_inner)
#    POST: new family A(C·k) with M = m_inner
#    POST: W_new = W_outer @ C  (chain rule!)
#    WHY:  Pre-compose with linear map — reduce parameter space dimension.
#  EXAMPLE:
#    >>> C = np.eye(4, 2)  # select first 2 params
#    >>> fam2 = compose_linear(fam, C)  # M: 4→2
# ────────────────────────────────────────────────────
def compose_linear(outer: OperatorFamily, C: np.ndarray) -> OperatorFamily:
    C = np.asarray(C, dtype=np.float64)
    if C.ndim != 2 or C.shape[0] != outer.M:
        raise ValueError(f"C must have shape ({outer.M}, m), got {C.shape}")
    if outer.basis_kind != "dense" and getattr(outer._basis_backend, "materialized_elements", 0) > 10_000_000:
        warnings.warn(
            "compose_linear is materializing a structured basis; use a smaller projection or dense family.",
            ResourceWarning,
            stacklevel=2,
        )
    outer_stack = np.stack(outer.basis) if outer.M > 0 else np.empty((0, outer.N, outer.N))
    basis_new = np.tensordot(C.T, outer_stack, axes=((1,), (0,)))
    return _attach_algebra_metadata(OperatorFamily(outer.A0.copy(), list(basis_new)), "compose", (outer,))
# ────────────────────────────────────────────────────
#  tensor_sum(A, B)  →  OperatorFamily
#  CONTRACT:
#    PRE:  A.N, B.N > 0
#    POST: A⊗I + I⊗B with N = A.N·B.N, M = A.M + B.M
#    POST: λ(A⊗B) = λ(A) + λ(B)  (pairwise sum)
#    WHY:  Combine two families into product space — eigenvalues add.
#  EXAMPLE:
#    >>> fam = tensor_sum(a, b)
#    >>> lam_all = np.sort(np.add.outer(a.lam0, b.lam0).ravel())
#    >>> np.allclose(np.sort(fam.lam0), lam_all)
# ────────────────────────────────────────────────────
def tensor_sum(fam_a: OperatorFamily, fam_b: OperatorFamily) -> OperatorFamily:
    na, nb = fam_a.N, fam_b.N
    ma, mb = fam_a.M, fam_b.M
    A0_kron = np.kron(fam_a.A0, np.eye(nb)) + np.kron(np.eye(na), fam_b.A0)
    basis_kron = []
    for j in range(ma):
        basis_kron.append(np.kron(fam_a.basis[j], np.eye(nb)))
    for j in range(mb):
        basis_kron.append(np.kron(np.eye(na), fam_b.basis[j]))
    return _attach_algebra_metadata(OperatorFamily(A0_kron, basis_kron), "tensor", (fam_a, fam_b))


def jordan_fingerprint(A: np.ndarray, tol: float = 1e-9) -> JordanFingerprint:
    """Return a compact Jordan-chain fingerprint for a nilpotent candidate."""
    arr = np.asarray(A)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"A must be a square matrix, got shape {arr.shape}")
    n = int(arr.shape[0])
    ranks: list[int] = []
    nullities: list[int] = []
    norms: list[float] = []
    nilpotent_index: int | None = None
    for power in range(1, n + 1):
        Ap = np.linalg.matrix_power(arr, power)
        rank = int(np.linalg.matrix_rank(Ap, tol=tol))
        norm = float(np.linalg.norm(Ap))
        ranks.append(rank)
        nullities.append(n - rank)
        norms.append(norm)
        if nilpotent_index is None and norm <= tol:
            nilpotent_index = power
    return JordanFingerprint(
        order=n,
        rank_sequence=tuple(ranks),
        nullity_sequence=tuple(nullities),
        nilpotent_index=nilpotent_index,
        geometric_multiplicity=nullities[0] if nullities else 0,
        norms=tuple(norms),
        tol=float(tol),
    )


def is_single_jordan_chain(A: np.ndarray, order: int | None = None,
                           tol: float = 1e-9) -> bool:
    """True when ``A`` has one nilpotent Jordan chain of requested order."""
    fp = jordan_fingerprint(A, tol=tol)
    expected = fp.order if order is None else int(order)
    return fp.order == expected and fp.is_single_chain


def _coupling_matrix(n: int, coupling: tuple[int, int] | np.ndarray,
                     strength: complex) -> np.ndarray:
    if isinstance(coupling, tuple):
        if len(coupling) != 2:
            raise ValueError("coupling tuple must be (row, col)")
        row, col = map(int, coupling)
        if not (0 <= row < n and 0 <= col < n):
            raise ValueError(f"coupling indices must be in [0, {n}), got {coupling}")
        C = np.zeros((n, n), dtype=np.result_type(strength, complex))
        C[row, col] = strength
        return C
    C = np.asarray(coupling)
    if C.shape != (n, n):
        raise ValueError(f"coupling matrix must have shape ({n}, {n}), got {C.shape}")
    return np.asarray(C * strength, dtype=np.result_type(C, strength, complex))


def _find_fusing_coupling(A0: np.ndarray, strength: complex,
                         tol: float) -> tuple[int, int]:
    n = A0.shape[0]
    for row in range(n):
        for col in range(n):
            if row == col:
                continue
            candidate = A0 + _coupling_matrix(n, (row, col), strength)
            if is_single_jordan_chain(candidate, order=n, tol=tol):
                return (row, col)
    raise ValueError("no rank-one matrix-unit coupling fuses this tensor sum")


def _closure_basis(n: int) -> list[np.ndarray]:
    closure = np.zeros((n, n), dtype=np.complex128)
    closure[n - 1, 0] = 1.0
    return [closure, 1j * closure]


def _best_bridge(A0: np.ndarray, strength: complex,
                 tol: float) -> tuple[int, int]:
    n = A0.shape[0]
    best: tuple[tuple[int, int, int, int], tuple[int, int]] | None = None
    for row in range(n):
        for col in range(n):
            if row == col:
                continue
            candidate = A0 + _coupling_matrix(n, (row, col), strength)
            fp = jordan_fingerprint(candidate, tol=tol)
            if not fp.is_nilpotent:
                continue
            score = (
                int(fp.is_single_chain),
                fp.nilpotent_index or 0,
                -fp.geometric_multiplicity,
                fp.rank_sequence[0] if fp.rank_sequence else 0,
            )
            if best is None or score > best[0]:
                best = (score, (row, col))
    if best is None:
        raise ValueError("no nilpotent bridge candidate found")
    return best[1]


def _deterministic_fuse_couplings(fam_a: OperatorFamily, fam_b: OperatorFamily) -> list[tuple[int, int]] | None:
    """Fast bridge formulas for canonical single-chain tensor products."""
    fp_a = jordan_fingerprint(fam_a.A0)
    fp_b = jordan_fingerprint(fam_b.A0)
    if not (fp_a.is_single_chain and fp_b.is_single_chain):
        return None
    ma, mb = fam_a.N, fam_b.N
    if ma == mb:
        return [(row * mb - 1, row * mb) for row in range(1, ma)]
    if mb == 2:
        return [(2 * ma - 2, 1)]
    return None


def jordan_fuse(fam_a: OperatorFamily, fam_b: OperatorFamily,
                coupling: tuple[int, int] | np.ndarray | str | None = None,
                strength: complex = 1.0, search: bool = False,
                add_closure: bool = True, tol: float = 1e-9) -> OperatorFamily:
    """Tensor sum plus a chain-fusing coupling.

    ``EP2 ⊗ EP2`` is reducible under a plain tensor sum.  Adding one
    off-chain coupling, by default ``(1, 2)`` for 2x2 factors, fuses the
    tensor-product Jordan chains into a single EP4 block.  By default a
    chain-closing complex perturbation is appended to the basis so the
    resulting family exposes the expected 1/order Puiseux scaling.
    """
    if fam_a.N <= 0 or fam_b.N <= 0:
        raise ValueError("families must be non-empty")
    strength = complex(strength)
    if not np.isfinite(strength.real) or not np.isfinite(strength.imag):
        raise ValueError("strength must be finite")

    fused = tensor_sum(fam_a, fam_b)
    n = fused.N
    if coupling is None:
        if fam_a.N == 2 and fam_b.N == 2:
            coupling = (1, 2)
        elif search:
            coupling = _find_fusing_coupling(fused.A0, strength, tol)
        else:
            raise ValueError("coupling is required unless fusing 2x2 families or search=True")
    elif coupling == "search":
        coupling = _find_fusing_coupling(fused.A0, strength, tol)

    C = _coupling_matrix(n, coupling, strength)
    A0 = fused.A0 + C
    basis = list(fused.basis)
    if add_closure:
        basis.extend(_closure_basis(n))
    result = OperatorFamily(A0, basis, hermitian=False)
    result.jordan_coupling = coupling
    result.jordan_couplings = [coupling] if isinstance(coupling, tuple) else ["matrix"]
    result.jordan_fingerprint = jordan_fingerprint(A0, tol=tol)
    return _attach_algebra_metadata(result, "jordan_fuse", (fam_a, fam_b))


def multi_jordan_fuse(fam_a: OperatorFamily, fam_b: OperatorFamily,
                      couplings: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
                      strength: complex = 1.0, max_bridges: int | None = None,
                      add_closure: bool = True, tol: float = 1e-9) -> OperatorFamily:
    """Fuse all reducible Jordan blocks in a tensor sum using multiple bridges.

    A single ``jordan_fuse`` bridge is enough for EP2xEP2 and EP4xEP2.  For
    EP4xEP4 the tensor sum decomposes into four Jordan chains, so three bridges
    are required.  This routine greedily inserts nilpotent matrix-unit bridges
    until the tensor sum becomes one Jordan chain or the bridge budget is spent.
    """
    strength = complex(strength)
    if not np.isfinite(strength.real) or not np.isfinite(strength.imag):
        raise ValueError("strength must be finite")

    fused = tensor_sum(fam_a, fam_b)
    A0 = fused.A0.copy()
    n = fused.N
    bridge_budget = max_bridges
    if bridge_budget is None:
        bridge_budget = max(1, min(fam_a.N, fam_b.N) - 1)

    used: list[tuple[int, int]] = []
    if couplings is not None:
        for coupling in couplings:
            A0 = A0 + _coupling_matrix(n, coupling, strength)
            used.append(tuple(map(int, coupling)))
    else:
        deterministic = _deterministic_fuse_couplings(fam_a, fam_b)
        if deterministic is not None and len(deterministic) <= int(bridge_budget):
            candidate = A0.copy()
            for coupling in deterministic:
                candidate = candidate + _coupling_matrix(n, coupling, strength)
            if jordan_fingerprint(candidate, tol=tol).is_single_chain:
                A0 = candidate
                used.extend(deterministic)
        if not used:
            for _ in range(int(bridge_budget)):
                fp = jordan_fingerprint(A0, tol=tol)
                if fp.is_single_chain:
                    break
                coupling = _best_bridge(A0, strength, tol)
                A0 = A0 + _coupling_matrix(n, coupling, strength)
                used.append(coupling)

    fp = jordan_fingerprint(A0, tol=tol)
    if not fp.is_single_chain:
        raise ValueError(
            f"multi_jordan_fuse did not produce one Jordan chain after {len(used)} bridge(s)"
        )

    basis = list(fused.basis)
    if add_closure:
        basis.extend(_closure_basis(n))
    result = OperatorFamily(A0, basis, hermitian=False)
    result.jordan_coupling = used[0] if len(used) == 1 else tuple(used)
    result.jordan_couplings = tuple(used)
    result.jordan_fingerprint = fp
    return _attach_algebra_metadata(result, "multi_jordan_fuse", (fam_a, fam_b))


jordan_fuse_chain = multi_jordan_fuse


oplus = direct_sum
otimes = tensor_sum


def compose(*args):
    """Compose immediately or return a deferred transform.

    ``compose(fam, C)`` is equivalent to ``compose_linear(fam, C)``.
    ``compose(C)`` returns a transform usable with ``family.then(...)``.
    """
    if len(args) == 1:
        return AlgebraTransform("compose", np.asarray(args[0], dtype=np.float64))
    if len(args) == 2:
        return compose_linear(args[0], args[1])
    raise TypeError("compose expects C or (family, C)")


def pullback(C: np.ndarray) -> AlgebraTransform:
    """Deferred parameter-space pullback ``A(k) -> A(Ck)``."""
    return AlgebraTransform("pullback", np.asarray(C, dtype=np.float64))


def estimate_cost(family: OperatorFamily, operation: str | None = None):
    from .operator_algebra import OperatorCost
    return OperatorCost.estimate(family, operation=operation)


# ────────────────────────────────────────────────────
#  expectation(fam, mu, sigma, n_samples, seed) → dict
#  CONTRACT:
#    PRE:  len(mu) == fam.M, sigma > 0
#    POST: returns {mean_lam, lam_at_mean, gap, sigma, n_samples}
#    POST: gap = max|E[λ(k)] − λ(E[k])|  (Jensen gap measure)
#    WHY:  Quantify nonlinearity of λ(k).  Large gap → need 2nd order (Hessian).
#  EXAMPLE:
#    >>> res = expectation(fam, np.zeros(3), 0.1, n_samples=200)
#    >>> print(f"Jensen gap: {res['gap']:.4f}")
# ────────────────────────────────────────────────────
def expectation(family: OperatorFamily, mu: np.ndarray, sigma: float,
                n_samples: int = 300, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    K_all = rng.standard_normal((n_samples, family.M)) * sigma + mu
    lams = np.array([family.spectrum(k) for k in K_all])
    mean_lam = lams.mean(axis=0)
    lam_mu = family.spectrum(mu)
    gap = float(np.max(np.abs(mean_lam - lam_mu)))
    return {"mean_lam": mean_lam, "lam_at_mean": lam_mu, "gap": gap,
            "sigma": sigma, "n_samples": n_samples}
