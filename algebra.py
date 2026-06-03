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
