"""
sft.inversion — Inversion strategies: bottleneck, fixed-point, monodromy.

╔══════════════════════════════════════════════════════════════════════╗
║  INTENT                                                             ║
║  Three alternative inverse-design strategies beyond damped Newton:  ║
║    • bottleneck_inverse — factor W via truncated SVD, invert        ║
║      through a low-dimensional bottleneck (Hack A: separation).     ║
║    • fixed_point_inverse — freeze nonlinear params, solve linear,   ║
║      iterate to self-consistency (Hack B: freezing).               ║
║    • monodromy_inverse — walk 2π around a branch cut to force       ║
║      eigenvalue swap (Hack D: bifurcation).                        ║
╚══════════════════════════════════════════════════════════════════════╝
║  DEPENDENCIES                                                       ║
║  ┌── sft.core.OperatorFamily                                       ║
║  └── scipy.linalg (svd, pinv)                                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""
import numpy as np
from scipy import linalg
from .core import OperatorFamily


def bottleneck_inverse(family: OperatorFamily, target: np.ndarray,
                       bottleneck_dim: int, steps: int = 30, alpha: float = 0.5) -> np.ndarray:
    """W = UΣV → A=U√Σ, B=√ΣV → k += α·B⁺·A⁺·residual."""
    W = family.W; U, s, Vt = linalg.svd(W, full_matrices=False)
    b = min(bottleneck_dim, len(s))
    A = U[:, :b] * np.sqrt(s[:b])[None, :]
    B = np.sqrt(s[:b])[:, None] * Vt[:b, :]
    Ap, Bp = linalg.pinv(A), linalg.pinv(B)
    k = np.zeros(family.M); family.set_reference(family.build(k))
    for _ in range(steps):
        residual = target - family.spectrum(k); n_res = min(len(residual), family.M)
        dk = alpha * (Bp @ (Ap[:, :n_res] @ residual[:n_res]))
        k += dk; family.set_reference(family.build(k))
    return k


def fixed_point_inverse(family: OperatorFamily, target: np.ndarray,
                        linear_basis_indices: list[int],
                        outer_steps: int = 10, inner_steps: int = 5,
                        alpha: float = 0.3) -> np.ndarray:
    """Freeze nonlinear params → solve linear → update.  Repeat outer_steps×inner_steps."""
    k = np.zeros(family.M)
    lmask = np.isin(np.arange(family.M), linear_basis_indices).astype(float)
    for _ in range(outer_steps):
        family.set_reference(family.build(k))
        for _ in range(inner_steps):
            residual = target - family.spectrum(k)
            n_r = min(len(residual), family.W_pinv.shape[1])
            dk = alpha * (family.W_pinv[:, :n_r] @ residual[:n_r])
            if len(dk) > family.M: dk = dk[:family.M]
            k += dk * lmask
            family.set_reference(family.build(k))
    return k


def monodromy_inverse(target_func, n_pts_circle: int = 60,
                      radius: float = 1.0) -> tuple[np.ndarray, float]:
    """Complex bifurcation: walk 2π around branch cut, track eigenvalues."""
    theta = np.linspace(0, 2 * np.pi, n_pts_circle)
    lams = np.array([target_func(np.array([radius * np.cos(t), radius * np.sin(t)]))[0]
                     for t in theta])
    return lams, float(np.max(np.abs(lams[-1] - lams[0])))
