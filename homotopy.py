"""
sft.homotopy — Homotopy continuation, regularised pinv, trust-region.

╔══════════════════════════════════════════════════════════════════════╗
║  INTENT                                                             ║
║  Solve inverse problems via homotopy: H_τ = (1−τ)H_easy + τ·H_target║
║  Track eigenvalues from τ=0 (easy, known) to τ=1 (target).          ║
║  Uses Tikhonov-regularised W⁺ and trust-region corrector.           ║
╚══════════════════════════════════════════════════════════════════════╝
║  DEPENDENCIES: sft.core, numpy, scipy.linalg.solve                  ║
╚══════════════════════════════════════════════════════════════════════╝
"""
import numpy as np
from scipy import linalg
from .core import OperatorFamily


def regularised_pinv(W: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """Tikhonov: W⁺_reg = (W^T·W + reg·I)^(−1)·W^T.  Stable for ill-conditioned W."""
    N, M = W.shape
    return linalg.solve(W.T @ W + reg * np.eye(min(N, M)),
                        W.T if M <= N else (W @ W.T + reg * np.eye(N)).T @ W, assume_a='pos')


def trust_region_corrector(x: np.ndarray, target: np.ndarray,
                           W: np.ndarray, radius: float = 1.0) -> np.ndarray:
    """Dogleg trust-region step for min||d||≤r ||W·d − target||."""
    g = -W.T @ target; B = W.T @ W
    gn_step = -np.linalg.solve(B + np.eye(len(x)) * 1e-10, g)
    gn_norm = np.linalg.norm(gn_step)
    if gn_norm <= radius: return gn_step
    sd_step = -g; sd_norm = np.linalg.norm(sd_step)
    return sd_step * (radius / max(sd_norm, 1e-15)) if sd_norm > 1e-15 else np.zeros_like(x)


def track_homotopy(target_lam: np.ndarray, n_steps: int = 20,
                   family: OperatorFamily | None = None,
                   easy_lam: np.ndarray | None = None, adaptive: bool = True):
    """H_τ path tracking: λ(τ) = (1−τ)λ_easy + τ·λ_target.  Uses SFT prediction + correction."""
    if family is None: raise ValueError("family required")
    easy = easy_lam or np.ones(family.N)
    k = np.zeros(family.M); step_size = 1.0 / n_steps
    tau = np.linspace(0, 1, n_steps)
    for i, t in enumerate(tau):
        if i == 0: continue
        target_t = (1 - t) * easy + t * target_lam
        family.set_reference(family.build(k))
        residual = target_t - family.spectrum(k)
        n_r = min(len(residual), family.W.shape[1], family.W.shape[0])
        Wp = regularised_pinv(family.W, 1e-4)
        if n_r == Wp.shape[1]:
            dk = step_size * (Wp @ residual[:n_r])
        else:
            dk = step_size * (Wp[:, :n_r] @ residual[:n_r])
        k += dk
        if adaptive and i > 1:
            err_prev = float(np.max(np.abs(family.spectrum(k) - target_t)))
            step_size = min(1.0 / n_steps, step_size * min(2.0, max(0.1, 0.1 / (err_prev + 1e-10))))
    lam_final = family.spectrum(k)
    err = float(np.max(np.abs(lam_final[:len(target_lam)] - target_lam)))
    return k, err, err < 0.05
