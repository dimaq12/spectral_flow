"""
sft.transport — Optimal transport: 1D Monge map, Wasserstein-2.

╔══════════════════════════════════════════════════════════════════════╗
║  INTENT                                                             ║
║  optimal_transport_map(mu, nu) → T = F_ν⁻¹∘F_μ, Wasserstein-2.     ║
║  1D OT via CDF matching — closed-form, no Sinkhorn needed.          ║
╚══════════════════════════════════════════════════════════════════════╝
║  DEPENDENCIES: numpy (searchsorted, concatenate)                    ║
╚══════════════════════════════════════════════════════════════════════╝
"""
import numpy as np


def optimal_transport_map(mu: np.ndarray, nu: np.ndarray,
                          grid: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """T = F_ν⁻¹∘F_μ.  Returns (T(grid), W2)."""
    mu_s = np.sort(np.asarray(mu, np.float64).ravel())
    nu_s = np.sort(np.asarray(nu, np.float64).ravel())
    if grid is None: grid = mu_s
    T = np.array([nu_s[max(0, min(int(np.searchsorted(mu_s, x) / len(mu_s) * (len(nu_s) - 1)),
                                   len(nu_s) - 1))] for x in grid])
    cg = np.sort(np.concatenate([mu_s[:min(500, len(mu_s))], nu_s[:min(500, len(nu_s))]]))
    qm = np.searchsorted(mu_s, cg) / len(mu_s)
    qn = np.searchsorted(nu_s, cg) / len(nu_s)
    dq = np.abs(qm - qn); dx = np.diff(cg) if len(cg) > 1 else np.array([0.0])
    if len(dx) < len(dq): dx = np.append(dx, dx[-1] if len(dx) > 0 else 1.0)
    return T, float(np.sqrt(np.sum(dq * dx[:len(dq)])))
