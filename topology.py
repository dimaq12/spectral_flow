"""
sft.topology — Spectral topology: monodromy, braiding, Berry holonomy, EP locus.

╔══════════════════════════════════════════════════════════════════════╗
║  INTENT                                                             ║
║  Topological invariants of the parameter→spectrum mapping.          ║
║  Track eigenvalues around closed loops to detect braiding (swaps)   ║
║  and Z₂ sign flips (Möbius topology).  Find exceptional points     ║
║  where eigenvalues collide.                                         ║
╚══════════════════════════════════════════════════════════════════════╝
║  CONCEPT                                                            ║
║  • monodromy: λ tracked by eigenvector overlap around a loop.       ║
║    Hermitian: no true EP → levels repel → no braiding.             ║
║    Non-Hermitian: true EP → eigenvalues SWAP after 2π loop.        ║
║  • berry_holonomy: Z₂ ±1 — eigenvector sign flip after a loop.     ║
║  • exceptional_point_locus: grid-scan min gap → EP where gap→0.    ║
╚══════════════════════════════════════════════════════════════════════╝
║  DEPENDENCIES                                                       ║
║  ┌── sft.core.OperatorFamily (build, spectrum)                     ║
║  ├── scipy.linalg (eigh)                                           ║
║  ├── scipy.optimize.linear_sum_assignment (Hungarian matching)     ║
║  └── joblib (parallel eigh — optional)                             ║
╚══════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations
import numpy as np
from scipy import linalg
from scipy.optimize import linear_sum_assignment
from .core import OperatorFamily


# ────────────────────────────────────────────────────
#  monodromy(family, loop) → (tracked, swapped)
#  CONTRACT:
#    PRE:  loop is list of k-vectors, k[0] ≈ k[-1]
#    POST: tracked (n_steps, N) — eigenvalues tracked by eigenvector overlap
#    POST: swapped — list of (i,j) pairs that exchanged places
#    PERF: parallel eigh via joblib (GIL release) on multi-core.
#  EXAMPLE:
#    >>> tracked, swaps = monodromy(fam, [k1, k2, ..., k1])
#    >>> if len(swaps) > 0: print("Braiding detected!")
# ────────────────────────────────────────────────────
def monodromy(family: OperatorFamily,
              loop: list[np.ndarray],
              n_eigs: int | None = None,
              which: str = "SM") -> tuple[np.ndarray, list[tuple[int, int]]]:
    n_steps = len(loop)
    N = n_eigs or family.N

    def _step(kvec):
        return family.eigensystem(kvec, n_eigs=n_eigs, which=which)

    use_parallel = (N >= 64 and n_steps >= 32) or n_steps >= 512
    try:
        if use_parallel:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=-1)(delayed(_step)(kvec) for kvec in loop)
        else:
            results = [_step(kvec) for kvec in loop]
    except (ImportError, ModuleNotFoundError):
        results = [_step(kvec) for kvec in loop]

    dtype = np.result_type(*[lam for lam, _ in results])
    lams = np.zeros((n_steps, N), dtype=dtype)
    vec_dtype = np.result_type(*[v for _, v in results])
    vecs_hist = np.empty((n_steps, family.N, N), dtype=vec_dtype)
    for i, (lam, v) in enumerate(results):
        lams[i] = lam; vecs_hist[i] = v

    tracked = np.zeros((n_steps, N), dtype=dtype); tracked[0] = lams[0]
    for i_step in range(1, n_steps):
        cost = -np.abs(vecs_hist[i_step - 1].conj().T @ vecs_hist[i_step])
        _, col_ind = linear_sum_assignment(cost)
        tracked[i_step] = lams[i_step, col_ind]

    swapped = []
    d0, d_last = tracked[0], tracked[-1]
    for i in range(N):
        for j in range(i + 1, N):
            if (abs(d_last[i] - d0[j]) < 1e-6 and abs(d_last[j] - d0[i]) < 1e-6
                and abs(d0[i] - d0[j]) > 1e-6):
                swapped.append((i, j))
    return tracked, swapped


def eigenvalue_winding(family: OperatorFamily, loop: list[np.ndarray],
                       pair: tuple[int, int] = (0, 1)) -> float:
    """Winding of eigenvalue gap λ_i-λ_j around zero along a closed loop."""
    lams = np.array([family.eigensystem(kvec)[0] for kvec in loop])
    i, j = pair
    gap = np.empty(len(loop), dtype=np.result_type(lams, complex))
    gap[0] = lams[0, j] - lams[0, i]
    for t in range(1, len(loop)):
        raw = lams[t, j] - lams[t, i]
        gap[t] = raw if abs(raw - gap[t - 1]) <= abs(-raw - gap[t - 1]) else -raw
    phase = np.unwrap(np.angle(gap))
    return float((phase[-1] - phase[0]) / (2 * np.pi))


def complex_monodromy(family: OperatorFamily,
                      loop: list[np.ndarray]) -> dict[str, object]:
    """Complex-valued monodromy summary for non-Hermitian families."""
    tracked, swaps = monodromy(family, loop)
    windings = {}
    for i in range(tracked.shape[1]):
        for j in range(i + 1, tracked.shape[1]):
            windings[(i, j)] = eigenvalue_winding(family, loop, (i, j))
    return {"tracked": tracked, "swaps": swaps, "windings": windings}


# ────────────────────────────────────────────────────
#  berry_holonomy(family, loop, level) → ±1
#  CONTRACT:
#    PRE:  loop closed (k[0]≈k[-1])
#    POST: +1 = trivial bundle (no sign flip), −1 = Möbius topology
#    WHY:  Z₂ invariant: the eigenvector picks up a ± sign after 2π walk.
#  EXAMPLE:
#    >>> holo = berry_holonomy(fam_ac, circle_loop, level=1)
#    >>> assert holo == -1  # avoided crossing → Möbius
# ────────────────────────────────────────────────────
def berry_holonomy(family: OperatorFamily, loop: list[np.ndarray], level: int = 0) -> int:
    vecs_arr = np.array([family.eigensystem(kvec)[1][:, level] for kvec in loop])
    for i in range(1, len(vecs_arr)):
        if np.dot(vecs_arr[i], vecs_arr[i - 1]) < 0:
            vecs_arr[i] = -vecs_arr[i]
    return -1 if float(np.dot(vecs_arr[-1], vecs_arr[0])) < 0 else 1


# ────────────────────────────────────────────────────
#  exceptional_point_locus(family, grid_res, range) → (grid, gap_map)
#  CONTRACT:
#    PRE:  family.M == 2
#    POST: grid (2×G, G×G) gap_map — min eigenvalue gap at each grid point
#    O(grid²·N³) — use parallel eigh for speed.
# ────────────────────────────────────────────────────
def exceptional_point_locus(family: OperatorFamily, grid_resolution: int = 50,
                             k_range: tuple[float, float] = (-1.0, 1.0)) -> tuple[np.ndarray, np.ndarray]:
    if family.M != 2:
        raise ValueError("EP locus only implemented for M=2 families")
    k1 = np.linspace(k_range[0], k_range[1], grid_resolution)
    k2 = np.linspace(k_range[0], k_range[1], grid_resolution)
    gap_map = np.zeros((grid_resolution, grid_resolution))
    for i, x in enumerate(k1):
        row_lams = np.array([family.eigensystem(np.array([x, y]))[0] for y in k2])
        gaps = np.diff(row_lams, axis=1)
        gap_map[i] = gaps.min(axis=1) if gaps.shape[1] > 0 else 0.0
    return np.array([k1, k2]), gap_map


# ────────────────────────────────────────────────────
#  exceptional_point_locus_nd(family, ...) → (grid, gap_map)
#  CONTRACT:
#    PRE:  any M, projects onto 2D slice (axis_pair)
#    POST: gap map on 2D slice through k-space
# ────────────────────────────────────────────────────
def exceptional_point_locus_nd(family: OperatorFamily, grid_resolution: int = 30,
                               k_range: tuple[float, float] = (-1.0, 1.0),
                               axis_pair: tuple[int, int] = (0, 1)) -> tuple[np.ndarray, np.ndarray]:
    a1, a2 = axis_pair
    k1 = np.linspace(k_range[0], k_range[1], grid_resolution)
    k2 = np.linspace(k_range[0], k_range[1], grid_resolution)
    gap_map = np.zeros((grid_resolution, grid_resolution))
    for i, x in enumerate(k1):
        for j, y in enumerate(k2):
            kvec = np.zeros(family.M); kvec[a1] = x; kvec[a2] = y
            lam = family.eigensystem(kvec)[0]
            gaps = np.diff(lam)
            gap_map[i, j] = np.min(gaps) if len(gaps) > 0 else 0.0
    return np.array([k1, k2]), gap_map


def exceptional_point_atlas(family: OperatorFamily, grid_resolution: int = 41,
                            k_range: tuple[float, float] = (-1.0, 1.0),
                            axis_pair: tuple[int, int] = (0, 1),
                            threshold: float | None = None) -> dict[str, object]:
    """Grid/refinement-ready atlas of near exceptional points.

    The atlas reports small eigenvalue gaps and eigenvector coalescence scores.
    It is intentionally numerical: candidates are hypotheses to inspect, not
    theorem-level certificates.
    """
    a1, a2 = axis_pair
    xs = np.linspace(k_range[0], k_range[1], grid_resolution)
    ys = np.linspace(k_range[0], k_range[1], grid_resolution)
    gap_map = np.zeros((grid_resolution, grid_resolution))
    coalescence = np.zeros_like(gap_map)
    candidates: list[dict[str, object]] = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            kvec = np.zeros(family.M)
            kvec[a1] = x; kvec[a2] = y
            lam, vecs = family.eigensystem(kvec)
            if len(lam) < 2:
                gap = 0.0
            else:
                diffs = np.abs(lam[:, None] - lam[None, :])
                diffs[diffs == 0] = np.inf
                finite = diffs[np.isfinite(diffs)]
                gap = float(np.min(finite)) if finite.size else 0.0
            gap_map[i, j] = gap
            if vecs.shape[1] >= 2:
                gram = np.abs(vecs.conj().T @ vecs)
                np.fill_diagonal(gram, 0.0)
                coalescence[i, j] = float(np.max(gram))
    cutoff = threshold if threshold is not None else float(np.percentile(gap_map, 5))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if gap_map[i, j] <= cutoff:
                candidates.append({
                    "k": np.array([x, y]),
                    "gap": float(gap_map[i, j]),
                    "coalescence": float(coalescence[i, j]),
                })
    candidates.sort(key=lambda c: (c["gap"], -c["coalescence"]))
    return {
        "grid": np.array([xs, ys]),
        "gap_map": gap_map,
        "coalescence": coalescence,
        "candidates": candidates,
        "min_gap": float(np.min(gap_map)),
        "threshold": float(cutoff),
    }


def spectral_flow(family: OperatorFamily, path: list[np.ndarray]) -> np.ndarray:
    """Track eigenvalues along path.  Wraps monodromy."""
    return monodromy(family, path)[0]
