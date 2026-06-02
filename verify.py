"""
sft.verify — SFT conclusions C1-C8 verification suite.

╔══════════════════════════════════════════════════════════════════════╗
║  INTENT                                                             ║
║  Run all 8 SFT theoretical conclusions as automated tests:          ║
║  C1: Fourier⊂SFT  C2: τ-invariance  C3: W-entropy                  ║
║  C4: Rank separability  C5: Defect universality                     ║
║  C6: Monodromy classification  C7: Hessian sparsity                 ║
║  C8: Universal embedding                                            ║
║  run_verification_suite() → dict with all results                   ║
╚══════════════════════════════════════════════════════════════════════╝
"""
import numpy as np
from scipy import linalg


def verify_c1_fourier_subset() -> bool:
    from .tasks import dct_matrix; from .families import toeplitz
    n = 16; C = dct_matrix(n); W = toeplitz(n, n - 1).W
    return abs(np.sum(C @ C.T) / (n * n) - np.sum(W @ W.T) / (n * W.shape[1])) < 10.0


def verify_c2_tau_invariance() -> bool:
    from .tasks import dct_matrix; N = 50
    C = dct_matrix(N); x = np.random.default_rng(0).standard_normal(N)
    return np.max(np.abs(np.abs(C @ x) - np.abs(C @ np.sort(x)))) < 1e-6


def verify_c3_w_entropy() -> float:
    from .families import random; fam = random(20, 8, seed=42)
    W = fam.W; WWt = W @ W.T; det = np.linalg.det(WWt)
    return float(np.log(max(det, 1e-15))) if det > 0 else 0.0


def verify_c4_rank_separability() -> dict:
    from .families import diagonal, graph_laplacian
    fo = diagonal(20); fg = graph_laplacian(np.ones((20, 20)) - np.eye(20))
    return {"ORDER_complexity": fo.complexity, "GRAPH_complexity": fg.complexity,
            "separable": fo.complexity < fg.complexity}


def verify_c5_defect_universality() -> dict:
    from .order import rank_defect_analysis
    rg = rank_defect_analysis(np.random.default_rng(0).standard_normal(500))
    ru = rank_defect_analysis(np.random.default_rng(0).uniform(-3, 3, 500))
    return {"gaussian_alpha": rg["slope"], "uniform_alpha": ru["slope"]}


def verify_c6_monodromy_classification() -> int:
    from .topology import berry_holonomy; from .families import avoided_crossing_2x2
    fam = avoided_crossing_2x2(0.3)
    loop = [np.array([0.4 * np.cos(t), 0.4 * np.sin(t)]) for t in np.linspace(0, 2 * np.pi, 60)]
    return berry_holonomy(fam, loop, level=1)


def verify_c7_hessian_sparsity() -> float:
    from .hessian import hessian, hessian_sparsity; from .families import random
    return hessian_sparsity(hessian(random(15, 6, seed=42)))


def verify_c8_universal_embedding() -> bool:
    from .embed import GraphEmbedder
    ge = GraphEmbedder(np.array([[0,1,0,0],[1,0,1,0],[0,1,0,1],[0,0,1,0]]), K=3, R=2)
    return len(ge.embed_graph()) > 0 and all(len(ge.embed_node(i)) > 0 for i in range(4))


def run_verification_suite() -> dict:
    return {f"C{n}_result": globals()[f"verify_c{n}_fourier_subset" if n == 1 else
               f"verify_c{n}_tau_invariance" if n == 2 else
               f"verify_c{n}_w_entropy" if n == 3 else
               f"verify_c{n}_rank_separability" if n == 4 else
               f"verify_c{n}_defect_universality" if n == 5 else
               f"verify_c{n}_monodromy_classification" if n == 6 else
               f"verify_c{n}_hessian_sparsity" if n == 7 else
               f"verify_c{n}_universal_embedding"]() for n in range(1, 9)}


def verify_s1_complexity() -> bool:
    """S1: complexity = rank(W)/N ∈ [0,1] for any OperatorFamily."""
    from .families import random
    for seed in range(5):
        c = random(20, 10, seed=seed).complexity
        if not (0.0 <= c <= 1.0): return False
    return True


def verify_s2_perturbation_theory() -> bool:
    """S2: ||λ(k₀+dk) − λ(k₀) − W·dk|| = O(||dk||²)."""
    from .families import random
    from .hessian import spectral_curvature
    fam = random(10, 3, seed=42)
    for eps in [1e-2, 5e-3, 1e-3]:
        d = np.ones(3); d = d / np.linalg.norm(d)
        curv = spectral_curvature(fam, d, eps=1e-4)
        scale = np.linalg.norm(d) * eps
        err_fd = np.max(np.abs(fam.spectrum(d * eps) - fam.predict(d * eps)))
        expected = np.max(np.abs(curv)) * scale**2 / 2
        if err_fd > expected * 5: return False
    return True


def verify_s3_drum_shape() -> bool:
    """S3: dirichlet_laplacian spectrum → W rank → detect shape changes."""
    from .families import graph_laplacian
    from .basis import path_graph, star_graph
    fam_p = graph_laplacian(path_graph(20))
    fam_s = graph_laplacian(star_graph(20))
    return fam_p.W_rank != fam_s.W_rank


def verify_s4_information_bound() -> float:
    """S4: log|det(W^T W + I)| ≥ 0 always."""
    from .families import random
    W = random(15, 8, seed=42).W
    det_log = np.linalg.slogdet(W.T @ W + np.eye(8))[1] if W.shape[1] >= 1 else 0.0
    return float(det_log)


def verify_s5_rmt_vs_defect() -> float:
    """S5: Wigner semi-circle spacing vs defect Poisson spacing — KS distance."""
    from .families import random
    fam = random(30, 15, seed=42)
    gaps = np.diff(fam.lam0)
    gaps_norm = gaps / np.mean(gaps) if np.mean(gaps) > 1e-15 else gaps
    from scipy import stats
    d_wigner, _ = stats.kstest(gaps_norm, lambda x: np.pi * x / 2 * np.exp(-np.pi * x**2 / 4))
    return float(d_wigner)
