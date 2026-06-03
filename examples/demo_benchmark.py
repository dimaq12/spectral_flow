#!/usr/bin/env python3
"""SFT benchmark: scale test across N, M — build time, predict, inverse."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from scipy import sparse
import sft

OUT = os.path.join(os.path.dirname(__file__), "reports", "benchmark_scale.md")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

configs = [
    (50, 10), (100, 20), (200, 50), (500, 100),
]
results = []

for N, M in configs:
    t0 = time.perf_counter()
    fam = sft.families.random(N, M, seed=42)
    build_t = (time.perf_counter() - t0) * 1000

    dk = np.ones(M) * 0.01
    t0 = time.perf_counter()
    lam_pred = fam.predict(dk)
    pred_t = (time.perf_counter() - t0) * 1e6  # μs

    t0 = time.perf_counter()
    lam_exact = fam.spectrum(dk)
    exact_t = (time.perf_counter() - t0) * 1000

    pred_err = float(np.max(np.abs(lam_exact - lam_pred)))

    DK = np.tile(dk, (1000, 1))
    t0 = time.perf_counter()
    _ = fam.predict_many(DK)
    pred_many_t = (time.perf_counter() - t0) * 1000

    results.append((N, M, build_t, pred_t, pred_many_t, exact_t, pred_err, fam.W_rank, fam.complexity))

# Heavy inverse test
N_heavy, M_heavy = 100, 30
t0 = time.perf_counter()
fam_h = sft.families.random(N_heavy, M_heavy, seed=99)
target = np.sort(fam_h.lam0 + np.linspace(-0.15, 0.15, N_heavy))
inv_result = fam_h.inverse(target, steps=30, alpha=0.3, refresh_every=8)
k, err, ok = inv_result
inv_t = (time.perf_counter() - t0) * 1000

# Graph adapter / structured basis test
N_graph = 200
adj_graph = sft.graph_gen.random_graph(N_graph, p=0.05, seed=7)
t0 = time.perf_counter()
graph_adapter = sft.graph(adj_graph)
graph_build_t = (time.perf_counter() - t0) * 1000
graph_edges = graph_adapter.n_edges
graph_kind = graph_adapter.family.basis_kind
graph_dense_equiv_mb = graph_edges * N_graph * N_graph * 8 / (1024 ** 2)

# Sparse partial spectrum test
N_sparse = 1000
row = np.arange(N_sparse - 1)
adj_sparse = sparse.coo_matrix(
    (np.ones(2 * (N_sparse - 1)),
     (np.r_[row, row + 1], np.r_[row + 1, row])),
    shape=(N_sparse, N_sparse),
).tocsr()
t0 = time.perf_counter()
sparse_family = sft.families.graph_laplacian(adj_sparse)
sparse_build_t = (time.perf_counter() - t0) * 1000
t0 = time.perf_counter()
partial_lam = sparse_family.spectrum(np.zeros(sparse_family.M), n_eigs=16)
partial_t = (time.perf_counter() - t0) * 1000

# 0.3 geometry tests
ep_family = sft.physics.exceptional_point_2x2().family()
ep_loop = [np.array([0.25 * np.cos(t), 0.25 * np.sin(t)]) for t in np.linspace(0, 2 * np.pi, 80)]
t0 = time.perf_counter()
ep_summary = sft.topology.complex_monodromy(ep_family, ep_loop)
ep_t = (time.perf_counter() - t0) * 1000
ep_winding = ep_summary["windings"][(0, 1)]

t0 = time.perf_counter()
hess_result = fam_h.inverse(target, steps=5, alpha=0.25, method="hessian")
hess_inv_t = (time.perf_counter() - t0) * 1000

x_pde = np.linspace(0.0, 1.0, 66)[1:-1]
pde_family = sft.physics.schrodinger_1d(x_pde, np.zeros_like(x_pde), max_potential_params=8).family()
t0 = time.perf_counter()
pde_lam = pde_family.spectrum(np.zeros(pde_family.M), n_eigs=8)
pde_t = (time.perf_counter() - t0) * 1000

# Monodromy test
fam2 = sft.families.avoided_crossing_2x2(0.3)
n_loop = 60
loop = [np.array([0.4*np.cos(2*np.pi*i/n_loop), 0.4*np.sin(2*np.pi*i/n_loop)]) for i in range(n_loop)]
t0 = time.perf_counter()
tracked, swaps = sft.topology.monodromy(fam2, loop)
mono_t = (time.perf_counter() - t0) * 1000

# Hessian test
N_hess, M_hess = 30, 10
t0 = time.perf_counter()
fam_hess = sft.families.random(N_hess, M_hess, seed=42)
H = sft.hessian.hessian(fam_hess)
hess_t = (time.perf_counter() - t0) * 1000

rows = "\n".join(
    f"| {N} | {M} | {bt:.0f}ms | {pt:.0f}μs | {pmt:.1f}ms | {et:.0f}ms | {pe:.1e} | {rk}/{N} | {cx:.3f} |"
    for N, M, bt, pt, pmt, et, pe, rk, cx in results
)

md = f"""# SFT Performance — Scale Benchmark

Build + predict + inverse timing across operator sizes.

---

## Setup

- Random symmetric operator families, seed=42
- dk = ones(M) × 0.01
- All times: single thread, laptop CPU

## Scale test

| N | M | Build | Predict | Predict×1000 | Exact eigh | ‖λ_pred − λ_exact‖_∞ | rank(W) | complexity |
|---|----|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
{rows}

## Heavy operations

| Operation | Config | Time | Result |
|-----------|--------|:---:|--------|
| Inverse design | N={N_heavy}, M={M_heavy}, 30 steps | **{inv_t:.0f}ms** | err={err:.2e}, converged={ok}, eigh={inv_result.eigh_count}, refresh={inv_result.n_refresh} |
| Hessian inverse | N={N_heavy}, M={M_heavy}, 5 steps | **{hess_inv_t:.0f}ms** | err={hess_result.error:.2e}, hess={hess_result.hessian_count}, method={hess_result.method} |
| Graph adapter | N={N_graph}, E={graph_edges} | **{graph_build_t:.0f}ms** | basis={graph_kind}, dense_equiv={graph_dense_equiv_mb:.1f}MB |
| Sparse partial spectrum | N={N_sparse}, E={N_sparse - 1}, K=16 | build **{sparse_build_t:.0f}ms**, spectrum **{partial_t:.0f}ms** | sparse={sparse_family._sparse_mode}, backend={sparse_family.basis_kind}, λ0={partial_lam[0]:.2e} |
| Quantum EP monodromy | 2×2 non-Hermitian, loop 80 pts | **{ep_t:.0f}ms** | gap_winding={ep_winding:.3f}, complex_W={np.iscomplexobj(ep_family.W)} |
| PDE partial spectrum | Schrödinger 1D, N={pde_family.N}, K=8 | **{pde_t:.0f}ms** | sparse={pde_family._sparse_mode}, λ0={pde_lam[0]:.3f} |
| Monodromy | 2×2, loop {n_loop} pts | **{mono_t:.0f}ms** | swaps={len(swaps)} |
| Hessian FD | N={N_hess}, M={M_hess} | **{hess_t:.0f}ms** | shape={H.shape} |

## Key observations

- **Build time scales as O(N³ + M·N²)** — dominated by eigh at N≥200
- **Predict time O(N·M)** — microseconds even for N=500, M=100 (vs {results[-1][5]:.0f}ms for exact eigh)
- **predict_many** batches 1000 perturbations with one matrix multiply
- **Inverse design** reports eigensolve count and refresh count for regression tracking
- **Graph adapter** uses `{graph_kind}` basis — avoids materialising ~{graph_dense_equiv_mb:.1f}MB of edge basis matrices
- **Sparse partial spectrum** tracks K modes without dense edge-basis materialization
- **Monodromy** is O(|loop|·N³) — unavoidable, every point needs eigh
- **Hessian FD** is O(M·N³) — {hess_t:.0f}ms for M=10, scales linearly with M

Generated by `demo_benchmark.py` on {time.strftime('%Y-%m-%d %H:%M')}.
"""
with open(OUT, 'w') as f: f.write(md)
print(f"✓ benchmark_scale.md written — N up to {configs[-1][0]}, inverse {inv_t:.0f}ms, sparse partial {partial_t:.0f}ms, graph {graph_build_t:.0f}ms, monodromy {mono_t:.0f}ms")
