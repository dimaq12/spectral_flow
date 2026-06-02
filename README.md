# spectral_flow

**One formula. One kernel. All operators.**

```
W(i,j) = v_iᵀ · B_j · v_i  =  ∂λ_i/∂k_j
```

A(k) = A₀ + Σ kⱼ·Bⱼ — a linear operator family.  
W is the spectral flow kernel: one diagonalization → all eigenvalue derivatives w.r.t. all parameters.

---

## Why

| Before | After |
|--------|-------|
| Every perturbation k → new `eigh` O(N³) | One `eigh` at build, then `W·dk` at O(N·M) |
| Spectral flow as abstract theory | W as the operator's Jacobian — prediction, inverse design, topology |
| Each domain — custom code | 12 adapters: audio, image, graph, text, video, voxel, point cloud, molecule, finance, tabular, mesh |

---

## 15 seconds to results

```python
import sft

# Core
fam = sft.families.random(N=100, M=30)
fam.W                    # (100,30) — ∂λ/∂k
fam.predict(dk)          # λ(k₀+dk) in O(N·M)
fam.inverse(target)      # find k matching target spectrum
fam.complexity           # rank(W)/N — structural complexity

# Adapters — killer feature
pic   = sft.image(pixels, patch_size=8)
sound = sft.audio(signal, sr=44100, n_bands=16)
net   = sft.graph(adjacency)

# Natural language → operator
fam = sft.from_task("sort these numbers", data)

# Graphs — structural analysis O(1) after precompute
gop = sft.graphop.GraphOperator(edges)
gop.is_bridge(0, 1)       # O(1)
gop.is_articulation(5)    # O(1)

# Topology
tracked, swaps = sft.topology.monodromy(fam, loop)
holonomy = sft.topology.berry_holonomy(fam, loop)

# Operator algebra
fam_sum = sft.algebra.direct_sum(a, b)   # ⊕
fam_ten = sft.algebra.tensor_sum(a, b)   # ⊗

# Invariants — 5 keys in one call
fp = sft.invariants.all_invariants(fam)

# Streaming CDF
stream = sft.streaming.StreamingCDF(capacity=1000)
for x in data_stream: stream.add(x)
stream.cdf(threshold)
```

---

## 22 modules

| Module | Purpose |
|--------|---------|
| `core` | `OperatorFamily`, W, W⁺, predict, inverse, nullspace |
| `algebra` | ⊕ (direct sum), ∘ (composition), ⊗ (Kronecker sum), ∫ (expectation) |
| `topology` | monodromy, Berry phase, exceptional points, spectral flow |
| `hessian` | ∂²λ/∂k² — analytic and finite-difference |
| `families` | random, graph_laplacian, toeplitz, diagonal, avoided_crossing |
| `adapters` | 12 domain adapters (Audio…Mesh) |
| `tasks` | classify_task, cdf_rank_sort, dct_matrix, filter_via_dct |
| `constructor` | from_task("sort", data) → OperatorFamily |
| `graphop` | bridges, articulation points, k-core in O(1) queries |
| `embed` | deterministic graph embeddings |
| `order` | CDF, rank, quantile, defect α-spectroscopy |
| `cluster` | spectral clustering, kNN, auto-basis selection |
| `compress` | spectral compression, DCT codec |
| `transport` | 1D optimal transport (Monge map + W₂ distance) |
| `streaming` | online CDF and ORDER operators |
| `carleman` | GF(2)/GF(3) operators, complex Hermitian check |
| `homotopy` | homotopy continuation, Tikhonov W⁺, trust-region |
| `inversion` | bottleneck, fixed-point, monodromy inverse strategies |
| `invariants` | 5 global invariants: kurtosis, sparsity, preimage, coherence, zeta |
| `basis` | Toeplitz, DCT/Fourier, Gaussian affinity, graph generators |
| `arnoldi` | Arnoldi iteration, Ritz values, Krylov solver |
| `codec` | InstantSpectralCodec: M·Δk encode/decode |
| `verify` | C1-C8 + S1-S5 theoretical verification suite |

---

## Theoretical foundations

SFT generalizes the Fourier transform to non-commutative operators.
Fourier ⊂ SFT: for circulant operators, W coincides with DFT.
For arbitrary A(k) = A₀ + Σ kⱼ·Bⱼ, W gives the full spectral Jacobian.

**rank(W) = computational complexity of the task:**
- rank(W) = 1 → ORDER regime (sorting, CDF)
- rank(W) ≪ N → structure (graphs, filters)
- rank(W) ≈ N → random (no shortcut)

**nullspace(W) = isospectral manifold:**
dim(ker(W)) directions in k-space do not change the spectrum.
The operator "cannot hear" them.

---

## Install

```bash
pip install -e .
```

Dependencies: `numpy>=1.24`, `scipy>=1.10`. Python ≥ 3.10.

---

## License

MIT
