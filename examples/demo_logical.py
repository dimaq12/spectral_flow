#!/usr/bin/env python3
"""SFT demo: logical graph embeddings — typed edges AND/NOT/IMPLY."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import sft

OUT = os.path.join(os.path.dirname(__file__), "reports", "logical_embeddings.md")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# Simulate Shakespeare-style co-occurrence + logic graph
rng = np.random.default_rng(42)
n = 100
# AND edges: words that co-occur
and_edges = [(i, (i+1)%n) for i in range(n-1)] + [(0, n-1)]
# NOT edges: negation context
not_edges = [(i, (i+n//2)%n) for i in range(0, n//2)]
# IMPLY edges: conditional context
imply_edges = [(i, (i+n//4)%n) for i in range(0, n//4)]

t0 = time.perf_counter()
lemb = sft.embed.LogicalGraphEmbedder(n, and_edges, not_edges, imply_edges, K=32)
build_t = (time.perf_counter() - t0) * 1000

# Nearest neighbor check for node 0
emb = np.array([lemb.embed_node(i) for i in range(n)])
dist_0 = np.array([np.linalg.norm(emb[0] - emb[i]) for i in range(1, n)])
nn_idx = np.argmin(dist_0) + 1
nn_dist = float(dist_0[nn_idx - 1])

md = f"""# Logical Graph Embeddings — Logic Baked into Edges

Embed knowledge graphs using **typed edges:** AND, NOT, IMPLY.
The Laplacian has signed off-diagonals — negation literally repels in embedding space.
No training. No SGD. Deterministic.

---

## Setup

- **{n} nodes** (word vocabulary)
- AND edges: **{len(and_edges)}** — co-occurrence, attract in embedding space
- NOT edges: **{len(not_edges)}** — negation context, REPEL in embedding space
- IMPLY edges: **{len(imply_edges)}** — conditional context, asymmetric coupling
- Embedding dimension: **32** (top eigenvectors of signed Laplacian)

## Metrics

| Metric | Value |
|--------|-------|
| Build time | **{build_t:.0f}ms** |
| Embedding | {n} nodes × 32-d spectral space |
| Nearest neighbor to node 0 | node {nn_idx} at distance **{nn_dist:.4f}** |

## Interpretation

- **NOT edges create repulsion** — the Laplacian off-diagonals are positive for NOT edges,
  unlike standard Laplacians where all off-diagonals are negative.

- **IMPLY edges are asymmetric** — A → B couples A to B stronger than B to A,
  encoding directional logical flow in the spectral geometry.

- The embedding is **deterministic** — same graph always produces the same vectors.
  No random initialization. No SGD. No hyperparameters beyond K.

- Build time scales with sparse eigsh: **O(K·n·E)** — practical for graphs up to 10⁵ nodes.

## Code

```python
import sft
lemb = sft.embed.LogicalGraphEmbedder(n, and_edges, not_edges, imply_edges, K=64)
vec = lemb.embed_node(0)  # 64-d spectral coordinates with logical constraints
```
"""
with open(OUT, 'w') as f: f.write(md)
print(f"✓ logical_embeddings.md written — {n} nodes, build {build_t:.0f}ms, nn dist {nn_dist:.4f}")
