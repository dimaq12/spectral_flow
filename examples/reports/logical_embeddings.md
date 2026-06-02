# Logical Graph Embeddings — Logic Baked into Edges

Embed knowledge graphs using **typed edges:** AND, NOT, IMPLY.
The Laplacian has signed off-diagonals — negation literally repels in embedding space.
No training. No SGD. Deterministic.

---

## Setup

- **100 nodes** (word vocabulary)
- AND edges: **100** — co-occurrence, attract in embedding space
- NOT edges: **50** — negation context, REPEL in embedding space
- IMPLY edges: **25** — conditional context, asymmetric coupling
- Embedding dimension: **32** (top eigenvectors of signed Laplacian)

## Metrics

| Metric | Value |
|--------|-------|
| Build time | **1ms** |
| Embedding | 100 nodes × 32-d spectral space |
| Nearest neighbor to node 0 | node 99 at distance **0.3281** |

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
