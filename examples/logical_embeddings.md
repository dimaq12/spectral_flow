# Logical Graph Embeddings — logic baked into edges

Embed words, molecules, or knowledge graphs using typed edges: **AND, NOT, IMPLY.**

The Laplacian has signed off-diagonals — negation literally repels in embedding space. No training. No SGD. Deterministic. Built at 110ms for 200 nodes.

**Setup:** Shakespeare sonnets → 200-word graph. AND=185 edges (co-occurrence), NOT=397 (negation context), IMPLY=431 (conditional context).

**Result:**
- Build: **110ms** — 200 nodes × 64-d spectral space
- Embedding dimension: 64 (top eigenvectors of signed Laplacian)

|| Query | Nearest neighbors (cosine distance) |
||-------|-------------------------------------|
|| love | sweet (0.052), virtue (0.055), die (0.059) |
|| beauty | sweet (0.055), self (0.062), write (0.069) |
|| death | look (0.276), glass (0.309), pride (0.322) |
|| truth | sweet (0.052), love (0.063), roses (0.069) |

```python
import sft

# AND edges: words co-occurring in same line — attract in embedding space
# NOT edges: words in negated context — REPEL in embedding space
# IMPLY edges: words in conditional context — asymmetric coupling

lemb = sft.embed.LogicalGraphEmbedder(
    n_vertices=200,
    and_edges=[(0, 1), (1, 2), ...],
    not_edges=[(3, 4), ...],
    imply_edges=[(5, 6), ...],
    K=64
)
vec = lemb.embed_node(0)  # 64-d spectral coordinates with logical constraints
# → "beauty" nearest to "fair" at cosine distance 0.08
```
