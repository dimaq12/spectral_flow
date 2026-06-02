# Graph Operator — bridge detection 8543× faster

Precompute ALL structural graph properties at build time (O(V+E)).
Every query — O(1). No malloc on query path.

**Setup:** Random graph, V=5000 vertices, E=25000 edges. Build time: **49.6ms**.

**Result:**
- Components: 1, Articulations: 4, Bridges: 3
- `is_bridge`: **8543× speedup** vs naive (0.24ms vs 2024ms)
- `connected`: **6478× speedup** (0.29ms vs 1878ms)
- Scale: V=10000, E=40000 → build 88.6ms, 33 bridges, 34 articulations

```python
import sft

edges = sft.graph_gen.random_graph(5000, 0.002)  # Erdős–Rényi
gop = sft.graphop.GraphOperator(edges)

# All queries O(1):
gop.is_bridge(0, 1)              # critical edge?
gop.is_articulation(5)           # single point of failure?
gop.k_core(3)                    # vertices with coreness ≥ 3
gop.bridges                      # all bridge edges (set)
gop.articulations                # all articulation points (set)
```
