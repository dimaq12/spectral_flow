# Graph Operator — O(1) Structural Queries After Precompute

Precompute ALL structural graph properties at build time — O(V+E).
Every query is O(1). No malloc on query path.

---

## Setup

- **V = 5000** vertices, **E ≈ 24991** edges
- Random graph G(V, E/V)
- Three build phases: BFS connectivity + Tarjan DFS + Batagelj-Zaveršnik k-core

## Metrics

| Metric | Value |
|--------|-------|
| Build time | **67.5ms** |
| Components | 7  |
| Bridges | **2** |
| Articulation points | **3** |
| 1000 bridge queries | **0.40ms** |
| 1000 articulation queries | **0.15ms** |

## Scale test

| V | E | Build | Bridges | Articulations |
|---|----|:---:|:-----:|:----------:|
| 100 | 400 | 0.6ms | — | — |
| **5000** | **24991** | **67.5ms** | **2** | **3** |
| 10000 | 40000 | 88.6ms | 33 | 34 |

## Interpretation

- **2 bridges** in a 5000-node graph — O(V+E) Tarjan DFS detects them all.
- **3 articulation points** — single points of failure in the network.
- Every query is O(1): set/dict lookup for bridges, list lookup for articulations, array for coreness.
- **No malloc** on any query path — all buffers preallocated at build.

## Code

```python
import sft
gop = sft.graphop.GraphOperator(edges)
print(f"bridges={len(gop.bridges)}, articulations={len(gop.articulations)}")
print(gop.is_bridge(0, 1))  # O(1)
```
