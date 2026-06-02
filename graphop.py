"""
sft.graphop — GraphOperator: bridges, articulations, coreness via Tarjan+BZ.

╔══════════════════════════════════════════════════════════════════════╗
║  INTENT                                                             ║
║  Precompute structural graph properties at build time.              ║
║  Bridges: O(1) query, articulation points: O(1) query.             ║
║  Coreness: k-core decomposition via Batagelj-Zaversnik (BZ).       ║
║  Iterative Tarjan DFS — no recursion, no stack overflow at V>3000. ║
╚══════════════════════════════════════════════════════════════════════╝
║  DEPENDENCIES: numpy only                                          ║
╚══════════════════════════════════════════════════════════════════════╝
"""
import numpy as np


class GraphOperator:
    """Precompute bridges, articulation points, and coreness.  O(V+E) build."""

    def __init__(self, edges: list[tuple[int, int]], n_vertices: int | None = None):
        self.edges = edges
        self.n_vertices = n_vertices or (max(max(u, v) for u, v in edges) + 1 if edges else 0)
        self.n_edges = len(edges)
        self._adj = [[] for _ in range(self.n_vertices)]
        for u, v in edges: self._adj[u].append(v); self._adj[v].append(u)
        self._bridges: set = set()
        self._articulations: set = set()
        self._coreness = np.zeros(self.n_vertices, dtype=int)
        self._build()

    def _build(self):
        self._find_bridges_and_articulations()
        self._compute_coreness()

    def _find_bridges_and_articulations(self):
        """Iterative Tarjan — O(V+E), no recursion."""
        n = self.n_vertices
        tin = np.full(n, -1, dtype=int); low = np.full(n, -1, dtype=int)
        visited = np.zeros(n, bool); parent = np.full(n, -1, dtype=int)
        children = np.zeros(n, int); timer = 0
        for start in range(n):
            if visited[start] or not self._adj[start]: continue
            stack = [(start, -1, 0)]
            while stack:
                v, p_caller, state = stack.pop()
                if state == 0:
                    if visited[v]: continue
                    visited[v] = True; parent[v] = p_caller; timer += 1; tin[v] = low[v] = timer
                    stack.append((v, p_caller, 1))
                    for to in reversed(self._adj[v]):
                        if to == parent[v]: continue
                        if visited[to]: low[v] = min(low[v], tin[to])
                        else: stack.append((to, v, 0))
                else:
                    if parent[v] != -1:
                        p = parent[v]; low[p] = min(low[p], low[v])
                        if low[v] > tin[p]: self._bridges.add((min(p, v), max(p, v)))
                        if low[v] >= tin[p]: self._articulations.add(p)
                        children[p] += 1
                    elif children[v] > 1: self._articulations.add(v)

    def _compute_coreness(self):
        """BZ k-core decomposition."""
        n = self.n_vertices
        degree = np.array([len(self._adj[i]) for i in range(n)])
        max_deg = int(degree.max()) + 1
        vert = [[] for _ in range(max_deg)]
        for v in range(n): vert[degree[v]].append(v)
        removed = np.zeros(n, bool)
        for deg in range(max_deg):
            while vert[deg]:
                v = vert[deg].pop()
                if removed[v] or degree[v] != deg: continue
                removed[v] = True; self._coreness[v] = deg
                for u in self._adj[v]:
                    if not removed[u] and degree[u] > deg:
                        degree[u] = max(degree[u] - 1, deg); vert[degree[u]].append(u)

    @property
    def bridges(self) -> set: return self._bridges
    @property
    def articulations(self) -> set: return self._articulations
    @property
    def coreness(self) -> np.ndarray: return self._coreness

    def is_bridge(self, u: int, v: int) -> bool:
        return (min(u, v), max(u, v)) in self._bridges

    def is_articulation(self, v: int) -> bool:
        return v in self._articulations

    def k_core(self, k: int) -> list[int]:
        return list(np.where(self._coreness >= k)[0])
