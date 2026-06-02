"""
sft.embed — GraphEmbedder, LogicalGraphEmbedder, build_ternary_laplacian.

╔══════════════════════════════════════════════════════════════════════╗
║  INTENT                                                             ║
║  Deterministic node+graph embeddings from spectral + structural      ║
║  features.  Logical (AND/NOT/IMPLY) and ternary (0/1/2) variants.   ║
╚══════════════════════════════════════════════════════════════════════╝
║  DEPENDENCIES: numpy, scipy.linalg.eigh                            ║
╚══════════════════════════════════════════════════════════════════════╝
"""
import numpy as np
from scipy import linalg


class GraphEmbedder:
    """Spectral + structural → (2K+4) node vectors, (K+R+5) graph vectors."""
    def __init__(self, adjacency: np.ndarray, K: int = 8, R: int = 4):
        self.adj = np.asarray(adjacency, np.float64); self.N = self.adj.shape[0]
        self.K, self.R = min(K, self.N), min(R, self.N)
        D = np.diag(np.sum(self.adj, axis=1)); self.L = D - self.adj
        self.lam, self.V = linalg.eigh(self.L)
        self.top_lam = self.lam[:self.K]; self.top_vecs = self.V[:, :self.K]
        self.degrees = np.sum(self.adj, axis=1)
        self.spread = self.lam[-1] - self.lam[0]
        self.spectral_radius = max(abs(self.lam[0]), abs(self.lam[-1]))
        self.harmonic = np.sum(1.0 / np.maximum(self.degrees, 1))

    def embed_node(self, i: int) -> np.ndarray:
        spectral = self.top_vecs[i]; d = self.degrees[i]
        nbs = np.where(self.adj[i] > 0)[0]; n_n = len(nbs); clust = 0.0
        if n_n > 0:
            sub = self.adj[np.ix_(nbs, nbs)]
            tri = np.trace(sub @ sub @ sub) / 6.0 if n_n >= 3 else 0.0
            clust = tri / (n_n * (n_n - 1) / 2.0) if n_n > 1 else 0.0
        struct = np.array([d / max(self.degrees.max(), 1), n_n / max(self.N - 1, 1), clust, 0.0])
        return np.concatenate([spectral, struct[:self.R]])

    def embed_graph(self) -> np.ndarray:
        K, R = self.K, self.R; tl = self.top_lam; f = list(tl)
        if K >= 2: f.extend(np.diff(tl)[:K - 1]); f.append(np.std(tl))
        else: f.extend([0.0] * K)
        f.extend([self.spread, self.spectral_radius, np.mean(self.degrees), self.harmonic, float(self.N)])
        return np.array(f[:K + R + 5])


class LogicalGraphEmbedder:
    """Typed edges: AND (+1), NOT (−1), IMPLY (0.5).  Signed Laplacian."""
    def __init__(self, n: int, and_e: list, not_e: list | None = None,
                 imply_e: list | None = None, K: int = 8):
        self.n = n; self.K = min(K, n - 1)
        L = np.zeros((n, n))
        for u, v in and_e: L[u, v] = L[v, u] = -1.0; L[u, u] += 1.0; L[v, v] += 1.0
        for u, v in (not_e or []): L[u, v] = -1.0; L[v, u] = 1.0
        for u, v in (imply_e or []): L[u, v] = -0.5; L[v, u] = 0.5
        self.L = (L + L.T) / 2; self.lam, self.vecs = linalg.eigh(self.L)

    def embed_node(self, i: int) -> np.ndarray: return self.vecs[i, :self.K]

    def embed_graph(self) -> np.ndarray:
        lam = self.lam[:self.K]; f = list(lam)
        if self.K >= 2: f.extend(np.diff(lam)[:self.K - 1])
        f.append(self.lam[-1] - self.lam[0])
        return np.array(f)


def build_ternary_laplacian(n: int, e01: list, e02: list | None = None) -> np.ndarray:
    """Ternary GF(3) Laplacian: weight-1 edges from e01, weight-2 from e02."""
    L = np.zeros((n, n))
    for u, v in e01: L[u, v] = L[v, u] = -1.0; L[u, u] += 1.0; L[v, v] += 1.0
    if e02:
        for u, v in e02: L[u, v] = L[v, u] = -2.0; L[u, u] += 2.0; L[v, v] += 2.0
    return L
