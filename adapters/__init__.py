"""
sft.adapters — Domain adapters (12 domains): Audio, Image, Graph, Text,
              Timeseries, Video, Voxel, PointCloud, Molecular,
              Financial, Tabular, Mesh.

╔══════════════════════════════════════════════════════════════════════╗
║  INTENT                                                             ║
║  Each adapter takes RAW DATA and builds an OperatorFamily with      ║
║  domain-interpretable parameters.  The user never touches matrices  ║
║  directly — they just provide data, and SFT does the rest.          ║
║                                                                     ║
║  All adapters expose the same interface:                            ║
║    .family   → OperatorFamily                                       ║
║    .kernel   → W = ∂λ/∂k  (sensitivity matrix)                     ║
║    .predict  → λ(k₀+dk) ≈ λ₀ + W·dk                                ║
║    .inverse  → find k such that λ(k) ≈ target                      ║
║    .rank     → structural rank of W                                 ║
║    .complexity → rank/N                                             ║
║    .spectrum → eigenvalues at current reference                     ║
╚══════════════════════════════════════════════════════════════════════╝
║  CONCEPT                                                            ║
║  ┌─────────────┬──────────────────────┬───────────────────────────┐ ║
║  │ ADAPTER     │ OPERATOR             │ PARAMETERS                │ ║
║  ├─────────────┼──────────────────────┼───────────────────────────┤ ║
║  │ Audio       │ frame autocorrelation│ band gains (EQ)           │ ║
║  │ Image       │ patch covariance     │ region brightness         │ ║
║  │ Graph       │ graph Laplacian      │ edge weights              │ ║
║  │ Text        │ word co-occurrence   │ word-pair associations    │ ║
║  │ Timeseries  │ Hankel lag-covariance│ per-lag weights           │ ║
║  │ Video       │ spatiotemporal patch │ spatiotemporal zones      │ ║
║  │ Voxel       │ 3D cube covariance   │ spatial zone density      │ ║
║  │ PointCloud  │ k-NN graph Laplacian │ per-point density         │ ║
║  │ Molecular   │ Coulomb+bond Hessian │ per-bond stiffness        │ ║
║  │ Financial   │ asset correlation    │ sector + per-asset vol    │ ║
║  │ Tabular     │ feature covariance   │ group + per-feature weight│ ║
║  │ Mesh        │ cotangent Laplacian  │ per-edge stiffness        │ ║
║  └─────────────┴──────────────────────┴───────────────────────────┘ ║
╚══════════════════════════════════════════════════════════════════════╝
║  DEPENDENCIES: sft.core.OperatorFamily, numpy, sliding_window_view  ║
╚══════════════════════════════════════════════════════════════════════╝
"""
from ..core import OperatorFamily
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class BaseAdapter:
    """Base class for all domain adapters. Provides uniform SFT interface."""

    _family: OperatorFamily

    @property
    def family(self) -> OperatorFamily:
        return self._family

    @property
    def kernel(self) -> np.ndarray:
        return self._family.W

    def predict(self, changes: np.ndarray) -> np.ndarray:
        return self._family.predict(changes)

    def inverse(self, target_spectrum: np.ndarray, **kw):
        return self._family.inverse(target_spectrum, **kw)

    @property
    def rank(self) -> int:
        return self._family.W_rank

    @property
    def complexity(self) -> float:
        return self._family.complexity

    @property
    def spectrum(self) -> np.ndarray:
        return self._family.lam0.copy()


class AudioAdapter(BaseAdapter):
    """
    Audio signal → autocorrelation operator → SFT.

    Operator: frame-wise autocorrelation → averaged Toeplitz matrix.
    Parameters: band gains (EQ).
    Basis: diagonal band masks in frequency domain.

    Example
    -------
    >>> sound = AudioAdapter(signal, sr=44100, n_bands=16)
    >>> W = sound.kernel       # (n_bands, n_bands) spectral response matrix
    >>> lam = sound.predict(gain_changes)  # predict spectrum after EQ
    """

    def __init__(self, signal: np.ndarray, sample_rate: float = 44100.0,
                 frame_size: int = 1024, hop_size: int | None = None,
                 n_bands: int = 16):
        self.signal = np.asarray(signal, dtype=np.float64).ravel()
        self.sr = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size or frame_size // 2
        self.n_bands = n_bands
        self._build()

    def _build(self):
        n = len(self.signal)
        n_frames = max((n - self.frame_size) // self.hop_size + 1, 1)
        self.n_frames = n_frames

        if n >= self.frame_size:
            idx = np.arange(n_frames) * self.hop_size
            frames = self.signal[idx[:, None] + np.arange(self.frame_size)]
        else:
            frames = np.zeros((n_frames, self.frame_size))
            frames[0, :n] = self.signal

        win = np.hanning(self.frame_size)
        frames = frames * win[None, :]

        fft_size = self.frame_size // 2 + 1
        spectrum = np.abs(np.fft.rfft(frames, n=self.frame_size))[:, :fft_size]
        self.power = np.mean(spectrum ** 2, axis=0)

        freqs = np.linspace(0, self.sr / 2, fft_size)
        if self.n_bands > 1:
            lo = np.logspace(np.log10(max(freqs[1], 1)), np.log10(self.sr / 2),
                             self.n_bands + 1)
        else:
            lo = np.array([0, self.sr / 2])
        self.band_edges = lo
        self.band_center = np.sqrt(lo[:-1] * lo[1:])

        masks = np.array([(freqs >= lo[j]) & (freqs < lo[j + 1])
                          for j in range(self.n_bands)], dtype=np.float64)
        self._basis = [np.diag(masks[j]) for j in range(self.n_bands)]

        A0 = np.diag(self.power)
        self._family = OperatorFamily(A0, self._basis)

    @property
    def freqs(self) -> np.ndarray:
        return np.linspace(0, self.sr / 2, len(self.power))

    @property
    def band_freqs(self) -> np.ndarray:
        return self.band_center.copy()


class ImageAdapter(BaseAdapter):
    """
    Image → patch covariance operator → SFT.

    Operator: covariance matrix of image patches.
    Parameters: per-pixel or per-region brightness/contrast.
    Basis: pixel-region masks.

    Example
    -------
    >>> img = ImageAdapter(pixels, patch_size=8, n_regions=16)
    >>> W = img.kernel
    >>> lam = img.predict(region_changes)
    """

    def __init__(self, pixels: np.ndarray, patch_size: int = 8,
                 stride: int | None = None, n_regions: int = 16):
        self.pixels = np.asarray(pixels, dtype=np.float64)
        if self.pixels.ndim == 3:
            self.pixels = 0.299 * self.pixels[:, :, 0] + \
                         0.587 * self.pixels[:, :, 1] + \
                         0.114 * self.pixels[:, :, 2]
        self.image = self.pixels
        self.patch_size = patch_size
        self.stride = stride or patch_size // 2
        self.n_regions = n_regions
        self._build()

    def _build(self):
        h, w = self.image.shape
        ps = self.patch_size

        if h < ps or w < ps:
            raise ValueError(
                f"Image dimensions ({h}x{w}) must be >= patch_size ({ps}x{ps})"
            )

        patches_view = sliding_window_view(self.image, (ps, ps))[::self.stride, ::self.stride]
        n_i, n_j = patches_view.shape[0], patches_view.shape[1]
        patches = patches_view.reshape(n_i * n_j, ps * ps)
        n_patches = len(patches)
        self.n_patches = n_patches

        self.patch_mean = patches.mean(axis=0)
        patches_centered = patches - self.patch_mean[None, :]
        Cov = (patches_centered.T @ patches_centered) / max(n_patches - 1, 1)
        self.Cov = (Cov + Cov.T) / 2

        n_side = int(np.ceil(np.sqrt(self.n_regions)))
        region_h = np.ceil(h / n_side).astype(int)
        region_w = np.ceil(w / n_side).astype(int)

        self._basis = []
        count = 0
        for ri in range(n_side):
            r0 = ri * region_h
            r1 = min(r0 + region_h, h)
            for rj in range(n_side):
                if count >= self.n_regions:
                    break
                c0 = rj * region_w
                c1 = min(c0 + region_w, w)
                mask = np.zeros((h, w))
                mask[r0:r1, c0:c1] = 1.0

                indices_pi = np.tile(np.arange(ps).reshape(-1, 1), (1, ps)).ravel()
                indices_pj = np.tile(np.arange(ps), ps)
                safe_pi = np.minimum(indices_pi, h - 1)
                safe_pj = np.minimum(indices_pj, w - 1)
                mask_flat = mask[safe_pi, safe_pj]
                Bj = np.diag(mask_flat)
                self._basis.append(Bj)
                count += 1
            if count >= self.n_regions:
                break

        self._family = OperatorFamily(self.Cov, self._basis)


class GraphAdapter(BaseAdapter):
    """
    Graph → Laplacian operator → GraphSFT.

    Operator: graph Laplacian L = D − A.
    Parameters: edge weights.
    Basis: edge Laplacian B_e = (e_u − e_v)(e_u − e_v)^T.

    kernel:  W(i,e) = (v_i(u) − v_i(v))²

    Example
    -------
    >>> net = GraphAdapter(adjacency)
    >>> W = net.kernel
    >>> lam = net.predict(edge_weight_changes)
    """

    def __init__(self, adjacency: np.ndarray):
        self.adjacency = np.asarray(adjacency, dtype=np.float64)
        if self.adjacency.shape[0] != self.adjacency.shape[1]:
            raise ValueError("Adjacency must be square")
        self.n_nodes = self.adjacency.shape[0]
        self._build()

    def _build(self):
        N = self.n_nodes
        row, col = np.triu(self.adjacency != 0, 1).nonzero()
        edges = list(zip(row.tolist(), col.tolist()))
        self.n_edges = len(edges)

        M = len(edges)
        basis_arr = np.zeros((M, N, N))
        for k, (u, v) in enumerate(edges):
            basis_arr[k, u, u] = 1.0
            basis_arr[k, v, v] = 1.0
            basis_arr[k, u, v] = -1.0
            basis_arr[k, v, u] = -1.0

        D = np.diag(np.sum(self.adjacency, axis=1))
        A0 = D - self.adjacency
        self._family = OperatorFamily(A0, list(basis_arr) if M > 0 else [])

    @property
    def isospectral_dim(self) -> int:
        """Number of edge directions that don't change the spectrum."""
        return self._family.isospectral_dimension()


class TextAdapter(BaseAdapter):
    """
    Text → co-occurrence operator → SFT.

    Operator: normalised word co-occurrence Laplacian.
    Parameters: per-word importance weights.
    Basis: diagonal perturbation per word.

    Example
    -------
    >>> doc = TextAdapter("hello world hello")
    >>> W = doc.kernel
    >>> lam = doc.predict(association_changes)
    """

    def __init__(self, texts: str | list[str],
                 max_words: int = 500, window: int = 5):
        if isinstance(texts, str):
            texts = [texts]
        self.texts = texts
        self.max_words = max_words
        self.window = window
        self._build()

    def _build(self):
        from collections import Counter
        all_tokens = []
        for text in self.texts:
            all_tokens.extend(text.lower().split())
        counter = Counter(all_tokens)
        vocab = [w for w, _ in counter.most_common(self.max_words)]
        n = len(vocab)
        self.vocab = vocab
        self.n_words = n

        word_to_idx = {w: i for i, w in enumerate(vocab)}

        cooc = np.zeros((n, n))
        for text in self.texts:
            tokens = [word_to_idx.get(w, -1) for w in text.lower().split()]
            tokens = [t for t in tokens if t >= 0]
            for i, ti in enumerate(tokens):
                for j in range(max(0, i - self.window),
                               min(len(tokens), i + self.window + 1)):
                    if i != j:
                        cooc[ti, tokens[j]] += 1

        self.cooc = cooc
        deg = np.sum(cooc, axis=1) + 1e-10
        A0 = np.diag(1.0 / np.sqrt(deg)) @ cooc @ np.diag(1.0 / np.sqrt(deg))
        A0 = (A0 + A0.T) / 2

        pairs = []
        for u in range(n):
            for v in range(u, n):
                if cooc[u, v] > 0:
                    pairs.append((u, v))
        max_pairs = min(len(pairs), 200)
        pairs = sorted(pairs, key=lambda uv: -cooc[uv[0], uv[1]])[:max_pairs]

        M = 1 + len(pairs)
        basis_arr = np.zeros((M, n, n))
        basis_arr[0] = np.eye(n)
        for k, (u, v) in enumerate(pairs):
            basis_arr[k + 1, u, v] = 1.0
            basis_arr[k + 1, v, u] = 1.0

        self._family = OperatorFamily(A0, list(basis_arr))


class TimeseriesAdapter(BaseAdapter):
    """
    Timeseries → Hankel/trajectory matrix → SFT (singular spectrum).

    Operator: trajectory (Hankel) matrix H_{i,j} = x[i+j-1].
    Parameters: window coefficients — per-lag weighting.
    Basis: diagonal perturbation per lag.

    Example
    -------
    >>> ts = TimeseriesAdapter(series, window_len=50)
    >>> W = ts.kernel
    >>> lam = ts.predict(lag_weight_changes)
    """

    def __init__(self, series: np.ndarray, window_len: int = 50):
        self.series = np.asarray(series, dtype=np.float64).ravel()
        self.window_len = window_len
        n = len(self.series)
        if self.window_len < 2:
            raise ValueError(f"window_len must be >= 2, got {window_len}")
        if self.window_len >= n:
            raise ValueError(
                f"window_len ({window_len}) must be < series length ({n})"
            )
        self._build()

    def _build(self):
        n = len(self.series)
        L = min(self.window_len, n - self.window_len + 1)
        K = n - L + 1

        H = sliding_window_view(self.series, L)
        H = H.reshape(K, L)
        Cov = (H.T @ H) / K
        Cov = (Cov + Cov.T) / 2
        self.Cov = Cov
        self.L = L

        self._basis = [np.diag(np.eye(L)[lag]) for lag in range(L)]
        self._family = OperatorFamily(Cov, self._basis)


# ═══════════════════════════════════════════════════════════
# VIDEO
# ═══════════════════════════════════════════════════════════

class VideoAdapter(BaseAdapter):
    """
    Video → spatiotemporal patch covariance → SFT.

    Operator: covariance of (T×ps×ps) spatiotemporal patches.
    Parameters: spatiotemporal region gains.
    Basis: region masks in (time, space) flattened coordinates.

    Each patch captures local motion/texture over T consecutive frames
    in a ps×ps spatial window. The operator is the covariance of all such
    patches across the video — a natural spectral summary of video content.

    Example
    -------
    >>> vid = VideoAdapter(frames, patch_t=5, patch_xy=8, n_regions=16)
    >>> W = vid.kernel
    >>> lam = vid.predict(region_changes)
    """

    def __init__(self, frames: np.ndarray,
                 patch_t: int = 5, patch_xy: int = 8,
                 stride_t: int | None = None,
                 stride_xy: int | None = None,
                 n_regions: int = 16):
        frames = np.asarray(frames, dtype=np.float64)
        if frames.ndim == 4 and frames.shape[-1] == 3:
            frames = 0.299 * frames[:, :, :, 0] + \
                    0.587 * frames[:, :, :, 1] + \
                    0.114 * frames[:, :, :, 2]
        elif frames.ndim == 4:
            frames = frames.mean(axis=-1)
        self.frames = frames if frames.ndim == 3 else frames[None]
        self.n_frames_total, self.height, self.width = self.frames.shape
        self.patch_t = patch_t
        self.patch_xy = patch_xy
        self.stride_t = stride_t or max(patch_t // 2, 1)
        self.stride_xy = stride_xy or max(patch_xy // 2, 1)
        self.n_regions = n_regions
        self._build()

    def _build(self):
        nf, h, w = self.frames.shape
        pt, pxy = self.patch_t, self.patch_xy

        if h < pxy or w < pxy or nf < pt:
            pad_frames = np.zeros((max(nf, pt), max(h, pxy), max(w, pxy)))
            pad_frames[:nf, :h, :w] = self.frames
            self.frames = pad_frames
            nf, h, w = pad_frames.shape

        patches_view = sliding_window_view(self.frames, (pt, pxy, pxy))[
            ::self.stride_t, ::self.stride_xy, ::self.stride_xy]
        nt, ni, nj = patches_view.shape[:3]
        patches = patches_view.reshape(nt * ni * nj, pt * pxy * pxy)
        n_patches = len(patches)
        self.n_patches = n_patches

        self.patch_mean = patches.mean(axis=0)
        patches_centered = patches - self.patch_mean[None, :]
        Cov = (patches_centered.T @ patches_centered) / max(n_patches - 1, 1)
        self.Cov = (Cov + Cov.T) / 2

        patch_dim = pt * pxy * pxy
        n_side = int(np.ceil(np.sqrt(self.n_regions)))
        region_t = max(1, pt // n_side)
        region_h = max(1, h // n_side)
        region_w = max(1, w // n_side)

        self._basis = []
        count = 0
        for rt in range(n_side):
            t0 = rt * region_t
            t1 = min(t0 + region_t, pt)
            for ri in range(n_side):
                r0 = ri * region_h
                r1 = min(r0 + region_h, h)
                for rj in range(n_side):
                    if count >= self.n_regions:
                        break
                    c0 = rj * region_w
                    c1 = min(c0 + region_w, w)
                    mask = np.zeros((pt, h, w))
                    mask[t0:t1, r0:r1, c0:c1] = 1.0
                    mask_flat = mask.ravel()[:patch_dim]
                    Bj = np.diag(mask_flat)
                    self._basis.append(Bj)
                    count += 1
                if count >= self.n_regions:
                    break
            if count >= self.n_regions:
                break

        self._family = OperatorFamily(self.Cov, self._basis)


# ═══════════════════════════════════════════════════════════
# VOXEL (3D volumetric data)
# ═══════════════════════════════════════════════════════════

class VoxelAdapter(BaseAdapter):
    """
    3D volume (MRI, CT, seismic) → 3D patch covariance → SFT.

    Operator: covariance of 3D patches (cubes) extracted from the volume.
    Parameters: per-zone density/scattering coefficients.
    Basis: spatial zone masks.

    Example
    -------
    >>> mri = VoxelAdapter(volume, patch_size=4, n_zones=8)
    >>> W = mri.kernel
    >>> lam = mri.predict(zone_changes)
    """

    def __init__(self, volume: np.ndarray,
                 patch_size: int = 4,
                 stride: int | None = None,
                 n_zones: int = 8):
        self.volume = np.asarray(volume, dtype=np.float64)
        if self.volume.ndim != 3:
            raise ValueError("Volume must be 3D (depth × height × width)")
        self.depth, self.height, self.width = self.volume.shape
        self.patch_size = patch_size
        self.stride = stride or max(patch_size // 2, 1)
        self.n_zones = n_zones
        self._build()

    def _build(self):
        d, h, w = self.volume.shape
        ps = self.patch_size
        st = self.stride

        if d < ps or h < ps or w < ps:
            raise ValueError(f"Volume ({d}x{h}x{w}) must be >= patch_size ({ps}) in each dim")

        patches_view = sliding_window_view(self.volume, (ps, ps, ps))[
            ::st, ::st, ::st]
        nd, nh, nw = patches_view.shape[:3]
        n_patches = nd * nh * nw
        patches = patches_view.reshape(n_patches, ps ** 3)

        self.n_patches = n_patches
        self.patch_mean = patches.mean(axis=0)
        patches_centered = patches - self.patch_mean[None, :]
        Cov = (patches_centered.T @ patches_centered) / max(n_patches - 1, 1)
        self.Cov = (Cov + Cov.T) / 2

        n_side = int(np.ceil(self.n_zones ** (1.0 / 3.0)))
        zone_d = max(1, d // n_side)
        zone_h = max(1, h // n_side)
        zone_w = max(1, w // n_side)

        self._basis = []
        count = 0
        mask_indices = np.indices((ps, ps, ps)).reshape(3, -1)
        for zd in range(n_side):
            d0, d1 = zd * zone_d, min((zd + 1) * zone_d, d)
            for zh in range(n_side):
                h0, h1 = zh * zone_h, min((zh + 1) * zone_h, h)
                for zw in range(n_side):
                    if count >= self.n_zones:
                        break
                    w0, w1 = zw * zone_w, min((zw + 1) * zone_w, w)
                    mask = np.zeros((d, h, w))
                    mask[d0:d1, h0:h1, w0:w1] = 1.0
                    safe_d = np.minimum(mask_indices[0], d - 1)
                    safe_h = np.minimum(mask_indices[1], h - 1)
                    safe_w = np.minimum(mask_indices[2], w - 1)
                    mask_flat = mask[safe_d, safe_h, safe_w]
                    self._basis.append(np.diag(mask_flat))
                    count += 1
                if count >= self.n_zones:
                    break
            if count >= self.n_zones:
                break

        self._family = OperatorFamily(self.Cov, self._basis)


# ═══════════════════════════════════════════════════════════
# POINT CLOUD
# ═══════════════════════════════════════════════════════════

class PointCloudAdapter(BaseAdapter):
    """
    3D point cloud → graph Laplacian + density operator → SFT.

    Operator: weighted k-NN graph Laplacian built from pairwise Euclidean
    distances, weighted by local density estimates.
    Parameters: per-point density/scale.
    Basis: per-point diagonal perturbations.

    Example
    -------
    >>> pts = np.random.randn(500, 3)
    >>> pc = PointCloudAdapter(pts, k=15)
    >>> W = pc.kernel
    >>> lam = pc.predict(point_weight_changes)
    """

    def __init__(self, points: np.ndarray, k: int = 15,
                 sigma: float | None = None):
        self.points = np.asarray(points, dtype=np.float64)
        if self.points.ndim != 2:
            raise ValueError("Points must be (n_points, d)")
        self.n_points = self.points.shape[0]
        self.k = max(min(k, self.n_points - 1), 1)
        self.sigma = sigma or np.max(np.std(self.points, axis=0))
        self._build()

    def _build(self):
        N = self.n_points
        X = self.points

        from scipy.spatial import cKDTree
        from scipy import sparse as sp

        tree = cKDTree(X)
        k_query = min(self.k + 1, N)
        dist, idx = tree.query(X, k=k_query)
        if k_query == 1:
            dist = dist.reshape(-1, 1)
            idx = idx.reshape(-1, 1)

        weight = np.exp(-dist**2 / (2.0 * self.sigma ** 2))

        i_idx = np.repeat(np.arange(N), k_query)
        j_idx = idx.ravel()
        mask = i_idx != j_idx
        ii, jj = i_idx[mask], j_idx[mask]
        ww = weight.ravel()[mask]

        adj = sp.coo_matrix((np.ones(len(ii)), (ii, jj)), shape=(N, N)).tocsr()
        adj = adj.maximum(adj.T).tolil()
        adj.setdiag(0)
        self.adjacency = adj.toarray()

        W_sp = sp.coo_matrix((ww, (ii, jj)), shape=(N, N)).tocsr()
        W_sp = W_sp.maximum(W_sp.T)

        row, col = adj.nonzero()
        mask_tri = row < col
        edges = list(zip(row[mask_tri].tolist(), col[mask_tri].tolist()))

        M = len(edges)
        max_params = min(M, max(10, 20000 // N) if N > 0 else 10, 500)
        if M > max_params:
            rng_perm = np.random.default_rng(M)
            edges = [edges[p] for p in rng_perm.permutation(M)[:max_params]]
        else:
            max_params = M

        basis_arr = np.zeros((max_params, N, N))
        edge_arr = np.array(edges, dtype=int) if max_params > 0 else np.empty((0, 2), dtype=int)
        if max_params > 0:
            u_idx = edge_arr[:, 0]
            v_idx = edge_arr[:, 1]
            basis_arr[np.arange(max_params), u_idx, u_idx] = 1.0
            basis_arr[np.arange(max_params), v_idx, v_idx] = 1.0
            basis_arr[np.arange(max_params), u_idx, v_idx] = -1.0
            basis_arr[np.arange(max_params), v_idx, u_idx] = -1.0

        W_dense = W_sp.toarray()
        D = np.diag(np.asarray(W_sp.sum(axis=1)).ravel())
        A0 = D - W_dense
        self._family = OperatorFamily(A0, list(basis_arr) if max_params > 0 else [])


# ═══════════════════════════════════════════════════════════
# MOLECULAR
# ═══════════════════════════════════════════════════════════

class MolecularAdapter(BaseAdapter):
    """
    Molecule → force-field Hessian / bond operator → SFT.

    Operator: weighted adjacency (bond connectivity) + atom-type on diagonal.
    Parameters: per-bond stiffness coefficients.
    Basis: edge Laplacian for each bond.

    Uses Coulomb-like interatomic weighting: w_{ij} ∝ Z_i Z_j / r_{ij}
    for disconnected atoms, standard connectivity for bonded pairs.

    Example
    -------
    >>> mol = MolecularAdapter(positions, atom_types, bonds)
    >>> W = mol.kernel
    >>> lam = mol.predict(bond_parameter_changes)
    """

    def __init__(self, positions: np.ndarray,
                 atom_types: list[str],
                 bonds: list[tuple[int, int]],
                 atom_charges: dict[str, float] | None = None):
        self.positions = np.asarray(positions, dtype=np.float64)
        self.atom_types = atom_types
        self.n_atoms = len(atom_types)
        self.bonds = bonds

        default_charges = {"H": 1.0, "C": 6.0, "N": 7.0, "O": 8.0, "F": 9.0,
                          "Na": 11.0, "Mg": 12.0, "P": 15.0, "S": 16.0,
                          "Cl": 17.0, "K": 19.0, "Ca": 20.0, "Fe": 26.0, "Zn": 30.0}
        self.atom_charges = atom_charges or default_charges
        self._build()

    def _build(self):
        N = self.n_atoms
        X = self.positions
        charges = np.array([self.atom_charges.get(a, 6.0) for a in self.atom_types])

        diff = X[:, None, :] - X[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-10)
        bond_set = set((min(u, v), max(u, v)) for u, v in self.bonds)

        A0 = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                pair = (i, j)
                if pair in bond_set or (j, i) in bond_set:
                    A0[i, j] = A0[j, i] = 1.0
                else:
                    w = charges[i] * charges[j] / dist[i, j]
                    A0[i, j] = A0[j, i] = np.tanh(w * 0.01) * 0.1

        A0[np.diag_indices(N)] = -np.sum(A0, axis=1)

        M = len(self.bonds)
        basis_arr = np.zeros((M, N, N))
        for k, (u, v) in enumerate(self.bonds):
            basis_arr[k, u, u] = 1.0
            basis_arr[k, v, v] = 1.0
            basis_arr[k, u, v] = -1.0
            basis_arr[k, v, u] = -1.0

        self._family = OperatorFamily(A0, list(basis_arr) if M > 0 else [])


# ═══════════════════════════════════════════════════════════
# FINANCIAL
# ═══════════════════════════════════════════════════════════

class FinancialAdapter(BaseAdapter):
    """
    Multi-asset financial data → correlation/covariance operator → SFT.

    Operator: (T, N) returns → N×N covariance matrix.
    Parameters: per-sector or per-asset volatility/weight.
    Basis: block-diagonal sector masks.

    Performs eigenportfolio decomposition: the kernel W tells how
    much each asset's volatility parameter affects each principal
    component (market mode, sector mode, etc.).

    Example
    -------
    >>> fin = FinancialAdapter(returns, sectors=['tech','tech','fin','fin'])
    >>> W = fin.kernel
    >>> lam = fin.predict(volatility_changes)
    """

    def __init__(self, returns: np.ndarray,
                 sectors: list[str] | None = None,
                 asset_names: list[str] | None = None):
        self.returns = np.asarray(returns, dtype=np.float64)
        if self.returns.ndim == 1:
            self.returns = self.returns.reshape(-1, 1)
        self.T, self.N = self.returns.shape
        self.sectors = sectors or ["asset"] * self.N
        self.asset_names = asset_names or [f"A{i}" for i in range(self.N)]
        self._build()

    def _build(self):
        N = self.N
        returns_centered = self.returns - self.returns.mean(axis=0)[None, :]
        Cov = (returns_centered.T @ returns_centered) / max(self.T - 1, 1)
        Cov = (Cov + Cov.T) / 2
        self.Cov = Cov
        self.Corr = Cov / (np.sqrt(np.diag(Cov))[:, None] *
                            np.sqrt(np.diag(Cov))[None, :] + 1e-15)

        unique_sectors = sorted(set(self.sectors))
        sector_to_idx = {s: [i for i, sec in enumerate(self.sectors) if sec == s]
                         for s in unique_sectors}
        M = len(unique_sectors) + N
        basis_arr = np.zeros((M, N, N))

        for k, (sec, indices) in enumerate(sector_to_idx.items()):
            for idx in indices:
                basis_arr[k, idx, idx] = 1.0 / max(len(indices), 1)

        for offset, i in enumerate(range(N)):
            k = len(unique_sectors) + offset
            basis_arr[k, i, i] = 1.0

        self._family = OperatorFamily(self.Corr, list(basis_arr))


# ═══════════════════════════════════════════════════════════
# TABULAR
# ═══════════════════════════════════════════════════════════

class TabularAdapter(BaseAdapter):
    """
    Generic tabular data → feature covariance → SFT.

    Operator: (samples, features) → feature-feature covariance matrix.
    Parameters: per-feature-group and per-feature weights.
    Basis: block-diagonal group masks + per-feature diagonal.

    Works with any rectangular data (numeric). Handles missing values
    via pairwise covariance. Categorical columns should be one-hot encoded.

    Example
    -------
    >>> tab = TabularAdapter(data, feature_groups=['demog','demog','lab','lab','lab'])
    >>> W = tab.kernel
    >>> lam = tab.predict(feature_weight_changes)
    """

    def __init__(self, data: np.ndarray,
                 feature_names: list[str] | None = None,
                 feature_groups: list[str] | None = None):
        self.data = np.asarray(data, dtype=np.float64)
        if self.data.ndim == 1:
            self.data = self.data.reshape(-1, 1)
        self.n_samples, self.n_features = self.data.shape
        self.feature_names = feature_names or [f"f{i}" for i in range(self.n_features)]
        self.feature_groups = feature_groups or ["all"] * self.n_features
        self._build()

    def _build(self):
        D = self.n_features
        data_centered = self.data - np.nanmean(self.data, axis=0)[None, :]
        mask = ~np.isnan(data_centered)

        Cov = np.zeros((D, D))
        for i in range(D):
            for j in range(D):
                valid = mask[:, i] & mask[:, j]
                if valid.sum() > 1:
                    Cov[i, j] = (data_centered[valid, i] *
                                  data_centered[valid, j]).sum() / (valid.sum() - 1)
        Cov = (Cov + Cov.T) / 2
        self.Cov = Cov

        groups = sorted(set(self.feature_groups))
        group_indices = {g: [i for i, fg in enumerate(self.feature_groups) if fg == g]
                          for g in groups}
        M = len(groups) + D
        basis_arr = np.zeros((M, D, D))

        for k, (g, indices) in enumerate(group_indices.items()):
            for idx in indices:
                basis_arr[k, idx, idx] = 1.0 / max(len(indices), 1)

        for offset, i in enumerate(range(D)):
            basis_arr[len(groups) + offset, i, i] = 1.0

        self._family = OperatorFamily(Cov, list(basis_arr))


# ═══════════════════════════════════════════════════════════
# MESH
# ═══════════════════════════════════════════════════════════

class MeshAdapter(BaseAdapter):
    """
    3D triangular mesh → cotangent-weight Laplacian → SFT.

    Operator: cotangent-weight Laplacian derived from mesh geometry.
    Parameters: per-face or per-vertex stiffness.
    Basis: per-edge Laplacian (each edge weight is a parameter).

    The cotangent Laplacian is the standard discrete Laplace-Beltrami
    operator used across geometry processing, shape analysis, and PDE
    solvers on surfaces.

    Example
    -------
    >>> mesh = MeshAdapter(vertices, faces)
    >>> W = mesh.kernel
    >>> lam = mesh.predict(edge_weight_changes)
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=int)
        if self.faces.shape[1] != 3:
            raise ValueError("Faces must be triangles (N_faces, 3)")
        self.n_vertices = self.vertices.shape[0]
        self.n_faces = self.faces.shape[0]
        self._build()

    def _build(self):
        V = self.vertices
        F = self.faces
        Nv = self.n_vertices
        Nf = self.n_faces

        v0 = V[F[:, 0]]
        v1 = V[F[:, 1]]
        v2 = V[F[:, 2]]
        e01 = v1 - v0
        e12 = v2 - v1
        e20 = v0 - v2

        def _cot(a, b):
            dot_val = np.sum(a * b, axis=1)
            cross = np.cross(a, b)
            cross_norm = np.sqrt(np.sum(cross ** 2, axis=1))
            return dot_val / (cross_norm + 1e-15)

        cot0 = _cot(e20, -e12)
        cot1 = _cot(e01, -e20)
        cot2 = _cot(e12, -e01)

        L = np.zeros((Nv, Nv))
        for f_idx in range(Nf):
            i0, i1, i2 = F[f_idx]
            L[i0, i1] -= cot2[f_idx]
            L[i1, i0] -= cot2[f_idx]
            L[i1, i2] -= cot0[f_idx]
            L[i2, i1] -= cot0[f_idx]
            L[i2, i0] -= cot1[f_idx]
            L[i0, i2] -= cot1[f_idx]

        row_sum = np.sum(L, axis=1) + 1e-15
        for i in range(Nv):
            L[i, i] -= row_sum[i]

        L = (L + L.T) / 2
        self.Laplacian = L

        row, col = np.triu(np.abs(L) > 1e-15, 1).nonzero()
        edges = list(zip(row.tolist(), col.tolist()))
        self.n_edges = len(edges)

        M = len(edges)
        basis_arr = np.zeros((M, Nv, Nv))
        for k, (u, v) in enumerate(edges):
            basis_arr[k, u, u] = 1.0
            basis_arr[k, v, v] = 1.0
            basis_arr[k, u, v] = -1.0
            basis_arr[k, v, u] = -1.0

        self._family = OperatorFamily(L, list(basis_arr) if M > 0 else [])
