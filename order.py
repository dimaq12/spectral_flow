"""
sft.order — ORDER/CDF rank, quantile, defect α-spectroscopy.

╔══════════════════════════════════════════════════════════════════════╗
║  INTENT                                                             ║
║  Empirical CDF operations: rank, quantile, defect analysis.         ║
║  UniversalRankOperator — from samples → CDF → predicted ranks.     ║
║  DefectPrecomputedCDF — precompute, O(log n) query.               ║
║  rank_defect_analysis — α-spectroscopy across bin resolutions.     ║
║  carleman_cdf — Gram-Charlier CDF from moments.                    ║
╚══════════════════════════════════════════════════════════════════════╝
║  DEPENDENCIES: numpy, scipy.stats, scipy.special (Hermite)         ║
╚══════════════════════════════════════════════════════════════════════╝
"""
import bisect, numpy as np
from scipy import stats


class UniversalRankOperator:
    """CDF rank estimator from samples.  Build CDF → predict rank via interpolation."""
    def __init__(self, data: np.ndarray, n_bins: int = 100):
        x = np.asarray(data, np.float64).ravel(); self.n = len(x)
        self.sorted_data = np.sort(x); self.n_bins = n_bins
        self._min, self._max = x.min(), x.max()
        self._edges = np.linspace(self._min, self._max, n_bins + 1)
        self._bin_cdf = np.cumsum(np.histogram(x, bins=self._edges)[0]) / self.n

    def rank(self, values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, np.float64).ravel()
        idx = np.clip(np.digitize(v, self._edges) - 1, 0, self.n_bins - 1)
        return (self._bin_cdf[idx] * self.n).astype(int)

    def quantile(self, q): return np.percentile(self.sorted_data, np.asarray(q) * 100)

    def defect_field(self) -> np.ndarray:
        d = np.abs(np.argsort(np.argsort(self.sorted_data)) - self.rank(self.sorted_data))
        return d.astype(float)


class DefectPrecomputedCDF:
    """Precompute sorted array → O(log n) rank/quantile queries."""
    def __init__(self, data: np.ndarray):
        self.sorted_data = np.sort(np.asarray(data, np.float64).ravel())
        self.n = len(self.sorted_data)

    def rank(self, value: float) -> float:
        return min(bisect.bisect_right(self.sorted_data, value), self.n - 1) / max(self.n - 1, 1)

    def quantile(self, q: float) -> float:
        return self.sorted_data[max(0, min(int(q * (self.n - 1)), self.n - 1))]

    @property
    def median(self) -> float: return self.quantile(0.5)
    def iqr(self) -> float: return self.quantile(0.75) - self.quantile(0.25)


def rank_defect_analysis(arr: np.ndarray, bins_list: list[int] | None = None) -> dict:
    """α-spectroscopy: log₂(||D_coarse||/||D_fine||) across bin scales."""
    x = np.asarray(arr, np.float64).ravel()
    bl = bins_list or [10, 20, 40, 80, 160, 320]; bl = [min(b, len(x)) for b in bl if b > 1]
    norms = []
    for b in bl:
        d = UniversalRankOperator(x, n_bins=b).defect_field()
        norms.append(float(np.linalg.norm(d)))
    ln, ld = np.log2(bl), np.log2(np.array(norms) + 1e-15)
    slope, _, r, _, _ = stats.linregress(ln, ld) if len(ln) >= 2 else (0.0, 0, 0, 0, 0)
    return {"alphas": dict(zip(bl, norms)), "slope": float(slope), "r_squared": float(r)}


def carleman_cdf(moments: np.ndarray, n_points: int = 200) -> np.ndarray:
    """Reconstruct CDF from moments via Gram-Charlier (Gaussian + Hermite).  Returns (n,2) [x,CDF]."""
    mu, sigma2 = moments[0], moments[1]; sigma = np.sqrt(max(sigma2, 1e-15))
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, n_points)
    z = (x - mu) / sigma; pdf = stats.norm.pdf(z); cdf = stats.norm.cdf(z)
    from scipy.special import eval_hermite
    if len(moments) >= 3: cdf += moments[2] / sigma**3 * eval_hermite(3, z) / 6.0 * pdf
    if len(moments) >= 4: cdf += (moments[3] / sigma**4 - 3) / 24.0 * eval_hermite(4, z) / 24.0 * pdf
    return np.column_stack([x, np.clip(cdf, 0.0, 1.0)])


class BDSAnalyzer:
    """
    BDS (Born-Dyson Spectrometer): build database of α-fingerprints and
    identify unknown distributions by matching.

    Example
    -------
    >>> bds = BDSAnalyzer()
    >>> bds.register("gauss", np.random.randn(500), bins_list=[10,30,90])
    >>> bds.register("uniform", np.random.uniform(-3,3,500), bins_list=[10,30,90])
    >>> best = bds.identify(unknown_sample)  # → 'gauss'
    """

    def __init__(self):
        self._db: dict[str, np.ndarray] = {}
        self._bins_list: list[int] | None = None

    def register(self, name: str, sample: np.ndarray, bins_list: list[int] | None = None):
        """Store the α-curve (slope per bin) for a named distribution."""
        bins = bins_list or [10, 20, 40, 80, 160]
        result = rank_defect_analysis(sample, bins_list=bins)
        norms = np.array([result["alphas"][b] for b in bins])
        self._db[name] = norms / (np.linalg.norm(norms) + 1e-15)
        if self._bins_list is None:
            self._bins_list = bins

    def identify(self, sample: np.ndarray) -> str:
        """Return the best-matching distribution name by cosine similarity."""
        bins = self._bins_list or [10, 20, 40, 80, 160]
        result = rank_defect_analysis(sample, bins_list=bins)
        query = np.array([result["alphas"][b] for b in bins])
        query = query / (np.linalg.norm(query) + 1e-15)
        best_name, best_sim = "unknown", -1.0
        for name, fp in self._db.items():
            sim = float(np.dot(query, fp))
            if sim > best_sim:
                best_sim, best_name = sim, name
        return best_name

    @property
    def fingerprints(self) -> dict[str, np.ndarray]:
        return dict(self._db)
