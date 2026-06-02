"""
sft.streaming — StreamingCDF, StreamingOrderOnline.

╔══════════════════════════════════════════════════════════════════════╗
║  INTENT                                                             ║
║  Online CDF approximation via reservoir sampling.                   ║
║  StreamingCDF: add values → CDF/quantile via sorted buffer.         ║
║  StreamingOrderOnline: rank/quantile with insert+search.            ║
╚══════════════════════════════════════════════════════════════════════╝
"""
import bisect, numpy as np


class StreamingCDF:
    """Reservoir-sampled CDF.  O(capacity) memory, O(log capacity) queries."""
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity; self._buffer = []; self._count = 0
        self._rng = np.random.default_rng()

    def add(self, value: float) -> None:
        self._count += 1
        if len(self._buffer) < self.capacity: bisect.insort(self._buffer, value)
        else:
            j = self._rng.integers(0, self._count)
            if j < self.capacity:
                self._buffer.pop(j % self.capacity); bisect.insort(self._buffer, value)

    def cdf(self, x: float) -> float:
        return float(bisect.bisect_right(self._buffer, x) / len(self._buffer)) if self._buffer else 0.0

    def quantile(self, q: float) -> float:
        if not self._buffer: return 0.0
        q = max(0.0, min(1.0, q)); return self._buffer[int(q * (len(self._buffer) - 1))]

    @property
    def median(self) -> float: return self.quantile(0.5)


class StreamingOrderOnline:
    """Online ORDER: insert, rank O(log n), quantile O(1)."""
    def __init__(self, capacity: int = 5000):
        self._cdf = StreamingCDF(capacity); self._data = []

    def insert(self, value: float) -> None:
        self._cdf.add(value); bisect.insort(self._data, value)

    def rank(self, value: float) -> float:
        return float(bisect.bisect_left(self._data, value) / len(self._data)) if self._data else 0.0

    def quantile(self, q: float) -> float: return self._cdf.quantile(q)

    @property
    def size(self) -> int: return len(self._data)
