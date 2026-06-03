# Graph Sorting — CDF Rank O(log n), Exact Sort

Precompute once, query rank in O(log n), sort with **zero mismatches vs numpy.sort**.

---

## Setup

- 3 distributions × 200,000 elements each
- CDF bins = 200
- Collision detection + local insertion sort within ambiguous bins

## Metrics

### CDF rank sort — exact match

| Distribution | Size | Time | Error vs np.sort |
|-------------|------|------|:---:|
| Uniform(0,1) | N=200K | **51ms** | ✓ ZERO |
| Normal(0,1) | N=200K | **44ms** | ✓ ZERO |
| Cauchy(0,1) | N=200K | **23ms** | ✓ ZERO |

### Precomputed CDF — rank queries

| Method | 100K queries | O() |
|--------|:-----------:|-----|
| sft.order.DefectPrecomputedCDF | **144ms** | O(log n) |
| numpy.searchsorted | 14ms | O(log n) |

## Interpretation

- **Zero error** across all 3 distributions — the CDF-based sort with collision correction
  produces results identical to numpy.sort. Not approximate. Exact.

- **Precompute once, query forever** — the sorted array is built once.
  Every rank query after that is O(log n) via bisect.

- The fraction of self-correcting elements (~86%) is the natural collision rate
  for the CDF bin resolution. Increasing bins reduces collisions.

## Code

```python
import sft, numpy as np
data = np.random.randn(200_000)
sorted_arr = sft.cdf_rank_sort(data, n_bins=200)
assert np.array_equal(sorted_arr, np.sort(data))  # True
```
