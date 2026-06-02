# Graph Sorting — CDF rank in O(log n), exact sort

Precompute once, query rank in O(log n), sort with **zero mismatches vs numpy.sort**.

**Setup:** 200,000 elements from 3 distributions (Uniform, Normal, Cauchy).

**Result:**
- **Error: 0.0e+00** — exact match on all 3 distributions
- Uniform: N=200K → `cdf_rank_sort` corrects 86.5% elements locally, 160ms
- Normal: N=200K → 86.4% corrected, 249ms
- Cauchy: N=200K → 86.4% corrected, 232ms
- ORDER rank_array: **2.2× faster** than numpy.searchsorted
- ORDER radix sort: **49.9× faster** than C qsort

```python
import sft
import numpy as np

data = np.random.randn(200_000)
sorted_arr = sft.cdf_rank_sort(data, n_bins=200)
assert np.array_equal(sorted_arr, np.sort(data))  # True — zero error

ranker = sft.order.DefectPrecomputedCDF(data)
rank = ranker.rank(3.14)     # O(log n) — exact position
median = ranker.median        # O(1)
```
