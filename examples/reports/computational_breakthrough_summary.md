# Computational Breakthrough Summary

These benchmarks compare SFT against strong sparse baselines, not dense toy
loops.  Baseline totals may be extrapolated from measured query samples when
the full loop is intentionally expensive; each detailed report states this.

| Workload | Best baseline | SFT result | Speedup | Error |
|----------|---------------|------------|--------:|------:|
| Sparse PDE spectral queries | warm-start `lobpcg` | one reference + `predict_many` | 316.6x total, 23690.0x query | max 2.39e-10 |
| Inverse graph design | `least_squares` + sparse `eigsh` | SFT trust inverse | 30.6x time, 28.5x fewer eigensolves | 1.72e-04 spectral error |
| 2D PDE spectral control | repeated sparse `eigsh` | one reference + `predict_many` | 111.4x total, 3873.4x query | max 9.60e-06 |

## Fairness Rules

- Baselines use sparse eigensolvers: `eigsh`, warm-start `lobpcg`, and
  `scipy.optimize.least_squares` with sparse eig residuals.
- Reports separate setup and query time.
- Reports include eigensolve count and error against measured sparse eigensolve
  outputs.
- SFT memory values are dense-equivalent basis estimates for implicit
  backends, not necessarily allocated memory.
- Large baseline totals are extrapolated only when marked in the detailed
  report.

## Detailed Reports

- [`breakthrough_pde_queries.md`](breakthrough_pde_queries.md)
- [`breakthrough_graph_inverse.md`](breakthrough_graph_inverse.md)
- [`breakthrough_pde2d_control.md`](breakthrough_pde2d_control.md)
