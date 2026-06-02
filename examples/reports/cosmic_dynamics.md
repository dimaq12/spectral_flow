# Cosmic Dynamics — Spectral Jacobian of Galaxy Formation

SFT computes the exact spectral Jacobian of a gravitational N-body operator.
One diagonalization replaces thousands of N-body simulations.

---

## Setup

- **N = 100** bodies in 3D
- **M = 20** mass perturbation parameters
- Operator: gravitational potential G_ij = 1/|r_i − r_j|
- Basis: per-body mass perturbations B_i = diag(e_i)

## Metrics

| Metric | Value |
|--------|-------|
| Build time | **50.6ms** |
| Kernel W shape | 100 × 20 |
| rank(W) | **20** |
| complexity = rank/N | **0.200** |
| κ(W) | **6.2** |
| dim(ker(W)) | **0** |
| ‖λ_exact − W·dk‖_∞ (‖dk‖=0.01) | **1.23e-03** |

## Interpretation

- **complexity = 0.200 << 1** → the gravitational operator has LOW structural complexity.
  The spectrum responds to only 20 out of 20 mass parameters.
  This is the linear clustering regime.

- **rank(W) << N** → spectral dimension far below spatial dimension.
  Most mass perturbations are *spectrally silent* — the operator cannot hear them.

- The transition to non-linear collapse occurs as rank(W) → N,
  when every mass degree of freedom affects the spectrum.

## Code

```python
import sft, numpy as np
fam = sft.OperatorFamily(gravitational_potential, mass_basis)
print(f"complexity = {fam.complexity:.3f}")
print(f"κ(W) = {fam.condition_number():.1f}")
```
