# Cosmic dynamics — spectral Jacobian of galaxy formation

Compute the exact spectral Jacobian of a gravitational N-body operator.

**Setup:** 100-body gravitational operator. 20 mass parameters. Random initial positions.

**Result:**
- Kernel size: 100×20 — all eigenvalue derivatives in one matrix
- rank(W) = 20/100 = **0.20** → linear clustering regime (low structural complexity)
- `fam.complexity` detects the phase transition to non-linear collapse as rank(W) approaches N

```python
import numpy as np
import sft

# Build gravitational operator from N-body configuration
N = 100; M = 20
rng = np.random.default_rng(42)
positions = rng.standard_normal((N, 3))
# G_ij = 1/|r_i - r_j| — gravitational potential matrix
diff = positions[:, None] - positions[None, :]
dist = np.sqrt(np.sum(diff**2, axis=-1)) + 1e-10
G = 1.0 / dist; np.fill_diagonal(G, 0)
G = (G + G.T) / 2

# Mass perturbations as basis
mass_basis = [np.diag(np.eye(N)[i]) for i in range(M)]

fam = sft.OperatorFamily(G, mass_basis)
print(f"complexity = rank(W)/N = {fam.complexity:.3f}")
print(f"κ(W) = {fam.condition_number():.1f}")
```
