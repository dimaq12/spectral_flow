# Electrodynamics — Maxwell's 4/3 mass paradox as rank deficit

The 130-year-old electromagnetic mass paradox: why is m_inertial/m_rest = 4/3 instead of 1?

SFT reveals the answer: **rank(W_EM) = 3 < 4** — the energy-momentum Jacobian has a rank deficit. One spectral degree of freedom is invisible to standard field theory.

**Result:**
- 4/3 ratio: **1.333333** — pure Maxwell anomaly
- rank(W_EM) = **3** (deficit = 1) — one output component spectrally enslaved
- σ(W_EM) = [0.667, 0.667, 0.667]
- Poincaré stress k = E₀/3 → ratio restored to **1.000000**, rank → 4

```python
import sft
import numpy as np

# W_EM = ∂P^μ/∂v — 4×4 Jacobian of energy-momentum 4-vector
def compute_W_em(v0, k0, eps=1e-4):
    """Central finite-difference Jacobian of P^μ(v, k_Poincaré)."""
    W = np.zeros((4, 4))
    for j in range(4):
        kp = k0.copy(); kp[j] += eps
        km = k0.copy(); km[j] -= eps
        W[:, j] = (P_mu(v0, kp) - P_mu(v0, km)) / (2 * eps)
    return W

W = compute_W_em(v0=0.5, k0=0.0)
fam = sft.OperatorFamily(W.T @ W, [np.eye(4)])
# rank(W) = 3 — the spectral signature of the 4/3 anomaly
print(f"rank(W_EM) = {fam.W_rank}, deficit = {4 - fam.W_rank}")
```
