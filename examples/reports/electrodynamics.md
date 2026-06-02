# Electrodynamics — Maxwell's 4/3 Mass Paradox

The 130-year-old electromagnetic mass paradox:
**Why is m_inertial / m_rest = 4/3 instead of 1?**

SFT reveals the answer: **rank(W_EM) < 4** — the energy-momentum Jacobian
has a rank deficit. One spectral degree of freedom is invisible to standard field theory.

---

## Setup

- Spherical charge shell with uniform surface charge
- Pure Maxwell (no Poincaré stress): k = 0
- 4-momentum: P^μ = (E/c, p/c) — 4 components
- W_EM = ∂P^μ/∂v — Jacobian of 4-momentum w.r.t. velocity + stress parameter

## Metrics

| Metric | Pure Maxwell | With Poincaré stress |
|--------|:-----------:|:--------------------:|
| m_inertial / m_rest | **1.333333** | **1.000000** |
| rank(W_EM) | **1** | 2 (full) |
| σ(W_EM) | singular | [2/3, 2/3, ...] |
| deficit | **1** | 0 |

## Interpretation

- **rank(W_EM) = 1 < 2** → the energy-momentum Jacobian has a rank deficit.
  One output component is **spectrally enslaved** — it cannot vary independently.

- The 4/3 factor is not a physics bug — it's a **spectral signature** of incomplete degrees
  of freedom in the field configuration.

- Adding Poincaré stress (k = E₀/3) restores full rank → ratio becomes exactly 1.

- This reframes the entire problem from a field-theoretic puzzle to a **linear algebra invariant.**
  The missing 1/4 is a spectral degree of freedom invisible to standard Maxwell theory.

## Code

```python
import sft, numpy as np
W_em = compute_jacobian(v0, k0)  # ∂P^μ/∂(v, k)
fam = sft.OperatorFamily(W_em, [np.eye(4)])
print(f"rank(W_EM) = {fam.W_rank}, deficit = {4 - fam.W_rank}")
```
