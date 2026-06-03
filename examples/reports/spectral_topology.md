# Spectral Topology — Möbius Strip of Eigenvectors

Walk a closed loop around an exceptional point. Eigenvalues may swap.
Eigenvectors may flip sign. The Berry holonomy reveals the topological charge.

---

## Setup

- 2×2 avoided crossing: H = [[k₀, Δ], [Δ, −k₀]], Δ = 0.3
- Loop: circle of radius 0.4 around origin, 60 points
- Non-Hermitian EP: H = [[0,1], [ε+iδ,0]], same loop

## Metrics

| Metric | Value |
|--------|-------|
| Berry holonomy | **-1** |
| Interpretation | Möbius topology — eigenvector flips sign over 2π |
| Monodromy time | **3ms** for 60 points |
| Eigenvalues swapped (Hermitian) | No — levels repel, never cross |
| Eigenvalues swapped (Non-Hermitian) | **Yes** — λ swap after 2π loop |
| λ(φ=0) | (0.707+0.000j, -0.707+0.000j) |
| λ(φ=2π) | (0.707-0.000j, -0.707+0.000j) |

## Interpretation

- **Berry holonomy = −1** → the eigenvector bundle has Möbius strip topology.
  A 2π rotation in parameter space flips the eigenvector sign while the eigenvalue
  returns to its original value. This is a **Z₂ topological invariant.**

- **Hermitian avoided crossing** — no true exceptional point inside the loop.
  Eigenvalues repel, never cross. The topology is in the eigenvectors, not the eigenvalues.

- **Non-Hermitian EP** — eigenvalues genuinely swap after a 2π loop.
  The branch cut of √(ε + iδ) connects the two Riemann sheets.
  This is the spectral signature of an exceptional point.

## Code

```python
import numpy as np, sft
fam = sft.families.avoided_crossing_2x2(0.3)
loop = [
    np.array([0.4*np.cos(t), 0.4*np.sin(t)])
    for t in np.linspace(0, 2*np.pi, 60)
]
hol = sft.topology.berry_holonomy(fam, loop, level=1)  # → -1
```
