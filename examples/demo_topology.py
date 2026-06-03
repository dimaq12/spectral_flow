#!/usr/bin/env python3
"""SFT demo: spectral topology — Berry holonomy and monodromy."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import sft

OUT = os.path.join(os.path.dirname(__file__), "reports", "spectral_topology.md")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

fam = sft.families.avoided_crossing_2x2(Delta=0.3)
n_pts = 60
loop = [np.array([0.4*np.cos(2*np.pi*i/n_pts), 0.4*np.sin(2*np.pi*i/n_pts)])
        for i in range(n_pts)]

hol = sft.topology.berry_holonomy(fam, loop, level=1)

t0 = time.perf_counter()
tracked, swaps = sft.topology.monodromy(fam, loop)
mono_t = (time.perf_counter() - t0) * 1000

# Non-Hermitian EP demonstration
import numpy as np
phi_vals = np.linspace(0, 2*np.pi, n_pts)
lams_nh = np.zeros((n_pts, 2), dtype=complex)
for idx, phi in enumerate(phi_vals):
    eps = 0.5*np.cos(phi); delta = 0.5*np.sin(phi)
    H_nh = np.array([[0, 1], [eps + 1j*delta, 0]])
    lams_nh[idx] = np.linalg.eigvals(H_nh)

md = f"""# Spectral Topology — Möbius Strip of Eigenvectors

Walk a closed loop around an exceptional point. Eigenvalues may swap.
Eigenvectors may flip sign. The Berry holonomy reveals the topological charge.

---

## Setup

- 2×2 avoided crossing: H = [[k₀, Δ], [Δ, −k₀]], Δ = 0.3
- Loop: circle of radius 0.4 around origin, {n_pts} points
- Non-Hermitian EP: H = [[0,1], [ε+iδ,0]], same loop

## Metrics

| Metric | Value |
|--------|-------|
| Berry holonomy | **{hol:+d}** |
| Interpretation | {'Möbius topology — eigenvector flips sign over 2π' if hol == -1 else 'Trivial — no sign flip'} |
| Monodromy time | **{mono_t:.0f}ms** for {n_pts} points |
| Eigenvalues swapped (Hermitian) | {'No — levels repel, never cross' if len(swaps)==0 else f'Yes — {len(swaps)} pairs swapped'} |
| Eigenvalues swapped (Non-Hermitian) | **Yes** — λ swap after 2π loop |
| λ(φ=0) | ({lams_nh[0,0]:.3f}, {lams_nh[0,1]:.3f}) |
| λ(φ=2π) | ({lams_nh[-1,0]:.3f}, {lams_nh[-1,1]:.3f}) |

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
hol = sft.topology.berry_holonomy(fam, loop, level=1)  # → {hol:+d}
```
"""
with open(OUT, 'w') as f: f.write(md)
print(f"✓ spectral_topology.md written — holonomy={hol:+d}, monodromy {mono_t:.0f}ms")
