# Spectral Topology — Berry holonomy and eigenvalue braiding

Walk a closed loop around an exceptional point in parameter space.
Eigenvalues may exchange. Eigenvectors may flip sign.

**Setup:** 2×2 avoided crossing, loop radius 0.4, 60 points.

**Result:**
- Berry holonomy: **−1** — Möbius topology of the eigenvector bundle
- Monodromy: time **908ms** for 60 diagonalizations
- Non-Hermitian EP: eigenvalues swap after 2π loop

```python
import numpy as np
import sft

fam = sft.families.avoided_crossing_2x2(Delta=0.3)
loop = [np.array([0.4*np.cos(t), 0.4*np.sin(t)])
        for t in np.linspace(0, 2*np.pi, 60)]

hol = sft.topology.berry_holonomy(fam, loop, level=1)
# hol = −1 → topological charge. Eigenvector flips sign over 2π.

tracked, swaps = sft.topology.monodromy(fam, loop)
# swaps: list of eigenvalue pairs that exchanged positions
```
