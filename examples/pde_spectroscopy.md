# PDE Defect Spectroscopy — machine-zero spectral recovery

Recover the EXACT operator spectrum from a black-box PDE solver using only 3 resolution runs.

**Setup:** Fractional heat equation ∂_t u = −(−Δ)^{γ/2} u. Backward Euler at n₀=16, 2n₀=32, 4n₀=64.

**Result:**
- Per-mode ratio: **1.0000 → machine zero**
- α-spectroscopy on 6 distributions: Normal(α=1.22), Cauchy(α=1.42), Uniform(α=1.77), Bimodal(α=2.19), Exponential(α=1.24), Power-law(α=1.48)
- Heat and Kuramoto-Sivashinsky fail — mean Δ% up to 10¹⁵%

```python
import sft

# Defect signal from 3 resolutions
alphas = sft.rank_defect_analysis(solver_output, bins_list=[8, 16, 32, 64])
print(f"α = {alphas['slope']:.3f}  r² = {alphas['r_squared']:.4f}")
# α determines the spectral modality: F(z) ∝ z^{α−1}
```
