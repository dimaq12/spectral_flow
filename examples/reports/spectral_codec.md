# Spectral Codec — Machine-Zero Roundtrip

Encode any signal through a spectral operator. Decode back.
One matrix multiply each way. **Roundtrip error at machine precision.**

---

## Setup

- Operator: 64×64 random symmetric, 16 parameters
- Encode: y = W·dk (spectral response)
- Decode: dk ≈ W⁺·y (pseudoinverse reconstruction)

## Metrics

| Metric | Value |
|--------|-------|
| Operator size | 64 × 64, M = 16 |
| rank(W) | **16** / 16 |
| Encode time | **0μs** |
| Decode time | **0μs** |
| Roundtrip error ‖dk_hat − dk‖_∞ | **1.0e-15** |

## Interpretation

- **1.0e-15** — this is **machine precision**. The roundtrip is numerically exact.
  W⁺·W = I to within floating-point error for full-rank W.

- Encode + decode together take ~**0μs** — two matrix-vector products.
  Compare to eigh which takes **~2000μs** for the same matrix.

- For structured operators with rank(W) < M, the codec has inherent compression:
  only rank(W) dimensions carry information. See `codec.capacity`.

## Code

```python
from sft.codec import InstantSpectralCodec
from sft.families import random
import numpy as np

fam = random(64, 16, seed=42)
codec = InstantSpectralCodec(fam)
dk = np.linspace(-0.5, 0.5, 16)
y = codec.encode(dk)
dk_hat = codec.decode(y)
err = np.max(np.abs(dk_hat - dk))  # → 1.0e-15
```
