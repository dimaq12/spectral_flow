# Spectral Codec — encode in microseconds, machine-zero roundtrip

Encode any signal through a spectral operator. Decode back. One matrix multiply each way.

**Setup:** Random 64×64 operator family. 16 parameters.

**Result:**
- rank(W) = **16/16** (full rank)
- Roundtrip error: **9.99e-16** — machine precision
- Encode: `y = W·dk`, Decode: `dk ≈ W⁺·y`

```python
import numpy as np
from sft.codec import InstantSpectralCodec
from sft.families import random

fam = random(64, 16, seed=42)
codec = InstantSpectralCodec(fam)
dk = np.linspace(-0.5, 0.5, 16)
y = codec.encode(dk)           # microseconds
dk_hat = codec.decode(y)        # microseconds
err = np.max(np.abs(dk_hat - dk))
# err = 9.99e-16 → machine zero
```
