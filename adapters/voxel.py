from .base import BaseAdapter
from ..core import OperatorFamily
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class VoxelAdapter(BaseAdapter):
    """
    3D volume (MRI, CT, seismic) → 3D patch covariance → SFT.

    Operator: covariance of 3D patches (cubes) extracted from the volume.
    Parameters: per-zone density/scattering coefficients.
    Basis: spatial zone masks.

    Example
    -------
    >>> mri = VoxelAdapter(volume, patch_size=4, n_zones=8)
    >>> W = mri.kernel
    >>> lam = mri.predict(zone_changes)
    """

    def __init__(self, volume: np.ndarray,
                 patch_size: int = 4,
                 stride: int | None = None,
                 n_zones: int = 8):
        self.volume = np.asarray(volume, dtype=np.float64)
        if self.volume.ndim != 3:
            raise ValueError("Volume must be 3D (depth × height × width)")
        self.depth, self.height, self.width = self.volume.shape
        self.patch_size = patch_size
        self.stride = stride or max(patch_size // 2, 1)
        self.n_zones = n_zones
        self._build()

    def _build(self):
        d, h, w = self.volume.shape
        ps = self.patch_size
        st = self.stride

        if d < ps or h < ps or w < ps:
            raise ValueError(f"Volume ({d}x{h}x{w}) must be >= patch_size ({ps}) in each dim")

        patches_view = sliding_window_view(self.volume, (ps, ps, ps))[
            ::st, ::st, ::st]
        nd, nh, nw = patches_view.shape[:3]
        n_patches = nd * nh * nw
        patches = patches_view.reshape(n_patches, ps ** 3)

        self.n_patches = n_patches
        self.patch_mean = patches.mean(axis=0)
        patches_centered = patches - self.patch_mean[None, :]
        Cov = (patches_centered.T @ patches_centered) / max(n_patches - 1, 1)
        self.Cov = (Cov + Cov.T) / 2

        n_side = int(np.ceil(self.n_zones ** (1.0 / 3.0)))
        zone_d = max(1, d // n_side)
        zone_h = max(1, h // n_side)
        zone_w = max(1, w // n_side)

        self._basis = []
        count = 0
        mask_indices = np.indices((ps, ps, ps)).reshape(3, -1)
        for zd in range(n_side):
            d0, d1 = zd * zone_d, min((zd + 1) * zone_d, d)
            for zh in range(n_side):
                h0, h1 = zh * zone_h, min((zh + 1) * zone_h, h)
                for zw in range(n_side):
                    if count >= self.n_zones:
                        break
                    w0, w1 = zw * zone_w, min((zw + 1) * zone_w, w)
                    mask = np.zeros((d, h, w))
                    mask[d0:d1, h0:h1, w0:w1] = 1.0
                    safe_d = np.minimum(mask_indices[0], d - 1)
                    safe_h = np.minimum(mask_indices[1], h - 1)
                    safe_w = np.minimum(mask_indices[2], w - 1)
                    mask_flat = mask[safe_d, safe_h, safe_w]
                    self._basis.append(np.diag(mask_flat))
                    count += 1
                if count >= self.n_zones:
                    break
            if count >= self.n_zones:
                break

        self._family = OperatorFamily(self.Cov, self._basis)
