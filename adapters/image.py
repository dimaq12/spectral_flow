from .base import BaseAdapter
from ..core import OperatorFamily
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class ImageAdapter(BaseAdapter):
    """
    Image → patch covariance operator → SFT.

    Operator: covariance matrix of image patches.
    Parameters: per-pixel or per-region brightness/contrast.
    Basis: pixel-region masks.

    Example
    -------
    >>> img = ImageAdapter(pixels, patch_size=8, n_regions=16)
    >>> W = img.kernel
    >>> lam = img.predict(region_changes)
    """

    def __init__(self, pixels: np.ndarray, patch_size: int = 8,
                 stride: int | None = None, n_regions: int = 16):
        self.pixels = np.asarray(pixels, dtype=np.float64)
        if self.pixels.ndim == 3:
            self.pixels = 0.299 * self.pixels[:, :, 0] + \
                         0.587 * self.pixels[:, :, 1] + \
                         0.114 * self.pixels[:, :, 2]
        self.image = self.pixels
        self.patch_size = patch_size
        self.stride = stride or patch_size // 2
        self.n_regions = n_regions
        self._build()

    def _build(self):
        h, w = self.image.shape
        ps = self.patch_size

        if h < ps or w < ps:
            raise ValueError(
                f"Image dimensions ({h}x{w}) must be >= patch_size ({ps}x{ps})"
            )

        patches_view = sliding_window_view(self.image, (ps, ps))[::self.stride, ::self.stride]
        n_i, n_j = patches_view.shape[0], patches_view.shape[1]
        patches = patches_view.reshape(n_i * n_j, ps * ps)
        n_patches = len(patches)
        self.n_patches = n_patches

        self.patch_mean = patches.mean(axis=0)
        patches_centered = patches - self.patch_mean[None, :]
        Cov = (patches_centered.T @ patches_centered) / max(n_patches - 1, 1)
        self.Cov = (Cov + Cov.T) / 2

        n_side = int(np.ceil(np.sqrt(self.n_regions)))
        region_h = np.ceil(h / n_side).astype(int)
        region_w = np.ceil(w / n_side).astype(int)

        self._basis = []
        count = 0
        for ri in range(n_side):
            r0 = ri * region_h
            r1 = min(r0 + region_h, h)
            for rj in range(n_side):
                if count >= self.n_regions:
                    break
                c0 = rj * region_w
                c1 = min(c0 + region_w, w)
                mask = np.zeros((h, w))
                mask[r0:r1, c0:c1] = 1.0

                indices_pi = np.tile(np.arange(ps).reshape(-1, 1), (1, ps)).ravel()
                indices_pj = np.tile(np.arange(ps), ps)
                safe_pi = np.minimum(indices_pi, h - 1)
                safe_pj = np.minimum(indices_pj, w - 1)
                mask_flat = mask[safe_pi, safe_pj]
                Bj = np.diag(mask_flat)
                self._basis.append(Bj)
                count += 1
            if count >= self.n_regions:
                break

        self._family = OperatorFamily(self.Cov, self._basis)
