from .base import BaseAdapter
from ..core import OperatorFamily
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class VideoAdapter(BaseAdapter):
    """
    Video → spatiotemporal patch covariance → SFT.

    Operator: covariance of (T×ps×ps) spatiotemporal patches.
    Parameters: spatiotemporal region gains.
    Basis: region masks in (time, space) flattened coordinates.

    Each patch captures local motion/texture over T consecutive frames
    in a ps×ps spatial window. The operator is the covariance of all such
    patches across the video — a natural spectral summary of video content.

    Example
    -------
    >>> vid = VideoAdapter(frames, patch_t=5, patch_xy=8, n_regions=16)
    >>> W = vid.kernel
    >>> lam = vid.predict(region_changes)
    """

    def __init__(self, frames: np.ndarray,
                 patch_t: int = 5, patch_xy: int = 8,
                 stride_t: int | None = None,
                 stride_xy: int | None = None,
                 n_regions: int = 16):
        frames = np.asarray(frames, dtype=np.float64)
        if frames.ndim == 4 and frames.shape[-1] == 3:
            frames = 0.299 * frames[:, :, :, 0] + \
                    0.587 * frames[:, :, :, 1] + \
                    0.114 * frames[:, :, :, 2]
        elif frames.ndim == 4:
            frames = frames.mean(axis=-1)
        self.frames = frames if frames.ndim == 3 else frames[None]
        self.n_frames_total, self.height, self.width = self.frames.shape
        self.patch_t = patch_t
        self.patch_xy = patch_xy
        self.stride_t = stride_t or max(patch_t // 2, 1)
        self.stride_xy = stride_xy or max(patch_xy // 2, 1)
        self.n_regions = n_regions
        self._build()

    def _build(self):
        nf, h, w = self.frames.shape
        pt, pxy = self.patch_t, self.patch_xy

        if h < pxy or w < pxy or nf < pt:
            pad_frames = np.zeros((max(nf, pt), max(h, pxy), max(w, pxy)))
            pad_frames[:nf, :h, :w] = self.frames
            self.frames = pad_frames
            nf, h, w = pad_frames.shape

        patches_view = sliding_window_view(self.frames, (pt, pxy, pxy))[
            ::self.stride_t, ::self.stride_xy, ::self.stride_xy]
        nt, ni, nj = patches_view.shape[:3]
        patches = patches_view.reshape(nt * ni * nj, pt * pxy * pxy)
        n_patches = len(patches)
        self.n_patches = n_patches

        self.patch_mean = patches.mean(axis=0)
        patches_centered = patches - self.patch_mean[None, :]
        Cov = (patches_centered.T @ patches_centered) / max(n_patches - 1, 1)
        self.Cov = (Cov + Cov.T) / 2

        patch_dim = pt * pxy * pxy
        n_side = int(np.ceil(np.sqrt(self.n_regions)))
        region_t = max(1, pt // n_side)
        region_h = max(1, h // n_side)
        region_w = max(1, w // n_side)

        self._basis = []
        count = 0
        for rt in range(n_side):
            t0 = rt * region_t
            t1 = min(t0 + region_t, pt)
            for ri in range(n_side):
                r0 = ri * region_h
                r1 = min(r0 + region_h, h)
                for rj in range(n_side):
                    if count >= self.n_regions:
                        break
                    c0 = rj * region_w
                    c1 = min(c0 + region_w, w)
                    mask = np.zeros((pt, h, w))
                    mask[t0:t1, r0:r1, c0:c1] = 1.0
                    mask_flat = mask.ravel()[:patch_dim]
                    Bj = np.diag(mask_flat)
                    self._basis.append(Bj)
                    count += 1
                if count >= self.n_regions:
                    break
            if count >= self.n_regions:
                break

        self._family = OperatorFamily(self.Cov, self._basis)
