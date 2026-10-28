"""Decode the model's depth channel.

The model predicts depth as normalized disparity in [-1, 1]:  depth_norm = 2 / (1 + d / s) - 1,
with s a per-clip scale. Inverting gives depth up to that unknown scale (s = 1 here), which is
why every depth metric is computed after a scale (or scale + shift) alignment.
"""
import numpy as np


def depth_from_norm(depth_norm: np.ndarray, percentile_clip: float | None = 2.0, eps: float = 1e-3) -> np.ndarray:
    """(T, H, W) normalized disparity -> (T, H, W) relative depth.

    `percentile_clip` floors disparity at its p-th percentile over the whole clip before the
    1/disp inversion, so far/sky pixels stay bounded. 2.0 is the historical default; 10 measured
    better on KITTI, Sintel and ScanNet++.
    """
    disp = (np.asarray(depth_norm, dtype=np.float32) + 1.0) * 0.5
    lo = eps
    if percentile_clip is not None:
        valid = (disp > 0) & np.isfinite(disp)
        if valid.any():
            lo = max(float(np.percentile(disp[valid], percentile_clip)), eps)
    disp = np.clip(disp, lo, 1.0 - eps)
    return (1.0 / disp - 1.0).astype(np.float32)


def encode_depth(depth, scale: float | None = None) -> np.ndarray:
    """(T, H, W) depth (metric or relative; <= 0 marks invalid) -> the model's depth input.

    Training encodes depth as 2 / (1 + d / s) - 1. With cameras on the same clip, `s` must be
    the camera moment scale (ugg.cameras.encode_cameras). Without cameras, `s` is the median
    valid depth, the gauge training used for depth-only data.
    """
    d = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0)
    if scale is None:
        scale = float(np.median(d[valid])) if valid.any() else 1.0
    out = 2.0 / (1.0 + np.clip(d, 0, None) / scale) - 1.0
    return np.where(valid, out, -1.0).astype(np.float32)
