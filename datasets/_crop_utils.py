"""Aspect-preserving center-crop + resize helpers for geometry-aware datasets.

Usage pattern inside a dataset's `_load_video`:

    from datasets._crop_utils import (
        center_crop_box_for_aspect,
        apply_spatial_crop,
        adjust_K_for_crop_resize,
    )

    # After loading raw frames/depth/valid and reading raw K, BEFORE generating raymaps:
    box = center_crop_box_for_aspect(H_src, W_src, target_H, target_W)
    frames = apply_spatial_crop(frames, box)      # (T, C, H, W) → cropped
    depth  = apply_spatial_crop(depth,  box)
    valid  = apply_spatial_crop(valid,  box)
    K      = adjust_K_for_crop_resize(K, box, target_H, target_W)

    frames = F.interpolate(frames, size=(target_H, target_W), mode="bilinear", align_corners=False)
    depth  = F.interpolate(depth,  size=(target_H, target_W), mode="bilinear", align_corners=False)
    valid  = F.interpolate(valid.float(), size=(target_H, target_W), mode="nearest").bool()

    # Now K, frames, depth, valid are all aligned on the (target_H, target_W) grid,
    # and camera_to_raymap(K, extrinsics, target_H, target_W) produces a raymap whose
    # moments are consistent with the stored depth and extrinsics.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import torch

BoxTTLR = Tuple[int, int, int, int]   # (top, bottom, left, right), right/bottom exclusive


def center_crop_box_for_aspect(
    H_src: int, W_src: int, target_H: int, target_W: int
) -> BoxTTLR:
    """Largest centered crop of (H_src, W_src) with aspect `target_W / target_H`.

    Returns (top, bottom, left, right). Right/bottom exclusive.
    Never enlarges — if the source already matches, returns the full box.
    """
    target_ar = target_W / target_H
    src_ar = W_src / H_src
    if src_ar > target_ar:
        # source too wide → crop width
        new_H = H_src
        new_W = int(round(H_src * target_ar))
    else:
        # source too tall → crop height
        new_W = W_src
        new_H = int(round(W_src / target_ar))
    new_H = min(new_H, H_src)
    new_W = min(new_W, W_src)
    top = (H_src - new_H) // 2
    left = (W_src - new_W) // 2
    return top, top + new_H, left, left + new_W


def apply_spatial_crop(
    tensor: Union[torch.Tensor, np.ndarray], box: BoxTTLR
) -> Union[torch.Tensor, np.ndarray]:
    """Crop the last two dims by `box` (works for torch or numpy)."""
    top, bot, left, right = box
    return tensor[..., top:bot, left:right]


def adjust_K_for_crop_resize(
    K: Union[torch.Tensor, np.ndarray],
    box: BoxTTLR,
    target_H: int,
    target_W: int,
) -> Union[torch.Tensor, np.ndarray]:
    """Update a pinhole intrinsics matrix for (center-crop → resize).

    K may be (3, 3) or leading-batched (..., 3, 3). Returns same shape+dtype.
    Principal point is shifted by the crop offset, then both (fx, cx) and
    (fy, cy) are scaled by the per-axis resize factor.
    """
    top, _bot, left, _right = box
    crop_H = _bot - top
    crop_W = _right - left
    sx = target_W / crop_W
    sy = target_H / crop_H

    if isinstance(K, torch.Tensor):
        K = K.clone()
    else:
        K = np.array(K, copy=True)

    K[..., 0, 2] = (K[..., 0, 2] - left) * sx
    K[..., 1, 2] = (K[..., 1, 2] - top) * sy
    K[..., 0, 0] = K[..., 0, 0] * sx
    K[..., 1, 1] = K[..., 1, 1] * sy
    return K


def aspect_crop_resize_rgb_depth(
    frames: torch.Tensor,          # (T, C, H, W) or (C, H, W)
    depth: torch.Tensor,           # (T, 1, H, W) or None
    valid: torch.Tensor,           # (T, 1, H, W) bool or None
    K: Union[torch.Tensor, np.ndarray],   # (3,3) or (T,3,3)
    target_H: int,
    target_W: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Union[torch.Tensor, np.ndarray], BoxTTLR]:
    """One-shot pass: crop to aspect → resize → adjust K.

    Returns (frames', depth', valid', K', box). `depth` / `valid` may be None
    (returned as None then). `K` intrinsics are assumed in the same pixel frame
    as `frames` prior to the crop.
    """
    import torch.nn.functional as F

    H_src, W_src = frames.shape[-2:]
    box = center_crop_box_for_aspect(H_src, W_src, target_H, target_W)

    frames = apply_spatial_crop(frames, box)
    if depth is not None:
        depth = apply_spatial_crop(depth, box)
    if valid is not None:
        valid = apply_spatial_crop(valid, box)
    K = adjust_K_for_crop_resize(K, box, target_H, target_W)

    if frames.dim() == 3:
        frames = frames.unsqueeze(0)
        frames = F.interpolate(frames.float(), size=(target_H, target_W),
                               mode="bilinear", align_corners=False).squeeze(0)
    else:
        frames = F.interpolate(frames.float(), size=(target_H, target_W),
                               mode="bilinear", align_corners=False)
    if depth is not None:
        depth = F.interpolate(depth.float(), size=(target_H, target_W),
                              mode="bilinear", align_corners=False)
    if valid is not None:
        valid = F.interpolate(valid.float(), size=(target_H, target_W),
                              mode="nearest").bool()
    return frames, depth, valid, K, box
