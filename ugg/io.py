"""Video input: read frames, pick a legal clip length, center-crop and resize."""
import glob
import os
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F

from datasets._crop_utils import apply_spatial_crop, center_crop_box_for_aspect

IMG_EXTS = (".png", ".jpg", ".jpeg")


def legal_num_frames(n: int, cap: int = 50) -> int:
    """Largest M <= min(n, cap) with M % 4 == 2 (the first/last-frame VAE chunking)."""
    m = min(int(n), int(cap))
    while m > 2 and m % 4 != 2:
        m -= 1
    if m < 2:
        raise ValueError(f"need at least 2 frames, got {n}")
    return m


def select_frames(n_src: int, m: int, mode: str = "contig_first") -> np.ndarray:
    """Indices of the m frames the model sees. contig_first = first m frames (the protocol);
    uniform = m frames spread evenly over the whole video, endpoints included."""
    if mode == "contig_first":
        return np.arange(m, dtype=np.int64)
    if mode == "uniform":
        return np.unique(np.linspace(0, n_src - 1, m).round().astype(np.int64))
    raise ValueError(f"unknown frame mode {mode!r} (contig_first | uniform)")


def read_frames(path) -> np.ndarray:
    """A video file or a directory of images -> (N, H, W, 3) uint8."""
    path = Path(path)
    if path.is_dir():
        files = sorted(f for f in glob.glob(str(path / "*")) if f.lower().endswith(IMG_EXTS))
        if not files:
            raise FileNotFoundError(f"no {IMG_EXTS} images in {path}")
        return np.stack([np.asarray(iio.imread(f))[..., :3] for f in files])
    if not path.exists():
        raise FileNotFoundError(path)
    return np.asarray(iio.imread(path, plugin="pyav"))[..., :3]


def center_crop_resize(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """(T, 3, H, W) in [0, 1] -> largest centered crop with the target aspect, resized."""
    box = center_crop_box_for_aspect(*frames.shape[-2:], height, width)
    frames = apply_spatial_crop(frames, box)
    return F.interpolate(frames, size=(height, width), mode="bilinear", align_corners=False)


def prepare_video(video, height: int = 288, width: int = 384, max_frames: int = 50,
                  frame_mode: str = "contig_first"):
    """Anything video-like -> ((M, 3, height, width) float in [-1, 1], frame indices).

    `video` is a path (file or image folder), a (N, H, W, 3) uint8 array, or a (N, 3, H, W)
    float tensor in [0, 1].
    """
    if isinstance(video, (str, os.PathLike)):
        video = read_frames(video)
    if isinstance(video, np.ndarray):
        video = torch.from_numpy(np.ascontiguousarray(video)).permute(0, 3, 1, 2).float() / 255.0
    video = video.float()
    idx = select_frames(len(video), legal_num_frames(len(video), max_frames), frame_mode)
    frames = center_crop_resize(video[torch.from_numpy(idx)], height, width)
    return (frames * 2 - 1).clamp(-1, 1), idx
