"""Spring dataset: RGB + GT disparity(.dsp5) -> metric depth + per-frame cameras.

High-res (1920x1080) Blender-rendered stereo benchmark (Mehl et al., CVPR'23). We use
the LEFT view only: frame_left/*.png + disp1_left/*.dsp5 + cam_data/{intrinsics,
extrinsics}.txt. This is exactly the subset Geo4D re-hosts for training.

Geometry is made IDENTICAL to TartanAir (datasets/tartanair.py) so the two mix cleanly:
the geometry / normalization helpers (first-frame-identity, raymap generation, mutual
scale, depth->modified-disparity) are inherited verbatim from TartanAirDataset. Only the
record walk, file IO and camera convention differ.

Conventions (verified by reprojection, see datasets/geometry_verification/spring/;
median frame-to-frame reprojection depth error ~0.05% on camera-moving sequences):
- Intrinsics: per-frame pinhole `fx fy cx cy` at full-HD (cx=960, cy=540). Some scenes
  zoom, so K varies per frame.
- Depth: Z = fx * B / disp, baseline B = 0.065 m. disp1_left is stored at 2x resolution
  (2160x3840) with values in HD pixels -> subsample [::2,::2] to (1080,1920). NaN/<=0
  (sky) and Z >= FAR_PLANE are invalid.
- Poses: extrinsics.txt is a 4x4 WORLD-TO-CAMERA matrix already in OpenCV optical axes
  (x-right, y-down, z-forward). camera-to-world c2w = inv(E) directly -> SAME output
  frame as TartanAir (no NED permutation needed).
- Output depths / raymaps / scaling: inherited from TartanAir (modified disparity in
  [-1,1], Plucker raymaps, per-clip mutual scale with MIN_SCALE=1.0).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import imageio.v3 as iio
import numpy as np
from ._frame_io import map_frames
import torch
import torch.nn.functional as F

from ._crop_utils import (
    adjust_K_for_crop_resize,
    apply_spatial_crop,
    center_crop_box_for_aspect,
)
from .tartanair import TartanAirDataset, INVALID_DEPTH_SENTINEL

RGB_H, RGB_W = 1080, 1920
BASELINE_M = 0.065                   # Spring orthoparallel stereo baseline (FAQ: always 0.065 m)
FAR_PLANE_M = 1000.0                 # depth >= this (tiny disparity / sky) -> invalid
MIN_DISP = 1e-3                      # disparity at/below this -> invalid (sky, NaN)


class SpringDataset(TartanAirDataset):
    """One record per Spring sequence (left view). Reuses TartanAir geometry verbatim."""

    SCENE_ID_KEY = "seq_id"   # record["seq_id"] matches captions.csv scene_id

    def __init__(self, *args, **kwargs):
        # per-sequence caches: intrinsics (T,4) and extrinsics (T,4,4)
        self._intr_cache: Dict[str, np.ndarray] = {}
        self._extr_cache: Dict[str, np.ndarray] = {}
        super().__init__(*args, **kwargs)

    # ---- Records ----

    def _load_records(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for seq_dir in sorted(p for p in self.data_root.iterdir() if p.is_dir()):
            rgb_dir = seq_dir / "frame_left"
            disp_dir = seq_dir / "disp1_left"
            cam_dir = seq_dir / "cam_data"
            if not (rgb_dir.is_dir() and disp_dir.is_dir()
                    and (cam_dir / "intrinsics.txt").exists()
                    and (cam_dir / "extrinsics.txt").exists()):
                continue
            n_frames = sum(1 for _ in rgb_dir.glob("frame_left_*.png"))
            if n_frames < 2:
                continue
            records.append({
                "seq_id": f"spring_{seq_dir.name}",
                "traj_dir": str(seq_dir),
                "env": seq_dir.name,
                "difficulty": "",
                "n_frames": n_frames,
                "fps": int(getattr(self.cfg, "fps_source", 10)),
                "height": RGB_H,
                "width": RGB_W,
                "caption": "",
            })

        if not records:
            raise FileNotFoundError(f"No Spring sequences found under {self.data_root}")

        if not self.filtering.disable:
            pre = len(records)
            records = [r for r in records if self._filter_record(r)]
            print(f"Spring: filtered {pre - len(records)} from {pre}, kept {len(records)}")
        else:
            print(f"Spring: loaded {len(records)} sequences from {self.data_root}")

        random.Random(0).shuffle(records)

        # train / holdout split by sequence (records already shuffled deterministically)
        if self.split != "all":
            n_test = int(len(records) * self.test_percentage)
            if self.split == "training":
                records = records[n_test:] if n_test > 0 else records
            else:  # validation / test
                records = records[:n_test] if n_test > 0 else records[:1]

        records = [self.preprocess_record(r) for r in records]
        return records

    # ---- Per-sequence IO ----

    def _load_intr(self, seq_dir: str) -> np.ndarray:
        if seq_dir not in self._intr_cache:
            arr = np.loadtxt(Path(seq_dir) / "cam_data" / "intrinsics.txt")
            self._intr_cache[seq_dir] = arr[None] if arr.ndim == 1 else arr
        return self._intr_cache[seq_dir]

    def _load_extr(self, seq_dir: str) -> np.ndarray:
        if seq_dir not in self._extr_cache:
            arr = np.loadtxt(Path(seq_dir) / "cam_data" / "extrinsics.txt")
            arr = arr[None] if arr.ndim == 1 else arr
            self._extr_cache[seq_dir] = arr.reshape(-1, 4, 4)
        return self._extr_cache[seq_dir]

    @staticmethod
    def _read_rgb_frames(rgb_dir: Path, frame_ids: np.ndarray) -> np.ndarray:
        def _one(fid):
            p = rgb_dir / f"frame_left_{int(fid) + 1:04d}.png"      # files are 1-indexed
            return np.asarray(iio.imread(str(p)))[..., :3]
        return np.stack(map_frames(_one, frame_ids))

    def _read_depth_frames(
        self, disp_dir: Path, intr: np.ndarray, frame_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        def _one(fid):
            p = disp_dir / f"disp1_left_{int(fid) + 1:04d}.dsp5"
            # each call opens its OWN read-only handle, so this is safe to thread even
            # though HDF5 serialises internally on a non-threadsafe build
            with h5py.File(str(p), "r") as f:
                disp = np.asarray(f["disparity"]).astype(np.float32)  # (2160,3840) HD-pixel disparity
            disp = disp[::2, ::2]                                      # -> (1080,1920)
            fx = float(intr[int(fid), 0])
            valid = np.isfinite(disp) & (disp > MIN_DISP)
            d = np.where(valid, fx * BASELINE_M / np.where(valid, disp, 1.0), 0.0).astype(np.float32)
            valid = valid & (d > 0.0) & (d < FAR_PLANE_M)
            return np.where(valid, d, 0.0).astype(np.float32), valid
        out = map_frames(_one, frame_ids)
        return np.stack([o[0] for o in out]), np.stack([o[1] for o in out])

    def _load_cameras(
        self, record: Dict[str, Any], frame_ids: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        seq_dir = record["traj_dir"]
        intr = self._load_intr(seq_dir)
        extr = self._load_extr(seq_dir)
        K_list, r_list, t_list = [], [], []
        for fid in frame_ids:
            fx, fy, cx, cy = intr[int(fid)]
            K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
            c2w = np.linalg.inv(extr[int(fid)].astype(np.float64))     # E is world-to-camera (OpenCV)
            K_list.append(K)
            r_list.append(c2w[:3, :3].copy())
            t_list.append(c2w[:3, 3].copy())
        return K_list, r_list, t_list

    # ---- Per-clip assembly (mirrors TartanAir._load_video; Spring IO + per-frame K) ----

    def _load_video(
        self, record: Dict[str, Any]
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, float,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        seq_dir = Path(record["traj_dir"])
        rgb_dir = seq_dir / "frame_left"
        disp_dir = seq_dir / "disp1_left"
        intr = self._load_intr(str(seq_dir))

        temporal_indices = self._temporal_sample(record["n_frames"], record["fps"])

        # --- RGB ---
        rgb_np = self._read_rgb_frames(rgb_dir, temporal_indices)
        frames = torch.from_numpy(rgb_np).float().permute(0, 3, 1, 2).contiguous() / 255.0

        # --- Depth (metric, from disparity) ---
        depth_np, valid_np = self._read_depth_frames(disp_dir, intr, temporal_indices)
        d_m = torch.from_numpy(depth_np).unsqueeze(1).float()
        v = torch.from_numpy(valid_np).unsqueeze(1).bool()

        # --- Cameras (per-frame K at source res; c2w in OpenCV optical frame) ---
        K_list, r_list, t_list = self._load_cameras(record, temporal_indices)

        # --- Aspect-preserving center-crop + resize to (self.height, self.width) ---
        H_src, W_src = frames.shape[-2:]
        box = center_crop_box_for_aspect(H_src, W_src, self.height, self.width)
        frames = apply_spatial_crop(frames, box)
        d_m = apply_spatial_crop(d_m, box)
        v = apply_spatial_crop(v, box)
        frames = F.interpolate(frames, size=(self.height, self.width),
                               mode="bilinear", align_corners=False, antialias=True)
        d_m = F.interpolate(d_m, size=(self.height, self.width), mode="nearest")
        v = F.interpolate(v.float(), size=(self.height, self.width), mode="nearest").bool()
        frames = self.img_normalize(frames).clamp(-1.0, 1.0)
        K_list = [adjust_K_for_crop_resize(K, box, self.height, self.width) for K in K_list]

        depths_raw = d_m.clone()
        depths_valid = v
        d_for_scale = torch.where(v, d_m, torch.full_like(d_m, INVALID_DEPTH_SENTINEL))

        # --- Raymaps + mutual scale + modified-disparity depth (inherited from TartanAir) ---
        r_norm, t_norm = self._normalize_first_identity(r_list, t_list)
        raymap_h = getattr(self.cfg, "raymap_height", None) or self.height
        raymap_w = getattr(self.cfg, "raymap_width", None) or self.width
        raymaps_raw, ext_gt, K_gt = self._generate_raymaps_raw(
            K_list, r_norm, t_norm, (raymap_h, raymap_w),
            frame_hw=(self.height, self.width),
        )
        raymaps, depths, scale = self._apply_mutual_scale(raymaps_raw, d_for_scale)
        depths = torch.where(depths_valid, depths, torch.full_like(depths, -1.0))
        return frames, raymaps, depths, scale, ext_gt, K_gt, depths_raw, depths_valid
