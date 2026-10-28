"""MVS-Synth (DeepMVS, Huang et al. 2018): GTA-V synthetic street/aerial video.

Layout (<data_root>/GTAV_540/):
    <seq %04d>/images/%04d.png    960x540 RGB, 100 frames/seq, 120 seqs
    <seq %04d>/depths/%04d.exr    float32 'Y' channel, Z-DEPTH metres, inf = sky
    <seq %04d>/poses/%04d.json    {f_x,f_y,c_x,c_y, extrinsic: 4x4 WORLD-TO-CAMERA}

Geometry (verified datasets/geometry_verification/mvs_synth/check_raw_reproj.py on
3 seqs: 0.08-0.13% median rel depth reproj error; ray-depth 6-12%, OpenGL axes 30-50%,
c2w-as-stored lands off-screen):
- extrinsic is w2c in OpenCV axes -> c2w = inv(extrinsic). Per-frame K from the json.
- Depth is plain z-depth (metres); inf/far masked via FAR_PLANE_M.
- Crop/resize, raymaps, mutual scale, modified disparity inherited from TartanAir.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F

from ._crop_utils import (
    adjust_K_for_crop_resize,
    apply_spatial_crop,
    center_crop_box_for_aspect,
)
from .tartanair import TartanAirDataset, INVALID_DEPTH_SENTINEL

FAR_PLANE_M = 1000.0
MIN_DEPTH_M = 0.1
MIN_FRAMES = 16


def _read_exr_y(path: str) -> np.ndarray:
    import OpenEXR, Imath
    f = OpenEXR.InputFile(path)
    try:
        h = f.header()
        dw = h["dataWindow"]
        W, H = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
        d = np.frombuffer(f.channel("Y", Imath.PixelType(Imath.PixelType.FLOAT)),
                          dtype=np.float32).reshape(H, W)
    finally:
        f.close()
    return d


class MVSSynthDataset(TartanAirDataset):
    """One record per GTAV_540 sequence. Reuses TartanAir geometry helpers."""

    SCENE_ID_KEY = "seq_id"   # record["seq_id"] (= "GTAV_540/<seq>") matches captions.csv scene_id

    def __init__(self, *args, **kwargs):
        self._pose_cache: Dict[str, list] = {}
        super().__init__(*args, **kwargs)

    # ---- Records ----

    def _load_records(self) -> List[Dict[str, Any]]:
        import random as _random
        root = Path(self.cfg.data_root) / "GTAV_540"
        records: List[Dict[str, Any]] = []
        for seq_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            img_dir = seq_dir / "images"
            if not (img_dir.is_dir() and (seq_dir / "depths").is_dir()
                    and (seq_dir / "poses").is_dir()):
                continue
            n = sum(1 for _ in img_dir.glob("*.png"))
            if n < MIN_FRAMES:
                continue
            records.append({
                "seq_id": f"GTAV_540/{seq_dir.name}",
                "traj_dir": str(seq_dir),
                "env": seq_dir.name,
                "difficulty": "",
                "n_frames": n,
                "fps": int(getattr(self.cfg, "fps_source", 10)),
                "height": 540,
                "width": 960,
                "caption": "",
            })

        if not records:
            raise FileNotFoundError(f"No MVS-Synth sequences found under {root}")

        if not self.filtering.disable:
            pre = len(records)
            records = [r for r in records if self._filter_record(r)]
            print(f"MVSSynth: filtered {pre - len(records)} from {pre}, kept {len(records)}")
        else:
            print(f"MVSSynth: loaded {len(records)} sequences from {root}")

        _random.Random(0).shuffle(records)

        if self.split != "all":
            n_test = int(len(records) * self.test_percentage)
            if self.split == "training":
                records = records[n_test:] if n_test > 0 else records
            else:
                records = records[:n_test] if n_test > 0 else records[:1]

        records = [self.preprocess_record(r) for r in records]
        return records

    # ---- IO ----

    @staticmethod
    def _read_rgb_frames(seq_dir: Path, frame_ids: np.ndarray) -> np.ndarray:
        return np.stack([
            np.asarray(iio.imread(str(seq_dir / "images" / f"{int(fid):04d}.png")))[..., :3]
            for fid in frame_ids
        ])

    @staticmethod
    def _read_depth_frames(seq_dir: Path, frame_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        depths, valids = [], []
        for fid in frame_ids:
            d = _read_exr_y(str(seq_dir / "depths" / f"{int(fid):04d}.exr"))
            m = np.isfinite(d) & (d > MIN_DEPTH_M) & (d < FAR_PLANE_M)
            depths.append(np.where(m, d, 0.0).astype(np.float32))
            valids.append(m)
        return np.stack(depths), np.stack(valids)

    def _load_cameras(
        self, record: Dict[str, Any], frame_ids: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        seq_dir = Path(record["traj_dir"])
        K_list, r_list, t_list = [], [], []
        for fid in frame_ids:
            p = json.load(open(seq_dir / "poses" / f"{int(fid):04d}.json"))
            K = np.array([[p["f_x"], 0.0, p["c_x"]],
                          [0.0, p["f_y"], p["c_y"]],
                          [0.0, 0.0, 1.0]], dtype=np.float64)
            c2w = np.linalg.inv(np.array(p["extrinsic"], dtype=np.float64))  # w2c verified
            K_list.append(K)
            r_list.append(c2w[:3, :3].copy())
            t_list.append(c2w[:3, 3].copy())
        return K_list, r_list, t_list

    # ---- Per-clip assembly (mirrors bedlam/tartanair; per-frame K) ----

    def _load_video(
        self, record: Dict[str, Any]
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, float,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        seq_dir = Path(record["traj_dir"])
        temporal_indices = self._temporal_sample(record["n_frames"], record["fps"])

        rgb_np = self._read_rgb_frames(seq_dir, temporal_indices)
        frames = torch.from_numpy(rgb_np).float().permute(0, 3, 1, 2).contiguous() / 255.0

        depth_np, valid_np = self._read_depth_frames(seq_dir, temporal_indices)
        d_m = torch.from_numpy(depth_np).unsqueeze(1).float()
        v = torch.from_numpy(valid_np).unsqueeze(1).bool()

        K_list, r_list, t_list = self._load_cameras(record, temporal_indices)

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
