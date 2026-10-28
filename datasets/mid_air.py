"""Mid-Air dataset (midair.ulg.ac.be): low-altitude drone RGB + GT depth + poses.

Layout (per climate under <data_root>/MidAir/{Kite_training,PLE_training}/<climate>/):
    sensor_records.hdf5                      trajectory_XXXX/groundtruth/{position,attitude}
                                             at 100 Hz (GT idx = 4 * frame idx), camera_data
                                             path lists at 25 Hz
    color_left/trajectory_XXXX/NNNNNN.JPEG   1024x1024 RGB, 25 fps
    depth/trajectory_XXXX/NNNNNN.PNG         uint16 container of float16 metres

Geometry (verified by datasets/geometry_verification/mid_air/check_raw_reproj.py:
median rel depth reprojection error 0.12% at 4- and 16-frame baselines; wrong
hypotheses 2-5%):
- Intrinsics FIXED: 1024x1024, fov 90 deg -> fx=fy=cx=cy=512.
- Attitude quaternion is (w,x,y,z), body(NED, x-fwd/y-right/z-down)->world; the
  color_left optical axis is body +x, so OpenCV-optical c2w = T @ c2w_ned @ T^-1
  with the same NED permutation TartanAir uses.
- Depth PNGs store EUCLIDEAN RAY DISTANCE (not z): z = d / sqrt(((u-cx)/f)^2 +
  ((v-cy)/f)^2 + 1). Sky renders ~65500 (f16 max); masked via FAR_PLANE_M.
- Everything else (crop/resize, raymaps, mutual scale, modified disparity) is
  inherited verbatim from TartanAirDataset so Mid-Air mixes cleanly.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import random
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

RGB_HW = 1024
F_LEN = 512.0
C_PP = 512.0
K_FIXED = np.array([[F_LEN, 0.0, C_PP], [0.0, F_LEN, C_PP], [0.0, 0.0, 1.0]], dtype=np.float64)

FAR_PLANE_M = 1000.0                 # depth >= this (sky ~65500) -> invalid
MIN_DEPTH_M = 0.1
MIN_FRAMES = 16
GT_PER_FRAME = 4                     # groundtruth at 100 Hz, frames at 25 Hz

# NED (x-fwd, y-right, z-down) -> OpenCV optical (x-right, y-down, z-fwd)
_T_NED = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float64)

# ray-distance -> z-depth divisor, computed once at native res
_uu, _vv = np.meshgrid(np.arange(RGB_HW) + 0.5, np.arange(RGB_HW) + 0.5, indexing="xy")
_RAY_NORM = np.sqrt(((_uu - C_PP) / F_LEN) ** 2 + ((_vv - C_PP) / F_LEN) ** 2 + 1.0).astype(np.float32)


def _quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float64)


class MidAirDataset(TartanAirDataset):
    """One record per Mid-Air trajectory. Reuses TartanAir geometry helpers."""

    SCENE_ID_KEY = "seq_id"   # record["seq_id"] (= "<dataset>/<climate>/<traj>") matches captions.csv scene_id

    def __init__(self, *args, **kwargs):
        self._pose_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        # per-trajectory ZipFile handles (frames kept zipped for inode economy — one frames.zip
        # per trajectory instead of ~2200 loose files). Opened lazily in the worker (fork-safe).
        self._zip_cache: Dict[str, zipfile.ZipFile] = {}
        super().__init__(*args, **kwargs)

    def _zip(self, zip_path: str) -> zipfile.ZipFile:
        if not hasattr(self, "_zip_cache"):
            self._zip_cache = {}
        zf = self._zip_cache.get(zip_path)
        if zf is None:
            zf = self._zip_cache[zip_path] = zipfile.ZipFile(zip_path, "r")
        return zf

    def __getstate__(self):
        # don't ship open zip handles to spawn workers; each reopens lazily in-worker.
        s = self.__dict__.copy()
        s.pop("_zip_cache", None)
        return s

    # ---- Records ----

    def _load_records(self) -> List[Dict[str, Any]]:
        import random as _random
        root = Path(self.cfg.data_root)
        records: List[Dict[str, Any]] = []
        for h5 in sorted(root.glob("MidAir/*/*/sensor_records.hdf5")):
            climate_dir = h5.parent
            for traj_rgb in sorted((climate_dir / "color_left").glob("trajectory_*")):
                traj = traj_rgb.name
                depth_dir = climate_dir / "depth" / traj
                # frames are kept as one frames.zip per trajectory (color + depth) — see
                # _dl/download_zip_keep.sh. Members are NNNNNN.JPEG / NNNNNN.PNG (flat).
                rgb_zip = traj_rgb / "frames.zip"
                depth_zip = depth_dir / "frames.zip"
                if not (rgb_zip.is_file() and depth_zip.is_file()):
                    continue
                try:
                    n_frames = sum(1 for m in zipfile.ZipFile(rgb_zip).namelist() if m.endswith(".JPEG"))
                    n_depth = sum(1 for m in zipfile.ZipFile(depth_zip).namelist() if m.endswith(".PNG"))
                except zipfile.BadZipFile:
                    continue
                n = min(n_frames, n_depth)
                if n < MIN_FRAMES:
                    continue
                records.append({
                    "seq_id": f"{climate_dir.parent.name}/{climate_dir.name}/{traj}",
                    "traj_dir": str(traj_rgb),
                    "depth_dir": str(depth_dir),
                    "rgb_zip": str(rgb_zip),
                    "depth_zip": str(depth_zip),
                    "h5_path": str(h5),
                    "traj": traj,
                    "env": f"{climate_dir.parent.name}/{climate_dir.name}",
                    "difficulty": "",
                    "n_frames": n,
                    "fps": int(getattr(self.cfg, "fps_source", 25)),
                    "height": RGB_HW,
                    "width": RGB_HW,
                    "caption": "",
                })

        if not records:
            raise FileNotFoundError(f"No Mid-Air trajectories found under {root}")

        if not self.filtering.disable:
            pre = len(records)
            records = [r for r in records if self._filter_record(r)]
            print(f"MidAir: filtered {pre - len(records)} from {pre}, kept {len(records)}")
        else:
            print(f"MidAir: loaded {len(records)} trajectories from {root}")

        _random.Random(0).shuffle(records)

        if self.split != "all":
            n_test = int(len(records) * self.test_percentage)
            if self.split == "training":
                records = records[n_test:] if n_test > 0 else records
            else:
                records = records[:n_test] if n_test > 0 else records[:1]

        records = [self.preprocess_record(r) for r in records]
        return records

    # ---- Per-trajectory IO ----

    def _load_pose_rows(self, record: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """(position (M,3), attitude wxyz (M,4)) at 100 Hz, cached as plain numpy
        (h5py handles are NOT kept open -> fork-safe for DataLoader workers)."""
        key = f"{record['h5_path']}::{record['traj']}"
        if key not in self._pose_cache:
            import h5py
            with h5py.File(record["h5_path"], "r") as f:
                g = f[record["traj"]]["groundtruth"]
                self._pose_cache[key] = (np.asarray(g["position"]), np.asarray(g["attitude"]))
        return self._pose_cache[key]

    def _read_rgb_frames(self, rgb_zip: str, frame_ids: np.ndarray) -> np.ndarray:
        zf = self._zip(rgb_zip)
        return np.stack([
            np.asarray(iio.imread(io.BytesIO(zf.read(f"{int(fid):06d}.JPEG"))))[..., :3]
            for fid in frame_ids
        ])

    def _read_depth_frames(self, depth_zip: str, frame_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        zf = self._zip(depth_zip)
        depths, valids = [], []
        for fid in frame_ids:
            raw = iio.imread(io.BytesIO(zf.read(f"{int(fid):06d}.PNG")))
            d_ray = raw.view(np.float16).astype(np.float32)
            d = d_ray / _RAY_NORM                      # ray distance -> z-depth
            m = np.isfinite(d) & (d > MIN_DEPTH_M) & (d < FAR_PLANE_M)
            depths.append(np.where(m, d, 0.0).astype(np.float32))
            valids.append(m)
        return np.stack(depths), np.stack(valids)

    def _load_cameras(
        self, record: Dict[str, Any], frame_ids: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        pos, att = self._load_pose_rows(record)
        K_list, r_list, t_list = [], [], []
        for fid in frame_ids:
            gi = min(GT_PER_FRAME * int(fid), len(pos) - 1)
            P = np.eye(4)
            P[:3, :3] = _quat_wxyz_to_R(att[gi])
            P[:3, 3] = pos[gi]
            M = np.eye(4); M[:3, :3] = _T_NED
            c2w = M @ P @ np.linalg.inv(M)             # verified: full_conj + wxyz
            K_list.append(K_FIXED.copy())
            r_list.append(c2w[:3, :3].copy())
            t_list.append(c2w[:3, 3].copy())
        return K_list, r_list, t_list

    # ---- Per-clip assembly (mirrors bedlam/tartanair _load_video) ----

    def _load_video(
        self, record: Dict[str, Any]
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, float,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        rgb_zip = record.get("rgb_zip") or str(Path(record["traj_dir"]) / "frames.zip")
        depth_zip = record.get("depth_zip") or str(Path(record["depth_dir"]) / "frames.zip")

        temporal_indices = self._temporal_sample(record["n_frames"], record["fps"])
        # Static-scene temporal reverse (opt-in): valid backward camera path (see tartanair.py).
        if getattr(self.cfg, "allow_temporal_reverse", False) and random.random() < 0.5:
            temporal_indices = temporal_indices[::-1].copy()

        rgb_np = self._read_rgb_frames(rgb_zip, temporal_indices)
        frames = torch.from_numpy(rgb_np).float().permute(0, 3, 1, 2).contiguous() / 255.0

        depth_np, valid_np = self._read_depth_frames(depth_zip, temporal_indices)
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
