"""MatrixCity (Li et al. 2023, BoDai/MatrixCity): Unreal Engine 5 synthetic city.

STREET split only: smooth vehicle-path video (verified in-bounds 100% over
100-frame baselines). The AERIAL split is a grid scan (~21% of the image leaves the
view EVERY frame) — not usable as continuous video clips; excluded.

Layout (<data_root>/):
    small_city/street/<split>/<road>/transforms.json   frames: {frame_index, rot_mat}
    small_city/street/<split>/<road>/%04d.png          1000x1000 RGB (fov 90)
    small_city_depth/street/<split>/<road>_depth/%04d.exr  float16 R=G=B, 65504=sky

Geometry (verified datasets/geometry_verification/matrixcity/check_raw_reproj.py):
- rot_mat is c2w in OpenCV axes with the ROTATION SCALED by 0.01 — normalize by
  cbrt(det). Translation kept as stored.
- STREET depth shares the pose unit directly (divisor 1; z-depth). (Aerial would
  need /100 — another reason it stays excluded.)
- Intrinsics from camera_angle_x (fov 90 -> fx=fy=cx=cy=500 at 1000x1000).
- A road json may concatenate several sweeps: records are split at teleports
  (translation step > TELEPORT_FACTOR x median step or non-consecutive frame_index)
  so a temporal window never crosses a jump.
"""
from __future__ import annotations

import random
import io
import json
import math
import os
import tempfile
import zipfile
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

FAR_SENTINEL = 60000.0               # f16 sky is 65504; anything near it is invalid
MIN_DEPTH = 1e-3
MIN_FRAMES = 16
TELEPORT_FACTOR = 5.0                # split record when step > factor x median step
STREET_SPLITS = ("train", "train_dense")
CITIES = ("small_city", "big_city")   # big_city street downloaded 2026-07-12 (phase 2)


def _read_exr_r(path: str) -> np.ndarray:
    import OpenEXR, Imath
    f = OpenEXR.InputFile(path)
    try:
        h = f.header()
        dw = h["dataWindow"]
        W, H = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
        d = np.frombuffer(f.channel("R", Imath.PixelType(Imath.PixelType.HALF)),
                          dtype=np.float16).reshape(H, W)
    finally:
        f.close()
    return d.astype(np.float32)


def _read_exr_r_bytes(buf: bytes) -> np.ndarray:
    """OpenEXR is path-only, so spill the zip member to a RAM tempfile (/dev/shm — no
    filesystem inode/byte cost) and reuse _read_exr_r."""
    d = "/dev/shm" if os.path.isdir("/dev/shm") else None
    tf = tempfile.NamedTemporaryFile(suffix=".exr", dir=d, delete=False)
    try:
        tf.write(buf); tf.close()
        return _read_exr_r(tf.name)
    finally:
        try: os.unlink(tf.name)
        except OSError: pass


class MatrixCityDataset(TartanAirDataset):
    """One record per contiguous street sweep. Reuses TartanAir geometry helpers.

    Frames are kept as one frames.zip per block (road/frames.zip for RGB pngs,
    <road>_depth/frames.zip for EXR depth) for filesystem inode economy — see
    _dl/repackage_tars_to_zip.py. transforms.json stays loose. Loader reads members
    via zipfile (ASE/mid_air pattern)."""

    SCENE_ID_KEY = "seq_id"   # record["seq_id"] matches captions.csv scene_id

    def _zip(self, zip_path: str) -> zipfile.ZipFile:
        if not hasattr(self, "_zip_cache"):
            self._zip_cache: Dict[str, zipfile.ZipFile] = {}
        zf = self._zip_cache.get(zip_path)
        if zf is None:
            zf = self._zip_cache[zip_path] = zipfile.ZipFile(zip_path, "r")
        return zf

    def __getstate__(self):
        # never ship open zip handles across a process boundary (spawn workers) — each worker
        # reopens lazily. (Fork safety comes from _load_records not caching in the main process.)
        s = self.__dict__.copy()
        s.pop("_zip_cache", None)
        return s

    # ---- Records ----

    def _load_records(self) -> List[Dict[str, Any]]:
        import random as _random
        root = Path(self.cfg.data_root)
        records: List[Dict[str, Any]] = []
        for city in CITIES:
          for split in STREET_SPLITS:
            for road in sorted((root / city / "street" / split).glob("*/")):
                tj = road / "transforms.json"
                depth_dir = root / f"{city}_depth" / "street" / split / f"{road.name}_depth"
                rgb_zip = road / "frames.zip"
                depth_zip = depth_dir / "frames.zip"
                if not (tj.exists() and rgb_zip.is_file() and depth_zip.is_file()):
                    continue
                try:
                    # TRANSIENT ZipFile (not self._zip) so no handle is cached in the MAIN
                    # process — forked DataLoader workers must not inherit shared zip fds.
                    with zipfile.ZipFile(str(rgb_zip)) as _z:
                        rgb_names = set(_z.namelist())
                    with zipfile.ZipFile(str(depth_zip)) as _z:
                        depth_names = set(_z.namelist())
                except zipfile.BadZipFile:
                    continue
                meta = json.load(open(tj))
                ang = meta.get("camera_angle_x", math.pi / 2)
                frames = sorted(meta["frames"], key=lambda m: int(m["frame_index"]))
                # keep only frames whose rgb+depth members exist (zips can be partial)
                keep = []
                for m in frames:
                    fi = int(m["frame_index"])
                    if f"{fi:04d}.png" in rgb_names and f"{fi:04d}.exr" in depth_names:
                        M = np.array(m["rot_mat"], dtype=np.float64)
                        keep.append((fi, M))
                if len(keep) < MIN_FRAMES:
                    continue
                # split into contiguous sweeps at teleports / index gaps
                pos = np.array([m[:3, 3] for _, m in keep])
                idxs = np.array([fi for fi, _ in keep])
                steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
                med = max(float(np.median(steps)), 1e-9)
                cut = np.where((steps > TELEPORT_FACTOR * med) | (np.diff(idxs) != 1))[0]
                bounds = [0] + (cut + 1).tolist() + [len(keep)]
                for si, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
                    if hi - lo < MIN_FRAMES:
                        continue
                    records.append({
                        "seq_id": f"{city}/street/{split}/{road.name}/sweep{si:02d}",
                        "traj_dir": str(road),
                        "depth_dir": str(depth_dir),
                        "rgb_zip": str(rgb_zip),
                        "depth_zip": str(depth_zip),
                        "frame_ids": [int(fi) for fi, _ in keep[lo:hi]],
                        "camera_angle_x": float(ang),
                        "env": f"{city}/{split}/{road.name}",
                        "difficulty": "",
                        "n_frames": hi - lo,
                        "fps": int(getattr(self.cfg, "fps_source", 10)),
                        "height": 1000,
                        "width": 1000,
                        "caption": "",
                    })

        if not records:
            raise FileNotFoundError(f"No MatrixCity street sweeps found under {root}")

        if not self.filtering.disable:
            pre = len(records)
            records = [r for r in records if self._filter_record(r)]
            print(f"MatrixCity: filtered {pre - len(records)} from {pre}, kept {len(records)}")
        else:
            print(f"MatrixCity: loaded {len(records)} street sweeps from {root}")

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

    def _load_pose_map(self, record: Dict[str, Any]) -> Dict[int, np.ndarray]:
        key = record["traj_dir"]
        if not hasattr(self, "_mc_pose_cache"):
            self._mc_pose_cache: Dict[str, Dict[int, np.ndarray]] = {}
        if key not in self._mc_pose_cache:
            meta = json.load(open(Path(key) / "transforms.json"))
            pm = {}
            for m in meta["frames"]:
                M = np.array(m["rot_mat"], dtype=np.float64)
                s = np.cbrt(np.linalg.det(M[:3, :3]))
                M[:3, :3] = M[:3, :3] / s              # undo the 0.01 rotation scale
                pm[int(m["frame_index"])] = M
            self._mc_pose_cache[key] = pm
        return self._mc_pose_cache[key]

    # ---- Per-clip assembly ----

    def _load_video(
        self, record: Dict[str, Any]
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, float,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        road = Path(record["traj_dir"])
        depth_dir = Path(record["depth_dir"])
        rgb_zip = record.get("rgb_zip") or str(road / "frames.zip")
        depth_zip = record.get("depth_zip") or str(depth_dir / "frames.zip")
        fids_all = record["frame_ids"]

        # MatrixCity street depth AND pose translation are in UE5 CENTIMETRES. Convert both
        # (same factor -> reprojection preserved) to metres via unit_to_meters (0.01 in yaml).
        u = float(self.cfg.get("unit_to_meters", 1.0))

        temporal_indices = self._temporal_sample(record["n_frames"], record["fps"])
        # Static-scene temporal reverse (opt-in): valid backward camera path (see tartanair.py).
        if getattr(self.cfg, "allow_temporal_reverse", False) and random.random() < 0.5:
            temporal_indices = temporal_indices[::-1].copy()
        fids = [fids_all[int(i)] for i in temporal_indices]

        rz = self._zip(rgb_zip)
        rgb_np = np.stack([
            np.asarray(iio.imread(io.BytesIO(rz.read(f"{fi:04d}.png"))))[..., :3] for fi in fids
        ])
        frames = torch.from_numpy(rgb_np).float().permute(0, 3, 1, 2).contiguous() / 255.0

        dz = self._zip(depth_zip)
        depths, valids = [], []
        for fi in fids:
            d = _read_exr_r_bytes(dz.read(f"{fi:04d}.exr"))   # raw EXR in cm
            m = np.isfinite(d) & (d > MIN_DEPTH) & (d < FAR_SENTINEL)  # mask in native cm
            depths.append((np.where(m, d, 0.0) * u).astype(np.float32))  # cm -> metres
            valids.append(m)
        d_m = torch.from_numpy(np.stack(depths)).unsqueeze(1).float()
        v = torch.from_numpy(np.stack(valids)).unsqueeze(1).bool()

        H_src, W_src = frames.shape[-2:]
        fl = W_src / (2.0 * math.tan(record["camera_angle_x"] / 2.0))
        K = np.array([[fl, 0.0, W_src / 2.0], [0.0, fl, H_src / 2.0], [0.0, 0.0, 1.0]])
        pose_map = self._load_pose_map(record)
        K_list, r_list, t_list = [], [], []
        for fi in fids:
            c2w = pose_map[fi]
            K_list.append(K.copy())
            r_list.append(c2w[:3, :3].copy())
            t_list.append(c2w[:3, 3].copy() * u)            # cm -> metres (same factor as depth)

        box = center_crop_box_for_aspect(H_src, W_src, self.height, self.width)
        frames = apply_spatial_crop(frames, box)
        d_m = apply_spatial_crop(d_m, box)
        v = apply_spatial_crop(v, box)
        frames = F.interpolate(frames, size=(self.height, self.width),
                               mode="bilinear", align_corners=False, antialias=True)
        d_m = F.interpolate(d_m, size=(self.height, self.width), mode="nearest")
        v = F.interpolate(v.float(), size=(self.height, self.width), mode="nearest").bool()
        frames = self.img_normalize(frames).clamp(-1.0, 1.0)
        K_list = [adjust_K_for_crop_resize(Ki, box, self.height, self.width) for Ki in K_list]

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
