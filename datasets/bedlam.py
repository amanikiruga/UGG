"""BEDLAM dataset: RGB + GT metric depth (EXR) + per-frame Unreal cameras -> raymaps.

Max-Planck synthetic-humans dataset (bedlam.is.tue.mpg.de). We use it purely as a
geometry source (RGB + metric depth + moving cameras). Supports BOTH releases, which
share an identical camera-CSV schema and on-disk layout (only sub-paths differ):

  v1 (Intelligent-Systems/BEDLAM[-depth]):
      <batch>/png/seq_XXXXXX/seq_XXXXXX_NNNN.png
      <batch>/depth/seq_XXXXXX/seq_XXXXXX_NNNN_depth.exr          (channel 'Depth')
      <batch>/ground_truth/camera/seq_XXXXXX_camera.csv
  v2 (Intelligent-Systems/BEDLAM2[-depth]):
      <batch>/png/seq_XXXXXX/seq_XXXXXX_NNNN.png
      <batch>/exr_depth/seq_XXXXXX/seq_XXXXXX_NNNN_depth.exr      (multilayer ...WorldDepth.R)
      <batch>/ground_truth/meta_exr_csv/seq_XXXXXX_camera.csv

Geometry is made IDENTICAL to TartanAir so the datasets mix cleanly: the geometry /
normalization helpers (first-frame-identity, Plucker raymaps, mutual scale, depth->
modified-disparity) are inherited verbatim from TartanAirDataset. Only the record walk,
file IO and the Unreal->OpenCV camera convention differ.

Conventions (camera convention verified by reprojection on camera-moving orbit/zoom
sequences, see datasets/geometry_verification/bedlam/):
- Camera CSV per frame: name,x,y,z,yaw,pitch,roll,focal_length,sensor_width,
  sensor_height,hfov.  Position in cm (Unreal world X-fwd/Y-right/Z-up, left-handed);
  yaw/pitch/roll in degrees (Unreal FRotator).
- Intrinsics: fx = fl_mm * IMG_W / SENSOR_W, fy = fl_mm * IMG_H / SENSOR_H, cx=W/2,
  cy=H/2 (SENSOR_W=36, SENSOR_H=20.25 -> square pixels).  Per-frame (some scenes zoom).
- Extrinsics: c2w in OpenCV optical frame = R_actor2world(yaw,pitch,roll) @ A, with
  A = [[0,0,1],[1,0,0],[0,-1,0]] (optical right/down/fwd = Unreal +Y/-Z/+X) and the
  camera centre = (x,y,z)/100 (cm->m).  This det(-1) map folds in the UE left->right
  handedness; after first-frame-identity normalization the relative poses are proper
  rotations in the same right-handed optical frame TartanAir uses.
- Depth: EXR z-buffer in cm -> /100 metres.  Sky/background renders as a huge value
  (>1e4 cm); masked invalid via FAR_PLANE_M.
- Output depths / raymaps / scaling: inherited from TartanAir (modified disparity in
  [-1,1], Plucker raymaps, per-clip mutual scale with MIN_SCALE=1.0).
"""
from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import imageio.v3 as iio
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.nn.functional as F

from ._crop_utils import (
    adjust_K_for_crop_resize,
    apply_spatial_crop,
    center_crop_box_for_aspect,
)
from .tartanair import TartanAirDataset, INVALID_DEPTH_SENTINEL

SENSOR_W, SENSOR_H = 36.0, 20.25     # BEDLAM virtual DSLR sensor (mm); square pixels
FAR_PLANE_M = 1000.0                 # depth >= this (sky/background) -> invalid
MIN_DEPTH_M = 0.05                   # depth <= this -> invalid
MIN_FRAMES = 16                      # skip sequences shorter than this

# Sideways sequences: some batches (v1 *closeup*, v2 vcam/portrait) were rendered with
# the camera genuinely rolled ~+-90 deg (world-up along image x). Pixels, depth and
# extrinsics are mutually consistent as stored, but the CONTENT is sideways, which we
# must not train on. The roll is in the GT extrinsics (see the per-batch scan in
# vwm_eval_data/training_data_report/bedlam_bg_audit/roll_scan.json: cleanly bimodal,
# ~0 or ~-90, constant per batch), so each clip is un-rolled by an exact multiple of
# 90 deg: a lossless rot90 of RGB/depth/valid (z-buffer depth is invariant to in-plane
# rotation), axis-swapped K, and c2w := c2w @ Rz. The residual few-degree handheld
# roll remains in BOTH pixels and cameras, i.e. stays consistent.
_UP_WORLD = np.array([0.0, 0.0, 1.0])          # Unreal world up = +Z
# Rz(+90) as column-vector rotation; Rz(-90) is its transpose.
_RZ_90 = {
    1: np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    -1: np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
}

# Optical-axis map: columns = OpenCV optical axes (right, down, fwd) in Unreal actor frame.
# UE camera looks along actor +X (fwd); image-right = actor +Y; image-down = -actor-Z.
_A_OPT = np.array([[0.0, 0.0, 1.0],
                   [1.0, 0.0, 0.0],
                   [0.0, -1.0, 0.0]], dtype=np.float64)


def _ue_rotmat(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Unreal FRotationMatrix (degrees) -> actor-to-world rotation (column-vector form).
    UE's matrix rows are the world images of the local X/Y/Z axes (row-vector convention),
    so the column-vector actor->world rotation is its transpose (columns = fwd/right/up)."""
    cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    cp, sp = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
    cr, sr = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    M = np.array([
        [cp * cy,                   cp * sy,                  sp],
        [sr * sp * cy - cr * sy,    sr * sp * sy + cr * cy,  -sr * cp],
        [-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy,   cr * cp],
    ], dtype=np.float64)
    return M.T


def bedlam_cam_to_K_c2w(row: np.ndarray, H: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
    """One camera CSV row (x,y,z,yaw,pitch,roll,fl) -> (K 3x3, c2w 4x4 OpenCV optical)."""
    x, y, z, yaw, pitch, roll, fl = row[:7]
    fx = fl * W / SENSOR_W
    fy = fl * H / SENSOR_H
    K = np.array([[fx, 0.0, W / 2.0], [0.0, fy, H / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = _ue_rotmat(yaw, pitch, roll) @ _A_OPT
    c2w[:3, 3] = np.array([x, y, z], dtype=np.float64) / 100.0   # cm -> m
    return K, c2w


def _read_exr_depth_m(path: str) -> np.ndarray:
    """Read a BEDLAM depth EXR -> (H,W) float32 metres. Handles v1 'Depth' and v2
    multilayer '...WorldDepth.R'. EXR stores z-buffer in cm -> /100."""
    import OpenEXR, Imath
    f = OpenEXR.InputFile(path)
    try:
        h = f.header()
        chans = list(h["channels"].keys())
        if "Depth" in chans:
            ch = "Depth"
        else:
            ch = next((c for c in chans if c.endswith("WorldDepth.R") or c.endswith("Depth.R")), chans[0])
        dw = h["dataWindow"]
        W = dw.max.x - dw.min.x + 1
        Hh = dw.max.y - dw.min.y + 1
        raw = f.channel(ch, Imath.PixelType(Imath.PixelType.FLOAT))
        d = np.frombuffer(raw, dtype=np.float32).reshape(Hh, W).astype(np.float32)
    finally:
        f.close()
    return d / 100.0


_RGB_THREADS = int(os.environ.get("BEDLAM_RGB_THREADS", "16"))
_DEPTH_CACHE_ROOT = os.environ.get(
    "BEDLAM_DEPTH_CACHE", os.path.expanduser("~/.cache/ugg/bedlam_depth_cache"))


class BedlamDataset(TartanAirDataset):
    """One record per BEDLAM sequence (a <batch>/.../seq_XXXXXX). Reuses TartanAir geometry."""

    SCENE_ID_KEY = "seq_id"   # record["seq_id"] (= "batch/seq") matches captions.csv scene_id

    def __init__(self, *args, **kwargs):
        self._cam_cache: Dict[str, np.ndarray] = {}     # seq_id -> (T,7) camera rows
        super().__init__(*args, **kwargs)

    # ---- pre-decoded depth cache ----

    def _cached_depth(self, record, frame_ids):
        """Depth window from the pre-decoded cache, already rot90'd/cropped/resized to
        (self.height, self.width) -- verified bit-exact against the EXR path.

        BEDLAM depth frames are 20-channel multilayer EXRs; pulling the one WorldDepth
        channel decompresses all 20 and the OpenEXR bindings hold the GIL, so the EXR path
        reads at 63 MB/s (4.17s per 50-frame clip). The cache is chunked+compressed npz,
        which numpy loads lazily per key, so only the 1-2 chunks a window touches are
        decompressed: 0.088s, a 47x speedup. Returns None to fall back to EXR.
        """
        root = getattr(self.cfg, "depth_cache_root", None) or _DEPTH_CACHE_ROOT
        if not root:
            return None
        # bedlam and bedlam2 share this class, so key the cache on the data root
        name = Path(str(self.data_root)).name
        f = Path(root) / f"{name}_{self.height}x{self.width}" / \
            f"{str(record['seq_id']).replace('/', '__')}.npz"
        try:
            if not f.exists():
                return None
            z = np.load(f)
            ch = 64
            ids = np.asarray(frame_ids, dtype=int)
            lo, hi = int(ids.min()) // ch, int(ids.max()) // ch
            buf = np.concatenate([z[f"c{c}"] for c in range(lo, hi + 1)])
            return buf[ids - lo * ch].astype(np.float32)
        except Exception:
            return None            # any cache problem -> silent fall back to EXR

    # ---- layout helpers (v1 vs v2) ----

    @staticmethod
    def _depth_subdir(batch: Path) -> str:
        return "exr_depth" if (batch / "exr_depth").is_dir() else "depth"

    @staticmethod
    def _camera_subdir(batch: Path) -> str:
        if (batch / "ground_truth" / "meta_exr_csv").is_dir():
            return "ground_truth/meta_exr_csv"
        return "ground_truth/camera"

    # ---- Records ----

    def _load_records(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for batch in sorted(p for p in self.data_root.iterdir() if p.is_dir()):
            # v2 *_hdri batches are photo-backdrop renders with NO real background
            # geometry (see bedlam_bg_audit/) — excluded permanently, and this also
            # keeps record-building safe while their on-disk deletion is in flight.
            if "_hdri" in batch.name:
                continue
            png_root = batch / "png"
            if not png_root.is_dir():
                continue
            depth_sub = self._depth_subdir(batch)
            cam_sub = self._camera_subdir(batch)
            depth_root = batch / depth_sub
            cam_root = batch / cam_sub
            if not (depth_root.is_dir() and cam_root.is_dir()):
                continue
            for seq_dir in sorted(p for p in png_root.iterdir() if p.is_dir()):
                seq = seq_dir.name
                cam_csv = cam_root / f"{seq}_camera.csv"
                depth_seq = depth_root / seq
                if not (cam_csv.exists() and depth_seq.is_dir()):
                    continue
                n_frames = sum(1 for _ in seq_dir.glob(f"{seq}_*.png"))
                if n_frames < MIN_FRAMES:
                    continue
                records.append({
                    "seq_id": f"{batch.name}/{seq}",
                    "traj_dir": str(seq_dir),                 # png/seq dir
                    "batch_dir": str(batch),
                    "depth_dir": str(depth_seq),
                    "cam_csv": str(cam_csv),
                    "seq": seq,
                    "env": batch.name,
                    "difficulty": "",
                    "n_frames": n_frames,
                    "fps": int(getattr(self.cfg, "fps_source", 30)),
                    "height": 720,
                    "width": 1280,
                    "caption": "",
                })

        if not records:
            raise FileNotFoundError(f"No BEDLAM sequences found under {self.data_root}")

        if not self.filtering.disable:
            pre = len(records)
            records = [r for r in records if self._filter_record(r)]
            print(f"BEDLAM: filtered {pre - len(records)} from {pre}, kept {len(records)}")
        else:
            print(f"BEDLAM: loaded {len(records)} sequences from {self.data_root}")

        random.Random(0).shuffle(records)

        if self.split != "all":
            n_test = int(len(records) * self.test_percentage)
            if self.split == "training":
                records = records[n_test:] if n_test > 0 else records
            else:
                records = records[:n_test] if n_test > 0 else records[:1]

        records = [self.preprocess_record(r) for r in records]
        return records

    # ---- Per-sequence IO ----

    def _load_cam_rows(self, record: Dict[str, Any]) -> np.ndarray:
        key = record["cam_csv"]
        if key not in self._cam_cache:
            rows = []
            with open(key) as fh:
                for r in csv.DictReader(fh):
                    rows.append([float(r["x"]), float(r["y"]), float(r["z"]),
                                 float(r["yaw"]), float(r["pitch"]), float(r["roll"]),
                                 float(r["focal_length"])])
            self._cam_cache[key] = np.array(rows, dtype=np.float64)
        return self._cam_cache[key]

    @staticmethod
    def _read_rgb_frames(rgb_dir: Path, seq: str, frame_ids: np.ndarray) -> np.ndarray:
        """Threaded PNG decode. PIL releases the GIL inside the decoder, so 50 frames go
        from 1.41s serial to 0.10s at 16 threads (measured); the filesystem sustains
        ~726 MB/s under this pattern, so serial reads were leaving 13x on the table."""
        def _one(fid):
            return np.asarray(iio.imread(str(rgb_dir / f"{seq}_{int(fid):04d}.png")))[..., :3]
        with ThreadPoolExecutor(_RGB_THREADS) as ex:
            return np.stack(list(ex.map(_one, list(frame_ids))))

    @staticmethod
    def _read_depth_frames(depth_dir: Path, seq: str, frame_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        depths, valids = [], []
        for fid in frame_ids:
            # v1 names depth EXRs "<seq>_NNNN_depth.exr"; v2 (exr_depth/) names
            # them plain "<seq>_NNNN.exr". Every v2 load raised OSError before
            # this fallback existed, which the recursive-retry __getitem__ then
            # amplified into ~100GB worker OOMs.
            p = depth_dir / f"{seq}_{int(fid):04d}_depth.exr"
            if not p.exists():
                p = depth_dir / f"{seq}_{int(fid):04d}.exr"
            d = _read_exr_depth_m(str(p))
            m = np.isfinite(d) & (d > MIN_DEPTH_M) & (d < FAR_PLANE_M)
            d = np.where(m, d, 0.0).astype(np.float32)
            depths.append(d)
            valids.append(m)
        return np.stack(depths), np.stack(valids)

    def _load_cameras(
        self, record: Dict[str, Any], frame_ids: np.ndarray, hw: Tuple[int, int]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        rows = self._load_cam_rows(record)
        H, W = hw
        K_list, r_list, t_list = [], [], []
        for fid in frame_ids:
            K, c2w = bedlam_cam_to_K_c2w(rows[int(fid)], H, W)
            K_list.append(K)
            r_list.append(c2w[:3, :3].copy())
            t_list.append(c2w[:3, 3].copy())
        return K_list, r_list, t_list

    # ---- Sideways (rolled-camera) correction ----

    @staticmethod
    def _clip_roll_deg(r_list: List[np.ndarray]) -> float:
        """Median in-plane roll of the clip, from world-up expressed in camera coords.

        g = R_c2w^T @ up_world; upright (OpenCV y down) has g ~ (0,-1,0) -> 0 deg,
        sideways-stored has g ~ (+-1,0,0) -> +-90 deg. Frames whose optical axis is
        near-vertical (|g_xy| small, roll ill-defined) are skipped."""
        rolls = []
        for R in r_list:
            g = R.T @ _UP_WORLD
            if float(np.hypot(g[0], g[1])) < 0.2:
                continue
            rolls.append(np.degrees(np.arctan2(g[0], -g[1])))
        return float(np.median(rolls)) if rolls else 0.0

    @staticmethod
    def _rot90_step(
        frames: torch.Tensor, d_m: torch.Tensor, v: torch.Tensor,
        K_list: List[np.ndarray], r_list: List[np.ndarray], k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[np.ndarray], List[np.ndarray]]:
        """One exact +-90 deg in-plane rotation of the whole clip (k=+1 CCW, k=-1 CW).

        Pixel-center convention is `index + 0.5` (matches camera_to_raymap /
        adjust_K_for_crop_resize), under which rot90 maps continuous coords
          k=-1 (CW):  u' = H - v,  v' = u        -> cx' = H - cy, cy' = cx
          k=+1 (CCW): u' = v,      v' = W - u    -> cx' = cy,     cy' = W - cx
        with fx' = fy, fy' = fx, and the new camera basis x'/y' = a Rz(k*90) remap
        of x/y (z, and thus z-buffer depth and the camera centre, unchanged)."""
        H_src, W_src = frames.shape[-2:]
        frames = torch.rot90(frames, k, dims=(-2, -1)).contiguous()
        d_m = torch.rot90(d_m, k, dims=(-2, -1)).contiguous()
        v = torch.rot90(v, k, dims=(-2, -1)).contiguous()
        rz = _RZ_90[k]
        K_new, r_new = [], []
        for K, R in zip(K_list, r_list):
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            if k == -1:
                cx2, cy2 = H_src - cy, cx
            else:
                cx2, cy2 = cy, W_src - cx
            K_new.append(np.array([[fy, 0.0, cx2], [0.0, fx, cy2], [0.0, 0.0, 1.0]],
                                  dtype=np.float64))
            r_new.append(R @ rz)
        return frames, d_m, v, K_new, r_new

    # ---- Per-clip assembly (mirrors TartanAir._load_video; BEDLAM IO + per-frame K) ----

    def _load_video(
        self, record: Dict[str, Any]
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, float,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        rgb_dir = Path(record["traj_dir"])
        depth_dir = Path(record["depth_dir"])
        seq = record["seq"]

        temporal_indices = self._temporal_sample(record["n_frames"], record["fps"])

        # --- RGB ---
        rgb_np = self._read_rgb_frames(rgb_dir, seq, temporal_indices)
        frames = torch.from_numpy(rgb_np).float().permute(0, 3, 1, 2).contiguous() / 255.0

        # --- Depth: pre-decoded cache when present, else the EXR path ---
        cached_depth = self._cached_depth(record, temporal_indices)
        if cached_depth is None:
            depth_np, valid_np = self._read_depth_frames(depth_dir, seq, temporal_indices)
            d_m = torch.from_numpy(depth_np).unsqueeze(1).float()
            v = torch.from_numpy(valid_np).unsqueeze(1).bool()
        else:
            d_m = v = None        # already final-resolution; skip rot90/crop/resize below

        # --- Cameras (per-frame K at source res; c2w in OpenCV optical frame) ---
        H_src, W_src = frames.shape[-2:]
        K_list, r_list, t_list = self._load_cameras(record, temporal_indices, (H_src, W_src))

        # --- Un-roll sideways clips (see _RZ_90 note above). k in {-1,0,1,2}; the
        # roll distribution is bimodal at 0 / -90 so round() is unambiguous. ---
        n_rot = int(round(self._clip_roll_deg(r_list) / 90.0))
        for _ in range(abs(n_rot)):
            # cache is stored post-rotation, so rotate a dummy when depth is cached
            _d = d_m if d_m is not None else torch.zeros(1, 1, *frames.shape[-2:])
            _v = v if v is not None else torch.zeros(1, 1, *frames.shape[-2:], dtype=torch.bool)
            frames, _d, _v, K_list, r_list = self._rot90_step(
                frames, _d, _v, K_list, r_list, 1 if n_rot > 0 else -1
            )
            if d_m is not None:
                d_m, v = _d, _v
        H_src, W_src = frames.shape[-2:]

        # --- Aspect-preserving center-crop + resize to (self.height, self.width) ---
        box = center_crop_box_for_aspect(H_src, W_src, self.height, self.width)
        frames = apply_spatial_crop(frames, box)
        frames = F.interpolate(frames, size=(self.height, self.width),
                               mode="bilinear", align_corners=False, antialias=True)
        if cached_depth is None:
            d_m = apply_spatial_crop(d_m, box)
            v = apply_spatial_crop(v, box)
            d_m = F.interpolate(d_m, size=(self.height, self.width), mode="nearest")
            v = F.interpolate(v.float(), size=(self.height, self.width), mode="nearest").bool()
        else:
            # invalid pixels were written as 0 before caching, and both depth and mask are
            # resized with the SAME nearest grid, so mask == (depth > 0) exactly.
            d_m = torch.from_numpy(cached_depth).unsqueeze(1).float()
            v = d_m > 0
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
