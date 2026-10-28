"""ASE (Aria Synthetic Environments) dataset: fisheye RGB + GT range-depth + poses.

Meta's 100k synthetic Aria interior scenes. On Isambard the data is 58 SquashFS
images (mount with /projects/u6el/aak81/datasets/projectaria/mount_ase.sh).

Geometry is reverse-engineered from the authoritative projectaria_tools source
(AseCalibrationProvider.cpp) and verified to 0.06-0.10% frame i->j reprojection
error (datasets/geometry_verification/aria/check_raw_reproj.py). The legacy
datasets/aria.py uses a DIFFERENT (Harvard) convention that is WRONG for ASE
(fabricated hardware rotation, baked-in image roll + rot90, z-instead-of-range,
inverted extrinsic); do not reuse it. This loader is the correct one.

Conventions (verified):
- Raw frames: fisheye 704x704 RGB (rgb/vignette*.jpg); depth (depth/depth*.png)
  uint16 = RANGE in mm (Euclidean distance from the camera centre, NOT planar z).
- Undistort fisheye->pinhole 640x480 via the offline maps (ase_undistort_maps.npz),
  RGB bilinear / depth nearest. NO rot90 (content is already upright).
- Pinhole intrinsics K = [[350,0,319.5],[0,350,239.5],[0,0,1]] (get_linear_camera_
  calibration(640,480,350)). Range depth is converted to planar z for consistency
  with the repo's other 3D datasets: z = range / ||[(u-cx)/fx,(v-cy)/fy,1]||.
- device_T_camera = quat(w,x,y,z)=(0.94419,0.32641,0.02927,0.03336),
  t=(-0.00753,-0.01091,-0.00360), used DIRECTLY (no inverse).
- world_T_camera (= c2w) = world_T_device(trajectory.csv) @ device_T_camera.

Structurally mirrors datasets/tartanair.py (normalization/raymap helpers reused).
"""
from __future__ import annotations

import io
import random
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
cv2.setNumThreads(0)  # libpng+zlib decoder is NOT thread-safe on multi-core nodes -> "bad parameters to zlib" -> imread returns None. Force single-threaded decode (dataloader workers + this module).

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as transforms

from ._crop_utils import (
    adjust_K_for_crop_resize,
    apply_spatial_crop,
    center_crop_box_for_aspect,
)
from .aria_new import camera_to_raymap
from .video_base import VideoDataset
from ._geometry_constants import GLOBAL_SCALE
from ._scale_norm import is_scale_norm_mode, apply_scale_norm

# Offline undistort maps (pinhole 640x480 <- fisheye 704x704). Verified against
# projectaria_tools distort_by_calibration to sub-pixel accuracy.
# PORTRAIT undistort target (480x640): rot90(k=3) -> 640x480 upright LANDSCAPE (4:3), so no
# crop is needed and the full sensor FOV is kept (~85deg H x 69deg V vs ~69x54 with the old
# 640x480 maps that had to be cropped after rotation).
_MAPS_PATH = Path(__file__).parent / "geometry_verification" / "aria" / "ase_undistort_maps_portrait.npz"

PIN_W, PIN_H = 480, 640              # undistorted pinhole resolution (portrait; remap out is HxW=640x480)
FAR_PLANE_M = 1000.0
MIN_SCALE = 1.0                     # per-clip mutual scale floor (same as TartanAir/VKITTI2)


def _quat_wxyz_to_R(w: float, x: float, y: float, z: float) -> np.ndarray:
    n = (w * w + x * x + y * y + z * z) ** 0.5
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


# device_T_camera, used directly (NO inverse) — see module docstring / .cpp.
_DEVICE_T_CAMERA = np.eye(4, dtype=np.float64)
_DEVICE_T_CAMERA[:3, :3] = _quat_wxyz_to_R(
    0.9441858687689326, 0.326409343828490850, 0.029274992008313648, 0.033361059956531547)
_DEVICE_T_CAMERA[:3, 3] = [-0.007530096566173914, -0.010908549841580260, -0.003598063315542823]


def _Rz(deg: float) -> np.ndarray:
    t = np.deg2rad(deg); c, s = np.cos(t), np.sin(t)
    M = np.eye(4); M[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return M


# The Aria RGB sensor is mounted rotated, so the undistorted (portrait 640x480) frame is sideways.
# Make it UPRIGHT with np.rot90(k=3) (90deg CW) of the image, paired with a +90deg roll of the
# camera optical frame and a re-centred intrinsics. With the portrait maps the rotated frame is
# 640x480 LANDSCAPE (4:3) directly -> no crop. Verified self-consistent: reproj 0.0007, identical
# to the un-rotated frame (datasets/geometry_verification/aria/check_upright.py).
_ROT_K = 3
_UP_W, _UP_H = PIN_H, PIN_W                                   # 640 x 480 after rot90 (landscape 4:3)
_K_UPRIGHT = np.array([[350.0, 0.0, (_UP_W - 1) / 2.0],
                       [0.0, 350.0, (_UP_H - 1) / 2.0],
                       [0.0, 0.0, 1.0]], dtype=np.float64)
_DEVICE_T_CAMERA_UPRIGHT = _DEVICE_T_CAMERA @ _Rz(90.0).T


def _quat_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Quaternion (x,y,z,w), Hamilton -> 3x3 rotation."""
    return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float64)


class ASEDataset(VideoDataset):
    """One record per ASE scene (numeric dir with rgb/, depth/, trajectory.csv)."""

    SCENE_ID_KEY = "seq_id"   # record["seq_id"] matches captions.csv scene_id

    def __init__(self, *args, **kwargs):
        self._pose_cache: Dict[str, np.ndarray] = {}
        # per-scene ZipFile handles (packed scenes), opened lazily in the worker
        # process so they are fork-safe. Keyed by zip path.
        self._zip_cache: Dict[str, zipfile.ZipFile] = {}
        # undistort maps + the range->z norm map (depend only on K) are cached here
        maps = np.load(_MAPS_PATH)
        self._map_x = maps["map_x"].astype(np.float32)
        self._map_y = maps["map_y"].astype(np.float32)
        self._K_pin = maps["K_dst"].astype(np.float64)          # 640x480 pinhole intrinsics
        fx, fy = self._K_pin[0, 0], self._K_pin[1, 1]
        cx, cy = self._K_pin[0, 2], self._K_pin[1, 2]
        uu, vv = np.meshgrid(np.arange(PIN_W), np.arange(PIN_H))
        # ||[(u-cx)/fx,(v-cy)/fy,1]||: divides range to give planar z
        self._ray_norm = np.sqrt(((uu - cx) / fx) ** 2 + ((vv - cy) / fy) ** 2 + 1.0).astype(np.float32)
        super().__init__(*args, **kwargs)

    # ---- Records ----

    def _load_records(self) -> List[Dict[str, Any]]:
        # 20,880 scenes over a FUSE union mount: do ONE iterdir and DON'T touch each
        # scene (no per-scene glob/stat). n_frames is filled lazily in _load_video from
        # the trajectory (which is read there anyway). Scenes are uniform 0..N-1.
        # A scene is EITHER a loose dir `<id>/` OR a packed archive `<id>.zip` (one
        # inode instead of ~1800; see ASE_ZIP_PACKING). Enumerate both in one iterdir.
        records: List[Dict[str, Any]] = []
        loose, packed = {}, {}
        for p in self.data_root.iterdir():
            if p.name.isdigit():
                loose[p.name] = str(p)
            elif p.suffix == ".zip" and p.stem.isdigit():
                packed[p.stem] = str(p)
        for name in sorted(set(loose) | set(packed), key=int):
            zp = packed.get(name)
            records.append({
                "seq_id": f"ase_{name}",
                "scene_dir": loose.get(name, str(self.data_root / name)),
                "zip_path": zp,                         # None if loose
                "packed": zp is not None,
                "n_frames": None,                       # lazy (filled from trajectory.csv)
                "fps": int(getattr(self.cfg, "fps_source", 10)),
                "height": _UP_H,                        # upright portrait source dims
                "width": _UP_W,
                "caption": "",
            })
        if not records:
            raise FileNotFoundError(f"No ASE scenes found under {self.data_root}")

        if not self.filtering.disable:
            pre = len(records)
            records = [r for r in records if self._filter_record(r)]
            print(f"ASE: filtered {pre - len(records)} from {pre}, kept {len(records)}")
        else:
            print(f"ASE: loaded {len(records)} scenes from {self.data_root}")

        random.Random(0).shuffle(records)

        if self.split != "all":
            n_test = int(len(records) * self.test_percentage)
            if self.split == "training":
                records = records[n_test:] if n_test > 0 else records
            else:
                records = records[:n_test] if n_test > 0 else records[:1]

        records = [self.preprocess_record(r) for r in records]
        return records

    # ---- Transforms ----

    def _build_video_transforms(self, augment: bool = True):
        if augment and (
            self.augmentation.random_flip is not None
            or self.augmentation.ratio is not None
            or self.augmentation.scale is not None
        ):
            raise ValueError(
                "ASEDataset does not support random crop/flip/scale augmentations; "
                "they invalidate depth/raymap alignment."
            )
        return transforms.Compose([
            transforms.Resize(size=(self.height, self.width),
                              interpolation=transforms.InterpolationMode.BICUBIC)
        ])

    # ---- Per-sequence IO ----

    def _read_member(self, record: Dict[str, Any], rel: str) -> bytes:
        """Raw bytes of a scene member (`rgb/vignetteNNN.jpg`, `depth/depthNNN.png`,
        `trajectory.csv`) from either the packed zip or the loose dir."""
        if record.get("packed"):
            zp = record["zip_path"]
            zf = self._zip_cache.get(zp)
            if zf is None:
                zf = self._zip_cache[zp] = zipfile.ZipFile(zp, "r")
            return zf.read(rel)
        with open(Path(record["scene_dir"]) / rel, "rb") as f:
            return f.read()

    def _load_poses(self, record: Dict[str, Any]) -> np.ndarray:
        scene_dir = record["scene_dir"]
        if scene_dir not in self._pose_cache:
            rows = []
            text = self._read_member(record, "trajectory.csv").decode("utf-8", "replace")
            lines = text.splitlines()
            for line in lines[1:]:
                    p = line.strip().split(",")
                    if len(p) < 10:
                        continue
                    rows.append([float(x) for x in (p[3], p[4], p[5], p[6], p[7], p[8], p[9])])
            self._pose_cache[scene_dir] = np.asarray(rows, dtype=np.float64)
        return self._pose_cache[scene_dir]

    def _undistort_rgb(self, buf: bytes, tag: str) -> np.ndarray:
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            # missing/truncated frame (e.g. scene from an interrupted download chunk);
            # raise OSError so __getitem__'s bounded retry resamples instead of the
            # bare AttributeError killing the DataLoader worker.
            raise OSError(f"unreadable rgb {tag}")
        img = img[..., ::-1]                                      # BGR->RGB, 704x704
        return cv2.remap(np.ascontiguousarray(img), self._map_x, self._map_y,
                         interpolation=cv2.INTER_LINEAR)          # 480x640x3

    def _undistort_depth_z(self, buf: bytes, tag: str) -> Tuple[np.ndarray, np.ndarray]:
        """Decode fisheye range-depth (mm) -> undistort -> metric planar z (m) + valid mask."""
        d = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_UNCHANGED)
        if d is None:
            # cv2's libpng path sporadically returns None for these uint16 PNGs
            # depending on process state (zlib decoder-state bug family) even
            # though the bytes are intact; imageio decodes them reliably.
            try:
                import imageio.v3 as _iio
                d = np.asarray(_iio.imread(io.BytesIO(buf)))
            except Exception as e:
                raise OSError(f"unreadable depth {tag}: {e}")
        d = d.astype(np.float32)                                  # 704x704 range mm
        rng = cv2.remap(d, self._map_x, self._map_y, interpolation=cv2.INTER_NEAREST) / 1000.0
        valid = np.isfinite(rng) & (rng > 0.0) & (rng < FAR_PLANE_M)
        z = np.where(valid, rng / self._ray_norm, 0.0).astype(np.float32)  # range -> planar z
        return z, valid

    def _read_frames(self, record: Dict[str, Any], frame_ids: np.ndarray):
        # undistort (640x480 sideways) -> rot90(k=3) to upright portrait 480x640.
        # z is planar depth along the optical axis, invariant under the roll, so it is
        # computed on the sideways frame then rotated as a plain array.
        rgbs, zs, vs = [], [], []
        for fid in frame_ids:
            rgb = self._undistort_rgb(
                self._read_member(record, f"rgb/vignette{int(fid):07d}.jpg"),
                f"{record['seq_id']}#{int(fid)}")
            z, v = self._undistort_depth_z(
                self._read_member(record, f"depth/depth{int(fid):07d}.png"),
                f"{record['seq_id']}#{int(fid)}")
            rgbs.append(np.rot90(rgb, _ROT_K))
            zs.append(np.rot90(z, _ROT_K))
            vs.append(np.rot90(v, _ROT_K))
        return np.stack(rgbs), np.stack(zs), np.stack(vs)

    def _load_cameras(self, poses: np.ndarray, frame_ids: np.ndarray):
        """world_T_camera (c2w) for the UPRIGHT frame = world_T_device @ device_T_camera_upright
        (device_T_camera with the +90deg optical-axis roll baked in)."""
        K_list, r_list, t_list = [], [], []
        for fid in frame_ids:
            row = poses[int(fid)]
            wTd = np.eye(4, dtype=np.float64)
            wTd[:3, :3] = _quat_to_rotmat(row[3], row[4], row[5], row[6])
            wTd[:3, 3] = row[:3]
            c2w = wTd @ _DEVICE_T_CAMERA_UPRIGHT
            K_list.append(_K_UPRIGHT.copy())
            r_list.append(c2w[:3, :3].copy())
            t_list.append(c2w[:3, 3].copy())
        return K_list, r_list, t_list

    # ---- Geometry / normalization (reused from TartanAir/VKITTI2) ----

    @staticmethod
    def _normalize_first_identity(r_list, t_list):
        r0_inv = r_list[0].T
        t0 = t_list[0]
        r_norm = [r0_inv @ r for r in r_list]
        t_norm = [r0_inv @ (t - t0) for t in t_list]
        return r_norm, t_norm

    @staticmethod
    def _generate_raymaps_raw(K_frame_list, r_c2w_list, t_c2w_list, raymap_hw, frame_hw):
        H, W = raymap_hw
        H_f, W_f = frame_hw
        sx, sy = W / W_f, H / H_f
        raymaps, exts, ks = [], [], []
        for K_frame, R, t in zip(K_frame_list, r_c2w_list, t_c2w_list):
            K = torch.tensor(
                [[K_frame[0, 0] * sx, 0.0,                K_frame[0, 2] * sx],
                 [0.0,                K_frame[1, 1] * sy, K_frame[1, 2] * sy],
                 [0.0,                0.0,                1.0]], dtype=torch.float32)
            E = torch.eye(4, dtype=torch.float32)
            E[:3, :3] = torch.from_numpy(R).float()
            E[:3, 3] = torch.from_numpy(t).float()
            rays_d, rays_m = camera_to_raymap(K, E, H, W)
            raymaps.append(torch.cat([rays_d, rays_m], dim=0))
            exts.append(E); ks.append(K)
        return torch.stack(raymaps, 0), torch.stack(exts, 0), torch.stack(ks, 0)

    def _apply_mutual_scale(self, raymaps_raw, depth_metric):
        # scale_preserving=True -> ONE global constant (GLOBAL_SCALE) divides BOTH camera
        # moments and depth so metric scale is preserved across the mixture. Else per-clip.
        moments = raymaps_raw[:, 3:6]
        if is_scale_norm_mode(self.cfg):
            # Scheme C (SCALE_NORMALIZATION_ANALYSIS.md): per-clip s = kappa*median(valid depth).
            # d_for_scale stuffs invalid pixels with FAR_PLANE_M*10; exclude them.
            return apply_scale_norm(raymaps_raw, depth_metric, self.cfg, FAR_PLANE_M * 10)
        elif self.cfg.get("scale_preserving", False):
            scale = float(self.cfg.get("mutual_scale", None) or GLOBAL_SCALE)   # per-dataset (yaml) else global
        else:
            scale = max(float(moments.abs().max().item()), MIN_SCALE)
        if scale < 1e-8:
            scale = 1.0
        raymaps_norm = raymaps_raw.clone()
        raymaps_norm[:, 3:6] = torch.clamp(moments / scale, -1.0, 1.0)
        depth_scaled = torch.clamp(depth_metric / scale, min=0.0)
        disparity = 1.0 / (1.0 + depth_scaled)
        depth_out = disparity * 2.0 - 1.0
        return raymaps_norm, depth_out, scale

    def _load_video(self, record):
        poses = self._load_poses(record)
        n_frames = record["n_frames"]
        if n_frames is None:
            n_frames = record["n_frames"] = len(poses)      # lazy fill from trajectory
        temporal_indices = self._temporal_sample(n_frames, record["fps"])
        # Static-scene temporal reverse (opt-in): valid backward camera path (see tartanair.py).
        if getattr(self.cfg, "allow_temporal_reverse", False) and random.random() < 0.5:
            temporal_indices = temporal_indices[::-1].copy()

        rgb_np, z_np, valid_np = self._read_frames(record, temporal_indices)
        frames = torch.from_numpy(rgb_np).float().permute(0, 3, 1, 2).contiguous() / 255.0
        d_m = torch.from_numpy(z_np).unsqueeze(1).float()
        v = torch.from_numpy(valid_np).unsqueeze(1).bool()

        K_list, r_list, t_list = self._load_cameras(poses, temporal_indices)

        # Aspect-preserving crop + resize to (self.height, self.width). 640x480 and the
        # default 320x240 target share 4:3, so this is a uniform downscale (no crop).
        H_src, W_src = frames.shape[-2:]
        box = center_crop_box_for_aspect(H_src, W_src, self.height, self.width)
        frames = apply_spatial_crop(frames, box)
        d_m = apply_spatial_crop(d_m, box)
        v = apply_spatial_crop(v, box)
        frames = F.interpolate(frames, size=(self.height, self.width), mode="bilinear", align_corners=False,
                               antialias=True)
        d_m = F.interpolate(d_m, size=(self.height, self.width), mode="nearest")
        v = F.interpolate(v.float(), size=(self.height, self.width), mode="nearest").bool()
        frames = self.img_normalize(frames).clamp(-1.0, 1.0)
        K_list = [adjust_K_for_crop_resize(K, box, self.height, self.width) for K in K_list]

        depths_raw = d_m.clone()
        depths_valid = v
        d_for_scale = torch.where(v, d_m, torch.full_like(d_m, FAR_PLANE_M * 10))

        r_norm, t_norm = self._normalize_first_identity(r_list, t_list)
        raymap_h = getattr(self.cfg, "raymap_height", None) or self.height
        raymap_w = getattr(self.cfg, "raymap_width", None) or self.width
        raymaps_raw, ext_gt, K_gt = self._generate_raymaps_raw(
            K_list, r_norm, t_norm, (raymap_h, raymap_w), frame_hw=(self.height, self.width))

        raymaps, depths, scale = self._apply_mutual_scale(raymaps_raw, d_for_scale)
        depths = torch.where(depths_valid, depths, torch.full_like(depths, -1.0))
        return frames, raymaps, depths, scale, ext_gt, K_gt, depths_raw, depths_valid

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Bounded retry LOOP (not recursion): recursive retries pin each level's
        # partially-decoded tensors via the exception traceback; with many failing
        # records (e.g. bedlam2's misnamed depth files) DataLoader workers ballooned
        # to ~100GB and were OOM-killed, silently losing the sample.
        for _attempt in range(8):
            record = self.records[idx]
            try:
                frames, raymaps, depths, scale, ext_gt, K_gt, depths_raw, depths_valid = self._load_video(record)
                break
            except (OSError, ValueError, KeyError, RuntimeError) as e:
                print(f"Warning: skipping {record.get('seq_id', idx)}: {e}")
                # backoff: transient filesystem stalls fail all instant
                # retries at once; wait it out before resampling
                time.sleep(min(5.0, 0.3 * 2 ** _attempt))
                idx = random.randint(0, len(self.records) - 1)
        else:
            record = self.records[0]
            frames, raymaps, depths, scale, ext_gt, K_gt, depths_raw, depths_valid = self._load_video(record)

        video_metadata = {
            "num_frames": frames.shape[0],
            "height": frames.shape[2],
            "width": frames.shape[3],
            "has_caption": False,
        }
        # Use the caption cache when configured. `_select_caption` returns
        # (caption, ABSOLUTE prompt-embed path) for this scene, picking a random
        # caption variant per __getitem__. Previously this loader read only
        # record["caption"] and its _load_prompt_embed returned a ZERO tensor, so a
        # configured caption_cache_dir was silently ignored and every clip trained
        # with a null text embedding.
        _cap = self._select_caption(record)
        _embed_override = None
        if _cap is not None:
            _caption, _embed_override = _cap
        else:
            _caption = record.get("caption", "")
        prompts = self.id_token + _caption
        prompt_embeds, prompt_embed_len = None, None
        # A cache miss is NOT a reason to drop the sample. `_load_prompt_embed_or_null`
        # falls back to the null embedding (what this subset trained on before captions
        # were wired up) instead of raising into MixtureDataset's except/continue, which
        # would make every uncaptioned scene unreachable.
        _has_caption = _embed_override is not None or bool(_caption)
        if self.load_prompt_embed:
            (
                prompt_embeds,
                prompt_embed_len,
                _has_caption,
            ) = self._load_prompt_embed_or_null(
                record, path_override=_embed_override
            )
        # Keep the metadata honest: a cache hit means this clip really does have a
        # caption, regardless of what record["caption"] said at index time.
        video_metadata["has_caption"] = _has_caption

        record["height"] = video_metadata["height"]
        record["width"] = video_metadata["width"]
        has_bbox, bbox_render = self._render_bbox(record)

        output: Dict[str, Any] = {
            "videos": frames,
            "raymaps": raymaps,
            "depths": depths,
            "raymap_scale": torch.tensor(scale, dtype=torch.float32),
            "extrinsics_gt": ext_gt,
            "intrinsics_gt": K_gt,
            "depths_raw": depths_raw,
            "depths_valid": depths_valid,
            "video_metadata": video_metadata,
            "bbox_render": bbox_render,
            "has_bbox": has_bbox,
            "prompts": prompts,
        }
        if self._clip_motion_too_fast(ext_gt, depths_raw, depths_valid):
            # PER-CLIP motion gate: ASE is head-mounted walking at an intrinsic
            # 10 fps (100 ms steps, nothing denser to sample), so some windows
            # whip around far faster than the rest of the mixture. Those clips
            # supervise DEPTH ONLY: dropping the camera keys makes the mixture
            # set has_raymap=False -> ray tokens noise-filled + loss-masked, and
            # the algorithm additionally bars settings 2/3 (rays-as-history)
            # for has_raymap=False samples. Calm windows keep full supervision.
            for k in ("raymaps", "raymap_scale", "extrinsics_gt", "intrinsics_gt"):
                output.pop(k)
        if prompt_embeds is not None:
            output["prompt_embeds"] = prompt_embeds
            output["prompt_embed_len"] = prompt_embed_len
        return output

    def _clip_motion_too_fast(self, ext_gt, depths_raw, depths_valid) -> bool:
        """True if this sampled window's camera motion is out-of-family:
        mean adjacent rotation > cfg.raymap_max_rot_deg_per_frame (default 4.0)
        OR mean adjacent translation / median scene depth (parallax/frame)
        > cfg.raymap_max_parallax (default 0.08). Both None/0 => gate disabled."""
        max_rot = self.cfg.get("raymap_max_rot_deg_per_frame", None)
        max_par = self.cfg.get("raymap_max_parallax", None)
        if not max_rot and not max_par:
            return False
        R = ext_gt[:, :3, :3]
        tvec = ext_gt[:, :3, 3]
        if max_rot:
            R_rel = torch.matmul(R[1:], R[:-1].transpose(1, 2))
            tr = R_rel.diagonal(dim1=1, dim2=2).sum(-1)
            adj_rot = torch.rad2deg(torch.arccos(((tr - 1) / 2).clamp(-1, 1)))
            if float(adj_rot.mean()) > float(max_rot):
                return True
        if max_par:
            dv = depths_raw[depths_valid.bool()]
            dv = dv[dv > 0]
            if dv.numel():
                adj_tra = (tvec[1:] - tvec[:-1]).norm(dim=-1)
                if float(adj_tra.mean()) / float(dv.median()) > float(max_par):
                    return True
        return False

