"""TartanAir dataset: RGB + GT metric depth (.npy) + NED poses -> Plucker raymaps.

Synthetic photorealistic SLAM dataset (CMU AirLab). Left camera only.

- Intrinsics: FIXED pinhole, 640x480, fov 90deg -> fx=fy=320, cx=320, cy=240.
- Depth: float32 .npy, z-buffer depth in metres. Sky/far rendered as a large
  value (>1e3); masked out as invalid.
- Poses: pose_left.txt, one line per frame `tx ty tz qx qy qz qw` in the NED
  world frame (x-forward, y-right, z-down). The quaternion is camera-to-world.
  Converted to OpenCV optical frame (x-right, y-down, z-forward) by the fixed
  permutation T (NED->cam) applied as a similarity: c2w_opt = T @ c2w_ned @ T^-1.
  This convention is empirically verified perfect (median frame-to-frame
  reprojection depth error ~0.1%); see datasets/geometry_verification/tartanair/.
- Output depths: modified disparity 1/(1+d/scale) * 2 - 1 in [-1, 1] (invalid=-1).
- Raymap moments: moments/scale clamped to [-1, 1].
- scale = max(|raw_moments|, MIN_SCALE), MIN_SCALE = 1.0 (same as VKITTI2;
  TartanAir spans indoor+outdoor so a depth-median floor is inappropriate).

Structurally identical to datasets/vkitti2.py (the geometry/normalization helpers
are reused verbatim); only record-walking, file IO and the pose convention differ.
"""
from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import imageio.v3 as iio
import numpy as np
from ._frame_io import map_frames
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

RGB_H, RGB_W = 480, 640
FX = FY = 320.0
CX, CY = 320.0, 240.0
K_FIXED = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]], dtype=np.float64)

FAR_PLANE_M = 1000.0                 # depth >= this is sky/far -> invalid
INVALID_DEPTH_SENTINEL = 1.0e4       # stuffed into invalid pixels so disparity_norm -> -1
MIN_SCALE = 1.0                      # floor for per-clip mutual scale
# Scale-preserving global divisor for TartanAir V1 (metres). ONE fixed scale (not per-clip) divides
# BOTH camera moments AND depth, so global metric scale is preserved across the whole dataset (you
# can recover metres by multiplying back) and parallax (camera/depth ratio) stays metric. Computed
# from a sample of the training set (scripts/tartanair_global_scale.py). [FILLED FROM P90 BELOW]
# Measured (scripts/tartanair_global_scale.py, 60 clips each): V1 depth-P90=22.1, V2 depth-P90=30.8;
# moment-max V1=19.7, V2=29.2. Use ONE shared scale = 30 m for V1+V2 (both truly metric -> single
# consistent metric frame; >= both moment-maxes so camera motion isn't clipped; depth disparity
# spreads well). Recover metres by multiplying back by this constant.
GLOBAL_SCALE_P90 = GLOBAL_SCALE   # single source of truth (datasets/_geometry_constants.py)

DIFFICULTIES = ("Easy", "Hard")

# NED (x-fwd, y-right, z-down) -> OpenCV optical (x-right, y-down, z-fwd)
_T3 = np.array([[0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0]], dtype=np.float64)


def _quat_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Quaternion (x,y,z,w), Hamilton convention -> 3x3 rotation matrix."""
    return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float64)


def ned_pose_to_c2w_opt(row: np.ndarray) -> np.ndarray:
    """pose row (tx ty tz qx qy qz qw) in NED -> 4x4 c2w in OpenCV optical frame."""
    t = row[:3].astype(np.float64)
    R_ned = _quat_to_rotmat(*row[3:7])
    R_opt = _T3 @ R_ned @ _T3.T      # body NED->optical (essential)
    t_opt = _T3 @ t                  # world NED->optical (cancels under first-id norm)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R_opt
    c2w[:3, 3] = t_opt
    return c2w


class TartanAirDataset(VideoDataset):
    """One record per (environment, difficulty, trajectory) triple."""

    SCENE_ID_KEY = "seq_id"   # record["seq_id"] matches captions.csv scene_id

    def __init__(self, *args, **kwargs):
        # cache parsed poses per trajectory dir: {traj_dir: (T,7) ndarray}
        self._pose_cache: Dict[str, np.ndarray] = {}
        super().__init__(*args, **kwargs)

    # ---- Records ----

    def _load_records(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for env_dir in sorted(p for p in self.data_root.iterdir() if p.is_dir()):
            for diff in DIFFICULTIES:
                diff_dir = env_dir / diff
                if not diff_dir.is_dir():
                    continue
                for traj_dir in sorted(p for p in diff_dir.iterdir() if p.is_dir()):
                    rgb_dir = traj_dir / "image_left"
                    depth_dir = traj_dir / "depth_left"
                    pose_path = traj_dir / "pose_left.txt"
                    if not (rgb_dir.is_dir() and depth_dir.is_dir() and pose_path.exists()):
                        continue
                    n_frames = sum(1 for _ in rgb_dir.glob("*_left.png"))
                    if n_frames < 2:
                        continue
                    records.append({
                        "seq_id": f"{env_dir.name}_{diff}_{traj_dir.name}",
                        "traj_dir": str(traj_dir),
                        "env": env_dir.name,
                        "difficulty": diff,
                        "n_frames": n_frames,
                        "fps": int(getattr(self.cfg, "fps_source", 10)),
                        "height": RGB_H,
                        "width": RGB_W,
                        "caption": "",
                    })

        if not records:
            raise FileNotFoundError(f"No TartanAir trajectories found under {self.data_root}")

        if not self.filtering.disable:
            pre = len(records)
            records = [r for r in records if self._filter_record(r)]
            print(f"TartanAir: filtered {pre - len(records)} from {pre}, kept {len(records)}")
        else:
            print(f"TartanAir: loaded {len(records)} trajectories from {self.data_root}")

        random.Random(0).shuffle(records)

        # train / holdout split by trajectory (records already shuffled deterministically)
        if self.split != "all":
            n_test = int(len(records) * self.test_percentage)
            if self.split == "training":
                records = records[n_test:] if n_test > 0 else records
            else:  # validation / test
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
                "TartanAirDataset does not support random crop/flip/scale augmentations; "
                "they invalidate depth/raymap alignment."
            )
        return transforms.Compose([
            transforms.Resize(
                size=(self.height, self.width),
                interpolation=transforms.InterpolationMode.BICUBIC,
            )
        ])

    # ---- Per-sequence IO ----

    def _load_poses(self, traj_dir: str) -> np.ndarray:
        if traj_dir not in self._pose_cache:
            self._pose_cache[traj_dir] = np.loadtxt(Path(traj_dir) / "pose_left.txt")
        return self._pose_cache[traj_dir]

    @staticmethod
    def _read_rgb_frames(rgb_dir: Path, frame_ids: np.ndarray) -> np.ndarray:
        def _one(fid):
            p = rgb_dir / f"{int(fid):06d}_left.png"
            return np.asarray(iio.imread(str(p)))[..., :3]
        return np.stack(map_frames(_one, frame_ids))

    @staticmethod
    def _read_depth_frames(depth_dir: Path, frame_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        def _one(fid):
            p = depth_dir / f"{int(fid):06d}_left_depth.npy"
            d = np.load(str(p)).astype(np.float32)
            m = np.isfinite(d) & (d > 0.0) & (d < FAR_PLANE_M)
            return np.where(m, d, 0.0).astype(np.float32), m
        out = map_frames(_one, frame_ids)
        return np.stack([o[0] for o in out]), np.stack([o[1] for o in out])

    def _load_cameras(
        self, record: Dict[str, Any], frame_ids: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        poses = self._load_poses(record["traj_dir"])
        K_list, r_list, t_list = [], [], []
        for fid in frame_ids:
            c2w = ned_pose_to_c2w_opt(poses[int(fid)])
            K_list.append(K_FIXED.copy())
            r_list.append(c2w[:3, :3].copy())
            t_list.append(c2w[:3, 3].copy())
        return K_list, r_list, t_list

    # ---- Geometry / normalization (reused verbatim from VKITTI2) ----

    @staticmethod
    def _normalize_first_identity(
        r_list: List[np.ndarray], t_list: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        r0_inv = r_list[0].T
        t0 = t_list[0]
        r_norm = [r0_inv @ r for r in r_list]
        t_norm = [r0_inv @ (t - t0) for t in t_list]
        return r_norm, t_norm

    @staticmethod
    def _generate_raymaps_raw(
        K_frame_list: List[np.ndarray],
        r_c2w_list: List[np.ndarray],
        t_c2w_list: List[np.ndarray],
        raymap_hw: Tuple[int, int],
        frame_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        H, W = raymap_hw
        H_f, W_f = frame_hw
        sx = W / W_f
        sy = H / H_f
        raymaps, exts, ks = [], [], []
        for K_frame, R, t in zip(K_frame_list, r_c2w_list, t_c2w_list):
            K = torch.tensor(
                [[K_frame[0, 0] * sx, 0.0,                K_frame[0, 2] * sx],
                 [0.0,                K_frame[1, 1] * sy, K_frame[1, 2] * sy],
                 [0.0,                0.0,                1.0]],
                dtype=torch.float32,
            )
            E = torch.eye(4, dtype=torch.float32)
            E[:3, :3] = torch.from_numpy(R).float()
            E[:3, 3] = torch.from_numpy(t).float()
            rays_d, rays_m = camera_to_raymap(K, E, H, W)
            raymaps.append(torch.cat([rays_d, rays_m], dim=0))
            exts.append(E)
            ks.append(K)
        return torch.stack(raymaps, dim=0), torch.stack(exts, dim=0), torch.stack(ks, dim=0)

    def _apply_mutual_scale(
        self,
        raymaps_raw: torch.Tensor,
        depth_metric: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        moments = raymaps_raw[:, 3:6]
        if is_scale_norm_mode(self.cfg):
            # Scheme C (SCALE_NORMALIZATION_ANALYSIS.md): s = kappa * median(valid depth),
            # dividing BOTH moments and depth so moments become scene-relative parallax and
            # depth lands at a fixed operating point. Shared with every geometry loader.
            return apply_scale_norm(raymaps_raw, depth_metric, self.cfg, INVALID_DEPTH_SENTINEL)
        elif self.cfg.get("scale_preserving", False):
            # GLOBAL fixed scale (not per-clip) -> preserves global metric scale across the dataset.
            # The same scale divides BOTH camera moments AND depth (below), so parallax stays metric
            # and the mapping is invertible (x GLOBAL_SCALE_P90 -> metres).
            scale = float(self.cfg.get("mutual_scale", None) or GLOBAL_SCALE_P90)   # per-dataset (yaml) else global
        else:
            moment_max = float(moments.abs().max().item())
            if self.cfg.get("depth_only", False):
                # depth_only drops the camera downstream (has_raymap=False, ray tokens
                # noise-filled and loss-masked), so there is no moment channel left to
                # condition and no depth<->pose alignment to preserve -- the two reasons
                # MIN_SCALE exists. Gauging depth on camera motion is then meaningless.
                # Measured on matrixcity: |m| median 1.71 against a 13.68 m scene, so
                # depth/scale ~ 8 and 92% of clips came out crushed (range used 0.261).
                # Gauge on the clip's own median depth instead.
                valid = (depth_metric > 0) & (depth_metric < INVALID_DEPTH_SENTINEL)
                zbar = float(depth_metric[valid].median().item()) if bool(valid.any()) else 0.0
                scale = zbar if zbar > 1e-8 else 1.0
            else:
                scale = max(moment_max, MIN_SCALE)
            if scale < 1e-8:
                scale = 1.0

        raymaps_norm = raymaps_raw.clone()
        raymaps_norm[:, 3:6] = torch.clamp(moments / scale, -1.0, 1.0)

        depth_scaled = torch.clamp(depth_metric / scale, min=0.0)
        disparity = 1.0 / (1.0 + depth_scaled)          # (0, 1]
        depth_out = disparity * 2.0 - 1.0               # (-1, 1]
        return raymaps_norm, depth_out, scale

    def _load_video(
        self, record: Dict[str, Any]
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, float,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        traj_dir = Path(record["traj_dir"])
        rgb_dir = traj_dir / "image_left"
        depth_dir = traj_dir / "depth_left"

        temporal_indices = self._temporal_sample(record["n_frames"], record["fps"])
        # Static-scene temporal reverse augmentation (opt-in via allow_temporal_reverse).
        # Reverses frames+depth+poses together (all indexed by temporal_indices) -> a valid
        # BACKWARD camera path through the unchanging scene. Only enable for STATIC datasets;
        # dynamic subclasses (spring/bedlam/mvs_synth override _load_video and don't set it).
        if getattr(self.cfg, "allow_temporal_reverse", False) and random.random() < 0.5:
            temporal_indices = temporal_indices[::-1].copy()

        # --- RGB ---
        rgb_np = self._read_rgb_frames(rgb_dir, temporal_indices)
        frames = torch.from_numpy(rgb_np).float().permute(0, 3, 1, 2).contiguous() / 255.0

        # --- Depth ---
        depth_np, valid_np = self._read_depth_frames(depth_dir, temporal_indices)
        d_m = torch.from_numpy(depth_np).unsqueeze(1).float()
        v = torch.from_numpy(valid_np).unsqueeze(1).bool()

        # --- Cameras (fixed K at source res; c2w in optical frame) ---
        K_list, r_list, t_list = self._load_cameras(record, temporal_indices)

        # --- Aspect-preserving center-crop + resize to (self.height, self.width) ---
        H_src, W_src = frames.shape[-2:]
        box = center_crop_box_for_aspect(H_src, W_src, self.height, self.width)
        frames = apply_spatial_crop(frames, box)
        d_m = apply_spatial_crop(d_m, box)
        v = apply_spatial_crop(v, box)
        frames = F.interpolate(frames, size=(self.height, self.width),
                               mode="bilinear", align_corners=False, antialias=True)
        d_m = F.interpolate(d_m, size=(self.height, self.width),
                            mode="nearest")
        v = F.interpolate(v.float(), size=(self.height, self.width),
                          mode="nearest").bool()
        frames = self.img_normalize(frames).clamp(-1.0, 1.0)
        K_list = [adjust_K_for_crop_resize(K, box, self.height, self.width) for K in K_list]

        depths_raw = d_m.clone()
        depths_valid = v
        d_for_scale = torch.where(v, d_m, torch.full_like(d_m, INVALID_DEPTH_SENTINEL))

        # --- Raymaps (uniform stride to raymap grid) ---
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
        if self.cfg.get("depth_only", False):
            # RGB+depth-ONLY dataset (e.g. matrixcity): drop the camera keys so MixtureDataset
            # sets has_raymap=False -> the algorithm noise-fills + loss-masks all ray tokens.
            # With diffusion_forcing setting [1] the objective is RGB history -> depth future.
            # Cameras were still used internally above to normalize depth's per-clip scale.
            for k in ("raymaps", "raymap_scale", "extrinsics_gt", "intrinsics_gt"):
                output.pop(k, None)
        if prompt_embeds is not None:
            output["prompt_embeds"] = prompt_embeds
            output["prompt_embed_len"] = prompt_embed_len
        return output

