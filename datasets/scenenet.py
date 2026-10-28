"""SceneNet RGBD: RGB + uint16-mm depth + protobuf-derived OpenCV c2w + Plucker raymaps.

- Camera: OpenCV c2w (X right, Y down, Z forward). Derived from pySceneNetRGBD's
  look-at convention; see `_lookat_to_opencv_c2w` for the math.
- Depth PNG: uint16 mm, valid in (0, 60000) mm. SceneNet stores **Euclidean ray
  length** ("depth is defined as the euclidean ray length from the camera position
  to the first point of intersection"), so we divide by sqrt(x_n^2 + y_n^2 + 1) at
  the native resolution to convert it to z-depth before any downstream use. This
  way standard pinhole unprojection (x_n*z, y_n*z, z) recovers the correct point.
- Output depths: modified disparity 1/(1 + d/scale) * 2 - 1 in [-1, 1] (invalid=-1).
- Raymap moments: moments/scale clamped to [-1, 1].
- scale = max(|raw_moments|.max(), depth_median) — point_odyssey-style indoor floor.
"""
from __future__ import annotations

import io
import random
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
cv2.setNumThreads(0)  # libpng+zlib PNG decode is NOT thread-safe on multi-core -> "bad parameters
# to zlib" -> cv2.imread returns None intermittently under dataloader workers. Force single-thread.
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2 as transforms

from ._crop_utils import (
    adjust_K_for_crop_resize,
    apply_spatial_crop,
    center_crop_box_for_aspect,
)
from .aria_new import camera_to_raymap
from ._scale_norm import apply_scale_norm, is_scale_norm_mode
from .video_base import VideoDataset

# ---- Source-format constants (do NOT depend on cfg) ----
SCENENET_SRC_H = 240
SCENENET_SRC_W = 320
SCENENET_VFOV_DEG = 45.0
SCENENET_HFOV_DEG = 60.0
SCENENET_FX = (SCENENET_SRC_W / 2.0) / np.tan(np.deg2rad(SCENENET_HFOV_DEG / 2.0))   # ≈ 277.128
SCENENET_FY = (SCENENET_SRC_H / 2.0) / np.tan(np.deg2rad(SCENENET_VFOV_DEG / 2.0))   # ≈ 289.708
SCENENET_CX = SCENENET_SRC_W / 2.0
SCENENET_CY = SCENENET_SRC_H / 2.0

DEPTH_MM_PER_METER = 1000.0
DEPTH_MAX_VALID_MM = 60000
INVALID_DEPTH_SENTINEL = 1.0e4   # meters; stuffed into invalid pixels for mutual-scale
EPS = 1e-8


def _ray_length_to_z_grid() -> np.ndarray:
    """Per-pixel sqrt((x-cx)^2/fx^2 + (y-cy)^2/fy^2 + 1) at native 320x240.

    SceneNet depth values are Euclidean ray lengths; dividing by this grid
    converts them to z-depth (perpendicular distance to the image plane).
    """
    ys, xs = np.mgrid[0:SCENENET_SRC_H, 0:SCENENET_SRC_W].astype(np.float32)
    x_n = (xs + 0.5 - SCENENET_CX) / SCENENET_FX
    y_n = (ys + 0.5 - SCENENET_CY) / SCENENET_FY
    return np.sqrt(x_n * x_n + y_n * y_n + 1.0).astype(np.float32)


_RAY_NORM_NATIVE: np.ndarray = _ray_length_to_z_grid()  # (240, 320)


def _lookat_to_opencv_c2w(camera: np.ndarray, lookat: np.ndarray,
                          world_up: np.ndarray = np.array([0.0, 1.0, 0.0])) -> np.ndarray:
    """SceneNet (camera_pos, lookat_pos) -> OpenCV c2w (4,4) float64.

    Mirrors pySceneNetRGBD/util.py::world_to_camera_with_pose: the camera frame
    has +X right, +Y down (= -world_up after the cross product), +Z forward
    (camera -> lookat). This is OpenCV convention, NOT OpenGL.
    """
    forward = lookat - camera
    forward = forward / (np.linalg.norm(forward) + EPS)
    right = np.cross(forward, world_up)
    right = right / (np.linalg.norm(right) + EPS)
    down = np.cross(forward, right)
    down = down / (np.linalg.norm(down) + EPS)
    R_c2w = np.column_stack([right, down, forward])      # 3x3
    t_c2w = np.asarray(camera, dtype=np.float64)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R_c2w
    c2w[:3, 3] = t_c2w
    return c2w


def scenenet_intrinsics() -> np.ndarray:
    """Pixel-unit K at the native 320x240 resolution. Constant across the dataset."""
    return np.array([
        [SCENENET_FX, 0.0,         SCENENET_CX],
        [0.0,         SCENENET_FY, SCENENET_CY],
        [0.0,         0.0,         1.0],
    ], dtype=np.float64)


class SceneNetRGBDDataset(VideoDataset):
    """SceneNet RGBD with GT camera poses (read from `cfg.gt_pose_root`).

    A shared dataset directory is read-only for most users, so
    the per-trajectory `gt_pose/cameras.npz` files (generated once by
    `scripts/preprocess_scenenet_protobufs.py`) live at a separate writable
    root. If `cfg.gt_pose_root` is null/missing, falls back to `cfg.data_root`
    (useful if you've extracted the poses into the dataset directory itself).
    """

    SCENE_ID_KEY = "video_path"   # record["video_path"] matches captions.csv scene_id

    # ---- IO ----

    def _zip(self, zip_path: str) -> zipfile.ZipFile:
        if not hasattr(self, "_zip_cache"):
            self._zip_cache: Dict[str, zipfile.ZipFile] = {}
        zf = self._zip_cache.get(zip_path)
        if zf is None:
            zf = self._zip_cache[zip_path] = zipfile.ZipFile(zip_path, "r")
        return zf

    def __getstate__(self):
        # don't ship open zip handles to spawn workers; each reopens lazily in-worker.
        s = self.__dict__.copy()
        s.pop("_zip_cache", None)
        return s

    def _frame_bytes(self, traj_dir: Path, sub: str, fn: int, ext: str) -> bytes:
        """Read frame <fn> of subdir <sub> from the trajectory's frames.zip (inode-packed
        format: member '<sub>/<fn>.<ext>'), falling back to the loose file if no zip."""
        zp = traj_dir / "frames.zip"
        if zp.is_file():
            return self._zip(str(zp)).read(f"{sub}/{int(fn)}.{ext}")
        return (traj_dir / sub / f"{int(fn)}.{ext}").read_bytes()

    def _read_rgb_frames(self, traj_dir: Path, frame_nums: np.ndarray) -> np.ndarray:
        frames = []
        for fn in frame_nums:
            with Image.open(io.BytesIO(self._frame_bytes(traj_dir, "photo", fn, "jpg"))) as im:
                frames.append(np.asarray(im.convert("RGB")))
        return np.stack(frames)

    def _read_depth_frames(self, traj_dir: Path, frame_nums: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
        depths, valids = [], []
        for fn in frame_nums:
            # PIL (thread-safe) instead of cv2.imread, which returns None intermittently on
            # these 16-bit PNGs under multi-threaded libpng (matches scenenet_depth.py).
            try:
                with Image.open(io.BytesIO(self._frame_bytes(traj_dir, "depth", fn, "png"))) as im:
                    u16 = np.array(im)
            except (OSError, ValueError, KeyError) as e:
                raise OSError(f"cannot read depth {fn} in {traj_dir}: {e}")
            if u16 is None or u16.ndim != 2:
                raise OSError(f"cannot read depth {fn} in {traj_dir}")
            valid = (u16 > 0) & (u16 < DEPTH_MAX_VALID_MM)
            # SceneNet stores Euclidean ray length; convert to z-depth so that
            # standard pinhole unprojection (x_n*z, y_n*z, z) recovers the
            # correct 3D point.
            d_eucl_m = u16.astype(np.float32) / DEPTH_MM_PER_METER
            d_z = d_eucl_m / _RAY_NORM_NATIVE
            d_z = np.where(valid, d_z, 0.0).astype(np.float32)
            depths.append(d_z)
            valids.append(valid)
        return np.stack(depths), np.stack(valids)

    # ---- Geometry ----

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
        K_list: List[np.ndarray],
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
        for K_frame, R, t in zip(K_list, r_c2w_list, t_c2w_list):
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
        return torch.stack(raymaps, 0), torch.stack(exts, 0), torch.stack(ks, 0)

    def _apply_mutual_scale(
        self,
        raymaps_raw: torch.Tensor,
        depth_metric_for_scale: torch.Tensor,
        depth_metric_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """scale = max(|raw moments|, depth_median).

        SceneNet is indoor with bounded depths (<= 60 m). PointOdyssey-style
        depth_median floor anchors near-stationary clips so |M| ~= 0 cases
        don't blow disparity to -1.

        Under scheme C (`dataset.scale_mode=parallax`) this instead divides BOTH the
        moments and depth by the single per-clip scalar s = kappa*median(valid z-depth),
        with NO moment clamp, so depth<->pose stay EXACTLY aligned (identical recipe and
        alignment guarantee as every other geometry loader; SCALE_NORMALIZATION_ANALYSIS.md).
        `depth_metric_for_scale` carries INVALID_DEPTH_SENTINEL at invalid pixels.
        """
        if is_scale_norm_mode(self.cfg):
            return apply_scale_norm(raymaps_raw, depth_metric_for_scale, self.cfg,
                                  INVALID_DEPTH_SENTINEL)

        moments = raymaps_raw[:, 3:6]
        moment_max = float(moments.abs().max().item())

        depth_valid_only = depth_metric_for_scale[depth_metric_valid]
        if depth_valid_only.numel() > 0:
            depth_median = float(depth_valid_only.median().item())
        else:
            depth_median = 1.0
        scale = max(moment_max, depth_median)
        if scale < EPS:
            scale = 1.0

        raymaps_norm = raymaps_raw.clone()
        raymaps_norm[:, 3:6] = torch.clamp(moments / scale, -1.0, 1.0)

        depth_scaled = torch.clamp(depth_metric_for_scale / scale, min=0.0)
        disparity = 1.0 / (1.0 + depth_scaled)
        depth_out = disparity * 2.0 - 1.0
        return raymaps_norm, depth_out, scale

    # ---- Build transforms ----

    def _build_video_transforms(self, augment: bool = True):
        if augment and (
            self.augmentation.random_flip is not None
            or self.augmentation.ratio is not None
            or self.augmentation.scale is not None
        ):
            raise ValueError(
                "SceneNetRGBDDataset does not support random crop/flip/scale augmentations; "
                "they invalidate depth/raymap alignment."
            )
        return transforms.Compose([
            transforms.Resize(
                size=(self.height, self.width),
                interpolation=transforms.InterpolationMode.BICUBIC,
            )
        ])

    # ---- Records ----

    def _gt_pose_root(self) -> Path:
        gt_root_cfg = getattr(self.cfg, "gt_pose_root", None)
        return Path(gt_root_cfg).resolve() if gt_root_cfg else self.data_root

    def _pose_path_for(self, record: Dict[str, Any]) -> Path:
        """gt_pose/cameras.npz path derived from scene_id/trajectory_id/split.

        Kept independent of any cached `pose_path` field so a stale records-manifest
        (written before this field existed, or by a base walk) can't break loading.
        """
        gt_root = self._gt_pose_root()
        scene_id = int(record["scene_id"])
        traj_id = int(record["trajectory_id"])
        split_dir = "train" if record.get("split", "training") == "training" else "val"
        return gt_root / split_dir / str(scene_id) / str(traj_id) / "gt_pose" / "cameras.npz"

    def _load_records(self) -> List[Dict[str, Any]]:
        records = super()._load_records()
        kept: List[Dict[str, Any]] = []
        n_missing_pose = 0
        for r in records:
            pose_path = self._pose_path_for(r)
            if not pose_path.is_file():
                n_missing_pose += 1
                continue
            r["pose_path"] = str(pose_path)   # absolute (gt_root may differ from data_root)
            kept.append(r)
        print(f"SceneNetRGBD: filtered {n_missing_pose} records missing gt_pose/cameras.npz; "
              f"kept {len(kept)} / {len(kept) + n_missing_pose}")
        return kept

    # ---- Per-record loading ----

    def _load_video(
        self, record: Dict[str, Any]
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, float,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        """Returns:
          frames       (T, 3, H, W)   in [-1, 1]
          raymaps      (T, 6, rH, rW) dirs unit, moments in [-1, 1]
          depths       (T, 1, H, W)   modified-disparity in [-1, 1] (invalid=-1)
          raymap_scale float
          ext_gt       (T, 4, 4)      OpenCV c2w, first-frame identity
          K_gt         (T, 3, 3)      pixel-unit K at raymap resolution
          depths_raw   (T, 1, H, W)   float32 meters (invalid=0)
          depths_valid (T, 1, H, W)   bool
        """
        # Trajectory dir holds frames.zip (inode-packed: members photo/<f>.jpg, depth/<f>.png)
        # or the loose photo/ + depth/ subdirs. video_path is "<split>/<scene>/<traj>/photo".
        traj_dir = (self.data_root / record["video_path"]).parent
        # Resolve pose path from ids rather than trusting a cached field: a records-manifest
        # written by a base walk (or before pose_path existed) stores pose_path=None.
        pose_path = record.get("pose_path") or str(self._pose_path_for(record))
        pose = np.load(pose_path)
        c2w_all = pose["c2w"]                # (T_src, 4, 4) float32
        frame_nums_all = pose["frame_nums"]  # (T_src,) int32
        K_native = pose["K"].astype(np.float64)  # (3, 3)

        n_src = int(frame_nums_all.shape[0])
        # SceneNet photos are ALREADY sparsely sampled (one every 25 render-frames, ~300/traj);
        # record["fps"]=30 is the RENDER rate, not the photo cadence. Feeding 30 as src_fps makes
        # _temporal_sample decimate 3x (target_len=round(n_frames/self.fps*30) -> stride 3, skips
        # 2/3 of the photos). Treat the photos as the frames at the target rate -> stride 1, no
        # skipping (self.fps as src_fps => target_len == n_frames). Effective output fps = self.fps.
        fps = int(self.fps)

        # --- Temporal sample (use base class fps-aware version) ---
        temporal_indices = self._temporal_sample(n_src, fps)
        if not getattr(self.cfg, "no_augmentations", False) and random.random() < 0.5:
            temporal_indices = temporal_indices[::-1].copy()
        sel = np.asarray(temporal_indices, dtype=np.int64)
        sel_frame_nums = frame_nums_all[sel]                      # (T,) int32

        # --- RGB ---
        rgb_np = self._read_rgb_frames(traj_dir, sel_frame_nums)
        frames = torch.from_numpy(rgb_np).float().permute(0, 3, 1, 2).contiguous() / 255.0

        # --- Depth ---
        depth_np, valid_np = self._read_depth_frames(traj_dir, sel_frame_nums)
        d_m = torch.from_numpy(depth_np).unsqueeze(1).float()
        v = torch.from_numpy(valid_np).unsqueeze(1).bool()

        # --- Cameras (pixel-unit K constant; c2w per frame) ---
        K_list = [K_native.copy() for _ in range(sel.shape[0])]
        r_list = [c2w_all[i, :3, :3].astype(np.float64) for i in sel]
        t_list = [c2w_all[i, :3, 3].astype(np.float64) for i in sel]

        # --- Aspect-preserving center-crop + resize to (self.height, self.width) ---
        H_src, W_src = frames.shape[-2:]
        box = center_crop_box_for_aspect(H_src, W_src, self.height, self.width)
        frames = apply_spatial_crop(frames, box)
        d_m = apply_spatial_crop(d_m, box)
        v = apply_spatial_crop(v, box)
        frames = F.interpolate(frames, size=(self.height, self.width),
                               mode="bilinear", align_corners=False, antialias=True)
        d_m = F.interpolate(d_m, size=(self.height, self.width),
                            mode="nearest")  # nearest, NOT bilinear: bilinear mixes the 0-stuffed
                            # invalid pixels into valid depths at mask boundaries and creates
                            # flying-pixel depths at object edges (mask is nearest -> misaligned).
                            # Matches tartanair/spring/bedlam/ase/mvs_synth/matrixcity/mid_air.
        v = F.interpolate(v.float(), size=(self.height, self.width),
                          mode="nearest").bool()
        frames = self.img_normalize(frames).clamp(-1.0, 1.0)
        K_list = [adjust_K_for_crop_resize(K, box, self.height, self.width) for K in K_list]

        depths_raw = d_m.clone()
        depths_valid = v
        d_for_scale = torch.where(v, d_m, torch.full_like(d_m, INVALID_DEPTH_SENTINEL))

        # --- Raymaps (first-frame identity, then mutual scale) ---
        r_norm, t_norm = self._normalize_first_identity(r_list, t_list)
        raymap_h = getattr(self.cfg, "raymap_height", None) or self.height
        raymap_w = getattr(self.cfg, "raymap_width", None) or self.width
        raymaps_raw, ext_gt, K_gt = self._generate_raymaps_raw(
            K_list, r_norm, t_norm, (raymap_h, raymap_w),
            frame_hw=(self.height, self.width),
        )

        raymaps, depths, scale = self._apply_mutual_scale(raymaps_raw, d_for_scale, depths_valid)
        depths = torch.where(depths_valid, depths, torch.full_like(depths, -1.0))
        return frames, raymaps, depths, scale, ext_gt, K_gt, depths_raw, depths_valid

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Bounded retry LOOP (not recursion), matching tartanair: recursive retries pin
        # each level's partially-decoded tensors via the exception traceback, and when a
        # whole subset is unreadable they turn a clean OSError into a RecursionError --
        # which is exactly how this loader's pose tree going missing stayed invisible.
        for _attempt in range(8):
            record = self.records[idx]
            try:
                frames, raymaps, depths, scale, ext_gt, K_gt, depths_raw, depths_valid = self._load_video(record)
                break
            except (OSError, ValueError, KeyError, RuntimeError) as e:
                traj_id = record.get("trajectory_id", "?")
                scene_id = record.get("scene_id", "?")
                print(f"Warning: skipping scene {scene_id} traj {traj_id}: {e}")
                # backoff: transient filesystem stalls fail all instant retries at once
                time.sleep(min(5.0, 0.3 * 2 ** _attempt))
                idx = random.randint(0, len(self.records) - 1)
        else:
            record = self.records[0]
            frames, raymaps, depths, scale, ext_gt, K_gt, depths_raw, depths_valid = self._load_video(record)

        video_metadata = {
            "num_frames": frames.shape[0],
            "height": frames.shape[2],
            "width": frames.shape[3],
            "has_caption": bool(record.get("caption", "")),
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
        if prompt_embeds is not None:
            output["prompt_embeds"] = prompt_embeds
            output["prompt_embed_len"] = prompt_embed_len
        return output



if __name__ == "__main__":
    import imageio as iio_write
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "data_root": "data/scenenetrgbd",
        "metadata_path": "metadata.csv",
        "gt_pose_root": "data/scenenet_rgbd_gt_poses",
        "auto_download": False, "force_download": False, "test_percentage": 0.0,
        "id_token": "", "height": 240, "width": 320, "n_frames": 13, "fps": 10,
        "trim_mode": "random_cut", "pad_mode": "slowdown",
        "filtering": {"disable": True, "height": None, "width": None, "n_frames": None, "fps": None},
        "load_video_latent": False, "load_prompt_embed": False,
        "augmentation": {"random_flip": None, "ratio": None, "scale": None},
        "image_to_video": False, "max_text_tokens": 1, "check_video_path": False,
        "no_augmentations": True, "raymap_height": 30, "raymap_width": 40, "debug": True,
    })
    print("=" * 80); print("Testing SceneNetRGBDDataset"); print("=" * 80)
    ds = SceneNetRGBDDataset(cfg, split="validation")
    print(f"size: {len(ds)}")
    item = ds[0]
    for k, v in item.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)} dtype={v.dtype} range=[{v.min().item():.3f}, {v.max().item():.3f}]")
        elif isinstance(v, dict):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v!r}")

    assert item["videos"].shape == (cfg.n_frames, 3, cfg.height, cfg.width)
    assert item["depths"].shape == (cfg.n_frames, 1, cfg.height, cfg.width)
    assert item["raymaps"].shape[:2] == (cfg.n_frames, 6)
    assert item["raymaps"][:, :3].abs().max().item() <= 1.0 + 1e-6
    assert item["raymaps"][:, 3:].abs().max().item() <= 1.0 + 1e-6
    assert item["depths"].abs().max().item() <= 1.0 + 1e-6
    assert item["extrinsics_gt"].shape == (cfg.n_frames, 4, 4)
    assert item["intrinsics_gt"].shape == (cfg.n_frames, 3, 3)
    assert torch.allclose(item["extrinsics_gt"][0], torch.eye(4), atol=1e-3)
    print("all assertions passed")

    rays_up = F.interpolate(item["raymaps"], size=item["videos"].shape[-2:],
                            mode="bilinear", align_corners=False)
    panel = torch.cat([item["videos"], rays_up[:, :3], rays_up[:, 3:],
                       item["depths"].expand(-1, 3, -1, -1)], dim=-1)
    vis = ((panel.permute(0, 2, 3, 1) / 2 + 0.5).clamp(0, 1).numpy() * 255).astype(np.uint8)
    iio_write.mimwrite("test_scenenet.mp4", vis, fps=cfg.fps, quality=8)
    print("wrote test_scenenet.mp4")
