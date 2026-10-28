"""Virtual KITTI 2 dataset: RGB + uint16 depth + Plucker raymaps.

- Camera: OpenCV c2w (extrinsic.txt stores w2c -> invert).
- Depth PNG: uint16, meters = u16/100. Sky/invalid = 65535.
- Output depths: modified disparity 1/(1+d/scale) * 2 - 1 in [-1, 1] (invalid=-1).
- Raymap moments: moments/scale clamped to [-1, 1].
- scale = max(|raw_moments|, MIN_SCALE), MIN_SCALE = 1.0 (same as OmniWorld-Game).
  VKITTI scenes are outdoor / sky-dominated; a depth-median floor (PointOdyssey
  style) would collapse moments because valid depths span tens of meters.
"""
from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
cv2.setNumThreads(0)  # libpng+zlib decoder is NOT thread-safe on multi-core nodes -> "bad parameters to zlib" -> imread returns None. Force single-threaded decode (dataloader workers + this module).

import imageio
import imageio.v3 as iio
import numpy as np
from ._frame_io import map_frames
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
from .video_base import VideoDataset
from ._geometry_constants import GLOBAL_SCALE
from ._scale_norm import is_scale_norm_mode, apply_scale_norm

RGB_H, RGB_W = 375, 1242
DEPTH_CM_PER_UNIT = 100.0            # meters = u16 / 100
DEPTH_SKY_SENTINEL_U16 = 65535       # VKITTI2 sky/invalid marker
INVALID_DEPTH_SENTINEL = 1.0e4       # stuffed into invalid pixels so 1/(1+d/s) ~= 0 -> disparity_norm=-1
MIN_SCALE = 1.0                      # floor for per-clip mutual scale (meters / rays_m magnitude)

SCENES = ("Scene01", "Scene02", "Scene06", "Scene18", "Scene20")
VARIANTS = (
    "clone", "fog", "morning", "overcast", "rain", "sunset",
    "15-deg-left", "15-deg-right", "30-deg-left", "30-deg-right",
)
CAMERAS = ("Camera_0", "Camera_1")


def _parse_intrinsic_txt(path: Path) -> Dict[Tuple[int, int], np.ndarray]:
    """Returns {(frame_id, cam_id): K (3,3) float64}.

    File format:
        frame cameraID K[0,0] K[1,1] K[0,2] K[1,2]
        0 0 725.0087 725.0087 620.5 187
        0 1 ...
    """
    table: Dict[Tuple[int, int], np.ndarray] = {}
    with open(path) as handle:
        header = handle.readline()
        assert header.startswith("frame cameraID"), f"unexpected header in {path}: {header!r}"
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            frame = int(parts[0])
            cam = int(parts[1])
            fx, fy, cx, cy = (float(x) for x in parts[2:6])
            K = np.array(
                [[fx, 0.0, cx],
                 [0.0, fy, cy],
                 [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            table[(frame, cam)] = K
    return table


def _parse_extrinsic_txt(path: Path) -> Dict[Tuple[int, int], np.ndarray]:
    """Returns {(frame_id, cam_id): w2c (4,4) float64}.

    File format (after header):
        frame cameraID r1,1 r1,2 r1,3 t1 r2,1 r2,2 r2,3 t2 r3,1 r3,2 r3,3 t3 0 0 0 1
    """
    table: Dict[Tuple[int, int], np.ndarray] = {}
    with open(path) as handle:
        header = handle.readline()
        assert header.startswith("frame cameraID"), f"unexpected header in {path}: {header!r}"
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            frame = int(parts[0])
            cam = int(parts[1])
            values = np.array([float(x) for x in parts[2:18]], dtype=np.float64)
            assert values.shape == (16,), f"expected 16 matrix values, got {values.shape} in {path}"
            table[(frame, cam)] = values.reshape(4, 4)
    return table


class VirtualKitti2Dataset(VideoDataset):
    """One record per (scene, variant, camera) triple."""

    _SPLIT_MAP = {"training": "train", "validation": "val", "test": "test"}
    SCENE_ID_KEY = "seq_id"   # record["seq_id"] matches captions.csv scene_id

    def __init__(self, *args, **kwargs):
        # cache per sequence: {(scene, variant): (K_tbl, E_tbl)} avoids re-parsing txt each __getitem__
        self._anno_cache: Dict[Tuple[str, str], Tuple[
            Dict[Tuple[int, int], np.ndarray], Dict[Tuple[int, int], np.ndarray]
        ]] = {}
        super().__init__(*args, **kwargs)

    # ---- Records ----

    @staticmethod
    def _mean_adj_rotation_deg_from_E_tbl(
        E_tbl: Dict[Tuple[int, int], np.ndarray], cam_id: int
    ) -> float:
        """Mean adjacent-frame rotation (deg) over frames of a single camera in
        the parsed extrinsic.txt table. Rotation angle of adjacent frames is
        invariant under inversion, so using w2c rows gives the same answer.
        """
        frames = sorted([(fid, mat) for (fid, c), mat in E_tbl.items() if c == cam_id])
        if len(frames) < 2:
            return float("nan")
        R = np.stack([m[:3, :3] for _, m in frames]).astype(np.float64)
        R_rel = np.matmul(R[1:], np.transpose(R[:-1], (0, 2, 1)))
        tr = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
        cos_a = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_a)).mean())

    def _load_records(self) -> List[Dict[str, Any]]:
        max_mean_rot = getattr(self.cfg, "max_mean_adj_rotation_deg", None)
        records: List[Dict[str, Any]] = []
        skipped_high_rotation = 0
        for scene in SCENES:
            for variant in VARIANTS:
                seq_root = self.data_root / scene / variant
                if not (seq_root / "intrinsic.txt").exists():
                    continue
                if max_mean_rot is not None:
                    try:
                        E_tbl = _parse_extrinsic_txt(seq_root / "extrinsic.txt")
                    except (OSError, ValueError, AssertionError):
                        E_tbl = None
                else:
                    E_tbl = None
                for camera in CAMERAS:
                    rgb_dir = seq_root / "frames" / "rgb" / camera
                    depth_dir = seq_root / "frames" / "depth" / camera
                    if not rgb_dir.is_dir() or not depth_dir.is_dir():
                        continue
                    n_frames = sum(1 for _ in rgb_dir.glob("rgb_*.jpg"))
                    if n_frames < 2:
                        continue
                    if max_mean_rot is not None:
                        cam_id = int(camera.split("_")[-1])
                        if E_tbl is None:
                            skipped_high_rotation += 1
                            continue
                        mean_rot = self._mean_adj_rotation_deg_from_E_tbl(E_tbl, cam_id)
                        if not np.isfinite(mean_rot) or mean_rot > float(max_mean_rot):
                            skipped_high_rotation += 1
                            continue
                    records.append({
                        "seq_id": f"{scene}_{variant}_{camera}",
                        "scene": scene,
                        "variant": variant,
                        "camera": camera,
                        "cam_id": int(camera.split("_")[-1]),
                        "n_frames": n_frames,
                        "fps": int(getattr(self.cfg, "fps_source", 10)),
                        "height": RGB_H,
                        "width": RGB_W,
                        "caption": "",
                    })

        if not records:
            raise FileNotFoundError(f"No VKITTI2 sequences found under {self.data_root}")

        if not self.filtering.disable:
            pre = len(records)
            records = [r for r in records if self._filter_record(r)]
            print(f"VKITTI2: filtered {pre - len(records)} from {pre}, kept {len(records)} "
                  f"(skipped {skipped_high_rotation} high-rotation @ >{max_mean_rot}°)")
        else:
            print(f"VKITTI2: loaded {len(records)} sequences from {self.data_root} "
                  f"(skipped {skipped_high_rotation} high-rotation @ >{max_mean_rot}°)")

        random.Random(0).shuffle(records)
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
                "VirtualKitti2Dataset does not support random crop/flip/scale augmentations; "
                "they invalidate depth/raymap alignment."
            )
        return transforms.Compose([
            transforms.Resize(
                size=(self.height, self.width),
                interpolation=transforms.InterpolationMode.BICUBIC,
            )
        ])

    # ---- Per-sequence IO ----

    def _load_anno(self, scene: str, variant: str):
        key = (scene, variant)
        if key not in self._anno_cache:
            seq_root = self.data_root / scene / variant
            K_tbl = _parse_intrinsic_txt(seq_root / "intrinsic.txt")
            E_tbl = _parse_extrinsic_txt(seq_root / "extrinsic.txt")
            self._anno_cache[key] = (K_tbl, E_tbl)
        return self._anno_cache[key]

    @staticmethod
    def _read_rgb_frames(rgb_dir: Path, frame_ids: np.ndarray) -> np.ndarray:
        def _one(fid):
            p = rgb_dir / f"rgb_{int(fid):05d}.jpg"
            return np.asarray(iio.imread(str(p)))
        return np.stack(map_frames(_one, frame_ids))

    @staticmethod
    def _read_depth_frames(depth_dir: Path, frame_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        depths, valids = [], []
        for fid in frame_ids:
            p = depth_dir / f"depth_{int(fid):05d}.png"
            # PIL (thread-safe) instead of cv2.imread: cv2's libpng+zlib path returns None
            # intermittently on these 16-bit PNGs under dataloader load even with
            # setNumThreads(0) (matches the scenenet fix). The file is fine; cv2 is flaky.
            try:
                with Image.open(p) as im:
                    u16 = np.array(im)
            except (OSError, ValueError) as e:
                raise OSError(f"cannot read depth {p}: {e}")
            if u16 is None or u16.ndim != 2:
                raise OSError(f"cannot read depth {p}")
            d = u16.astype(np.float32) / DEPTH_CM_PER_UNIT
            m = (u16 > 0) & (u16 < DEPTH_SKY_SENTINEL_U16)
            d = np.where(m, d, 0.0).astype(np.float32)
            depths.append(d)
            valids.append(m)
        return np.stack(depths), np.stack(valids)

    def _load_cameras(
        self, record: Dict[str, Any], frame_ids: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        K_tbl, E_tbl = self._load_anno(record["scene"], record["variant"])
        cam = record["cam_id"]
        K_list, r_list, t_list = [], [], []
        for fid in frame_ids:
            i = int(fid)
            K = K_tbl[(i, cam)].copy()
            c2w = np.linalg.inv(E_tbl[(i, cam)])
            K_list.append(K)
            r_list.append(c2w[:3, :3].copy())
            t_list.append(c2w[:3, 3].copy())
        return K_list, r_list, t_list

    # ---- Geometry (reused verbatim from PointOdyssey/OmniWorld) ----

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
        """K is pixel-unit at `frame_hw`; uniform rescale to `raymap_hw`.

        Since both frame and raymap grids share the same aspect (we already did
        the center-crop-to-target-aspect step upstream), sx == sy here — K stays
        geometrically valid at raymap resolution.
        """
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
        """scale_preserving=True -> ONE global constant (GLOBAL_SCALE) divides BOTH camera
        moments and depth so metric scale is preserved across the mixture. Otherwise per-clip
        scale = max(|raw moments|, MIN_SCALE). VKITTI depths span tens-hundreds of meters
        (outdoor driving, often with sky), so a depth-median floor would shrink moments to ~0.
        """
        moments = raymaps_raw[:, 3:6]
        if is_scale_norm_mode(self.cfg):
            # Scheme C (SCALE_NORMALIZATION_ANALYSIS.md): per-clip s = kappa*median(valid depth).
            return apply_scale_norm(raymaps_raw, depth_metric, self.cfg, INVALID_DEPTH_SENTINEL)
        elif self.cfg.get("scale_preserving", False):
            scale = float(self.cfg.get("mutual_scale", None) or GLOBAL_SCALE)   # per-dataset (yaml) else global
        else:
            moment_max = float(moments.abs().max().item())
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
        seq_root = self.data_root / record["scene"] / record["variant"]
        rgb_dir = seq_root / "frames" / "rgb" / record["camera"]
        depth_dir = seq_root / "frames" / "depth" / record["camera"]

        temporal_indices = self._temporal_sample(record["n_frames"], record["fps"])
        # if not getattr(self.cfg, "no_augmentations", False) and random.random() < 0.5:
        #     temporal_indices = temporal_indices[::-1].copy()

        # --- RGB ---
        rgb_np = self._read_rgb_frames(rgb_dir, temporal_indices)
        frames = torch.from_numpy(rgb_np).float().permute(0, 3, 1, 2).contiguous() / 255.0

        # --- Depth ---
        depth_np, valid_np = self._read_depth_frames(depth_dir, temporal_indices)
        d_m = torch.from_numpy(depth_np).unsqueeze(1).float()
        v = torch.from_numpy(valid_np).unsqueeze(1).bool()

        # --- Cameras (at source res) ---
        K_list, r_list, t_list = self._load_cameras(record, temporal_indices)

        # --- Aspect-preserving center-crop + resize to (self.height, self.width) ---
        # Everything (RGB, depth, mask, K) is transformed by the same crop+resize so
        # the raymaps generated below are geometrically valid at the frame grid.
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

        # --- Raymaps (uniform stride to raymap grid) ---
        r_norm, t_norm = self._normalize_first_identity(r_list, t_list)
        raymap_h = getattr(self.cfg, "raymap_height", None) or self.height
        raymap_w = getattr(self.cfg, "raymap_width", None) or self.width
        raymaps_raw, ext_gt, K_gt = self._generate_raymaps_raw(
            K_list, r_norm, t_norm, (raymap_h, raymap_w),
            frame_hw=(self.height, self.width),
        )

        raymaps, depths, scale = self._apply_mutual_scale(raymaps_raw, d_for_scale)
        # Invalid depth pixels stuffed as disparity_norm = -1 explicitly; idempotent but clearer
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
        if prompt_embeds is not None:
            output["prompt_embeds"] = prompt_embeds
            output["prompt_embed_len"] = prompt_embed_len
        return output



if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "data_root": "./data/vkitti2",
        "debug": True,
        "metadata_path": "",
        "auto_download": False,
        "force_download": False,
        "test_percentage": 0.0,
        "id_token": "",
        "height": 288,
        "width": 960,
        "n_frames": 13,
        "fps": 5,
        "fps_source": 10,
        "trim_mode": "random_cut",
        "pad_mode": "slowdown",
        "filtering": {"disable": True, "height": None, "width": None, "n_frames": None, "fps": None},
        "load_video_latent": False,
        "load_prompt_embed": False,
        "augmentation": {"random_flip": None, "ratio": None, "scale": None},
        "image_to_video": False,
        "max_text_tokens": 1,
        "check_video_path": False,
        "no_augmentations": True,
        "raymap_height": 36,
        "raymap_width": 120,
    })

    print("=" * 80)
    print("Testing VirtualKitti2Dataset")
    print("=" * 80)

    dataset = VirtualKitti2Dataset(cfg, split="training")
    print(f"dataset size: {len(dataset)}")

    item = dataset[0]
    for key, value in item.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {tuple(value.shape)} dtype={value.dtype} "
                  f"range=[{value.min().item():.3f}, {value.max().item():.3f}]")
        elif isinstance(value, dict):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value!r}")

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

    rays_up = F.interpolate(
        item["raymaps"], size=item["videos"].shape[-2:], mode="bilinear", align_corners=False
    )
    panel = torch.cat(
        [
            item["videos"],
            item["depths"].expand(-1, 3, -1, -1),
            rays_up[:, :3],
            rays_up[:, 3:],
        ],
        dim=-1,
    )
    vis = ((panel.permute(0, 2, 3, 1) / 2 + 0.5).clamp(0, 1).numpy() * 255).astype(np.uint8)
    imageio.mimwrite("test_vkitti2.mp4", vis, fps=5, quality=8)
    print("wrote test_vkitti2.mp4")
