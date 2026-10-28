"""DL3DV with the ORIGINAL COLMAP-derived poses (not ViPE) — NVS-only.

Reads a processed DL3DV tree that stores DL3DV's official COLMAP-derived cameras in two
layouts (verified against ViPE poses: 0.08-0.10 deg rotation / translation-direction
error for the convention used here, 60-100 deg for the alternatives):

  dense (1K~7K): <split>/<scene>/dense/rgb/frame_%05d.png  (954x536)
                 <split>/<scene>/dense/cam/frame_%05d.npz  {intrinsic (3,3) at
                 rgb res, pose (4,4) = c2w OpenCV}
  flat  (8K~10K): <split>/<scene>/images_4/frame_%05d.png  (960x540)
                  <split>/<scene>/transforms.json  (nerfstudio: c2w OpenGL,
                  K at full 3840x2160; frames contiguous, every frame has a png)

COLMAP scale is ARBITRARY per scene (measured ~0.5-0.6x ViPE-metric on probe
scenes) and there is NO depth — so this is a pure novel-view-synthesis subset
like re10k: rays as history, RGB as target (diffusion-forcing settings [2, 3]).
The mixture auto-injects has_depth=False (depths key absent).
"""
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as transforms
from tqdm import tqdm

from ._crop_utils import (
    adjust_K_for_crop_resize,
    apply_spatial_crop,
    center_crop_box_for_aspect,
)
from .aria_new import camera_to_raymap
from .video_base import VideoDataset

MIN_RAYMAP_SCALE = 1.0  # floor for per-clip max-abs moment normalization
# P90 of raw |moments| over 60 random clips at mixture geometry (first-frame-
# identity, 288x384), measured 2026-07-07 -> datasets/dl3dv_colmap_moment_stats.json
# (p50 0.37 / p90 1.97 / p95 2.86 / max 9.5). Used when scale_preserving=True so
# moment magnitude stays comparable across clips (COLMAP scale is arbitrary per
# scene, so unlike the metric datasets this preserves only the DATASET-level
# moment distribution, not metres). Unclamped, like re10k/dl3dv's fixed divisors.
GLOBAL_MOMENT_DIVISOR_P90 = 2.0

# nerfstudio/OpenGL c2w -> OpenCV c2w (flip y/z camera axes)
_GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


class DL3DVColmapDataset(VideoDataset):

    def _load_records(self) -> List[Dict[str, Any]]:
        data_root = Path(self.cfg.data_root)
        splits = list(getattr(self.cfg, "splits", None) or [])
        splits_flat = list(getattr(self.cfg, "splits_flat", None) or [])

        records: List[Dict[str, Any]] = []
        for split in splits:
            split_dir = data_root / split
            if not split_dir.is_dir():
                continue
            for scene in tqdm(sorted(d.name for d in split_dir.iterdir() if d.is_dir()),
                              desc=f"Loading DL3DV-COLMAP {split} (dense)"):
                cam_dir = split_dir / scene / "dense" / "cam"
                rgb_dir = split_dir / scene / "dense" / "rgb"
                if not cam_dir.is_dir() or not rgb_dir.is_dir():
                    continue
                records.append({
                    "obj_id": scene, "split": split, "layout": "dense",
                    "cam_dir": str(cam_dir), "rgb_dir": str(rgb_dir),
                    "fps": 30,  # DL3DV captures are 30 fps walkthroughs
                    "caption": self._caption_for(split, scene),
                })
        for split in splits_flat:
            split_dir = data_root / split
            if not split_dir.is_dir():
                continue
            for scene in tqdm(sorted(d.name for d in split_dir.iterdir() if d.is_dir()),
                              desc=f"Loading DL3DV-COLMAP {split} (flat)"):
                tj = split_dir / scene / "transforms.json"
                rgb_dir = split_dir / scene / "images_4"
                if not tj.is_file() or not rgb_dir.is_dir():
                    continue
                records.append({
                    "obj_id": scene, "split": split, "layout": "flat",
                    "transforms_path": str(tj), "rgb_dir": str(rgb_dir),
                    "fps": 30,
                    "caption": self._caption_for(split, scene),
                })

        print(f"Found {len(records)} DL3DV-COLMAP scenes "
              f"(dense splits {splits}, flat splits {splits_flat}).")

        if not self.filtering.disable:
            records = [r for r in records if self._filter_record(r)]

        # Same prompt-embed layout as DL3DVDataset: <root>/<split>/<obj_id>.pt,
        # zeros fallback at load time when the file doesn't exist.
        embeds_root = Path(self.cfg.prompt_embeds_root)
        n_with_embed = 0
        for r in records:
            embed_path = embeds_root / r["split"] / f"{r['obj_id']}.pt"
            r["prompt_embed_path"] = str(embed_path)
            if embed_path.exists():
                n_with_embed += 1
        print(f"DL3DV-COLMAP prompt embeds: {n_with_embed} / {len(records)} cached "
              f"under {embeds_root}; the rest use zero embeddings.")

        random.Random(0).shuffle(records)
        n_test = int(len(records) * self.test_percentage)
        if self.split == "training":
            records = records[n_test:]
        elif self.split in ["validation", "test"]:
            records = records[:n_test]
        return [self.preprocess_record(r) for r in records]

    def preprocess_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return record

    def _caption_for(self, split, obj_id):
        """captions.csv at <data_root>/captions.csv (split,scene_id,frame_indices,caption)
        — same file the ViPE-based DL3DVDataset uses (rgb_root == this data_root)."""
        if not hasattr(self, "_caption_map"):
            self._caption_map = {}
            import csv as _csv
            cap_csv = Path(self.cfg.data_root) / "captions.csv"
            try:
                with open(cap_csv, newline="") as f:
                    for row in _csv.DictReader(f):
                        self._caption_map[(row.get("split"), row.get("scene_id"))] = \
                            row.get("caption", "") or ""
            except Exception as e:
                print(f"[dl3dv_colmap] captions.csv not loaded ({e}); captions empty")
        return self._caption_map.get((split, obj_id), "")

    def _build_video_transforms(self, augment: bool = True):
        """Resize only — no random crop to preserve camera alignment."""
        return transforms.Compose([
            transforms.Resize(size=(self.height, self.width),
                              interpolation=transforms.InterpolationMode.BICUBIC)
        ])

    # ------------------------------------------------------------------ cameras

    def _load_cameras_dense(
        self, record: Dict[str, Any], indices: np.ndarray, cam_files: List[Path]
    ) -> Tuple[List[str], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Per-frame npz: intrinsic at rgb resolution, pose = c2w OpenCV.
        Loads ONLY the temporally-sampled frames (scenes have ~300 npz files)."""
        names, K_list, R_list, t_list = [], [], [], []
        for i in indices:
            f = cam_files[int(i)]
            with np.load(f) as z:
                K_list.append(z["intrinsic"].astype(np.float64))
                pose = z["pose"].astype(np.float64)
            R_list.append(pose[:3, :3])
            t_list.append(pose[:3, 3])
            names.append(f.stem + ".png")
        return names, K_list, R_list, t_list

    def _load_cameras_flat(
        self, record: Dict[str, Any]
    ) -> Tuple[List[str], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """transforms.json: nerfstudio c2w OpenGL at full res -> OpenCV; K is a
        single full-res header entry, rescaled below to the actual png size."""
        with open(record["transforms_path"]) as f:
            tj = json.load(f)
        # png in images_4 is the 4x downscale of (tj.w, tj.h); probe actual size
        # once per clip instead of hardcoding the factor.
        frames = sorted(tj["frames"], key=lambda fr: fr["file_path"])
        probe_name = Path(frames[0]["file_path"]).name
        probe = iio.imread(Path(record["rgb_dir"]) / probe_name)
        h_img, w_img = probe.shape[:2]
        sx, sy = w_img / float(tj["w"]), h_img / float(tj["h"])
        K = np.array([
            [tj["fl_x"] * sx, 0.0, tj["cx"] * sx],
            [0.0, tj["fl_y"] * sy, tj["cy"] * sy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        names, K_list, R_list, t_list = [], [], [], []
        for fr in frames:
            pose = np.asarray(fr["transform_matrix"], dtype=np.float64) @ _GL_TO_CV
            names.append(Path(fr["file_path"]).name)
            K_list.append(K)
            R_list.append(pose[:3, :3])
            t_list.append(pose[:3, 3])
        return names, K_list, R_list, t_list

    @staticmethod
    def _normalize_first_identity(
        r_list: List[np.ndarray], t_list: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        r0_inv = r_list[0].T
        t0 = t_list[0]
        return [r0_inv @ r for r in r_list], [r0_inv @ (t - t0) for t in t_list]

    def _normalize_ray_moments(self, raymaps: torch.Tensor, clip_idx=None,
                               obj_id: Optional[str] = None) -> Tuple[torch.Tensor, float]:
        """Per-clip max-abs moment normalization (COLMAP scale is arbitrary, so
        this loses nothing metric); fixed P90 divisor when scale_preserving so
        moment magnitudes stay comparable across clips in the mixture."""
        from ._scale_norm import (is_parallax_mode, parallax_ray_encoding,
                                  companded_origin_ray_norm)
        from ._depth_gauge import gauge_scale, is_gauge_mode
        if is_parallax_mode(self.cfg) and parallax_ray_encoding(self.cfg) == "companded_origin":
            # scheme-C consistency with the geometry loaders: spatially-constant mu-law origin
            # channels; gauge = per-clip max origin (COLMAP scale is arbitrary anyway).
            return companded_origin_ray_norm(raymaps, self.cfg, MIN_RAYMAP_SCALE)
        if is_gauge_mode(self.cfg) and obj_id is not None:
            # G3: identical gauge to re10k.py and the geometry loaders --
            # s = kappa * median(metric scene depth) in this scene's COLMAP units.
            # `clip_idx` are FRAME INDICES, which is what the DL3DV extractor stored as
            # `ts`: both sides sort dense cams by globbed filename and flat frames by
            # transforms.json `file_path`, so the two indexings agree. Per-frame medians
            # vary a lot within a DL3DV scene (1.7 m to 5.5 m is typical), so the gauge is
            # taken over the SAMPLED frames rather than the whole scene.
            # A scene absent from the table RAISES; ungauged and poorly-constrained scenes
            # are pruned from the records offline. Falling through to max-norm would put a
            # second magnitude code on the ray channel -- the bug the gauge removes, and at
            # dl3dv_colmap's ~9% mixture share it would dwarf the RE10K case it fixes.
            out = raymaps.clone()
            s = gauge_scale(self.cfg, obj_id, clip_idx)
            out[:, 3:6] = raymaps[:, 3:6] / s
            # Bounded now: this exact line fed 9.24e10 to the model on 2026-09-02 when a
            # diverged COLMAP pose (frame_00184, |t|=3.77e12) landed in a sampled window.
            from ._scale_norm import clamp_moments_
            clamp_moments_(out, self.cfg)
            return out, s
        moments = raymaps[:, 3:6]
        if self.cfg.get("scale_preserving", False):
            divisor = float(self.cfg.get("mutual_scale", None) or GLOBAL_MOMENT_DIVISOR_P90)
            normalized = raymaps.clone()
            normalized[:, 3:6] = moments / divisor
            return normalized, 1.0
        scale = max(float(moments.abs().max().item()), MIN_RAYMAP_SCALE)
        if scale < 1e-8:
            scale = 1.0
        normalized = raymaps.clone()
        normalized[:, 3:6] = torch.clamp(moments / scale, -1.0, 1.0)
        return normalized, scale

    # ------------------------------------------------------------------ loading

    def _load_video(
        self, record: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, torch.Tensor]:
        if record["layout"] == "dense":
            cam_files = sorted(Path(record["cam_dir"]).glob("frame_*.npz"))
            if len(cam_files) < 2:
                raise ValueError(f"{record['obj_id']}: only {len(cam_files)} posed frames")
            temporal_indices = self._temporal_sample(len(cam_files), record["fps"])
            if not getattr(self.cfg, "no_augmentations", False) and random.random() < 0.5:
                temporal_indices = temporal_indices[::-1].copy()
            names, K_sel, R_sel, t_sel = self._load_cameras_dense(
                record, temporal_indices, cam_files)
        else:
            names, K_list, R_list, t_list = self._load_cameras_flat(record)
            if len(names) < 2:
                raise ValueError(f"{record['obj_id']}: only {len(names)} posed frames")
            temporal_indices = self._temporal_sample(len(names), record["fps"])
            if not getattr(self.cfg, "no_augmentations", False) and random.random() < 0.5:
                temporal_indices = temporal_indices[::-1].copy()
            names = [names[int(i)] for i in temporal_indices]
            K_sel = [K_list[int(i)] for i in temporal_indices]
            R_sel = [R_list[int(i)] for i in temporal_indices]
            t_sel = [t_list[int(i)] for i in temporal_indices]

        rgb_dir = Path(record["rgb_dir"])
        frames_np = np.stack([iio.imread(rgb_dir / n) for n in names])
        frames = torch.from_numpy(frames_np).float().permute(0, 3, 1, 2).contiguous() / 255.0

        # Aspect-preserving center-crop + resize; the same crop transforms K so
        # raymaps below stay geometrically valid on the frame grid.
        H_src, W_src = frames.shape[-2:]
        box = center_crop_box_for_aspect(H_src, W_src, self.height, self.width)
        frames = apply_spatial_crop(frames, box)
        frames = F.interpolate(frames, size=(self.height, self.width),
                               mode="bilinear", align_corners=False, antialias=True)
        frames = self.img_normalize(frames)
        K_sel = [adjust_K_for_crop_resize(K, box, self.height, self.width) for K in K_sel]

        R_sel, t_sel = self._normalize_first_identity(R_sel, t_sel)

        raymap_h = getattr(self.cfg, "raymap_height", None) or self.height
        raymap_w = getattr(self.cfg, "raymap_width", None) or self.width
        sx = raymap_w / self.width
        sy = raymap_h / self.height

        raymaps_list, ext_list, K_gt_list = [], [], []
        for K_f, R, t in zip(K_sel, R_sel, t_sel):
            K_r = torch.tensor([
                [K_f[0, 0] * sx, 0.0, K_f[0, 2] * sx],
                [0.0, K_f[1, 1] * sy, K_f[1, 2] * sy],
                [0.0, 0.0, 1.0],
            ], dtype=torch.float32)
            E = torch.eye(4, dtype=torch.float32)
            E[:3, :3] = torch.from_numpy(R).float()
            E[:3, 3] = torch.from_numpy(t).float()
            rays_d, rays_m = camera_to_raymap(K_r, E, raymap_h, raymap_w)
            raymaps_list.append(torch.cat([rays_d, rays_m], dim=0))
            ext_list.append(E)
            K_gt_list.append(K_r)

        raymaps = torch.stack(raymaps_list, dim=0)
        ext_gt = torch.stack(ext_list, dim=0)
        K_gt = torch.stack(K_gt_list, dim=0)

        raymaps, scale = self._normalize_ray_moments(
            raymaps, clip_idx=temporal_indices, obj_id=record["obj_id"])
        return frames, raymaps, scale, ext_gt, K_gt

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Bounded retry (as dl3dv.py): avoid unbounded recursion on bad files.
        for _attempt in range(8):
            record = self.records[idx]
            try:
                frames, raymaps, scale, ext_gt, K_gt = self._load_video(record)
                break
            except Exception as e:
                print(f"Warning: skipping dl3dv_colmap {record.get('obj_id', '?')}: {e}")
                # backoff: transient filesystem stalls fail all instant
                # retries at once; wait it out before resampling
                time.sleep(min(5.0, 0.3 * 2 ** _attempt))
                idx = random.randint(0, len(self.records) - 1)
        else:
            record = self.records[0]
            frames, raymaps, scale, ext_gt, K_gt = self._load_video(record)

        video_metadata = {
            "num_frames": frames.shape[0],
            "height": frames.shape[2],
            "width": frames.shape[3],
            "has_caption": bool(record.get("caption", "")),
        }
        prompts = self.id_token + record.get("caption", "")
        prompt_embeds, prompt_embed_len = None, None
        if self.load_prompt_embed:
            prompt_embeds, prompt_embed_len = self._load_prompt_embed(record)

        record["height"] = video_metadata["height"]
        record["width"] = video_metadata["width"]
        has_bbox, bbox_render = self._render_bbox(record)

        # NVS-only: NO "depths" key — MixtureDataset injects a randn placeholder
        # and has_depth=False, and the algorithm masks the depth stream.
        output: Dict[str, Any] = {
            "videos": frames,
            "raymaps": raymaps,
            "raymap_scale": torch.tensor(scale, dtype=torch.float32),
            "extrinsics_gt": ext_gt,
            "intrinsics_gt": K_gt,
            "video_metadata": video_metadata,
            "bbox_render": bbox_render,
            "has_bbox": has_bbox,
            "prompts": prompts,
        }
        if prompt_embeds is not None:
            output["prompt_embeds"] = prompt_embeds
            output["prompt_embed_len"] = prompt_embed_len
        return output

    def _load_prompt_embed(self, record):
        embed_path = Path(record["prompt_embed_path"])
        if not embed_path.exists():
            return torch.zeros(self.max_text_tokens, 4096), self.max_text_tokens
        prompt_embed = torch.load(embed_path, map_location="cpu", weights_only=True)
        prompt_embed_len = prompt_embed.size(0)
        if prompt_embed_len < self.max_text_tokens:
            padding = torch.zeros(self.max_text_tokens - prompt_embed.size(0),
                                  prompt_embed.size(1), dtype=prompt_embed.dtype)
            prompt_embed = torch.cat([prompt_embed, padding], dim=0)
        return prompt_embed, prompt_embed_len
