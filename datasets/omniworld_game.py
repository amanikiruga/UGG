import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from torchvision.transforms import v2 as transforms

from ._crop_utils import (
    adjust_K_for_crop_resize,
    apply_spatial_crop,
    center_crop_box_for_aspect,
)
from ._geometry_constants import GLOBAL_SCALE
from ._scale_norm import is_scale_norm_mode, apply_scale_norm
from .aria_new import camera_to_raymap
from .video_base import VideoDataset


ORIG_H, ORIG_W = 720, 1280  # pixels; source resolution of OmniWorld-Game frames
INVALID_DEPTH_SENTINEL = 1.0e4  # meters; sentinel stuffed into invalid pixels so they saturate disparity
MIN_SCALE = 1.0  # meters; floor for mutual scale (see _apply_mutual_scale)


class OmniWorldGameDataset(VideoDataset):

    # captions.csv keys on (scene_id, split_idx): a scene has many clips, each
    # its own caption. Match record["scene_id"]+record["split_idx"] so a clip
    # gets ITS caption, not a sibling clip's. (Embeds: train__<scene>__<idx>.pt.)
    CAPTION_SCENE_KEY_COLS = ["scene_id", "split_idx"]

    def _mean_adj_rotation_deg_for_record(self, record: Dict[str, Any]) -> float:
        """Mean adjacent-frame rotation (deg) for a record from
        camera/split_<split_idx>.json quats. The quats array is already
        split-local (one entry per frame in this split — see
        omniworld_game._load_cameras which indexes cam["quats"][idx] by the
        same temporal indices used to slice the video), so we use it directly
        without re-indexing by frame_indices.
        """
        scene_dir = self.data_root / str(record["scene_id"])
        cam_path = scene_dir / "camera" / f"split_{record['split_idx']}.json"
        try:
            with open(cam_path) as f:
                cam = json.load(f)
        except (OSError, json.JSONDecodeError):
            return float("nan")
        try:
            q = np.asarray(cam["quats"], dtype=np.float64)
        except (KeyError, ValueError):
            return float("nan")
        if q.ndim != 2 or q.shape[1] != 4 or q.shape[0] < 2:
            return float("nan")
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        dots = np.clip(np.abs(np.sum(q[1:] * q[:-1], axis=1)), 0.0, 1.0)
        return float(np.degrees(2.0 * np.arccos(dots)).mean())

    def _load_records(self) -> List[Dict[str, Any]]:
        records = super()._load_records()
        max_mean_rot = getattr(self.cfg, "max_mean_adj_rotation_deg", None)
        if max_mean_rot is None:
            return records
        kept: List[Dict[str, Any]] = []
        skipped_high_rotation = 0
        for r in records:
            mean_rot = self._mean_adj_rotation_deg_for_record(r)
            if not np.isfinite(mean_rot) or mean_rot > float(max_mean_rot):
                skipped_high_rotation += 1
                continue
            kept.append(r)
        print(f"OmniWorldGame: high-rotation filter @ >{max_mean_rot}°: "
              f"kept {len(kept)} / {len(records)} (skipped {skipped_high_rotation})")
        return kept

    def preprocess_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich each record with frame indices and caption from text/ dir."""
        scene_id = str(record["scene_id"])
        split_idx = int(record["split_idx"])
        scene_dir = self.data_root / scene_id

        record["video_path"] = f"{scene_id}/color.mp4"

        with open(scene_dir / "split_info.json") as f:
            split_info = json.load(f)
        record["frame_indices"] = split_info["split"][split_idx]

        # if not record.get("caption"):
        #     record["caption"] = self._find_caption(scene_dir, record["frame_indices"])

        return record

    def _find_caption(self, scene_dir: Path, frame_indices: List[int]) -> str:
        """Find best-matching text caption JSON for the split's frame range."""
        text_dir = scene_dir / "text"
        if not text_dir.exists():
            return ""

        split_start = min(frame_indices)
        split_end = max(frame_indices)

        best_caption = ""
        best_overlap = 0

        for tf in text_dir.iterdir():
            if tf.suffix != ".json":
                continue
            parts = tf.stem.split("_")
            if len(parts) != 2:
                continue
            try:
                tf_start, tf_end = int(parts[0]), int(parts[1])
            except ValueError:
                continue

            overlap = max(0, min(tf_end, split_end) - max(tf_start, split_start))
            if overlap > best_overlap:
                best_overlap = overlap
                try:
                    with open(tf) as f:
                        data = json.load(f)
                    best_caption = data.get("captions", {}).get("Video_Caption", "")
                except (json.JSONDecodeError, KeyError):
                    pass

        return best_caption

    def _build_video_transforms(self, augment: bool = True):
        """Resize only -- no random crop to preserve camera alignment."""
        return transforms.Compose([
            transforms.Resize(
                size=(self.height, self.width),
                interpolation=transforms.InterpolationMode.BICUBIC,
            )
        ])

    @staticmethod
    def _quat_wxyz_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
        """Quaternion [w, x, y, z] -> 3x3 rotation matrix."""
        q_wxyz = q_wxyz / (np.linalg.norm(q_wxyz) + 1e-12)
        q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
        return Rotation.from_quat(q_xyzw).as_matrix()

    @staticmethod
    def _decode_depth(raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode OmniWorld uint16 depth to metric depth.
        Formula from OmniWorld scripts/utils.py.
        """
        depth = raw.astype(np.float32) / 65535.0
        near_mask = depth < 0.0015
        far_mask = depth > (65500.0 / 65535.0)
        near, far = 1.0, 1000.0
        depth = depth / (far - depth * (far - near)) / 0.004
        valid = ~(near_mask | far_mask)
        depth[~valid] = 0.0
        return depth, valid

    @staticmethod
    def _normalize_first_identity(
        r_list: List[np.ndarray], t_list: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        r0_inv = r_list[0].T
        t0 = t_list[0]
        r_norm = [r0_inv @ r for r in r_list]
        t_norm = [r0_inv @ (t - t0) for t in t_list]
        return r_norm, t_norm

    def _apply_mutual_scale(
        self,
        raymaps_raw: torch.Tensor,
        depth_metric: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """scale = max(|raw moments|, MIN_SCALE).

        Using the moment max alone collapses near-stationary clips: |M| = |O x D|
        is ~0 when the camera barely moves, so depth/scale blows up and every
        valid depth saturates at disparity = -1. A fixed MIN_SCALE floor keeps
        both streams well-conditioned: near-stationary clips get a stable scene
        anchor, and high-motion clips are unaffected because |M| dominates.

        Note: unlike PointOdyssey we deliberately avoid a depth-median floor.
        OmniWorld-Game depths can be huge (sky-box, long corridors) and a
        large depth median would divide moments down until they are ~0 — no
        raymap signal left for the model to learn from.

        Returns (raymaps_norm, depth_as_signed_disparity, scale).
        """
        moments = raymaps_raw[:, 3:6]

        if is_scale_norm_mode(self.cfg):
            # Scheme C (SCALE_NORMALIZATION_ANALYSIS.md): per-clip s = kappa*median(valid depth).
            # OmniWorld units are game-engine-arbitrary, but parallax (moments/depth) is
            # dimensionless so the scheme still applies; median is robust to skybox outliers.
            return apply_scale_norm(raymaps_raw, depth_metric, self.cfg, INVALID_DEPTH_SENTINEL)
        elif self.cfg.get("scale_preserving", False):
            # GLOBAL fixed scale (not per-clip) -> preserves metric scale across the dataset.
            # Same constant divides BOTH camera moments AND depth so parallax stays metric
            # and the mapping is invertible (x mutual_scale -> metres). Matches tartanair.
            scale = float(self.cfg.get("mutual_scale", None) or GLOBAL_SCALE)
        elif self.cfg.get("depth_only", False):
            # depth_only discards the camera downstream (has_raymap=False, ray tokens
            # noise-filled and loss-masked), so there is no moment channel left to keep
            # well-conditioned and no depth<->pose alignment to preserve -- the two
            # reasons the MIN_SCALE floor exists. Gauging depth on camera motion is then
            # meaningless: OmniWorld's |M| is ~0.02 of range, so MIN_SCALE always binds
            # and the depth channel collapses into the top of [-1,1] (measured: 42% of
            # clips crushed, range used 0.415 median, 0.014 worst). Gauge on the scene.
            valid = (depth_metric > 0) & (depth_metric < INVALID_DEPTH_SENTINEL)
            zbar = float(depth_metric[valid].median().item()) if bool(valid.any()) else 0.0
            scale = zbar if zbar > 1e-8 else 1.0
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

    def _read_video_frames(self, video_path: Path, frame_ids: np.ndarray) -> np.ndarray:
        """Read specific frames from MP4 via sequential forward seeking."""
        sort_idx = np.argsort(frame_ids)
        sorted_ids = frame_ids[sort_idx]

        frames = [None] * len(frame_ids)
        reader = imageio.get_reader(str(video_path))
        try:
            pos = int(sorted_ids[0])
            reader.set_image_index(pos)
            current_frame = reader.get_next_data()
            frames[sort_idx[0]] = current_frame

            for i, target in enumerate(sorted_ids[1:], start=1):
                target = int(target)
                while pos < target:
                    current_frame = reader.get_next_data()
                    pos += 1
                frames[sort_idx[i]] = current_frame
        finally:
            reader.close()

        return np.stack(frames)

    def _load_cameras(
        self, record: Dict[str, Any], temporal_indices: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Load per-frame OpenCV cameras for the sampled indices.

        Raw camera JSON stores w2c as (quats wxyz, trans). We invert to c2w.
        scipy's Rotation.from_quat normalizes non-unit quats, so we do not
        reject frames on quat magnitude. We do honor the dataset-provided
        `invalid_frame` list when present.

        Returns per-frame:
            K:      (3, 3) pixel-unit intrinsics at original 720x1280 resolution
            R_c2w:  (3, 3) c2w rotation (OpenCV)
            t_c2w:  (3,)   c2w translation (OpenCV)
        """
        scene_dir = self.data_root / str(record["scene_id"])
        cam_path = scene_dir / "camera" / f"split_{record['split_idx']}.json"
        with open(cam_path) as f:
            cam = json.load(f)

        cx_orig = cam["cx"]
        cy_orig = cam["cy"]
        invalid_set = set(cam.get("invalid_frame") or [])

        K_list: List[np.ndarray] = []
        r_list: List[np.ndarray] = []
        t_list: List[np.ndarray] = []

        for idx in temporal_indices:
            idx = int(idx)
            q = np.asarray(cam["quats"][idx], dtype=np.float64)
            t = np.asarray(cam["trans"][idx], dtype=np.float64)
            focal = float(cam["focals"][idx])

            if idx in invalid_set:
                raise ValueError(
                    f"frame {idx} in split {record['split_idx']} of "
                    f"{record['scene_id']} is marked invalid"
                )

            R_w2c = self._quat_wxyz_to_rotmat(q)  # scipy normalizes internally
            R_c2w = R_w2c.T
            t_c2w = -R_w2c.T @ t

            K_list.append(np.array(
                [[focal, 0.0,   cx_orig],
                 [0.0,   focal, cy_orig],
                 [0.0,   0.0,   1.0]],
                dtype=np.float64,
            ))
            r_list.append(R_c2w)
            t_list.append(t_c2w)

        return K_list, r_list, t_list

    @staticmethod
    def _generate_raymaps_raw(
        K_frame_list: List[np.ndarray],
        r_c2w_list: List[np.ndarray],
        t_c2w_list: List[np.ndarray],
        raymap_hw: Tuple[int, int],
        frame_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """K is pixel-unit at `frame_hw`; uniform rescale to `raymap_hw`.

        Upstream we center-crop to the target aspect, so sx == sy here and K
        remains geometrically valid at raymap resolution.
        """
        H, W = raymap_hw
        H_f, W_f = frame_hw
        sx = W / W_f
        sy = H / H_f

        raymaps, exts, ks = [], [], []
        for K_frame, R, t in zip(K_frame_list, r_c2w_list, t_c2w_list):
            K = torch.tensor(
                [
                    [K_frame[0, 0] * sx, 0.0,                K_frame[0, 2] * sx],
                    [0.0,                K_frame[1, 1] * sy, K_frame[1, 2] * sy],
                    [0.0,                0.0,                1.0],
                ],
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

    def _load_video(
        self, record: Dict[str, Any]
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, float,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        """Returns:
          frames       (T, 3, H, W)       in [-1, 1]
          raymaps      (T, 6, rH, rW)     dirs unit, moments in [-1, 1]
          depths       (T, 1, H, W)       modified-disparity in [-1, 1]
          raymap_scale float               shared scalar used for moments + depth
          ext_gt       (T, 4, 4)          OpenCV c2w, first-frame identity
          K_gt         (T, 3, 3)          pixel-unit K at raymap resolution
          depths_raw   (T, 1, H, W)       float32 meters (invalid=0)
          depths_valid (T, 1, H, W)       bool
        """
        scene_dir = self.data_root / str(record["scene_id"])
        frame_indices = np.asarray(record["frame_indices"])
        n_split_frames = len(frame_indices)
        fps = record["fps"]
        # import ipdb; ipdb.set_trace()

        temporal_indices = self._temporal_sample(n_split_frames, fps)
        # if not getattr(self.cfg, "no_augmentations", False) and random.random() < 0.5:
        #     temporal_indices = temporal_indices[::-1].copy()

        global_ids = np.asarray([frame_indices[int(i)] for i in temporal_indices])

        # --- RGB ---
        frames_np = self._read_video_frames(scene_dir / "color.mp4", global_ids)
        frames = torch.from_numpy(frames_np).float().permute(0, 3, 1, 2).contiguous() / 255.0

        # --- Depth (metric meters) ---
        with h5py.File(scene_dir / "depth.h5", "r") as f:
            raw_depth = np.stack([f["depth"][int(fid)] for fid in global_ids])
        decoded_depth, valid_mask = self._decode_depth(raw_depth)
        d_m = torch.from_numpy(decoded_depth).unsqueeze(1).float()
        v = torch.from_numpy(valid_mask).unsqueeze(1).bool()

        # --- Cameras (at source res) ---
        K_list, r_list, t_list = self._load_cameras(record, temporal_indices)

        # --- Aspect-preserving center-crop + resize to (self.height, self.width) ---
        # Same pattern as vkitti2/point_odyssey/scannetpp: one crop applies to
        # RGB, depth, mask, and K so raymaps below are geometrically valid.
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
        frames = self.img_normalize(frames)
        K_list = [adjust_K_for_crop_resize(K, box, self.height, self.width) for K in K_list]

        depths_raw = d_m.clone()
        depths_valid = v
        d_for_scale = torch.where(
            v, d_m, torch.full_like(d_m, INVALID_DEPTH_SENTINEL)
        )

        # --- Raymaps ---
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
                scene_id = record.get("scene_id", str(idx))
                split_idx = record.get("split_idx", "?")
                print(f"Warning: skipping {scene_id}/split_{split_idx}: {e}")
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
        if self.cfg.get("depth_only", False):
            # Treat OmniWorld as an RGB+depth-ONLY dataset: omit the camera keys so
            # MixtureDataset injects has_raymap=False -> the algorithm noise-fills and
            # loss-masks all ray tokens. With diffusion-forcing setting [1] the training
            # objective is then exactly RGB history -> depth future (no pose, no NVS).
            # Cameras are still used INTERNALLY above only to normalize depth's scale.
            for k in ("raymaps", "raymap_scale", "extrinsics_gt", "intrinsics_gt"):
                output.pop(k)
        if prompt_embeds is not None:
            output["prompt_embeds"] = prompt_embeds
            output["prompt_embed_len"] = prompt_embed_len
        return output



if __name__ == "__main__":
    import imageio as iio_write
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "data_root": "./data/OmniWorld_game",
        "debug": True,
        "metadata_path": "metadata.csv",
        "auto_download": False,
        "force_download": False,
        "test_percentage": 0.1,
        "id_token": "",
        "height": 480,
        "width": 832,
        "n_frames": 13,
        "fps": 5,
        "trim_mode": "random_cut",
        "pad_mode": "slowdown",
        "filtering": {"disable": True, "height": None, "width": None,
                      "n_frames": None, "fps": None},
        "load_video_latent": False,
        "load_prompt_embed": False,
        "augmentation": {"random_flip": None, "ratio": None, "scale": None},
        "image_to_video": False,
        "max_text_tokens": 1,
        "check_video_path": False,
        "no_augmentations": True,
        "raymap_height": 60,
        "raymap_width": 104,
    })

    print("=" * 80)
    print("Testing OmniWorld-Game Dataset")
    print("=" * 80)

    dataset = OmniWorldGameDataset(cfg, split="training")
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
            rays_up[:, :3],
            rays_up[:, 3:],
            item["depths"].expand(-1, 3, -1, -1),
        ],
        dim=-1,
    )
    vis = ((panel.permute(0, 2, 3, 1) / 2 + 0.5).clamp(0, 1).numpy() * 255).astype(np.uint8)
    iio_write.mimwrite("test_omniworld.mp4", vis, fps=5, quality=8)
    print("wrote test_omniworld.mp4")
