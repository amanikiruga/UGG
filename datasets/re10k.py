
import os
import random
import time
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import cv2
import imageio.v3 as iio
import decord  # seek-based frame reading (avoids full-video decode OOM)
import torch.nn.functional as F
from typing import Any, Dict, Tuple, Optional, List
from torchvision.transforms import v2 as transforms
from tqdm import tqdm
from ._crop_utils import (
    adjust_K_for_crop_resize,
    apply_spatial_crop,
    center_crop_box_for_aspect,
)
from .video_base import VideoDataset
from utils.re10k_correctness import (
    exclude_invalid_video_records,
    select_re10k_clip,
)
# from algorithms.wan.utils.raymap import camera_to_raymap


MIN_RAYMAP_SCALE = 1.0  # floor for per-clip max-abs moment normalization
# Global per-channel divisor for Plücker moments, anchored on DL3DV-dominated
# p95 of raw |moments| (scripts/collect_raymap_moment_stats.py). Used only
# when cfg.scale_preserving=True; preserves inter-clip translation extent.
GLOBAL_MOMENT_DIVISOR = (1.3, 5.0, 1.9)


def camera_to_raymap(
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    height: int,
    width: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert camera intrinsics and extrinsics to Plücker raymap.
    
    Args:
        intrinsics: Camera intrinsic matrix (B, 3, 3) or (3, 3)
            [[fx, 0, cx],
             [0, fy, cy],
             [0,  0,  1]]
        extrinsics: Camera extrinsic matrix (B, 4, 4) or (4, 4)
            Camera-to-world transformation matrix (c2w)
        height: Image height
        width: Image width
    
    Returns:
        rays_d: Ray directions (B, 3, H, W), L2-normalized unit vectors
        rays_m: Ray moments (B, 3, H, W), Plücker moments (O × D)
    
    Note:
        Plücker coordinates: [D, M] where D is direction, M = O × D is moment
        - D encodes ray direction (unit vector)
        - M encodes the closest point to origin on the ray
    """
    # Handle unbatched input
    if intrinsics.dim() == 2:
        intrinsics = intrinsics.unsqueeze(0)
    if extrinsics.dim() == 2:
        extrinsics = extrinsics.unsqueeze(0)
    
    B = intrinsics.shape[0]
    device = intrinsics.device
    dtype = intrinsics.dtype
    
    # Extract intrinsic parameters
    fx = intrinsics[:, 0, 0]  # (B,)
    fy = intrinsics[:, 1, 1]  # (B,)
    cx = intrinsics[:, 0, 2]  # (B,)
    cy = intrinsics[:, 1, 2]  # (B,)
    
    # Create pixel grid
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing='ij'
    )
    # Use pixel centers
    x = x + 0.5  # (H, W)
    y = y + 0.5  # (H, W)
    
    # Unproject to camera coordinates (z=1 plane)
    # rays_d_cam = [(x - cx) / fx, (y - cy) / fy, 1]
    x = x.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
    y = y.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
    
    rays_d_cam_x = (x - cx.view(B, 1, 1)) / fx.view(B, 1, 1)  # (B, H, W)
    rays_d_cam_y = (y - cy.view(B, 1, 1)) / fy.view(B, 1, 1)  # (B, H, W)
    rays_d_cam_z = torch.ones_like(rays_d_cam_x)  # (B, H, W)
    
    rays_d_cam = torch.stack([rays_d_cam_x, rays_d_cam_y, rays_d_cam_z], dim=1)  # (B, 3, H, W)
    
    # Transform ray directions to world coordinates
    R_c2w = extrinsics[:, :3, :3]  # (B, 3, 3)
    t_c2w = extrinsics[:, :3, 3]   # (B, 3)
    
    # rays_d_world = R_c2w @ rays_d_cam
    rays_d_cam_flat = rays_d_cam.view(B, 3, -1)  # (B, 3, H*W)
    rays_d_world_flat = torch.bmm(R_c2w, rays_d_cam_flat)  # (B, 3, H*W)
    rays_d_world = rays_d_world_flat.view(B, 3, height, width)  # (B, 3, H, W)
    
    # L2 normalize ray directions
    rays_d = F.normalize(rays_d_world, dim=1)  # (B, 3, H, W)
    
    # Camera origin in world coordinates
    rays_o = t_c2w.view(B, 3, 1, 1).expand(-1, -1, height, width)  # (B, 3, H, W)
    
    # Compute Plücker moment: M = O × D
    rays_m = torch.cross(rays_o, rays_d, dim=1)  # (B, 3, H, W)
    
    # Squeeze batch dim if input was unbatched
    if B == 1 and intrinsics.shape[0] == 1:
        rays_d = rays_d.squeeze(0)
        rays_m = rays_m.squeeze(0)
    
    return rays_d, rays_m


def _intersect_skew_lines(points: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """
    Estimate a single point that best intersects multiple skew lines in least squares.

    Args:
        points: (N, 3) points on each line
        directions: (N, 3) line directions
    Returns:
        center: (3,) least-squares intersection point
    """
    directions = F.normalize(directions, dim=-1)
    eye = torch.eye(3, dtype=points.dtype, device=points.device).unsqueeze(0)
    proj = eye - directions.unsqueeze(-1) * directions.unsqueeze(-2)  # (N, 3, 3)

    lhs = proj.sum(dim=0)  # (3, 3)
    rhs = torch.bmm(proj, points.unsqueeze(-1)).sum(dim=0).squeeze(-1)  # (3,)
    return torch.linalg.lstsq(lhs, rhs).solution


def raymap_to_camera(
    raymaps: torch.Tensor,
    raymap_scale: float = 1.0,
    max_points: int = 5000,
    ransac_reproj_threshold: float = 1.0,
    min_abs_z: float = 1e-6,
    ray_encoding: str = "auto",
    mu: float = 255.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert normalized raymaps back to camera intrinsics/extrinsics.

    Channels 3:6 are either the Plucker moment (`rays_m_raw = raymaps[:, 3:] * raymap_scale`)
    or — under scale_mode=parallax with parallax_ray_encoding=companded_origin — the mu-law
    companded camera origin (spatially constant per frame; `t = raymap_scale *
    mulaw_expand(mean over pixels, mu)`). K and R always come from the direction channels.

    Args:
        raymaps: (6, H, W) or (B, 6, H, W), channels are [rays_d, rays_m_norm]
        raymap_scale: scalar produced at generation time; use 1.0 if you pass raw moments.
        max_points: max number of rays used for homography estimation
        ransac_reproj_threshold: RANSAC reprojection threshold in pixels
        min_abs_z: minimal |D_z| for stable projective conversion
        ray_encoding: 'plucker' | 'companded_origin' | 'auto' (detect: the companded-origin
            channel is (near-)spatially-constant per frame, the moment field is not)
        mu: mu-law gain used at encode time (parallax_mu; only for companded_origin)
    Returns:
        intrinsics: (3, 3) or (B, 3, 3)
        extrinsics: (4, 4) or (B, 4, 4), camera-to-world matrices (OpenCV)
    """
    if raymaps.dim() == 3:
        raymaps = raymaps.unsqueeze(0)
        squeeze_output = True
    elif raymaps.dim() == 4:
        squeeze_output = False
    else:
        raise ValueError(f"Expected raymaps with 3 or 4 dims, got shape {raymaps.shape}.")

    if raymaps.shape[1] != 6:
        raise ValueError(f"Expected channel dim to be 6, got {raymaps.shape[1]}.")

    raymaps = raymaps.clone()
    if ray_encoding == "auto":
        # companded-origin channels are constant over pixels (exactly for GT, nearly for
        # predictions); Plucker moment fields vary at O(|m|) across the image.
        ch = raymaps[:, 3:]
        spatial_std = ch.std(dim=(-2, -1)).mean()
        overall = ch.abs().mean().clamp_min(1e-6)
        ray_encoding = "companded_origin" if (spatial_std / overall) < 0.25 else "plucker"
    if ray_encoding == "plucker":
        raymaps[:, 3:] = raymaps[:, 3:] * raymap_scale  # companded_origin: expand AFTER mean

    B, _, H, W = raymaps.shape
    device = raymaps.device
    dtype = raymaps.dtype

    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=dtype, device=device),
        torch.arange(W, dtype=dtype, device=device),
        indexing="ij",
    )
    pixel_x = (xx + 0.5).reshape(-1)
    pixel_y = (yy + 0.5).reshape(-1)

    intrinsics_list = []
    extrinsics_list = []

    for b in range(B):
        rays_d = F.normalize(raymaps[b, :3], dim=0)
        rays_m = raymaps[b, 3:]

        rays_d_flat = rays_d.permute(1, 2, 0).reshape(-1, 3)  # (H*W, 3)
        rays_m_flat = rays_m.permute(1, 2, 0).reshape(-1, 3)  # (H*W, 3)

        if ray_encoding == "companded_origin":
            # spatial mean -> mu-law expand -> shared scale. Exact inverse of the encoder.
            from ._scale_norm import mulaw_expand
            cam_center = mulaw_expand(rays_m.mean(dim=(-2, -1)), mu) * raymap_scale  # (3,)
        else:
            # Convert Plucker rays [D, M] to points-on-lines using P = D x M.
            ray_points = torch.cross(rays_d_flat, rays_m_flat, dim=-1)
            cam_center = _intersect_skew_lines(ray_points, rays_d_flat)  # (3,)

        valid = (
            torch.isfinite(rays_d_flat).all(dim=-1)
            & torch.isfinite(rays_m_flat).all(dim=-1)
            & (rays_d_flat[:, 2].abs() > min_abs_z)
        )
        valid_idx = torch.where(valid)[0]
        if valid_idx.numel() < 8:
            raise ValueError(
                f"Not enough valid rays to estimate homography: {valid_idx.numel()}."
            )

        if valid_idx.numel() > max_points:
            sample_idx = torch.linspace(
                0,
                valid_idx.numel() - 1,
                steps=max_points,
                dtype=torch.float32,
                device=device,
            ).long()
            valid_idx = valid_idx[sample_idx]

        rays_d_sel = rays_d_flat[valid_idx]
        src = torch.stack(
            [
                rays_d_sel[:, 0] / rays_d_sel[:, 2],
                rays_d_sel[:, 1] / rays_d_sel[:, 2],
            ],
            dim=-1,
        )  # normalized world ray-plane coords
        dst = torch.stack([pixel_x[valid_idx], pixel_y[valid_idx]], dim=-1)  # pixel coords

        src_np = src.detach().cpu().numpy().astype(np.float32)
        dst_np = dst.detach().cpu().numpy().astype(np.float32)

        homography, _ = cv2.findHomography(
            src_np,
            dst_np,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_reproj_threshold,
        )
        if homography is None:
            homography, _ = cv2.findHomography(src_np, dst_np, method=0)
        if homography is None:
            raise ValueError("Homography estimation failed from ray directions.")

        # Treat homography as camera matrix up to scale: H ~ K * R_w2c.
        projection = np.concatenate(
            [homography.astype(np.float64), np.zeros((3, 1), dtype=np.float64)],
            axis=1,
        )
        K_np, R_w2c_np, *_ = cv2.decomposeProjectionMatrix(projection)
        K_np = K_np / (K_np[2, 2] + 1e-12)

        # Fix common sign ambiguities to keep focal lengths positive.
        if K_np[0, 0] < 0:
            K_np[:, 0] *= -1
            R_w2c_np[0, :] *= -1
        if K_np[1, 1] < 0:
            K_np[:, 1] *= -1
            R_w2c_np[1, :] *= -1

        # Re-orthogonalize rotation.
        U, _, Vt = np.linalg.svd(R_w2c_np)
        R_w2c_np = U @ Vt
        if np.linalg.det(R_w2c_np) < 0:
            U[:, -1] *= -1
            R_w2c_np = U @ Vt

        intrinsics = torch.from_numpy(K_np).to(device=device, dtype=dtype)
        R_c2w = torch.from_numpy(R_w2c_np.T).to(device=device, dtype=dtype)

        extrinsics = torch.eye(4, dtype=dtype, device=device)
        extrinsics[:3, :3] = R_c2w
        extrinsics[:3, 3] = cam_center

        intrinsics_list.append(intrinsics)
        extrinsics_list.append(extrinsics)

    intrinsics_out = torch.stack(intrinsics_list, dim=0)
    extrinsics_out = torch.stack(extrinsics_list, dim=0)

    if squeeze_output:
        intrinsics_out = intrinsics_out.squeeze(0)
        extrinsics_out = extrinsics_out.squeeze(0)

    return intrinsics_out, extrinsics_out


class RE10KDataset(VideoDataset):

    def __init__(self, cfg, split: str = "training") -> None:
        super().__init__(cfg, split)

        # Production normally loads VideoDataset's cached JSON manifest, which
        # bypasses RE10KDataset._load_records(). Filter after that boundary so
        # cached and freshly scanned records have identical semantics.
        invalid_video_ids = self.cfg.get("invalid_video_ids", []) or []
        self.records, removed = exclude_invalid_video_records(
            self.records, invalid_video_ids
        )
        if invalid_video_ids:
            print(
                f"RE10KDataset: excluded {removed} confirmed-invalid videos "
                "after records cache/load"
            )

    @staticmethod
    def _mean_adj_rotation_deg_from_anno_txt(anno_path: Path) -> float:
        """Mean adjacent-frame rotation (deg) from a RealEstate10K .txt
        annotation. Each non-header line packs a 3x4 row-major w2c at indices
        7..18 (after timestamp at 0 and intrinsics at 1..6).
        """
        rs = []
        try:
            with open(anno_path) as f:
                f.readline()  # skip URL header
                for line in f:
                    parts = line.split()
                    if len(parts) < 19:
                        continue
                    try:
                        rs.append([
                            float(parts[7]),  float(parts[8]),  float(parts[9]),
                            float(parts[11]), float(parts[12]), float(parts[13]),
                            float(parts[15]), float(parts[16]), float(parts[17]),
                        ])
                    except ValueError:
                        continue
        except OSError:
            return float("nan")
        if len(rs) < 2:
            return float("nan")
        R = np.asarray(rs, dtype=np.float64).reshape(-1, 3, 3)
        R_rel = np.matmul(R[1:], np.transpose(R[:-1], (0, 2, 1)))
        tr = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
        cos_a = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_a)).mean())

    def _load_records(self) -> List[Dict[str, Any]]:
        """
        Load RE10K records directly from directory structure.
        Each record contains video_path, anno_path, and other metadata.
        """
        # Determine which split to load
        if self.split == "training":
            split_folder = "train"
        elif self.split in ["validation", "test"]:
            split_folder = "test"
        elif self.split == "all":
            split_folder = None  # Load both
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be one of ['training', 'validation', 'test', 'all']")

        max_mean_rot = getattr(self.cfg, "max_mean_adj_rotation_deg", None)
        skipped_high_rotation = 0
        invalid_video_ids = set(self.cfg.get("invalid_video_ids", []) or [])
        skipped_invalid_manifest = 0

        records = []
        total_anno_files = 0
        anno_base_path = self.data_root / "RealEstate10K"
        video_dir = self.data_root / "videos"

        # Collect annotation files from the appropriate split(s)
        if split_folder is None:
            # Load from both train and test
            split_folders = ["train", "test"]
        else:
            split_folders = [split_folder]

        for split in split_folders:
            split_path = anno_base_path / split
            if not split_path.exists():
                print(f"Warning: Split folder {split_path} does not exist, skipping.")
                continue

            anno_files = sorted(split_path.glob("*.txt"))
            total_anno_files += len(anno_files)

            for anno_file in tqdm(anno_files, desc=f"Loading {split} annotations"):
            # for anno_file in anno_files[:10]:
                # The annotation file name (without .txt) should match the video ID
                video_id = anno_file.stem
                
                # # Get video metadata using cv2
                # cap = cv2.VideoCapture(str(video_path))
                # if not cap.isOpened():
                #     if self.debug:
                #         print(f"Warning: Cannot open video {video_path}, skipping.")
                #     continue
                
                # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                # fps = cap.get(cv2.CAP_PROP_FPS)
                # n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                # cap.release()
                
                # Count annotations to estimate usable frames
                with open(anno_file, 'r') as f:
                    videoPathURL = f.readline().rstrip()  # skip first line (video URL)
                    # n_annotations = sum(1 for _ in f)
                    youtubeIDOffset = videoPathURL.find("/watch?v=") + len('/watch?v=')
                    youtubeID = videoPathURL[youtubeIDOffset:]
                
                video_path = video_dir / f"{youtubeID}.mp4"

                if youtubeID in invalid_video_ids:
                    skipped_invalid_manifest += 1
                    continue

                # Check if video exists
                if not video_path.exists():
                    if self.debug:
                        print(f"Warning: Video {video_path} not found for annotation {anno_file}, skipping.")
                    continue

                if max_mean_rot is not None:
                    mean_rot = self._mean_adj_rotation_deg_from_anno_txt(anno_file)
                    if not np.isfinite(mean_rot) or mean_rot > float(max_mean_rot):
                        skipped_high_rotation += 1
                        continue

                # caption is injected below from data_root/captions.csv keyed by scene_id
                record = {
                    "video_path": f"videos/{youtubeID}.mp4",
                    "anno_path": f"RealEstate10K/{split}/{video_id}.txt",
                }

                records.append(record)

        print(
            f"Loaded {len(records)} / {total_anno_files} records from RE10K dataset "
            f"(skipped {skipped_high_rotation} high-rotation @ >{max_mean_rot}°, "
            f"{skipped_invalid_manifest} confirmed-invalid videos)"
        )
        
        # Apply filtering
        len_pre_filter = len(records)
        if not self.filtering.disable:
            records = [record for record in records if self._filter_record(record)]
        len_post_filter = len(records)
        
        print(
            f"RE10K: filtered {len_pre_filter - len_post_filter} records from {len_pre_filter} to {len_post_filter}, "
            f"retaining rate: {len_post_filter / len_pre_filter:.2%}"
        )

        # Inject captions from captions.csv and drop scenes without a valid caption
        # (or, when load_prompt_embed is set, without a cached embed file).
        captions_csv = self.data_root / "captions.csv"
        if not captions_csv.exists():
            raise FileNotFoundError(
                f"RE10K captions.csv not found at {captions_csv}. "
                "Run scripts/cache_re10k_prompt_embeds.py to set up captions/embeddings."
            )
        cap_df = pd.read_csv(captions_csv)
        cap_df = cap_df.dropna(subset=["caption"])
        cap_df = cap_df[~cap_df["caption"].str.startswith("__ERROR__")]
        cap_lookup = dict(zip(cap_df["scene_id"], cap_df["caption"]))

        records_with_caption = []
        n_dropped_no_caption = 0
        n_dropped_no_embed = 0
        for r in records:
            scene_id = Path(r["anno_path"]).stem
            cap = cap_lookup.get(scene_id)
            if cap is None:
                n_dropped_no_caption += 1
                continue
            if self.load_prompt_embed:
                split_dir = Path(r["anno_path"]).parent.name
                embed_path = self.data_root / "prompt_embeds" / split_dir / f"{scene_id}.pt"
                if not embed_path.exists():
                    n_dropped_no_embed += 1
                    continue
            r["caption"] = cap
            records_with_caption.append(r)
        print(
            f"RE10K captions: kept {len(records_with_caption)} / {len(records)} "
            f"(dropped {n_dropped_no_caption} no-caption, {n_dropped_no_embed} no-embed)"
        )
        records = records_with_caption

        # Handle split selection for validation/test split within test folder
        if self.split == "validation" or self.split == "test":
            # Use test_percentage to split the test folder
            if self.split == "validation":
                records = records[: int(len(records) * self.test_percentage)]
            else:  # test
                records = records[int(len(records) * self.test_percentage) :]
        
        # Shuffle with fixed seed for reproducibility
        random.Random(0).shuffle(records)
        
        records = [self.preprocess_record(record) for record in records]
        
        return records

    def _build_video_transforms(self, augment: bool = True):
        # resize instead of random resized crop because of alignemnt with camera pose 
        # in case it crops and the camera center is not at the center of the image and 
        # focal length is not scaled.
        trans = []
        
        trans.append(
            transforms.Resize(
                size=(self.height, self.width),
                interpolation=transforms.InterpolationMode.BICUBIC,
            )
        )
        return transforms.Compose(trans)
    
    def _load_video(
        self, record: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, torch.Tensor, torch.Tensor]:
        """
        Given a record, return:
            frames: (n_frames, 3, H, W)
            raymaps: (n_frames, 6, H, W) with rays_m in [-1, 1]
            raymaps_inv: (n_frames, 6, H, W) with rays_m in [-1, 1]
            raymap_scale: float, rays_m_raw = raymaps[:, 3:6] * raymap_scale
            raymap_inv_scale: float, analogous for raymaps_inv
            extrinsics_gt: (n_frames, 4, 4) OpenCV c2w, first camera = identity
            intrinsics_gt: (n_frames, 3, 3) pixel-unit K at raymap resolution
        """

        video_path = self.data_root / record['video_path']
        # frames = iio.imread(video_path)
        # frames = torch.from_numpy(frames) # (T, H, W, 3)
        # n_frames = frames.shape[0]

        # get annotations
        anno_path = self.data_root / record['anno_path']
        annos = []
        with open(anno_path, 'r') as f:
            f.readline().strip()  # skip first line
            for l in f.readlines():
                line = l.split(' ')
                timeStamp = int(line[0])
                intrinsics = [float(i) for i in line[1:7]]
                pose = [float(i) for i in line[7:19]]
                annos.append({
                    'timeStamp': timeStamp, 
                    'intrinsics': intrinsics,
                    'pose': pose})

        # Seek-read ONLY the annotated frames instead of decoding the whole video.
        # re10k mp4s are full-length YouTube videos, so iio.imread(index=None) was
        # decoding the entire clip (often many GB) into RAM per worker -> OOM. decord
        # seeks to the requested frames. Returned frames are bit-equivalent (ffmpeg).
        # num_threads=1 avoids decord's threaded-decoder EAGAIN bug
        # (avcodec_send_packet -11); we only decode ~hundreds of frames so single
        # thread is plenty fast and far more robust.
        vr = decord.VideoReader(str(video_path), num_threads=1)
        fps = float(vr.get_avg_fps())
        total_frames = len(vr)

        # Convert annotation timestamps to frame indices
        # timestamp is in microseconds, convert to seconds and multiply by fps
        frame_indices = []
        for anno in annos:
            timestamp_sec = anno['timeStamp'] / 1e6  # microseconds to seconds
            frame_idx = int(round(timestamp_sec * fps))
            frame_indices.append(frame_idx)

        # Filter out frame indices that are out of bounds
        valid_frame_indices = []
        valid_anno_indices = []
        for anno_idx, frame_idx in enumerate(frame_indices):
            if 0 <= frame_idx < total_frames:
                valid_frame_indices.append(frame_idx)
                valid_anno_indices.append(anno_idx)

        if len(valid_frame_indices) == 0:
            raise ValueError(f"No valid frame indices for video: {video_path}")

        # Keep only valid annotations
        annos = [annos[i] for i in valid_anno_indices]

        # Choose the final clip before decode. Long RE10K videos can contain
        # hundreds of valid annotations, but training consumes only n_frames.
        indices = self._temporal_sample(len(valid_frame_indices), fps)

        # random flip in timewise
        if not getattr(self.cfg, 'no_augmentations', False) and random.random() < 0.5:
            indices = indices[::-1].copy()

        selected_frame_indices, annos = select_re10k_clip(
            valid_frame_indices, annos, indices
        )
        frames = torch.from_numpy(
            vr.get_batch(selected_frame_indices).asnumpy()
        )  # (self.n_frames, H, W, 3) uint8
        
        if len(frames) != self.n_frames:
            raise ValueError(
                f"Expected {len(frames)=} to be equal to {self.n_frames=}."
            )
        
        # Convert to float — normalization applied after crop+resize below.
        frames = frames.float().permute(0, 3, 1, 2).contiguous() / 255.0

        # Aspect-preserving center-crop + uniform resize to (self.height, self.width).
        H_src, W_src = frames.shape[-2:]
        box = center_crop_box_for_aspect(H_src, W_src, self.height, self.width)
        frames = apply_spatial_crop(frames, box)
        frames = F.interpolate(frames, size=(self.height, self.width),
                               mode="bilinear", align_corners=False, antialias=True)
        frames = self.img_normalize(frames)

        # Raymap + K at raymap res — pass the crop box + source HxW so normalized
        # intrinsics can be converted to pixel K, shifted by the crop, rescaled.
        raymap_h = getattr(self.cfg, 'raymap_height', None) or frames.shape[-2]
        raymap_w = getattr(self.cfg, 'raymap_width', None) or frames.shape[-1]

        raymaps, raymaps_inv, raymap_scale, raymap_inv_scale, extrinsics_gt, intrinsics_gt = self._generate_raymaps(
            annos,
            np.arange(len(annos), dtype=np.int64),
            image_size=(H_src, W_src),
            raymap_size=(raymap_h, raymap_w),
            crop_box=box,
            frame_hw=(self.height, self.width),
            scene_id=Path(record["anno_path"]).stem,
        )

        return frames, raymaps, raymaps_inv, raymap_scale, raymap_inv_scale, extrinsics_gt, intrinsics_gt
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        # Bounded retry LOOP (not recursion): recursive retries pin each level's
        # partially-decoded tensors via the exception traceback; with many failing
        # records (e.g. bedlam2's misnamed depth files) DataLoader workers ballooned
        # to ~100GB and were OOM-killed, silently losing the sample.
        for _attempt in range(8):
            record = self.records[idx]
            try:
                videos, raymaps, raymaps_inv, raymap_scale, raymap_inv_scale, extrinsics_gt, intrinsics_gt = self._load_video(record)
                break
            except (OSError, ValueError, RuntimeError, decord.DECORDError) as e:
                video_path = record.get('video_path', str(idx))
                print(f"Warning: skipping corrupted/invalid video {video_path}: {e}")
                # backoff: transient filesystem stalls fail all instant
                # retries at once; wait it out before resampling
                time.sleep(min(5.0, 0.3 * 2 ** _attempt))
                idx = random.randint(0, len(self.records) - 1)
        else:
            record = self.records[0]
            videos, raymaps, raymaps_inv, raymap_scale, raymap_inv_scale, extrinsics_gt, intrinsics_gt = self._load_video(record)
        # images = videos[:1].clone() if self.image_to_video else None
        image_latents, video_latents = None, None
        video_metadata = {
            "num_frames": videos.shape[0],
            "height": videos.shape[2],
            "width": videos.shape[3],
        }

        if self.load_video_latent:
            image_latents, video_latents = self._load_video_latent(record)
            # This is hardcoded for now.
            # The VAE's temporal compression ratio is 4.
            # The VAE's spatial compression ratio is 8.
            latent_num_frames = video_latents.size(1)
            if latent_num_frames % 2 == 0:
                n_frames = latent_num_frames * 4
            else:
                n_frames = (latent_num_frames - 1) * 4 + 1

            height = video_latents.size(2) * 8
            width = video_latents.size(3) * 8

            assert video_metadata["num_frames"] == n_frames, "num_frames changed"
            assert video_metadata["height"] == height, "height changed"
            assert video_metadata["width"] == width, "width changed"

        # Load prompt data - either raw or preprocessed embeddings
        caption = ""
        if "caption" in record:
            caption = record["caption"]
        elif "gemini_caption" in record:
            caption = record["gemini_caption"]
        elif "original_caption" in record:
            caption = record["original_caption"]
        video_metadata["has_caption"] = caption != ""
        prompts = self.id_token + caption
        prompt_embeds = None
        prompt_embed_len = None
        if self.load_prompt_embed:
            prompt_embeds, prompt_embed_len = self._load_prompt_embed(record)

        record["height"] = video_metadata["height"]
        record["width"] = video_metadata["width"]
        has_bbox, bbox_render = self._render_bbox(record)

        output = {
            "videos": videos,
            "raymaps": raymaps,
            "raymaps_inv": raymaps_inv,
            "raymap_scale": torch.tensor(raymap_scale, dtype=torch.float32),
            "raymap_inv_scale": torch.tensor(raymap_inv_scale, dtype=torch.float32),
            "extrinsics_gt": extrinsics_gt,
            "intrinsics_gt": intrinsics_gt,
            "video_metadata": video_metadata,
            "bbox_render": bbox_render,
            "has_bbox": has_bbox,
        }

        if prompts is not None:
            output["prompts"] = prompts
        # if images is not None:
        #     output["images"] = images
        if prompt_embeds is not None:
            output["prompt_embeds"] = prompt_embeds
            output["prompt_embed_len"] = prompt_embed_len
        if image_latents is not None:
            output["image_latents"] = image_latents
        if video_latents is not None:
            output["video_latents"] = video_latents

        return output


    def _generate_raymaps(
        self,
        annos: list,
        temporal_indices: np.ndarray,
        image_size: Tuple[int, int],
        raymap_size: Optional[Tuple[int, int]] = None,
        crop_box: Optional[Tuple[int, int, int, int]] = None,
        frame_hw: Optional[Tuple[int, int]] = None,
        scene_id: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate Plucker raymap from camera annotations.

        Annotations store `intrinsics` in normalized [0,1] coords of the source
        video. If `crop_box` is given (top, bot, left, right), normalized K is
        converted to pixel K at (H_src, W_src), shifted by the crop offset, and
        uniformly rescaled to (frame_hw), then uniformly rescaled to (raymap_size)
        — so the resulting raymaps are geometrically valid on the frame grid.

        Args:
            annos: List of annotations (already filtered for valid frames)
            temporal_indices: Temporal sampling indices applied to annotations
            image_size: (H_src, W_src) of the raw video frames
            raymap_size: (H, W) target raymap grid
            crop_box: (top, bot, left, right) aspect-preserving center crop.
                      If None, falls back to legacy behaviour (K directly at raymap res).
            frame_hw: (H_out, W_out) frame resolution after crop+resize.
                      Required when crop_box is given.
        Returns:
            raymaps: (n_frames, 6, H, W) with rays_m normalized to [-1, 1].
            raymaps_inv: (n_frames, 6, H, W) from inverse camera poses, also
                normalized to [-1, 1].
            raymap_scale: float; rays_m_raw = raymaps[:, 3:6] * raymap_scale.
            raymap_inv_scale: float; analogous for raymaps_inv.
        """
        H_src, W_src = image_size
        H, W = raymap_size if raymap_size is not None else image_size

        use_crop = crop_box is not None and frame_hw is not None

        # Get the final selected annotations after temporal sampling
        final_annos = [annos[idx] for idx in temporal_indices]

        # First pass: parse all camera parameters
        intrinsics_list = []
        R_c2w_list = []
        t_c2w_list = []

        for anno in final_annos:
            focal_length_x = anno['intrinsics'][0]
            focal_length_y = anno['intrinsics'][1]
            principal_point_x = anno['intrinsics'][2]
            principal_point_y = anno['intrinsics'][3]

            if use_crop:
                # Convert normalized K → pixel K at source res → crop+resize → raymap uniform rescale.
                K_src = np.array([
                    [focal_length_x * W_src, 0.0, principal_point_x * W_src],
                    [0.0, focal_length_y * H_src, principal_point_y * H_src],
                    [0.0, 0.0, 1.0],
                ], dtype=np.float64)
                H_f, W_f = frame_hw
                K_frame = adjust_K_for_crop_resize(K_src, crop_box, H_f, W_f)
                sx = W / W_f
                sy = H / H_f
                fx = K_frame[0, 0] * sx
                fy = K_frame[1, 1] * sy
                cx = K_frame[0, 2] * sx
                cy = K_frame[1, 2] * sy
            else:
                fx = W * focal_length_x
                fy = H * focal_length_y
                cx = W * principal_point_x
                cy = H * principal_point_y

            intrinsics = torch.tensor([
                [fx, 0., cx],
                [0., fy, cy],
                [0., 0., 1.]
            ], dtype=torch.float32)
            
            # Parse extrinsics: 3x4 matrix [R | t]
            pose = anno['pose']
            R_w2c = torch.tensor([
                [pose[0], pose[1], pose[2]],
                [pose[4], pose[5], pose[6]],
                [pose[8], pose[9], pose[10]]
            ], dtype=torch.float32)
            
            t_w2c = torch.tensor([pose[3], pose[7], pose[11]], dtype=torch.float32)
            
            # Convert to camera-to-world (c2w)
            R_c2w = R_w2c.T
            t_c2w = -R_c2w @ t_w2c
            
            intrinsics_list.append(intrinsics)
            R_c2w_list.append(R_c2w)
            t_c2w_list.append(t_c2w)
        
        
        travel = sum(torch.norm(t_c2w_list[i] - t_c2w_list[i-1]).item() for i in range(1, len(t_c2w_list)))
        # print(f"camera travel: {travel:.4f}", "min_camera_travel:", getattr(self.cfg, 'min_camera_travel', 0.0))

        # # Filter near-static sequences
        # min_travel = getattr(self.cfg, 'min_camera_travel', 0.0)
        # if min_travel > 0.0:
        #     travel = sum(torch.norm(t_c2w_list[i] - t_c2w_list[i-1]).item() for i in range(1, len(t_c2w_list)))
        #     if travel < min_travel:
        #         raise ValueError(f"Near-static camera (travel={travel:.4f} < {min_travel}), skipping.")

        # Normalize cameras: make first camera identity
        R_c2w_normalized, t_c2w_normalized = self._normalize_cameras(
            R_c2w_list, t_c2w_list
        )
        
        # Generate raymaps with normalized cameras
        raymaps_list = []
        raymaps_inv_list = []
        extrinsics_gt_list = []
        intrinsics_gt_list = []
        for intrinsics, R_c2w, t_c2w in zip(intrinsics_list, R_c2w_normalized, t_c2w_normalized):
            # Build 4x4 extrinsic matrix (camera-to-world)
            extrinsics = torch.eye(4, dtype=torch.float32)
            extrinsics[:3, :3] = R_c2w
            extrinsics[:3, 3] = t_c2w

            # Invert normalized c2w to induce opposite camera motion.
            # Since camera 0 is normalized to identity, it remains identity after inversion.
            extrinsics_inv = self._invert_c2w(extrinsics)

            # Generate raymaps at target resolution for correct geometry
            rays_d, rays_m = camera_to_raymap(
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                height=H,
                width=W
            )
            rays_d_inv, rays_m_inv = camera_to_raymap(
                intrinsics=intrinsics,
                extrinsics=extrinsics_inv,
                height=H,
                width=W
            )

            # Concatenate direction and moment: [D, M] shape (6, H, W)
            raymap = torch.cat([rays_d, rays_m], dim=0)
            raymap_inv = torch.cat([rays_d_inv, rays_m_inv], dim=0)
            raymaps_list.append(raymap)
            raymaps_inv_list.append(raymap_inv)
            extrinsics_gt_list.append(extrinsics)
            intrinsics_gt_list.append(intrinsics)

        raymaps = torch.stack(raymaps_list, dim=0)  # (n_frames, 6, H, W)
        raymaps_inv = torch.stack(raymaps_inv_list, dim=0)  # (n_frames, 6, H, W)
        extrinsics_gt = torch.stack(extrinsics_gt_list, dim=0)  # (n_frames, 4, 4) c2w OpenCV
        intrinsics_gt = torch.stack(intrinsics_gt_list, dim=0)  # (n_frames, 3, 3) pixel K at raymap res

        # Per-sample max-abs normalization of the Plucker moments.
        # Forward and inverse raymaps encode different camera motions; scale
        # each independently so both fill the [-1, 1] range.
        # Under the G3 depth gauge both raymaps share ONE scene-depth scale (same clip),
        # matching the geometry loaders' shared-scalar convention.
        clip_ts = [a["timeStamp"] for a in final_annos]
        raymaps, raymap_scale = self._normalize_ray_moments(
            raymaps, scene_id=scene_id, clip_timestamps=clip_ts)
        raymaps_inv, raymap_inv_scale = self._normalize_ray_moments(
            raymaps_inv, scene_id=scene_id, clip_timestamps=clip_ts)

        return raymaps, raymaps_inv, raymap_scale, raymap_inv_scale, extrinsics_gt, intrinsics_gt

    

    def _invert_c2w(self, c2w: torch.Tensor) -> torch.Tensor:
        """
        Invert a camera-to-world transform.
        """
        R = c2w[:3, :3]
        t = c2w[:3, 3]

        inv_c2w = torch.eye(4, dtype=c2w.dtype, device=c2w.device)
        R_t = R.T
        inv_c2w[:3, :3] = R_t
        inv_c2w[:3, 3] = -R_t @ t
        return inv_c2w
    
    def _normalize_cameras(
        self,
        R_c2w_list: list,
        t_c2w_list: list
    ) -> Tuple[list, list]:
        """
        Normalize cameras so first camera has identity rotation and zero translation.
        
        Args:
            R_c2w_list: List of rotation matrices (camera-to-world)
            t_c2w_list: List of translation vectors (camera centers in world)
        
        Returns:
            R_normalized: List of normalized rotation matrices
            t_normalized: List of normalized translation vectors
        """
        # Get first camera parameters
        R_first = R_c2w_list[0]  # (3, 3)
        t_first = t_c2w_list[0]  # (3,)
        
        # Transform to make first camera identity
        # We want: R_first_new = R_first^T @ R_first = I
        # For other cameras: R_new = R_first^T @ R_old
        # For translation: t_new = R_first^T @ (t_old - t_first)
        
        R_first_inv = R_first.T
        
        R_normalized = []
        t_normalized = []
        
        for R_c2w, t_c2w in zip(R_c2w_list, t_c2w_list):
            # Apply transformation
            R_new = R_first_inv @ R_c2w
            t_new = R_first_inv @ (t_c2w - t_first)
            
            R_normalized.append(R_new)
            t_normalized.append(t_new)
        
        return R_normalized, t_normalized
    
    def _normalize_ray_moments(self, raymaps: torch.Tensor,
                               scene_id: Optional[str] = None,
                               clip_timestamps=None) -> Tuple[torch.Tensor, float]:
        """
        Per-sample max-abs normalization of the Plucker moment channels.

        Divides channels 3:6 by a single scalar (max absolute moment across the
        whole clip) so the output moment lies in [-1, 1]. Directions (0:3) are
        already unit-norm, so each component is inherently in [-1, 1].

        The scale is floored at MIN_RAYMAP_SCALE: near-stationary RE10K clips
        have |M| ≈ 0, and dividing by that tiny max would amplify noise and
        saturate every moment to ±1. The floor keeps slow clips well-conditioned
        and is a no-op for high-motion clips where |M| dominates.

        Args:
            raymaps: (T, 6, H, W)

        Returns:
            normalized: (T, 6, H, W) with channels 3:6 in [-1, 1]
            scale: float such that rays_m_raw = rays_m_norm * scale
        """
        from ._scale_norm import (is_parallax_mode, parallax_ray_encoding,
                                  companded_origin_ray_norm)
        if is_parallax_mode(self.cfg) and parallax_ray_encoding(self.cfg) == "companded_origin":
            # keep ray-channel semantics consistent with the geometry loaders under scheme C
            # (spatially-constant mu-law origins); gauge = per-clip max origin (no metric depth).
            return companded_origin_ray_norm(raymaps, self.cfg, MIN_RAYMAP_SCALE)
        from ._depth_gauge import gauge_scale, is_gauge_mode
        if is_gauge_mode(self.cfg) and scene_id is not None:
            # G3: same gauge as the geometry loaders -- moments / (kappa * median metric
            # scene depth, expressed in this scene's COLMAP units). NOT clamped: clamping
            # is the only op that can desync the pose channel from the depth channel.
            # A scene missing from the table RAISES (see _depth_gauge.gauge_scale): there
            # is no silent fall-through to max-norm, because that would put two magnitude
            # codes on one ray channel. Prune ungauged scenes from the records instead.
            out = raymaps.clone()
            s_g3 = gauge_scale(self.cfg, scene_id, clip_timestamps)
            out[:, 3:6] = raymaps[:, 3:6] / s_g3
            return out, s_g3
        moments = raymaps[:, 3:6, :, :]
        if self.cfg.get("scale_preserving", False):
            divisor = torch.tensor(GLOBAL_MOMENT_DIVISOR,
                                   dtype=moments.dtype, device=moments.device).view(1, 3, 1, 1)
            normalized = raymaps.clone()
            normalized[:, 3:6] = moments / divisor
            return normalized, 1.0
        max_abs = float(moments.abs().max().item())
        scale = max(max_abs, MIN_RAYMAP_SCALE)
        if scale < 1e-8:
            scale = 1.0
        raymaps = raymaps.clone()
        raymaps[:, 3:6, :, :] = torch.clamp(raymaps[:, 3:6, :, :] / scale, -1.0, 1.0)
        return raymaps, scale
    
    def _get_best_matching_frame(self, frame_timestamp: int, annos: list, max_distance: float) -> Optional[Dict[str, Any]]:
        """
        Find the best matching annotation for a given frame timestamp.
        Returns None if no match within max_distance.
        """
        best_match = None
        min_distance = float('inf')
        
        for anno_idx, anno in enumerate(annos):
            distance = abs(anno['timeStamp'] - frame_timestamp)
            if distance < max_distance and distance < min_distance:
                min_distance = distance
                best_match = {
                    'caseIdx': anno_idx,
                    'distance': distance
                }
        
        return best_match
    
    def _load_prompt_embed(self, record):
        """Load cached umT5-XXL embedding for this scene.

        Layout: <data_root>/prompt_embeds/<split>/<scene_id>.pt where
        <split> is the parent dir of record['anno_path'] ('train' or 'test')
        and <scene_id> is the stem of record['anno_path'].
        """
        anno_path = Path(record["anno_path"])
        split_dir = anno_path.parent.name
        scene_id = anno_path.stem
        embed_path = self.data_root / "prompt_embeds" / split_dir / f"{scene_id}.pt"

        prompt_embed = torch.load(embed_path, map_location="cpu", weights_only=True)
        prompt_embed_len = prompt_embed.size(0)
        if prompt_embed_len < self.max_text_tokens:
            padding = torch.zeros(
                self.max_text_tokens - prompt_embed.size(0),
                prompt_embed.size(1),
                dtype=prompt_embed.dtype,
                device=prompt_embed.device,
            )
            prompt_embed = torch.cat([prompt_embed, padding], dim=0)
        return prompt_embed, prompt_embed_len


if __name__ == "__main__":
    # Test script for loading RE10K videos and annotations
    from pathlib import Path
    from omegaconf import DictConfig
    
    # data_root = Path(__file__).parent.parent / "data" / "re10k"
    data_root = Path("./data/re10k")
    
    cfg = DictConfig({
        "data_root": str(data_root),
        "debug": True,
        "metadata_path": "",  # Not used for RE10K
        "auto_download": False,
        "force_download": False,
        "test_percentage": 0.1,
        "id_token": "",
        "height": 480,
        "width": 640,
        "n_frames": 49,
        "fps": 30,
        "trim_mode": "random_cut",
        "pad_mode": "slowdown",
        "filtering": DictConfig({
            "disable": True,  # Disable filtering for testing
            "height": None,
            "width": None,
            "n_frames": None,
            "fps": None,
        }),
        "load_video_latent": False,
        "load_prompt_embed": True,
        "augmentation": DictConfig({
            "random_flip": None,
            "ratio": None,
            "scale": None,
        }),
        "image_to_video": False,
        "max_text_tokens": 1,
        "check_video_path": False,
    })
    
    print("=" * 80)
    print("Testing RE10K Dataset")
    print("=" * 80)
    
    # Test loading dataset
    print("\n1. Creating dataset instance...")
    dataset = RE10KDataset(cfg, split="training")
    print(f"   ✓ Dataset created with {len(dataset)} records")
    
    # Test loading a single item
    print("\n2. Loading first item...")
    try:
        item = dataset[0]
        rays_d = item["raymaps"][:, :3]
        rays_m = item["raymaps"][:, 3:]
        rays_m_inv = item["raymaps_inv"][:, 3:]
        print(f"   ✓ videos:            {item['videos'].shape}  range=[{item['videos'].min():.2f}, {item['videos'].max():.2f}]")
        print(f"   ✓ raymaps:           {item['raymaps'].shape}")
        print(f"     rays_d range       [{rays_d.min():.3f}, {rays_d.max():.3f}]")
        print(f"     rays_m range       [{rays_m.min():.3f}, {rays_m.max():.3f}]")
        print(f"     raymap_scale       {float(item['raymap_scale']):.4f}")
        print(f"     rays_m_inv range   [{rays_m_inv.min():.3f}, {rays_m_inv.max():.3f}]")
        print(f"     raymap_inv_scale   {float(item['raymap_inv_scale']):.4f}")
        assert rays_d.abs().max() <= 1.0 + 1e-6, "rays_d out of [-1, 1]"
        assert rays_m.abs().max() <= 1.0 + 1e-6, "rays_m not normalized to [-1, 1]"
        assert rays_m_inv.abs().max() <= 1.0 + 1e-6, "rays_m_inv not normalized to [-1, 1]"

        item["raymaps"] = torch.nn.functional.interpolate(
            item["raymaps"], size=item["videos"].shape[-2:], mode="bilinear", align_corners=False)
        all_videos = torch.cat([item['videos'], item['raymaps'][:, :3], item['raymaps'][:, 3:]], dim=-1)
        import imageio
        imageio.mimwrite('test.mp4', ((all_videos.permute(0, 2, 3, 1) / 2 + 0.5).clamp(0., 1.).numpy() * 255).astype(np.uint8))
    except Exception as e:
        print(f"   ✗ Error loading item: {e}")
        import traceback
        traceback.print_exc()
    
    # Test multiple items
    print("\n3. Testing batch loading (first 3 items)...")
    for i in range(min(3, len(dataset))):
        try:
            item = dataset[i]
            print(f"   ✓ Item {i}: video_path={dataset.records[i]['video_path']}")
        except Exception as e:
            print(f"   ✗ Item {i} failed: {e}")
    
    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)
