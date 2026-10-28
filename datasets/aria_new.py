
import os
import json
import random
import torch
import numpy as np
import cv2
import imageio.v3 as iio
import torch.nn.functional as F
from typing import Any, Dict, Tuple, Optional, List
from torchvision.transforms import v2 as transforms
from tqdm import tqdm
from .video_base import VideoDataset
# import path
from pathlib import Path
# from algorithms.wan.utils.raymap import camera_to_raymap

DEPTH_STATS = None
DEPTH_STATS_PATH = "depth_stats.json"

def load_depth_stats(depth_stats_path: Path): 
    global DEPTH_STATS
    with open(depth_stats_path, "r") as f: 
        DEPTH_STATS = json.load(f)
        # convert depth from millimeters to meters
        DEPTH_STATS["min"] = DEPTH_STATS["min"] / 1000.0
        DEPTH_STATS["max"] = DEPTH_STATS["max"] / 1000.0
# done

# Global ray moment statistics computed over the Aria training split.
# Used to normalize moments to [-1, 1] consistently across all sequences.
_MOMENT_MIN: float = -24.74799156188965
_MOMENT_MAX: float =  24.935894012451172

_ARIA_FOCAL: float = 350.0
_ARIA_SRC_W: int = 640 
_ARIA_SRC_H: int = 480


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


# The inverse of `camera_to_raymap` is `datasets/re10k.py::raymap_to_camera`, which is the
# copy the model and the evaluation path use. A second, older implementation used to sit
# here; it predated normalized moments (no raymap_scale, no companded-origin encoding) and
# nothing in the repo imported it, so two divergent inverses were being carried for one
# forward map. Removed 2026-08-27.



class AriaDataset(VideoDataset):

    def _preprocess_depth(self, depth: np.ndarray) -> np.ndarray: 
        if DEPTH_STATS is not None: 
            # convert depth from millimeters to meters
            depth = depth / 1000.0
            depth = depth.clip(DEPTH_STATS["min"], DEPTH_STATS["max"])
            depth = (depth - DEPTH_STATS["min"]) / (DEPTH_STATS["max"] - DEPTH_STATS["min"])
        else: 
            raise ValueError("Depth stats are not loaded")
        return depth.astype(np.float32)

    def _read_depth(self, depth_dir: Path, frame_files: List[Path]) -> np.ndarray: 
        """Read depth PNGs"""
        frames = [] 
        for f in frame_files: 
            depth_file_name = f.stem.replace("vignette", "depth") + ".png"
            depth_file = depth_dir / depth_file_name
            depth = np.array(iio.imread(str(depth_file))).astype(np.float32)
            frames.append(depth)
        return np.stack(frames)

    def _load_records(self) -> List[Dict[str, Any]]:
        """
        Load Aria records directly train_undistorted directory structure
        In each sequence dir there is a subdir whose name is a number which contains rgb/, trajectory.csv and depth/
        """
        # Determine which split to load
        
        seq_dirs = sorted(
            [d for d in self.data_root.iterdir() if d.is_dir() and d.name.isdigit()],
            key=lambda p: int(p.name),
        )
        records = []
        for seq_dir in tqdm(seq_dirs, desc="Loading Aria sequences"): 
            rgb_dir = seq_dir / "rgb"
            traj_path = seq_dir / "trajectory.csv"
            if not rgb_dir.exists() or not traj_path.exists(): 
                continue
            record = {
                "seq_id": seq_dir.name,
                "rgb_dir": str(seq_dir.relative_to(self.data_root) / "rgb"),
                "traj_path": str(seq_dir.relative_to(self.data_root) / "trajectory.csv"),
                "fps": 10,
                "caption": "",
            }
            depth_dir = seq_dir / "depth"
            if depth_dir.exists(): 
                record["depth_dir"] = str(seq_dir.relative_to(self.data_root) / "depth")
            records.append(record)
            #done

        print(f"Found {len(records)} Aria sequences in {self.data_root}")
        
        # Apply filtering
        len_pre_filter = len(records)
        if not self.filtering.disable:
            records = [record for record in records if self._filter_record(record)]
        len_post_filter = len(records)
        
        print(
            f"Aria: filtered {len_pre_filter - len_post_filter} records", 
            f"retaining {len_post_filter} ({len_post_filter / max(len_pre_filter, 1):.2%})"
        )
        # Shuffle with fixed seed for reproducibility
        random.Random(0).shuffle(records)

        n_test = int(len(records) * self.test_percentage)

        if self.split == "training": 
            records = records[n_test:]
        elif self.split == "validation": 
            records = records[:n_test]
        elif self.split == "test": 
            records = records[:n_test]
        
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

    @staticmethod 
    def _quat_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray: 
        """Quaternion (x,y,z,w) -> 3x3 rotation matrix."""
        return np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)], 
            [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)], 
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]], 
            dtype=np.float64
        )
    
    def _load_video(self, record: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given a record, return:
            frames: (n_frames, 3, H, W)
            raymaps: (n_frames, 6, H, W)
            raymaps_inv: (n_frames, 6, H, W)
            depths: (n_frames, 1, H, W) or None
        """

        
        rgb_dir = self.data_root / record['rgb_dir']
        traj_path = self.data_root / record['traj_path']

        # Normalized intrinsics for the image AFTER 90° CW rotation.
        # The rotation turns (H=480, W=640) into (H=640, W=480), so the
        # width direction is the old height axis and vice versa.
        fx_n = _ARIA_FOCAL / _ARIA_SRC_H  # focal / new_width (480)
        fy_n = _ARIA_FOCAL / _ARIA_SRC_W  # focal / new_height (640)
        cx_n = 0.5
        cy_n = 0.5
        

        annos = [] 
        with open(traj_path, 'r') as f: 
            f.readline()
            for line in f: 
                parts = line.strip().split(',') # csv 
                if len(parts) < 10: 
                    raise ValueError(f"Invalid trajectory file: {traj_path}")
                timestamp_us = int(parts[1])
                tx, ty, tz = float(parts[3]), float(parts[4]), float(parts[5])
                qx, qy, qz, qw = float(parts[6]), float(parts[7]), float(parts[8]), float(parts[9])

                R_c2w = self._quat_to_rotmat(qx, qy, qz, qw)
                t_c2w = np.array([tx, ty, tz])

                R_w2c = R_c2w.T 
                t_w2c = -R_w2c @ t_c2w

                pose = [
                    R_w2c[0, 0], R_w2c[0, 1], R_w2c[0, 2], t_w2c[0],
                    R_w2c[1, 0], R_w2c[1, 1], R_w2c[1, 2], t_w2c[1],
                    R_w2c[2, 0], R_w2c[2, 1], R_w2c[2, 2], t_w2c[2],
                ]
                annos.append({
                    'timestamp': timestamp_us, 
                    'intrinsics': [fx_n, fy_n, cx_n, cy_n, 0.0, 0.0], 
                    'pose': pose,
                })
        
        if len(annos) == 0: 
            raise ValueError(f"No trajectory entries for {traj_path}")
        
        # load RGB frames
        frame_files = sorted(rgb_dir.glob("vignette*.jpg"))
        total_frames = len(frame_files)
        if total_frames == 0 or total_frames != len(annos): 
            raise ValueError(f"Expected {len(annos)} frames, got {total_frames}")
        
        fps = record.get('fps', 10)
        n_frames = total_frames 
        indices = self._temporal_sample(n_frames, fps)

        if not getattr(self.cfg, 'no_augmentations', False) and random.random() < 0.5: 
            indices = indices[::-1].copy()
        
        selected_files = [frame_files[i] for i in indices]
        frames_np = np.stack([np.array(iio.imread(str(f))) for f in selected_files])
        frames = torch.from_numpy(frames_np)
        # frames are rotated anti-clockwise by 90 degrees
        # let's rotate them back
        frames = torch.rot90(frames, k=3, dims=[1, 2])
        if len(frames) != self.n_frames:
            raise ValueError(f"Expected {len(frames)=} to be equal to {self.n_frames=}.")
        
        # Convert to float and normalize
        frames = frames.float().permute(0, 3, 1, 2).contiguous() / 255.0
        
        # Apply transforms
        if "has_bbox" in record and record["has_bbox"]:
            frames = self.no_augment_transforms(frames)
        else:
            frames = self.augment_transforms(frames)
        frames = self.img_normalize(frames)
        
        # Generate raymap from camera parameters with normalization
        raymap_h = getattr(self.cfg, 'raymap_height', None) or frames.shape[-2]
        raymap_w = getattr(self.cfg, 'raymap_width', None) or frames.shape[-1]
        
        raymaps = self._generate_raymaps(
            annos,
            indices,
            frames.shape[-2:],
            raymap_size = (raymap_h, raymap_w)
        )

        depths = None
        if "depth_dir" in record: 
            depth_dir = self.data_root / record["depth_dir"]
            selected_depth_files = [frame_files[i] for i in indices]
            depth_np = self._read_depth(depth_dir, selected_depth_files)
            depth_np = self._preprocess_depth(depth_np)
            depths = torch.from_numpy(depth_np).unsqueeze(1) # (n_frames, 1, H, W)
            depths = torch.rot90(depths, k=3, dims=[2, 3])
            depths = F.interpolate(depths, size=frames.shape[-2:], mode='bilinear', align_corners=False)
            depth_normalize = transforms.Normalize(mean=[0.5], std=[0.5])
            depths = depth_normalize(depths)

        return frames, raymaps, depths
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        record = self.records[idx]

        if DEPTH_STATS is None and "depth_dir" in record: 
            depth_stats_path = self.data_root / DEPTH_STATS_PATH
            if depth_stats_path.exists(): 
                load_depth_stats(depth_stats_path)

        # Load video data - either raw or preprocessed latents
        try:
            videos, raymaps, depths = self._load_video(record)
        except (OSError, ValueError) as e:
            video_path = record.get('video_path', str(idx))
            print(f"Warning: skipping corrupted/invalid video {video_path}: {e}")
            with open("ignore/failed_videos.txt", "a") as f:
                f.write(video_path + "\n")
            return self.__getitem__(random.randint(0, len(self.records) - 1))
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
            "video_metadata": video_metadata,
            "bbox_render": bbox_render,
            "has_bbox": has_bbox,
        }

        if depths is not None: 
            output["depths"] = depths
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
        raymap_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Generate Plucker raymap from camera annotations.

        Normalizes cameras such that the first camera has identity rotation
        and zero translation.

        Args:
            annos: List of annotations (already filtered for valid frames)
            temporal_indices: Temporal sampling indices applied to annotations
            image_size: (H, W) of the processed frames
        Returns:
            raymaps: Tensor of shape (n_frames, 6, H, W) containing [rays_d, rays_m]
        """
        H, W = raymap_size if raymap_size is not None else image_size

        # Get the final selected annotations after temporal sampling
        final_annos = [annos[idx] for idx in temporal_indices]

        # First pass: parse all camera parameters
        intrinsics_list = []
        R_c2w_list = []
        t_c2w_list = []

        for anno in final_annos:
            # Parse intrinsics (normalized coordinates)
            focal_length_x = anno['intrinsics'][0]
            focal_length_y = anno['intrinsics'][1]
            principal_point_x = anno['intrinsics'][2]
            principal_point_y = anno['intrinsics'][3]

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

        # Normalize cameras: make first camera identity
        R_c2w_normalized, t_c2w_normalized = self._normalize_cameras(
            R_c2w_list, t_c2w_list
        )

        # Generate raymaps with normalized cameras
        raymaps_list = []
        for intrinsics, R_c2w, t_c2w in zip(intrinsics_list, R_c2w_normalized, t_c2w_normalized):
            extrinsics = torch.eye(4, dtype=torch.float32)
            extrinsics[:3, :3] = R_c2w
            extrinsics[:3, 3] = t_c2w

            rays_d, rays_m = camera_to_raymap(
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                height=H,
                width=W
            )
            raymaps_list.append(torch.cat([rays_d, rays_m], dim=0))

        raymaps = torch.stack(raymaps_list, dim=0)  # (n_frames, 6, H, W)

        # Normalize ray moments to [-1, 1] using global dataset stats
        raymaps = self._normalize_ray_moments(raymaps)

        return raymaps

    

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
    
    def _normalize_ray_moments(self, raymaps: torch.Tensor) -> torch.Tensor:
        """
        Normalize ray moments to [-1, 1] using global dataset-wide min/max.

        Args:
            raymaps: (n_frames, 6, H, W) — channels 0:3 are directions, 3:6 are moments

        Returns:
            raymaps with channels 3:6 mapped to [-1, 1] and clamped.
        """
        # ray_moments = raymaps[:, 3:6, :, :]
        # scale = (_MOMENT_MAX - _MOMENT_MIN) / 2.0          # 21.2669
        # shift = (_MOMENT_MAX + _MOMENT_MIN) / 2.0          # -2.4903
        # raymaps[:, 3:6, :, :] = torch.clamp((ray_moments - shift) / scale, -1.0, 1.0)
        
        max_abs = max(abs(_MOMENT_MIN), abs(_MOMENT_MAX))
        raymaps[:, 3:6, :, :] = torch.clamp(raymaps[:, 3:6, :, :] / max_abs, -1.0, 1.0)

        # log norm 

        # import math
        # _LOG_MOMENT_MIN = math.copysign(math.log1p(abs(_MOMENT_MIN)), _MOMENT_MIN)
        # _LOG_MOMENT_MAX = math.copysign(math.log1p(abs(_MOMENT_MAX)), _MOMENT_MAX)
        # log_scale = (_LOG_MOMENT_MAX - _LOG_MOMENT_MIN) / 2.0
        # log_shift = (_LOG_MOMENT_MAX + _LOG_MOMENT_MIN) / 2.0
        # log_moments = torch.sign(ray_moments) * torch.log1p(ray_moments.abs())
        # raymaps[:, 3:6, :, :] = torch.clamp((log_moments - log_shift) / log_scale, -1.0, 1.0)

        return raymaps
    
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
        return torch.zeros(self.max_text_tokens, 4096), self.max_text_tokens


if __name__ == "__main__":
    # Test script for loading Aria videos and annotations
    from pathlib import Path
    from omegaconf import DictConfig
    
    # data_root = Path(__file__).parent.parent / "data" / "Aria"
    data_root = Path("./data/Aria")
    
    cfg = DictConfig({
        "data_root": str(data_root),
        "debug": True,
        "metadata_path": "",  # Not used for Aria
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
    print("Testing Aria Dataset")
    print("=" * 80)
    
    # Test loading dataset
    print("\n1. Creating dataset instance...")
    dataset = AriaDataset(cfg, split="training")
    print(f"   ✓ Dataset created with {len(dataset)} records")
    
    # Test loading a single item
    print("\n2. Loading first item...")
    try:
        item = dataset[0]
        print(f"   ✓ Item loaded successfully")
        print(f"   - Videos shape: {item['videos'].shape}")
        if isinstance(item['videos'], tuple):
            frames, raymaps = item['videos']
            print(f"   - Frames shape: {frames.shape}")
            print(f"   - Raymaps shape: {raymaps.shape}")
        print(f"   - Prompts: '{item['prompts']}'")
        print(f"   - Video metadata: {item['video_metadata']}")
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