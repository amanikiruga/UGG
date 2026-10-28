"""Camera recovery from UGG's predicted ray maps (prediction-only; never reads GT).

The model outputs a 6-channel ray map per frame: Plücker direction (3) and moment (3).
`raymap_to_camera` turns each frame's rays into intrinsics K and a camera-to-world pose by
RANSAC/PnP on the ray field; `recover_cameras` optionally refits ONE shared intrinsic matrix
for the whole clip (`shared_intrinsics=True`, recommended: 1.4-10x better rotation error).

Vendored verbatim from the evaluation harness that produced the paper's pose tables
(video_world_model_new/datasets/aria.py and scripts/shared_intrinsics_camera.py), so poses
recovered here are identical to the published ones.
"""
from __future__ import annotations
import dataclasses
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple

from ugg._shared_intrinsics import recover_cameras_shared_intrinsics

# [-1, 1] normalized Plücker moments -> raw moments (the legacy Aria normalisation constant).
_MOMENT_ABS_MAX = max(abs(-24.74799156188965), abs(24.935894012451172))

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
    max_points: int = 5000,
    ransac_reproj_threshold: float = 1.0,
    min_abs_z: float = 1e-6,
    moments_are_normalized: bool = True,
    norm_source: str = "aria",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert Plucker raymaps back to camera intrinsics/extrinsics.

    This assumes ray moments are raw Plucker moments (not normalized/rescaled):
        M = O x D
    where O is camera center and D is unit ray direction.

    Args:
        raymaps: (6, H, W) or (B, 6, H, W), channels are [rays_d, rays_m]
        max_points: max number of rays used for homography estimation
        ransac_reproj_threshold: RANSAC reprojection threshold in pixels
        min_abs_z: minimal |D_z| for stable projective conversion
    Returns:
        intrinsics: (3, 3) or (B, 3, 3)
        extrinsics: (4, 4) or (B, 4, 4), camera-to-world matrices
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
    if moments_are_normalized:
        raymaps[:, 3:] = raymaps[:, 3:] * _MOMENT_ABS_MAX
    
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


@dataclasses.dataclass
class Cameras:
    """Per-frame cameras for one clip (OpenCV convention, first frame at the origin)."""
    intrinsics: np.ndarray   # (T, 3, 3)
    c2w: np.ndarray          # (T, 4, 4)


def recover_cameras(raymaps: torch.Tensor, shared_intrinsics: bool = True) -> Cameras:
    """(T, 6, h, w) predicted ray maps -> Cameras.

    shared_intrinsics=False reproduces the per-frame ("legacy") recovery; True fits one K for
    the whole clip, which is the recommended setting (the paper's `release_si` rows).
    Camera centres are defined up to a global scale; the shared path returns them in the
    model's native ray units.
    """
    raymaps = raymaps.detach().float()
    if not shared_intrinsics:
        K, c2w = raymap_to_camera(raymaps)
        return Cameras(K.cpu().numpy().astype(np.float64), c2w.cpu().numpy().astype(np.float64))
    if raymaps.ndim != 4 or len(raymaps) < 2:
        raise ValueError("shared-intrinsics recovery needs at least two (T, 6, h, w) frames")
    K_native, _ = raymap_to_camera(raymaps, moments_are_normalized=False)
    res = recover_cameras_shared_intrinsics(
        raymaps[:, :3].cpu().numpy(), raymaps[:, 3:].cpu().numpy(),
        K_native.cpu().numpy().astype(np.float64), max_points_per_frame=600, max_nfev=80)
    return Cameras(res.intrinsics.astype(np.float64), res.extrinsics.astype(np.float64))


def encode_cameras(c2w, intrinsics, image_hw, raymap_hw, min_scale: float = 1.0):
    """Cameras -> the model's ray conditioning, encoded exactly as in training.

    c2w: (T, 4, 4) OpenCV camera-to-world; intrinsics: (3, 3) or (T, 3, 3) in pixels of the
    model input (image_hw, after the center crop + resize). Poses are re-expressed relative to
    the first camera, rays are Pluecker [direction | origin x direction] on the raymap grid, and
    moments are divided by the clip's largest moment (floored at `min_scale`), clamped to [-1, 1].
    Returns ((T, 6, h, w) float tensor, scale). Depth conditioning on the same clip must use the
    same scale (see ugg.depth.encode_depth).
    """
    from datasets.aria_new import camera_to_raymap

    c2w = np.asarray(c2w, dtype=np.float64)
    K = np.asarray(intrinsics, dtype=np.float64)
    K = np.broadcast_to(K, (len(c2w), 3, 3)) if K.ndim == 2 else K
    first_inv = np.linalg.inv(c2w[0])
    (H, W), (h, w) = image_hw, raymap_hw
    sx, sy = w / W, h / H
    rays = []
    for T_i, K_i in zip(c2w, K):
        E = torch.from_numpy(first_inv @ T_i).float()
        Kr = torch.tensor([[K_i[0, 0] * sx, 0, K_i[0, 2] * sx], [0, K_i[1, 1] * sy, K_i[1, 2] * sy], [0, 0, 1]],
                          dtype=torch.float32)
        d, m = camera_to_raymap(Kr, E, h, w)
        rays.append(torch.cat([d, m], dim=0))
    rays = torch.stack(rays)
    scale = max(float(rays[:, 3:].abs().max()), min_scale)
    rays[:, 3:] = (rays[:, 3:] / scale).clamp(-1, 1)
    return rays, scale
