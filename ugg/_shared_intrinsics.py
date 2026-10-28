from __future__ import annotations
#!/usr/bin/env python3
"""Prediction-only camera recovery with one intrinsic matrix per clip.

The existing raymap-to-camera implementation decomposes every frame's ray
homography independently.  This module keeps the Pluecker-center recovery but
fits a single upper-triangular pinhole calibration across all frames.  Only
predicted ray directions/moments and (optionally) independently recovered
predicted intrinsics are consumed; dataset camera annotations are not inputs.
"""


import dataclasses
import math
from typing import Any

import numpy as np
from scipy.optimize import least_squares


@dataclasses.dataclass(frozen=True)
class SharedIntrinsicsResult:
    intrinsics: np.ndarray
    extrinsics: np.ndarray
    shared_intrinsic: np.ndarray
    initial_shared_intrinsic: np.ndarray
    optimizer_success: bool
    optimizer_message: str
    optimizer_nfev: int
    initial_ray_angle_mean_deg: float
    initial_ray_angle_median_deg: float
    initial_ray_angle_p95_deg: float
    final_ray_angle_mean_deg: float
    final_ray_angle_median_deg: float
    final_ray_angle_p95_deg: float
    observations: int

    def summary(self) -> dict[str, Any]:
        return {
            "shared_intrinsic": self.shared_intrinsic.tolist(),
            "initial_shared_intrinsic": self.initial_shared_intrinsic.tolist(),
            "optimizer_success": self.optimizer_success,
            "optimizer_message": self.optimizer_message,
            "optimizer_nfev": self.optimizer_nfev,
            "initial_ray_angle_mean_deg": self.initial_ray_angle_mean_deg,
            "initial_ray_angle_median_deg": self.initial_ray_angle_median_deg,
            "initial_ray_angle_p95_deg": self.initial_ray_angle_p95_deg,
            "final_ray_angle_mean_deg": self.final_ray_angle_mean_deg,
            "final_ray_angle_median_deg": self.final_ray_angle_median_deg,
            "final_ray_angle_p95_deg": self.final_ray_angle_p95_deg,
            "observations": self.observations,
        }


def intrinsic_components(intrinsics: np.ndarray) -> np.ndarray:
    """Return ``[fx, fy, skew, cx, cy]`` for each intrinsic matrix."""

    k = np.asarray(intrinsics, dtype=np.float64)
    if k.ndim == 2:
        k = k[None]
    if k.ndim != 3 or k.shape[1:] != (3, 3):
        raise ValueError(f"Expected intrinsics [T,3,3], got {k.shape}")
    return np.column_stack(
        [k[:, 0, 0], k[:, 1, 1], k[:, 0, 1], k[:, 0, 2], k[:, 1, 2]]
    )


def summarize_intrinsic_variation(intrinsics: np.ndarray) -> dict[str, Any]:
    """Describe within-clip calibration drift without consulting ground truth."""

    values = intrinsic_components(intrinsics)
    names = ("fx", "fy", "skew", "cx", "cy")
    repeated_first = np.repeat(values[:1], len(values), axis=0)
    summary: dict[str, Any] = {
        "frames": int(len(values)),
        "exactly_constant": bool(np.array_equal(values, repeated_first)),
        "allclose_1e-6": bool(
            np.allclose(values, repeated_first, rtol=1e-6, atol=1e-6)
        ),
    }
    for column, name in enumerate(names):
        current = values[:, column]
        median = float(np.median(current))
        span = float(np.ptp(current))
        summary[f"{name}_median_px"] = median
        summary[f"{name}_std_px"] = float(np.std(current))
        summary[f"{name}_range_px"] = span
        summary[f"{name}_range_relative"] = span / max(abs(median), 1.0e-12)
        if len(current) > 1:
            summary[f"{name}_adjacent_abs_change_mean_px"] = float(
                np.mean(np.abs(np.diff(current)))
            )
        else:
            summary[f"{name}_adjacent_abs_change_mean_px"] = 0.0
    return summary


def _sample_pixel_indices(height: int, width: int, max_points: int) -> np.ndarray:
    if max_points < 8:
        raise ValueError("max_points_per_frame must be at least 8")
    if max_points >= height * width:
        return np.arange(height * width, dtype=np.int64)
    rows = max(2, int(round(math.sqrt(max_points * height / width))))
    columns = max(2, int(max_points // rows))
    while rows * columns > max_points:
        if columns >= rows:
            columns -= 1
        else:
            rows -= 1
    yy = np.unique(np.rint(np.linspace(0, height - 1, rows)).astype(np.int64))
    xx = np.unique(np.rint(np.linspace(0, width - 1, columns)).astype(np.int64))
    return (yy[:, None] * width + xx[None, :]).reshape(-1)


def _prepare_observations(
    ray_directions: np.ndarray,
    max_points_per_frame: int,
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int, int]]:
    directions = np.asarray(ray_directions, dtype=np.float64)
    if directions.ndim != 4 or directions.shape[1] != 3:
        raise ValueError(f"Expected ray directions [T,3,H,W], got {directions.shape}")
    _, _, height, width = directions.shape
    indices = _sample_pixel_indices(height, width, max_points_per_frame)
    yy, xx = np.divmod(indices, width)
    base_pixels = np.column_stack(
        [xx.astype(np.float64) + 0.5, yy.astype(np.float64) + 0.5, np.ones(len(indices))]
    )
    world_by_frame: list[np.ndarray] = []
    pixels_by_frame: list[np.ndarray] = []
    for frame in directions:
        world = frame.reshape(3, -1).T[indices]
        norm = np.linalg.norm(world, axis=1)
        valid = np.isfinite(world).all(axis=1) & np.isfinite(norm) & (norm > 1.0e-10)
        if int(valid.sum()) < 8:
            raise ValueError(f"Only {int(valid.sum())} valid ray directions in a frame")
        world_by_frame.append(world[valid] / norm[valid, None])
        pixels_by_frame.append(base_pixels[valid])
    return world_by_frame, pixels_by_frame, (height, width)


def _parameters_to_intrinsic(parameters: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    height, width = image_hw
    log_fx, log_fy, skew, cx, cy = np.asarray(parameters, dtype=np.float64)
    return np.asarray(
        [
            [math.exp(log_fx) * width, skew * width, cx * width],
            [0.0, math.exp(log_fy) * height, cy * height],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _intrinsic_to_parameters(intrinsic: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    height, width = image_hw
    k = np.asarray(intrinsic, dtype=np.float64)
    if not np.isfinite(k).all() or k[0, 0] <= 0.0 or k[1, 1] <= 0.0:
        raise ValueError(f"Invalid intrinsic initialization:\n{k}")
    return np.asarray(
        [
            math.log(k[0, 0] / width),
            math.log(k[1, 1] / height),
            k[0, 1] / width,
            k[0, 2] / width,
            k[1, 2] / height,
        ],
        dtype=np.float64,
    )


def _robust_rotation(camera: np.ndarray, world: np.ndarray, huber_delta: float) -> np.ndarray:
    """Return an SO(3) matrix mapping column camera rays to world rays."""

    weights = np.ones(len(camera), dtype=np.float64)
    rotation = np.eye(3, dtype=np.float64)
    for _ in range(3):
        covariance = (camera * weights[:, None]).T @ world
        u, _, vt = np.linalg.svd(covariance)
        row_transform = u @ vt
        if np.linalg.det(row_transform) < 0.0:
            u[:, -1] *= -1.0
            row_transform = u @ vt
        rotation = row_transform.T
        residual = np.linalg.norm(camera @ rotation.T - world, axis=1)
        weights = np.minimum(1.0, huber_delta / np.maximum(residual, 1.0e-12))
    return rotation


def _fit_rotations_and_residuals(
    intrinsic: np.ndarray,
    world_by_frame: list[np.ndarray],
    pixels_by_frame: list[np.ndarray],
    huber_delta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inverse_k = np.linalg.inv(intrinsic)
    rotations, residuals, angles = [], [], []
    for world, pixels in zip(world_by_frame, pixels_by_frame):
        camera = pixels @ inverse_k.T
        camera /= np.maximum(np.linalg.norm(camera, axis=1, keepdims=True), 1.0e-12)
        rotation = _robust_rotation(camera, world, huber_delta)
        predicted = camera @ rotation.T
        residuals.append(predicted - world)
        angles.append(
            np.degrees(
                np.arccos(np.clip(np.sum(predicted * world, axis=1), -1.0, 1.0))
            )
        )
        rotations.append(rotation)
    return np.stack(rotations), np.concatenate(residuals), np.concatenate(angles)


def _recover_centers(ray_directions: np.ndarray, ray_moments: np.ndarray) -> np.ndarray:
    directions = np.asarray(ray_directions, dtype=np.float64)
    moments = np.asarray(ray_moments, dtype=np.float64)
    if directions.shape != moments.shape or directions.ndim != 4 or directions.shape[1] != 3:
        raise ValueError(
            f"Expected matching direction/moment [T,3,H,W], got {directions.shape}/{moments.shape}"
        )
    centers = []
    identity = np.eye(3, dtype=np.float64)
    for direction_frame, moment_frame in zip(directions, moments):
        direction = direction_frame.reshape(3, -1).T
        moment = moment_frame.reshape(3, -1).T
        norm = np.linalg.norm(direction, axis=1)
        valid = (
            np.isfinite(direction).all(axis=1)
            & np.isfinite(moment).all(axis=1)
            & np.isfinite(norm)
            & (norm > 1.0e-10)
        )
        if int(valid.sum()) < 3:
            raise ValueError(f"Only {int(valid.sum())} valid Pluecker rays in a frame")
        direction = direction[valid] / norm[valid, None]
        moment = moment[valid]
        points = np.cross(direction, moment)
        projectors = identity[None] - direction[:, :, None] * direction[:, None, :]
        lhs = np.sum(projectors, axis=0)
        rhs = np.einsum("nij,nj->i", projectors, points)
        centers.append(np.linalg.lstsq(lhs, rhs, rcond=None)[0])
    return np.stack(centers)


def recover_cameras_shared_intrinsics(
    ray_directions: np.ndarray,
    ray_moments: np.ndarray,
    initial_intrinsics: np.ndarray,
    *,
    max_points_per_frame: int = 600,
    max_nfev: int = 80,
    robust_f_scale: float = 0.035,
    rotation_huber_delta: float = 0.05,
) -> SharedIntrinsicsResult:
    """Recover clip cameras with one jointly fitted predicted intrinsic matrix.

    ``ray_moments`` must be in the desired output translation units.  For the
    model's raw predictions this means passing the saved normalized moment
    channels directly, yielding model-native camera centers without using the
    dataset-derived visualization scale.
    """

    per_frame_k = np.asarray(initial_intrinsics, dtype=np.float64)
    world_by_frame, pixels_by_frame, image_hw = _prepare_observations(
        ray_directions, max_points_per_frame
    )
    if per_frame_k.shape != (len(world_by_frame), 3, 3):
        raise ValueError(
            f"Expected initial intrinsics {(len(world_by_frame), 3, 3)}, got {per_frame_k.shape}"
        )
    components = intrinsic_components(per_frame_k)
    finite = np.isfinite(components).all(axis=1) & (components[:, 0] > 0.0) & (
        components[:, 1] > 0.0
    )
    if not np.any(finite):
        raise ValueError("No finite positive predicted intrinsic initialization")
    median = np.median(components[finite], axis=0)
    initial_shared = np.asarray(
        [[median[0], median[2], median[3]], [0.0, median[1], median[4]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    initial_parameters = _intrinsic_to_parameters(initial_shared, image_hw)
    lower = np.asarray(
        [math.log(0.04), math.log(0.04), -1.0, -1.0, -1.0], dtype=np.float64
    )
    upper = np.asarray(
        [math.log(10.0), math.log(10.0), 1.0, 2.0, 2.0], dtype=np.float64
    )
    initial_parameters = np.clip(initial_parameters, lower + 1.0e-9, upper - 1.0e-9)

    _, initial_residuals, initial_angles = _fit_rotations_and_residuals(
        initial_shared,
        world_by_frame,
        pixels_by_frame,
        rotation_huber_delta,
    )

    def residual_function(parameters: np.ndarray) -> np.ndarray:
        intrinsic = _parameters_to_intrinsic(parameters, image_hw)
        _, residuals, _ = _fit_rotations_and_residuals(
            intrinsic,
            world_by_frame,
            pixels_by_frame,
            rotation_huber_delta,
        )
        return residuals.reshape(-1)

    optimized = least_squares(
        residual_function,
        initial_parameters,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=robust_f_scale,
        x_scale="jac",
        max_nfev=max_nfev,
    )
    candidate = _parameters_to_intrinsic(optimized.x, image_hw)
    rotations, final_residuals, final_angles = _fit_rotations_and_residuals(
        candidate,
        world_by_frame,
        pixels_by_frame,
        rotation_huber_delta,
    )
    initial_robust_cost = float(
        np.sum(2.0 * robust_f_scale**2 * (np.sqrt(1.0 + (initial_residuals / robust_f_scale) ** 2) - 1.0))
    )
    final_robust_cost = float(
        np.sum(2.0 * robust_f_scale**2 * (np.sqrt(1.0 + (final_residuals / robust_f_scale) ** 2) - 1.0))
    )
    if not np.isfinite(final_robust_cost) or final_robust_cost > initial_robust_cost:
        candidate = initial_shared
        rotations, final_residuals, final_angles = _fit_rotations_and_residuals(
            candidate,
            world_by_frame,
            pixels_by_frame,
            rotation_huber_delta,
        )
        accepted_message = f"optimizer result rejected; using median initialization ({optimized.message})"
        accepted_success = False
    else:
        accepted_message = str(optimized.message)
        accepted_success = bool(optimized.success)

    centers = _recover_centers(ray_directions, ray_moments)
    extrinsics = np.repeat(np.eye(4, dtype=np.float64)[None], len(rotations), axis=0)
    extrinsics[:, :3, :3] = rotations
    extrinsics[:, :3, 3] = centers
    intrinsics = np.repeat(candidate[None], len(rotations), axis=0)
    return SharedIntrinsicsResult(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        shared_intrinsic=candidate,
        initial_shared_intrinsic=initial_shared,
        optimizer_success=accepted_success,
        optimizer_message=accepted_message,
        optimizer_nfev=int(optimized.nfev),
        initial_ray_angle_mean_deg=float(np.mean(initial_angles)),
        initial_ray_angle_median_deg=float(np.median(initial_angles)),
        initial_ray_angle_p95_deg=float(np.percentile(initial_angles, 95)),
        final_ray_angle_mean_deg=float(np.mean(final_angles)),
        final_ray_angle_median_deg=float(np.median(final_angles)),
        final_ray_angle_p95_deg=float(np.percentile(final_angles, 95)),
        observations=int(len(final_angles)),
    )
