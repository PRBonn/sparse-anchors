from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import poselib
import torch

from sparse_anchors.data.colmap import Frame
from sparse_anchors.localization.types import PnPConfig, PoseEstimate
from sparse_anchors.splatting.gaussians import GaussianModel
from sparse_anchors.splatting.renderer import render
from sparse_anchors.utils.geometry import transform_points


@torch.no_grad()
def render_reference_depth(
    model: GaussianModel,
    reference_frame: Frame,
    sh_degree: int,
    packed: bool = False,
) -> np.ndarray:
    output = render(
        model,
        reference_frame,
        sh_degree=sh_degree,
        packed=packed,
        render_latents=False,
    )
    return output.depth.detach().cpu().numpy()


def lift_reference_matches(
    points_query: np.ndarray,
    points_reference: np.ndarray,
    reference_depth: np.ndarray,
    reference_K: np.ndarray,
    reference_c2w: np.ndarray,
    min_depth: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Lift depth-valid reference match pixels into reference-world 3D points.

    Each returned query pixel remains aligned with its corresponding world point.
    Correspondences whose reference rendered depth is invalid are removed.
    """
    points_query = np.asarray(points_query, dtype=np.float32).reshape(-1, 2)
    points_reference = np.asarray(points_reference, dtype=np.float32).reshape(-1, 2)

    depths, valid = sample_depth_bilinear(reference_depth, points_reference)
    valid &= np.isfinite(depths) & (depths > min_depth)

    query_points = points_query[valid]
    reference_points = points_reference[valid]

    # Backproject valid reference pixels in the reference camera frame,
    # then map them into the reference SFM/gsplat world frame.
    camera_points = backproject_pixels(
        reference_points,
        depths[valid],
        reference_K,
    )
    world_points = transform_points(reference_c2w, camera_points)

    return query_points, world_points


def estimate_pose_generalized_pnp(
    points_2d_by_camera: Mapping[str, np.ndarray],
    points_3d_by_camera: Mapping[str, np.ndarray],
    camera_from_rig_by_camera: Mapping[str, np.ndarray],
    K_by_camera: Mapping[str, np.ndarray],
    image_size_by_camera: Mapping[str, tuple[int, int]],
    config: PnPConfig,
) -> PoseEstimate | None:
    """Estimate one rig pose from per-camera 2D-3D correspondences.

    PoseLib uses camera_from_rig fixed extrinsics and returns rig_from_world.
    The PoseEstimate follows the existing convention: w2c is rig_from_world and
    c2w is world_from_rig.
    """
    camera_names = [
        camera_name for camera_name in sorted(points_2d_by_camera) if len(points_2d_by_camera[camera_name]) > 0
    ]
    if not camera_names:
        return None

    total_points = sum(len(points_2d_by_camera[camera_name]) for camera_name in camera_names)
    if total_points < max(4, config.min_points):
        return None

    points2d = []
    points3d = []
    camera_ext = []
    cameras = []
    offsets = []
    offset = 0
    for camera_name in camera_names:
        image_points = np.asarray(points_2d_by_camera[camera_name], dtype=np.float64).reshape(-1, 2)
        world_points = np.asarray(points_3d_by_camera[camera_name], dtype=np.float64).reshape(-1, 3)

        points2d.append(image_points)
        points3d.append(world_points)
        camera_ext.append(_poselib_pose(camera_from_rig_by_camera[camera_name]))
        cameras.append(_poselib_pinhole_camera(K_by_camera[camera_name], image_size_by_camera[camera_name]))
        offsets.append(offset)
        offset += len(image_points)

    ransac_options = {
        "max_reproj_error": float(config.reprojection_error_px),
        "max_iterations": int(config.iterations),
        "success_prob": float(config.confidence),
    }
    try:
        rig_pose, info = poselib.estimate_generalized_absolute_pose(
            points2d,
            points3d,
            camera_ext,
            cameras,
            ransac_options,
            {},
        )
    except RuntimeError:
        return None

    inlier_indices = _generalized_inlier_indices(info.get("inliers"), offsets)
    if len(inlier_indices) < max(4, config.min_points):
        return None

    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = np.asarray(rig_pose.R, dtype=np.float64)
    w2c[:3, 3] = np.asarray(rig_pose.t, dtype=np.float64)
    c2w = np.linalg.inv(w2c)

    errors = _generalized_reprojection_errors(
        points2d,
        points3d,
        camera_ext,
        cameras,
        w2c,
        info.get("inliers"),
    )

    return PoseEstimate(
        c2w=c2w,
        w2c=w2c,
        inlier_indices=inlier_indices,
        reprojection_error_px=float(np.mean(errors)) if len(errors) else float("nan"),
    )


def _poselib_pose(transform: np.ndarray) -> poselib.CameraPose:
    transform = np.asarray(transform, dtype=np.float64)
    pose = poselib.CameraPose()
    pose.R = transform[:3, :3]
    pose.t = transform[:3, 3]
    return pose


def _poselib_pinhole_camera(K: np.ndarray, image_size: tuple[int, int]) -> poselib.Camera:
    K = np.asarray(K, dtype=np.float64)
    width, height = image_size
    return poselib.Camera(
        "PINHOLE",
        [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])],
        int(width),
        int(height),
    )


def _generalized_inlier_indices(inliers, offsets: list[int]) -> np.ndarray:
    if inliers is None:
        return np.empty((0,), dtype=np.int64)

    indices = []
    for camera_offset, camera_inliers in zip(offsets, inliers, strict=True):
        mask = np.asarray(camera_inliers, dtype=bool).reshape(-1)
        indices.extend((camera_offset + np.flatnonzero(mask)).tolist())
    return np.asarray(indices, dtype=np.int64)


def _generalized_reprojection_errors(
    points2d: list[np.ndarray],
    points3d: list[np.ndarray],
    camera_ext: list[poselib.CameraPose],
    cameras: list[poselib.Camera],
    rig_w2c: np.ndarray,
    inliers,
) -> np.ndarray:
    errors = []
    for camera_index, (image_points, world_points, camera_pose, camera) in enumerate(
        zip(points2d, points3d, camera_ext, cameras, strict=True)
    ):
        if inliers is None:
            mask = np.ones(len(image_points), dtype=bool)
        else:
            mask = np.asarray(inliers[camera_index], dtype=bool).reshape(-1)
        if not mask.any():
            continue

        camera_from_rig = np.eye(4, dtype=np.float64)
        camera_from_rig[:3, :3] = np.asarray(camera_pose.R, dtype=np.float64)
        camera_from_rig[:3, 3] = np.asarray(camera_pose.t, dtype=np.float64)
        camera_w2c = camera_from_rig @ rig_w2c

        camera_points = transform_points(camera_w2c, world_points[mask])
        projected = camera_points[:, :2] / camera_points[:, 2:3]
        projected[:, 0] = projected[:, 0] * camera.focal_x() + camera.principal_point()[0]
        projected[:, 1] = projected[:, 1] * camera.focal_y() + camera.principal_point()[1]
        errors.append(np.linalg.norm(projected - image_points[mask], axis=1))

    if not errors:
        return np.empty((0,), dtype=np.float64)
    return np.concatenate(errors)


def sample_depth_bilinear(
    depth: np.ndarray,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a depth image at floating-point pixel coordinates.

    Returned values and validity flags are aligned with points. Points outside
    the image are marked invalid; their sampled values should be ignored.
    """
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]

    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    height, width = depth.shape

    x = points[:, 0]
    y = points[:, 1]
    valid = (x >= 0.0) & (x <= width - 1) & (y >= 0.0) & (y <= height - 1)

    # Clip only for safe indexing. Invalid points are discarded by the caller.
    x0 = np.floor(np.clip(x, 0, width - 1)).astype(np.int64)
    y0 = np.floor(np.clip(y, 0, height - 1)).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)

    wx = x - x0
    wy = y - y0
    values = (
        depth[y0, x0] * (1.0 - wx) * (1.0 - wy)
        + depth[y0, x1] * wx * (1.0 - wy)
        + depth[y1, x0] * (1.0 - wx) * wy
        + depth[y1, x1] * wx * wy
    )
    return values.astype(np.float32), valid


def backproject_pixels(
    points: np.ndarray,
    depths: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Backproject image pixels with camera z-depth into camera-frame 3D points."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    depths = np.asarray(depths, dtype=np.float32).reshape(-1)

    K = np.asarray(K, dtype=np.float64)
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    x = (points[:, 0] - cx) * depths / fx
    y = (points[:, 1] - cy) * depths / fy
    return np.stack([x, y, depths], axis=1).astype(np.float32)
