from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def rotation_y(deg: float) -> np.ndarray:
    angle = np.deg2rad(deg)
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=float,
    )


def rotation_x_down(deg: float) -> np.ndarray:
    angle = np.deg2rad(deg)
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, s],
            [0.0, -s, c],
        ],
        dtype=float,
    )


def rotation_z(deg: float) -> np.ndarray:
    angle = np.deg2rad(deg)
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def view_rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    return rotation_y(yaw_deg) @ rotation_x_down(pitch_deg)


def pose_row_to_matrix(row: dict[str, str]) -> np.ndarray:
    T = np.eye(4, dtype=float)
    quat_xyzw = np.array(
        [
            float(row["qx"]),
            float(row["qy"]),
            float(row["qz"]),
            float(row["qw"]),
        ]
    )
    T[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = [float(row["tx"]), float(row["ty"]), float(row["tz"])]
    return T


def matrix_to_pose_fields(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    translation = T[:3, 3]
    quat_xyzw = Rotation.from_matrix(T[:3, :3]).as_quat()
    return translation, quat_xyzw


def camera_centers(c2w: np.ndarray) -> np.ndarray:
    return c2w[..., :3, 3].copy()


def similarity_from_camera_centers(c2w: np.ndarray) -> np.ndarray:
    centers = camera_centers(c2w)
    center = np.median(centers, axis=0)
    scale = 1.0 / np.median(np.linalg.norm(centers - center, axis=-1))

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] *= scale
    transform[:3, 3] = -scale * center
    return transform


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def transform_c2w(transform: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """Apply a source-world to destination-world Sim(3) to rigid camera poses.

    For transform=(s, R, t), centers become s*R*center+t and rotations become
    R*camera_rotation. Supports one pose or a batch; inputs are not modified.
    """
    transformed = transform @ c2w
    scale = np.linalg.norm(transform[:3, 0])
    transformed[..., :3, :3] /= scale
    return transformed


def estimate_similarity_transform(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> np.ndarray:
    """Estimate a similarity transform from source points to target points."""
    source_mean = source_points.mean(axis=0)
    target_mean = target_points.mean(axis=0)

    source_centered = source_points - source_mean
    target_centered = target_points - target_mean

    covariance = target_centered.T @ source_centered / len(source_points)
    u, singular_values, vt = np.linalg.svd(covariance)

    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))

    rotation = u @ correction @ vt
    scale = float(np.sum(singular_values * np.diag(correction)) / np.mean(np.sum(source_centered**2, axis=1)))
    translation = target_mean - scale * rotation @ source_mean

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation
    return transform


def skew_symmetric(vector: torch.Tensor) -> torch.Tensor:
    """Return the 3x3 skew matrix for a 3-vector."""
    zero = torch.zeros((), dtype=vector.dtype, device=vector.device)
    wx, wy, wz = vector.unbind()
    return torch.stack(
        [
            torch.stack([zero, -wz, wy]),
            torch.stack([wz, zero, -wx]),
            torch.stack([-wy, wx, zero]),
        ]
    )


def so3_exp_torch(omega: torch.Tensor) -> torch.Tensor:
    """Map a 3-vector rotation update to a 3x3 rotation matrix."""
    omega_hat = skew_symmetric(omega)
    omega_hat_sq = omega_hat @ omega_hat

    theta_sq = torch.dot(omega, omega)
    theta_safe = torch.sqrt(theta_sq.clamp_min(torch.finfo(omega.dtype).eps))
    small = theta_sq < 1e-8

    a = torch.where(
        small,
        1.0 - theta_sq / 6.0 + theta_sq * theta_sq / 120.0,
        torch.sin(theta_safe) / theta_safe,
    )
    b = torch.where(
        small,
        0.5 - theta_sq / 24.0 + theta_sq * theta_sq / 720.0,
        (1.0 - torch.cos(theta_safe)) / (theta_safe * theta_safe),
    )
    eye3 = torch.eye(3, dtype=omega.dtype, device=omega.device)
    return eye3 + a * omega_hat + b * omega_hat_sq


def se3_exp_torch(xi: torch.Tensor) -> torch.Tensor:
    """Map xi=[omega, translation] to a 4x4 SE(3) matrix."""
    rotation = so3_exp_torch(xi[:3])
    transform = torch.eye(4, dtype=xi.dtype, device=xi.device)
    transform[:3, :3] = rotation
    transform[:3, 3] = xi[3:]
    return transform


def compose_pose_update(
    xi: torch.Tensor,
    initial_w2c: torch.Tensor,
) -> torch.Tensor:
    """Left-compose a per-pose SE(3) update with the initial camera-to-world pose."""
    initial_c2w = torch.linalg.inv(initial_w2c)
    updated_c2w = se3_exp_torch(xi) @ initial_c2w
    return torch.linalg.inv(updated_c2w)
