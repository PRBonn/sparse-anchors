"""Fixed-scale pose graph optimization for a sequence of rigid camera poses."""

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation


def sigmas(translation: float, rotation_deg: float) -> np.ndarray:
    """Return translation and rotation standard deviations in solver order."""
    return np.array([translation] * 3 + [np.deg2rad(rotation_deg)] * 3)


def optimize_pose_graph(
    base: np.ndarray,
    anchor_indices: np.ndarray,
    anchor_poses: np.ndarray,
    relative_sigmas: np.ndarray,
    anchor_sigmas: np.ndarray,
    prior_sigmas: np.ndarray,
    initial_poses: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Adjust c2w poses toward PnP anchors while preserving sequence motion.

    Each sigma has translation xyz followed by rotation xyz in radians. The
    poses are already in the reference map frame; only SE(3) corrections vary.
    Optional initial poses change the starting point, not the graph measurements.
    """
    n, m = len(base), len(anchor_indices)
    base_rotations = base[:, :3, :3]
    base_centers = base[:, :3, 3]
    base_inverse_rotations = base_rotations.transpose(0, 2, 1)
    measured_t = np.einsum("nij,nj->ni", base_inverse_rotations[:-1], np.diff(base_centers, axis=0))
    measured_r = base_inverse_rotations[:-1] @ base_rotations[1:]
    anchor_inverse_rotations = anchor_poses[:, :3, :3].transpose(0, 2, 1)

    def corrected_poses(x):
        corrections = x.reshape(n, 6)
        rotations = Rotation.from_rotvec(corrections[:, 3:]).as_matrix() @ base_rotations
        centers = base_centers + corrections[:, :3]
        return rotations, centers

    def objective(x):
        rotations, centers = corrected_poses(x)
        inverse_rotations = rotations.transpose(0, 2, 1)

        relative_t = np.einsum("nij,nj->ni", inverse_rotations[:-1], np.diff(centers, axis=0)) - measured_t
        relative_r = Rotation.from_matrix(
            measured_r.transpose(0, 2, 1) @ inverse_rotations[:-1] @ rotations[1:]
        ).as_rotvec()
        anchor_t = centers[anchor_indices] - anchor_poses[:, :3, 3]
        anchor_r = Rotation.from_matrix(anchor_inverse_rotations @ rotations[anchor_indices]).as_rotvec()

        return np.concatenate(
            [
                (np.c_[relative_t, relative_r] / relative_sigmas).ravel(),
                (np.c_[anchor_t, anchor_r] / anchor_sigmas).ravel(),
                (x.reshape(n, 6) / prior_sigmas).ravel(),
            ]
        )

    # A relative edge touches two poses; each anchor and prior touches one.
    sparsity = lil_matrix((6 * (n - 1 + m + n), 6 * n), dtype=int)
    for i in range(n - 1):
        sparsity[6 * i : 6 * i + 6, 6 * i : 6 * i + 12] = 1
    anchor_offset = 6 * (n - 1)
    for j, i in enumerate(anchor_indices):
        sparsity[anchor_offset + 6 * j : anchor_offset + 6 * j + 6, 6 * i : 6 * i + 6] = 1
    prior_offset = anchor_offset + 6 * m
    for i in range(n):
        sparsity[prior_offset + 6 * i : prior_offset + 6 * i + 6, 6 * i : 6 * i + 6] = 1

    if initial_poses is None:
        initial = np.zeros(6 * n)
    else:
        initial = np.c_[
            initial_poses[:, :3, 3] - base_centers,
            Rotation.from_matrix(initial_poses[:, :3, :3] @ base_inverse_rotations).as_rotvec(),
        ].ravel()

    initial_squared_residual = float(np.sum(objective(initial) ** 2))
    fit = least_squares(
        objective,
        initial,
        jac_sparsity=sparsity.tocsr(),
        loss="soft_l1",
        max_nfev=100,
        ftol=1e-7,
        xtol=1e-7,
        gtol=1e-7,
    )
    rotations, centers = corrected_poses(fit.x)
    optimized = base.copy()
    optimized[:, :3, :3] = rotations
    optimized[:, :3, 3] = centers
    return optimized, {
        "success": bool(fit.success),
        "message": fit.message,
        "nfev": fit.nfev,
        "cost": float(fit.cost),
        "initial_squared_residual": initial_squared_residual,
        "max_center_correction": float(np.linalg.norm(centers - base_centers, axis=1).max()),
        "fixed_scale": True,
    }
