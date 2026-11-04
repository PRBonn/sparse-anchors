"""Fit and score query-SfM to reference-map transforms from oriented rig poses."""

from itertools import combinations

import numpy as np
from scipy.spatial.transform import Rotation


def fit_oriented_sim3(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray | None:
    """Fit a Sim(3) from paired c2w poses in query and reference frames."""
    weights = np.ones(len(source)) if weights is None else np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    # Each pose pair votes for the world-frame rotation: target_R = R @ source_R.
    relative_rotations = target[:, :3, :3] @ source[:, :3, :3].transpose(0, 2, 1)
    rotation = Rotation.from_matrix(relative_rotations).mean(weights=weights).as_matrix()

    # With rotation fixed, fit one scale and translation to the camera centers.
    rotated_source = source[:, :3, 3] @ rotation.T
    target_centers = target[:, :3, 3]
    source_mean = np.sum(weights[:, None] * rotated_source, axis=0)
    target_mean = np.sum(weights[:, None] * target_centers, axis=0)
    centered_source = rotated_source - source_mean
    centered_target = target_centers - target_mean
    variance = np.sum(weights * np.sum(centered_source**2, axis=1))
    if variance < 1e-10:
        return None

    scale = float(np.sum(weights * np.sum(centered_source * centered_target, axis=1)) / variance)
    if not np.isfinite(scale) or scale <= 0:
        return None

    transform = np.eye(4)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = target_mean - scale * source_mean
    return transform


def pose_residuals(
    transform: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-pose center distances and rotation errors in degrees.

    The Sim(3) maps query-local SfM poses to the normalized reference map.
    Translation errors therefore use normalized reference-map distance units.
    """
    # Camera centers transform with the full scale, rotation, and translation.
    source_centers = source[:, :3, 3]
    target_centers = target[:, :3, 3]
    predicted_centers = source_centers @ transform[:3, :3].T + transform[:3, 3]
    translation_errors = np.linalg.norm(predicted_centers - target_centers, axis=1)

    # Camera orientations stay rigid: remove scale from the Sim(3) rotation block.
    scale = np.linalg.norm(transform[:3, 0])
    rotation = transform[:3, :3] / scale
    predicted_rotations = rotation @ source[:, :3, :3]
    target_rotations = target[:, :3, :3]
    relative_rotations = target_rotations @ predicted_rotations.transpose(0, 2, 1)
    rotation_errors = np.rad2deg(Rotation.from_matrix(relative_rotations).magnitude())

    return translation_errors, rotation_errors


def _spatial_cell_indices(source: np.ndarray, trajectory: np.ndarray, fraction: float) -> np.ndarray:
    """Number the query-path cells occupied by the source anchors."""
    trajectory_centers = trajectory[:, :3, 3]
    origin = trajectory_centers.min(axis=0)
    extent = np.ptp(trajectory_centers, axis=0)
    cell_width = fraction * max(float(np.linalg.norm(extent)), 1e-8)
    cells = np.floor((source[:, :3, 3] - origin) / cell_width).astype(int)
    _, cell_index = np.unique(cells, axis=0, return_inverse=True)
    return cell_index


def _cell_weights(cell_index: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Give every occupied query-path cell equal total weight."""
    counts = np.bincount(cell_index[mask], minlength=cell_index.max() + 1)
    return 1.0 / counts[cell_index[mask]]


def initial_consensus(
    source: np.ndarray,
    target: np.ndarray,
    trajectory: np.ndarray,
    translation_threshold: float = 0.15,
    rotation_threshold_deg: float = 15.0,
    cell_fraction: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Find an initial map transform and its inlier mask.

    source[i] is a query-local SfM rig pose; target[i] is its PnP pose in the
    normalized reference map. trajectory contains all query-local rig poses.
    The returned transform is a seed for the later weighted refit.
    """
    # Use the complete query path to define cells, so repeated nearby anchors
    # count as one spatial vote when ranking candidate transforms.
    target_centers = target[:, :3, 3]
    cell_index = _spatial_cell_indices(source, trajectory, cell_fraction)

    best_score = None
    best_transform = None
    best_inliers = None
    for i, j in combinations(range(len(source)), 2):
        # Nearly coincident PnP centers cannot constrain the pair's scale well.
        pair_distance = np.linalg.norm(target_centers[i] - target_centers[j])
        if pair_distance < 2 * translation_threshold:
            continue

        candidate = fit_oriented_sim3(source[[i, j]], target[[i, j]])
        if candidate is None:
            continue

        # Both position and orientation must agree with the same map transform.
        translation_errors, rotation_errors = pose_residuals(candidate, source, target)
        inliers = (translation_errors <= translation_threshold) & (rotation_errors <= rotation_threshold_deg)
        num_inliers = np.count_nonzero(inliers)
        if num_inliers < 3:
            continue

        # Prefer broad coverage, then the number of anchors, then tighter fit.
        # Strict > keeps the first candidate when all three scores tie.
        num_cells = len(np.unique(cell_index[inliers]))
        median_error = float(np.median(translation_errors[inliers]))
        score = (num_cells, num_inliers, -median_error)
        if best_score is None or score > best_score:
            best_score = score
            best_transform = candidate
            best_inliers = inliers

    assert best_transform is not None
    return best_transform, best_inliers


def refine_consensus(
    source: np.ndarray,
    target: np.ndarray,
    trajectory: np.ndarray,
    transform: np.ndarray,
    mask: np.ndarray,
    translation_threshold: float = 0.15,
    rotation_threshold_deg: float = 15.0,
    cell_fraction: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Refit the seed transform from its spatially balanced inliers."""
    cell_index = _spatial_cell_indices(source, trajectory, cell_fraction)

    for _ in range(5):
        # Give each occupied cell equal total weight, regardless of how many
        # nearby query timestamps contributed PnP anchors to that cell.
        fitted = fit_oriented_sim3(source[mask], target[mask], _cell_weights(cell_index, mask))

        distance, angle = pose_residuals(fitted, source, target)
        candidate = (distance <= translation_threshold) & (angle <= rotation_threshold_deg)
        if candidate.sum() < 3:
            break

        transform = fitted
        if np.array_equal(mask, candidate):
            break
        mask = candidate

    # The returned mask must describe the returned transform, including when
    # the last refit stopped because the mask no longer changed.
    distance, angle = pose_residuals(transform, source, target)
    mask = (distance <= translation_threshold) & (angle <= rotation_threshold_deg)
    assert mask.sum() >= 3
    return transform, mask


def consensus_quality(
    source: np.ndarray,
    target: np.ndarray,
    trajectory: np.ndarray,
    transform: np.ndarray,
    mask: np.ndarray,
    cell_fraction: float = 0.05,
    stable_min_anchors: int = 6,
    stable_min_cells: int = 3,
    stable_span_fraction: float = 0.30,
    stable_jackknife_p90: float = 0.05,
) -> dict:
    """Measure spatial support and sensitivity to removing one query-path cell."""
    cell_index = _spatial_cell_indices(source, trajectory, cell_fraction)
    support = np.unique(cell_index[mask])
    source_centers = source[mask, :3, 3]
    trajectory_centers = trajectory[:, :3, 3]
    source_diameter = max(float(np.linalg.norm(np.ptp(source_centers, axis=0))), 1e-8)
    trajectory_diameter = max(float(np.linalg.norm(np.ptp(trajectory_centers, axis=0))), 1e-8)
    span = source_diameter / trajectory_diameter

    # Compare the mapped whole trajectory with a refit that omits each
    # supported cell in turn. The worst 90th-percentile shift is the score.
    predicted = trajectory_centers @ transform[:3, :3].T + transform[:3, 3]
    shifts = []
    for cell in support:
        keep = mask & (cell_index != cell)
        if keep.sum() < 3:
            continue
        alternate = fit_oriented_sim3(source[keep], target[keep], _cell_weights(cell_index, keep))
        if alternate is None:
            continue
        alternate_centers = trajectory_centers @ alternate[:3, :3].T + alternate[:3, 3]
        shifts.append(float(np.quantile(np.linalg.norm(alternate_centers - predicted, axis=1), 0.9)))

    jackknife = max(shifts) if len(shifts) >= 3 else None
    stable = (
        mask.sum() >= stable_min_anchors
        and len(support) >= stable_min_cells
        and span >= stable_span_fraction
        and jackknife is not None
        and jackknife <= stable_jackknife_p90
    )
    distance, angles = pose_residuals(transform, source, target)
    return {
        "stable": bool(stable),
        "inliers": int(mask.sum()),
        "independent_cells": len(support),
        "span_fraction": float(span),
        "jackknife_p90_max": jackknife,
        "translation_residual_max": float(distance[mask].max()),
        "rotation_residual_max_deg": float(angles[mask].max()),
    }
