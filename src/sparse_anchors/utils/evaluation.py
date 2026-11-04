"""Ground-truth alignment and retrieval diagnostics, independent of pose selection."""

from __future__ import annotations

import numpy as np

from sparse_anchors.data.sessions import SessionData
from sparse_anchors.utils.geometry import estimate_similarity_transform


def estimate_gt_from_reference_sfm(
    reference_data: SessionData,
    reference_gt_poses: dict[str, np.ndarray],
) -> np.ndarray:
    """Estimate SFM-to-GT Sim(3) from reference camera centers."""
    reference_sfm_poses = {frame.image_name: frame.c2w for frame in reference_data.frames}
    names = sorted(set(reference_sfm_poses) & set(reference_gt_poses))

    sfm_centers = np.stack([reference_sfm_poses[name][:3, 3] for name in names])
    gt_centers = np.stack([reference_gt_poses[name][:3, 3] for name in names])

    return estimate_similarity_transform(sfm_centers, gt_centers)
