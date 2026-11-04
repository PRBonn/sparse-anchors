"""Records shared by localization algorithms and experiment scripts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class PoseEstimate:
    c2w: np.ndarray
    w2c: np.ndarray
    inlier_indices: np.ndarray
    reprojection_error_px: float


@dataclass(frozen=True)
class PnPConfig:
    reprojection_error_px: float = 8.0
    confidence: float = 0.999
    iterations: int = 100
    min_points: int = 4


@dataclass(frozen=True)
class GroupedImageDescriptors:
    """VPR descriptors in timestamp order and each group's input image order."""

    timestamps: tuple[str, ...]
    timestamp_descriptors: torch.Tensor  # [timestamps, channels]
    image_descriptors_by_timestamp: dict[str, torch.Tensor]  # [views, channels] per timestamp


@dataclass(frozen=True)
class FeatureMetricRefinement:
    """Result of optimizing an SE(3) pose update against feature loss."""

    initial_w2c: torch.Tensor
    final_w2c: torch.Tensor
    xi: torch.Tensor
    initial_loss: float
    final_loss: float
    best_iteration: int
    losses: tuple[float, ...]
