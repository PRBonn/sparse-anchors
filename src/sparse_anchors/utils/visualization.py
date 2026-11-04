"""PCA colors, image comparisons, and diagnostic plots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class PcaProjection:
    """A fixed three-component projection for diagnostic latent colors."""

    mean: torch.Tensor  # [D]
    components: torch.Tensor  # [D, 3]
    low: torch.Tensor  # [3]
    high: torch.Tensor  # [3]


@torch.no_grad()
def fit_pca_projection(values: torch.Tensor) -> PcaProjection:
    values = values.detach().float()
    mean = values.mean(dim=0)
    _, _, right_singular_vectors = torch.linalg.svd(values - mean, full_matrices=False)
    components = right_singular_vectors[:3].transpose(0, 1)
    projected = (values - mean) @ components
    low = torch.quantile(projected, 0.01, dim=0)
    high = torch.quantile(projected, 0.99, dim=0)
    return PcaProjection(mean=mean, components=components, low=low, high=high)


@torch.no_grad()
def pca_rgb(values: torch.Tensor, projection: PcaProjection) -> torch.Tensor:
    projected = (values - projection.mean) @ projection.components
    denominator = (projection.high - projection.low).clamp_min(torch.finfo(projected.dtype).eps)
    return ((projected - projection.low) / denominator).clamp(0.0, 1.0)


def diagonal_rgb_composite(
    render_rgb: np.ndarray | torch.Tensor,
    real_rgb: np.ndarray | torch.Tensor,
    separator_width: int = 2,
    separator_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Combine same-size RGB images: render above the diagonal, real image below."""
    render_rgb = _rgb_numpy(render_rgb)
    real_rgb = _rgb_numpy(real_rgb)
    height, width = render_rgb.shape[:2]
    yy, xx = np.indices((height, width))
    diagonal_y = xx * (height - 1) / max(width - 1, 1)
    render_mask = yy <= diagonal_y
    composite = np.where(render_mask[..., None], render_rgb, real_rgb)

    if separator_width > 0:
        separator = np.abs(yy - diagonal_y) <= separator_width / 2
        composite[separator] = np.asarray(separator_color, dtype=np.float32)

    return np.clip(composite, 0.0, 1.0).astype(np.float32)


def plot_vpr_matches_map(
    query_xy_by_timestamp: Mapping[str, np.ndarray],
    reference_xy_by_timestamp: Mapping[str, np.ndarray],
    retrievals: Sequence[Mapping],
    correct_distance_m: float = 5.0,
    floorplan: np.ndarray | None = None,
    floorplan_extent: tuple[float, float, float, float] | None = None,
    show: bool = True,
):
    """Show query/reference trajectories and VPR links colored by GT XY distance."""
    fig, ax = _map_figure(floorplan, floorplan_extent)
    _plot_pose_tracks(ax, query_xy_by_timestamp, reference_xy_by_timestamp)

    correct_count = 0
    plotted_count = 0
    for row in retrievals:
        query_timestamp = str(row["query_timestamp"])
        reference_timestamp = str(row["reference_timestamp"])
        query_xy = _xy(query_xy_by_timestamp[query_timestamp])
        reference_xy = _xy(reference_xy_by_timestamp[reference_timestamp])
        distance = float(np.linalg.norm(query_xy - reference_xy))
        is_correct = distance <= correct_distance_m
        correct_count += int(is_correct)
        plotted_count += 1
        color = "tab:green" if is_correct else "tab:red"
        ax.plot(
            [query_xy[0], reference_xy[0]],
            [query_xy[1], reference_xy[1]],
            color=color,
            linewidth=0.9,
            alpha=0.45,
        )

    ax.set_title(f"VPR matches: {correct_count}/{plotted_count} within {correct_distance_m:g} m")
    _finish_map(ax, show)
    return fig, ax


def plot_image_matching_survivors_map(
    query_xy_by_timestamp: Mapping[str, np.ndarray],
    reference_xy_by_timestamp: Mapping[str, np.ndarray],
    match_rows: Sequence[Mapping],
    min_inliers: int = 1,
    floorplan: np.ndarray | None = None,
    floorplan_extent: tuple[float, float, float, float] | None = None,
    show: bool = True,
):
    """Show which retrieved poses survive image matching thresholds."""
    fig, ax = _map_figure(floorplan, floorplan_extent)
    _plot_pose_tracks(ax, query_xy_by_timestamp, reference_xy_by_timestamp)

    accepted = []
    for row in match_rows:
        if int(row["num_inliers"]) >= min_inliers:
            accepted.append(row)

    for row in accepted:
        query_timestamp = str(row["query_timestamp"])
        reference_timestamp = str(row["reference_timestamp"])
        query_xy = _xy(query_xy_by_timestamp[query_timestamp])
        reference_xy = _xy(reference_xy_by_timestamp[reference_timestamp])
        num_inliers = int(row["num_inliers"])
        ax.plot(
            [query_xy[0], reference_xy[0]],
            [query_xy[1], reference_xy[1]],
            color="tab:purple",
            linewidth=max(0.8, min(3.0, 0.01 * num_inliers)),
            alpha=0.45,
        )
        ax.scatter(query_xy[0], query_xy[1], s=24, color="tab:orange", zorder=5)

    ax.set_title(f"Image matching survivors: {len(accepted)}/{len(match_rows)} (inliers>={min_inliers})")
    _finish_map(ax, show)
    return fig, ax


def plot_pnp_estimates_map(
    query_xy_by_timestamp: Mapping[str, np.ndarray],
    reference_xy_by_timestamp: Mapping[str, np.ndarray],
    estimates: Sequence[Mapping],
    floorplan: np.ndarray | None = None,
    floorplan_extent: tuple[float, float, float, float] | None = None,
    show: bool = True,
):
    """Show GT query/reference trajectories together with PnP estimated query poses."""
    fig, ax = _map_figure(floorplan, floorplan_extent)
    _plot_pose_tracks(ax, query_xy_by_timestamp, reference_xy_by_timestamp)

    estimated_xy = []
    for row in estimates:
        xy = np.asarray(row["c2w"], dtype=np.float64)[:2, 3].astype(np.float32)
        estimated_xy.append(xy)
        gt_xy = _xy(query_xy_by_timestamp[str(row["query_timestamp"])])
        reference_xy = _xy(reference_xy_by_timestamp[str(row["reference_timestamp"])])
        ax.plot([gt_xy[0], xy[0]], [gt_xy[1], xy[1]], color="0.45", linewidth=0.8, alpha=0.5)
        ax.plot([reference_xy[0], xy[0]], [reference_xy[1], xy[1]], color="tab:purple", linewidth=0.8, alpha=0.3)

    if estimated_xy:
        estimated_xy = np.stack(estimated_xy)
        ax.scatter(
            estimated_xy[:, 0],
            estimated_xy[:, 1],
            s=30,
            color="tab:red",
            marker="x",
            label="PnP estimate",
            zorder=6,
        )

    ax.set_title(f"PnP estimates: {len(estimated_xy)}/{len(estimates)}")
    _finish_map(ax, show)
    return fig, ax


def _map_figure(
    floorplan: np.ndarray | None,
    floorplan_extent: tuple[float, float, float, float] | None,
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    if floorplan is not None:
        # PNG row 0 is the top of the image; the floorplan map frame starts at the bottom-left pixel.
        ax.imshow(floorplan, extent=floorplan_extent, origin="upper", alpha=0.45)
    return fig, ax


def _plot_pose_tracks(
    ax,
    query_xy_by_timestamp: Mapping[str, np.ndarray],
    reference_xy_by_timestamp: Mapping[str, np.ndarray],
) -> None:
    query_xy = _ordered_xy(query_xy_by_timestamp)
    reference_xy = _ordered_xy(reference_xy_by_timestamp)

    if len(reference_xy):
        ax.plot(reference_xy[:, 0], reference_xy[:, 1], color="tab:blue", linewidth=1.4, label="reference GT")
        ax.scatter(reference_xy[:, 0], reference_xy[:, 1], s=10, color="tab:blue", alpha=0.5)
    if len(query_xy):
        ax.plot(query_xy[:, 0], query_xy[:, 1], color="tab:orange", linewidth=1.4, label="query GT")
        ax.scatter(query_xy[:, 0], query_xy[:, 1], s=10, color="tab:orange", alpha=0.5)


def _ordered_xy(xy_by_timestamp: Mapping[str, np.ndarray]) -> np.ndarray:
    if not xy_by_timestamp:
        return np.empty((0, 2), dtype=np.float32)
    return np.stack([_xy(xy_by_timestamp[timestamp]) for timestamp in sorted(xy_by_timestamp, key=float)])


def _xy(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)[:2]


def _rgb_numpy(image: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    return np.asarray(image, dtype=np.float32)


def _finish_map(ax, show: bool) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    if show:
        import matplotlib.pyplot as plt

        plt.show()


def resize_rgb(image: np.ndarray | torch.Tensor, output_size: tuple[int, int]) -> np.ndarray:
    image = torch.as_tensor(image).detach().cpu().numpy()
    height, width = output_size
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.float32)


def side_by_side(
    before: np.ndarray | torch.Tensor,
    after: np.ndarray | torch.Tensor,
    separator_width: int = 8,
) -> np.ndarray:
    before = torch.as_tensor(before).detach().cpu().numpy()
    after = torch.as_tensor(after).detach().cpu().numpy()
    separator = np.ones((before.shape[0], separator_width, 3), dtype=np.float32)
    return np.concatenate([before, separator, after], axis=1)


def make_contact_sheet(
    images: list[np.ndarray],
    columns: int,
) -> np.ndarray:
    height, width = images[0].shape[:2]
    rows = int(np.ceil(len(images) / columns))
    sheet = np.zeros((rows * height, columns * width, 3), dtype=np.float32)
    for index, image in enumerate(images):
        row = index // columns
        column = index % columns
        sheet[row * height : (row + 1) * height, column * width : (column + 1) * width] = image
    return sheet
