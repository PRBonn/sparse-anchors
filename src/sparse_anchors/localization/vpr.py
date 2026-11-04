from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from sparse_anchors.localization.types import GroupedImageDescriptors
from sparse_anchors.utils.image import load_rgb


def load_megaloc(device: torch.device) -> torch.nn.Module:
    return torch.hub.load("gmberton/MegaLoc", "get_trained_model").eval().to(device)


@torch.inference_mode()
def describe_image_paths(
    model: torch.nn.Module,
    image_paths: Sequence[Path],
    device: torch.device,
    batch_size: int = 8,
    image_size: int | None = None,
    normalize: bool = True,
    desc: str = "describing images",
    cache_path: Path | None = None,
    use_cache: bool = False,
) -> torch.Tensor:
    """Return one descriptor per input image, in input order.

    With use_cache=True, load the complete descriptor tensor from cache_path.
    Otherwise, describe every image and optionally save the complete result.
    """
    image_paths = list(image_paths)

    if use_cache:
        descriptors = torch.from_numpy(np.load(cache_path))
        return descriptors.to(device)

    descriptors = []

    for start in tqdm(range(0, len(image_paths), batch_size), desc=desc, unit="batch"):
        batch_paths = image_paths[start : start + batch_size]
        images = torch.stack(
            [torch.from_numpy(load_rgb(image_path)).permute(2, 0, 1) for image_path in batch_paths]
        ).to(device)

        if image_size is not None:
            images = F.interpolate(
                images,
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            )

        batch_descriptors = model(images)
        if normalize:
            batch_descriptors = F.normalize(batch_descriptors, dim=-1)

        descriptors.append(batch_descriptors.cpu())

    descriptors = torch.cat(descriptors, dim=0)

    # A provided cache path means this fresh result should be saved for reuse.
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, descriptors.numpy())

    return descriptors.to(device)


def describe_image_groups(
    model: torch.nn.Module,
    image_groups: dict[str, tuple[Path, ...]],
    device: torch.device,
    batch_size: int = 8,
    image_size: int | None = None,
    desc: str = "describing image groups",
    cache_path: Path | None = None,
    use_cache: bool = False,
) -> GroupedImageDescriptors:
    """Describe every image and pool descriptors by timestamp.

    The descriptor row at
    image_descriptors_by_timestamp[timestamp][i] corresponds to
    image_groups[timestamp][i].
    """
    timestamps = tuple(sorted(image_groups, key=float))
    image_paths = [image_path for timestamp in timestamps for image_path in image_groups[timestamp]]

    # The cache, when used, contains descriptors for this full ordered image list.
    all_image_descriptors = describe_image_paths(
        model,
        image_paths,
        device=device,
        batch_size=batch_size,
        image_size=image_size,
        desc=desc,
        cache_path=cache_path,
        use_cache=use_cache,
    )

    timestamp_descriptors = []
    image_descriptors_by_timestamp = {}
    start = 0

    for timestamp in timestamps:
        end = start + len(image_groups[timestamp])

        # Rows remain aligned with the paths in image_groups[timestamp].
        group_image_descriptors = all_image_descriptors[start:end]
        image_descriptors_by_timestamp[timestamp] = group_image_descriptors

        # One normalized descriptor represents this multi-camera timestamp.
        timestamp_descriptors.append(group_image_descriptors.mean(dim=0))
        start = end

    return GroupedImageDescriptors(
        timestamps=timestamps,
        timestamp_descriptors=F.normalize(torch.stack(timestamp_descriptors), dim=-1),
        image_descriptors_by_timestamp=image_descriptors_by_timestamp,
    )


def retrieve(
    query_timestamps: Sequence[str],
    query_descriptors: torch.Tensor,
    reference_timestamps: Sequence[str],
    reference_descriptors: torch.Tensor,
    top_k: int = 1,
) -> list[dict]:
    scores = image_similarity_matrix(query_descriptors, reference_descriptors)
    top_indices = np.argsort(-scores, axis=1)[:, :top_k]

    return [
        {
            "query_timestamp": query_timestamps[query_index],
            "reference_timestamp": reference_timestamps[int(reference_index)],
            "rank": rank,
            "score": float(scores[query_index, reference_index]),
        }
        for query_index, indices_row in enumerate(top_indices)
        for rank, reference_index in enumerate(indices_row, start=1)
    ]


def far_margin_scores(
    query_descriptors: torch.Tensor,
    reference_descriptors: torch.Tensor,
    reference_centers: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Compare each query's best match with references outside its map neighborhood."""
    scores = image_similarity_matrix(query_descriptors, reference_descriptors)
    margins = np.full(scores.shape[0], np.nan, dtype=np.float32)

    for q in range(scores.shape[0]):
        winner_idx = np.argmax(scores[q])
        winner_score = scores[q, winner_idx]

        distances = np.linalg.norm(
            reference_centers - reference_centers[winner_idx],
            axis=1,
        )

        far_mask = distances > radius

        if np.any(far_mask):
            best_far_score = np.max(scores[q, far_mask])
            margins[q] = winner_score - best_far_score

    return margins


def image_similarity_matrix(query_descriptors: torch.Tensor, reference_descriptors: torch.Tensor) -> np.ndarray:
    query_descriptors = F.normalize(query_descriptors.detach(), dim=-1)
    reference_descriptors = F.normalize(reference_descriptors.detach(), dim=-1)
    return (query_descriptors @ reference_descriptors.T).float().cpu().numpy()


def top_diverse_image_pairs(
    query_descriptors: torch.Tensor,
    reference_descriptors: torch.Tensor,
    *,
    top_k: int,
) -> list[tuple[int, int, float]]:
    """Return the top-scoring pairs from the best one-to-one assignment."""
    scores = image_similarity_matrix(query_descriptors, reference_descriptors)
    query_indices, reference_indices = linear_sum_assignment(-scores)
    assigned = [
        (int(query_index), int(reference_index), float(scores[query_index, reference_index]))
        for query_index, reference_index in zip(query_indices, reference_indices, strict=True)
    ]
    assigned.sort(key=lambda item: item[2], reverse=True)
    return assigned[:top_k]
