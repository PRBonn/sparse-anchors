from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from sparse_anchors.data.colmap import Frame
from sparse_anchors.utils.geometry import pose_row_to_matrix


def frame_timestamp(frame: Frame) -> str:
    return Path(frame.image_name).stem


def frame_camera_name(frame: Frame) -> str:
    return Path(frame.image_name).parent.name


def timestamp_sort_key(timestamp: str) -> float:
    return float(timestamp)


def group_frames_by_timestamp(frames: Sequence[Frame]) -> dict[str, tuple[Frame, ...]]:
    groups: dict[str, list[Frame]] = defaultdict(list)

    for frame in frames:
        groups[frame_timestamp(frame)].append(frame)

    return {
        timestamp: tuple(sorted(timestamp_frames, key=frame_camera_name))
        for timestamp, timestamp_frames in sorted(groups.items(), key=lambda item: timestamp_sort_key(item[0]))
    }


class SessionData:
    """Frames indexed by image name and timestamp, in COLMAP order."""

    def __init__(self, frames: tuple[Frame, ...]) -> None:
        self.frames = frames
        self.frames_by_name = {frame.image_name: frame for frame in frames}
        self.frames_by_timestamp = group_frames_by_timestamp(frames)


def load_gt_poses(
    frames_csv: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load per-image GT poses and one XY position per timestamp."""
    poses_by_image = {}
    xy_by_timestamp = {}

    with frames_csv.open(newline="") as file:
        for row in csv.DictReader(file):
            image_name = image_name_from_frames_csv(row["image_path"])

            c2w = pose_row_to_matrix(row)

            poses_by_image[image_name] = c2w
            xy_by_timestamp.setdefault(
                row["timestamp"],
                c2w[:2, 3].astype(np.float32),
            )

    return poses_by_image, xy_by_timestamp


def image_name_from_frames_csv(image_path: str) -> str:
    path = Path(image_path)
    if path.parts[0] == "images":
        return str(Path(*path.parts[1:]))
    return str(path)
