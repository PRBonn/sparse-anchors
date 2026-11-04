"""Scene metadata for the splatting pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from sparse_anchors.data.colmap import ColmapModel, Frame, load_colmap_model
from sparse_anchors.splatting.features import FeatureManifest, load_feature_manifest
from sparse_anchors.utils.geometry import similarity_from_camera_centers, transform_c2w, transform_points


@dataclass(frozen=True)
class Scene:
    colmap: ColmapModel
    features: FeatureManifest
    train_frames: tuple[Frame, ...]
    eval_frames: tuple[Frame, ...]
    world_transform: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))


def load_scene(
    sparse_dir: Path,
    images_dir: Path,
    feature_manifest_path: Path,
    test_every: int = 8,
) -> Scene:

    colmap = load_colmap_model(sparse_dir, images_dir)
    features = load_feature_manifest(feature_manifest_path)

    eval_frames = colmap.frames[::test_every] if test_every else ()
    train_frames = colmap.frames
    if test_every > 1:
        train_frames = tuple(frame for index, frame in enumerate(colmap.frames) if index % test_every != 0)
    return Scene(colmap=colmap, features=features, train_frames=train_frames, eval_frames=eval_frames)


def normalize_scene(scene: Scene) -> Scene:
    train_c2w = np.stack([frame.c2w for frame in scene.train_frames])
    transform = similarity_from_camera_centers(train_c2w)

    return transform_scene(scene, transform)


def transform_scene(scene: Scene, transform: np.ndarray) -> Scene:
    transform = np.asarray(transform, dtype=np.float64)
    frames = []

    for frame in scene.colmap.frames:
        c2w = transform_c2w(transform, frame.c2w)
        frames.append(
            replace(
                frame,
                c2w=c2w,
                w2c=np.linalg.inv(c2w),
            )
        )

    normalized_frames = tuple(frames)
    frames_by_id = {frame.image_id: frame for frame in normalized_frames}

    normalized_colmap = replace(
        scene.colmap,
        frames=normalized_frames,
        points=replace(
            scene.colmap.points,
            positions=transform_points(
                transform,
                scene.colmap.points.positions,
            ),
        ),
    )

    return Scene(
        colmap=normalized_colmap,
        features=scene.features,
        train_frames=tuple(frames_by_id[frame.image_id] for frame in scene.train_frames),
        eval_frames=tuple(frames_by_id[frame.image_id] for frame in scene.eval_frames),
        world_transform=transform @ scene.world_transform,
    )
