"""COLMAP reconstruction records and loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pycolmap import Reconstruction


@dataclass(frozen=True)
class Frame:
    """One registered image together with its calibration and pose."""

    image_id: int
    camera_id: int
    image_name: str
    image_path: Path
    width: int
    height: int
    K: np.ndarray
    w2c: np.ndarray
    c2w: np.ndarray


@dataclass(frozen=True)
class SparsePointCloud:
    positions: np.ndarray
    colors: np.ndarray


@dataclass(frozen=True)
class ColmapModel:
    frames: tuple[Frame, ...]
    points: SparsePointCloud


def load_colmap_model(sparse_dir: Path, images_dir: Path) -> ColmapModel:
    reconstruction = Reconstruction(str(sparse_dir))
    frames: list[Frame] = []

    for image_id, image in reconstruction.images.items():
        camera = reconstruction.cameras[image.camera_id]
        cam_from_world = image.cam_from_world()

        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :4] = cam_from_world.matrix()

        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :4] = cam_from_world.inverse().matrix()

        image_name = str(image.name)
        frames.append(
            Frame(
                image_id=int(image_id),
                camera_id=int(image.camera_id),
                image_name=image_name,
                image_path=Path(images_dir) / image_name,
                width=int(camera.width),
                height=int(camera.height),
                K=np.array(camera.calibration_matrix(), dtype=np.float64, copy=True),
                w2c=w2c,
                c2w=c2w,
            )
        )

    frames.sort(key=lambda frame: frame.image_name)
    point3ds = reconstruction.points3D.values()
    points = SparsePointCloud(
        positions=np.asarray([point.xyz for point in point3ds], dtype=np.float32).reshape(-1, 3),
        colors=np.asarray([point.color for point in point3ds], dtype=np.uint8).reshape(-1, 3),
    )
    return ColmapModel(frames=tuple(frames), points=points)
