#!/usr/bin/env python3
"""Prepare COLMAP rig configuration and per-camera image lists for a HILTI scene."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def quaternion_to_rotation(qx: float, qy: float, qz: float, qw: float) -> list[list[float]]:
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0:
        raise ValueError("Encountered a zero-norm pose quaternion")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ]


def transpose(a: list[list[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*a)]


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def matvec(a: list[list[float]], v: list[float]) -> list[float]:
    return [sum(a[i][k] * v[k] for k in range(3)) for i in range(3)]


def rotation_to_quaternion(m: list[list[float]]) -> list[float]:
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        s = math.sqrt(trace + 1) * 2
        qw, qx, qy, qz = 0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1 + m[0][0] - m[1][1] - m[2][2]) * 2
        qw, qx, qy, qz = (m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1 + m[1][1] - m[0][0] - m[2][2]) * 2
        qw, qx, qy, qz = (m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1 + m[2][2] - m[0][0] - m[1][1]) * 2
        qw, qx, qy, qz = (m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    return [qw / norm, qx / norm, qy / norm, qz / norm]


def pose(row: dict[str, str]) -> tuple[list[list[float]], list[float]]:
    rotation = quaternion_to_rotation(*(float(row[key]) for key in ("qx", "qy", "qz", "qw")))
    translation = [float(row[key]) for key in ("tx", "ty", "tz")]
    return rotation, translation


def image_relative_path(raw_path: str, images_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        path = path.resolve().relative_to(images_dir.resolve())
    elif path.parts and path.parts[0] == images_dir.name:
        path = Path(*path.parts[1:])
    return path


def prepare(scene: Path, output: Path, reference_sensor: str) -> None:
    images_dir = scene / "images"
    cameras = json.loads((scene / "cameras.json").read_text(encoding="utf-8"))["virtual_cameras"]
    if len(cameras) != 8:
        raise ValueError(f"Expected 8 virtual cameras, found {len(cameras)}")
    if reference_sensor not in cameras:
        raise ValueError(f"Reference sensor {reference_sensor!r} is not in cameras.json")

    with (scene / "frames.csv").open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"timestamp", "virtual_camera", "image_path", "tx", "ty", "tz", "qx", "qy", "qz", "qw"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"frames.csv must contain: {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError("frames.csv contains no images")

    per_camera: dict[str, dict[str, str]] = defaultdict(dict)
    by_timestamp: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        camera = row["virtual_camera"]
        if camera not in cameras:
            raise ValueError(f"Unknown camera in frames.csv: {camera}")
        path = image_relative_path(row["image_path"], images_dir)
        if not path.parts or path.parts[0] != camera:
            raise ValueError(f"Image path must start with {camera}/: {path}")
        if not (images_dir / path).is_file():
            raise ValueError(f"Missing image: {images_dir / path}")
        suffix = Path(*path.parts[1:]).as_posix()
        per_camera[camera][suffix] = path.as_posix()
        if camera in by_timestamp[row["timestamp"]]:
            raise ValueError(f"Duplicate {camera} entry at timestamp {row['timestamp']}")
        by_timestamp[row["timestamp"]][camera] = row

    suffix_sets = [set(per_camera[camera]) for camera in cameras]
    complete = set.intersection(*suffix_sets)
    all_suffixes = set.union(*suffix_sets)
    if complete != all_suffixes:
        raise ValueError(f"Rig frames are incomplete: {len(complete)} complete / {len(all_suffixes)} total suffixes")

    output.mkdir(parents=True, exist_ok=True)
    list_dir = output / "image_lists"
    list_dir.mkdir(parents=True, exist_ok=True)
    order = [reference_sensor] + [camera for camera in cameras if camera != reference_sensor]
    for camera in order:
        with (list_dir / f"{camera}.txt").open("w", encoding="utf-8") as file:
            for suffix in sorted(per_camera[camera], key=lambda value: float(Path(value).stem)):
                file.write(per_camera[camera][suffix] + "\n")

    selected_timestamp = next(
        (
            timestamp
            for timestamp in sorted(by_timestamp, key=float)
            if all(c in by_timestamp[timestamp] for c in cameras)
        ),
        None,
    )
    if selected_timestamp is None:
        raise ValueError("No timestamp contains all 8 cameras")
    selected = by_timestamp[selected_timestamp]
    ref_rotation, ref_translation = pose(selected[reference_sensor])
    rig = []
    for camera in order:
        calibration: dict[str, Any] = cameras[camera]
        entry: dict[str, Any] = {
            "image_prefix": f"{camera}/",
            "camera_model_name": "PINHOLE",
            "camera_params": [float(calibration[key]) for key in ("fx", "fy", "cx", "cy")],
        }
        if camera == reference_sensor:
            entry["ref_sensor"] = True
        else:
            camera_rotation, camera_translation = pose(selected[camera])
            camera_from_world = transpose(camera_rotation)
            camera_from_reference = matmul(camera_from_world, ref_rotation)
            translation = matvec(
                camera_from_world,
                [ref_translation[i] - camera_translation[i] for i in range(3)],
            )
            entry["cam_from_rig_rotation"] = rotation_to_quaternion(camera_from_reference)
            entry["cam_from_rig_translation"] = translation
        rig.append(entry)

    (output / "rig_config.json").write_text(json.dumps([{"cameras": rig}], indent=2), encoding="utf-8")
    with (output / "cameras.tsv").open("w", encoding="utf-8") as file:
        for camera in order:
            calibration = cameras[camera]
            params = ",".join(str(float(calibration[key])) for key in ("fx", "fy", "cx", "cy"))
            file.write(f"{camera}\t{list_dir / (camera + '.txt')}\t{params}\t{len(per_camera[camera])}\n")
    print(f"Prepared 8 cameras, {len(complete)} synchronized frames, {len(complete) * 8} images")
    print(f"Rig extrinsics derived from synchronized timestamp {selected_timestamp}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path, help="Prepared scene containing images/, frames.csv, and cameras.json")
    parser.add_argument("output", type=Path, help="COLMAP workspace directory")
    parser.add_argument("--reference-sensor", default="cam0_right")
    args = parser.parse_args()
    prepare(args.scene, args.output, args.reference_sensor)


if __name__ == "__main__":
    main()
