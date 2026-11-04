#!/usr/bin/env python3
"""Convert exported Hilti26 fisheye images to virtual pinhole camera views.

Usage:
python3 tools/hilti/convert_to_pinhole.py \
  --input hilti26_export/floor_2_2025-05-05_run_1 \
  --out hilti26_export/floor_2_2025-05-05_run_1_pinhole \
  --clean
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from sparse_anchors.utils.geometry import (
    matrix_to_pose_fields,
    pose_row_to_matrix,
    rotation_z,
    view_rotation,
)

DEFAULT_INPUT_DIR = Path("hilti26_export/floor_2_2025-05-05_run_1")
DEFAULT_OUT_DIR = Path("hilti26_export/floor_2_2025-05-05_run_1_pinhole")

# 8-view virtual camera rig.
#
# The earlier 4-yaw x 3-pitch layout put some views near the cam0/cam1 overlap
# seam.  That is exactly where fisheye calibration/parallax errors are most
# visible.  This layout keeps each output image sampled from a single physical
# fisheye.
#
# Local yaw/pitch values are expressed relative to the owner camera optical axis.
# OpenCV convention: x right, y down, z forward.
VIEW_SPECS = [
    ("cam0_left", "cam0", -38.0, 0.0),
    ("cam0_right", "cam0", 38.0, 0.0),
    ("cam0_up", "cam0", 0.0, -32.0),
    ("cam0_down", "cam0", 0.0, 32.0),
    ("cam1_left", "cam1", -38.0, 0.0),
    ("cam1_right", "cam1", 38.0, 0.0),
    ("cam1_up", "cam1", 0.0, -32.0),
    ("cam1_down", "cam1", 0.0, 32.0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--fov-deg", type=float, default=90.0)
    parser.add_argument("--max-source-angle-deg", type=float, default=85.0)
    parser.add_argument("--sphere-radius-m", type=float, default=10.0)
    parser.add_argument("--roll-deg", type=float, default=180.0)
    parser.add_argument("--fusion-mode", choices=("best", "blend"), default="best")
    parser.add_argument("--max-frames", type=int, default=0, help="Debug limit on source fisheye frames. 0 means all.")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def build_virtual_pinhole_maps(
    source_cam: dict[str, Any],
    source_camera: str,
    R_cam0_virtual: np.ndarray,
    t_cam0_virtual: np.ndarray,
    T_cam0_cam1: np.ndarray,
    size: int,
    fov_deg: float,
    max_source_angle_deg: float,
    sphere_radius_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], np.ndarray]:
    virtual_focal = (size * 0.5) / np.tan(np.deg2rad(fov_deg) * 0.5)
    cx_virtual = (size - 1) * 0.5
    cy_virtual = (size - 1) * 0.5

    u, v = np.meshgrid(
        np.arange(size, dtype=np.float64),
        np.arange(size, dtype=np.float64),
    )

    x = (u - cx_virtual) / virtual_focal
    y = (v - cy_virtual) / virtual_focal
    z = np.ones_like(x)

    rays_virtual = np.stack([x, y, z], axis=-1)
    rays_virtual /= np.linalg.norm(rays_virtual, axis=-1, keepdims=True)

    rays_cam0 = rays_virtual.reshape(-1, 3) @ R_cam0_virtual.T
    points_cam0 = np.asarray(t_cam0_virtual, dtype=float).reshape(1, 3) + rays_cam0 * float(sphere_radius_m)
    if source_camera == "cam0":
        rays_source = points_cam0
    else:
        R_cam0_cam1 = T_cam0_cam1[:3, :3]
        t_cam0_cam1 = T_cam0_cam1[:3, 3]
        rays_source = (points_cam0 - t_cam0_cam1) @ R_cam0_cam1.T

    X = rays_source[:, 0]
    Y = rays_source[:, 1]
    Z = rays_source[:, 2]
    dist = np.linalg.norm(rays_source, axis=1)
    rho = np.sqrt(X * X + Y * Y)
    theta = np.arctan2(rho, Z)

    source_focal = 0.5 * (float(source_cam["fx"]) + float(source_cam["fy"]))
    cx = float(source_cam["cx"])
    cy = float(source_cam["cy"])
    k1, k2, k3, k4 = [float(k) for k in source_cam["distortion_coeffs"]]

    theta2 = theta * theta
    theta_d = theta * (
        1.0
        + k1 * theta2
        + k2 * theta2 * theta2
        + k3 * theta2 * theta2 * theta2
        + k4 * theta2 * theta2 * theta2 * theta2
    )
    scale = np.divide(theta_d, rho, out=np.zeros_like(theta_d), where=rho > 1e-12)

    map_x = source_focal * X * scale + cx
    map_y = source_focal * Y * scale + cy

    width = int(source_cam["width"])
    height = int(source_cam["height"])
    valid = (
        (map_x >= 0)
        & (map_x < width - 1)
        & (map_y >= 0)
        & (map_y < height - 1)
        & (theta <= np.deg2rad(max_source_angle_deg))
    )
    weight = np.divide(1.0, np.maximum(dist, 1e-6), out=np.zeros_like(theta), where=valid)

    map_x = map_x.reshape(size, size).astype(np.float32)
    map_y = map_y.reshape(size, size).astype(np.float32)
    valid = valid.reshape(size, size)
    weight = weight.reshape(size, size).astype(np.float32)
    rays_virtual = rays_virtual.reshape(size, size, 3)

    virtual_cam = {
        "width": size,
        "height": size,
        "fx": float(virtual_focal),
        "fy": float(virtual_focal),
        "cx": float(cx_virtual),
        "cy": float(cy_virtual),
        "camera_model": "pinhole",
        "fov_deg": float(fov_deg),
    }
    return map_x, map_y, valid, weight, virtual_cam, rays_virtual


def view_rotation_in_cam0(
    owner_camera: str,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    T_cam0_cam1: np.ndarray,
) -> np.ndarray:
    R_owner_virtual = view_rotation(yaw_deg, pitch_deg) @ rotation_z(roll_deg)
    if owner_camera == "cam0":
        return R_owner_virtual
    R_cam0_cam1 = T_cam0_cam1[:3, :3]
    return R_cam0_cam1 @ R_owner_virtual


def view_translation_in_cam0(owner_camera: str, T_cam0_cam1: np.ndarray) -> np.ndarray:
    if owner_camera == "cam0":
        return np.zeros(3, dtype=float)
    return T_cam0_cam1[:3, 3].astype(float)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def group_frames_by_timestamp(frames: list[dict[str, str]]) -> list[tuple[str, dict[str, dict[str, str]]]]:
    groups: dict[str, dict[str, dict[str, str]]] = {}
    for frame in frames:
        groups.setdefault(frame["timestamp"], {})[frame["camera"]] = frame
    return sorted(groups.items(), key=lambda item: float(item[0]))


def clean_output(out_dir: Path) -> None:
    for path in [
        out_dir / "frames.csv",
        out_dir / "cameras.json",
        out_dir / "export_report.json",
        out_dir / "contact_sheet.png",
    ]:
        if path.exists():
            path.unlink()
    image_root = out_dir / "images"
    if image_root.exists():
        for image_path in image_root.glob("*/*.png"):
            image_path.unlink()
    mask_root = out_dir / "masks"
    if mask_root.exists():
        for mask_path in mask_root.glob("*.png"):
            mask_path.unlink()


def write_contact_sheet(out_dir: Path, image_paths: list[Path]) -> None:
    thumbs = []
    for path in image_paths[:12]:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        label = path.parent.name
        cv2.putText(image, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(image, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        thumbs.append(image)
    if not thumbs:
        return
    rows = []
    for start in range(0, len(thumbs), 6):
        row = thumbs[start : start + 6]
        while len(row) < 6:
            row.append(np.zeros_like(thumbs[0]))
        rows.append(np.hstack(row))
    cv2.imwrite(str(out_dir / "contact_sheet.png"), np.vstack(rows))


def build_owner_masks(
    view_rotations: dict[str, np.ndarray],
    rays_by_view: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    names = list(view_rotations)
    axes = np.stack([view_rotations[name] @ np.array([0.0, 0.0, 1.0]) for name in names])
    masks = {}
    for name in names:
        rays_cam0 = rays_by_view[name].reshape(-1, 3)
        owner = np.argmax(rays_cam0 @ axes.T, axis=1)
        masks[name] = owner.reshape(rays_by_view[name].shape[:2]) == names.index(name)
    return masks


def render_from_fisheyes(
    source_images: dict[str, np.ndarray],
    view_maps: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    fusion_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    sampled_images = {}
    valid_maps = {}
    weight_maps = {}
    for camera_name, (map_x, map_y, valid, weight) in view_maps.items():
        sampled_images[camera_name] = cv2.remap(
            source_images[camera_name],
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        valid_maps[camera_name] = valid
        weight_maps[camera_name] = weight

    valid_any = np.zeros(next(iter(valid_maps.values())).shape, dtype=bool)
    for valid in valid_maps.values():
        valid_any |= valid

    if len(sampled_images) == 1:
        camera_name = next(iter(sampled_images))
        image = np.zeros_like(sampled_images[camera_name])
        image[valid_maps[camera_name]] = sampled_images[camera_name][valid_maps[camera_name]]
        image[~valid_any] = 0
        return image, valid_any

    if fusion_mode == "best":
        use_cam0 = valid_maps["cam0"] & (~valid_maps["cam1"] | (weight_maps["cam0"] >= weight_maps["cam1"]))
        use_cam1 = valid_maps["cam1"] & (~valid_maps["cam0"] | (weight_maps["cam1"] > weight_maps["cam0"]))
        image = np.zeros_like(sampled_images["cam0"])
        image[use_cam0] = sampled_images["cam0"][use_cam0]
        image[use_cam1] = sampled_images["cam1"][use_cam1]
    else:
        accum = (
            sampled_images["cam0"].astype(np.float32) * weight_maps["cam0"][..., None]
            + sampled_images["cam1"].astype(np.float32) * weight_maps["cam1"][..., None]
        )
        total_weight = weight_maps["cam0"][..., None] + weight_maps["cam1"][..., None]
        image = np.divide(accum, np.maximum(total_weight, 1e-6), out=np.zeros_like(accum), where=total_weight > 0)
        image = np.clip(image, 0, 255).astype(np.uint8)

    image[~valid_any] = 0
    return image, valid_any


def convert_dataset(args: argparse.Namespace) -> dict[str, Any]:
    cameras = json.loads((args.input / "cameras.json").read_text())
    frames = load_csv(args.input / "frames.csv")
    if args.max_frames > 0:
        timestamps = {timestamp for timestamp, _ in group_frames_by_timestamp(frames)[: args.max_frames]}
        frames = [frame for frame in frames if frame["timestamp"] in timestamps]

    if args.clean:
        clean_output(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "masks").mkdir(parents=True, exist_ok=True)

    T_cam0_cam1 = np.array(cameras["T_cam0_cam1"], dtype=float)
    maps = {}
    rays_by_view = {}
    view_rotations = {}
    view_translations = {}
    virtual_cameras = {}
    for view_name, owner_camera, yaw_deg, pitch_deg in VIEW_SPECS:
        R_cam0_virtual = view_rotation_in_cam0(owner_camera, yaw_deg, pitch_deg, args.roll_deg, T_cam0_cam1)
        t_cam0_virtual = view_translation_in_cam0(owner_camera, T_cam0_cam1)
        maps[view_name] = {}
        virtual_cam = None
        rays_virtual = None
        map_x, map_y, valid, weight, virtual_cam, rays_virtual = build_virtual_pinhole_maps(
            cameras[owner_camera],
            source_camera=owner_camera,
            R_cam0_virtual=R_cam0_virtual,
            t_cam0_virtual=t_cam0_virtual,
            T_cam0_cam1=T_cam0_cam1,
            size=args.size,
            fov_deg=args.fov_deg,
            max_source_angle_deg=args.max_source_angle_deg,
            sphere_radius_m=args.sphere_radius_m,
        )
        maps[view_name][owner_camera] = (map_x, map_y, valid, weight)
        rays_by_view[view_name] = rays_virtual.reshape(-1, 3) @ R_cam0_virtual.T
        rays_by_view[view_name] = rays_by_view[view_name].reshape(args.size, args.size, 3)
        view_rotations[view_name] = R_cam0_virtual
        view_translations[view_name] = t_cam0_virtual
        virtual_cameras[view_name] = {
            **virtual_cam,
            "source_camera": owner_camera,
            "yaw_deg_owner_to_virtual": yaw_deg,
            "pitch_deg_owner_to_virtual": pitch_deg,
            "roll_deg_virtual": args.roll_deg,
            "tx_cam0_to_virtual": float(t_cam0_virtual[0]),
            "ty_cam0_to_virtual": float(t_cam0_virtual[1]),
            "tz_cam0_to_virtual": float(t_cam0_virtual[2]),
            "ownership_mask_path": f"masks/{view_name}.png",
        }
        (args.out / "images" / view_name).mkdir(parents=True, exist_ok=True)

    owner_masks = build_owner_masks(view_rotations, rays_by_view)
    for view_name, owner_mask in owner_masks.items():
        valid_any = np.zeros(owner_mask.shape, dtype=bool)
        for _map_x, _map_y, valid, _weight in maps[view_name].values():
            valid_any |= valid
        cv2.imwrite(str(args.out / "masks" / f"{view_name}.png"), ((owner_mask & valid_any) * 255).astype(np.uint8))

    output_rows = []
    preview_paths = []
    count_by_camera = {name: 0 for name in virtual_cameras}
    source_count_by_camera = {name: {"cam0": 0, "cam1": 0, "fused": 0} for name in virtual_cameras}

    for timestamp_text, timestamp_frames in group_frames_by_timestamp(frames):
        if "cam0" not in timestamp_frames or "cam1" not in timestamp_frames:
            continue

        source_images = {}
        for source_camera in ("cam0", "cam1"):
            source_path = args.input / timestamp_frames[source_camera]["image_path"]
            source_image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            source_images[source_camera] = source_image

        T_map_cam0 = pose_row_to_matrix(timestamp_frames["cam0"])
        timestamp = float(timestamp_text)

        for view_name, owner_camera, _yaw_deg, _pitch_deg in VIEW_SPECS:
            virtual_image, _valid_any = render_from_fisheyes(source_images, maps[view_name], args.fusion_mode)

            filename = f"{timestamp:.9f}.png"
            rel_path = Path("images") / view_name / filename
            out_path = args.out / rel_path
            out_path.write_bytes(cv2.imencode(out_path.suffix, virtual_image)[1].tobytes())

            T_source_virtual = np.eye(4, dtype=float)
            T_source_virtual[:3, :3] = view_rotations[view_name]
            T_source_virtual[:3, 3] = view_translations[view_name]
            T_map_virtual = T_map_cam0 @ T_source_virtual
            trans, quat_xyzw = matrix_to_pose_fields(T_map_virtual)

            output_rows.append(
                [
                    f"{timestamp:.9f}",
                    owner_camera,
                    view_name,
                    rel_path.as_posix(),
                    f"{trans[0]:.9f}",
                    f"{trans[1]:.9f}",
                    f"{trans[2]:.9f}",
                    f"{quat_xyzw[0]:.12f}",
                    f"{quat_xyzw[1]:.12f}",
                    f"{quat_xyzw[2]:.12f}",
                    f"{quat_xyzw[3]:.12f}",
                ]
            )
            count_by_camera[view_name] += 1
            source_count_by_camera[view_name][owner_camera] += 1
            if len(preview_paths) < 12:
                preview_paths.append(out_path)

    with (args.out / "frames.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "source_camera",
                "virtual_camera",
                "image_path",
                "tx",
                "ty",
                "tz",
                "qx",
                "qy",
                "qz",
                "qw",
            ]
        )
        writer.writerows(output_rows)

    with (args.out / "cameras.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "pose_convention": "frames.csv stores T_map_virtual as tx ty tz qx qy qz qw",
                "source_dataset": str(args.input),
                "virtual_cameras": virtual_cameras,
            },
            f,
            indent=2,
        )

    report = {
        "input": str(args.input),
        "out": str(args.out),
        "size": args.size,
        "fov_deg": args.fov_deg,
        "max_source_angle_deg": args.max_source_angle_deg,
        "sphere_radius_m": args.sphere_radius_m,
        "roll_deg": args.roll_deg,
        "virtual_view_layout": "single_source_8_cross",
        "cam1_projection_transform": "finite_T_cam0_cam1_formula",
        "fusion_mode": args.fusion_mode,
        "source_frames": len(frames),
        "virtual_frames": len(output_rows),
        "count_by_camera": count_by_camera,
        "source_count_by_camera": source_count_by_camera,
    }
    with (args.out / "export_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    write_contact_sheet(args.out, preview_paths)
    return report


def main() -> None:
    args = parse_args()
    report = convert_dataset(args)
    print("source frames:", report["source_frames"])
    print("virtual frames:", report["virtual_frames"])
    print("out:", report["out"])
    print("contact sheet:", Path(report["out"]) / "contact_sheet.png")


if __name__ == "__main__":
    main()
