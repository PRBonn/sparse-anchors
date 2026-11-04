"""Cache raw pose-conditioned DA3 depths for all eight HILTI cameras with temporal radius 2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from evo.core.geometry import GeometryException

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "thirdparty/Depth-Anything-3/src"))
sys.path.insert(0, str(ROOT / "src"))

from depth_anything_3.api import DepthAnything3  # noqa: E402

from sparse_anchors.data.colmap import Frame, load_colmap_model  # noqa: E402
from sparse_anchors.data.sessions import group_frames_by_timestamp  # noqa: E402

MODEL = "depth-anything/DA3-LARGE-1.1"
MODEL_REVISION = "0e109ae307c5982f319a67cf6f9f99ccdc0ec97c"
TEMPORAL_RADIUS = 2
PROCESS_RES = 504
CAMERAS = tuple(f"cam{rig}_{direction}" for rig in range(2) for direction in ("down", "left", "right", "up"))


def handle_collinear_pose_windows(model: DepthAnything3) -> None:
    """Recover DA3's pose scale when Umeyama cannot align a straight rig trajectory."""
    align = model._align_to_input_extrinsics_intrinsics

    def align_depth(extrinsics, intrinsics, prediction, align_to_input_ext_scale=True, ransac_view_thresh=10):
        try:
            return align(extrinsics, intrinsics, prediction, align_to_input_ext_scale, ransac_view_thresh)
        except GeometryException:
            known = extrinsics.numpy()
            predicted = prediction.extrinsics
            known_centers = -np.einsum("nji,nj->ni", known[:, :3, :3], known[:, :3, 3])
            predicted_centers = -np.einsum("nji,nj->ni", predicted[:, :3, :3], predicted[:, :3, 3])
            known_distances = np.linalg.norm(known_centers[:, None] - known_centers[None, :], axis=-1)
            predicted_distances = np.linalg.norm(predicted_centers[:, None] - predicted_centers[None, :], axis=-1)
            upper = np.triu_indices(len(known), 1)
            known_distances, predicted_distances = known_distances[upper], predicted_distances[upper]
            valid = (known_distances > 1e-6) & (predicted_distances > 1e-6)
            prediction.depth /= np.median(predicted_distances[valid] / known_distances[valid])
            prediction.intrinsics = intrinsics.numpy()
            prediction.extrinsics = known[:, :3, :]
            print("Recovered DA3 pose scale from camera baselines for a collinear window.", flush=True)
            return prediction

    model._align_to_input_extrinsics_intrinsics = align_depth


def write_timestamp(
    model: DepthAnything3,
    groups: dict[str, tuple[Frame, ...]],
    timestamps: list[str],
    index: int,
    output: Path,
) -> None:
    timestamp = timestamps[index]
    window = timestamps[max(0, index - TEMPORAL_RADIUS) : index + TEMPORAL_RADIUS + 1]
    frames = tuple(frame for time in window for frame in groups[time])
    print(f"[{index + 1}/{len(timestamps)}] {timestamp}: {len(frames)} input images", flush=True)
    prediction = model.inference(
        image=[str(frame.image_path) for frame in frames],
        extrinsics=np.stack([frame.w2c for frame in frames]).astype(np.float32),
        intrinsics=np.stack([frame.K for frame in frames]).astype(np.float32),
        align_to_input_ext_scale=True,
        ref_view_strategy="saddle_balanced",
        process_res=PROCESS_RES,
        process_res_method="upper_bound_resize",
    )
    indices = [i for i, frame in enumerate(frames) if Path(frame.image_name).stem == timestamp]
    center_frames = groups[timestamp]
    path = output / f"{timestamp}.npz"
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(
            file,
            depth=prediction.depth[indices].astype(np.float32),
            conf=prediction.conf[indices].astype(np.float32),
            intrinsics=prediction.intrinsics[indices].astype(np.float32),
            extrinsics=np.stack([frame.w2c for frame in center_frames]).astype(np.float32),
            image_names=np.asarray([frame.image_name for frame in center_frames]),
            camera_names=np.asarray(CAMERAS),
            timestamp=np.asarray(timestamp),
            window_timestamps=np.asarray(window),
        )
    temporary.replace(path)
    print(f"wrote {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene", type=Path, action="append", required=True, help="COLMAP scene; repeat for several runs"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate completed caches in depth/da3")
    args = parser.parse_args()

    model = DepthAnything3.from_pretrained(MODEL, revision=MODEL_REVISION).to(args.device).eval()
    handle_collinear_pose_windows(model)
    for scene in args.scene:
        colmap = load_colmap_model(scene / "sparse/0", scene / "images")
        groups = group_frames_by_timestamp(colmap.frames)
        timestamps = list(groups)
        for timestamp, frames in groups.items():
            cameras = tuple(Path(frame.image_name).parent.name for frame in frames)
            if cameras != CAMERAS:
                raise ValueError(f"{scene}: {timestamp} must contain all eight cameras; found {cameras}")
        output = scene / "depth/da3"
        output.mkdir(parents=True, exist_ok=True)
        config = {
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "temporal_radius": TEMPORAL_RADIUS,
            "process_res": PROCESS_RES,
            "process_res_method": "upper_bound_resize",
            "cameras": CAMERAS,
            "depth_convention": "camera_z_in_sfm_units",
            "pose_convention": "world_to_camera",
        }
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        print(
            f"{scene.parent.name}: {len(timestamps)} timestamps, all 8 cameras, temporal radius 2 -> {output}",
            flush=True,
        )
        for index, timestamp in enumerate(timestamps):
            if (output / f"{timestamp}.npz").is_file() and not args.overwrite:
                print(f"[{index + 1}/{len(timestamps)}] {timestamp}: exists", flush=True)
                continue
            write_timestamp(model, groups, timestamps, index, output)


if __name__ == "__main__":
    main()
