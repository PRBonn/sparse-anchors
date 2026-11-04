"""Evaluate saved rig PnP hypotheses without changing the localization run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from configs import localization_exp as config
from sparse_anchors.data.colmap import load_colmap_model
from sparse_anchors.data.sessions import SessionData, load_gt_poses
from sparse_anchors.splatting.renderer import ViewCamera, render
from sparse_anchors.splatting.scene import load_scene, normalize_scene
from sparse_anchors.utils.evaluation import estimate_gt_from_reference_sfm
from sparse_anchors.utils.geometry import transform_c2w
from sparse_anchors.utils.image import load_rgb, write_rgb
from sparse_anchors.utils.io import load_splat_checkpoint
from sparse_anchors.utils.visualization import (
    diagonal_rgb_composite,
    make_contact_sheet,
    plot_image_matching_survivors_map,
    plot_pnp_estimates_map,
    plot_vpr_matches_map,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=config.output_dir)
    parser.add_argument("--diagonals", action="store_true", help="Render diagnostic comparison sheets for every pose")
    args = parser.parse_args()
    output_dir = args.output_dir

    retrievals = json.loads((output_dir / "retrievals.json").read_text())
    match_rows = json.loads((output_dir / "match_rows.json").read_text())
    pose_rows = json.loads((output_dir / "pose_estimates.json").read_text())["poses"]

    reference_scene = normalize_scene(
        load_scene(
            sparse_dir=config.reference_sparse_dir,
            images_dir=config.reference_images_dir,
            feature_manifest_path=config.feature_manifest,
            test_every=config.reference_test_every,
        )
    )
    reference_data = SessionData(reference_scene.colmap.frames)
    reference_gt_poses, reference_xy = load_gt_poses(config.reference_frames_csv)
    query_gt_poses, query_xy = load_gt_poses(config.query_frames_csv)
    gt_from_reference_sfm = estimate_gt_from_reference_sfm(reference_data, reference_gt_poses)
    map_pose_rows = [
        {**row, "c2w": transform_c2w(gt_from_reference_sfm, np.asarray(row["rig_c2w"]))} for row in pose_rows
    ]

    floorplan = load_rgb(config.floorplan) if config.floorplan else None
    extent = None
    if floorplan is not None:
        height, width = floorplan.shape[:2]
        extent = (0.0, width * config.floorplan_resolution_m, 0.0, height * config.floorplan_resolution_m)

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    figures = (
        (
            "vpr_matches_map.png",
            plot_vpr_matches_map(
                query_xy,
                reference_xy,
                retrievals,
                correct_distance_m=config.correct_distance_m,
                floorplan=floorplan,
                floorplan_extent=extent,
                show=False,
            ),
        ),
        (
            "image_matching_survivors_map.png",
            plot_image_matching_survivors_map(
                query_xy,
                reference_xy,
                match_rows,
                min_inliers=config.match_min_inliers,
                floorplan=floorplan,
                floorplan_extent=extent,
                show=False,
            ),
        ),
        (
            "pnp_estimates_map.png",
            plot_pnp_estimates_map(
                query_xy,
                reference_xy,
                map_pose_rows,
                floorplan=floorplan,
                floorplan_extent=extent,
                show=False,
            ),
        ),
    )
    for filename, (figure, _) in figures:
        figure.savefig(plot_dir / filename, dpi=200)
        plt.close(figure)

    summary = write_pose_metrics(map_pose_rows, query_gt_poses, output_dir)
    print(json.dumps(summary, indent=2))
    if args.diagonals:
        render_diagonals(pose_rows, output_dir)


def write_pose_metrics(pose_rows: list[dict], query_gt_poses: dict[str, np.ndarray], output_dir: Path) -> dict:
    metric_rows = []
    for row in pose_rows:
        gt_c2w = query_gt_poses[row["query_rig_image"]]
        c2w = np.asarray(row["c2w"], dtype=np.float64)
        translation_delta = c2w[:3, 3] - gt_c2w[:3, 3]
        rotation_delta = gt_c2w[:3, :3].T @ c2w[:3, :3]
        metric_rows.append(
            {
                "query_timestamp": row["query_timestamp"],
                "reference_timestamp": row["reference_timestamp"],
                "query_image": row["query_rig_image"],
                "reference_image": row["reference_image"],
                "translation_error_m": float(np.linalg.norm(translation_delta)),
                "xy_translation_error_m": float(np.linalg.norm(translation_delta[:2])),
                "z_translation_error_m": float(abs(translation_delta[2])),
                "rotation_error_deg": float(Rotation.from_matrix(rotation_delta).magnitude() * 180.0 / np.pi),
                "reprojection_error_px": float(row["reprojection_error_px"]),
                "num_inliers": int(row["depth_num_inliers"]),
                "rig_pnp_num_inliers": int(row["rig_pnp_num_inliers"]),
                "num_pairs": int(row["num_pairs"]),
                "masked_num_inliers": int(row["masked_num_inliers"]),
                "raw_num_inliers": int(row["raw_num_inliers"]),
                "vpr_score": float(row["vpr_score"]),
                "image_score": float(row["image_score"]),
            }
        )

    with (output_dir / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    def stats(key):
        values = np.asarray([row[key] for row in metric_rows], dtype=np.float64)
        return {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)),
        }

    summary = {
        "num_pose_estimates": len(pose_rows),
        "num_poses_with_gt": len(metric_rows),
        "translation_error_m": stats("translation_error_m"),
        "xy_translation_error_m": stats("xy_translation_error_m"),
        "rotation_error_deg": stats("rotation_error_deg"),
        "reprojection_error_px": stats("reprojection_error_px"),
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


@torch.no_grad()
def render_diagonals(pose_rows: list[dict], output_dir: Path):
    query_data = SessionData(
        load_colmap_model(sparse_dir=config.query_sparse_dir, images_dir=config.query_images_dir).frames
    )
    checkpoint = load_splat_checkpoint(config.checkpoint, torch.device(config.device))
    diagonal_dir = output_dir / "diagonal"
    diagonal_dir.mkdir(parents=True, exist_ok=True)

    for row in pose_rows:
        poses_by_image = {pose["image_name"]: pose for pose in row["virtual_camera_poses"]}
        tiles = []
        for frame in query_data.frames_by_timestamp[row["query_timestamp"]]:
            image_rgb = load_rgb(frame.image_path)
            rendered = render(
                checkpoint.model,
                ViewCamera(
                    w2c=np.asarray(poses_by_image[frame.image_name]["w2c"], dtype=np.float64),
                    K=np.asarray(frame.K, dtype=np.float64),
                    width=image_rgb.shape[1],
                    height=image_rgb.shape[0],
                ),
                sh_degree=checkpoint.sh_degree,
                packed=checkpoint.packed,
                render_latents=False,
            )
            tiles.append(diagonal_rgb_composite(rendered.rgb, image_rgb))
        write_rgb(
            diagonal_dir / f"{row['query_timestamp']}__{row['reference_timestamp']}__rig.png",
            make_contact_sheet(tiles, columns=5),
        )


if __name__ == "__main__":
    main()
