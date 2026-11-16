"""Evaluate every saved localization stage against query ground truth."""

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
from sparse_anchors.data.sessions import SessionData, frame_camera_name, load_gt_poses
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
    plot_pose_result_map,
    plot_vpr_matches_map,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=config.output_dir)
    parser.add_argument("--diagonals", action="store_true", help="Render comparison sheets for PnP poses")
    args = parser.parse_args()
    run_dir = args.output_dir
    pnp_dir = run_dir / "pnp" if (run_dir / "pnp" / "pose_estimates.json").exists() else run_dir
    evaluation_dir = run_dir / "evaluation"
    plot_dir = evaluation_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    pnp_payload = json.loads((pnp_dir / "pose_estimates.json").read_text())
    pose_rows = pnp_payload["poses"]
    retrievals_path = pnp_dir / "retrievals.json"
    match_rows_path = pnp_dir / "match_rows.json"
    retrievals = json.loads(retrievals_path.read_text()) if retrievals_path.exists() else []
    match_rows = json.loads(match_rows_path.read_text()) if match_rows_path.exists() else []

    reference_scene = normalize_scene(
        load_scene(
            sparse_dir=config.reference_sparse_dir,
            images_dir=config.reference_images_dir,
            feature_manifest_path=config.feature_manifest,
            test_every=config.reference_test_every,
        )
    )
    reference_data = SessionData(reference_scene.colmap.frames)
    query_model = load_colmap_model(config.query_sparse_dir, config.query_images_dir)
    query_data = SessionData(query_model.frames)
    reference_gt_poses, reference_xy = load_gt_poses(config.reference_frames_csv)
    query_gt_poses, query_xy = load_gt_poses(config.query_frames_csv)
    gt_from_reference_sfm = estimate_gt_from_reference_sfm(reference_data, reference_gt_poses)

    query_rig_image_by_timestamp = {
        timestamp: next(frame.image_name for frame in frames if frame_camera_name(frame) == config.rig_camera_name)
        for timestamp, frames in query_data.frames_by_timestamp.items()
    }
    result_sets = {}
    result_sets["pnp_hypotheses"] = [
        {
            "query_timestamp": row["query_timestamp"],
            "c2w": transform_c2w(gt_from_reference_sfm, np.asarray(row["rig_c2w"])),
        }
        for row in pose_rows
    ]

    consensus_dir = run_dir / "consensus"
    consensus_path = consensus_dir / "consensus.json"
    if consensus_path.exists():
        consensus = json.loads(consensus_path.read_text())
        consensus_anchors = [
            {
                "query_timestamp": row["query_timestamp"],
                "c2w": transform_c2w(gt_from_reference_sfm, np.asarray(row["rig_c2w"])),
            }
            for row in consensus["inlier_rows"]
        ]
        result_sets["consensus_anchors"] = consensus_anchors
        result_sets["consensus_alignment"] = load_trajectory(
            consensus_dir / "consensus_trajectory.json", gt_from_reference_sfm
        )

    pose_graph_dir = run_dir / "pose_graph"
    for name in ("pnp_all", "consensus"):
        trajectory_path = pose_graph_dir / f"{name}_trajectory.json"
        if trajectory_path.exists():
            result_sets[f"pose_graph_{name}"] = load_trajectory(trajectory_path, gt_from_reference_sfm)
    for name in ("pnp_all", "consensus"):
        trajectory_path = consensus_dir / f"{name}_pgo_trajectory.json"
        if trajectory_path.exists():
            result_sets[f"pose_graph_{name}"] = load_trajectory(trajectory_path, gt_from_reference_sfm)

    featuremetric_dir = run_dir / "featuremetric"
    refinements_path = featuremetric_dir / "refinements.json"
    if refinements_path.exists():
        refinements = json.loads(refinements_path.read_text())
        result_sets["featuremetric_anchors"] = [
            {
                "query_timestamp": row["query_timestamp"],
                "c2w": transform_c2w(gt_from_reference_sfm, np.asarray(row["final_c2w"])),
            }
            for row in refinements
        ]
    final_trajectory_path = featuremetric_dir / "trajectory.json"
    if final_trajectory_path.exists():
        result_sets["final_featuremetric_pgo"] = load_trajectory(final_trajectory_path, gt_from_reference_sfm)

    floorplan = load_rgb(config.floorplan) if config.floorplan else None
    extent = None
    if floorplan is not None:
        height, width = floorplan.shape[:2]
        extent = (0.0, width * config.floorplan_resolution_m, 0.0, height * config.floorplan_resolution_m)

    if retrievals_path.exists():
        figure, _ = plot_vpr_matches_map(
            query_xy,
            reference_xy,
            retrievals,
            correct_distance_m=config.correct_distance_m,
            floorplan=floorplan,
            floorplan_extent=extent,
            show=False,
        )
        save_figure(figure, plot_dir / "vpr_matches.png")
    if match_rows_path.exists():
        figure, _ = plot_image_matching_survivors_map(
            query_xy,
            reference_xy,
            match_rows,
            min_inliers=config.match_min_inliers,
            floorplan=floorplan,
            floorplan_extent=extent,
            show=False,
        )
        save_figure(figure, plot_dir / "image_matching.png")

    for name, poses in result_sets.items():
        figure, _ = plot_pose_result_map(
            query_xy,
            reference_xy,
            poses,
            title=name.replace("_", " ").title(),
            floorplan=floorplan,
            floorplan_extent=extent,
            show=False,
        )
        save_figure(figure, plot_dir / f"{name}.png")

    metric_rows = []
    summaries = []
    for name, poses in result_sets.items():
        rows = pose_metrics(name, poses, query_gt_poses, query_rig_image_by_timestamp)
        metric_rows.extend(rows)
        summaries.append(summarize(name, rows, len(query_rig_image_by_timestamp)))

    write_metrics(evaluation_dir, metric_rows, summaries)
    summary_text = format_summary(summaries)
    (evaluation_dir / "summary.txt").write_text(summary_text)
    print(summary_text, end="")
    if args.diagonals:
        render_diagonals(pose_rows, pnp_dir)


def load_trajectory(path: Path, gt_from_reference_sfm: np.ndarray) -> list[dict]:
    trajectory = json.loads(path.read_text())
    return [
        {"query_timestamp": timestamp, "c2w": transform_c2w(gt_from_reference_sfm, np.asarray(pose))}
        for timestamp, pose in trajectory.items()
    ]


def pose_metrics(
    stage: str,
    poses: list[dict],
    query_gt_poses: dict[str, np.ndarray],
    query_rig_image_by_timestamp: dict[str, str],
) -> list[dict]:
    rows = []
    for pose in poses:
        timestamp = str(pose["query_timestamp"])
        image_name = query_rig_image_by_timestamp[timestamp]
        gt_c2w = query_gt_poses[image_name]
        c2w = np.asarray(pose["c2w"], dtype=np.float64)
        translation_delta = c2w[:3, 3] - gt_c2w[:3, 3]
        rotation_delta = gt_c2w[:3, :3].T @ c2w[:3, :3]
        rows.append(
            {
                "stage": stage,
                "query_timestamp": timestamp,
                "query_image": image_name,
                "translation_error_m": float(np.linalg.norm(translation_delta)),
                "xy_translation_error_m": float(np.linalg.norm(translation_delta[:2])),
                "rotation_error_deg": float(Rotation.from_matrix(rotation_delta).magnitude() * 180.0 / np.pi),
            }
        )
    return rows


def summarize(stage: str, rows: list[dict], num_query_poses: int) -> dict:
    translation = np.asarray([row["translation_error_m"] for row in rows])
    rotation = np.asarray([row["rotation_error_deg"] for row in rows])
    within_five = translation < 5.0
    best_error_by_timestamp = {
        timestamp: min(row["translation_error_m"] for row in rows if row["query_timestamp"] == timestamp)
        for timestamp in dict.fromkeys(row["query_timestamp"] for row in rows)
    }
    return {
        "stage": stage,
        "poses": len(rows),
        "translation_rmse_m": float(np.sqrt(np.mean(translation**2))),
        "translation_rmse_under_5m_m": float(np.sqrt(np.mean(translation[within_five] ** 2))),
        "rotation_rmse_under_5m_deg": float(np.sqrt(np.mean(rotation[within_five] ** 2))),
        "recall_at_0_5m": float(sum(error < 0.5 for error in best_error_by_timestamp.values()) / num_query_poses),
        "recall_at_1m": float(sum(error < 1.0 for error in best_error_by_timestamp.values()) / num_query_poses),
        "recall_at_5m": float(sum(error < 5.0 for error in best_error_by_timestamp.values()) / num_query_poses),
    }


def write_metrics(evaluation_dir: Path, metric_rows: list[dict], summaries: list[dict]) -> None:
    with (evaluation_dir / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    (evaluation_dir / "metrics.json").write_text(json.dumps(summaries, indent=2) + "\n")


def format_summary(summaries: list[dict]) -> str:
    header = "stage                         poses   t-RMSE[m]  t-RMSE<5m[m]  r-RMSE<5m[deg]  R@0.5m   R@1m    R@5m"
    lines = [header]
    for row in summaries:
        lines.append(
            f"{row['stage']:<29} {row['poses']:>5d}   {row['translation_rmse_m']:>9.3f}  "
            f"{row['translation_rmse_under_5m_m']:>12.3f}  {row['rotation_rmse_under_5m_deg']:>14.3f}  "
            f"{row['recall_at_0_5m']:>6.1%}  {row['recall_at_1m']:>6.1%}  {row['recall_at_5m']:>6.1%}"
        )
    return "\n".join(lines) + "\n"


def save_figure(figure, path: Path) -> None:
    figure.savefig(path, dpi=200)
    plt.close(figure)


@torch.no_grad()
def render_diagonals(pose_rows: list[dict], output_dir: Path) -> None:
    query_data = SessionData(load_colmap_model(config.query_sparse_dir, config.query_images_dir).frames)
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
