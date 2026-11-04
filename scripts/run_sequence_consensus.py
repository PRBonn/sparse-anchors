"""Align a query SfM sequence to the reference map using PnP consensus."""

import argparse
import json
from pathlib import Path

import numpy as np

from configs import localization_exp as config
from sparse_anchors.data.colmap import load_colmap_model
from sparse_anchors.data.sessions import SessionData, frame_camera_name
from sparse_anchors.localization.consensus import consensus_quality, initial_consensus, refine_consensus
from sparse_anchors.utils.geometry import transform_c2w


def select_best_pnp_poses(rows: list[dict]) -> list[dict]:
    """Keep the lowest-error PnP pose per query, breaking ties by inliers and VPR rank."""
    best = {}

    def score(row):
        return (
            row["reprojection_error_px"],
            -row["rig_pnp_num_inliers"],
            row["vpr_rank"],
        )

    for row in rows:
        timestamp = row["query_timestamp"]

        if timestamp not in best or score(row) < score(best[timestamp]):
            best[timestamp] = row

    return sorted(best.values(), key=lambda row: float(row["query_timestamp"]))


def load_query_rig_trajectory(data: SessionData, rig_camera_name: str) -> tuple[list[str], np.ndarray]:
    """Return one query-SfM c2w rig pose for each timestamp, in numeric order."""
    timestamps = []
    poses = []

    for timestamp, frames in data.frames_by_timestamp.items():
        rig_frame = next(frame for frame in frames if frame_camera_name(frame) == rig_camera_name)
        timestamps.append(timestamp)
        poses.append(np.asarray(rig_frame.c2w, dtype=np.float64))

    return timestamps, np.stack(poses)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-file", type=Path, default=config.output_dir / "pose_estimates.json")
    parser.add_argument("--output-dir", type=Path, default=config.output_dir / "consensus")
    args = parser.parse_args()

    payload = json.loads(args.pose_file.read_text())
    selected = select_best_pnp_poses(payload["poses"])

    # Query COLMAP poses are in their own local SfM frame, before map alignment.
    query_model = load_colmap_model(config.query_sparse_dir, config.query_images_dir)
    data = SessionData(query_model.frames)
    timestamps, trajectory = load_query_rig_trajectory(data, config.rig_camera_name)
    index_by_timestamp = {timestamp: i for i, timestamp in enumerate(timestamps)}

    # Index the complete trajectory at timestamps with a successful PnP estimate.
    anchor_indices = np.asarray(
        [index_by_timestamp[row["query_timestamp"]] for row in selected],
        dtype=int,
    )

    # Rows stay paired: source[i] is query-local; target[i] is in the normalized
    # reference map. Consensus will fit subsets because some PnP poses are outliers.
    source = trajectory[anchor_indices]
    target = np.stack([np.asarray(row["rig_c2w"], dtype=np.float64) for row in selected])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "pnp_selected_anchors.json"
    output_path.write_text(
        json.dumps(
            {
                "pose_frame": payload["pose_frame"],
                "source": str(args.pose_file),
                "poses": selected,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Selected {len(selected)} anchors from {len(payload['poses'])} PnP hypotheses")
    print(f"Trajectory: {trajectory.shape}; paired anchors: {source.shape} -> {target.shape}")
    print(f"Saved {output_path}")

    # Pairwise search finds the supported alignment; the weighted refit uses
    # all its inliers while giving each occupied query-path cell equal weight.
    seed, seed_mask = initial_consensus(
        source,
        target,
        trajectory,
        translation_threshold=config.consensus_translation,
        rotation_threshold_deg=config.consensus_rotation_deg,
        cell_fraction=config.spatial_cell_fraction,
    )
    transform, mask = refine_consensus(
        source,
        target,
        trajectory,
        seed,
        seed_mask,
        translation_threshold=config.consensus_translation,
        rotation_threshold_deg=config.consensus_rotation_deg,
        cell_fraction=config.spatial_cell_fraction,
    )
    quality = consensus_quality(
        source,
        target,
        trajectory,
        transform,
        mask,
        cell_fraction=config.spatial_cell_fraction,
        stable_min_anchors=config.stable_min_anchors,
        stable_min_cells=config.stable_min_cells,
        stable_span_fraction=config.stable_span_fraction,
        stable_jackknife_p90=config.stable_jackknife_p90,
    )

    # The mask indexes selected in exactly the same order as source and target.
    inlier_rows = [row for row, keep in zip(selected, mask, strict=True) if keep]
    consensus_path = args.output_dir / "consensus.json"
    consensus_path.write_text(
        json.dumps(
            {
                "pose_frame": payload["pose_frame"],
                "transform": transform.tolist(),
                "selected_rows": selected,
                "inlier_mask": mask.tolist(),
                "inlier_rows": inlier_rows,
                "quality": quality,
            },
            indent=2,
        )
        + "\n"
    )

    # Sim(3) scales camera centers, but camera rotations must remain rigid.
    aligned_trajectory = transform_c2w(transform, trajectory)
    trajectory_path = args.output_dir / "consensus_trajectory.json"
    trajectory_path.write_text(
        json.dumps(
            {timestamp: pose.tolist() for timestamp, pose in zip(timestamps, aligned_trajectory, strict=True)},
            indent=2,
        )
        + "\n"
    )
    print(f"Consensus: {len(inlier_rows)} inliers from {len(selected)} selected anchors; stable={quality['stable']}")
    print(f"Saved {consensus_path} and {trajectory_path}")


if __name__ == "__main__":
    main()
