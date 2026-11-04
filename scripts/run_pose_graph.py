"""Run fixed-scale pose graph optimization with all PnP and consensus anchors."""

import argparse
import json
from pathlib import Path

import numpy as np

from configs import localization_exp as config
from sparse_anchors.localization.pose_graph import optimize_pose_graph, sigmas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consensus-dir", type=Path, default=config.output_dir / "consensus")
    args = parser.parse_args()

    consensus_dir = args.consensus_dir
    consensus = json.loads((consensus_dir / "consensus.json").read_text())
    trajectory = json.loads((consensus_dir / "consensus_trajectory.json").read_text())
    timestamps = list(trajectory)
    base = np.asarray(list(trajectory.values()), dtype=np.float64)
    index_by_timestamp = {timestamp: i for i, timestamp in enumerate(timestamps)}

    relative_sigmas = sigmas(config.pgo_relative_translation_sigma, config.pgo_relative_rotation_deg)
    anchor_sigmas = sigmas(config.pgo_anchor_translation_sigma, config.pgo_anchor_rotation_deg)
    prior_sigmas = sigmas(config.pgo_prior_translation_sigma, config.pgo_prior_rotation_deg)

    solutions = {}
    for name, rows in (("pnp_all", consensus["selected_rows"]), ("consensus", consensus["inlier_rows"])):
        anchor_indices = np.asarray([index_by_timestamp[row["query_timestamp"]] for row in rows], dtype=int)
        anchor_poses = np.asarray([row["rig_c2w"] for row in rows], dtype=np.float64)
        optimized, info = optimize_pose_graph(
            base, anchor_indices, anchor_poses, relative_sigmas, anchor_sigmas, prior_sigmas
        )
        assert info["success"], info["message"]
        solutions[name] = (len(rows), optimized, info)

    metrics = {
        "pose_frame": consensus["pose_frame"],
        "relative_sigmas": relative_sigmas.tolist(),
        "anchor_sigmas": anchor_sigmas.tolist(),
        "prior_sigmas": prior_sigmas.tolist(),
    }
    for name, (num_anchors, optimized, info) in solutions.items():
        output_path = consensus_dir / f"{name}_pgo_trajectory.json"
        output_path.write_text(
            json.dumps(
                {timestamp: pose.tolist() for timestamp, pose in zip(timestamps, optimized, strict=True)}, indent=2
            )
            + "\n"
        )
        metrics[name] = {"anchors": num_anchors, **info}
        print(f"{name}: {num_anchors} anchors, cost={info['cost']:.6f}; saved {output_path}")

    metrics_path = consensus_dir / "pose_graph_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved {metrics_path}")


if __name__ == "__main__":
    main()
