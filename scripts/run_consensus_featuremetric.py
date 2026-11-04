"""Refine consensus PnP anchors against the Gaussian map, then run the final graph."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from configs import localization_exp as config
from sparse_anchors.data.colmap import load_colmap_model
from sparse_anchors.data.sessions import SessionData
from sparse_anchors.features.dinov3 import DinoV3FeatureExtractor, hf_model_from_manifest
from sparse_anchors.localization.featuremetric import RigPose, build_rig_views, refine_rig_pose_bounded
from sparse_anchors.localization.pose_graph import optimize_pose_graph, sigmas
from sparse_anchors.utils.io import load_splat_checkpoint


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consensus-dir", type=Path, default=config.output_dir / "consensus")
    parser.add_argument("--output-dir", type=Path, default=config.output_dir / "featuremetric")
    parser.add_argument("--max-anchors", type=int, help="Limit refinements for a small test run")
    args = parser.parse_args()

    consensus = json.loads((args.consensus_dir / "consensus.json").read_text())
    base_trajectory = json.loads((args.consensus_dir / "consensus_trajectory.json").read_text())
    pnp_trajectory = json.loads((args.consensus_dir / "consensus_pgo_trajectory.json").read_text())
    timestamps = list(base_trajectory)
    index = {timestamp: i for i, timestamp in enumerate(timestamps)}
    base = np.asarray(list(base_trajectory.values()), dtype=np.float64)
    pnp_graph = np.asarray([pnp_trajectory[timestamp] for timestamp in timestamps], dtype=np.float64)
    pnp_rows = consensus["inlier_rows"]
    fm_rows = pnp_rows[: args.max_anchors] if args.max_anchors is not None else pnp_rows

    device = torch.device(config.device)
    checkpoint = load_splat_checkpoint(config.checkpoint, device=device)
    extractor = DinoV3FeatureExtractor(device, hf_model_from_manifest(checkpoint.feature_manifest["extractor"]))
    query = SessionData(load_colmap_model(config.query_sparse_dir, config.query_images_dir).frames)

    args.output_dir.mkdir(parents=True)
    refinements = []
    for position, row in enumerate(fm_rows):
        timestamp = row["query_timestamp"]
        rig = RigPose(
            position,
            timestamp,
            row["reference_timestamp"],
            np.asarray(row["rig_w2c"], dtype=np.float64),
            tuple(row["virtual_camera_poses"]),
        )
        views = build_rig_views(
            rig,
            query,
            extractor,
            {},
            config.query_root / "masks",
            config.featuremetric_long_side,
            config.featuremetric_mask_dilation_pixels,
            device,
        )
        initial = pnp_graph[index[timestamp]]
        final, info = refine_rig_pose_bounded(
            checkpoint,
            views,
            initial,
            config.featuremetric_steps,
            config.featuremetric_learning_rate,
            config.featuremetric_alpha,
            config.featuremetric_max_translation,
            config.featuremetric_max_rotation_deg,
            config.featuremetric_prior_weight,
        )
        refinements.append(
            {"query_timestamp": timestamp, "initial_c2w": initial.tolist(), "final_c2w": final.tolist(), **info}
        )
        write_json(args.output_dir / "refinements.json", refinements)
        print(
            f"{position + 1}/{len(fm_rows)} {timestamp}: {info['initial_feature_loss']:.6f} -> {info['final_feature_loss']:.6f}",
            flush=True,
        )

    relative_sigmas = sigmas(config.pgo_relative_translation_sigma, config.pgo_relative_rotation_deg)
    anchor_sigmas = sigmas(config.pgo_anchor_translation_sigma, config.pgo_anchor_rotation_deg)
    prior_sigmas = sigmas(config.pgo_prior_translation_sigma, config.pgo_prior_rotation_deg)
    anchor_indices = np.asarray([index[row["query_timestamp"]] for row in pnp_rows + refinements], dtype=int)
    anchor_poses = np.concatenate(
        [np.asarray([row["rig_c2w"] for row in pnp_rows]), np.asarray([row["final_c2w"] for row in refinements])]
    )
    final_graph, graph_info = optimize_pose_graph(
        base, anchor_indices, anchor_poses, relative_sigmas, anchor_sigmas, prior_sigmas, initial_poses=pnp_graph
    )
    assert graph_info["success"], graph_info["message"]

    write_json(
        args.output_dir / "trajectory.json", {t: p.tolist() for t, p in zip(timestamps, final_graph, strict=True)}
    )
    write_json(
        args.output_dir / "summary.json",
        {
            "pose_frame": consensus["pose_frame"],
            "pnp_anchors": len(pnp_rows),
            "featuremetric_anchors": len(refinements),
            "graph": graph_info,
        },
    )
    print(f"Final graph: {len(anchor_indices)} measurements, cost={graph_info['cost']:.6f}")


if __name__ == "__main__":
    main()
