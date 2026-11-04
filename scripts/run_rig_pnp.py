from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from vismatch import get_matcher

from configs import localization_exp as config
from sparse_anchors.data.colmap import load_colmap_model
from sparse_anchors.data.sessions import SessionData, frame_camera_name
from sparse_anchors.localization.pnp import (
    estimate_pose_generalized_pnp,
    lift_reference_matches,
    render_reference_depth,
)
from sparse_anchors.localization.vpr import (
    describe_image_groups,
    far_margin_scores,
    load_megaloc,
    retrieve,
    top_diverse_image_pairs,
)
from sparse_anchors.splatting.scene import load_scene, normalize_scene
from sparse_anchors.utils.io import load_splat_checkpoint


def main():
    device = torch.device(config.device)

    reference_scene = normalize_scene(
        load_scene(
            sparse_dir=config.reference_sparse_dir,
            images_dir=config.reference_images_dir,
            feature_manifest_path=config.feature_manifest,
            test_every=config.reference_test_every,
        )
    )

    reference_data = SessionData(reference_scene.colmap.frames)

    query_data = SessionData(
        load_colmap_model(sparse_dir=config.query_sparse_dir, images_dir=config.query_images_dir).frames
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_splat_checkpoint(config.checkpoint, device)
    model, sh_degree, packed = checkpoint.model, checkpoint.sh_degree, checkpoint.packed

    query_vpr_cache = config.output_dir / "cache" / "vpr" / "query.npy"
    reference_vpr_cache = config.output_dir / "cache" / "vpr" / "reference.npy"
    match_cache_dir = config.output_dir / "cache" / "matches"
    match_cache_dir.mkdir(parents=True, exist_ok=True)
    vpr_model = None
    if not query_vpr_cache.exists() or not reference_vpr_cache.exists():
        vpr_model = load_megaloc(device)
    matcher = None
    query_groups = image_groups_from_session(query_data)
    reference_groups = image_groups_from_session(reference_data)
    scale = float(np.linalg.norm(reference_scene.world_transform[:3, 0]))
    camera_from_rig_by_camera = build_rig_extrinsics(
        query_data, rig_camera_name=config.rig_camera_name, translation_scale=scale
    )
    K_by_camera, image_size_by_camera = camera_metadata_by_name(query_data)
    query_uniqueness_masks = load_static_uniqueness_masks(config.query_root / "masks")
    reference_uniqueness_masks = load_static_uniqueness_masks(config.reference_root / "masks")

    query_vpr = describe_image_groups(
        model=vpr_model,
        image_groups=query_groups,
        device=device,
        batch_size=config.vpr_batch_size,
        image_size=config.vpr_image_size,
        desc="describing query image",
        cache_path=query_vpr_cache,
        use_cache=query_vpr_cache.exists(),
    )

    reference_vpr = describe_image_groups(
        model=vpr_model,
        image_groups=reference_groups,
        device=device,
        batch_size=config.vpr_batch_size,
        image_size=config.vpr_image_size,
        desc="describing ref image",
        cache_path=reference_vpr_cache,
        use_cache=reference_vpr_cache.exists(),
    )
    del vpr_model

    rig_centers_by_timestamp = {
        timestamp: frame.c2w[:3, 3]
        for timestamp, frames in reference_data.frames_by_timestamp.items()
        for frame in frames
        if frame_camera_name(frame) == config.rig_camera_name
    }
    reference_centers = np.stack([rig_centers_by_timestamp[timestamp] for timestamp in reference_vpr.timestamps])
    reference_diameter = np.linalg.norm(reference_centers[:, None] - reference_centers[None], axis=-1).max()
    far_radius = config.vpr_far_radius_fraction * reference_diameter
    margins = far_margin_scores(
        query_vpr.timestamp_descriptors,
        reference_vpr.timestamp_descriptors,
        reference_centers,
        radius=far_radius,
    )
    margin_by_query = dict(zip(query_vpr.timestamps, margins, strict=True))

    retrievals = retrieve(
        query_vpr.timestamps,
        query_vpr.timestamp_descriptors,
        reference_vpr.timestamps,
        reference_vpr.timestamp_descriptors,
        top_k=config.vpr_top_k,
    )
    selected_retrievals = []
    for retrieval in retrievals:
        margin = margin_by_query[retrieval["query_timestamp"]]
        if np.isfinite(margin) and margin >= config.vpr_far_margin_threshold:
            selected_retrievals.append({**retrieval, "far_margin": float(margin)})
    retrievals = selected_retrievals
    kept_query_count = len({row["query_timestamp"] for row in retrievals})

    print(
        f"VPR retained {kept_query_count}/{len(query_vpr.timestamps)} query timestamps "
        f"with rank-1 far margin >= {config.vpr_far_margin_threshold} "
        f"(radius {far_radius:.3f} SfM units, {config.vpr_far_radius_fraction:.0%} of reference diameter); "
        f"{len(retrievals)} retrieval candidates ({config.vpr_top_k} per retained query)"
    )
    (config.output_dir / "retrievals.json").write_text(json.dumps(retrievals, indent=2) + "\n")
    match_rows = []
    pose_rows = []
    eligible_match_count = 0
    depth_by_reference_image = {}

    for retrieval in retrievals:
        query_timestamp = str(retrieval["query_timestamp"])
        reference_timestamp = str(retrieval["reference_timestamp"])

        query_frames = query_data.frames_by_timestamp[query_timestamp]
        reference_frames = reference_data.frames_by_timestamp[reference_timestamp]

        selected_pairs = top_diverse_image_pairs(
            query_vpr.image_descriptors_by_timestamp[query_timestamp],
            reference_vpr.image_descriptors_by_timestamp[reference_timestamp],
            top_k=config.image_pair_top_k,
        )
        if not selected_pairs:
            continue

        query_frames_by_camera = {frame_camera_name(frame): frame for frame in query_frames}
        rig_query_frame = query_frames_by_camera.get(config.rig_camera_name, query_frames[0])
        _, primary_reference_index, image_score = selected_pairs[0]
        primary_reference_frame = reference_frames[primary_reference_index]

        points_2d_by_camera = defaultdict(list)
        points_3d_by_camera = defaultdict(list)
        pair_rows = []
        raw_match_inliers = 0
        masked_correspondences = 0
        depth_correspondences = 0

        for pair_rank, (query_index, reference_index, pair_score) in enumerate(selected_pairs, start=1):
            query_frame = query_frames[query_index]
            reference_frame = reference_frames[reference_index]
            query_camera = frame_camera_name(query_frame)
            reference_camera = frame_camera_name(reference_frame)

            key = hashlib.sha1(
                f"{config.matcher_name}|4096|{query_frame.image_name}|{reference_frame.image_name}".encode()
            ).hexdigest()
            cache_path = match_cache_dir / f"{key}.npz"
            if cache_path.exists():
                with np.load(cache_path) as cached:
                    query_points = np.asarray(cached["points_query"], dtype=np.float32)
                    reference_points = np.asarray(cached["points_reference"], dtype=np.float32)
            else:
                if matcher is None:
                    matcher = get_matcher(config.matcher_name, device=config.device, max_num_keypoints=4096)
                    matcher.skip_ransac = True
                matches = matcher(query_frame.image_path, reference_frame.image_path)
                query_points = np.asarray(matches["matched_kpts0"], dtype=np.float32)
                reference_points = np.asarray(matches["matched_kpts1"], dtype=np.float32)
                np.savez(cache_path, points_query=query_points, points_reference=reference_points)
            raw_num_inliers = len(query_points)
            raw_match_inliers += raw_num_inliers

            query_points, reference_points = filter_unique_correspondences(
                query_points,
                reference_points,
                query_uniqueness_masks.get(query_camera),
                reference_uniqueness_masks.get(reference_camera),
            )
            masked_count = len(query_points) if raw_num_inliers >= config.pair_match_min_inliers else 0
            masked_correspondences += masked_count

            pair_row = {
                "rank": pair_rank,
                "query_image": query_frame.image_name,
                "reference_image": reference_frame.image_name,
                "query_camera": query_camera,
                "reference_camera": reference_camera,
                "image_score": pair_score,
                "raw_num_inliers": raw_num_inliers,
                "masked_num_inliers": masked_count,
                "depth_num_inliers": 0,
            }
            pair_rows.append(pair_row)

            if masked_count == 0:
                continue

            if reference_frame.image_name not in depth_by_reference_image:
                depth_by_reference_image[reference_frame.image_name] = render_reference_depth(
                    model, reference_frame, sh_degree=sh_degree, packed=packed
                )
            reference_depth = depth_by_reference_image[reference_frame.image_name]
            query_points, world_points = lift_reference_matches(
                query_points,
                reference_points,
                reference_depth=reference_depth,
                reference_K=reference_frame.K,
                reference_c2w=reference_frame.c2w,
            )
            pair_row["depth_num_inliers"] = len(query_points)
            depth_correspondences += len(query_points)

            if len(query_points) == 0:
                continue

            points_2d_by_camera[query_camera].append(query_points)
            points_3d_by_camera[query_camera].append(world_points)

        points_2d_by_camera = {camera: np.concatenate(values, axis=0) for camera, values in points_2d_by_camera.items()}
        points_3d_by_camera = {camera: np.concatenate(values, axis=0) for camera, values in points_3d_by_camera.items()}

        match_row = {
            "query_timestamp": query_timestamp,
            "reference_timestamp": reference_timestamp,
            "query_image": rig_query_frame.image_name,
            "reference_image": primary_reference_frame.image_name,
            "vpr_score": retrieval["score"],
            "vpr_rank": retrieval["rank"],
            "top1_far_margin": retrieval["far_margin"],
            "image_score": image_score,
            "num_pairs": len(selected_pairs),
            "raw_num_inliers": raw_match_inliers,
            "masked_num_inliers": masked_correspondences,
            "num_inliers": depth_correspondences,
            "pairs": pair_rows,
        }

        match_rows.append(match_row)

        if depth_correspondences < config.match_min_inliers:
            print(
                f"{query_timestamp} -> {reference_timestamp}: "
                f"{len(selected_pairs)} pairs, "
                f"{depth_correspondences} depth-backed correspondences, skipped"
            )
            continue
        eligible_match_count += 1

        estimate = estimate_pose_generalized_pnp(
            points_2d_by_camera=points_2d_by_camera,
            points_3d_by_camera=points_3d_by_camera,
            camera_from_rig_by_camera=camera_from_rig_by_camera,
            K_by_camera=K_by_camera,
            image_size_by_camera=image_size_by_camera,
            config=config.pnp,
        )

        if estimate is None:
            print(
                f"{query_timestamp} -> {reference_timestamp}: "
                f"{depth_correspondences} depth-backed correspondences, rig_pnp=no"
            )
            continue

        virtual_pose_rows = virtual_camera_pose_rows(
            rig_c2w=estimate.c2w,
            query_frames=query_frames,
            camera_from_rig_by_camera=camera_from_rig_by_camera,
        )

        pose_rows.append(
            {
                **match_row,
                "sfm_c2w": estimate.c2w,
                "sfm_w2c": estimate.w2c,
                "virtual_camera_poses": virtual_pose_rows,
                "rig_pnp_num_inliers": len(estimate.inlier_indices),
                "reprojection_error_px": estimate.reprojection_error_px,
            }
        )

        print(
            f"{query_timestamp} -> {reference_timestamp}: "
            f"{len(selected_pairs)} pairs, "
            f"{depth_correspondences} depth-backed correspondences, "
            f"rig_pnp=yes, inliers={len(estimate.inlier_indices)}, "
            f"reprojection={estimate.reprojection_error_px:.2f}px"
        )

    print(f"PnP succeeded for {len(pose_rows)}/{eligible_match_count} match-qualified image pairs")
    (config.output_dir / "match_rows.json").write_text(json.dumps(match_rows, indent=2) + "\n")
    pose_estimates_path = write_pose_estimates(pose_rows, config.output_dir, scale)
    print(f"saved pose estimates to {pose_estimates_path}")


def image_groups_from_session(
    data: SessionData,
) -> dict[str, tuple[Path, ...]]:
    """Return timestamp groups in the same order as SessionData frames."""
    return {
        timestamp: tuple(frame.image_path for frame in frames) for timestamp, frames in data.frames_by_timestamp.items()
    }


def build_rig_extrinsics(data: SessionData, rig_camera_name: str, translation_scale: float) -> dict[str, np.ndarray]:
    """Build fixed camera_from_rig extrinsics from one complete timestamp group."""
    for frames in data.frames_by_timestamp.values():
        frames_by_camera = {frame_camera_name(frame): frame for frame in frames}
        rig_frame = frames_by_camera.get(rig_camera_name)
        if rig_frame is None:
            continue

        camera_from_rig = {}
        for camera_name, frame in frames_by_camera.items():
            extrinsic = frame.w2c @ rig_frame.c2w
            extrinsic[:3, 3] *= translation_scale
            camera_from_rig[camera_name] = extrinsic
        return camera_from_rig


def camera_metadata_by_name(
    data: SessionData,
) -> tuple[dict[str, np.ndarray], dict[str, tuple[int, int]]]:
    K_by_camera = {}
    image_size_by_camera = {}
    for frame in data.frames:
        camera_name = frame_camera_name(frame)
        K_by_camera.setdefault(camera_name, frame.K)
        image_size_by_camera.setdefault(camera_name, (frame.width, frame.height))
    return K_by_camera, image_size_by_camera


def load_static_uniqueness_masks(mask_dir: Path) -> dict[str, np.ndarray]:
    masks = {}
    if not mask_dir.exists():
        return masks

    for mask_path in sorted(mask_dir.glob("*.png")):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        masks[mask_path.stem] = mask > 0
    return masks


def filter_unique_correspondences(
    query_points: np.ndarray,
    reference_points: np.ndarray,
    query_mask: np.ndarray | None,
    reference_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    query_points = np.asarray(query_points, dtype=np.float32).reshape(-1, 2)
    reference_points = np.asarray(reference_points, dtype=np.float32).reshape(-1, 2)

    keep = np.ones(len(query_points), dtype=bool)
    if query_mask is not None:
        keep &= sample_mask_nearest(query_mask, query_points)
    if reference_mask is not None:
        keep &= sample_mask_nearest(reference_mask, reference_points)
    return query_points[keep], reference_points[keep]


def sample_mask_nearest(mask: np.ndarray, points: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    height, width = mask.shape
    x = np.rint(points[:, 0]).astype(np.int64)
    y = np.rint(points[:, 1]).astype(np.int64)
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    valid = np.zeros(len(points), dtype=bool)
    valid[inside] = mask[y[inside], x[inside]]
    return valid


def virtual_camera_pose_rows(
    rig_c2w: np.ndarray,
    query_frames: tuple,
    camera_from_rig_by_camera: dict[str, np.ndarray],
) -> list[dict]:
    rows = []
    for frame in query_frames:
        camera_name = frame_camera_name(frame)
        rig_from_camera = np.linalg.inv(camera_from_rig_by_camera[camera_name])
        c2w = rig_c2w @ rig_from_camera
        rows.append(
            {
                "image_name": frame.image_name,
                "camera_name": camera_name,
                "c2w": c2w,
                "w2c": np.linalg.inv(c2w),
            }
        )
    return rows


def write_pose_estimates(
    pose_rows: list[dict],
    output_dir: Path,
    rig_extrinsic_translation_scale: float,
) -> Path:
    path = output_dir / "pose_estimates.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    poses = []
    for row in pose_rows:
        poses.append(
            {
                "query_timestamp": row["query_timestamp"],
                "reference_timestamp": row["reference_timestamp"],
                "rig_camera": config.rig_camera_name,
                "query_rig_image": row["query_image"],
                "reference_image": row["reference_image"],
                "vpr_score": float(row["vpr_score"]),
                "vpr_rank": int(row["vpr_rank"]),
                "top1_far_margin": float(row["top1_far_margin"]),
                "image_score": float(row["image_score"]),
                "num_pairs": int(row["num_pairs"]),
                "raw_num_inliers": int(row["raw_num_inliers"]),
                "masked_num_inliers": int(row["masked_num_inliers"]),
                "depth_num_inliers": int(row["num_inliers"]),
                "rig_pnp_num_inliers": int(row["rig_pnp_num_inliers"]),
                "reprojection_error_px": float(row["reprojection_error_px"]),
                "rig_w2c": np.asarray(row["sfm_w2c"], dtype=np.float64).tolist(),
                "rig_c2w": np.asarray(row["sfm_c2w"], dtype=np.float64).tolist(),
                "w2c": np.asarray(row["sfm_w2c"], dtype=np.float64).tolist(),
                "c2w": np.asarray(row["sfm_c2w"], dtype=np.float64).tolist(),
                "pairs": row["pairs"],
                "virtual_camera_poses": [
                    {
                        "image_name": pose["image_name"],
                        "camera_name": pose["camera_name"],
                        "w2c": np.asarray(pose["w2c"], dtype=np.float64).tolist(),
                        "c2w": np.asarray(pose["c2w"], dtype=np.float64).tolist(),
                    }
                    for pose in row["virtual_camera_poses"]
                ],
            }
        )

    payload = {
        "pose_frame": "normalized_reference_sfm",
        "rig_extrinsic_translation_scale": rig_extrinsic_translation_scale,
        "poses": poses,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


if __name__ == "__main__":
    main()
