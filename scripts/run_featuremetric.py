from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from gsplat.losses import ssim_loss

from configs import localization_exp as config
from sparse_anchors.data.colmap import load_colmap_model
from sparse_anchors.data.sessions import SessionData, load_gt_poses
from sparse_anchors.features.base import ImageFeatureExtractor, extract_image_features
from sparse_anchors.features.dinov3 import DinoV3FeatureExtractor, hf_model_from_manifest
from sparse_anchors.localization.featuremetric import (
    featuremetric_loss,
    refine_pose_featuremetric,
)
from sparse_anchors.localization.types import FeatureMetricRefinement
from sparse_anchors.splatting.features import decode_feature_map, load_feature_map
from sparse_anchors.splatting.renderer import ViewCamera, render
from sparse_anchors.splatting.scene import load_scene, normalize_scene
from sparse_anchors.utils.evaluation import estimate_gt_from_reference_sfm
from sparse_anchors.utils.geometry import compose_pose_update, transform_c2w
from sparse_anchors.utils.image import load_mask, load_rgb, write_rgb
from sparse_anchors.utils.io import load_splat_checkpoint
from sparse_anchors.utils.visualization import (
    diagonal_rgb_composite,
    fit_pca_projection,
    pca_rgb,
    resize_rgb,
    side_by_side,
)


@dataclass(frozen=True)
class SavedPose:
    index: int
    query_image: str
    reference_image: str
    w2c: np.ndarray


@dataclass(frozen=True)
class PoseError:
    translation: float
    rotation_deg: float


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)

    poses = load_saved_poses(args.pose_estimates)
    if not args.all_poses:
        poses = [poses[args.pose_index]]
    if args.max_poses is not None:
        poses = poses[: args.max_poses]

    reference_scene = normalize_scene(
        load_scene(
            sparse_dir=args.reference_sparse_dir,
            images_dir=args.reference_images_dir,
            feature_manifest_path=args.feature_manifest,
            test_every=args.reference_test_every,
        )
    )
    reference_data = SessionData(reference_scene.colmap.frames)
    query_data = SessionData(
        load_colmap_model(
            sparse_dir=args.query_sparse_dir,
            images_dir=args.query_images_dir,
        ).frames
    )
    checkpoint = load_splat_checkpoint(args.checkpoint, device=device)
    query_extractor = (
        DinoV3FeatureExtractor(
            device=device,
            hf_model=args.dinov3_hf_model or hf_model_from_manifest(reference_scene.features.extractor),
        )
        if args.target == "query"
        else None
    )
    gt_context = None
    if args.target == "query" and args.reference_frames_csv is not None and args.query_frames_csv is not None:
        reference_gt_poses, _ = load_gt_poses(args.reference_frames_csv)
        query_gt_poses, _ = load_gt_poses(args.query_frames_csv)
        gt_context = (
            estimate_gt_from_reference_sfm(reference_data, reference_gt_poses),
            query_gt_poses,
        )
    diagonal_output_dir = args.diagonal_output_dir
    if diagonal_output_dir is None:
        suffix = "featuremetric_query_diagonal" if args.target == "query" else "featuremetric_diagonal"
        diagonal_output_dir = config.output_dir / suffix

    if args.target == "query":
        rows = run_query_poses(
            args=args,
            poses=poses,
            query_data=query_data,
            checkpoint=checkpoint,
            feature_extractor=query_extractor,
            device=device,
            diagonal_output_dir=diagonal_output_dir,
            gt_context=gt_context,
        )
        write_json(args, diagonal_output_dir, rows)
        return

    rows: list[dict] = []
    for pose in poses:
        frame = reference_data.frames_by_name[pose.reference_image]
        image_rgb = load_rgb(frame.image_path)
        target_features = torch.from_numpy(load_feature_map(reference_scene.features, frame.image_name)).to(device)
        initial_w2c = perturb_w2c(
            frame.w2c,
            translation=args.perturb_translation,
            yaw_degrees=args.perturb_yaw_degrees,
        )
        initial_error = pose_error(initial_w2c, frame.w2c)

        refinement = refine_pose_featuremetric(
            model=checkpoint.model,
            decoder=checkpoint.decoder,
            camera=ViewCamera(
                w2c=initial_w2c,
                K=frame.K,
                width=frame.width,
                height=frame.height,
            ),
            target_features=target_features,
            sh_degree=checkpoint.sh_degree,
            packed=checkpoint.packed,
            alpha_threshold=args.feature_alpha_threshold,
            steps=args.steps,
            learning_rate=args.learning_rate,
            freeze_alpha_mask=not args.dynamic_alpha_mask,
            target_valid_mask=None,
        )

        final_error = pose_error(refinement.final_w2c.numpy(), frame.w2c)
        before_diagonal, after_diagonal, rgb_before_after, feature_before_after = write_diagonal_images(
            output_dir=diagonal_output_dir,
            pose_index=pose.index,
            image_name=frame.image_name,
            image_rgb=image_rgb,
            K=frame.K,
            initial_w2c=initial_w2c,
            final_w2c=refinement.final_w2c.numpy(),
            model=checkpoint.model,
            decoder=checkpoint.decoder,
            target_features=target_features,
            target_valid_mask=None,
            sh_degree=checkpoint.sh_degree,
            packed=checkpoint.packed,
        )
        rows.append(
            {
                "pose_index": pose.index,
                "query_image": pose.query_image,
                "reference_image": pose.reference_image,
                "initial_feature_loss": refinement.initial_loss,
                "final_feature_loss": refinement.final_loss,
                "initial_translation_error": initial_error.translation,
                "final_translation_error": final_error.translation,
                "initial_rotation_error_deg": initial_error.rotation_deg,
                "final_rotation_error_deg": final_error.rotation_deg,
                "best_iteration": refinement.best_iteration,
                "xi": refinement.xi.tolist(),
                "initial_w2c": refinement.initial_w2c.tolist(),
                "final_w2c": refinement.final_w2c.tolist(),
                "before_diagonal": str(before_diagonal),
                "after_diagonal": str(after_diagonal),
                "rgb_before_after": str(rgb_before_after),
                "feature_before_after": None if feature_before_after is None else str(feature_before_after),
                "losses": list(refinement.losses),
            }
        )
        print_refinement_row(rows[-1])

    write_json(args, diagonal_output_dir, rows)


def run_query_poses(
    args: argparse.Namespace,
    poses: list[SavedPose],
    query_data: SessionData,
    checkpoint,
    feature_extractor: ImageFeatureExtractor,
    device: torch.device,
    diagonal_output_dir: Path,
    gt_context: tuple[np.ndarray, dict[str, np.ndarray]] | None,
) -> list[dict]:
    rows = []
    for pose in poses:
        frame = query_data.frames_by_name[pose.query_image]
        image_rgb = load_rgb(frame.image_path)
        valid_mask = None
        if not args.disable_query_mask and args.query_masks_dir is not None:
            valid_mask = load_mask(args.query_masks_dir / Path(frame.image_name).with_suffix(".png.png"))
        feature_values, feature_mask = extract_image_features(
            feature_extractor=feature_extractor,
            image_rgb=image_rgb,
            long_side=args.feature_long_side,
            valid_mask=valid_mask,
            mask_dilation_pixels=args.mask_dilation_pixels,
        )
        target_features = torch.from_numpy(feature_values).to(device)
        target_valid_mask = None if feature_mask is None else torch.from_numpy(feature_mask).to(device)
        camera = ViewCamera(
            w2c=perturb_w2c(
                pose.w2c,
                translation=args.perturb_translation,
                yaw_degrees=args.perturb_yaw_degrees,
            )
            if args.perturb_initial_pose
            else pose.w2c,
            K=frame.K,
            width=frame.width,
            height=frame.height,
        )
        if args.objective == "feature":
            refinement = refine_pose_featuremetric(
                model=checkpoint.model,
                decoder=checkpoint.decoder,
                camera=camera,
                target_features=target_features,
                sh_degree=checkpoint.sh_degree,
                packed=checkpoint.packed,
                alpha_threshold=args.feature_alpha_threshold,
                steps=args.steps,
                learning_rate=args.learning_rate,
                freeze_alpha_mask=not args.dynamic_alpha_mask,
                target_valid_mask=target_valid_mask,
            )
        else:
            refinement = refine_pose_featuremetric_rgb(
                model=checkpoint.model,
                decoder=checkpoint.decoder,
                camera=camera,
                target_features=target_features,
                image_rgb=image_rgb,
                valid_rgb_mask=load_query_rgb_mask(
                    image_name=frame.image_name,
                    image_size=image_rgb.shape[:2],
                    masks_dir=None if args.disable_query_mask else args.query_masks_dir,
                ),
                sh_degree=checkpoint.sh_degree,
                packed=checkpoint.packed,
                alpha_threshold=args.feature_alpha_threshold,
                steps=args.steps,
                learning_rate=args.learning_rate,
                freeze_alpha_mask=not args.dynamic_alpha_mask,
                target_valid_mask=target_valid_mask,
                feature_loss_weight=args.feature_loss_weight,
                rgb_loss_weight=args.rgb_loss_weight,
                ssim_lambda=args.ssim_lambda,
            )
        initial_error = None
        final_error = None
        if gt_context is not None:
            gt_from_reference_sfm, query_gt_poses = gt_context
            gt_c2w = query_gt_poses.get(frame.image_name)
            initial_error = query_pose_error(np.asarray(camera.w2c), gt_c2w, gt_from_reference_sfm)
            final_error = query_pose_error(refinement.final_w2c.numpy(), gt_c2w, gt_from_reference_sfm)

        before_diagonal = after_diagonal = rgb_before_after = feature_before_after = None
        if not args.skip_diagonals:
            before_diagonal, after_diagonal, rgb_before_after, feature_before_after = write_diagonal_images(
                output_dir=diagonal_output_dir,
                pose_index=pose.index,
                image_name=frame.image_name,
                image_rgb=image_rgb,
                K=frame.K,
                initial_w2c=np.asarray(camera.w2c),
                final_w2c=refinement.final_w2c.numpy(),
                model=checkpoint.model,
                decoder=checkpoint.decoder,
                target_features=target_features,
                target_valid_mask=target_valid_mask,
                sh_degree=checkpoint.sh_degree,
                packed=checkpoint.packed,
            )
        row = {
            "pose_index": pose.index,
            "query_image": pose.query_image,
            "reference_image": pose.reference_image,
            "initial_feature_loss": refinement.initial_loss,
            "final_feature_loss": refinement.final_loss,
            "initial_translation_error": None if initial_error is None else initial_error.translation,
            "final_translation_error": None if final_error is None else final_error.translation,
            "initial_rotation_error_deg": None if initial_error is None else initial_error.rotation_deg,
            "final_rotation_error_deg": None if final_error is None else final_error.rotation_deg,
            "best_iteration": refinement.best_iteration,
            "xi": refinement.xi.tolist(),
            "initial_w2c": refinement.initial_w2c.tolist(),
            "final_w2c": refinement.final_w2c.tolist(),
            "before_diagonal": "" if before_diagonal is None else str(before_diagonal),
            "after_diagonal": "" if after_diagonal is None else str(after_diagonal),
            "rgb_before_after": "" if rgb_before_after is None else str(rgb_before_after),
            "feature_before_after": None if feature_before_after is None else str(feature_before_after),
            "losses": list(refinement.losses),
        }
        rows.append(row)
        print_refinement_row(row)
    return rows


def write_json(
    args: argparse.Namespace,
    diagonal_output_dir: Path,
    rows: list[dict],
) -> None:
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(
                {
                    "settings": {
                        "target": args.target,
                        "objective": args.objective,
                        "steps": args.steps,
                        "learning_rate": args.learning_rate,
                        "feature_loss_weight": args.feature_loss_weight,
                        "rgb_loss_weight": args.rgb_loss_weight,
                        "ssim_lambda": args.ssim_lambda,
                        "feature_alpha_threshold": args.feature_alpha_threshold,
                        "alpha_mask": "dynamic" if args.dynamic_alpha_mask else "initial_fixed",
                        "query_mask": None if args.disable_query_mask else str(args.query_masks_dir),
                        "feature_render_resolution": "native",
                        "perturb_translation": args.perturb_translation,
                        "perturb_yaw_degrees": args.perturb_yaw_degrees,
                        "perturb_initial_pose": args.perturb_initial_pose,
                        "max_poses": args.max_poses,
                        "diagonal_output_dir": str(diagonal_output_dir),
                    },
                    "rows": rows,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"wrote {args.output_json}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refine featuremetric poses and write render-vs-real diagonals.")
    parser.add_argument("--target", choices=("reference", "query"), default="query")
    parser.add_argument("--pose-estimates", type=Path, default=config.output_dir / "pose_estimates.json")
    parser.add_argument("--pose-index", type=int, default=0)
    parser.add_argument("--all-poses", action="store_true")
    parser.add_argument("--max-poses", type=int)
    parser.add_argument("--reference-sparse-dir", type=Path, default=config.reference_sparse_dir)
    parser.add_argument("--reference-images-dir", type=Path, default=config.reference_images_dir)
    parser.add_argument("--reference-frames-csv", type=Path, default=config.reference_frames_csv)
    parser.add_argument("--query-sparse-dir", type=Path, default=config.query_sparse_dir)
    parser.add_argument("--query-images-dir", type=Path, default=config.query_images_dir)
    parser.add_argument("--query-frames-csv", type=Path, default=config.query_frames_csv)
    parser.add_argument("--feature-manifest", type=Path, default=config.feature_manifest)
    parser.add_argument("--checkpoint", type=Path, default=config.checkpoint)
    parser.add_argument("--reference-test-every", type=int, default=config.reference_test_every)
    parser.add_argument("--device", default=config.device)
    parser.add_argument("--feature-long-side", type=int, default=768)
    parser.add_argument("--dinov3-hf-model", default=None)
    parser.add_argument("--feature-alpha-threshold", type=float, default=0.0)
    parser.add_argument("--query-masks-dir", type=Path, default=config.query_root / "colmap_sfm_rig" / "feature_masks")
    parser.add_argument("--mask-dilation-pixels", type=int, default=14)
    parser.add_argument("--disable-query-mask", action="store_true")
    parser.add_argument("--perturb-translation", type=float, default=0.07)
    parser.add_argument("--perturb-yaw-degrees", type=float, default=7.0)
    parser.add_argument("--perturb-initial-pose", action="store_true")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--objective", choices=("feature", "feature_rgb"), default="feature")
    parser.add_argument("--feature-loss-weight", type=float, default=0.0)
    parser.add_argument("--rgb-loss-weight", type=float, default=1)
    parser.add_argument("--ssim-lambda", type=float, default=0.2)
    parser.add_argument("--dynamic-alpha-mask", action="store_true")
    parser.add_argument("--skip-diagonals", action="store_true")
    parser.add_argument("--diagonal-output-dir", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser


def load_saved_poses(path: Path) -> list[SavedPose]:
    payload = json.loads(path.read_text())
    poses = payload["poses"]
    return [
        SavedPose(
            index=index,
            query_image=str(pose["query_rig_image"]),
            reference_image=str(pose["reference_image"]),
            w2c=np.asarray(pose["w2c"], dtype=np.float64),
        )
        for index, pose in enumerate(poses)
    ]


def load_query_rgb_mask(
    image_name: str,
    image_size: tuple[int, int],
    masks_dir: Path | None,
) -> torch.Tensor:
    if masks_dir is None:
        return torch.ones(image_size, dtype=torch.bool)
    mask = load_mask(masks_dir / Path(image_name).with_suffix(".png.png"))
    if mask.shape != image_size:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (image_size[1], image_size[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return torch.from_numpy(mask)


def refine_pose_featuremetric_rgb(
    model,
    decoder,
    camera: ViewCamera,
    target_features: torch.Tensor,
    image_rgb: np.ndarray,
    valid_rgb_mask: torch.Tensor,
    sh_degree: int,
    packed: bool,
    alpha_threshold: float,
    steps: int,
    learning_rate: float,
    freeze_alpha_mask: bool,
    target_valid_mask: torch.Tensor | None,
    feature_loss_weight: float,
    rgb_loss_weight: float,
    ssim_lambda: float,
):
    device = target_features.device
    initial_w2c = torch.as_tensor(camera.w2c, dtype=torch.float32, device=device).detach()
    target_rgb = torch.from_numpy(image_rgb).to(device=device, dtype=torch.float32)
    rgb_mask = valid_rgb_mask.to(device=device, dtype=torch.bool)
    xi = torch.zeros(6, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([xi], lr=learning_rate)

    fixed_valid_mask = None
    if freeze_alpha_mask:
        with torch.no_grad():
            initial_output = render(
                model,
                ViewCamera(w2c=initial_w2c, K=camera.K, width=camera.width, height=camera.height),
                sh_degree=sh_degree,
                packed=packed,
                render_latents=True,
            )
            initial_features = decode_feature_map(
                decoder,
                initial_output.latents,
                output_size=tuple(target_features.shape[:2]),
            )
            _, fixed_valid_mask = featuremetric_loss(
                initial_features,
                target_features,
                initial_output.alpha,
                alpha_threshold,
                target_valid_mask=target_valid_mask,
            )
            fixed_valid_mask = fixed_valid_mask.detach()

    losses: list[float] = []
    best_loss = float("inf")
    best_xi = xi.detach().clone()
    best_iteration = 0

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        total_loss = combined_pose_loss(
            model=model,
            decoder=decoder,
            camera=ViewCamera(
                w2c=compose_pose_update(xi, initial_w2c),
                K=camera.K,
                width=camera.width,
                height=camera.height,
            ),
            target_features=target_features,
            target_rgb=target_rgb,
            rgb_mask=rgb_mask,
            sh_degree=sh_degree,
            packed=packed,
            alpha_threshold=alpha_threshold,
            fixed_valid_mask=fixed_valid_mask,
            target_valid_mask=target_valid_mask,
            feature_loss_weight=feature_loss_weight,
            rgb_loss_weight=rgb_loss_weight,
            ssim_lambda=ssim_lambda,
        )
        loss_value = float(total_loss.detach().cpu())
        if loss_value < best_loss:
            best_loss = loss_value
            best_xi = xi.detach().clone()
            best_iteration = step
        total_loss.backward()
        optimizer.step()
        losses.append(loss_value)

    with torch.no_grad():
        final_w2c = compose_pose_update(best_xi, initial_w2c)
        final_loss = combined_pose_loss(
            model=model,
            decoder=decoder,
            camera=ViewCamera(w2c=final_w2c, K=camera.K, width=camera.width, height=camera.height),
            target_features=target_features,
            target_rgb=target_rgb,
            rgb_mask=rgb_mask,
            sh_degree=sh_degree,
            packed=packed,
            alpha_threshold=alpha_threshold,
            fixed_valid_mask=fixed_valid_mask,
            target_valid_mask=target_valid_mask,
            feature_loss_weight=feature_loss_weight,
            rgb_loss_weight=rgb_loss_weight,
            ssim_lambda=ssim_lambda,
        )

    return FeatureMetricRefinement(
        initial_w2c=initial_w2c.detach().cpu(),
        final_w2c=final_w2c.detach().cpu(),
        xi=best_xi.detach().cpu(),
        initial_loss=losses[0] if losses else float(final_loss.detach().cpu()),
        final_loss=float(final_loss.detach().cpu()),
        best_iteration=best_iteration,
        losses=tuple(losses),
    )


def combined_pose_loss(
    model,
    decoder,
    camera: ViewCamera,
    target_features: torch.Tensor,
    target_rgb: torch.Tensor,
    rgb_mask: torch.Tensor,
    sh_degree: int,
    packed: bool,
    alpha_threshold: float,
    fixed_valid_mask: torch.Tensor | None,
    target_valid_mask: torch.Tensor | None,
    feature_loss_weight: float,
    rgb_loss_weight: float,
    ssim_lambda: float,
) -> torch.Tensor:
    output = render(
        model,
        camera,
        sh_degree=sh_degree,
        packed=packed,
        render_latents=True,
    )
    predicted_features = decode_feature_map(
        decoder,
        output.latents,
        output_size=tuple(target_features.shape[:2]),
    )
    feature_loss, _ = featuremetric_loss(
        predicted_features,
        target_features,
        output.alpha,
        alpha_threshold,
        fixed_valid_mask=fixed_valid_mask,
        target_valid_mask=target_valid_mask,
    )
    rgb_loss = rgb_ssim_loss(output.rgb, target_rgb, rgb_mask, ssim_lambda)
    return feature_loss_weight * feature_loss + rgb_loss_weight * rgb_loss


def rgb_ssim_loss(
    predicted_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    valid_rgb_mask: torch.Tensor,
    ssim_lambda: float,
) -> torch.Tensor:
    l1 = F.l1_loss(predicted_rgb[valid_rgb_mask], target_rgb[valid_rgb_mask])
    predicted_ssim = predicted_rgb.masked_fill(~valid_rgb_mask[..., None], 0.0)
    target_ssim = target_rgb.masked_fill(~valid_rgb_mask[..., None], 0.0)
    ssim = ssim_loss(
        predicted_ssim.permute(2, 0, 1).unsqueeze(0),
        target_ssim.permute(2, 0, 1).unsqueeze(0),
    )
    return torch.lerp(l1, ssim, ssim_lambda)


def perturb_w2c(
    w2c: np.ndarray,
    translation: float,
    yaw_degrees: float,
) -> np.ndarray:
    c2w = np.linalg.inv(np.asarray(w2c, dtype=np.float64))
    yaw = np.deg2rad(yaw_degrees)
    rotation_delta = np.array(
        [
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ],
        dtype=np.float64,
    )
    perturbed_c2w = c2w.copy()
    perturbed_c2w[:3, :3] = c2w[:3, :3] @ rotation_delta
    perturbed_c2w[:3, 3] = c2w[:3, 3] + c2w[:3, 0] * translation
    return np.linalg.inv(perturbed_c2w)


def pose_error(
    estimate_w2c: np.ndarray,
    target_w2c: np.ndarray,
) -> PoseError:
    estimate_c2w = np.linalg.inv(np.asarray(estimate_w2c, dtype=np.float64))
    target_c2w = np.linalg.inv(np.asarray(target_w2c, dtype=np.float64))

    translation = float(np.linalg.norm(estimate_c2w[:3, 3] - target_c2w[:3, 3]))
    rotation_delta = target_c2w[:3, :3].T @ estimate_c2w[:3, :3]
    cos_angle = np.clip((np.trace(rotation_delta) - 1.0) * 0.5, -1.0, 1.0)
    rotation_deg = float(np.rad2deg(np.arccos(cos_angle)))
    return PoseError(translation=translation, rotation_deg=rotation_deg)


def query_pose_error(
    w2c: np.ndarray,
    gt_c2w: np.ndarray | None,
    gt_from_reference_sfm: np.ndarray,
) -> PoseError | None:
    if gt_c2w is None:
        return None
    c2w = np.linalg.inv(np.asarray(w2c, dtype=np.float64))
    aligned_c2w = transform_c2w(gt_from_reference_sfm, c2w)
    translation = float(np.linalg.norm(aligned_c2w[:3, 3] - gt_c2w[:3, 3]))
    rotation_delta = gt_c2w[:3, :3].T @ aligned_c2w[:3, :3]
    cos_angle = np.clip((np.trace(rotation_delta) - 1.0) * 0.5, -1.0, 1.0)
    rotation_deg = float(np.rad2deg(np.arccos(cos_angle)))
    return PoseError(translation=translation, rotation_deg=rotation_deg)


def print_refinement_row(row: dict) -> None:
    translation = format_metric_pair(row["initial_translation_error"], row["final_translation_error"], precision=4)
    rotation = format_metric_pair(row["initial_rotation_error_deg"], row["final_rotation_error_deg"], precision=2)
    print(
        f"{row['pose_index']:02d} | "
        f"loss {row['initial_feature_loss']:.4f}->{row['final_feature_loss']:.4f} | "
        f"trans {translation} | "
        f"rot {rotation} deg | "
        f"best_iter={row['best_iteration']} | "
        f"query={row['query_image']} | ref={row['reference_image']}"
    )


def format_metric_pair(
    initial: float | None,
    final: float | None,
    precision: int,
) -> str:
    if initial is None or final is None:
        return "n/a"
    return f"{initial:.{precision}f}->{final:.{precision}f}"


@torch.no_grad()
def write_diagonal_images(
    output_dir: Path,
    pose_index: int,
    image_name: str,
    image_rgb: np.ndarray,
    K: np.ndarray,
    initial_w2c: np.ndarray,
    final_w2c: np.ndarray,
    model,
    decoder,
    target_features: torch.Tensor | None,
    target_valid_mask: torch.Tensor | None,
    sh_degree: int,
    packed: bool,
) -> tuple[Path, Path, Path, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{pose_index:02d}_{clean_name(image_name)}"
    before_path = output_dir / f"{stem}_before.png"
    after_path = output_dir / f"{stem}_after.png"
    rgb_before_after_path = output_dir / f"{stem}_rgb_before_after.png"
    feature_before_after_path = None if target_features is None else output_dir / f"{stem}_feature_before_after.png"

    before, before_features = render_diagonal(
        image_rgb=image_rgb,
        K=K,
        w2c=initial_w2c,
        model=model,
        decoder=decoder,
        target_features=target_features,
        sh_degree=sh_degree,
        packed=packed,
    )
    after, after_features = render_diagonal(
        image_rgb=image_rgb,
        K=K,
        w2c=final_w2c,
        model=model,
        decoder=decoder,
        target_features=target_features,
        sh_degree=sh_degree,
        packed=packed,
    )
    write_rgb(before_path, before)
    write_rgb(after_path, after)
    write_rgb(rgb_before_after_path, side_by_side(before, after))
    if target_features is not None and before_features is not None and after_features is not None:
        feature_before_after = feature_diagonal_before_after(
            before_features=before_features,
            after_features=after_features,
            target_features=target_features,
            target_valid_mask=target_valid_mask,
            output_size=image_rgb.shape[:2],
        )
        write_rgb(feature_before_after_path, feature_before_after)
    return before_path, after_path, rgb_before_after_path, feature_before_after_path


def render_diagonal(
    image_rgb: np.ndarray,
    K: np.ndarray,
    w2c: np.ndarray,
    model,
    decoder,
    target_features: torch.Tensor | None,
    sh_degree: int,
    packed: bool,
) -> tuple[np.ndarray, torch.Tensor | None]:
    output = render(
        model,
        ViewCamera(
            w2c=w2c,
            K=K,
            width=image_rgb.shape[1],
            height=image_rgb.shape[0],
        ),
        sh_degree=sh_degree,
        packed=packed,
        render_latents=target_features is not None,
    )
    decoded_features = None
    if target_features is not None:
        decoded_features = decode_feature_map(
            decoder,
            output.latents,
            output_size=tuple(target_features.shape[:2]),
        )
    return diagonal_rgb_composite(output.rgb, image_rgb), decoded_features


def feature_diagonal_before_after(
    before_features: torch.Tensor,
    after_features: torch.Tensor,
    target_features: torch.Tensor,
    target_valid_mask: torch.Tensor | None,
    output_size: tuple[int, int],
) -> np.ndarray:
    if target_valid_mask is not None:
        valid = target_valid_mask.detach().to(device=before_features.device, dtype=torch.bool)
        projection_values = torch.cat(
            [
                before_features[valid].reshape(-1, before_features.shape[-1]),
                after_features[valid].reshape(-1, after_features.shape[-1]),
                target_features[valid].reshape(-1, target_features.shape[-1]),
            ],
            dim=0,
        )
    else:
        projection_values = torch.cat(
            [
                before_features.reshape(-1, before_features.shape[-1]),
                after_features.reshape(-1, after_features.shape[-1]),
                target_features.reshape(-1, target_features.shape[-1]),
            ],
            dim=0,
        )
    projection = fit_pca_projection(projection_values)
    before_rgb = pca_rgb(before_features, projection)
    after_rgb = pca_rgb(after_features, projection)
    target_rgb = pca_rgb(target_features, projection)
    if target_valid_mask is not None:
        mask = target_valid_mask.detach().cpu().bool()
        before_rgb = before_rgb.masked_fill(~mask[..., None].to(before_rgb.device), 0.0)
        after_rgb = after_rgb.masked_fill(~mask[..., None].to(after_rgb.device), 0.0)
        target_rgb = target_rgb.masked_fill(~mask[..., None].to(target_rgb.device), 0.0)

    before_diagonal = resize_rgb(
        diagonal_rgb_composite(before_rgb, target_rgb),
        output_size,
    )
    after_diagonal = resize_rgb(
        diagonal_rgb_composite(after_rgb, target_rgb),
        output_size,
    )
    return side_by_side(before_diagonal, after_diagonal)


def clean_name(name: str) -> str:
    return Path(name).with_suffix("").as_posix().replace("/", "_")


if __name__ == "__main__":
    main()
