from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from configs import localization_exp as config
from sparse_anchors.data.colmap import load_colmap_model
from sparse_anchors.data.sessions import SessionData
from sparse_anchors.features.dinov3 import DinoV3FeatureExtractor, hf_model_from_manifest
from sparse_anchors.localization.featuremetric import (
    RigPose,
    RigView,
    accumulate_rig_loss,
    build_rig_views,
    evaluate_featuremetric_pose,
    single_view_loss,
    view_camera,
)
from sparse_anchors.localization.types import FeatureMetricRefinement
from sparse_anchors.splatting.features import decode_feature_map
from sparse_anchors.splatting.renderer import render
from sparse_anchors.utils.geometry import compose_pose_update
from sparse_anchors.utils.image import rgb_to_uint8, write_rgb
from sparse_anchors.utils.io import load_splat_checkpoint
from sparse_anchors.utils.visualization import (
    diagonal_rgb_composite,
    fit_pca_projection,
    pca_rgb,
    resize_rgb,
)


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)

    rig_poses = load_rig_poses(args.pose_estimates)
    if args.max_poses is not None:
        rig_poses = rig_poses[: args.max_poses]
    query_data = SessionData(
        load_colmap_model(
            sparse_dir=args.query_sparse_dir,
            images_dir=args.query_images_dir,
        ).frames
    )

    for checkpoint_path in args.checkpoints:
        run_checkpoint_experiment(
            args=args,
            checkpoint_path=checkpoint_path,
            rig_poses=rig_poses,
            query_data=query_data,
            device=device,
        )


def run_checkpoint_experiment(
    args: argparse.Namespace,
    checkpoint_path: Path,
    rig_poses: list[RigPose],
    query_data: SessionData,
    device: torch.device,
) -> None:
    checkpoint_path = checkpoint_path.resolve()
    checkpoint = load_splat_checkpoint(checkpoint_path, device=device)
    manifest = checkpoint.feature_manifest
    output_dir = args.output_root / checkpoint_output_name(checkpoint_path)
    visualization_root = output_dir / "per_timestamp"
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_extractor = DinoV3FeatureExtractor(
        device=device,
        hf_model=args.dinov3_hf_model or hf_model_from_manifest(manifest["extractor"]),
    )

    rows: list[dict] = []
    feature_cache = {}
    print(f"running {checkpoint_path}")
    print(f"extractor={feature_extractor.name}, poses={len(rig_poses)}, output={output_dir}")

    for rig_pose in rig_poses:
        views = build_rig_views(
            rig_pose=rig_pose,
            query_data=query_data,
            feature_extractor=feature_extractor,
            feature_cache=feature_cache,
            query_masks_dir=None if args.disable_query_mask else args.query_masks_dir,
            feature_long_side=args.feature_long_side,
            mask_dilation_pixels=args.mask_dilation_pixels,
            device=device,
        )
        if not views:
            print(f"{rig_pose.index:02d}: no valid rig views, skipped")
            continue

        refinement = refine_rig_featuremetric(
            model=checkpoint.model,
            decoder=checkpoint.decoder,
            initial_rig_w2c=rig_pose.rig_w2c,
            views=views,
            sh_degree=checkpoint.sh_degree,
            packed=checkpoint.packed,
            alpha_threshold=args.feature_alpha_threshold,
            steps=args.steps,
            learning_rate=args.learning_rate,
            freeze_alpha_mask=not args.dynamic_alpha_mask,
            progress_every=args.progress_every,
            progress_prefix=f"{rig_pose.index:02d} {rig_pose.query_timestamp}",
        )
        paths = write_rig_visualizations(
            visualization_root=visualization_root,
            pose_index=rig_pose.index,
            query_timestamp=rig_pose.query_timestamp,
            views=views,
            model=checkpoint.model,
            decoder=checkpoint.decoder,
            initial_rig_w2c=refinement.initial_w2c.numpy(),
            final_rig_w2c=refinement.final_w2c.numpy(),
            sh_degree=checkpoint.sh_degree,
            packed=checkpoint.packed,
            canny_low=args.edge_canny_low,
            canny_high=args.edge_canny_high,
            skip_features=args.skip_feature_diagonals,
        )
        row = {
            "pose_index": rig_pose.index,
            "query_timestamp": rig_pose.query_timestamp,
            "reference_timestamp": rig_pose.reference_timestamp,
            "checkpoint": str(checkpoint_path),
            "extractor": feature_extractor.name,
            "num_views": len(views),
            "initial_feature_loss": refinement.initial_loss,
            "final_feature_loss": refinement.final_loss,
            "best_iteration": refinement.best_iteration,
            "xi": refinement.xi.tolist(),
            "initial_rig_w2c": refinement.initial_w2c.tolist(),
            "final_rig_w2c": refinement.final_w2c.tolist(),
            "visualization_dir": str(paths["visualization_dir"]),
            "per_view_figures": [str(path) for path in paths["per_view_figures"]],
            "rgb_before": str(paths["rgb_before"]),
            "rgb_after": str(paths["rgb_after"]),
            "rgb_before_after": str(paths["rgb_before_after"]),
            "feature_before": str(paths["feature_before"]),
            "feature_after": str(paths["feature_after"]),
            "feature_before_after": str(paths["feature_before_after"]),
            "edge_before": str(paths["edge_before"]),
            "edge_after": str(paths["edge_after"]),
            "edge_before_after": str(paths["edge_before_after"]),
            "losses": list(refinement.losses),
        }
        rows.append(row)
        print(
            f"{rig_pose.index:02d} {rig_pose.query_timestamp}: "
            f"loss {row['initial_feature_loss']:.4f}->{row['final_feature_loss']:.4f}, "
            f"views={row['num_views']}, best_iter={row['best_iteration']}"
        )

    write_results_json(args, output_dir, checkpoint_path, manifest, rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Joint featuremetric refinement for virtual-camera rigs.")
    parser.add_argument("--pose-estimates", type=Path, default=config.output_dir / "pose_estimates.json")
    parser.add_argument("--checkpoints", type=Path, nargs="+", default=[config.checkpoint])
    parser.add_argument("--output-root", type=Path, default=Path("media/registration/rig_featuremetric"))
    parser.add_argument("--query-sparse-dir", type=Path, default=config.query_sparse_dir)
    parser.add_argument("--query-images-dir", type=Path, default=config.query_images_dir)
    parser.add_argument("--query-masks-dir", type=Path, default=config.query_root / "colmap_sfm_rig" / "feature_masks")
    parser.add_argument("--disable-query-mask", action="store_true")
    parser.add_argument("--mask-dilation-pixels", type=int, default=14)
    parser.add_argument("--feature-long-side", type=int, default=768)
    parser.add_argument("--feature-alpha-threshold", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dynamic-alpha-mask", action="store_true")
    parser.add_argument("--max-poses", type=int)
    parser.add_argument("--device", default=config.device)
    parser.add_argument("--dinov3-hf-model", default=None)
    parser.add_argument("--edge-canny-low", type=int, default=80)
    parser.add_argument("--edge-canny-high", type=int, default=160)
    parser.add_argument("--skip-feature-diagonals", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser


def load_rig_poses(path: Path) -> list[RigPose]:
    payload = json.loads(path.read_text())
    poses = []
    for index, pose in enumerate(payload["poses"]):
        poses.append(
            RigPose(
                index=index,
                query_timestamp=str(pose["query_timestamp"]),
                reference_timestamp=str(pose["reference_timestamp"]),
                rig_w2c=np.asarray(pose["rig_w2c"] if "rig_w2c" in pose else pose["w2c"], dtype=np.float64),
                virtual_camera_poses=tuple(pose["virtual_camera_poses"]),
            )
        )
    return poses


def refine_rig_featuremetric(
    model,
    decoder,
    initial_rig_w2c: np.ndarray,
    views: list[RigView],
    sh_degree: int,
    packed: bool,
    alpha_threshold: float,
    steps: int,
    learning_rate: float,
    freeze_alpha_mask: bool,
    progress_every: int,
    progress_prefix: str,
) -> FeatureMetricRefinement:
    device = views[0].target_features.device
    initial_w2c = torch.as_tensor(initial_rig_w2c, dtype=torch.float32, device=device).detach()
    xi = torch.zeros(6, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([xi], lr=learning_rate)

    fixed_masks = [None for _ in views]
    if freeze_alpha_mask:
        with torch.no_grad():
            for index, view in enumerate(views):
                camera_w2c = view.camera_from_rig @ initial_w2c
                _, valid_mask = evaluate_featuremetric_pose(
                    model=model,
                    decoder=decoder,
                    camera=view_camera(view, camera_w2c),
                    target_features=view.target_features,
                    sh_degree=sh_degree,
                    packed=packed,
                    alpha_threshold=alpha_threshold,
                    target_valid_mask=view.target_valid_mask,
                )
                fixed_masks[index] = valid_mask.detach()

    with torch.no_grad():
        initial_loss = accumulate_rig_loss(
            model=model,
            decoder=decoder,
            initial_rig_w2c=initial_w2c,
            xi=xi,
            views=views,
            fixed_masks=fixed_masks,
            sh_degree=sh_degree,
            packed=packed,
            alpha_threshold=alpha_threshold,
            backward=False,
        )

    losses = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        view_index = step % len(views)
        view_loss = single_view_loss(
            model=model,
            decoder=decoder,
            initial_rig_w2c=initial_w2c,
            xi=xi,
            view=views[view_index],
            fixed_mask=fixed_masks[view_index],
            sh_degree=sh_degree,
            packed=packed,
            alpha_threshold=alpha_threshold,
        )
        view_loss.backward()
        loss_value = float(view_loss.detach().cpu())
        optimizer.step()
        losses.append(loss_value)
        if progress_every > 0 and (step == 0 or (step + 1) % progress_every == 0 or step + 1 == steps):
            print(
                f"{progress_prefix}: step {step + 1}/{steps}, "
                f"view={views[view_index].camera_name}, loss={loss_value:.4f}"
            )

    with torch.no_grad():
        final_xi = xi.detach().clone()
        final_w2c = compose_pose_update(final_xi, initial_w2c)
        final_loss = accumulate_rig_loss(
            model=model,
            decoder=decoder,
            initial_rig_w2c=initial_w2c,
            xi=final_xi,
            views=views,
            fixed_masks=fixed_masks,
            sh_degree=sh_degree,
            packed=packed,
            alpha_threshold=alpha_threshold,
            backward=False,
        )

    return FeatureMetricRefinement(
        initial_w2c=initial_w2c.detach().cpu(),
        final_w2c=final_w2c.detach().cpu(),
        xi=final_xi.detach().cpu(),
        initial_loss=float(initial_loss.detach().cpu()),
        final_loss=float(final_loss.detach().cpu()),
        best_iteration=max(steps - 1, 0),
        losses=tuple(losses),
    )


@torch.no_grad()
def write_rig_visualizations(
    visualization_root: Path,
    pose_index: int,
    query_timestamp: str,
    views: list[RigView],
    model,
    decoder,
    initial_rig_w2c: np.ndarray,
    final_rig_w2c: np.ndarray,
    sh_degree: int,
    packed: bool,
    canny_low: int,
    canny_high: int,
    skip_features: bool,
) -> dict[str, Path | list[Path]]:
    stem = f"{pose_index:02d}_{query_timestamp}"
    timestamp_dir = visualization_root / stem
    timestamp_dir.mkdir(parents=True, exist_ok=True)

    initial_rig_w2c_t = torch.as_tensor(initial_rig_w2c, dtype=torch.float32, device=views[0].target_features.device)
    final_rig_w2c_t = torch.as_tensor(final_rig_w2c, dtype=torch.float32, device=views[0].target_features.device)

    per_view_figures = []
    for view in views:
        initial_output, initial_features = render_view_outputs(
            view=view,
            model=model,
            decoder=decoder,
            rig_w2c=initial_rig_w2c_t,
            sh_degree=sh_degree,
            packed=packed,
            render_features=not skip_features,
        )
        final_output, final_features = render_view_outputs(
            view=view,
            model=model,
            decoder=decoder,
            rig_w2c=final_rig_w2c_t,
            sh_degree=sh_degree,
            packed=packed,
            render_features=not skip_features,
        )
        rgb_before = diagonal_rgb_composite(initial_output.rgb, view.image_rgb)
        rgb_after = diagonal_rgb_composite(final_output.rgb, view.image_rgb)
        before_edge = edge_overlay(view.image_rgb, initial_output.rgb, canny_low, canny_high)
        after_edge = edge_overlay(view.image_rgb, final_output.rgb, canny_low, canny_high)
        if initial_features is not None and final_features is not None:
            feature_before, feature_after = feature_diagonals(
                before_features=initial_features,
                after_features=final_features,
                target_features=view.target_features,
                target_valid_mask=view.target_valid_mask,
                output_size=view.image_rgb.shape[:2],
            )
        else:
            feature_before = np.zeros_like(view.image_rgb, dtype=np.float32)
            feature_after = np.zeros_like(view.image_rgb, dtype=np.float32)

        figure = per_view_summary_figure(
            query_rgb=view.image_rgb,
            render_before=initial_output.rgb,
            render_after=final_output.rgb,
            rgb_before=rgb_before,
            rgb_after=rgb_after,
            edge_before=before_edge,
            edge_after=after_edge,
            feature_before=feature_before,
            feature_after=feature_after,
        )
        path = timestamp_dir / f"{clean_image_name(view.image_name)}.png"
        write_rgb(path, figure)
        per_view_figures.append(path)

    paths = {
        "visualization_dir": timestamp_dir,
        "per_view_figures": per_view_figures,
        "rgb_before": per_view_figures[0],
        "rgb_after": per_view_figures[0],
        "rgb_before_after": per_view_figures[0],
        "feature_before": per_view_figures[0],
        "feature_after": per_view_figures[0],
        "feature_before_after": per_view_figures[0],
        "edge_before": per_view_figures[0],
        "edge_after": per_view_figures[0],
        "edge_before_after": per_view_figures[0],
    }
    return paths


def render_view_outputs(
    view: RigView,
    model,
    decoder,
    rig_w2c: torch.Tensor,
    sh_degree: int,
    packed: bool,
    render_features: bool,
):
    camera_w2c = view.camera_from_rig @ rig_w2c
    output = render(
        model,
        view_camera(view, camera_w2c),
        sh_degree=sh_degree,
        packed=packed,
        render_latents=render_features,
    )
    decoded_features = None
    if render_features:
        decoded_features = decode_feature_map(
            decoder,
            output.latents,
            output_size=tuple(view.target_features.shape[:2]),
        )
    return output, decoded_features


def per_view_summary_figure(
    query_rgb: np.ndarray,
    render_before: np.ndarray | torch.Tensor,
    render_after: np.ndarray | torch.Tensor,
    rgb_before: np.ndarray | torch.Tensor,
    rgb_after: np.ndarray | torch.Tensor,
    edge_before: np.ndarray,
    edge_after: np.ndarray,
    feature_before: np.ndarray,
    feature_after: np.ndarray,
    separator_width: int = 8,
) -> np.ndarray:
    height, width = query_rgb.shape[:2]
    panels = [
        titled_panel(resize_rgb(query_rgb, (height, width)), "query rgb"),
        titled_panel(resize_rgb(render_before, (height, width)), "render before"),
        titled_panel(resize_rgb(render_after, (height, width)), "render after"),
        titled_panel(resize_rgb(rgb_before, (height, width)), "rgb diagonal before"),
        titled_panel(resize_rgb(rgb_after, (height, width)), "rgb diagonal after"),
        titled_panel(resize_rgb(edge_before, (height, width)), "edges before"),
        titled_panel(resize_rgb(edge_after, (height, width)), "edges after"),
        titled_panel(resize_rgb(feature_before, (height, width)), "features before"),
        titled_panel(resize_rgb(feature_after, (height, width)), "features after"),
    ]
    rows = [
        horizontal_stack(panels[0:3], separator_width),
        horizontal_stack(panels[3:6], separator_width),
        horizontal_stack(panels[6:9], separator_width),
    ]
    separator = np.ones((separator_width, rows[0].shape[1], 3), dtype=np.float32)
    return np.concatenate([rows[0], separator, rows[1], separator, rows[2]], axis=0)


def titled_panel(image: np.ndarray, title: str, title_height: int = 34) -> np.ndarray:
    image = np.clip(image, 0.0, 1.0).astype(np.float32)
    strip = np.ones((title_height, image.shape[1], 3), dtype=np.float32)
    cv2.putText(
        strip,
        title,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0.0, 0.0, 0.0),
        1,
        cv2.LINE_AA,
    )
    return np.concatenate([strip, image], axis=0)


def horizontal_stack(images: list[np.ndarray], separator_width: int) -> np.ndarray:
    separator = np.ones((images[0].shape[0], separator_width, 3), dtype=np.float32)
    output = images[0]
    for image in images[1:]:
        output = np.concatenate([output, separator, image], axis=1)
    return output


def feature_diagonals(
    before_features: torch.Tensor,
    after_features: torch.Tensor,
    target_features: torch.Tensor,
    target_valid_mask: torch.Tensor | None,
    output_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    if target_valid_mask is not None:
        valid = target_valid_mask.detach().to(device=target_features.device, dtype=torch.bool)
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
        mask = target_valid_mask.detach().to(device=before_rgb.device, dtype=torch.bool)
        before_rgb = before_rgb.masked_fill(~mask[..., None], 0.0)
        after_rgb = after_rgb.masked_fill(~mask[..., None], 0.0)
        target_rgb = target_rgb.masked_fill(~mask[..., None], 0.0)
    return (
        resize_rgb(diagonal_rgb_composite(before_rgb, target_rgb), output_size),
        resize_rgb(diagonal_rgb_composite(after_rgb, target_rgb), output_size),
    )


def edge_overlay(
    query_rgb: np.ndarray | torch.Tensor,
    render_rgb: np.ndarray | torch.Tensor,
    low: int,
    high: int,
) -> np.ndarray:
    query_edges = canny_edges(query_rgb, low, high)
    render_edges = canny_edges(render_rgb, low, high)
    overlay = np.zeros((*query_edges.shape, 3), dtype=np.float32)
    overlay[query_edges] = (1.0, 1.0, 1.0)
    overlay[render_edges] = (1.0, 0.0, 0.0)
    return overlay


def canny_edges(image: np.ndarray | torch.Tensor, low: int, high: int) -> np.ndarray:
    image = torch.as_tensor(image).detach().cpu().numpy()
    image = rgb_to_uint8(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.Canny(gray, low, high) > 0


def clean_image_name(image_name: str) -> str:
    return Path(image_name).with_suffix("").as_posix().replace("/", "__")


def checkpoint_output_name(checkpoint_path: Path) -> str:
    return f"{checkpoint_path.parent.name}_{checkpoint_path.stem}"


def write_results_json(
    args: argparse.Namespace,
    output_dir: Path,
    checkpoint_path: Path,
    manifest: dict,
    rows: list[dict],
) -> None:
    path = output_dir / "rig_featuremetric_results.json"
    payload = {
        "settings": {
            "pose_estimates": str(args.pose_estimates),
            "checkpoint": str(checkpoint_path),
            "feature_manifest": {
                "extractor": manifest.get("extractor"),
                "embedding_dim": manifest.get("embedding_dim"),
            },
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "optimization_mode": "one_view_per_step_round_robin",
            "progress_every": args.progress_every,
            "feature_alpha_threshold": args.feature_alpha_threshold,
            "alpha_mask": "dynamic" if args.dynamic_alpha_mask else "initial_fixed",
            "feature_long_side": args.feature_long_side,
            "query_mask": None if args.disable_query_mask else str(args.query_masks_dir),
            "mask_dilation_pixels": args.mask_dilation_pixels,
            "edge_canny_low": args.edge_canny_low,
            "edge_canny_high": args.edge_canny_high,
        },
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
