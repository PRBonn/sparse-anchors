"""Command-line entry points for Gaussian splatting."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from statistics import fmean

import torch

from sparse_anchors.splatting.features import FeatureDecoder
from sparse_anchors.splatting.gaussians import GaussianModel
from sparse_anchors.splatting.renderer import render_checkpoint
from sparse_anchors.splatting.scene import Scene, load_scene, normalize_scene
from sparse_anchors.splatting.trainer import Trainer, TrainerConfig
from sparse_anchors.splatting.viewer import create_training_viewer, run_viewer


def _run_scene(
    run_dir: Path,
    scene_root: Path | None = None,
    feature_manifest: Path | None = None,
) -> tuple[Scene, dict]:
    run = json.loads((run_dir / "run.json").read_text())
    scene_root = scene_root or Path(run["scene"])
    feature_manifest = feature_manifest or Path(run["feature_manifest"])
    scene = load_scene(
        sparse_dir=scene_root / "sparse/0",
        images_dir=scene_root / "images",
        feature_manifest_path=feature_manifest,
        test_every=0,
    )
    frames = {frame.image_name: frame for frame in scene.colmap.frames}
    return replace(
        scene,
        train_frames=tuple(frames[name] for name in run["train_images"]),
        eval_frames=tuple(frames[name] for name in run["eval_images"]),
    ), run


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    target_steps = int(args.steps * args.steps_scaler)

    scene_root = args.scene.resolve()
    run = json.loads((args.output_dir / "run.json").read_text()) if args.resume else None
    feature_manifest = args.feature_manifest or (
        Path(run["feature_manifest"]) if run else scene_root / "features/dinov3/manifest.json"
    )
    feature_manifest = feature_manifest.absolute()
    masks_dir = (args.masks_dir or scene_root / "masks").resolve()
    depth_dir = args.da3_depth_dir or (
        Path(run["da3_depth_dir"]) if run and run.get("da3_depth_dir") else scene_root / "depth/da3"
    )
    depth_dir = depth_dir.absolute()

    if args.resume:
        scene, _ = _run_scene(args.output_dir, scene_root, feature_manifest)
    else:
        scene = load_scene(
            sparse_dir=scene_root / "sparse/0",
            images_dir=scene_root / "images",
            feature_manifest_path=feature_manifest,
            test_every=args.test_every,
        )

    if args.resume is not None:
        trainer = Trainer.load_checkpoint(
            args.resume,
            scene=scene,
            device=args.device,
            masks_dir=masks_dir,
            da3_depth_dir=depth_dir,
            mask_dilation_pixels=args.mask_dilation_pixels,
            da3_confidence_threshold=args.da3_confidence_threshold,
            data_loader_workers=args.data_loader_workers,
            data_loader_prefetch_factor=args.data_loader_prefetch_factor,
        )
    else:
        scene = normalize_scene(scene)
        model = GaussianModel.from_scene(
            scene,
            device=args.device,
            sh_degree=args.sh_degree,
            feature_latent_dim=args.feature_latent_dim,
        )
        decoder = FeatureDecoder(
            latent_dim=args.feature_latent_dim,
            embedding_dim=scene.features.embedding_dim,
        )
        trainer = Trainer(
            scene=scene,
            model=model,
            decoder=decoder,
            config=TrainerConfig(
                feature_loss_weight=args.feature_loss_weight,
                ssim_lambda=args.ssim_lambda,
                max_steps=target_steps,
                steps_scaler=args.steps_scaler,
                sh_degree=args.sh_degree,
                sh_degree_interval=int(args.sh_degree_interval * args.steps_scaler),
                da3_depth_loss_weight=args.da3_depth_loss_weight,
                da3_depth_start_step=int(args.da3_depth_start_step * args.steps_scaler),
                da3_confidence_threshold=args.da3_confidence_threshold,
                da3_alpha_threshold=args.da3_alpha_threshold,
                da3_log_smooth_l1_beta=args.da3_log_smooth_l1_beta,
                mask_dilation_pixels=args.mask_dilation_pixels,
                normalize_feature_loss=args.normalize_feature_loss,
                packed=args.packed,
                data_loader_workers=args.data_loader_workers,
                data_loader_prefetch_factor=args.data_loader_prefetch_factor,
            ),
            masks_dir=masks_dir,
            da3_depth_dir=depth_dir,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run = {
        "scene": str(scene_root),
        "feature_manifest": str(feature_manifest),
        "da3_depth_dir": str(trainer.da3_depth_dir) if trainer.da3_depth_dir else None,
        "train_images": [frame.image_name for frame in scene.train_frames],
        "eval_images": [frame.image_name for frame in scene.eval_frames],
    }

    def save_run(checkpoint_path: Path) -> None:
        run["checkpoint"] = checkpoint_path.name
        (args.output_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")

    print(f"training target: {target_steps} steps", flush=True)
    if args.viewer:
        trainer.viewer = create_training_viewer(
            trainer,
            output_dir=args.output_dir / "viewer",
            port=args.viewer_port,
            pca_refresh_every=args.viewer_pca_refresh_every,
            viewer_resolution=args.viewer_resolution,
        )

    metrics_window: deque[dict[str, float]] = deque(maxlen=args.log_every)
    log_start = time.perf_counter()
    while trainer.step < target_steps:
        metrics = trainer.train_step()
        metrics_window.append(metrics)

        if trainer.step % args.log_every == 0:
            window_seconds = time.perf_counter() - log_start
            print(
                f"step={trainer.step:06d} "
                f"gaussians={int(metrics['num_gaussians'])} "
                f"loss={fmean(item['loss'] for item in metrics_window):.6f} "
                f"rgb={fmean(item['rgb_loss'] for item in metrics_window):.6f} "
                f"feature={fmean(item['feature_loss'] for item in metrics_window):.6f} "
                f"da3_depth={fmean(item['da3_depth_loss'] for item in metrics_window):.6f} "
                f"sh={int(metrics['sh_degree'])} "
                f"it/s={len(metrics_window) / window_seconds:.2f}",
                flush=True,
            )
            log_start = time.perf_counter()

        if args.checkpoint_every > 0 and trainer.step % args.checkpoint_every == 0:
            checkpoint_path = args.output_dir / f"checkpoint_{trainer.step:06d}.pt"
            trainer.save_checkpoint(checkpoint_path)
            save_run(checkpoint_path)
            print(f"saved checkpoint: {checkpoint_path}", flush=True)

    checkpoint_path = args.output_dir / f"checkpoint_{trainer.step:06d}.pt"
    trainer.save_checkpoint(checkpoint_path)
    save_run(checkpoint_path)
    print(f"saved final checkpoint: {checkpoint_path}", flush=True)


def render(args: argparse.Namespace) -> None:
    scene, run = _run_scene(args.run_dir)
    render_checkpoint(
        checkpoint_path=args.run_dir / (args.checkpoint or run["checkpoint"]),
        scene=scene,
        output_dir=args.output_dir or args.run_dir / "renders",
        split=args.split,
        device=args.device,
        save_features=not args.no_save_features,
        max_frames=args.max_frames,
    )


def viewer(args: argparse.Namespace) -> None:
    scene, run = _run_scene(args.run_dir)
    run_viewer(
        checkpoint_path=args.run_dir / (args.checkpoint or run["checkpoint"]),
        scene=scene,
        output_dir=args.output_dir or args.run_dir / "viewer",
        port=args.port,
        device=args.device,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sparse-anchors")
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train")
    train_parser.add_argument("--scene", type=Path, required=True)
    train_parser.add_argument("--masks-dir", type=Path)
    train_parser.add_argument("--feature-manifest", type=Path)
    train_parser.add_argument("--da3-depth-dir", type=Path)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--test-every", type=int, default=8)
    train_parser.add_argument("--steps", type=int, default=TrainerConfig.max_steps)
    train_parser.add_argument("--steps-scaler", type=float, default=TrainerConfig.steps_scaler)
    train_parser.add_argument("--checkpoint-every", type=int, default=10_000)
    train_parser.add_argument("--log-every", type=int, default=250)
    train_parser.add_argument("--resume", type=Path)
    train_parser.add_argument("--device", default="cuda")
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--feature-latent-dim", type=int, default=128)
    train_parser.add_argument("--feature-loss-weight", type=float, default=TrainerConfig.feature_loss_weight)
    train_parser.add_argument("--ssim-lambda", type=float, default=TrainerConfig.ssim_lambda)
    train_parser.add_argument("--sh-degree", type=int, default=TrainerConfig.sh_degree)
    train_parser.add_argument("--sh-degree-interval", type=int, default=TrainerConfig.sh_degree_interval)
    train_parser.add_argument("--mask-dilation-pixels", type=int, default=TrainerConfig.mask_dilation_pixels)
    train_parser.add_argument("--normalize-feature-loss", action="store_true")
    train_parser.add_argument("--da3-depth-loss-weight", type=float, default=TrainerConfig.da3_depth_loss_weight)
    train_parser.add_argument("--da3-depth-start-step", type=int, default=TrainerConfig.da3_depth_start_step)
    train_parser.add_argument("--da3-confidence-threshold", type=float, default=TrainerConfig.da3_confidence_threshold)
    train_parser.add_argument("--da3-alpha-threshold", type=float, default=TrainerConfig.da3_alpha_threshold)
    train_parser.add_argument("--da3-log-smooth-l1-beta", type=float, default=TrainerConfig.da3_log_smooth_l1_beta)
    train_parser.add_argument("--packed", action="store_true")
    train_parser.add_argument("--data-loader-workers", type=int, default=TrainerConfig.data_loader_workers)
    train_parser.add_argument(
        "--data-loader-prefetch-factor", type=int, default=TrainerConfig.data_loader_prefetch_factor
    )
    train_parser.add_argument("--viewer", action="store_true")
    train_parser.add_argument("--viewer-port", type=int, default=8080)
    train_parser.add_argument("--viewer-pca-refresh-every", type=int, default=500)
    train_parser.add_argument("--viewer-resolution", type=int, default=768)
    train_parser.set_defaults(func=train)

    render_parser = commands.add_parser("render")
    render_parser.add_argument("--run-dir", type=Path, required=True)
    render_parser.add_argument("--checkpoint", type=Path)
    render_parser.add_argument("--output-dir", type=Path)
    render_parser.add_argument("--split", choices=("train", "eval", "all"), default="eval")
    render_parser.add_argument("--device", default="cuda")
    render_parser.add_argument("--no-save-features", action="store_true")
    render_parser.add_argument("--max-frames", type=int)
    render_parser.set_defaults(func=render)

    viewer_parser = commands.add_parser("viewer")
    viewer_parser.add_argument("--run-dir", type=Path, required=True)
    viewer_parser.add_argument("--checkpoint", type=Path)
    viewer_parser.add_argument("--output-dir", type=Path)
    viewer_parser.add_argument("--port", type=int, default=8080)
    viewer_parser.add_argument("--device", default="cuda")
    viewer_parser.set_defaults(func=viewer)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)
