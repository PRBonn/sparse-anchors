"""Gaussian rasterization and checkpoint rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from gsplat.rendering import rasterization

from sparse_anchors.data.colmap import Frame
from sparse_anchors.splatting.features import decode_feature_map, load_feature_map
from sparse_anchors.splatting.gaussians import GaussianModel
from sparse_anchors.splatting.scene import Scene, transform_scene
from sparse_anchors.utils.image import write_rgb
from sparse_anchors.utils.io import load_splat_checkpoint
from sparse_anchors.utils.visualization import fit_pca_projection, pca_rgb


@dataclass(frozen=True)
class ViewCamera:
    """The camera parameters required by Gaussian rasterization."""

    w2c: np.ndarray | torch.Tensor
    K: np.ndarray | torch.Tensor
    width: int
    height: int


@dataclass(frozen=True)
class RenderOutput:
    rgb: torch.Tensor  # [H, W, 3]
    alpha: torch.Tensor  # [H, W, 1]
    depth: torch.Tensor  # [H, W, 1]
    latents: torch.Tensor | None  # [H, W, D]
    info: dict  # gsplat metadata used by the densification strategy


def render(
    model: GaussianModel,
    camera: Frame | ViewCamera,
    sh_degree: int,
    packed: bool = False,
    absgrad: bool = False,
    render_latents: bool = True,
) -> RenderOutput:
    device = model.device

    params = model.params
    viewmats = torch.as_tensor(camera.w2c, dtype=torch.float32, device=device).unsqueeze(0)
    Ks = torch.as_tensor(camera.K, dtype=torch.float32, device=device).unsqueeze(0)
    sh_coeffs = torch.cat([params["sh0"], params["shN"]], dim=1)

    render_rgbd, render_alphas, info = rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=sh_coeffs,
        viewmats=viewmats,
        Ks=Ks,
        width=camera.width,
        height=camera.height,
        sh_degree=sh_degree,
        extra_signals=params["distill_features"] if render_latents else None,
        render_mode="RGB+ED",  # Expected camera-z depth for PnP backprojection.
        packed=packed,
        absgrad=absgrad,
    )
    depth = render_rgbd[0, ..., 3:4]
    return RenderOutput(
        rgb=render_rgbd[0, ..., :3],
        alpha=render_alphas[0],
        depth=depth,
        latents=info["render_extra_signals"][0] if render_latents else None,
        info=info,
    )


@torch.no_grad()
def render_checkpoint(
    checkpoint_path: Path,
    scene: Scene,
    output_dir: Path,
    split: Literal["train", "eval", "all"] = "eval",
    device: torch.device | str = "cuda",
    save_features: bool = True,
    max_frames: int | None = None,
) -> None:
    """Render a checkpoint and save RGB, depth, latent PCA, and DINO predictions.

    The PCA is fitted once to the checkpoint's Gaussian latents, so its colors
    stay comparable across all rendered views from that checkpoint.
    """
    checkpoint = load_splat_checkpoint(checkpoint_path, device=device)
    scene = transform_scene(scene, checkpoint.world_transform)
    output_dir.mkdir(parents=True, exist_ok=True)

    pca = fit_pca_projection(checkpoint.model.params["distill_features"])
    torch.save(
        {
            "mean": pca.mean.cpu(),
            "components": pca.components.cpu(),
            "low": pca.low.cpu(),
            "high": pca.high.cpu(),
        },
        output_dir / "latent_pca.pt",
    )

    if split == "train":
        frames = scene.train_frames
    elif split == "eval":
        frames = scene.eval_frames
    else:
        frames = scene.colmap.frames

    if max_frames is not None:
        frames = frames[:max_frames]

    for frame in frames:
        output = render(
            checkpoint.model,
            frame,
            sh_degree=checkpoint.sh_degree,
            packed=checkpoint.packed,
        )
        relative_path = Path(frame.image_name)

        write_rgb(output_dir / "rgb" / relative_path, output.rgb)
        write_rgb(output_dir / "latent_pca" / relative_path, pca_rgb(output.latents, pca))

        depth_path = output_dir / "depth" / relative_path.with_suffix(".npy")
        depth_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(depth_path, output.depth.squeeze(-1).cpu().numpy())

        if save_features:
            target = load_feature_map(scene.features, frame.image_name)
            decoded = decode_feature_map(
                checkpoint.decoder,
                output.latents,
                output_size=tuple(target.shape[:2]),
            )
            feature_path = output_dir / "features" / relative_path.with_suffix(".npz")
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                feature_path,
                values=decoded.cpu().numpy(),
            )
