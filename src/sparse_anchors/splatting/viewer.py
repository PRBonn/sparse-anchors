"""Interactive checkpoint and training viewer built on the gsplat viewer stack."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sparse_anchors.splatting.gaussians import GaussianModel
from sparse_anchors.splatting.renderer import RenderOutput, ViewCamera, render
from sparse_anchors.splatting.scene import Scene, transform_scene
from sparse_anchors.splatting.trainer import Trainer
from sparse_anchors.utils.io import load_splat_checkpoint
from sparse_anchors.utils.visualization import PcaProjection, fit_pca_projection, pca_rgb


def _display_image(
    mode: str,
    output: RenderOutput,
    latent_projection: PcaProjection,
) -> torch.Tensor:
    if mode == "rgb":
        return output.rgb
    if mode == "latent_pca":
        return pca_rgb(output.latents, latent_projection)

    from nerfview import apply_float_colormap

    if mode == "depth":
        values = output.depth / output.depth.max().clamp_min(torch.finfo(output.depth.dtype).eps)
        return apply_float_colormap(values, colormap="turbo")
    return apply_float_colormap(output.alpha, colormap="gray")


def _create_viewer(
    model: GaussianModel,
    scene: Scene,
    sh_degree: int,
    output_dir: Path,
    port: int,
    pca_refresh_every: int,
    viewer_resolution: int,
    trainer: Trainer | None = None,
) -> Any:
    import viser
    from nerfview import CameraState, RenderTabState, Viewer

    projection_cache = {
        "projection": fit_pca_projection(model.params["distill_features"]),
        "step": trainer.step if trainer is not None else 0,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    server = viser.ViserServer(port=port, verbose=False)
    camera_up = np.median(
        np.stack([-frame.c2w[:3, 1] for frame in scene.train_frames]),
        axis=0,
    )
    server.scene.set_up_direction(camera_up / np.linalg.norm(camera_up))
    with server.gui.add_folder("Splatting"):
        render_mode = server.gui.add_dropdown(
            "Render mode",
            ("rgb", "depth", "alpha", "latent_pca"),
            initial_value="rgb",
        )

    def render_fn(camera_state: CameraState, render_tab_state: RenderTabState) -> np.ndarray:
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height

        camera = ViewCamera(
            w2c=np.linalg.inv(camera_state.c2w),
            K=camera_state.get_K((width, height)),
            width=width,
            height=height,
        )
        step = trainer.step if trainer is not None else 0
        active_sh_degree = (
            min(trainer.config.sh_degree, step // trainer.config.sh_degree_interval)
            if trainer is not None
            else sh_degree
        )
        with torch.inference_mode():
            render_latents = render_mode.value == "latent_pca"
            if render_latents and step - projection_cache["step"] >= pca_refresh_every:
                projection_cache["projection"] = fit_pca_projection(model.params["distill_features"])
                projection_cache["step"] = step
            output = render(
                model,
                camera,
                sh_degree=active_sh_degree,
                packed=False,
                render_latents=render_latents,
            )
            image = _display_image(
                render_mode.value,
                output,
                projection_cache["projection"],
            )
        return (image.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()

    mode = "training" if trainer is not None else "rendering"
    viewer = Viewer(server=server, render_fn=render_fn, output_dir=output_dir, mode=mode)
    viewer.render_tab_state.viewer_res = viewer_resolution
    viewer._rendering_tab_handles["viewer_res_slider"].value = viewer_resolution

    initial_frame = scene.eval_frames[0] if scene.eval_frames else scene.train_frames[0]

    @server.on_client_connect
    def _set_initial_camera(client: viser.ClientHandle) -> None:
        c2w = initial_frame.c2w
        client.camera.position = c2w[:3, 3]
        client.camera.look_at = c2w[:3, 3] + c2w[:3, 2]
        client.camera.up = -c2w[:3, 1]

    render_mode.on_update(viewer.rerender)
    return viewer


def create_training_viewer(
    trainer: Trainer,
    output_dir: Path,
    port: int = 8080,
    pca_refresh_every: int = 500,
    viewer_resolution: int = 768,
) -> Any:
    """Create a viewer that renders the live Gaussian parameters during training."""
    viewer = _create_viewer(
        model=trainer.model,
        scene=trainer.scene,
        sh_degree=trainer.config.sh_degree,
        trainer=trainer,
        output_dir=output_dir,
        port=port,
        pca_refresh_every=pca_refresh_every,
        viewer_resolution=viewer_resolution,
    )
    print(f"Training viewer running at http://localhost:{port}.", flush=True)
    return viewer


def run_viewer(
    checkpoint_path: Path,
    scene: Scene,
    output_dir: Path,
    port: int = 8080,
    device: torch.device | str = "cuda",
) -> None:
    """Serve interactive RGB, depth, alpha, and latent-PCA checkpoint renders."""
    checkpoint = load_splat_checkpoint(checkpoint_path, device=device)
    scene = transform_scene(scene, checkpoint.world_transform)
    _create_viewer(
        model=checkpoint.model,
        scene=scene,
        sh_degree=checkpoint.sh_degree,
        output_dir=output_dir,
        port=port,
        pca_refresh_every=500,
        viewer_resolution=2048,
    )
    print(f"Viewer running at http://localhost:{port}. Press Ctrl+C to exit.", flush=True)
    while True:
        time.sleep(1.0)
