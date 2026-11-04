"""Feature-metric rendering and loss for pose refinement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sparse_anchors.data.sessions import SessionData
from sparse_anchors.features.base import ImageFeatureExtractor, extract_image_features
from sparse_anchors.localization.types import FeatureMetricRefinement
from sparse_anchors.splatting.features import (
    FeatureDecoder,
    decode_feature_map,
)
from sparse_anchors.splatting.gaussians import GaussianModel
from sparse_anchors.splatting.renderer import (
    ViewCamera,
    render,
)
from sparse_anchors.utils.geometry import compose_pose_update, so3_exp_torch
from sparse_anchors.utils.image import load_mask, load_rgb


@dataclass(frozen=True)
class RigPose:
    index: int
    query_timestamp: str
    reference_timestamp: str
    rig_w2c: np.ndarray
    virtual_camera_poses: tuple[dict, ...]


@dataclass(frozen=True)
class RigView:
    image_name: str
    camera_name: str
    image_rgb: np.ndarray
    K: np.ndarray
    width: int
    height: int
    camera_from_rig: torch.Tensor
    target_features: torch.Tensor
    target_valid_mask: torch.Tensor | None


def view_camera(view: RigView, w2c: torch.Tensor) -> ViewCamera:
    return ViewCamera(w2c=w2c, K=view.K, width=view.width, height=view.height)


def build_rig_views(
    rig_pose: RigPose,
    query_data: SessionData,
    feature_extractor: ImageFeatureExtractor,
    feature_cache: dict,
    query_masks_dir: Path | None,
    feature_long_side: int,
    mask_dilation_pixels: int,
    device: torch.device,
) -> list[RigView]:
    rig_c2w = np.linalg.inv(rig_pose.rig_w2c)
    views = []
    for virtual_pose in rig_pose.virtual_camera_poses:
        image_name = str(virtual_pose["image_name"])
        frame = query_data.frames_by_name[image_name]
        camera_w2c = np.asarray(virtual_pose["w2c"], dtype=np.float64)
        camera_from_rig = camera_w2c @ rig_c2w
        image_rgb = load_rgb(frame.image_path)
        features = feature_cache.get(image_name)
        if features is None:
            valid_mask = None
            if query_masks_dir is not None:
                valid_mask = load_mask(query_masks_dir / Path(image_name).with_suffix(".png.png"))
            values, mask = extract_image_features(
                feature_extractor=feature_extractor,
                image_rgb=image_rgb,
                long_side=feature_long_side,
                valid_mask=valid_mask,
                mask_dilation_pixels=mask_dilation_pixels,
            )
            features = torch.from_numpy(values).to(device), None if mask is None else torch.from_numpy(mask).to(device)
            feature_cache[image_name] = features
        target_features, target_valid_mask = features
        if target_valid_mask is not None and not bool(target_valid_mask.any()):
            continue
        views.append(
            RigView(
                image_name=image_name,
                camera_name=str(virtual_pose["camera_name"]),
                image_rgb=image_rgb,
                K=frame.K,
                width=frame.width,
                height=frame.height,
                camera_from_rig=torch.as_tensor(camera_from_rig, dtype=torch.float32, device=device),
                target_features=target_features,
                target_valid_mask=target_valid_mask,
            )
        )
    return views


def featuremetric_loss(
    predicted_features: torch.Tensor,
    target_features: torch.Tensor,
    alpha: torch.Tensor,
    alpha_threshold: float,
    fixed_valid_mask: torch.Tensor | None = None,
    target_valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return masked L1 feature loss and its valid-pixel mask."""
    feature_height, feature_width = predicted_features.shape[:2]

    # Alpha is rendered at native camera resolution. Resize it to the DINO grid
    # before using it only as a visibility/support mask.
    feature_alpha = F.interpolate(
        alpha.permute(2, 0, 1).unsqueeze(0),
        size=(feature_height, feature_width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    if fixed_valid_mask is None:
        valid_support = feature_alpha.detach() >= alpha_threshold
    else:
        valid_support = fixed_valid_mask.to(device=feature_alpha.device, dtype=torch.bool)
    if target_valid_mask is not None:
        valid_support = valid_support & target_valid_mask.to(device=feature_alpha.device, dtype=torch.bool)
    valid_mask = (
        valid_support & torch.isfinite(predicted_features).all(dim=-1) & torch.isfinite(target_features).all(dim=-1)
    )
    assert bool(valid_mask.any()), "no valid feature pixels"

    predicted_features = F.normalize(predicted_features, dim=-1, eps=1e-6)
    target_features = F.normalize(target_features, dim=-1, eps=1e-6)
    per_pixel_loss = torch.abs(predicted_features - target_features).mean(dim=-1)
    return per_pixel_loss[valid_mask].mean(), valid_mask


def evaluate_featuremetric_pose(
    model: GaussianModel,
    decoder: FeatureDecoder,
    camera: ViewCamera,
    target_features: torch.Tensor,
    sh_degree: int,
    packed: bool,
    alpha_threshold: float,
    fixed_valid_mask: torch.Tensor | None = None,
    target_valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return feature loss and its valid-pixel mask at a fixed camera pose."""
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

    return featuremetric_loss(
        predicted_features,
        target_features,
        output.alpha,
        alpha_threshold,
        fixed_valid_mask=fixed_valid_mask,
        target_valid_mask=target_valid_mask,
    )


def single_view_loss(
    model: GaussianModel,
    decoder: FeatureDecoder,
    initial_rig_w2c: torch.Tensor,
    xi: torch.Tensor,
    view: RigView,
    fixed_mask: torch.Tensor | None,
    sh_degree: int,
    packed: bool,
    alpha_threshold: float,
) -> torch.Tensor:
    rig_w2c = compose_pose_update(xi, initial_rig_w2c)
    loss, _ = evaluate_featuremetric_pose(
        model,
        decoder,
        view_camera(view, view.camera_from_rig @ rig_w2c),
        view.target_features,
        sh_degree,
        packed,
        alpha_threshold,
        fixed_valid_mask=fixed_mask,
        target_valid_mask=view.target_valid_mask,
    )
    return loss


def accumulate_rig_loss(
    model: GaussianModel,
    decoder: FeatureDecoder,
    initial_rig_w2c: torch.Tensor,
    xi: torch.Tensor,
    views: list[RigView],
    fixed_masks: list[torch.Tensor | None],
    sh_degree: int,
    packed: bool,
    alpha_threshold: float,
    backward: bool,
) -> torch.Tensor:
    total = torch.zeros((), dtype=torch.float32, device=initial_rig_w2c.device)
    for view, fixed_mask in zip(views, fixed_masks, strict=True):
        loss = single_view_loss(
            model, decoder, initial_rig_w2c, xi, view, fixed_mask, sh_degree, packed, alpha_threshold
        ) / len(views)
        if backward:
            loss.backward()
            total = total + loss.detach()
        else:
            total = total + loss
    return total


def refine_rig_pose_bounded(
    checkpoint,
    views: list[RigView],
    initial_c2w: np.ndarray,
    steps: int,
    learning_rate: float,
    alpha_threshold: float,
    max_translation: float,
    max_rotation_deg: float,
    prior_weight: float,
) -> tuple[np.ndarray, dict]:
    """Refine one rig pose with fixed feature support and a bounded SE(3) update."""
    device = views[0].target_features.device
    initial = torch.as_tensor(initial_c2w, dtype=torch.float32, device=device)
    initial_w2c = torch.linalg.inv(initial)
    xi = torch.zeros(6, device=device, requires_grad=True)

    # Keep the same rendered pixels in the loss throughout pose optimization.
    with torch.no_grad():
        masks = [
            evaluate_featuremetric_pose(
                checkpoint.model,
                checkpoint.decoder,
                view_camera(view, view.camera_from_rig @ initial_w2c),
                view.target_features,
                checkpoint.sh_degree,
                checkpoint.packed,
                alpha_threshold,
                target_valid_mask=view.target_valid_mask,
            )[1].detach()
            for view in views
        ]

    def feature_loss(backward: bool) -> torch.Tensor:
        return accumulate_rig_loss(
            checkpoint.model,
            checkpoint.decoder,
            initial_w2c,
            xi,
            views,
            masks,
            checkpoint.sh_degree,
            checkpoint.packed,
            alpha_threshold,
            backward,
        )

    angle_limit = np.deg2rad(max_rotation_deg)

    def pose_prior() -> torch.Tensor:
        center = so3_exp_torch(xi[:3]) @ initial[:3, 3] + xi[3:]
        return prior_weight * (
            ((center - initial[:3, 3]) / max_translation).square().sum() + (xi[:3] / angle_limit).square().sum()
        )

    with torch.no_grad():
        initial_loss = float(feature_loss(False))
    best_xi = xi.detach().clone()
    best_objective, best_loss, best_step = 1.0, initial_loss, 0
    optimizer = torch.optim.Adam([xi], lr=learning_rate)

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        feature_loss(True)
        xi.grad.div_(max(initial_loss, 1e-8))
        pose_prior().backward()
        assert torch.isfinite(xi.grad).all()
        optimizer.step()

        with torch.no_grad():
            # Bound the rig center and rotation in the normalized map frame.
            xi[:3].mul_(min(1.0, angle_limit / max(float(xi[:3].norm()), 1e-8)))
            rotated_center = so3_exp_torch(xi[:3]) @ initial[:3, 3]
            delta = rotated_center + xi[3:] - initial[:3, 3]
            delta.mul_(min(1.0, max_translation / max(float(delta.norm()), 1e-8)))
            xi[3:].copy_(initial[:3, 3] + delta - rotated_center)

            if (step + 1) % 8 == 0 or step + 1 == steps:
                loss = float(feature_loss(False))
                objective = loss / max(initial_loss, 1e-8) + float(pose_prior())
                assert np.isfinite(objective)
                if objective < best_objective:
                    best_xi, best_objective, best_loss, best_step = xi.clone(), objective, loss, step + 1

    with torch.no_grad():
        final = torch.linalg.inv(compose_pose_update(best_xi, initial_w2c)).cpu().numpy()
    return final, {
        "initial_feature_loss": initial_loss,
        "final_feature_loss": best_loss,
        "best_objective": best_objective,
        "best_step": best_step,
        "supported_cameras": [view.camera_name for view in views],
        "fixed_valid_pixels": [int(mask.sum()) for mask in masks],
        "translation_change": float(np.linalg.norm(final[:3, 3] - initial_c2w[:3, 3])),
        "rotation_change_deg": float(np.rad2deg(np.linalg.norm(best_xi[:3].cpu().numpy()))),
    }


def refine_pose_featuremetric(
    model: GaussianModel,
    decoder: FeatureDecoder,
    camera: ViewCamera,
    target_features: torch.Tensor,
    sh_degree: int,
    packed: bool,
    alpha_threshold: float,
    steps: int = 100,
    learning_rate: float = 2e-3,
    freeze_alpha_mask: bool = True,
    target_valid_mask: torch.Tensor | None = None,
) -> FeatureMetricRefinement:
    """Optimize only a six-DoF SE(3) update against featuremetric loss."""
    device = target_features.device
    initial_w2c = torch.as_tensor(camera.w2c, dtype=torch.float32, device=device).detach()
    xi = torch.zeros(6, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([xi], lr=learning_rate)

    losses: list[float] = []
    best_loss = float("inf")
    best_xi = xi.detach().clone()
    best_iteration = 0
    fixed_valid_mask = None
    if freeze_alpha_mask:
        with torch.no_grad():
            _, fixed_valid_mask = evaluate_featuremetric_pose(
                model=model,
                decoder=decoder,
                camera=ViewCamera(
                    w2c=initial_w2c,
                    K=camera.K,
                    width=camera.width,
                    height=camera.height,
                ),
                target_features=target_features,
                sh_degree=sh_degree,
                packed=packed,
                alpha_threshold=alpha_threshold,
                target_valid_mask=target_valid_mask,
            )
        fixed_valid_mask = fixed_valid_mask.detach()

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)

        w2c = compose_pose_update(xi, initial_w2c)
        loss, _ = evaluate_featuremetric_pose(
            model=model,
            decoder=decoder,
            camera=ViewCamera(
                w2c=w2c,
                K=camera.K,
                width=camera.width,
                height=camera.height,
            ),
            target_features=target_features,
            sh_degree=sh_degree,
            packed=packed,
            alpha_threshold=alpha_threshold,
            fixed_valid_mask=fixed_valid_mask,
            target_valid_mask=target_valid_mask,
        )

        loss_value = float(loss.detach().cpu())
        if loss_value < best_loss:
            best_loss = loss_value
            best_xi = xi.detach().clone()
            best_iteration = step

        loss.backward()
        optimizer.step()

        losses.append(loss_value)

    with torch.no_grad():
        final_w2c = compose_pose_update(best_xi, initial_w2c)
        final_loss, _ = evaluate_featuremetric_pose(
            model=model,
            decoder=decoder,
            camera=ViewCamera(
                w2c=final_w2c,
                K=camera.K,
                width=camera.width,
                height=camera.height,
            ),
            target_features=target_features,
            sh_degree=sh_degree,
            packed=packed,
            alpha_threshold=alpha_threshold,
            fixed_valid_mask=fixed_valid_mask,
            target_valid_mask=target_valid_mask,
        )
    initial_loss = losses[0] if losses else float(final_loss.detach().cpu())
    return FeatureMetricRefinement(
        initial_w2c=initial_w2c.detach().cpu(),
        final_w2c=final_w2c.detach().cpu(),
        xi=best_xi.detach().cpu(),
        initial_loss=initial_loss,
        final_loss=float(final_loss.detach().cpu()),
        best_iteration=best_iteration,
        losses=tuple(losses),
    )
