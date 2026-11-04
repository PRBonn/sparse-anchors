"""Training loop for RGBD and feature distillation."""

from __future__ import annotations

import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gsplat.losses import ssim_loss
from gsplat.strategy import DefaultStrategy

from sparse_anchors.splatting.data import create_train_loader
from sparse_anchors.splatting.features import FeatureDecoder, decode_feature_map
from sparse_anchors.splatting.gaussians import GaussianModel
from sparse_anchors.splatting.renderer import ViewCamera, render
from sparse_anchors.splatting.scene import Scene, transform_scene


@dataclass(frozen=True)
class TrainerConfig:
    feature_loss_weight: float = 1
    ssim_lambda: float = 0.2
    max_steps: int = 30_000
    steps_scaler: float = 1.0
    sh_degree: int = 3
    sh_degree_interval: int = 1_000
    decoder_lr: float = 1e-3
    packed: bool = False
    data_loader_workers: int = 4
    data_loader_prefetch_factor: int = 2
    mask_dilation_pixels: int = 5
    normalize_feature_loss: bool = False
    da3_depth_loss_weight: float = 0.05
    da3_depth_start_step: int = 7_000
    da3_confidence_threshold: float = 1.3
    da3_alpha_threshold: float = 0.1
    da3_log_smooth_l1_beta: float = 0.05


class Trainer:
    def __init__(
        self,
        scene: Scene,
        model: GaussianModel,
        decoder: FeatureDecoder,
        config: TrainerConfig = TrainerConfig(),
        masks_dir: Path | None = None,
        da3_depth_dir: Path | None = None,
    ) -> None:
        if config.da3_depth_loss_weight > 0:
            if da3_depth_dir is None or not da3_depth_dir.is_dir():
                raise FileNotFoundError(f"DA3 depth directory does not exist: {da3_depth_dir}")
        else:
            da3_depth_dir = None

        self.scene = scene
        self.model = model
        self.decoder = decoder.to(model.device)
        self.config = config
        self.masks_dir = masks_dir
        self.da3_depth_dir = da3_depth_dir
        self.scene_depth_scale = float(np.linalg.norm(scene.world_transform[:3, 0]))
        self.viewer = None
        self.step = 0

        loader_workers = config.data_loader_workers if model.device.type == "cuda" else 0
        self.train_loader = create_train_loader(
            features=scene.features,
            frames=scene.train_frames,
            masks_dir=masks_dir,
            mask_dilation_pixels=config.mask_dilation_pixels,
            da3_depth_dir=da3_depth_dir,
            num_workers=loader_workers,
            prefetch_factor=config.data_loader_prefetch_factor,
            pin_memory=model.device.type == "cuda",
        )
        self._train_loader_iterator = None

        self.optimizers = model.create_optimizers()

        self.decoder_optimizer = torch.optim.Adam(
            self.decoder.parameters(),
            lr=config.decoder_lr,
        )

        self.means_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizers["means"],
            gamma=0.01 ** (1.0 / config.max_steps),
        )
        self.distill_features_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizers["distill_features"],
            gamma=(1.6e-6 / 1e-3) ** (1.0 / config.max_steps),
        )

        self.strategy = DefaultStrategy()
        for name in (
            "refine_start_iter",
            "refine_stop_iter",
            "reset_every",
            "refine_every",
            "refine_scale2d_stop_iter",
            "pause_refine_after_reset",
        ):
            setattr(self.strategy, name, int(getattr(self.strategy, name) * config.steps_scaler))
        self.strategy.check_sanity(self.model.params, self.optimizers)
        self.strategy_state = self.strategy.initialize_state(scene_scale=1.0)

    def _set_means_schedule_step(self, step: int) -> None:
        lrs = [base_lr * self.means_scheduler.gamma**step for base_lr in self.means_scheduler.base_lrs]
        for parameter_group, lr in zip(self.optimizers["means"].param_groups, lrs, strict=True):
            parameter_group["lr"] = lr
        self.means_scheduler.last_epoch = step
        self.means_scheduler._step_count = step + 1
        self.means_scheduler._last_lr = lrs

    def save_checkpoint(self, path: Path) -> None:
        """Save all state needed to resume training."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "step": self.step,
            "config": asdict(self.config),
            "model": self.model.params.state_dict(),
            "decoder": self.decoder.state_dict(),
            "decoder_config": {
                "latent_dim": self.decoder.latent_dim,
                "embedding_dim": self.decoder.embedding_dim,
            },
            "optimizers": {name: optimizer.state_dict() for name, optimizer in self.optimizers.items()},
            "means_scheduler": self.means_scheduler.state_dict(),
            "distill_features_scheduler": self.distill_features_scheduler.state_dict(),
            "decoder_optimizer": self.decoder_optimizer.state_dict(),
            "strategy_state": self.strategy_state,
            "world_transform": self.scene.world_transform.copy(),
            "feature_manifest": {
                "extractor": self.scene.features.extractor,
                "embedding_dim": self.scene.features.embedding_dim,
                "image_paths": {name: str(path) for name, path in self.scene.features.image_paths.items()},
            },
            "random_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        torch.save(payload, temporary_path)
        temporary_path.replace(path)

    @classmethod
    def load_checkpoint(
        cls,
        path: Path,
        scene: Scene,
        device: torch.device | str = "cuda",
        masks_dir: Path | None = None,
        da3_depth_dir: Path | None = None,
        mask_dilation_pixels: int | None = None,
        da3_confidence_threshold: float | None = None,
        data_loader_workers: int | None = None,
        data_loader_prefetch_factor: int | None = None,
    ) -> "Trainer":
        """Restore a trainer from a checkpoint saved by :meth:`save_checkpoint`."""
        payload = torch.load(path, map_location=device, weights_only=False)
        scene = transform_scene(scene, payload["world_transform"])
        model = GaussianModel.from_checkpoint_state(payload["model"], device=device)
        decoder = FeatureDecoder(**payload["decoder_config"])
        config_fields = {field.name for field in fields(TrainerConfig)}
        config = TrainerConfig(**{name: value for name, value in payload["config"].items() if name in config_fields})
        if mask_dilation_pixels is not None:
            config = replace(config, mask_dilation_pixels=mask_dilation_pixels)
        if da3_confidence_threshold is not None:
            config = replace(config, da3_confidence_threshold=da3_confidence_threshold)
        if data_loader_workers is not None:
            config = replace(config, data_loader_workers=data_loader_workers)
        if data_loader_prefetch_factor is not None:
            config = replace(config, data_loader_prefetch_factor=data_loader_prefetch_factor)

        trainer = cls(
            scene=scene,
            model=model,
            decoder=decoder,
            config=config,
            masks_dir=masks_dir,
            da3_depth_dir=da3_depth_dir,
        )
        trainer.decoder.load_state_dict(payload["decoder"])
        for name, optimizer_state in payload["optimizers"].items():
            trainer.optimizers[name].load_state_dict(optimizer_state)
        trainer.decoder_optimizer.load_state_dict(payload["decoder_optimizer"])
        trainer.step = payload["step"]
        if "means_scheduler" in payload:
            trainer.means_scheduler.load_state_dict(payload["means_scheduler"])
        else:
            trainer._set_means_schedule_step(trainer.step)
        if "distill_features_scheduler" in payload:
            trainer.distill_features_scheduler.load_state_dict(payload["distill_features_scheduler"])
        trainer.strategy_state = payload["strategy_state"]

        random.setstate(payload["random_state"])
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        if payload["cuda_rng_states"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng_states"]])

        return trainer

    def _da3_depth_loss(
        self,
        batch: dict,
        output_depth: torch.Tensor,
        output_alpha: torch.Tensor,
        valid_rgb_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if "da3_depth" not in batch or self.step < self.config.da3_depth_start_step:
            return torch.zeros((), dtype=output_depth.dtype, device=output_depth.device)

        device = output_depth.device
        output_size = output_depth.shape[:2]

        target_depth = batch["da3_depth"].to(device=device, non_blocking=True).unsqueeze(0).unsqueeze(0)
        confidence = batch["da3_confidence"].to(device=device, non_blocking=True).unsqueeze(0).unsqueeze(0)

        target_depth = (
            F.interpolate(target_depth, size=output_size, mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
        )
        confidence = (
            F.interpolate(confidence, size=output_size, mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
        )
        target_depth = target_depth * self.scene_depth_scale

        pred_depth = output_depth[..., 0]
        valid = torch.ones(output_size, dtype=torch.bool, device=device)
        if valid_rgb_mask is not None:
            valid &= valid_rgb_mask
        valid &= confidence >= self.config.da3_confidence_threshold
        valid &= torch.isfinite(target_depth) & (target_depth > 0.0)
        valid &= torch.isfinite(pred_depth) & (pred_depth > 0.0)
        valid &= output_alpha[..., 0].detach() > self.config.da3_alpha_threshold
        if not torch.any(valid):
            return torch.zeros((), dtype=output_depth.dtype, device=device)

        pred_log = torch.log(pred_depth[valid].clamp_min(1e-6))
        target_log = torch.log(target_depth[valid].clamp_min(1e-6))
        return F.smooth_l1_loss(pred_log, target_log, beta=self.config.da3_log_smooth_l1_beta)

    def train_step(self) -> dict[str, float]:
        if self.viewer is not None:
            while self.viewer.state == "paused":
                time.sleep(0.01)

        viewer_lock = self.viewer.lock if self.viewer is not None else nullcontext()
        train_start = time.perf_counter()
        with viewer_lock:
            metrics, num_train_rays = self._train_step()

        if self.viewer is not None:
            train_seconds = time.perf_counter() - train_start
            self.viewer.render_tab_state.num_train_rays_per_sec = num_train_rays / max(train_seconds, 1e-10)
            self.viewer.update(self.step, num_train_rays)
        return metrics

    def _train_step(self) -> tuple[dict[str, float], int]:
        device = self.model.device

        if self._train_loader_iterator is None:
            self._train_loader_iterator = iter(self.train_loader)
        batch = next(self._train_loader_iterator, None)
        if batch is None:
            self._train_loader_iterator = iter(self.train_loader)
            batch = next(self._train_loader_iterator)

        target_rgb = batch["rgb"].to(device, non_blocking=True)
        target_features = batch["features"].to(device, non_blocking=True)
        camera = ViewCamera(
            w2c=batch["w2c"].to(device, non_blocking=True),
            K=batch["K"].to(device, non_blocking=True),
            width=target_rgb.shape[1],
            height=target_rgb.shape[0],
        )
        num_train_rays = target_rgb.shape[0] * target_rgb.shape[1]
        valid_rgb_mask = batch.get("valid_rgb_mask")
        valid_feature_mask = batch.get("valid_feature_mask")
        if valid_rgb_mask is not None:
            valid_rgb_mask = valid_rgb_mask.to(device, non_blocking=True)
            valid_feature_mask = valid_feature_mask.to(device, non_blocking=True)

        active_sh_degree = min(
            self.config.sh_degree,
            self.step // self.config.sh_degree_interval,
        )
        feature_loss_active = self.step >= self.config.da3_depth_start_step

        output = render(
            self.model,
            camera,
            sh_degree=active_sh_degree,
            packed=self.config.packed,
            absgrad=self.strategy.absgrad,
            render_latents=feature_loss_active,
        )

        if valid_rgb_mask is None:
            l1_loss = F.l1_loss(output.rgb, target_rgb)
            ssim_rgb = output.rgb
            ssim_target = target_rgb
        else:
            l1_loss = F.l1_loss(output.rgb[valid_rgb_mask], target_rgb[valid_rgb_mask])
            ssim_rgb = output.rgb.masked_fill(~valid_rgb_mask[..., None], 0.0)
            ssim_target = target_rgb.masked_fill(~valid_rgb_mask[..., None], 0.0)
        ssim = ssim_loss(
            ssim_rgb.permute(2, 0, 1).unsqueeze(0),
            ssim_target.permute(2, 0, 1).unsqueeze(0),
        )
        rgb_loss = torch.lerp(l1_loss, ssim, self.config.ssim_lambda)

        if feature_loss_active:
            decoded_features = decode_feature_map(
                self.decoder, output.latents, output_size=tuple(target_features.shape[:2])
            )
            if self.config.normalize_feature_loss:
                decoded_features = F.normalize(decoded_features, dim=-1, eps=1e-6)
                target_features = F.normalize(target_features, dim=-1, eps=1e-6)
            if valid_feature_mask is not None:
                feature_loss = F.l1_loss(decoded_features[valid_feature_mask], target_features[valid_feature_mask])
            else:
                feature_loss = F.l1_loss(decoded_features, target_features)
        else:
            feature_loss = torch.zeros((), dtype=rgb_loss.dtype, device=device)

        da3_depth_loss = self._da3_depth_loss(
            batch=batch,
            output_depth=output.depth,
            output_alpha=output.alpha,
            valid_rgb_mask=valid_rgb_mask,
        )

        loss = (
            rgb_loss
            + self.config.feature_loss_weight * feature_loss
            + self.config.da3_depth_loss_weight * da3_depth_loss
        )

        self.strategy.step_pre_backward(self.model.params, self.optimizers, self.strategy_state, self.step, output.info)

        loss.backward()

        for optimizer in self.optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        self.decoder_optimizer.step()
        self.decoder_optimizer.zero_grad(set_to_none=True)

        self.strategy.step_post_backward(
            self.model.params,
            self.optimizers,
            self.strategy_state,
            self.step,
            output.info,
            packed=self.config.packed,
        )
        self.means_scheduler.step()
        self.distill_features_scheduler.step()

        metrics = {
            "loss": loss.item(),
            "rgb_loss": rgb_loss.item(),
            "feature_loss": feature_loss.item(),
            "da3_depth_loss": da3_depth_loss.item(),
            "sh_degree": float(active_sh_degree),
            "num_gaussians": float(len(self.model.params["means"])),
        }

        self.step += 1
        return metrics, int(num_train_rays)
