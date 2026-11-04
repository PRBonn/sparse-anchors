"""Gaussian model state and initialization from sparse scene points."""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree

from sparse_anchors.splatting.scene import Scene

_SH_C0 = 0.28209479177387814
_LEARNING_RATES = {
    "means": 1.6e-4,
    "sh0": 2.5e-3,
    "shN": 1.25e-4,
    "opacities": 5e-2,
    "scales": 5e-3,
    "quats": 1e-3,
    "distill_features": 1e-3,
}


class GaussianModel:
    """Gaussian parameter state, initialization, and optimizer settings."""

    def __init__(self, params: torch.nn.ParameterDict) -> None:
        self.params = params

    @classmethod
    def from_scene(
        cls,
        scene: Scene,
        device: torch.device | str = "cuda",
        sh_degree: int = 3,
        feature_latent_dim: int = 128,
        init_opacity: float = 0.1,
    ) -> "GaussianModel":
        points = scene.colmap.points.positions
        colors = scene.colmap.points.colors

        distances, _ = cKDTree(points).query(points, k=4)
        spacing = np.sqrt((distances[:, 1:] ** 2).mean(axis=1))

        means = torch.tensor(points, dtype=torch.float32, device=device)
        rgbs = torch.tensor(colors, dtype=torch.float32, device=device) / 255.0
        scales = torch.log(torch.tensor(spacing, dtype=torch.float32, device=device)).unsqueeze(-1).repeat(1, 3)
        quats = torch.rand((len(points), 4), dtype=torch.float32, device=device)
        opacities = torch.logit(torch.full((len(points),), init_opacity, dtype=torch.float32, device=device))
        sh0 = ((rgbs - 0.5) / _SH_C0).unsqueeze(1)
        shN = torch.zeros(
            (len(points), (sh_degree + 1) ** 2 - 1, 3),
            dtype=torch.float32,
            device=device,
        )
        distill_features = torch.zeros((len(points), feature_latent_dim), dtype=torch.float32, device=device)

        return cls(
            torch.nn.ParameterDict(
                {
                    "means": torch.nn.Parameter(means),
                    "scales": torch.nn.Parameter(scales),
                    "quats": torch.nn.Parameter(quats),
                    "opacities": torch.nn.Parameter(opacities),
                    "sh0": torch.nn.Parameter(sh0),
                    "shN": torch.nn.Parameter(shN),
                    "distill_features": torch.nn.Parameter(distill_features),
                }
            )
        )

    @classmethod
    def from_checkpoint_state(
        cls,
        state: dict[str, torch.Tensor],
        device: torch.device | str,
    ) -> "GaussianModel":
        return cls(
            torch.nn.ParameterDict({name: torch.nn.Parameter(value.to(device)) for name, value in state.items()})
        )

    @property
    def device(self) -> torch.device:
        return self.params["means"].device

    def create_optimizers(self) -> dict[str, torch.optim.Adam]:
        adam_kwargs = {"eps": 1e-15, "betas": (0.9, 0.999)}
        if self.device.type == "cuda":
            adam_kwargs["fused"] = True

        return {
            name: torch.optim.Adam([{"params": self.params[name], "lr": lr, "name": name}], **adam_kwargs)
            for name, lr in _LEARNING_RATES.items()
        }
