"""Load trained maps and cached depth."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from sparse_anchors.splatting.features import FeatureDecoder
from sparse_anchors.splatting.gaussians import GaussianModel


@dataclass(frozen=True)
class SplatCheckpoint:
    model: GaussianModel
    decoder: FeatureDecoder
    sh_degree: int
    packed: bool
    feature_manifest: dict
    world_transform: np.ndarray


def load_splat_checkpoint(path: Path, device: torch.device | str = "cuda") -> SplatCheckpoint:
    """Load inference state; training resume belongs to Trainer.load_checkpoint."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = GaussianModel.from_checkpoint_state(payload["model"], device=device)
    decoder = FeatureDecoder(**payload["decoder_config"]).to(device)
    decoder.load_state_dict(payload["decoder"])
    model.params.requires_grad_(False)
    decoder.eval().requires_grad_(False)

    config = payload["config"]
    return SplatCheckpoint(
        model=model,
        decoder=decoder,
        sh_degree=min(int(config["sh_degree"]), int(payload["step"]) // int(config["sh_degree_interval"])),
        packed=bool(config["packed"]),
        feature_manifest=payload["feature_manifest"],
        world_transform=np.asarray(payload["world_transform"], dtype=np.float64),
    )


def load_da3_depth_map(depth_dir: Path, image_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return camera-z depth and confidence arrays for one COLMAP image."""
    image_path = Path(image_name)
    timestamp = image_path.stem
    cache_path = Path(depth_dir) / f"{timestamp}.npz"
    with np.load(cache_path, allow_pickle=False) as payload:
        image_names = payload["image_names"].astype(str)
        index = int(np.flatnonzero(image_names == image_name)[0])
        return (
            np.asarray(payload["depth"][index], dtype=np.float32),
            np.asarray(payload["conf"][index], dtype=np.float32),
        )
