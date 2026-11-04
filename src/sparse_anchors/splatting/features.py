"""Feature map extraction and save/load."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FeatureManifest:
    embedding_dim: int
    extractor: str
    image_paths: dict[str, Path]


def load_feature_manifest(path: Path) -> FeatureManifest:
    data = json.loads(path.read_text())
    return FeatureManifest(
        embedding_dim=data["embedding_dim"],
        extractor=data["extractor"],
        image_paths={
            image_name: path.parent / feature_path for image_name, feature_path in data["image_paths"].items()
        },
    )


def load_feature_map(manifest: FeatureManifest, image_name: str) -> np.ndarray:
    """Load a dense feature array [height, width, channels]."""
    return np.load(manifest.image_paths[image_name], allow_pickle=False)


def save_feature_manifest(path: Path, manifest: FeatureManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "embedding_dim": manifest.embedding_dim,
        "extractor": manifest.extractor,
        "image_paths": {image_name: str(feature_path) for image_name, feature_path in manifest.image_paths.items()},
    }
    path.write_text(json.dumps(data, indent=2))


def save_feature_map(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values, allow_pickle=False)


def decode_feature_map(decoder: FeatureDecoder, latents: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
    inputs = latents.permute(2, 0, 1).unsqueeze(0)  # [1, latent_dim, H, W]
    outputs = decoder(inputs)  # [1, embedding_dim, H, W]
    outputs = F.interpolate(outputs, size=output_size, mode="bilinear", align_corners=True)
    return outputs.squeeze(0).permute(1, 2, 0)  # [Hf, Wf, embedding_dim]


class FeatureDecoder(torch.nn.Module):
    def __init__(self, latent_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.network = torch.nn.Sequential(torch.nn.Conv2d(latent_dim, embedding_dim, 1))

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.network(latents)
