"""DINOv3 dense feature extraction."""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel

from sparse_anchors.features.base import ImageFeatureExtractor, image_to_tensor

DEFAULT_HF_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"


class DinoV3FeatureExtractor(ImageFeatureExtractor):
    """Extract DINOv3 patch tokens from a Hugging Face model."""

    def __init__(self, device: torch.device, hf_model: str = DEFAULT_HF_MODEL) -> None:
        self.device = device
        self.name = f"{hf_model.replace('/', '_').replace(':', '_')}_masked"
        self.model = AutoModel.from_pretrained(hf_model).eval().to(device)
        self.patch_size = self.model.config.patch_size
        self.embedding_dim = self.model.config.hidden_size

    @torch.inference_mode()
    def extract(self, image: np.ndarray, masks: np.ndarray | None = None) -> np.ndarray:
        feature_height = image.shape[0] // self.patch_size
        feature_width = image.shape[1] // self.patch_size
        tokens = self.model(pixel_values=image_to_tensor(image, self.device)).last_hidden_state
        patch_tokens = tokens[:, 1 + self.model.config.num_register_tokens :]
        if masks is not None:
            mask = torch.from_numpy(masks.reshape(1, -1)).to(device=self.device, dtype=torch.bool)
            patch_tokens = patch_tokens.masked_fill(mask[..., None], 0.0)
        return patch_tokens[0].reshape(feature_height, feature_width, -1).cpu().numpy().astype(np.float32)


def hf_model_from_manifest(manifest_extractor: str) -> str:
    """Recover the Hugging Face model ID from its saved extractor name."""
    return manifest_extractor.removesuffix("_masked").replace("_", "/", 1)
