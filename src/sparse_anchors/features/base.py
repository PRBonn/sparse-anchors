"""Common image feature extractor interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch

from sparse_anchors.utils.image import rgb_to_uint8

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageFeatureExtractor(ABC):
    """Backbone-specific dense image feature extraction."""

    name: str
    embedding_dim: int
    patch_size: int

    def resize_image(self, image: np.ndarray, long_side: int) -> np.ndarray:
        """Resize an RGB image with dimensions rounded down to patch multiples."""
        height, width = image.shape[:2]
        scale = long_side / max(height, width)

        resized_height = max(self.patch_size, int(height * scale) // self.patch_size * self.patch_size)
        resized_width = max(self.patch_size, int(width * scale) // self.patch_size * self.patch_size)
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        return cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)

    def invalid_patch_mask(self, valid_mask: np.ndarray, dilation_pixels: int = 0) -> np.ndarray:
        """Return a patch-grid mask where True means the feature patch is invalid."""
        valid_mask = valid_mask.astype(bool)
        invalid = ~valid_mask
        if dilation_pixels > 0:
            kernel = np.ones((dilation_pixels, dilation_pixels), dtype=np.uint8)
            invalid = cv2.dilate(invalid.astype(np.uint8), kernel).astype(bool)

        patch_height = valid_mask.shape[0] // self.patch_size
        patch_width = valid_mask.shape[1] // self.patch_size
        invalid = invalid[: patch_height * self.patch_size, : patch_width * self.patch_size]
        invalid = invalid.reshape(patch_height, self.patch_size, patch_width, self.patch_size)
        return invalid.any(axis=(1, 3))

    @abstractmethod
    def extract(self, image: np.ndarray, masks: np.ndarray | None = None) -> np.ndarray:
        """Extract dense feature values [Hf, Wf, D] from an RGB uint8 image."""


def extract_image_features(
    feature_extractor: ImageFeatureExtractor,
    image_rgb: np.ndarray,
    long_side: int,
    valid_mask: np.ndarray | None = None,
    mask_dilation_pixels: int = 0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return dense features and a valid-patch mask after resizing an RGB image."""
    image = feature_extractor.resize_image(rgb_to_uint8(image_rgb), long_side=long_side)
    invalid_patches = None
    if valid_mask is not None:
        valid_mask = cv2.resize(
            valid_mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        invalid_patches = feature_extractor.invalid_patch_mask(valid_mask, dilation_pixels=mask_dilation_pixels)
    values = feature_extractor.extract(image, masks=invalid_patches)
    return values, None if invalid_patches is None else ~invalid_patches


def image_to_tensor(
    image: np.ndarray,
    device: torch.device,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> torch.Tensor:
    """Convert an RGB uint8 image to normalized input tensor [1, 3, H, W]."""
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)
    return ((tensor - mean_tensor) / std_tensor).unsqueeze(0).to(device)
