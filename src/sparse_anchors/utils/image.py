"""RGB and mask I/O and pixel preprocessing.

RGB arrays use float32 values in [0, 1]. Masks use True for valid pixels.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return mask > 0


def rgb_to_uint8(image_rgb: np.ndarray) -> np.ndarray:
    """Convert float RGB in [0, 1] to uint8; leave uint8 images unchanged."""
    if image_rgb.dtype == np.uint8:
        return image_rgb
    return (np.clip(image_rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def dilate_invalid_mask(valid_mask: np.ndarray, pixels: int) -> np.ndarray:
    """Expand invalid mask regions by ``pixels`` and return the remaining valid mask."""
    if pixels <= 0:
        return valid_mask

    invalid = (~valid_mask).astype(np.uint8)
    kernel_size = 2 * pixels + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_invalid = cv2.dilate(invalid, kernel)
    return dilated_invalid == 0


def write_rgb(path: Path, image: np.ndarray | torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = torch.as_tensor(image).detach().clamp(0.0, 1.0).cpu().numpy()
    rgb = (rgb * 255.0).round().astype(np.uint8)
    path.write_bytes(cv2.imencode(path.suffix, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))[1].tobytes())


def downsample_valid_mask(mask: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
    """Keep a feature cell only when its source image region is fully valid."""
    values = F.interpolate(
        mask.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0),
        size=output_size,
        mode="area",
    )
    return values[0, 0] >= 0.999
