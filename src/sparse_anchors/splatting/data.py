"""Training data loading for Gaussian splatting."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from sparse_anchors.data.colmap import Frame
from sparse_anchors.splatting.features import FeatureManifest, load_feature_map
from sparse_anchors.utils.image import dilate_invalid_mask, downsample_valid_mask, load_mask, load_rgb
from sparse_anchors.utils.io import load_da3_depth_map


class TrainingDataset(Dataset[dict]):
    """Load one COLMAP view and its feature-supervision targets."""

    def __init__(
        self,
        features: FeatureManifest,
        frames: tuple[Frame, ...],
        masks_dir: Path | None = None,
        mask_dilation_pixels: int = 0,
        da3_depth_dir: Path | None = None,
    ) -> None:
        self.features = features
        self.frames = frames
        self.masks_dir = masks_dir
        self.mask_dilation_pixels = mask_dilation_pixels
        self.da3_depth_dir = da3_depth_dir

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> dict:
        frame = self.frames[index]
        features = torch.from_numpy(load_feature_map(self.features, frame.image_name)).float()
        batch = {
            "w2c": torch.from_numpy(frame.w2c).float(),
            "K": torch.from_numpy(frame.K).float(),
            "rgb": torch.from_numpy(load_rgb(frame.image_path)),
            "features": features,
        }

        if self.masks_dir is not None:
            mask_path = self.masks_dir / Path(frame.image_name).with_suffix(".png.png")
            valid_rgb_mask = torch.from_numpy(dilate_invalid_mask(load_mask(mask_path), self.mask_dilation_pixels))
            batch["valid_rgb_mask"] = valid_rgb_mask
            batch["valid_feature_mask"] = downsample_valid_mask(valid_rgb_mask, tuple(features.shape[:2]))

        if self.da3_depth_dir is not None:
            depth, confidence = load_da3_depth_map(self.da3_depth_dir, frame.image_name)
            batch["da3_depth"] = torch.from_numpy(depth)
            batch["da3_confidence"] = torch.from_numpy(confidence)

        return batch


def create_train_loader(
    features: FeatureManifest,
    frames: tuple[Frame, ...],
    masks_dir: Path | None,
    mask_dilation_pixels: int,
    da3_depth_dir: Path | None,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
) -> DataLoader[dict]:
    """Load one view per step, with optional asynchronous prefetching."""
    return DataLoader(
        TrainingDataset(features, frames, masks_dir, mask_dilation_pixels, da3_depth_dir),
        batch_size=None,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=pin_memory,
    )
