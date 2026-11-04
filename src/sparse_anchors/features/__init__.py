"""DINOv3 dense features and the shared extractor interface."""

from sparse_anchors.features.base import ImageFeatureExtractor
from sparse_anchors.features.dinov3 import DinoV3FeatureExtractor

__all__ = [
    "ImageFeatureExtractor",
    "DinoV3FeatureExtractor",
]
