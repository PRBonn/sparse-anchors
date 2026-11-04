"""Extract DINOv3 teacher features for registered COLMAP frames."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sparse_anchors.data.colmap import load_colmap_model
from sparse_anchors.features.base import extract_image_features
from sparse_anchors.features.dinov3 import DEFAULT_HF_MODEL, DinoV3FeatureExtractor
from sparse_anchors.splatting.features import (
    FeatureManifest,
    save_feature_manifest,
    save_feature_map,
)
from sparse_anchors.utils.image import load_mask, load_rgb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--masks-dir", type=Path)

    parser.add_argument("--dinov3-hf-model", default=DEFAULT_HF_MODEL)
    parser.add_argument("--long-side", type=int, default=768)
    parser.add_argument("--mask-dilation-pixels", type=int, default=14)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    output_dir = args.output_dir or args.scene / "features/dinov3"
    masks_dir = args.masks_dir or args.scene / "masks"

    device = torch.device(args.device)

    colmap = load_colmap_model(
        sparse_dir=args.scene / "sparse/0",
        images_dir=args.scene / "images",
    )

    extractor = DinoV3FeatureExtractor(
        device=device,
        hf_model=args.dinov3_hf_model,
    )

    image_paths: dict[str, Path] = {}

    for index, frame in enumerate(colmap.frames, start=1):
        relative_feature_path = Path("maps") / Path(frame.image_name).with_suffix(".npy")
        feature_path = output_dir / relative_feature_path

        image_paths[frame.image_name] = relative_feature_path

        if feature_path.is_file() and not args.overwrite:
            print(f"[{index}/{len(colmap.frames)}] exists: {frame.image_name}")
            continue

        valid_mask = load_mask(masks_dir / Path(frame.image_name).with_suffix(".png.png"))
        values, _ = extract_image_features(
            extractor,
            load_rgb(frame.image_path),
            long_side=args.long_side,
            valid_mask=valid_mask,
            mask_dilation_pixels=args.mask_dilation_pixels,
        )

        save_feature_map(feature_path, values)

        print(f"[{index}/{len(colmap.frames)}] wrote: {frame.image_name}")

    save_feature_manifest(
        output_dir / "manifest.json",
        FeatureManifest(
            embedding_dim=extractor.embedding_dim,
            extractor=extractor.name,
            image_paths=image_paths,
        ),
    )


if __name__ == "__main__":
    main()
