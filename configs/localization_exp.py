from pathlib import Path

from sparse_anchors.localization.types import PnPConfig

reference_root = Path("/path/to/hilti/floor_1_2025-05-05_run_1_pinhole")
query_root = Path("/path/to/hilti/floor_1_2025-07-07_run_1_pinhole")

reference_sparse_dir = reference_root / "colmap_loma_rig" / "sparse" / "0"
reference_images_dir = reference_root / "colmap_loma_rig" / "images"
reference_frames_csv = reference_root / "frames.csv"
query_sparse_dir = query_root / "colmap_loma_rig" / "sparse" / "0"
query_images_dir = query_root / "colmap_loma_rig" / "images"
query_frames_csv = query_root / "frames.csv"

feature_manifest = reference_root / "colmap_loma_rig/features/dinov3/manifest.json"
checkpoint = Path("/path/to/hilti/splat_runs/reference/checkpoint_060000.pt")

device = "cuda"
reference_test_every = 8
output_dir = Path("/path/to/hilti/localization/floor1_2025-05-05_to_2025-07-07")

vpr_image_size = 322
vpr_batch_size = 8
vpr_top_k = 3
vpr_far_radius_fraction = 0.20
vpr_far_margin_threshold = 0.01
matcher_name = "loma"
image_pair_top_k = 8
pair_match_min_inliers = 20
match_min_inliers = 20
rig_camera_name = "cam0_left"
pnp = PnPConfig(reprojection_error_px=12.0, confidence=0.999, iterations=1000, min_points=20)

consensus_translation = 0.15
consensus_rotation_deg = 15.0
spatial_cell_fraction = 0.05
stable_min_anchors = 6
stable_min_cells = 3
stable_span_fraction = 0.30
stable_jackknife_p90 = 0.05

pgo_relative_translation_sigma = 0.005
pgo_relative_rotation_deg = 0.5
pgo_anchor_translation_sigma = 0.03
pgo_anchor_rotation_deg = 2.0
pgo_prior_translation_sigma = 0.05
pgo_prior_rotation_deg = 3.0

featuremetric_steps = 32
featuremetric_learning_rate = 0.001
featuremetric_alpha = 0.25
featuremetric_max_translation = 0.03
featuremetric_max_rotation_deg = 3.0
featuremetric_prior_weight = 0.01
featuremetric_long_side = 768
featuremetric_mask_dilation_pixels = 14

floorplan = None
floorplan_resolution_m = 0.01
correct_distance_m = 5.0
