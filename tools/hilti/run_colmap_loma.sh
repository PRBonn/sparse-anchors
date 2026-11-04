#!/usr/bin/env bash
set -euo pipefail

# Reconstruct one prepared 8-camera HILTI scene.
# Input: <scene>/images/<camera>/<filename>, frames.csv, cameras.json
# Output: <scene>/colmap_loma_rig/{images,sparse/0}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 SCENE_DIR [OUTPUT_DIR]" >&2
  exit 2
fi

SCENE_DIR="$(realpath "$1")"
WORK="$(realpath -m "${2:-$SCENE_DIR/colmap_loma_rig}")"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGES="$SCENE_DIR/images"
DB="$WORK/database.db"
SPARSE="$WORK/sparse"

for command_name in colmap python3 sqlite3; do
  command -v "$command_name" >/dev/null || { echo "Missing command: $command_name" >&2; exit 1; }
done
[[ -d "$IMAGES" ]] || { echo "Missing: $IMAGES" >&2; exit 1; }
[[ -f "$SCENE_DIR/cameras.json" ]] || { echo "Missing: $SCENE_DIR/cameras.json" >&2; exit 1; }
[[ -f "$SCENE_DIR/frames.csv" ]] || { echo "Missing: $SCENE_DIR/frames.csv" >&2; exit 1; }

mkdir -p "$WORK"
if [[ ! -e "$WORK/images" && ! -L "$WORK/images" ]]; then
  ln -s "$IMAGES" "$WORK/images"
fi
if [[ "$(realpath "$WORK/images")" != "$IMAGES" ]]; then
  echo "Existing $WORK/images does not point to $IMAGES" >&2
  exit 1
fi

# Rebuild COLMAP's generated artifacts while retaining scene-level feature caches.
rm -f "$DB"
rm -rf "$SPARSE" "$WORK/image_lists"
mkdir -p "$SPARSE"
python3 "$REPO_ROOT/tools/hilti/prepare_colmap_rig.py" "$SCENE_DIR" "$WORK"

echo "=== [1/4] LoMa-B feature extraction ==="
while IFS=$'\t' read -r camera list params count; do
  echo "[$camera] $count images"
  colmap feature_extractor \
    --database_path "$DB" \
    --image_path "$IMAGES" \
    --image_list_path "$list" \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.camera_params "$params" \
    --ImageReader.single_camera 1 \
    --FeatureExtraction.type LOMA_B \
    --FeatureExtraction.use_gpu 1 \
    --FeatureExtraction.gpu_index 0 \
    --LomaExtraction.max_num_features 4096 \
    --LomaExtraction.use_bf16 0 \
    --LomaExtraction.use_fast_resize 0
done < "$WORK/cameras.tsv"

echo "=== [2/4] Rig configuration ==="
colmap rig_configurator \
  --database_path "$DB" \
  --rig_config_path "$WORK/rig_config.json"
sqlite3 "$DB" "UPDATE cameras SET prior_focal_length=1;"

RIGS="$(sqlite3 "$DB" 'SELECT COUNT(*) FROM rigs;')"
FRAMES="$(sqlite3 "$DB" 'SELECT COUNT(*) FROM frames;')"
CAMERAS="$(sqlite3 "$DB" 'SELECT COUNT(*) FROM cameras;')"
IMAGES_COUNT="$(sqlite3 "$DB" 'SELECT COUNT(*) FROM images;')"
echo "rigs=$RIGS frames=$FRAMES cameras=$CAMERAS images=$IMAGES_COUNT"
[[ "$RIGS" == "1" ]] || { echo "Expected 1 rig, got $RIGS" >&2; exit 1; }
[[ "$CAMERAS" == "8" ]] || { echo "Expected 8 cameras, got $CAMERAS" >&2; exit 1; }
[[ $((FRAMES * 8)) -eq "$IMAGES_COUNT" ]] || {
  echo "Expected images = 8 * frames, got $IMAGES_COUNT images and $FRAMES frames" >&2
  exit 1
}

echo "=== [3/4] LoMa-G rig-aware sequential matching ==="
colmap sequential_matcher \
  --database_path "$DB" \
  --FeatureMatching.type LOMA_B \
  --FeatureMatching.use_gpu 1 \
  --FeatureMatching.gpu_index 0 \
  --FeatureMatching.rig_verification 1 \
  --FeatureMatching.skip_geometric_verification 0 \
  --FeatureMatching.skip_image_pairs_in_same_frame 0 \
  --FeatureMatching.guided_matching 0 \
  --SequentialMatching.overlap 5 \
  --SequentialMatching.expand_rig_images 1 \
  --SequentialMatching.quadratic_overlap 1 \
  --SequentialMatching.loop_detection 0 \
  --LomaMatching.min_score 0.1 \
  --LomaMatching.use_bf16 0 \
  --TwoViewGeometry.max_error 2.0 \
  --TwoViewGeometry.min_num_inliers 30 \
  --TwoViewGeometry.min_inlier_ratio 0.25 \
  --TwoViewGeometry.confidence 0.999 \
  --TwoViewGeometry.max_num_trials 10000

sqlite3 "$DB" "
SELECT 'raw_pairs',COUNT(*),'raw_matches',COALESCE(SUM(rows),0) FROM matches;
SELECT 'verified_pairs',COUNT(*),'verified_inliers',COALESCE(SUM(rows),0)
FROM two_view_geometries WHERE rows>0;
"

echo "=== [4/4] Sparse reconstruction ==="
colmap mapper \
  --database_path "$DB" \
  --image_path "$IMAGES" \
  --output_path "$SPARSE" \
  --Mapper.ba_refine_sensor_from_rig 0 \
  --Mapper.ba_refine_focal_length 0 \
  --Mapper.ba_refine_principal_point 0 \
  --Mapper.ba_refine_extra_params 0 \
  --Mapper.ba_use_gpu 1 \
  --Mapper.ba_gpu_index 0 \
  --Mapper.ba_global_frames_ratio 1.4 \
  --Mapper.ba_global_points_ratio 1.4

MODEL_COUNT="$(find "$SPARSE" -mindepth 1 -maxdepth 1 -type d | wc -l)"
echo "Done. Workspace: $WORK; sparse models: $MODEL_COUNT"
if [[ "$MODEL_COUNT" -ne 1 ]]; then
  echo "Warning: reconstruction produced $MODEL_COUNT disconnected models." >&2
fi
