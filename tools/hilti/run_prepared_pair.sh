#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "Usage: $0" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python"
CLI="$REPO_ROOT/.venv/bin/sparse-anchors"
cd "$REPO_ROOT"

readarray -t EXPERIMENT_PATHS < <("$PYTHON" - <<'PY'
from configs import localization_exp as config

for value in (
    config.reference_root,
    config.query_root,
    config.reference_splat_run,
    config.output_dir,
):
    print(value)
PY
)

REFERENCE_ROOT="${EXPERIMENT_PATHS[0]}"
QUERY_ROOT="${EXPERIMENT_PATHS[1]}"
REFERENCE_SCENE="$REFERENCE_ROOT/colmap_loma_rig"
REFERENCE_SPLAT_RUN="${EXPERIMENT_PATHS[2]}"
LOCALIZATION_OUTPUT_DIR="${EXPERIMENT_PATHS[3]}"

echo "=== Cache reference DA3 depth ==="
"$PYTHON" "$REPO_ROOT/tools/hilti/cache_da3_depth.py" --scene "$REFERENCE_SCENE"

echo "=== Extract reference DINOv3 features ==="
"$PYTHON" "$REPO_ROOT/tools/splatting/extract_features.py" \
  --scene "$REFERENCE_SCENE" \
  --masks-dir "$REFERENCE_ROOT/masks"

echo "=== Train reference Gaussian map ==="
"$CLI" train \
  --scene "$REFERENCE_SCENE" \
  --masks-dir "$REFERENCE_ROOT/masks" \
  --output-dir "$REFERENCE_SPLAT_RUN" \
  --steps 30000 \
  --steps-scaler 2 \
  --checkpoint-every 5000 \
  --log-every 100 \
  --data-loader-workers 4 \
  --data-loader-prefetch-factor 2 \
  --da3-confidence-threshold 1.3

echo "=== Localize and evaluate query session ==="
"$REPO_ROOT/scripts/run_localization.sh"

echo "Done. Gaussian map: $REFERENCE_SPLAT_RUN"
echo "Localization results: $LOCALIZATION_OUTPUT_DIR"
