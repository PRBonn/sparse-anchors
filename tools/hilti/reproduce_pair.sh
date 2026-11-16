#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 QUERY_DYNAMIC_MASKS_DIR" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python"
cd "$REPO_ROOT"

readarray -t EXPERIMENT_PATHS < <("$PYTHON" - <<'PY'
from configs import localization_exp as config

for value in (
    config.data_root,
    config.reference_session,
    config.query_session,
    config.reference_root,
    config.query_root,
):
    print(value)
PY
)

DATA_ROOT="$(realpath "${EXPERIMENT_PATHS[0]}")"
REFERENCE_SOURCE_SESSION="$(basename "${EXPERIMENT_PATHS[1]}")"
QUERY_SOURCE_SESSION="$(basename "${EXPERIMENT_PATHS[2]}")"
REFERENCE_SOURCE_SESSION="${REFERENCE_SOURCE_SESSION%_pinhole}"
QUERY_SOURCE_SESSION="${QUERY_SOURCE_SESSION%_pinhole}"
REFERENCE_ROOT="${EXPERIMENT_PATHS[3]}"
QUERY_ROOT="${EXPERIMENT_PATHS[4]}"
QUERY_MASKS_DIR="$(realpath "$1")"

"$REPO_ROOT/tools/hilti/prepare_pair.sh" \
  "$DATA_ROOT" "$REFERENCE_SOURCE_SESSION" "$QUERY_SOURCE_SESSION"

mkdir -p "$QUERY_ROOT/masks"
cp -R "$QUERY_MASKS_DIR"/. "$QUERY_ROOT/masks/"

"$REPO_ROOT/tools/hilti/run_prepared_pair.sh"
