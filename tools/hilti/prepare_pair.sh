#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 HILTI_DATA_ROOT REFERENCE_SESSION QUERY_SESSION" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="$(realpath "$1")"
PYTHON="$REPO_ROOT/.venv/bin/python"
REFERENCE_SESSION="$2"
QUERY_SESSION="$3"

for session in "$REFERENCE_SESSION" "$QUERY_SESSION"; do
  source_dir="$DATA_ROOT/$session"
  pinhole_dir="$DATA_ROOT/${session}_pinhole"

  echo "=== Convert $session to pinhole views ==="
  "$PYTHON" "$REPO_ROOT/tools/hilti/convert_to_pinhole.py" \
    --input "$source_dir" \
    --out "$pinhole_dir" \
    --clean

  echo "=== Reconstruct $session ==="
  PATH="$REPO_ROOT/.venv/bin:$PATH" \
    "$REPO_ROOT/tools/hilti/run_colmap_loma.sh" "$pinhole_dir"
done

echo "Prepared reference and query sessions under $DATA_ROOT"
