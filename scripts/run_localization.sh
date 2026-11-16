#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python"
cd "$REPO_ROOT"

"$PYTHON" -m scripts.run_rig_pnp
"$PYTHON" -m scripts.run_sequence_consensus
"$PYTHON" -m scripts.run_pose_graph
"$PYTHON" -m scripts.run_consensus_featuremetric
"$PYTHON" -m scripts.evaluate
