#!/usr/bin/env bash
# Converts a trained 3DGS model's point_cloud.ply into a .splat file for the
# antimatter15/splat web viewer.
#
# Usage: bash viewer/package_for_viewer.sh scene1 [iteration]
set -euo pipefail

SCENE_NAME="${1:?usage: package_for_viewer.sh <scene_name> [iteration]}"
ITERATION="${2:-30000}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

VIEWER_REPO="third_party/splat-viewer"
if [ ! -d "$VIEWER_REPO" ]; then
  git clone https://github.com/antimatter15/splat.git "$VIEWER_REPO"
fi

PLY_PATH="dataset/${SCENE_NAME}/gaussian_splat_model/point_cloud/iteration_${ITERATION}/point_cloud.ply"
if [ ! -f "$PLY_PATH" ]; then
  echo "ERROR: $PLY_PATH not found. Did training for ${SCENE_NAME} finish at iteration ${ITERATION}?"
  exit 1
fi

mkdir -p viewer/exports
OUT_SPLAT="viewer/exports/${SCENE_NAME}.splat"

python "$VIEWER_REPO/convert.py" "$PLY_PATH" --output "$OUT_SPLAT"

echo "Wrote $OUT_SPLAT"
echo "Serve it: cd $VIEWER_REPO && python -m http.server 8080"
echo "Then open: http://localhost:8080/?url=/../../$OUT_SPLAT  (or drag-drop the file onto the page)"
