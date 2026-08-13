#!/usr/bin/env bash
# Runs the full pipeline (COLMAP sparse+dense, 3DGS training, Poisson mesh,
# classical-baseline rendering, evaluation) for a single scene.
#
# Usage: bash scripts/run_scene.sh scene1 [iterations]
set -euo pipefail

SCENE_NAME="${1:?usage: run_scene.sh <scene_name> [iterations]}"
ITERATIONS="${2:-30000}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

SCENE_DIR="dataset/${SCENE_NAME}"
GS_REPO="third_party/gaussian-splatting"

if [ ! -d "$SCENE_DIR/train" ]; then
  echo "ERROR: $SCENE_DIR/train not found. Expected raw captured photos there."
  exit 1
fi

echo "############################################"
echo "# Scene: ${SCENE_NAME}"
echo "############################################"

echo ""
echo "--- [1/5] COLMAP sparse SfM + undistortion + dense MVS ---"
python -m src.colmap_pipeline "$SCENE_DIR"

echo ""
echo "--- [2/5] 3D Gaussian Splatting training ---"
python -m src.train_splat "$SCENE_DIR" "$GS_REPO" --iterations "$ITERATIONS"

echo ""
echo "--- [3/5] Poisson surface reconstruction (classical mesh baseline) ---"
python -m src.poisson_baseline "$SCENE_DIR"

echo ""
echo "--- [4/5] Rendering classical baselines from held-out test poses ---"
python -m src.render_classical "$SCENE_DIR"

echo ""
echo "--- [5/5] Evaluation (PSNR/SSIM/coverage/timing/memory) ---"
python -m src.evaluate "$SCENE_DIR"

echo ""
echo "Done: ${SCENE_NAME}. See results/${SCENE_NAME}/metrics.json"
