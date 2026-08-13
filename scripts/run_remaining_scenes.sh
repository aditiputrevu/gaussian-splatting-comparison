#!/usr/bin/env bash
# Runs the full pipeline for one or more scenes automatically, using the
# exact settings validated on scene1: 15,000 3DGS iterations, half-resolution
# training (-r 2), COLMAP dense MVS, Poisson mesh, classical-baseline
# rendering (software-rendered via LIBGL_ALWAYS_SOFTWARE to avoid the WSL2
# EGL segfault), and full evaluation.
#
# Usage:
#   bash scripts/run_remaining_scenes.sh scene2 scene3 scene4
#   bash scripts/run_remaining_scenes.sh scene2          # just one scene
#
# Safe to re-run: if a scene's colmap/dense/fused.ply already exists, COLMAP
# steps are skipped for that scene so you don't redo already-finished work.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

GS_REPO="third_party/gaussian-splatting"
ITERATIONS=15000

export LIBGL_ALWAYS_SOFTWARE=1   # avoids the Open3D/EGL segfault seen under WSL2

if [ "$#" -eq 0 ]; then
  echo "Usage: bash scripts/run_remaining_scenes.sh <scene1> [<scene2> ...]"
  exit 1
fi

for SCENE_NAME in "$@"; do
  SCENE_DIR="dataset/${SCENE_NAME}"
  echo ""
  echo "############################################"
  echo "# Scene: ${SCENE_NAME}"
  echo "############################################"

  if [ ! -d "$SCENE_DIR/train" ]; then
    echo "ERROR: $SCENE_DIR/train not found -- skipping ${SCENE_NAME}."
    continue
  fi

  FUSED_PLY="$SCENE_DIR/colmap/dense/fused.ply"
  if [ -f "$FUSED_PLY" ]; then
    echo "--- [1/5] COLMAP already done for ${SCENE_NAME} (found fused.ply), skipping ---"
  else
    echo "--- [1/5] COLMAP sparse SfM + undistortion + dense MVS ---"
    python -m src.colmap_pipeline "$SCENE_DIR"
  fi

  MODEL_DIR="$SCENE_DIR/gaussian_splat_model"
  FINAL_PLY="$MODEL_DIR/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"
  if [ -f "$FINAL_PLY" ]; then
    echo "--- [2/5] 3DGS training already done for ${SCENE_NAME} (found iteration_${ITERATIONS}), skipping ---"
  else
    echo "--- [2/5] 3D Gaussian Splatting training (${ITERATIONS} iterations, half resolution) ---"
    python "$GS_REPO/train.py" \
      -s "$SCENE_DIR/colmap/dense" \
      -m "$MODEL_DIR" \
      --iterations "$ITERATIONS" \
      --eval \
      -r 2
  fi

  RESULTS_JSON="$MODEL_DIR/results.json"
  if [ -f "$RESULTS_JSON" ]; then
    echo "--- [3/5] 3DGS render/metrics already done for ${SCENE_NAME}, skipping ---"
  else
    echo "--- [3/5] Rendering test views + computing 3DGS PSNR/SSIM/LPIPS ---"
    python "$GS_REPO/render.py" -m "$MODEL_DIR" --skip_train
    python "$GS_REPO/metrics.py" -m "$MODEL_DIR"
  fi

  # Fold 3DGS results.json into results_3dgs_timing.json for evaluate.py.
  # Training time is estimated from the model folder's mtime range as a
  # reasonable approximation (we don't have the GPU-memory poller here since
  # we're calling train.py directly rather than through src/train_splat.py).
  python3 - "$SCENE_DIR" "$RESULTS_JSON" << 'PYEOF'
import json, sys
from pathlib import Path

scene_dir = Path(sys.argv[1])
results_json = Path(sys.argv[2])
raw = json.loads(results_json.read_text())
last_key = sorted(raw.keys())[-1]
metrics = raw[last_key]

out = {
    "train_time_sec": None,  # not measured for scenes run via this script
    "peak_gpu_memory_mb": None,
    **metrics,
}
(scene_dir / "results_3dgs_timing.json").write_text(json.dumps(out, indent=2))
print(f"Wrote {scene_dir / 'results_3dgs_timing.json'}")
PYEOF

  POISSON_MESH="$SCENE_DIR/colmap/dense/poisson_mesh.ply"
  if [ -f "$POISSON_MESH" ]; then
    echo "--- [4/5] Poisson mesh already done for ${SCENE_NAME}, skipping ---"
  else
    echo "--- [4/5] Poisson surface reconstruction ---"
    python -m src.poisson_baseline "$SCENE_DIR"
  fi

  if [ -d "$SCENE_DIR/renders_dense" ] && [ -d "$SCENE_DIR/renders_poisson" ]; then
    echo "--- [5/5] Classical baseline renders already done for ${SCENE_NAME}, skipping ---"
  else
    echo "--- [5/5] Rendering classical baselines from held-out test poses ---"
    python -m src.render_classical "$SCENE_DIR"
  fi

  echo "--- Evaluating ${SCENE_NAME} ---"
  python -m src.evaluate "$SCENE_DIR"

  echo "Done: ${SCENE_NAME}. See results/${SCENE_NAME}/metrics.json"
done

echo ""
echo "=== Aggregating all available scenes ==="
python src/aggregate_results.py
