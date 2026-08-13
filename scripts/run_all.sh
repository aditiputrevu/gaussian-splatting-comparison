#!/usr/bin/env bash
# Runs the full pipeline for all scenes found in dataset/, then aggregates.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

for scene_dir in dataset/scene*/; do
  scene_name="$(basename "$scene_dir")"
  bash scripts/run_scene.sh "$scene_name"
done

echo ""
echo "=== Aggregating results across all scenes ==="
python src/aggregate_results.py
