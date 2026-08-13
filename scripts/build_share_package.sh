#!/usr/bin/env bash
# Builds one combined zip to share with teammates: raw .ply files per scene,
# converted .splat files + web viewer, the comparison table/figures, and a
# README explaining how to view everything -- no GPU or setup needed on
# their end, just free viewer apps + Python for the local web server.
#
# Usage: bash scripts/build_share_package.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== [1/4] Gathering raw .ply files per scene ==="
rm -rf share
mkdir -p share/splats

for s in scene1 scene2 scene3 scene4; do
  mkdir -p "share/$s"
  cp "dataset/$s/colmap/dense/fused.ply" "share/$s/dense_point_cloud.ply" 2>/dev/null || echo "  (missing dense cloud for $s)"
  cp "dataset/$s/colmap/dense/poisson_mesh.ply" "share/$s/poisson_mesh.ply" 2>/dev/null || echo "  (missing poisson mesh for $s)"

  # Find whichever iteration folder actually exists (7000 or 15000)
  # Prefer the final iteration_15000 checkpoint; fall back to whatever's there.
  if [ -f "dataset/$s/gaussian_splat_model/point_cloud/iteration_15000/point_cloud.ply" ]; then
    PLY="dataset/$s/gaussian_splat_model/point_cloud/iteration_15000/point_cloud.ply"
  else
    PLY=$(find "dataset/$s/gaussian_splat_model/point_cloud" -name "point_cloud.ply" 2>/dev/null | sort -t_ -k2 -n | tail -1)
  fi
  if [ -n "$PLY" ]; then
    cp "$PLY" "share/$s/3dgs_splat_model.ply"
  else
    echo "  (missing 3DGS point cloud for $s)"
  fi
done

echo ""
echo "=== [2/4] Converting each scene's 3DGS model to .splat for the web viewer ==="
if [ ! -d "third_party/splat-viewer" ]; then
  git clone https://github.com/antimatter15/splat.git third_party/splat-viewer
fi

for s in scene1 scene2 scene3 scene4; do
  if [ -f "dataset/$s/gaussian_splat_model/point_cloud/iteration_15000/point_cloud.ply" ]; then
    PLY="dataset/$s/gaussian_splat_model/point_cloud/iteration_15000/point_cloud.ply"
  else
    PLY=$(find "dataset/$s/gaussian_splat_model/point_cloud" -name "point_cloud.ply" 2>/dev/null | sort -t_ -k2 -n | tail -1)
  fi
  if [ -n "$PLY" ]; then
    python third_party/splat-viewer/convert.py "$PLY" --output "share/splats/$s.splat"
  fi
done

echo ""
echo "=== [3/4] Copying the viewer app + comparison results + README ==="
cp -r third_party/splat-viewer share/splat-viewer
rm -rf share/splat-viewer/.git

mkdir -p share/comparison_results
cp -r report_figures share/comparison_results/ 2>/dev/null || true
cp results/comparison_table.md share/comparison_results/ 2>/dev/null || true
cp results/comparison_table.csv share/comparison_results/ 2>/dev/null || true
cp results/comparison_figures.png share/comparison_results/ 2>/dev/null || true

cp share_package_README.md share/README.md

echo ""
echo "=== [4/4] Zipping everything ==="
zip -r all_scenes_3d_models.zip share/ > /dev/null
echo "Wrote all_scenes_3d_models.zip"
echo ""
echo "Send this single zip file to your teammates. Tell them to start with README.md inside."
