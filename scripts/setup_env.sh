#!/usr/bin/env bash
# One-time setup: COLMAP, official 3D Gaussian Splatting repo, conda env.
# Run from the gaussian-splatting-comparison/ root:
#   bash scripts/setup_env.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== [1/5] Checking for NVIDIA GPU ==="
if ! command -v nvidia-smi &> /dev/null; then
  echo "WARNING: nvidia-smi not found. 3DGS training and COLMAP dense stereo"
  echo "will be extremely slow (or COLMAP dense stereo unavailable) on CPU only."
  echo "Continuing anyway, but strongly recommend running on a CUDA-capable machine."
else
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

echo ""
echo "=== [2/5] Checking for COLMAP ==="
if ! command -v colmap &> /dev/null; then
  echo "COLMAP not found. Installing via apt (requires sudo)..."
  sudo apt-get update
  sudo apt-get install -y colmap
  # If your Ubuntu's apt COLMAP is too old / lacks CUDA support, build from source instead:
  # https://colmap.github.io/install.html
else
  echo "Found: $(colmap -h 2>&1 | head -n 1)"
fi

echo ""
echo "=== [3/5] Cloning official 3D Gaussian Splatting repo (with submodules) ==="
mkdir -p third_party
if [ ! -d "third_party/gaussian-splatting" ]; then
  git clone https://github.com/graphdeco-inria/gaussian-splatting.git --recursive third_party/gaussian-splatting
else
  echo "third_party/gaussian-splatting already exists, skipping clone."
  git -C third_party/gaussian-splatting submodule update --init --recursive
fi

echo ""
echo "=== [4/5] Creating conda environment 'gs-compare' ==="
if ! command -v conda &> /dev/null; then
  echo "ERROR: conda not found. Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html"
  exit 1
fi

if conda env list | grep -q "^gs-compare"; then
  echo "conda env 'gs-compare' already exists, skipping creation."
else
  conda create -y -n gs-compare python=3.10
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gs-compare

echo ""
echo "=== [5/5] Installing Python dependencies ==="
# PyTorch w/ CUDA -- adjust cu121 -> your driver's CUDA version if needed (check nvidia-smi)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r "$PROJECT_ROOT/requirements.txt"

# Install the 3DGS repo's own CUDA extensions
pip install "$PROJECT_ROOT/third_party/gaussian-splatting/submodules/diff-gaussian-rasterization"
pip install "$PROJECT_ROOT/third_party/gaussian-splatting/submodules/simple-knn"

echo ""
echo "=== Setup complete ==="
echo "Activate the environment before running the pipeline:"
echo "    conda activate gs-compare"
