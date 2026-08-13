#!/usr/bin/env bash
# Builds COLMAP from source with CUDA support, since the apt package
# (colmap 3.7) ships CPU-only and dense MVS (patch_match_stereo) requires CUDA.
#
# Installs a system-wide CUDA toolkit (NVIDIA's official WSL-Ubuntu repo --
# separate from the conda cuda-toolkit used for the 3DGS extensions, since
# CMake-based builds expect CUDA at a standard system path like /usr/local/cuda).
#
# GTX 1650 = Turing architecture = compute capability 7.5 (set below).
set -euo pipefail

echo "=== [1/4] Installing COLMAP build dependencies ==="
sudo apt-get update
sudo apt-get install -y \
  git cmake ninja-build build-essential \
  libboost-program-options-dev libboost-graph-dev libboost-system-dev \
  libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev \
  libgoogle-glog-dev libgtest-dev libsqlite3-dev libglew-dev \
  qtbase5-dev libqt5opengl5-dev libcgal-dev libceres-dev

echo ""
echo "=== [2/4] Installing system CUDA toolkit (NVIDIA WSL-Ubuntu repo) ==="
if [ -d /usr/local/cuda ]; then
  echo "/usr/local/cuda already exists, skipping CUDA toolkit install."
else
  wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo apt-get update
  sudo apt-get install -y cuda-toolkit-12-1
  rm -f cuda-keyring_1.1-1_all.deb
fi

# Make system CUDA visible for this build (separate from the conda CUDA_HOME
# used for the 3DGS extensions -- that one stays as-is).
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH:-}

echo ""
echo "=== [3/4] Cloning and building COLMAP from source (with CUDA) ==="
mkdir -p ~/build
cd ~/build
if [ ! -d colmap ]; then
  git clone https://github.com/colmap/colmap.git
fi
cd colmap
git checkout 3.9.1
mkdir -p build && cd build

cmake .. -GNinja \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DCUDA_ENABLED=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc

ninja

echo ""
echo "=== [4/4] Installing (this replaces the apt colmap binary in PATH priority) ==="
sudo ninja install

echo ""
echo "=== Verifying ==="
hash -r  # clear bash's remembered PATH lookup for 'colmap'
which colmap
colmap -h | head -3
echo ""
echo "If 'which colmap' shows /usr/local/bin/colmap, you're using the new CUDA build."
echo "Test CUDA support directly with: colmap patch_match_stereo --help"
