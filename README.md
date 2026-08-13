# Gaussian Splatting for 3D Scene Reconstruction: A Multi-Method Comparison Study

Team: Alay Shah, Nina Gharachorloo, Brendan Fullerton, Aditi Putrevu

This codebase benchmarks **3D Gaussian Splatting (3DGS)** against two classical
baselines — **sparse SfM + dense MVS (COLMAP)** and **Poisson surface
reconstruction (Open3D)** — across 4 captured scenes of varying difficulty.

## 0. Directory layout expected

```
final_proj/
└── gaussian-splatting-comparison/
    ├── dataset/
    │   ├── scene1/train/*.jpg
    │   ├── scene2/train/*.jpg
    │   ├── scene3/train/*.jpg
    │   └── scene4/train/*.jpg
    ├── scripts/
    ├── src/
    ├── viewer/
    └── results/                # created by the pipeline
```

Your `dataset/sceneN/train/*.jpg` folders are exactly what the pipeline expects
as raw input — no need to reorganize them.

## 1. One-time environment setup

```bash
cd gaussian-splatting-comparison
bash scripts/setup_env.sh
```

This will:
- Check for / help you install COLMAP (needs a CUDA GPU for dense stereo to be fast).
- Clone the official 3DGS reference implementation (`graphdeco-inria/gaussian-splatting`)
  as a sibling folder `third_party/gaussian-splatting` with its submodules
  (`diff-gaussian-rasterization`, `simple-knn`).
- Create/activate a conda env `gs-compare` with the right PyTorch/CUDA + Open3D,
  scikit-image, etc.

**You must have an NVIDIA GPU with CUDA available on the G3-3590 for this to run
in reasonable time.** 3DGS training and COLMAP dense stereo are both GPU-bound.

Check first:
```bash
nvidia-smi
```
If that fails, none of the GPU-heavy steps (3DGS training, dense MVS) will work
without a lot of patience (CPU fallback is extremely slow — expect hours per scene).

## 2. Run one scene end-to-end (do this first, on scene1, before looping all 4)

```bash
bash scripts/run_scene.sh scene1
```

This runs, in order:
1. **COLMAP sparse SfM** (feature extraction → matching → mapper) → camera poses + sparse point cloud
2. **Held-out test split** — every 8th image (by filename order) is set aside as a test view (standard `llffhold=8` convention used by the 3DGS paper), the rest are "train" views for both 3DGS and the classical baselines
3. **Image undistortion** (COLMAP) — required before both 3DGS training and dense stereo
4. **COLMAP dense MVS** (patch match stereo + stereo fusion) → dense point cloud (`fused.ply`)
5. **3DGS training** on the train-view poses/images → trained splat model
6. **Poisson surface reconstruction** (Open3D) from the dense point cloud → mesh
7. **Evaluation** on the held-out test views for all three methods:
   - PSNR / SSIM (rendered vs. ground-truth held-out photo)
   - reconstruction completeness / coverage
   - training time, and peak GPU memory during training
8. Results are written to `results/scene1/metrics.json`

## 3. Run all 4 scenes

```bash
bash scripts/run_all.sh
```

(Just loops `run_scene.sh` over scene1–scene4. Expect this to take a while —
realistically hours total depending on image counts and GPU. Run it overnight if you can.)

## 4. Aggregate results into the comparison table/figures for the report

```bash
python src/aggregate_results.py
```

Produces `results/comparison_table.csv`, `results/comparison_table.md` (drop straight
into the report), and `results/comparison_figures.png` (bar charts per metric across
scenes and methods).

## 5. Package a viewer demo

```bash
bash viewer/package_for_viewer.sh scene1
```

See `viewer/README.md` — this converts your best 1–2 trained splats into a format
the lightweight web viewer (antimatter15/splat) can load, for the live in-class demo.

## Notes on what's assumed vs. what you'll need to tweak

- Filenames like `photo_10_2026-08-05_15-11-53.jpg` sort correctly enough
  numerically-adjusted (`sort_key.py` handles the `photo_<N>_...` numbering so
  `photo_2` sorts before `photo_10`) — held-out selection uses this order.
- Camera model assumed: `OPENCV` (COLMAP default, handles typical phone-camera
  distortion). If your captures are from very wide-angle/fisheye phone modes, you may
  need `OPENCV_FISHEYE` — see the comment in `src/colmap_pipeline.py`.
- 3DGS training uses the reference repo's default hyperparameters (30k iterations).
  For 4 scenes this is the safest apples-to-apples comparison; you can lower
  `--iterations` in `scripts/run_scene.sh` if you're short on time before the deadline
  (note it in the report's Experiments section if you do).
- GPU memory tracking for 3DGS polls `nvidia-smi` in a background thread during
  training (since we're shelling out to the external repo's `train.py`, we can't
  call `torch.cuda.max_memory_allocated()` directly in-process).
