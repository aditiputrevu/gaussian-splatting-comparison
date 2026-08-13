# Gaussian Splatting for 3D Scene Reconstruction: A Multi-Method Comparison Study

Team: Alay Shah, Nina Gharachorloo, Brendan Fullerton, Aditi Putrevu

This codebase benchmarks **3D Gaussian Splatting (3DGS)** against two classical
baselines — **sparse SfM + dense MVS (COLMAP)** and **Poisson surface
reconstruction (Open3D)** — across 4 captured scenes of varying difficulty.
See `report/report.tex` for the full writeup and final results.

## Actual settings used for our results

- **3DGS training: 15,000 iterations** (not the paper's default 30,000) —
  reduced due to GPU memory/time constraints on a 4GB laptop GPU.
- **Training resolution: half (`-r 2`) for scenes 1–3, quarter (`-r 4`) for
  scene 4** — scene 4's source photos were much higher resolution
  (4284×5712) and repeatedly hit CUDA out-of-memory errors even at half res.
- **Scene 4 was trained on a cloud GPU** (NVIDIA T4 via Google Colab) after
  the local GTX 1650 (4GB) couldn't fit it even at reduced resolution.
- **Held-out test split: every 8th image** (llffhold=8), same convention as
  the reference 3DGS paper, applied consistently across all three methods.
- **The interactive `.splat` web-viewer demo was built but not used in the
  final presentation** — `viewer/` still works if you want to try it.

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
    ├── report/
    └── results/                # created by the pipeline
```

## 1. One-time environment setup

```bash
cd gaussian-splatting-comparison
bash scripts/setup_env.sh
```

**Important — the apt/default COLMAP package on Ubuntu is usually CPU-only**
(no CUDA), which means dense stereo (`patch_match_stereo`) will fail with
`ERROR: Dense stereo reconstruction requires CUDA`. If you hit that, run:

```bash
bash scripts/build_colmap_cuda.sh
```

This rebuilds COLMAP from source with CUDA enabled (~20-30 min, needs
`build-essential`, a system CUDA toolkit, and enough patience for the ninja
build). Confirm it worked with `which colmap` → should print
`/usr/local/bin/colmap`, and `colmap patch_match_stereo --help` should run
without a CUDA error.

You also need `gcc`/`g++` and `ninja` to build the 3DGS CUDA extensions
(`diff-gaussian-rasterization`, `simple-knn`) — install with:
```bash
sudo apt-get install -y build-essential
pip install ninja
```
and make sure `CUDA_HOME` is set (see comments in `setup_env.sh`) before
installing those two submodules with `pip install --no-build-isolation`.

Check your GPU first:
```bash
nvidia-smi
```
If that fails, none of the GPU-heavy steps (3DGS training, dense MVS) will
work without a lot of patience (CPU fallback is extremely slow).

## 2. Run one scene end-to-end

```bash
bash scripts/run_scene.sh scene1 15000
```

(The second argument is the iteration count — we used 15000, not the
default 30000, for all scenes given GPU/time constraints. See
`scripts/run_remaining_scenes.sh` for the version actually used for
scenes 2–4, which also handles resuming a partially-completed scene and
uses `-r 2` training resolution automatically.)

This runs, in order:
1. COLMAP sparse SfM (feature extraction → matching → mapper) → camera poses + sparse point cloud
2. Held-out test split — every 8th image (llffhold=8 convention)
3. Image undistortion (COLMAP)
4. COLMAP dense MVS (patch match stereo + stereo fusion) → dense point cloud (`fused.ply`)
5. 3DGS training on the train-view poses/images → trained splat model
6. Poisson surface reconstruction (Open3D) from the dense point cloud → mesh
7. Evaluation on the held-out test views for all three methods (PSNR/SSIM/coverage/timing)

Results are written to `results/scene1/metrics.json`.

## 3. Run remaining scenes

```bash
bash scripts/run_remaining_scenes.sh scene2 scene3 scene4
```

This is what we actually used for scenes 2–4 — it skips any step that's
already completed for a scene (safe to re-run if interrupted), and applies
the same 15,000-iteration / half-resolution settings as scene 1 automatically.

**Known gotcha we hit repeatedly:** if a scene's photos are much
higher-resolution than the others (as scene 4's were), you may need to run
dense MVS with `--PatchMatchStereo.max_image_size 2000` and/or train 3DGS
with `-r 4` instead of `-r 2` to avoid CUDA OOM. See the comments in
`src/colmap_pipeline.py` and `scripts/run_remaining_scenes.sh`.

## 4. Aggregate results into the comparison table/figures for the report

```bash
python src/aggregate_results.py
```

Produces `results/comparison_table.csv`, `results/comparison_table.md`, and
`results/comparison_figures.png`.

## 5. Generate report-ready comparison images

```bash
python -m src.make_report_figures
```

Produces per-scene ground-truth/3DGS/Poisson/Dense-MVS comparison grids and
a pipeline-stages figure in `report_figures/` — these are the images used in
`report/report.tex`.

## 6. (Optional) Package a viewer demo

```bash
bash viewer/package_for_viewer.sh scene1
```

See `viewer/README.md`. Built and working, but **not used in our final
presentation** — included here in case it's useful for anyone continuing
this project.

## Notes on what's assumed vs. what you'll need to tweak

- Filenames like `photo_10_2026-08-05_15-11-53.jpg` sort correctly enough
  numerically-adjusted (`sort_key.py` handles the `photo_<N>_...` numbering).
- Camera model assumed: `OPENCV`. If your captures are wide-angle/fisheye,
  you may need `OPENCV_FISHEYE` — see the comment in `src/colmap_pipeline.py`.
- GPU memory tracking for 3DGS polls `nvidia-smi` in a background thread
  during training when using `src/train_splat.py`'s wrapper; scenes run via
  the direct `train.py` calls in `run_remaining_scenes.sh` don't capture
  this, so some scenes' `peak_gpu_memory_mb` in `results/*/metrics.json`
  will show `null` — this is expected, not a bug.
