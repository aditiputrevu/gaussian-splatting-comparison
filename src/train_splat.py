"""
Wraps the official 3D Gaussian Splatting reference implementation's train.py
(and render.py / metrics.py) so we get, per scene: a trained model, rendered
test-view images, and PSNR/SSIM/LPIPS on the held-out test split -- using the
--eval flag so the repo's own llffhold=8 split is used (matches our classical
baseline split in colmap_pipeline.compute_train_test_split).

Also tracks wall-clock training time and peak GPU memory (polled via
nvidia-smi in a background thread, since train.py runs as a subprocess).

Usage:
    python -m src.train_splat <scene_dir> <gaussian_splatting_repo_dir> [--iterations 30000]

Expects <scene_dir>/colmap/dense/ to already contain the COLMAP-undistorted
scene (images/, sparse/0/) produced by colmap_pipeline.undistort_images.
"""
import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path


class GpuMemPoller:
    """Polls `nvidia-smi` every 0.5s in a background thread, records peak
    memory used (MB) on GPU 0 while a subprocess is running."""

    def __init__(self, interval=0.5):
        self.interval = interval
        self.peak_mb = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output([
                    "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"
                ]).decode().strip()
                used_mb = int(out.splitlines()[0])
                self.peak_mb = max(self.peak_mb, used_mb)
            except Exception:
                pass
            time.sleep(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)


def run_logged(cmd, log_path):
    print(f"[train_splat] $ {' '.join(cmd)}")
    with open(log_path, "a") as f:
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("gs_repo_dir", type=Path, help="path to cloned graphdeco-inria/gaussian-splatting")
    ap.add_argument("--iterations", type=int, default=30000)
    args = ap.parse_args()

    dense_path = args.scene_dir / "colmap" / "dense"  # has images/ + sparse/0/, the format train.py expects
    model_out = args.scene_dir / "gaussian_splat_model"
    model_out.mkdir(parents=True, exist_ok=True)
    log_path = args.scene_dir / "gs_train_log.txt"

    train_py = args.gs_repo_dir / "train.py"
    render_py = args.gs_repo_dir / "render.py"
    metrics_py = args.gs_repo_dir / "metrics.py"

    t0 = time.time()
    with GpuMemPoller() as poller:
        run_logged([
            sys.executable, str(train_py),
            "-s", str(dense_path),
            "-m", str(model_out),
            "--iterations", str(args.iterations),
            "--eval",  # uses llffhold=8 split, same convention as our classical baselines
        ], log_path)
    train_time_sec = time.time() - t0
    peak_gpu_mb = poller.peak_mb

    # Render held-out test views + compute PSNR/SSIM/LPIPS via the repo's own scripts
    run_logged([sys.executable, str(render_py), "-m", str(model_out), "--skip_train"], log_path)
    run_logged([sys.executable, str(metrics_py), "-m", str(model_out)], log_path)

    results_json = model_out / "results.json"
    metrics = {}
    if results_json.exists():
        raw = json.loads(results_json.read_text())
        # results.json is keyed by method name (e.g. "ours_30000"); grab the last iteration's entry
        last_key = sorted(raw.keys())[-1]
        metrics = raw[last_key]

    summary = {
        "train_time_sec": train_time_sec,
        "peak_gpu_memory_mb": peak_gpu_mb,
        **metrics,  # SSIM / PSNR / LPIPS as reported by the reference repo's metrics.py
    }
    (args.scene_dir / "results_3dgs_timing.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
