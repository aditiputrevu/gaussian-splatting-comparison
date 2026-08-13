"""
Shared evaluation harness applied uniformly across all 3 methods for one scene:
  - held-out-view PSNR / SSIM (3DGS metrics come from the reference repo's
    metrics.py already; here we compute PSNR/SSIM for the two classical
    baselines against the same held-out ground-truth photos, using the
    identical image set, for an apples-to-apples number)
  - reconstruction completeness / coverage (fraction of the held-out frame
    that the reconstruction actually renders geometry into, i.e. non-background
    pixels -- a proxy for how much of the scene each method actually covers)
  - training/reconstruction time and peak memory, pulled from the per-stage
    JSON files written by colmap_pipeline.py, poisson_baseline.py, train_splat.py

Usage:
    python -m src.evaluate <scene_dir>

Writes: <scene_dir>/../results/<scene_name>/metrics.json
(also copies into results/ so aggregate_results.py can find every scene)
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def load_rgb(path: Path, resize_to=None):
    img = Image.open(path).convert("RGB")
    if resize_to is not None:
        img = img.resize(resize_to, Image.BILINEAR)
    return np.asarray(img).astype(np.float32) / 255.0


def coverage_fraction(render_path: Path, bg_value=0, tol=2):
    """Fraction of pixels that are NOT background (rendering produced geometry
    there). Open3D's OffscreenRenderer background defaults to black, so a
    near-black pixel with no geometry counts as background."""
    arr = np.asarray(Image.open(render_path).convert("RGB"))
    non_bg = np.any(arr > (bg_value + tol), axis=-1)
    return float(non_bg.mean())


def psnr_ssim(gt_path: Path, render_path: Path):
    gt = load_rgb(gt_path)
    render_img = Image.open(render_path).convert("RGB")
    render_img = render_img.resize((gt.shape[1], gt.shape[0]), Image.BILINEAR)
    render_arr = np.asarray(render_img).astype(np.float32) / 255.0
    p = psnr(gt, render_arr, data_range=1.0)
    s = ssim(gt, render_arr, data_range=1.0, channel_axis=-1)
    return float(p), float(s)


def evaluate_classical_method(scene_dir: Path, render_dir_name: str, test_names):
    psnrs, ssims, coverages = [], [], []
    for name in test_names:
        gt_path = scene_dir / "train" / name
        render_path = scene_dir / render_dir_name / name
        if not render_path.exists() or not gt_path.exists():
            continue
        p, s = psnr_ssim(gt_path, render_path)
        c = coverage_fraction(render_path)
        psnrs.append(p)
        ssims.append(s)
        coverages.append(c)
    return {
        "PSNR": float(np.mean(psnrs)) if psnrs else None,
        "SSIM": float(np.mean(ssims)) if ssims else None,
        "coverage": float(np.mean(coverages)) if coverages else None,
        "n_test_views_evaluated": len(psnrs),
    }


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("--results_root", type=Path, default=None,
                     help="defaults to <scene_dir>/../../results")
    args = ap.parse_args()

    scene_dir = args.scene_dir
    test_list_path = scene_dir / "colmap" / "test_list.txt"
    test_names = [n for n in test_list_path.read_text().splitlines() if n.strip()]

    colmap_timing = load_json(scene_dir / "results_colmap_timing.json")
    poisson_timing = load_json(scene_dir / "results_poisson_timing.json")
    gs_timing = load_json(scene_dir / "results_3dgs_timing.json")

    dense_metrics = evaluate_classical_method(scene_dir, "renders_dense", test_names)
    poisson_metrics = evaluate_classical_method(scene_dir, "renders_poisson", test_names)

    metrics = {
        "scene": scene_dir.name,
        "n_test_views": len(test_names),
        "3dgs": {
            "PSNR": gs_timing.get("PSNR"),
            "SSIM": gs_timing.get("SSIM"),
            "LPIPS": gs_timing.get("LPIPS"),
            "train_time_sec": gs_timing.get("train_time_sec"),
            "peak_gpu_memory_mb": gs_timing.get("peak_gpu_memory_mb"),
        },
        "colmap_dense_mvs": {
            **dense_metrics,
            "reconstruction_time_sec": (
                colmap_timing.get("feature_extraction_sec", 0)
                + colmap_timing.get("matching_sec", 0)
                + colmap_timing.get("mapper_sec", 0)
                + colmap_timing.get("undistortion_sec", 0)
                + colmap_timing.get("patch_match_stereo_sec", 0)
                + colmap_timing.get("stereo_fusion_sec", 0)
            ) or None,
        },
        "poisson_mesh": {
            **poisson_metrics,
            "reconstruction_time_sec": poisson_timing.get("poisson_reconstruction_sec"),
            "peak_memory_mb": poisson_timing.get("peak_python_memory_mb"),
            "output_vertices": poisson_timing.get("output_vertices"),
            "output_triangles": poisson_timing.get("output_triangles"),
        },
    }

    results_root = args.results_root or (scene_dir.parent.parent / "results")
    out_dir = results_root / scene_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
