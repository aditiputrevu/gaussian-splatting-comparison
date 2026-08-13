"""
Wraps the COLMAP CLI to produce, for one scene:
  - a sparse SfM reconstruction (camera poses + sparse point cloud)
  - undistorted images (required by 3DGS and useful for evaluation renders)
  - a dense MVS point cloud (patch match stereo + fusion)
  - a train/test image-name split (llffhold=8, matching the convention used
    by the official 3DGS reference implementation, so PSNR/SSIM comparisons
    across methods are apples-to-apples)

Usage:
    python -m src.colmap_pipeline <scene_dir> [--camera_model OPENCV]

<scene_dir> must contain a 'train/' subfolder of input images (as produced
by your capture pipeline), e.g. dataset/scene1/.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

from .colmap_model_io import read_images_binary


def run(cmd, log_path=None):
    print(f"[colmap_pipeline] $ {' '.join(cmd)}")
    t0 = time.time()
    if log_path:
        with open(log_path, "a") as f:
            subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd, check=True)
    return time.time() - t0


def sparse_reconstruction(scene_dir: Path, camera_model="OPENCV", log_path=None):
    """Feature extraction -> exhaustive matching -> incremental mapper."""
    image_path = scene_dir / "train"
    colmap_dir = scene_dir / "colmap"
    db_path = colmap_dir / "database.db"
    sparse_path = colmap_dir / "sparse"
    colmap_dir.mkdir(parents=True, exist_ok=True)
    sparse_path.mkdir(parents=True, exist_ok=True)

    timings = {}

    t = run([
        "colmap", "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(image_path),
        "--ImageReader.camera_model", camera_model,
        "--ImageReader.single_camera", "1",  # phone: assume one physical camera per scene
        "--SiftExtraction.use_gpu", "1",
    ], log_path)
    timings["feature_extraction_sec"] = t

    # Exhaustive matching is fine for the image counts here (~60-90 images/scene).
    # Switch to --sequential_matcher if a scene ever has >a few hundred images.
    t = run([
        "colmap", "exhaustive_matcher",
        "--database_path", str(db_path),
        "--SiftMatching.use_gpu", "1",
    ], log_path)
    timings["matching_sec"] = t

    t = run([
        "colmap", "mapper",
        "--database_path", str(db_path),
        "--image_path", str(image_path),
        "--output_path", str(sparse_path),
    ], log_path)
    timings["mapper_sec"] = t

    return colmap_dir, sparse_path / "0", timings


def undistort_images(scene_dir: Path, sparse_model_path: Path, log_path=None):
    image_path = scene_dir / "train"
    dense_path = scene_dir / "colmap" / "dense"
    dense_path.mkdir(parents=True, exist_ok=True)
    t = run([
        "colmap", "image_undistorter",
        "--image_path", str(image_path),
        "--input_path", str(sparse_model_path),
        "--output_path", str(dense_path),
        "--output_type", "COLMAP",
    ], log_path)
    return dense_path, {"undistortion_sec": t}


def nest_sparse_for_3dgs(dense_path: Path):
    """
    COLMAP's own tools (patch_match_stereo, stereo_fusion) expect the
    undistorted sparse model directly at dense/sparse/{cameras,images,points3D}.bin,
    and dense MVS must run against that layout. But the 3DGS reference
    repo's data loader expects it nested one level deeper at
    dense/sparse/0/... (mirroring the original sparse/0/ naming convention).
    Call this AFTER dense_mvs() has finished, not before -- nesting too early
    breaks patch_match_stereo, which can't find the files anymore.
    """
    undistorted_sparse = dense_path / "sparse"
    nested = undistorted_sparse / "0"
    if undistorted_sparse.exists() and not nested.exists():
        nested.mkdir(parents=True, exist_ok=True)
        for f in undistorted_sparse.glob("*"):
            if f.is_file():
                f.rename(nested / f.name)


def dense_mvs(dense_path: Path, log_path=None):
    """Patch match stereo (per-view depth/normal maps) + stereo fusion -> fused.ply"""
    timings = {}
    t = run([
        "colmap", "patch_match_stereo",
        "--workspace_path", str(dense_path),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true",
    ], log_path)
    timings["patch_match_stereo_sec"] = t

    fused_ply = dense_path / "fused.ply"
    t = run([
        "colmap", "stereo_fusion",
        "--workspace_path", str(dense_path),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", str(fused_ply),
    ], log_path)
    timings["stereo_fusion_sec"] = t

    # Now safe to nest the sparse model for 3DGS -- dense MVS is fully done.
    nest_sparse_for_3dgs(dense_path)

    return fused_ply, timings


def compute_train_test_split(sparse_model_path: Path, llffhold=8):
    """
    Mirrors the reference 3DGS implementation's convention: sort registered
    images by filename, then every `llffhold`-th image is a held-out test view.
    Returns (train_names, test_names) as lists of image filenames (strings).
    """
    images = read_images_binary(str(sparse_model_path / "images.bin"))
    names = sorted([img.name for img in images.values()])
    test_names = [n for i, n in enumerate(names) if i % llffhold == 0]
    train_names = [n for n in names if n not in set(test_names)]
    return train_names, test_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("--camera_model", default="OPENCV",
                     help="OPENCV for typical phone video/photo; OPENCV_FISHEYE for wide/fisheye captures")
    args = ap.parse_args()

    scene_dir = args.scene_dir
    log_path = scene_dir / "colmap" / "colmap_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    colmap_dir, sparse_model, t1 = sparse_reconstruction(scene_dir, args.camera_model, log_path)
    dense_path, t2 = undistort_images(scene_dir, sparse_model, log_path)
    fused_ply, t3 = dense_mvs(dense_path, log_path)
    train_names, test_names = compute_train_test_split(sparse_model)

    split_dir = scene_dir / "colmap"
    (split_dir / "train_list.txt").write_text("\n".join(train_names))
    (split_dir / "test_list.txt").write_text("\n".join(test_names))

    import json
    timings = {**t1, **t2, **t3}
    (scene_dir / "results_colmap_timing.json").write_text(json.dumps(timings, indent=2))

    print(f"\nScene: {scene_dir.name}")
    print(f"  Registered images: {len(train_names) + len(test_names)}")
    print(f"  Train / Test split: {len(train_names)} / {len(test_names)}")
    print(f"  Dense point cloud: {fused_ply}")
    print(f"  Timings: {timings}")


if __name__ == "__main__":
    sys.exit(main())
