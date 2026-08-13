"""
Generates report/slide-ready figures:

1. Per-scene comparison grids: for every held-out test view, a 4-panel image
   [Ground Truth | 3D Gaussian Splatting | Poisson Mesh | COLMAP Dense MVS],
   so you can pick the most illustrative one per scene for your report/slides.

2. A single "pipeline stages" figure for one scene (default: scene1),
   showing Sparse Point Cloud -> Dense Point Cloud -> Poisson Mesh side by
   side, rendered from a fixed viewpoint, to illustrate the reconstruction
   process itself (not a quality comparison).

Usage:
    python -m src.make_report_figures                     # all scenes, comparison grids
    python -m src.make_report_figures --pipeline_scene scene1

Output: report_figures/<scene>_test<i>_comparison.png
        report_figures/<scene>_pipeline_stages.png
"""
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "report_figures"

LABEL_H = 36  # px reserved for text label under each panel
PANEL_W = 400  # each panel resized to this width for consistent grids


def _font(size=20):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _panel(img_path: Path, label: str, missing_ok=True):
    """Load an image, resize to PANEL_W keeping aspect ratio, add a label strip below."""
    if img_path is None or not Path(img_path).exists():
        if not missing_ok:
            raise FileNotFoundError(img_path)
        img = Image.new("RGB", (PANEL_W, int(PANEL_W * 0.75)), (40, 40, 40))
        d = ImageDraw.Draw(img)
        d.text((10, 10), "missing", fill=(255, 80, 80), font=_font(16))
    else:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        new_h = int(h * (PANEL_W / w))
        img = img.resize((PANEL_W, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (img.width, img.height + LABEL_H), (20, 20, 20))
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    d.text((8, img.height + 6), label, fill=(255, 255, 255), font=_font(20))
    return canvas


def make_comparison_grid(gt_path, gs_path, poisson_path, dense_path, out_path):
    panels = [
        _panel(gt_path, "Ground Truth"),
        _panel(gs_path, "3D Gaussian Splatting"),
        _panel(poisson_path, "Poisson Mesh"),
        _panel(dense_path, "COLMAP Dense MVS"),
    ]
    max_h = max(p.height for p in panels)
    total_w = sum(p.width for p in panels) + 3 * 8  # 8px gaps
    grid = Image.new("RGB", (total_w, max_h), (10, 10, 10))
    x = 0
    for p in panels:
        grid.paste(p, (x, 0))
        x += p.width + 8
    grid.save(out_path)
    print(f"Wrote {out_path}")


def make_scene_grids(scene_dir: Path, out_dir: Path):
    test_list = scene_dir / "colmap" / "test_list.txt"
    if not test_list.exists():
        print(f"Skipping {scene_dir.name}: no test_list.txt")
        return
    test_names = sorted([n for n in test_list.read_text().splitlines() if n.strip()])

    gs_test_dir = scene_dir / "gaussian_splat_model" / "test" / "ours_15000"
    dense_dir = scene_dir / "renders_dense"
    poisson_dir = scene_dir / "renders_poisson"

    for i, name in enumerate(test_names):
        gt = gs_test_dir / "gt" / f"{i:05d}.png"
        gs_render = gs_test_dir / "renders" / f"{i:05d}.png"
        # Fallback to the original photo if 3DGS's own gt/ isn't there for some reason
        if not gt.exists():
            gt = scene_dir / "train" / name
        dense = dense_dir / name
        poisson = poisson_dir / name

        out_path = out_dir / f"{scene_dir.name}_test{i}_comparison.png"
        make_comparison_grid(gt, gs_render, poisson, dense, out_path)


def render_geometry_fixed_view(geometry, is_mesh, width=800, height=600):
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit" if is_mesh else "defaultUnlit"
    if not is_mesh:
        mat.point_size = 3.0
    renderer.scene.add_geometry("geom", geometry, mat)
    if is_mesh:
        renderer.scene.scene.enable_sun_light(True)
        renderer.scene.scene.set_sun_light([0.0, -1.0, 0.0], [1.0, 1.0, 1.0], 75000)

    bbox = geometry.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = np.linalg.norm(bbox.get_extent())
    eye = center + np.array([0.0, -extent * 0.3, extent * 1.2])
    renderer.setup_camera(60.0, center, eye, [0, -1, 0])

    img = renderer.render_to_image()
    return np.asarray(img)


def make_pipeline_stages_figure(scene_dir: Path, out_dir: Path):
    sparse_ply = scene_dir / "colmap" / "sparse" / "0" / "points.ply"
    dense_ply = scene_dir / "colmap" / "dense" / "fused.ply"
    poisson_ply = scene_dir / "colmap" / "dense" / "poisson_mesh.ply"

    # Sparse model is stored as .bin by default; export to PLY if not already done
    if not sparse_ply.exists():
        import subprocess
        subprocess.run([
            "colmap", "model_converter",
            "--input_path", str(scene_dir / "colmap" / "sparse" / "0"),
            "--output_path", str(sparse_ply),
            "--output_type", "PLY",
        ], check=True)

    stages = []
    if sparse_ply.exists():
        pcd = o3d.io.read_point_cloud(str(sparse_ply))
        arr = render_geometry_fixed_view(pcd, is_mesh=False)
        stages.append((arr, "1. Sparse Point Cloud (SfM)"))
    if dense_ply.exists():
        pcd = o3d.io.read_point_cloud(str(dense_ply))
        arr = render_geometry_fixed_view(pcd, is_mesh=False)
        stages.append((arr, "2. Dense Point Cloud (MVS)"))
    if poisson_ply.exists():
        mesh = o3d.io.read_triangle_mesh(str(poisson_ply))
        mesh.compute_vertex_normals()
        arr = render_geometry_fixed_view(mesh, is_mesh=True)
        stages.append((arr, "3. Poisson Mesh"))

    if not stages:
        print(f"No pipeline stage files found for {scene_dir.name}")
        return

    panels = []
    for arr, label in stages:
        img = Image.fromarray(arr)
        w, h = img.size
        new_h = int(h * (PANEL_W / w))
        img = img.resize((PANEL_W, new_h), Image.LANCZOS)
        canvas = Image.new("RGB", (img.width, img.height + LABEL_H), (20, 20, 20))
        canvas.paste(img, (0, 0))
        d = ImageDraw.Draw(canvas)
        d.text((8, img.height + 6), label, fill=(255, 255, 255), font=_font(20))
        panels.append(canvas)

    max_h = max(p.height for p in panels)
    total_w = sum(p.width for p in panels) + (len(panels) - 1) * 8
    grid = Image.new("RGB", (total_w, max_h), (10, 10, 10))
    x = 0
    for p in panels:
        grid.paste(p, (x, 0))
        x += p.width + 8

    out_path = out_dir / f"{scene_dir.name}_pipeline_stages.png"
    grid.save(out_path)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=None, help="defaults to all scene1-4 found in dataset/")
    ap.add_argument("--pipeline_scene", default="scene1", help="which scene to build the pipeline-stages figure for")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    dataset_root = PROJECT_ROOT / "dataset"

    scenes = args.scenes or sorted([p.name for p in dataset_root.glob("scene*") if p.is_dir()])

    for scene_name in scenes:
        scene_dir = dataset_root / scene_name
        print(f"\n=== {scene_name}: comparison grids ===")
        make_scene_grids(scene_dir, OUT_DIR)

    print(f"\n=== Pipeline stages figure ({args.pipeline_scene}) ===")
    make_pipeline_stages_figure(dataset_root / args.pipeline_scene, OUT_DIR)


if __name__ == "__main__":
    main()
