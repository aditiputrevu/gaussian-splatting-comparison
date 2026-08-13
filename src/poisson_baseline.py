"""
Classical mesh baseline: Poisson surface reconstruction (Kazhdan et al.) on
top of the COLMAP dense point cloud (fused.ply).

Usage:
    python -m src.poisson_baseline <scene_dir>

Reads:  <scene_dir>/colmap/dense/fused.ply
Writes: <scene_dir>/colmap/dense/poisson_mesh.ply
        <scene_dir>/results_poisson_timing.json
"""
import argparse
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import open3d as o3d


def run_poisson(fused_ply_path: Path, depth=9, density_quantile_trim=0.02):
    pcd = o3d.io.read_point_cloud(str(fused_ply_path))
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=30)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )

    # Trim low-density (spurious/extrapolated) faces, standard Poisson cleanup
    densities = np.asarray(densities)
    thresh = np.quantile(densities, density_quantile_trim)
    verts_to_remove = densities < thresh
    mesh.remove_vertices_by_mask(verts_to_remove)

    return mesh, pcd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("--depth", type=int, default=9, help="Poisson octree depth (higher = more detail, slower)")
    args = ap.parse_args()

    fused_ply = args.scene_dir / "colmap" / "dense" / "fused.ply"
    out_mesh = args.scene_dir / "colmap" / "dense" / "poisson_mesh.ply"

    tracemalloc.start()
    t0 = time.time()
    mesh, pcd = run_poisson(fused_ply, depth=args.depth)
    elapsed = time.time() - t0
    _, peak_mem_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    o3d.io.write_triangle_mesh(str(out_mesh), mesh)

    timing = {
        "poisson_reconstruction_sec": elapsed,
        "peak_python_memory_mb": peak_mem_bytes / (1024 * 1024),
        "input_points": len(pcd.points),
        "output_vertices": len(mesh.vertices),
        "output_triangles": len(mesh.triangles),
    }
    (args.scene_dir / "results_poisson_timing.json").write_text(json.dumps(timing, indent=2))
    print(json.dumps(timing, indent=2))
    print(f"Mesh written to {out_mesh}")


if __name__ == "__main__":
    main()
