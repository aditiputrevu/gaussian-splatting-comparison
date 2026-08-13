"""
Renders the classical baselines (COLMAP dense point cloud, Poisson mesh)
from the held-out test camera poses, so they can be compared pixel-wise
against the real held-out photos -- the same evaluation protocol used for
3DGS (render.py + metrics.py in the reference repo).

Usage:
    python -m src.render_classical <scene_dir>

Reads:
    <scene_dir>/colmap/sparse/0/{cameras.bin,images.bin}
    <scene_dir>/colmap/test_list.txt
    <scene_dir>/colmap/dense/fused.ply
    <scene_dir>/colmap/dense/poisson_mesh.ply
Writes:
    <scene_dir>/renders_dense/<test_image_name>.png
    <scene_dir>/renders_poisson/<test_image_name>.png
"""
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

from .colmap_model_io import (
    read_cameras_binary, read_images_binary, camera_to_K, image_world_to_cam,
)


def build_o3d_camera(K, R, t, width, height):
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    cam = o3d.camera.PinholeCameraParameters()
    cam.intrinsic = intrinsic
    cam.extrinsic = extrinsic
    return cam


def render_geometry_from_pose(geometry, cam_params, width, height, is_mesh):
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit" if is_mesh else "defaultUnlit"
    if not is_mesh:
        mat.point_size = 3.0
    renderer.scene.add_geometry("geom", geometry, mat)
    if is_mesh:
        # Meshes need a light source with defaultLit, otherwise they render black
        renderer.scene.scene.enable_sun_light(True)
        renderer.scene.scene.set_sun_light([0.0, -1.0, 0.0], [1.0, 1.0, 1.0], 75000)
    renderer.setup_camera(cam_params.intrinsic, cam_params.extrinsic)
    img = renderer.render_to_image()
    renderer.scene.clear_geometry()
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir", type=Path)
    args = ap.parse_args()

    sparse = args.scene_dir / "colmap" / "sparse" / "0"
    cameras = read_cameras_binary(str(sparse / "cameras.bin"))
    images = read_images_binary(str(sparse / "images.bin"))
    test_names = set((args.scene_dir / "colmap" / "test_list.txt").read_text().splitlines())

    name_to_image = {img.name: img for img in images.values()}

    fused_ply = args.scene_dir / "colmap" / "dense" / "fused.ply"
    poisson_mesh_path = args.scene_dir / "colmap" / "dense" / "poisson_mesh.ply"
    pcd = o3d.io.read_point_cloud(str(fused_ply)) if fused_ply.exists() else None
    mesh = o3d.io.read_triangle_mesh(str(poisson_mesh_path)) if poisson_mesh_path.exists() else None
    if mesh is not None:
        mesh.compute_vertex_normals()

    out_dense = args.scene_dir / "renders_dense"
    out_poisson = args.scene_dir / "renders_poisson"
    out_dense.mkdir(exist_ok=True)
    out_poisson.mkdir(exist_ok=True)

    for name in sorted(test_names):
        img_meta = name_to_image[name]
        cam = cameras[img_meta.camera_id]
        K = camera_to_K(cam)
        R, t = image_world_to_cam(img_meta)
        cam_params = build_o3d_camera(K, R, t, cam.width, cam.height)

        if pcd is not None:
            arr = render_geometry_from_pose(pcd, cam_params, cam.width, cam.height, is_mesh=False)
            o3d.io.write_image(str(out_dense / name), o3d.geometry.Image((arr * 255).astype(np.uint8) if arr.dtype != np.uint8 else arr))
        if mesh is not None:
            arr = render_geometry_from_pose(mesh, cam_params, cam.width, cam.height, is_mesh=True)
            o3d.io.write_image(str(out_poisson / name), o3d.geometry.Image((arr * 255).astype(np.uint8) if arr.dtype != np.uint8 else arr))

    print(f"Rendered {len(test_names)} test views for dense point cloud -> {out_dense}")
    print(f"Rendered {len(test_names)} test views for Poisson mesh -> {out_poisson}")


if __name__ == "__main__":
    main()
