"""
Minimal reader for COLMAP's binary sparse-model format
(cameras.bin, images.bin, points3D.bin), so we don't need to depend on
pycolmap. Implements the documented binary layout from COLMAP's
src/colmap/scene/reconstruction_io.h.

Only the fields we need for rendering/evaluation are parsed:
  - per-camera intrinsics (model, width, height, params)
  - per-image extrinsics (quaternion + translation, i.e. world-to-camera)
"""
import struct
import collections
import numpy as np

Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
Image = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name"])

# COLMAP camera model id -> (name, num_params)
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def _read_next_bytes(fid, num_bytes, format_char_sequence, endian_char="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_char + format_char_sequence, data)


def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            props = _read_next_bytes(fid, 24, "iiQQ")
            camera_id, model_id, width, height = props
            model_name, num_params = CAMERA_MODELS[model_id]
            params = _read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[camera_id] = Camera(
                id=camera_id, model=model_name, width=width, height=height,
                params=np.array(params),
            )
    return cameras


def read_images_binary(path):
    images = {}
    with open(path, "rb") as fid:
        num_reg_images = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = _read_next_bytes(fid, 64, "idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = _read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = _read_next_bytes(fid, 1, "c")[0]
            num_points2D = _read_next_bytes(fid, 8, "Q")[0]
            # skip the 2D point track data (x, y, point3D_id) — not needed here
            fid.read(24 * num_points2D)
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id, name=image_name,
            )
    return images


def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ])


def camera_to_K(camera: Camera):
    """Build a 3x3 intrinsics matrix from a COLMAP Camera (handles the
    common models used by phone captures: PINHOLE, SIMPLE_RADIAL, OPENCV)."""
    p = camera.params
    if camera.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
        f, cx, cy = p[0], p[1], p[2]
        fx = fy = f
    else:
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return K


def image_world_to_cam(image: Image):
    """Returns (R, t) such that X_cam = R @ X_world + t (COLMAP convention)."""
    R = qvec2rotmat(image.qvec)
    t = image.tvec
    return R, t
