#!/usr/bin/env python3
"""
Render an orbit camera path around an object center using COLMAP sparse model.

Why this script exists:
- Running from repo root avoids import/path confusion.
- We use COLMAP `sparse/0/images.bin` + `sparse/0/cameras.bin` to get a good prior
  (radius, FoV) from the real capture setup, then generate a smooth orbit path.

Outputs:
- PNG frames:  ./output/orbit/00000.png ...
- Video:       ./output/orbit.mp4
"""

import os
from argparse import ArgumentParser
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision

from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.scene.cameras import Simple_Camera
from gaussiansplatting.scene.colmap_loader import (
    read_extrinsics_binary,
    read_intrinsics_binary,
    qvec2rotmat,
    rotmat2qvec,
)
import collections
from gaussiansplatting.scene.vanilla_gaussian_model import GaussianModel
from gaussiansplatting.utils.graphics_utils import focal2fov, fov2focal, getWorld2View2
from gaussiansplatting.utils.system_utils import searchForMaxIteration
from gaussiansplatting.arguments import ModelParams, PipelineParams, get_combined_args
from gaussiansplatting.utils.general_utils import safe_state

# Colmap data structures
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
Image = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])


def _parse_center(s: str) -> np.ndarray:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"--center must be 'x,y,z' (got {s})")
    return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float32)


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v
    return v / n


def look_at_c2w(P: np.ndarray, C: np.ndarray, world_up: np.ndarray) -> np.ndarray:
    """
    Build camera-to-world (C2W) 4x4 matrix with OpenGL-style camera:
      - camera looks along -Z
      - +X right, +Y up
    """
    P = np.asarray(P, dtype=np.float32)
    C = np.asarray(C, dtype=np.float32)
    up = _normalize(np.asarray(world_up, dtype=np.float32))

    forward = _normalize(C - P)  # camera -> object direction in world
    # Use COLMAP/3DGS-friendly convention: camera +Z points forward (towards target).
    # Build a right-handed basis.
    right = np.cross(up, forward)
    if np.linalg.norm(right) < 1e-6:
        # forward ~ up; pick a fallback up
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(up, forward)
    right = _normalize(right)
    cam_up = np.cross(forward, right)

    # C2W rotation columns are camera axes in world coords: [X(right), Y(up), Z(forward)].
    R_c2w = np.stack([right, cam_up, forward], axis=1)  # (3,3)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R_c2w
    c2w[:3, 3] = P
    return c2w


def fovy_to_fovx(fovy: float, h: int, w: int) -> float:
    """Assume square pixels: fx = fy * (w/h)."""
    fy = fov2focal(float(fovy), int(h))
    fx = fy * (float(w) / float(h))
    return float(focal2fov(float(fx), int(w)))


def load_colmap_prior(colmap_path: str) -> Tuple[np.ndarray, float, dict]:
    """
    Returns:
      cam_centers_world: (N,3) camera centers in world coordinates
      fovy: vertical FoV in radians (from first camera model)
      colmap_cameras: dict camera_id -> Camera (from cameras.bin), for writing orbit Colmap with same intrinsics
    """
    sparse0 = os.path.join(os.path.abspath(colmap_path), "sparse", "0")
    images_bin = os.path.join(sparse0, "images.bin")
    cameras_bin = os.path.join(sparse0, "cameras.bin")

    images = read_extrinsics_binary(images_bin)
    cams = read_intrinsics_binary(cameras_bin)

    cam_centers: List[np.ndarray] = []
    first_intr = None
    for img in images.values():
        intr = cams[img.camera_id]
        if first_intr is None:
            first_intr = intr

        # Same convention as gaussiansplatting.scene.dataset_readers.readColmapCameras:
        R = np.transpose(qvec2rotmat(img.qvec)).astype(np.float32)
        T = np.array(img.tvec, dtype=np.float32)
        W2C = getWorld2View2(R, T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3].astype(np.float32))

    if not cam_centers or first_intr is None:
        raise RuntimeError("No registered cameras found in COLMAP images.bin")

    h = int(first_intr.height)
    if first_intr.model == "PINHOLE":
        fy = float(first_intr.params[1])
    elif first_intr.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
        fy = float(first_intr.params[0])
    else:
        fy = float(first_intr.params[0])
    fovy = float(focal2fov(fy, h))

    return np.stack(cam_centers, axis=0), fovy, cams


def save_orbit_cameras_to_colmap(
    cameras: List[Simple_Camera],
    output_dir: str,
    camera_id: int = 1,
    format: str = "txt",
    source_camera: Optional[Camera] = None,
):
    """
    Save orbit cameras to Colmap format (cameras.txt/images.txt or cameras.bin/images.bin).
    
    Args:
        cameras: List of Simple_Camera objects
        output_dir: Directory to save Colmap files (will create sparse/0 subdirectory)
        camera_id: Camera ID to use for all cameras (default: 1)
        format: "txt" or "bin" (default: "txt")
        source_camera: If given, use this Camera's model, width, height, params so output matches source cameras.bin
    """
    sparse_dir = os.path.join(output_dir, "sparse", "0")
    os.makedirs(sparse_dir, exist_ok=True)
    
    if format == "txt":
        cameras_path = os.path.join(sparse_dir, "cameras.txt")
        images_path = os.path.join(sparse_dir, "images.txt")
    else:
        cameras_path = os.path.join(sparse_dir, "cameras.bin")
        images_path = os.path.join(sparse_dir, "images.bin")
    
    if len(cameras) == 0:
        raise ValueError("No cameras to save")
    
    # Camera entry: use source_camera (from cameras.bin) when provided so output matches bin content
    if source_camera is not None:
        cam_dict = {
            camera_id: Camera(
                id=camera_id,
                model=source_camera.model,
                width=source_camera.width,
                height=source_camera.height,
                params=np.array(source_camera.params, dtype=np.float64),
            )
        }
    else:
        first_cam = cameras[0]
        h, w = first_cam.image_height, first_cam.image_width
        fy = fov2focal(first_cam.FoVy, h)
        fx = fov2focal(first_cam.FoVx, w)
        cx = w / 2.0
        cy = h / 2.0
        params = np.array([fx, fy, cx, cy], dtype=np.float64)
        cam_dict = {
            camera_id: Camera(
                id=camera_id,
                model="PINHOLE",
                width=w,
                height=h,
                params=params,
            )
        }
    
    # Create image entries
    images_dict = {}
    for i, cam in enumerate(cameras):
        # Convert R, T to qvec, tvec
        # cam.R is the rotation matrix (W2C convention, same as Colmap)
        # Colmap stores qvec such that R_w2c = transpose(qvec2rotmat(qvec))
        # So we need: qvec2rotmat(qvec) = R_w2c^T, which means qvec = rotmat2qvec(R_w2c^T)
        R_w2c = np.array(cam.R, dtype=np.float64) if isinstance(cam.R, (list, np.ndarray)) else cam.R.cpu().numpy().astype(np.float64)
        T_w2c = np.array(cam.T, dtype=np.float64) if isinstance(cam.T, (list, np.ndarray)) else cam.T.cpu().numpy().astype(np.float64)
        
        # Colmap convention: qvec represents rotation such that R_w2c = transpose(qvec2rotmat(qvec))
        # So we need: qvec2rotmat(qvec) = R_w2c^T
        # Therefore: qvec = rotmat2qvec(R_w2c^T)
        R_colmap = R_w2c.T
        qvec = rotmat2qvec(R_colmap)
        tvec = T_w2c
        
        # Empty point observations (orbit cameras don't have 2D points)
        xys = np.empty((0, 2), dtype=np.float64)
        point3D_ids = np.empty(0, dtype=np.int64)
        
        image_name = cam.image_name if hasattr(cam, "image_name") else f"orbit_{i:05d}.png"
        
        images_dict[i + 1] = Image(
            id=i + 1,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=image_name,
            xys=xys,
            point3D_ids=point3D_ids,
        )
    
    # Write files
    if format == "txt":
        _write_cameras_text(cam_dict, cameras_path)
        _write_images_text(images_dict, images_path)
    else:
        _write_cameras_binary(cam_dict, cameras_path)
        _write_images_binary(images_dict, images_path)
    
    print(f"Saved {len(cameras)} orbit cameras to {sparse_dir}/")
    print(f"  - cameras.{format}")
    print(f"  - images.{format}")


def _write_cameras_text(cameras: dict, path: str):
    """Write cameras to Colmap text format."""
    HEADER = "# Camera list with one line of data per camera:\n" + \
             "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n" + \
             "# Number of cameras: {}\n".format(len(cameras))
    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, cam in cameras.items():
            to_write = [cam.id, cam.model, cam.width, cam.height, *cam.params]
            line = " ".join([str(elem) for elem in to_write])
            fid.write(line + "\n")


def _write_images_text(images: dict, path: str):
    """Write images to Colmap text format."""
    if len(images) == 0:
        mean_observations = 0
    else:
        mean_observations = sum((len(img.point3D_ids) for _, img in images.items())) / len(images)
    HEADER = "# Image list with two lines of data per image:\n" + \
             "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n" + \
             "#   POINTS2D[] as (X, Y, POINT3D_ID)\n" + \
             "# Number of images: {}, mean observations per image: {}\n".format(len(images), mean_observations)
    
    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, img in images.items():
            image_header = [img.id, *img.qvec, *img.tvec, img.camera_id, img.name]
            first_line = " ".join(map(str, image_header))
            fid.write(first_line + "\n")
            
            # Empty points line (no 2D observations for orbit cameras)
            fid.write("\n")


def _write_cameras_binary(cameras: dict, path: str):
    """Write cameras to Colmap binary format."""
    import struct
    
    def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
        data_type = {
            "i": "i",
            "Q": "Q",
            "d": "d",
        }
        bytes_data = b""
        for fmt in format_char_sequence:
            if fmt in data_type:
                if isinstance(data, (list, tuple, np.ndarray)):
                    for d in data:
                        bytes_data += struct.pack(endian_character + data_type[fmt], d)
                else:
                    bytes_data += struct.pack(endian_character + data_type[fmt], data)
        fid.write(bytes_data)
    
    CAMERA_MODEL_NAMES = {
        "SIMPLE_PINHOLE": 0,
        "PINHOLE": 1,
        "SIMPLE_RADIAL": 2,
        "RADIAL": 3,
        "OPENCV": 4,
        "OPENCV_FISHEYE": 5,
        "FULL_OPENCV": 6,
        "FOV": 7,
        "SIMPLE_RADIAL_FISHEYE": 8,
        "RADIAL_FISHEYE": 9,
        "THIN_PRISM_FISHEYE": 10,
    }
    
    with open(path, "wb") as fid:
        write_next_bytes(fid, len(cameras), "Q")
        for _, cam in cameras.items():
            model_id = CAMERA_MODEL_NAMES[cam.model]
            camera_properties = [cam.id, model_id, cam.width, cam.height]
            write_next_bytes(fid, camera_properties, "iiQQ")
            for p in cam.params:
                write_next_bytes(fid, float(p), "d")


def _write_images_binary(images: dict, path: str):
    """Write images to Colmap binary format."""
    import struct
    
    def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
        data_type = {
            "i": "i",
            "Q": "Q",
            "d": "d",
            "c": "c",
        }
        bytes_data = b""
        for fmt in format_char_sequence:
            if fmt in data_type:
                if isinstance(data, (list, tuple, np.ndarray)):
                    for d in data:
                        if fmt == "c":
                            bytes_data += struct.pack(endian_character + "c", d.encode("utf-8"))
                        else:
                            bytes_data += struct.pack(endian_character + data_type[fmt], d)
                else:
                    if fmt == "c":
                        bytes_data += struct.pack(endian_character + "c", data.encode("utf-8"))
                    else:
                        bytes_data += struct.pack(endian_character + data_type[fmt], data)
        fid.write(bytes_data)
    
    with open(path, "wb") as fid:
        write_next_bytes(fid, len(images), "Q")
        for _, img in images.items():
            write_next_bytes(fid, img.id, "i")
            write_next_bytes(fid, img.qvec.tolist(), "dddd")
            write_next_bytes(fid, img.tvec.tolist(), "ddd")
            write_next_bytes(fid, img.camera_id, "i")
            for char in img.name:
                write_next_bytes(fid, char, "c")
            write_next_bytes(fid, b"\x00", "c")
            write_next_bytes(fid, len(img.point3D_ids), "Q")
            # Empty points for orbit cameras
            # for xy, p3d_id in zip(img.xys, img.point3D_ids):
            #     write_next_bytes(fid, [*xy, p3d_id], "ddq")


def load_gaussians(model_path: str, iteration: int, sh_degree: int):
    gaussians = GaussianModel(sh_degree)
    if iteration == -1:
        iteration = searchForMaxIteration(os.path.join(model_path, "point_cloud"))
    ply_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    print(f"Loading Gaussian model from {ply_path}")
    gaussians.load_ply(ply_path)
    return gaussians, iteration


def main():
    parser = ArgumentParser(description="Render orbit path using COLMAP priors + look-at constraint")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--colmap_path", type=str, required=True, help="Dataset root containing sparse/0")
    parser.add_argument("--center", type=str, required=True, help="Object center as 'x,y,z' in COLMAP world coords")
    parser.add_argument("--iteration", type=int, default=-1, help="Which iteration to render (-1 = latest)")

    parser.add_argument("--n_views", type=int, default=120)
    parser.add_argument(
        "--elevation_deg",
        type=float,
        default=15.0,
        help="Orbit elevation in degrees (set negative to auto from COLMAP cameras)",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=-1.0,
        help="Orbit radius; -1 uses COLMAP camera distance-to-center statistics",
    )
    parser.add_argument(
        "--radius_quantile",
        type=float,
        default=0.35,
        help="When --radius=-1, use this quantile of COLMAP distances to center (smaller -> closer)",
    )
    parser.add_argument(
        "--use_gaussian_center",
        action="store_true",
        help="Ignore --center and use Gaussian mean XYZ as orbit center",
    )
    parser.add_argument(
        "--force_user_center",
        action="store_true",
        help="Always use --center even if it looks inconsistent with COLMAP cameras",
    )
    parser.add_argument(
        "--azimuth_center_deg",
        type=float,
        default=-999.0,
        help="Azimuth center in degrees around C (set < -900 to auto from COLMAP cameras)",
    )
    parser.add_argument(
        "--azimuth_span_deg",
        type=float,
        default=120.0,
        help="Total azimuth span in degrees around center (e.g. 120 -> from -60 to +60 deg)",
    )
    parser.add_argument(
        "--orbit_y_offset",
        type=float,
        default=0.0,
        help="Add this value to camera position Y (world). Positive = orbit shifted up (larger Y). "
             "Use to put the whole orbit higher without changing elevation angle.",
    )

    parser.add_argument("--render_width", type=int, default=512)
    parser.add_argument("--render_height", type=int, default=512)

    parser.add_argument("--out_dir", type=str, default="output/orbit")
    parser.add_argument("--video_path", type=str, default="output/orbit.mp4")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--save_colmap",
        type=str,
        default=None,
        help="Save orbit cameras to Colmap format at this directory (creates sparse/0 subdirectory). "
             "Use 'txt' or 'bin' for format, or omit for txt format.",
    )
    parser.add_argument(
        "--colmap_format",
        type=str,
        default="txt",
        choices=["txt", "bin"],
        help="Colmap file format: 'txt' or 'bin' (default: txt)",
    )

    args = get_combined_args(parser)
    safe_state(args.quiet)

    gaussians, used_iter = load_gaussians(args.model_path, args.iteration, model.extract(args).sh_degree)
    gaussian_center = gaussians.get_xyz.detach().mean(dim=0).float().cpu().numpy()

    C_user = _parse_center(args.center)

    cam_centers_np, fovy, colmap_cameras = load_colmap_prior(args.colmap_path)  # (N,3), rad, dict

    # Choose a stable center automatically unless forced.
    if args.use_gaussian_center:
        C = gaussian_center
        center_msg = "gaussian(user)"
    else:
        d_user = np.linalg.norm(cam_centers_np - C_user[None, :], axis=1)
        d_g = np.linalg.norm(cam_centers_np - gaussian_center[None, :], axis=1)
        med_user = float(np.median(d_user))
        med_g = float(np.median(d_g))
        delta_norm = float(np.linalg.norm(C_user - gaussian_center))
        print(
            f"Center diagnostics: |user-gaussian|={delta_norm:.4f} | "
            f"median_dist_to_user={med_user:.4f} median_dist_to_gaussian={med_g:.4f}"
        )

        if args.force_user_center:
            C = C_user
            center_msg = "user(forced)"
        else:
            if med_g < med_user:
                C = gaussian_center
                center_msg = "gaussian(auto)"
            else:
                C = C_user
                center_msg = "user(auto)"

    vecs = cam_centers_np - C[None, :]
    dists_to_C = np.linalg.norm(vecs, axis=1)

    # Auto radius: use a lower quantile to stay closer by default.
    if float(args.radius) > 0:
        radius = float(args.radius)
        radius_msg = "user"
    else:
        q = float(np.clip(float(args.radius_quantile), 0.05, 0.95))
        radius = float(np.quantile(dists_to_C, q))
        radius_msg = f"colmap_quantile(q={q:.2f})"
    radius = float(max(radius, 1e-3))
    # Make camera 2x closer to center
    radius = radius * 0.8 ## 이게 물체와의 거리를 결정(값이 작아지면 물체와 가까워짐)

    # Auto elevation: median elevation of COLMAP cameras around C
    if float(args.elevation_deg) < 0:
        elevs = np.arcsin(np.clip(vecs[:, 1] / (np.linalg.norm(vecs, axis=1) + 1e-8), -1.0, 1.0))
        elevation_deg = float(np.median(elevs) * 180.0 / np.pi)
        elev_msg = "auto_median"
    else:
        elevation_deg = float(args.elevation_deg)
        elev_msg = "user"

    print(
        f"COLMAP stats wrt center[{center_msg}]: dist min/median/max = "
        f"{float(dists_to_C.min()):.4f}/{float(np.median(dists_to_C)):.4f}/{float(dists_to_C.max()):.4f} | "
        f"orbit radius={radius:.4f} ({radius_msg}), elevation={elevation_deg:.2f}deg ({elev_msg}), "
        f"orbit_y_offset={float(args.orbit_y_offset):.4f}, fovy={fovy:.4f}rad"
    )
    bg_color = [1, 1, 1] if model.extract(args).white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    os.makedirs(args.out_dir, exist_ok=True)
    frames = []
    orbit_cameras = []  # Store cameras for Colmap export

    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    elev = np.deg2rad(float(elevation_deg))

    # Azimuth center: auto from COLMAP if not provided
    if float(args.azimuth_center_deg) < -900.0:
        # project vecs onto XZ plane and compute azimuths
        flat = vecs.copy()
        flat[:, 1] = 0.0
        azs = np.arctan2(flat[:, 2], flat[:, 0])
        az_center = float(np.median(azs))
        az_center_msg = "auto_median"
    else:
        az_center = np.deg2rad(float(args.azimuth_center_deg))
        az_center_msg = "user"

    span_deg = float(args.azimuth_span_deg)
    span_rad = np.deg2rad(span_deg)
    az_start = az_center - span_rad / 2.0
    az_end = az_center + span_rad / 2.0

    print(
        f"Azimuth: center={np.rad2deg(az_center):.2f}deg ({az_center_msg}), "
        f"span={span_deg:.2f}deg -> [{np.rad2deg(az_start):.2f}, {np.rad2deg(az_end):.2f}]deg"
    )

    n_views = int(args.n_views)
    for i in range(n_views):
        t = i / float(max(n_views - 1, 1))
        az = az_start + (az_end - az_start) * t
        offset = np.array(
            [
                radius * np.cos(az) * np.cos(elev),
                radius * np.sin(elev),
                radius * np.sin(az) * np.cos(elev),
            ],
            dtype=np.float32,
        )
        P = C + offset
        P[1] += float(args.orbit_y_offset)  # shift orbit up/down (larger Y = higher)
        c2w = look_at_c2w(P, C, world_up)

        R_c2w = c2w[:3, :3].astype(np.float32)
        cam_center = c2w[:3, 3].astype(np.float32)
        # Simple_Camera expects R such that W2C = [R^T, T].
        # Given C2W=[R_c2w, cam_center], we set:
        #   R = R_c2w
        #   T = -R_c2w^T @ cam_center
        R = R_c2w
        T = (-(R_c2w.T) @ cam_center).astype(np.float32)

        h = int(args.render_height)
        w = int(args.render_width)
        fovx = fovy_to_fovx(fovy, h=h, w=w)

        cam = Simple_Camera(
            colmap_id=i,
            R=R,
            T=T,
            FoVx=float(fovx),
            FoVy=float(fovy),
            h=h,
            w=w,
            image_name=f"orbit_{i:05d}",
            uid=i,
            data_device="cuda",
            qvec=None,
        )
        orbit_cameras.append(cam)  # Store for Colmap export

        out = render(cam, gaussians, pipeline.extract(args), background)
        rgb = out["render"]
        if i == 0:
            vis = out.get("visibility_filter", None)
            if vis is not None:
                vis_ratio = float(vis.float().mean().item())
                print(f"[debug] orbit0 visibility_ratio={vis_ratio:.6f}")
            depth = out.get("depth_3dgs", None)
            if depth is not None:
                dmin = float(depth.min().item())
                dmax = float(depth.max().item())
                print(f"[debug] orbit0 depth_3dgs min/max = {dmin:.6f}/{dmax:.6f}")
            mean_rgb = float(rgb.mean().item())
            max_rgb = float(rgb.max().item())
            print(f"[debug] orbit0 rgb mean/max = {mean_rgb:.6f}/{max_rgb:.6f}")

        out_path = os.path.join(args.out_dir, f"{i:05d}.png")
        torchvision.utils.save_image(rgb, out_path)
        frames.append((rgb.clamp(0.0, 1.0) * 255.0).byte().permute(1, 2, 0).cpu().numpy())

        if (i + 1) % 10 == 0:
            print(f"Rendered {i+1}/{int(args.n_views)}")

    if args.video_path:
        import imageio

        os.makedirs(os.path.dirname(args.video_path), exist_ok=True)
        print(f"Writing video to {args.video_path} (fps={int(args.fps)})")
        imageio.mimsave(args.video_path, frames, fps=int(args.fps))
    
    # Save orbit cameras to Colmap format if requested (camera intrinsics match source cameras.bin)
    if args.save_colmap:
        source_cam = list(colmap_cameras.values())[0]  # use first camera from bin so output matches bin
        save_orbit_cameras_to_colmap(
            orbit_cameras,
            args.save_colmap,
            camera_id=1,
            format=args.colmap_format,
            source_camera=source_cam,
        )
        print(f"\nOrbit cameras saved to Colmap format at: {args.save_colmap}/sparse/0/")
        print(f"  You can use this path as --colmap_path in DGE for rendering these views.")


if __name__ == "__main__":
    main()

