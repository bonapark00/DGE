#!/usr/bin/env python3
"""
Generate-by-Lens: Intelligent camera view generation for 3D Gaussian Splatting editing.

This script selects optimal camera views for IP2P-based editing by analysing the
3D Gaussian distribution of the ROI (Region of Interest).

Pipeline:
  Step 1 – ROI Intrinsic Analysis   (Weighted PCA on ROI Gaussians)
  Step 2 – SAGE-Probing             (Sharpness-Aware Guided Editability probing)
            Finds optimal distance d* using:
              S = Focus - λ₁·Leakage - λ₂·Entropy(A) - Penalty_size
            Hard filter: H(A) > τ → unsafe view (-inf)
  Step 3 – Fibonacci Manifold Sampling (uniform candidate views on a sphere at d*)
  Step 4 – Energy-based Scoring     (visibility + canonical alignment)
  Step 5 – Diversity-aware Selection (Farthest Point Sampling on top candidates)

Outputs:
  - PNG frames in --out_dir
  - MP4 video  at --video_path
  - (Optional) COLMAP cameras via --save_colmap
"""

"""
python generate_by_lens.py \
        --ply_path /data/users/jaeyeonpark/3dgs-trained/3d-ovs/covered_desk/point_cloud/iteration_30000/point_cloud.ply \
        --colmap_path /data/users/jaeyeonpark/dataset/3d-ovs/covered_desk \
        --seg_prompt "head of the pooh" \
        --edit_prompt "Make the pooh wear sunglasses on his eyes" \
        --use_ip2p_scoring \
        --distance_multipliers "5.0, 6.0, 7.0, 8.0, 9.0, 10.0" \
        --visualize_roi \
        --n_select 20 \
        --guidance_scale 7.5 \
        --save_attn_grid output/attn_grid_pooh_sunglasses.jpg \
        --gpu 0 \
        --v_front_method scene_center \
        --num_inference_steps 5 --guidance_scale 7.5 --ip2p_batch_size 2
        

 python generate_by_lens.py \
        --ply_path /data/users/jaeyeonpark/3dgs-trained/3d-ovs/covered_desk/point_cloud/iteration_30000/point_cloud.ply \
        --colmap_path /data/users/jaeyeonpark/dataset/3d-ovs/covered_desk \
        --seg_prompt "red sweater" \
        --edit_prompt "Change the red sweater into a leather jacket" \
        --use_ip2p_scoring \
        --distance_multipliers "2.0, 2.5, 3.0, 4.0, 5.0, 6.0" \
        --visualize_roi \
        --n_select 20 \
        --save_attn_grid output/attn_grid.jpg \
        --gpu 0
        # --save_colmap output/lens_colmap \
        # --video_path output/lens_claude.mp4 \

 python generate_by_lens.py \
        --ply_path /data/users/jaeyeonpark/3dgs-trained/3d-ovs/covered_desk/point_cloud/iteration_30000/point_cloud.ply \
        --colmap_path /data/users/jaeyeonpark/dataset/3d-ovs/covered_desk \
        --seg_prompt "head of the pooh" \
        --edit_prompt "Make the pooh wear sunglasses on his eyes" \
        --use_ip2p_scoring \
        --distance_multipliers "5.0, 6.0, 7.0, 8.0, 9.0, 10.0" \
        --visualize_roi \
        --n_select 20 \
        --guidance_scale 4.0 \
        --save_attn_grid output/attn_grid_pooh_sunglasses.jpg \
        --gpu 0 \
        --v_front_method scene_center
        --distance_multipliers "3.0, 4.0, 5.0, 6.0, 7.0" \

 python generate_by_lens.py \
        --ply_path /data/users/jaeyeonpark/3dgs-trained/3d-ovs/covered_desk/point_cloud/iteration_30000/point_cloud.ply \
        --colmap_path /data/users/jaeyeonpark/dataset/3d-ovs/covered_desk \
        --seg_prompt "sweater of the pooh" \
        --edit_prompt "Change the sweather of the pooh into a fleece jacket" \
        --use_ip2p_scoring \
        --distance_multipliers "2.0, 2.5, 3.0, 4.0, 5.0, 6.0" \
        --visualize_roi \
        --n_select 20 \
        --guidance_scale 4.0 \
        --save_attn_grid output/attn_grid_pooh_sweater_to_leather_jacket.jpg \
        --gpu 1 \
        --v_front_method scene_center
 
 python generate_by_lens.py \
        --ply_path /data/users/jaeyeonpark/3dgs-trained/3d-ovs/covered_desk/point_cloud/iteration_30000/point_cloud.ply \
        --colmap_path /data/users/jaeyeonpark/dataset/3d-ovs/covered_desk \
        --seg_prompt "pooh" \
        --edit_prompt "Make the pooh look like a robot" \
        --use_ip2p_scoring \
        --distance_multipliers "2.0, 2.5, 3.0, 4.0, 5.0, 6.0" \
        --visualize_roi \
        --n_select 20 \
        --guidance_scale 4.0 \
        --save_attn_grid output/attn_grid_pooh_sweater_to_leather_jacket.jpg \
        --gpu 1 \
        --v_front_method scene_center


 python generate_by_lens.py \
        --ply_path /data/users/jaeyeonpark/3dgs-trained/3d-ovs/room/point_cloud/iteration_30000/point_cloud.ply \
        --colmap_path /data/users/jaeyeonpark/dataset/3d-ovs/room \
        --seg_prompt "eyes of the rabbit" \
        --edit_prompt "Make the rabbit wear sunglasses on his eyes" \
        --use_ip2p_scoring \
        --distance_multipliers "0.2, 0.5, 0.7, 1.0, 2.0" \
        --visualize_roi \
        --n_select 20 \
        --save_attn_grid output/attn_grid_rabbit_sunglasses.jpg \
        --gpu 1 \
        --v_front_method scene_center \
        --guidance_scale 6.0

python generate_by_lens.py \
        --ply_path /data/users/jaeyeonpark/3dgs-trained/3d-ovs/room/point_cloud/iteration_30000/point_cloud.ply \
        --colmap_path /data/users/jaeyeonpark/dataset/3d-ovs/room \
        --seg_prompt "face of the grey rabbit figure" \
        --edit_prompt "Give the rabbit figure a pair of small sunglasses" \
        --use_ip2p_scoring \
        --distance_multipliers " 1.0, 1.5, 2.0, 2.5" \
        --visualize_roi \
        --n_select 20 \
        --save_attn_grid output/attn_grid_rabbit_sunglasses.jpg \
        --gpu 1 \
        --v_front_method scene_center --guidance_scale 7.5

python generate_by_lens.py \
        --ply_path /data/users/jaeyeonpark/3dgs-trained/3d-ovs/room/point_cloud/iteration_30000/point_cloud.ply \
        --colmap_path /data/users/jaeyeonpark/dataset/3d-ovs/room \
        --seg_prompt "neck of the white rabbit" \
        --edit_prompt "Add a tiny bow tie to the rabbit's neck" \
        --use_ip2p_scoring \
        --distance_multipliers "0.7, 1.0, 1.5, 2.0, 3.0" \
        --visualize_roi \
        --n_select 20 \
        --save_attn_grid output/attn_grid_rabbit_bow_tie.jpg \
        --gpu 1 \
        --v_front_method scene_center --guidance_scale 5.0


 python generate_by_lens.py \
        --ply_path /data/users/jaeyeonpark/3dgs-trained/3d-ovs/room/point_cloud/iteration_30000/point_cloud.ply \
        --colmap_path /data/users/jaeyeonpark/dataset/3d-ovs/room \
        --seg_prompt "dinosaur" \
        --edit_prompt "Change the dinosaur figure to green" \
        --use_ip2p_scoring \
        --distance_multipliers "2.0, 2.5, 3.0, 4.0, 5.0, 6.0" \
        --visualize_roi \
        --n_select 20 \
        --save_attn_grid output/attn_grid_dino_green.jpg \
        --gpu 1 \
        --v_front_method scene_center


     python generate_by_lens.py \
        --ply_path /data/users/jaeyeonpark/3dgs-trained/in2n-GSEditor/face/point_cloud/iteration_30000/point_cloud.ply \
        --colmap_path /data/users/jaeyeonpark/dataset/in2n-GSEditor/face \
        --seg_prompt "man" \
        --edit_prompt "Turn the man into a spiderman with a mask" \
        --use_ip2p_scoring \
        --distance_multipliers "2.0, 2.5, 3.0, 4.0, 5.0, 6.0" \
        --visualize_roi \
        --n_select 20 \
        --save_attn_grid output/attn_grid_man_to_spiderman.jpg \
        --gpu 3 \
        --n_candidates 900 \
        --v_front_method scene_center \
        --cone_half_angle_deg 60 --diversity_y_variance_weight 1.0 --diversity_x_weight 30.0
  
        --distance_multipliers "2.0, 2.5, 3.0, 4.0, 5.0, 6.0" \

python generate_by_lens.py \
        --ply_path /data/users/jaeyeonpark/3dgs-trained/in2n-GSEditor/face/point_cloud/iteration_30000/point_cloud.ply \
        --colmap_path /data/users/jaeyeonpark/dataset/in2n-GSEditor/face \
        --seg_prompt "a face" \
        --edit_prompt "Turn the man into a clown" \
        --use_ip2p_scoring \
        --distance_multipliers "3.0, 3.5, 4.0, 4.5, 5.0" \
        --visualize_roi \
        --n_select 20 \
        --save_attn_grid output/attn_grid_man_to_clown.jpg \
        --gpu 2 \
        --n_candidates 900 \
        --v_front_method scene_center \
        --cone_half_angle_deg 60 --diversity_y_variance_weight 1.0 --diversity_x_weight 30.0 \
        --num_inference_steps 5 --guidance_scale 7.5 --ip2p_batch_size 2
  
python generate_by_lens.py \
        --ply_path /data/users/jaeyeonpark/3dgs-trained/3d-ovs/blue_sofa/point_cloud/iteration_30000/point_cloud.ply \
        --colmap_path /data/users/jaeyeonpark/dataset/3d-ovs/blue_sofa \
        --seg_prompt "speaker" \
        --edit_prompt "Change the speaker to a shiny gold texture" \
        --use_ip2p_scoring \
        --distance_multipliers "3.0, 4.0, 5.0, 6.0, 7.0" \
        --visualize_roi \
        --n_select 20 \
        --guidance_scale 5.5 \
        --save_attn_grid output/attn_grid_speaker_to_gold.jpg \
        --gpu 0 \
        --v_front_method scene_center \
        --num_inference_steps 5 --guidance_scale 5.5 --ip2p_batch_size 2
  
python generate_by_lens.py \
        --ply_path /data/users/jaeyeonpark/3dgs-trained/3d-ovs/blue_sofa/point_cloud/iteration_30000/point_cloud.ply \
        --colmap_path /data/users/jaeyeonpark/dataset/3d-ovs/blue_sofa \
        --seg_prompt "plush toy" \
        --edit_prompt "Change the plush toy's color to pink" \
        --use_ip2p_scoring \
        --distance_multipliers "3.0, 4.0, 5.0, 6.0, 7.0" \
        --visualize_roi \
        --n_select 20 \
        --guidance_scale 5.5 \
        --save_attn_grid output/attn_grid_plush_toy_pink.jpg \
        --gpu 1 \
        --v_front_method scene_center \
        --num_inference_steps 5 --guidance_scale 5.5 --ip2p_batch_size 2
  

python generate_by_lens.py \
        --ply_path /data/users/jaeyeonpark/3dgs-trained/3d-ovs/room/point_cloud/iteration_30000/point_cloud.ply \
        --colmap_path /data/users/jaeyeonpark/dataset/3d-ovs/room \
        --seg_prompt "neck of the yellow rubber chicken" \
        --edit_prompt "Add a necktie to the rubber chicken's neck" \
        --use_ip2p_scoring \
        --distance_multipliers "0.7, 1.0, 1.2, 1.4, 1.6, 2.0" \
        --visualize_roi \
        --n_select 20 \
        --save_attn_grid output/attn_grid_chicken_necktie.jpg \
        --gpu 0 \
        --v_front_method scene_center --guidance_scale 7.5 \
        --n_candidates 900 \
        --v_front_method scene_center \
        --cone_half_angle_deg 60 --diversity_y_variance_weight 1.0 --diversity_x_weight 30.0 


"""


# Compatibility patch for huggingface_hub  (must be first)
import hf_hub_patch  # noqa: E402, F401

# Set CUDA_VISIBLE_DEVICES from --gpu / --device before any torch import, so that
# GaussianModel, Simple_Camera, and the renderer (all use .cuda() / device="cuda")
# see a single GPU and avoid cross-device illegal memory access.
import os
import sys
for i, arg in enumerate(sys.argv):
    if arg == "--gpu" and i + 1 < len(sys.argv):
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(int(sys.argv[i + 1]))
            break
        except ValueError:
            pass
    if arg == "--device" and i + 1 < len(sys.argv):
        val = sys.argv[i + 1]
        if val.startswith("cuda:") and ":" in val:
            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(int(val.split(":")[1]))
                break
            except ValueError:
                pass
    if arg.startswith("--device="):
        val = arg.split("=", 1)[1]
        if val.startswith("cuda:") and ":" in val:
            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(int(val.split(":")[1]))
                break
            except ValueError:
                pass

import collections
import math
from argparse import ArgumentParser
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch
from torch import nn
import torchvision

from gaussiansplatting.arguments import ModelParams, PipelineParams, get_combined_args
from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.scene.cameras import Simple_Camera
from gaussiansplatting.scene.colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_intrinsics_binary,
    rotmat2qvec,
)
from gaussiansplatting.scene.vanilla_gaussian_model import GaussianModel
from gaussiansplatting.utils.general_utils import safe_state
from gaussiansplatting.utils.graphics_utils import (
    focal2fov,
    fov2focal,
    getWorld2View2,
)
from threestudio.utils.latency import LatencyLogger


@contextmanager
def latency_timeit(
    latency_logger: Optional[LatencyLogger],
    name: str,
    device: str,
):
    """Time a block with optional CUDA synchronization.

    CUDA ops are asynchronous; without synchronization, measured wall time can
    under-report actual GPU compute. This context manager synchronizes before
    starting and before stopping the timer when using CUDA.
    """
    if latency_logger is None:
        yield
        return

    do_sync = bool(device) and str(device).startswith("cuda") and torch.cuda.is_available()
    if do_sync:
        torch.cuda.synchronize()

    with latency_logger.timeit(name):
        try:
            yield
        finally:
            if do_sync:
                torch.cuda.synchronize()

# Re-use Colmap namedtuples from render_orbit_from_colmap
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"]
)
Image = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > eps else v


def look_at_c2w(
    eye: np.ndarray, target: np.ndarray, world_up: np.ndarray
) -> np.ndarray:
    # "Camera-to-world 4x4 (camera looks along +Z towards *target*)."
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = _normalize(np.asarray(world_up, dtype=np.float32))

    forward = _normalize(target - eye)
    right = np.cross(up, forward)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(up, forward)
    right = _normalize(right)
    cam_up = np.cross(forward, right)

    R_c2w = np.stack([right, cam_up, forward], axis=1)  # (3,3)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R_c2w
    c2w[:3, 3] = eye
    return c2w


def c2w_to_RT(c2w: np.ndarray):
    """Convert C2W 4x4 → (R, T) used by Simple_Camera (W2C convention)."""
    R_c2w = c2w[:3, :3].astype(np.float32)
    cam_center = c2w[:3, 3].astype(np.float32)
    R = R_c2w  # Simple_Camera stores R_c2w
    T = (-(R_c2w.T) @ cam_center).astype(np.float32)
    return R, T


def fovy_to_fovx(fovy: float, h: int, w: int) -> float:
    fy = fov2focal(float(fovy), int(h))
    fx = fy * (float(w) / float(h))
    return float(focal2fov(float(fx), int(w)))


# ---------------------------------------------------------------------------
# Gaussian / COLMAP loaders
# ---------------------------------------------------------------------------

def load_gaussians(ply_path: str, sh_degree: int):
    gaussians = GaussianModel(sh_degree)
    print(f"[lens] Loading Gaussians from {ply_path}")
    gaussians.load_ply(ply_path)
    return gaussians


def _prune_gaussians_by_mask(gaussians: GaussianModel, keep_mask: torch.Tensor) -> int:
    """
    Prune Gaussians in-place by keeping only points where keep_mask is True.
    Does not use optimizer (for inference-only models). Returns number of points removed.
    """
    device = gaussians.get_xyz.device
    keep_mask = keep_mask.to(device)
    n_before = keep_mask.shape[0]
    n_keep = int(keep_mask.sum().item())
    n_remove = n_before - n_keep
    if n_remove == 0:
        return 0
    # Replace parameters (model has no optimizer in generate_by_lens)
    gaussians._xyz = nn.Parameter(gaussians._xyz[keep_mask].detach().clone().requires_grad_(True))
    gaussians._features_dc = nn.Parameter(gaussians._features_dc[keep_mask].detach().clone().requires_grad_(True))
    gaussians._features_rest = nn.Parameter(gaussians._features_rest[keep_mask].detach().clone().requires_grad_(True))
    gaussians._opacity = nn.Parameter(gaussians._opacity[keep_mask].detach().clone().requires_grad_(True))
    gaussians._scaling = nn.Parameter(gaussians._scaling[keep_mask].detach().clone().requires_grad_(True))
    gaussians._rotation = nn.Parameter(gaussians._rotation[keep_mask].detach().clone().requires_grad_(True))
    if gaussians.max_radii2D.shape[0] == n_before:
        gaussians.max_radii2D = gaussians.max_radii2D[keep_mask].detach().clone()
    if gaussians.xyz_gradient_accum.shape[0] == n_before:
        gaussians.xyz_gradient_accum = gaussians.xyz_gradient_accum[keep_mask].detach().clone()
    if gaussians.denom.shape[0] == n_before:
        gaussians.denom = gaussians.denom[keep_mask].detach().clone()
    return n_remove


def load_colmap_prior(colmap_path: str):
    sparse0 = os.path.join(os.path.abspath(colmap_path), "sparse", "0")
    images = read_extrinsics_binary(os.path.join(sparse0, "images.bin"))
    cams = read_intrinsics_binary(os.path.join(sparse0, "cameras.bin"))

    cam_centers, cam_forwards = [], []
    first_intr = None
    for img in images.values():
        intr = cams[img.camera_id]
        if first_intr is None:
            first_intr = intr
        R = np.transpose(qvec2rotmat(img.qvec)).astype(np.float32)
        T_vec = np.array(img.tvec, dtype=np.float32)
        W2C = getWorld2View2(R, T_vec)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3].astype(np.float32))
        cam_forwards.append(C2W[:3, 2].astype(np.float32))  # +Z = forward

    h = int(first_intr.height)
    if first_intr.model == "PINHOLE":
        fy = float(first_intr.params[1])
    elif first_intr.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
        fy = float(first_intr.params[0])
    else:
        fy = float(first_intr.params[0])
    fovy = float(focal2fov(fy, h))
    print(f"fovy: {fovy}, fy: {fy}, h: {h}")

    return (
        np.stack(cam_centers, axis=0),
        np.stack(cam_forwards, axis=0),
        fovy,
        cams,
    )


# ===================================================================
# Step 1: ROI Intrinsic Analysis  (Weighted PCA)
# ===================================================================

def roi_intrinsic_analysis(
    gaussians: GaussianModel,
    roi_mask: Optional[torch.Tensor],
    cam_forwards: np.ndarray,
    cam_centers: Optional[np.ndarray] = None,
    v_front_method: str = "colmap_mean",
) -> Dict:
    """
    Perform weighted PCA on ROI Gaussians to find the object centre,
    principal axes, and a robust *front* direction.

    v_front_method:
        "colmap_mean": v_front from mean COLMAP view direction (minus v1 component).
        "scene_center": v_front from ROI center toward scene center (cam_centers.mean), then v1 removed.

    Returns dict with keys:
        center, v1, v2, v3 (eigenvectors, descending eigenvalue),
        eigenvalues, object_size, v_front
    """
    xyz = gaussians.get_xyz.detach()  # (N, 3)
    opacity = gaussians.get_opacity.detach().squeeze(-1)  # (N,)

    if roi_mask is not None:
        roi_mask = roi_mask.to(xyz.device).bool()
        xyz = xyz[roi_mask]
        opacity = opacity[roi_mask]

    weights = opacity  # α_i
    w_sum = weights.sum() + 1e-8

    # Weighted centre
    center = (weights[:, None] * xyz).sum(dim=0) / w_sum  # (3,)

    # Weighted covariance
    diff = xyz - center[None, :]  # (N, 3)
    C = (weights[:, None, None] * (diff.unsqueeze(2) * diff.unsqueeze(1))).sum(
        dim=0
    ) / w_sum  # (3,3)

    eigenvalues, eigenvectors = torch.linalg.eigh(C.float())  # ascending order
    # Flip to descending
    eigenvalues = eigenvalues.flip(0)
    eigenvectors = eigenvectors.flip(1)

    v1 = eigenvectors[:, 0].cpu().numpy()
    v2 = eigenvectors[:, 1].cpu().numpy()
    v3 = eigenvectors[:, 2].cpu().numpy()
    evals = eigenvalues.cpu().numpy()

    # Object size: 2σ along the *median* axis (avoids elongation bias)
    object_size = float(2.0 * np.sqrt(np.median(np.abs(evals))))

    center_np = center.cpu().numpy()

    # Front direction
    if v_front_method == "scene_center":
        if cam_centers is None or len(cam_centers) == 0:
            raise ValueError("v_front_method=scene_center requires cam_centers")
        scene_center = np.array(cam_centers, dtype=np.float64).mean(axis=0).astype(np.float32)
        raw = scene_center - center_np
        nrm = float(np.linalg.norm(raw))
        if nrm < 1e-8:
            v_front = v3.copy()
            print("[Step1] v_front=scene_center: degenerate (scene_center≈ROI center), using v3")
        else:
            # Use direction to scene center as-is (no v1 removal), so it actually differs from colmap_mean.
            # v1 removal would zero out when ROI→scene_center is parallel to v1, making result match v3/colmap_mean.
            v_front = (raw / nrm).astype(np.float32)
            print("[Step1] v_front=scene_center (ROI → scene center)")
    else:
        # colmap_mean: direction from object centre toward cameras (mean view direction)
        # cam_forwards points camera→scene, so negate to get scene→camera.
        mean_fwd = _normalize(cam_forwards.mean(axis=0))
        proj_on_v1 = np.dot(mean_fwd, v1) * v1
        v_front = _normalize(-(mean_fwd - proj_on_v1))
        if np.linalg.norm(v_front) < 1e-6:
            v_front = v3.copy()
            print("[Step1] v_front=colmap_mean: degenerate, using v3")
        else:
            print("[Step1] v_front=colmap_mean")

    print(
        f"[Step1] ROI center={center_np}, "
        f"eigenvalues={evals}, object_size={object_size:.4f}"
    )
    return dict(
        center=center_np,
        v1=v1,
        v2=v2,
        v3=v3,
        eigenvalues=evals,
        object_size=object_size,
        v_front=v_front,
    )


# ===================================================================
# Step 2: Scale Probing with Fast Preview
# ===================================================================

def _make_camera(
    eye: np.ndarray,
    target: np.ndarray,
    world_up: np.ndarray,
    fovy: float,
    h: int,
    w: int,
    uid: int,
    device: str = "cuda",
) -> Simple_Camera:
    c2w = look_at_c2w(eye, target, world_up)
    R, T = c2w_to_RT(c2w)
    fovx = fovy_to_fovx(fovy, h, w)
    return Simple_Camera(
        colmap_id=uid,
        R=R,
        T=T,
        FoVx=float(fovx),
        FoVy=float(fovy),
        h=h,
        w=w,
        image_name=f"probe_{uid:05d}",
        uid=uid,
        data_device=device,
        qvec=None,
    )


def compute_entropy(attention_map: torch.Tensor) -> float:
    """
    Calculate Shannon Entropy of an attention map.
    Lower entropy means sharper, more focused attention (better for editing).

    Args:
        attention_map: (H, W) tensor of attention values.
    Returns:
        Scalar entropy value.
    """
    epsilon = 1e-10
    # Normalize to probability distribution
    P = attention_map / (attention_map.sum() + epsilon)
    # Shannon entropy: -sum(p * log(p))
    entropy = -torch.sum(P * torch.log(P + epsilon))
    return float(entropy)


class _StoringAttnProcessor:
    """
    Custom AttnProcessor that *accumulates* cross-attention probabilities
    across all forward calls (i.e. across denoising timesteps).

    diffusers 0.19.3's default AttnProcessor2_0 uses F.scaled_dot_product_attention
    which never materializes attention weights. This processor falls back to explicit
    Q·K^T → softmax → V computation so we can capture the attention map.
    """

    def __init__(self):
        self.attn_probs_list: List[torch.Tensor] = []

    def reset(self):
        self.attn_probs_list = []

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # Accumulate across timesteps
        self.attn_probs_list.append(attention_probs.detach().cpu())

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class _StoringSAOnlySmallProcessor:
    """SA processor that only captures attention at a target spatial resolution.

    For self-attention layers whose sequence_length matches *target_spatial*
    (default 64 = 8×8), this behaves like _StoringAttnProcessor: explicit
    Q·K^T → softmax so we can store the attention map.

    For all other resolutions (16×16, 32×32, 64×64) it uses
    F.scaled_dot_product_attention (FlashAttention) — no attention map is
    materialised and no data is stored.  This avoids the O(n²) cost of
    naive attention at high resolutions while still capturing the 8×8 SA
    maps needed by _compute_sa_propagation_leakage.
    """

    def __init__(self, target_spatial: int = 64):
        self.target_spatial = target_spatial
        self.attn_probs_list: List[torch.Tensor] = []

    def reset(self):
        self.attn_probs_list = []

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None):
        import torch.nn.functional as F

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        if sequence_length == self.target_spatial:
            # Small resolution → explicit attention + store
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            self.attn_probs_list.append(attention_probs.detach().cpu())
            hidden_states = torch.bmm(attention_probs, value)
        else:
            # Large resolution → FlashAttention (no storage)
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask,
            )

        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


def _tokenize_and_find_keyword_indices(
    ip2p_pipe, prompt: str
) -> List[int]:
    """Return token indices for non-special, non-padding tokens in *prompt*."""
    tokenizer = ip2p_pipe.tokenizer
    tokens = tokenizer(prompt, return_tensors="pt", padding=False)
    input_ids = tokens["input_ids"][0].tolist()
    # Skip BOS (index 0) and EOS (last) — keep content tokens
    # BOS = 49406, EOS = 49407 for CLIP tokenizer
    indices = []
    for i, tid in enumerate(input_ids):
        if tid not in (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id):
            indices.append(i)
    return indices if indices else list(range(1, min(len(input_ids) - 1, 10)))


def _compute_sa_propagation_leakage(
    sa_storing_processors: Dict,
    roi_mask_2d: torch.Tensor,
    target_spatial: int = 256,
) -> float:
    """Measure self-attention propagation leakage from ROI to background.

    For each background pixel q, compute: sum of attention weights to ROI pixels.
    This captures how much the edit signal "flows" from ROI → background through
    self-attention, which cross-attention maps cannot reveal.

    Fast implementation:
    - Uses the smallest available SA spatial resolution (8×8=64) to avoid
      materializing large (256,256) matrices.
    - Online-averages across timesteps/layers instead of torch.stack,
      keeping memory at O(spatial²) = O(64) rather than O(N × spatial²).
    - Computes the leakage scalar directly from (bg_vec @ avg_sa @ roi_vec)
      without materializing the masked (spatial,spatial) product.

    Returns:
        sa_leakage in [0, 1]: average attention that background pixels pay to ROI.
        Higher = more edit signal leaks to background.
    """
    # Prefer the coarsest (smallest) resolution: 8×8=64 tokens.
    # If unavailable fall back upward through 16×16=256, 32×32=1024.
    PREFERRED = [64, 256, 1024]
    use_spatial: Optional[int] = None
    for sp_pref in PREFERRED:
        for sp in sa_storing_processors.values():
            for ap in sp.attn_probs_list:
                if ap.shape[-1] == sp_pref and ap.shape[-2] == sp_pref:
                    use_spatial = sp_pref
                    break
            if use_spatial is not None:
                break
        if use_spatial is not None:
            break

    if use_spatial is None:
        return 0.0

    side = int(math.sqrt(use_spatial))

    # --- Build ROI / BG masks at this resolution (cheap) ---
    M = roi_mask_2d.float().squeeze(0)  # (H, W)
    M_small = torch.nn.functional.interpolate(
        M[None, None], size=(side, side), mode="bilinear",
    ).squeeze().cpu()  # (side, side)
    M_flat = (M_small > 0.3).float().flatten()  # (use_spatial,)
    bg_flat = 1.0 - M_flat
    n_bg = float(bg_flat.sum())
    n_roi = float(M_flat.sum())
    if n_bg < 1 or n_roi < 1:
        return 0.0

    # --- Online mean over all SA maps at this resolution ---
    # avg_sa: (use_spatial, use_spatial) — accumulated in float32 on CPU
    # We never stack all maps; we add them one by one to save memory.
    avg_sa = torch.zeros(use_spatial, use_spatial, dtype=torch.float32)
    count = 0
    for sp in sa_storing_processors.values():
        for ap in sp.attn_probs_list:
            if ap.shape[-1] == use_spatial and ap.shape[-2] == use_spatial:
                # ap: (batch*heads, spatial, spatial) — average over head dim on-the-fly
                avg_sa += ap.float().mean(dim=0)  # (spatial, spatial)
                count += 1
    if count == 0:
        return 0.0
    avg_sa /= count  # true mean over timesteps × layers

    # --- Leakage scalar: bg_flat @ avg_sa @ roi_flat / n_bg ---
    # Uses two matrix-vector products instead of element-wise (spatial, spatial) mask:
    #   step 1: roi_attn_col = avg_sa @ roi_flat  → (spatial,)  "how much each pixel attends to ROI"
    #   step 2: bg_attention_to_roi = bg_flat · roi_attn_col    → scalar
    roi_attn_col = avg_sa.mv(M_flat)       # (spatial,) — sum of attention to ROI per query pixel
    sa_leakage = float((bg_flat * roi_attn_col).sum() / n_bg)

    return sa_leakage


def _aggregate_attention_at_resolution(
    storing_processors: Dict,
    target_spatial: int,
    keyword_indices: List[int],
) -> Optional[torch.Tensor]:
    """Aggregate stored attention maps for a single spatial resolution.

    Returns a normalised (side, side) attention map, or None.
    """
    collected = []
    for sp in storing_processors.values():
        for ap in sp.attn_probs_list:
            if ap.shape[-2] == target_spatial:
                collected.append(ap.float())
    if not collected:
        return None

    all_maps = torch.stack(collected)          # (N, batch*heads, spatial, 77)
    avg_map = all_maps.mean(dim=(0, 1))        # (spatial, 77)

    token_sel = [i for i in keyword_indices if i < avg_map.shape[-1]]
    if token_sel:
        avg_map = avg_map[:, token_sel]

    A_flat = avg_map.max(dim=-1).values
    side = int(math.sqrt(target_spatial))
    if side * side != target_spatial:
        return None
    return A_flat.view(side, side)


def _aggregate_attention_at_resolution_batched(
    storing_processors: Dict,
    target_spatial: int,
    keyword_indices: List[int],
    batch_size: int,
) -> List[Optional[torch.Tensor]]:
    """Per-image cross-attention maps from a batched IP2P run.

    With batch_size B, the stored attention probs have shape
    (3·B·H, spatial, tokens) where 3 is for CFG copies and H is heads.
    We reshape and average over CFG/heads to get one map per image.
    """
    collected = []
    for sp in storing_processors.values():
        for ap in sp.attn_probs_list:
            if ap.shape[-2] == target_spatial:
                collected.append(ap.float())
    if not collected:
        return [None] * batch_size

    B = batch_size
    side = int(math.sqrt(target_spatial))
    if side * side != target_spatial:
        return [None] * batch_size

    all_maps = torch.stack(collected)  # (N, 3*B*H, spatial, tokens)
    N_maps, total_bh, spatial, tokens = all_maps.shape
    H = total_bh // (3 * B)
    if H < 1 or total_bh != 3 * B * H:
        return [None] * batch_size

    all_maps = all_maps.reshape(N_maps, 3, B, H, spatial, tokens)
    per_image = all_maps.mean(dim=(0, 1, 3))  # (B, spatial, tokens)

    results = []
    for b in range(B):
        avg_map = per_image[b]
        token_sel = [i for i in keyword_indices if i < avg_map.shape[-1]]
        if token_sel:
            avg_map = avg_map[:, token_sel]
        A_flat = avg_map.max(dim=-1).values
        results.append(A_flat.view(side, side))
    return results


def _compute_sa_propagation_leakage_batched(
    sa_storing_processors: Dict,
    roi_masks_2d: List[torch.Tensor],
    batch_size: int,
) -> List[float]:
    """Batched SA propagation leakage — one scalar per image."""
    B = batch_size
    PREFERRED = [64, 256]
    use_spatial: Optional[int] = None
    for sp_pref in PREFERRED:
        for sp in sa_storing_processors.values():
            for ap in sp.attn_probs_list:
                if ap.shape[-1] == sp_pref and ap.shape[-2] == sp_pref:
                    use_spatial = sp_pref
                    break
            if use_spatial is not None:
                break
        if use_spatial is not None:
            break
    if use_spatial is None:
        return [0.0] * B

    side = int(math.sqrt(use_spatial))
    masks_flat = []
    for mask_2d in roi_masks_2d:
        M = mask_2d.float().squeeze(0)
        M_small = torch.nn.functional.interpolate(
            M[None, None], size=(side, side), mode="bilinear",
        ).squeeze().cpu()
        masks_flat.append((M_small > 0.3).float().flatten())

    avg_sa = torch.zeros(B, use_spatial, use_spatial, dtype=torch.float32)
    count = 0
    for sp in sa_storing_processors.values():
        for ap in sp.attn_probs_list:
            if ap.shape[-1] == use_spatial and ap.shape[-2] == use_spatial:
                total_bh = ap.shape[0]
                H = total_bh // (3 * B)
                if total_bh != 3 * B * H or H < 1:
                    continue
                per_img = ap.float().reshape(3, B, H, use_spatial, use_spatial).mean(dim=(0, 2)).cpu()
                avg_sa += per_img
                count += 1
    if count == 0:
        return [0.0] * B
    avg_sa /= count

    results = []
    for b in range(B):
        M_flat = masks_flat[b]
        bg_flat = 1.0 - M_flat
        n_bg = float(bg_flat.sum())
        n_roi = float(M_flat.sum())
        if n_bg < 1 or n_roi < 1:
            results.append(0.0)
            continue
        roi_attn_col = avg_sa[b].mv(M_flat)
        results.append(float((bg_flat * roi_attn_col).sum() / n_bg))
    return results


def _run_ip2p_and_collect(
    rendered_rgb: torch.Tensor,
    ip2p_pipe,
    prompt: str,
    num_steps: int = 20,
    seed: Optional[int] = None,
    guidance_scale: float = 7.5,
    image_guidance_scale: float = 1.5,
    latency_logger: Optional[LatencyLogger] = None,
    device: str = "cuda",
) -> Tuple[Dict, List[int], Optional[torch.Tensor]]:
    """Run IP2P and return (storing_processors, keyword_indices, edited_image).

    The edited_image is a (3, H, W) tensor on CPU in [0, 1], or None if
    the pipeline returned no images.
    """
    from PIL import Image as PILImage
    from torchvision.transforms import ToPILImage, ToTensor

    rgb_pil = ToPILImage()(rendered_rgb.cpu().clamp(0, 1))
    rgb_pil = rgb_pil.resize((512, 512), PILImage.BICUBIC)

    with latency_timeit(latency_logger, "step2.ip2p.tokenize", device):
        keyword_indices = _tokenize_and_find_keyword_indices(ip2p_pipe, prompt)

    with latency_timeit(latency_logger, "step2.ip2p.hook_attn_processors", device):
        original_processors = {}
        storing_processors = {}      # cross-attention (attn2)
        sa_storing_processors = {}   # self-attention (attn1)
        for name, mod in ip2p_pipe.unet.named_modules():
            if hasattr(mod, "processor"):
                if name.endswith(".attn2"):
                    original_processors[name] = mod.processor
                    sp = _StoringAttnProcessor()
                    storing_processors[name] = sp
                    mod.set_processor(sp)
                elif name.endswith(".attn1"):
                    original_processors[name] = mod.processor
                    sp = _StoringSAOnlySmallProcessor(target_spatial=64)
                    sa_storing_processors[name] = sp
                    mod.set_processor(sp)

    # Optional deterministic generator so repeated calls with the same seed
    # produce identical attention maps (used to align CLI vs grid entropy).
    generator = None
    if seed is not None:
        exec_device = getattr(ip2p_pipe, "_execution_device", None)
        if exec_device is None:
            exec_device = ip2p_pipe.unet.device
        generator = torch.Generator(device=str(exec_device)).manual_seed(seed)

    with latency_timeit(latency_logger, "step2.ip2p.forward", device):
        with torch.no_grad():
            result = ip2p_pipe(
                prompt=prompt,
                image=rgb_pil,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                image_guidance_scale=image_guidance_scale,
                output_type="pil",
                generator=generator,
            )

    # Restore original processors
    with latency_timeit(latency_logger, "step2.ip2p.restore_attn_processors", device):
        for name, mod in ip2p_pipe.unet.named_modules():
            if name in original_processors:
                mod.set_processor(original_processors[name])

    # Extract edited image from pipeline output
    with latency_timeit(latency_logger, "step2.ip2p.extract_edited", device):
        edited_t = None
        if result and result.images:
            edited_t = ToTensor()(result.images[0])  # (3, 512, 512)

    return storing_processors, sa_storing_processors, keyword_indices, edited_t


def _extract_attention_map(
    rendered_rgb: torch.Tensor,
    ip2p_pipe,
    prompt: str,
    num_steps: int = 20,
    seed: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """
    Run a fast IP2P preview and extract a spatial cross-attention map.

    Improvements over naïve extraction:
      1. Accumulates attention across all denoising timesteps.
      2. Uses 16×16 resolution layers (best semantic focus).
      3. Selects only content text tokens (skips BOS/EOS/padding).

    Returns:
        (H, W) attention map tensor on CPU, or None if extraction failed.
    """
    storing_processors, _sa_procs, keyword_indices, _ = _run_ip2p_and_collect(
        rendered_rgb, ip2p_pipe, prompt, num_steps, seed=seed,
    )

    TARGET_SPATIAL = 256  # 16 × 16
    A = _aggregate_attention_at_resolution(storing_processors, TARGET_SPATIAL, keyword_indices)

    if A is None:
        # Fallback: try any available resolution
        for sp in storing_processors.values():
            for ap in sp.attn_probs_list:
                A = _aggregate_attention_at_resolution(
                    storing_processors, ap.shape[-2], keyword_indices,
                )
                if A is not None:
                    break
            if A is not None:
                break

    return A


def _extract_attention_all_resolutions(
    rendered_rgb: torch.Tensor,
    ip2p_pipe,
    prompt: str,
    num_steps: int = 20,
    seed: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Like _extract_attention_map but returns maps at *all* resolutions.

    Returns dict ``{"8x8": Tensor, "16x16": Tensor, ...}``.
    """
    storing_processors, _sa_procs, keyword_indices, _ = _run_ip2p_and_collect(
        rendered_rgb, ip2p_pipe, prompt, num_steps, seed=seed,
    )

    result: Dict[str, torch.Tensor] = {}
    for target_spatial, label in [(64, "8x8"), (256, "16x16"), (1024, "32x32"), (4096, "64x64")]:
        A = _aggregate_attention_at_resolution(storing_processors, target_spatial, keyword_indices)
        if A is not None:
            result[label] = A
    return result


@torch.no_grad()
def _run_ip2p_edit(
    rendered_rgb: torch.Tensor,
    ip2p_pipe,
    prompt: str,
    num_steps: int = 20,
    seed: Optional[int] = None,
    guidance_scale: float = 7.5,
    image_guidance_scale: float = 1.5,
) -> Optional[torch.Tensor]:
    """Run IP2P for real editing (no attention hook). Returns edited image (3, H, W) on CPU in [0, 1]."""
    from PIL import Image as PILImage
    from torchvision.transforms import ToPILImage, ToTensor

    rgb_pil = ToPILImage()(rendered_rgb.cpu().clamp(0, 1))
    rgb_pil = rgb_pil.resize((512, 512), PILImage.BICUBIC)

    generator = None
    if seed is not None:
        exec_device = getattr(ip2p_pipe, "_execution_device", None) or ip2p_pipe.unet.device
        generator = torch.Generator(device=str(exec_device)).manual_seed(seed)

    out = ip2p_pipe(
        prompt=prompt,
        image=rgb_pil,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        image_guidance_scale=image_guidance_scale,
        output_type="pil",
        generator=generator,
    )
    if not out or not out.images:
        return None
    # out.images is List[PIL], take first and convert to (3, H, W) tensor
    edited_pil = out.images[0]
    edited_t = ToTensor()(edited_pil)  # (3, 512, 512)
    return edited_t


def _run_ip2p_unified(
    rendered_rgb: torch.Tensor,
    roi_mask_2d: torch.Tensor,
    ip2p_pipe,
    prompt: str,
    lambda_leak: float = 1.5,
    lambda_ent: float = 2.0,
    entropy_thresh: float = 0.97,
    num_steps: int = 20,
    seed: Optional[int] = None,
    guidance_scale: float = 7.5,
    image_guidance_scale: float = 1.5,
    latency_logger: Optional[LatencyLogger] = None,
    device: str = "cuda",
) -> Tuple[float, Dict, Dict[str, torch.Tensor], Optional[torch.Tensor]]:
    """Run IP2P once and return SAGE score, all-resolution attention maps, and edited image.

    Consolidates what previously required 3 separate IP2P forward passes
    (_compute_editability_score + _extract_attention_all_resolutions + _run_ip2p_edit)
    into a single pass.

    Returns:
        (sage_score, sage_details, all_res_attn_dict, edited_tensor)
    """
    with latency_timeit(latency_logger, "step2.ip2p_collect", device):
        storing_processors, sa_storing_processors, keyword_indices, edited_t = _run_ip2p_and_collect(
            rendered_rgb,
            ip2p_pipe,
            prompt,
            num_steps,
            seed=seed,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            latency_logger=latency_logger,
            device=device,
        )

    # --- Extract 16×16 attention for SAGE scoring ---
    TARGET_SPATIAL = 256  # 16 × 16
    with latency_timeit(latency_logger, "step2.attn.aggregate_16x16", device):
        A_16 = _aggregate_attention_at_resolution(storing_processors, TARGET_SPATIAL, keyword_indices)

    if A_16 is None:
        with latency_timeit(latency_logger, "step2.attn.aggregate_fallback", device):
            # Fallback: try any available resolution
            for sp in storing_processors.values():
                for ap in sp.attn_probs_list:
                    A_16 = _aggregate_attention_at_resolution(
                        storing_processors, ap.shape[-2], keyword_indices,
                    )
                    if A_16 is not None:
                        break
                if A_16 is not None:
                    break

    # --- Compute SA propagation leakage ---
    with latency_timeit(latency_logger, "step2.sa_leakage", device):
        sa_leak = _compute_sa_propagation_leakage(
            sa_storing_processors, roi_mask_2d, target_spatial=TARGET_SPATIAL,
        )

    # --- Compute SAGE score ---
    with latency_timeit(latency_logger, "step2.sage_score", device):
        if A_16 is not None:
            sage_score, sage_details = _compute_sage_score(
                A_16, roi_mask_2d,
                lambda_leak=lambda_leak,
                lambda_ent=lambda_ent,
                entropy_thresh=entropy_thresh,
                sa_leakage=sa_leak,
            )
            sage_details["mode"] = "sage"
        else:
            M = roi_mask_2d.float()
            occupancy = float(M.mean())
            optimal_ratio = 0.30
            focus = max(0.0, 1.0 - abs(occupancy - optimal_ratio) / optimal_ratio)
            size_penalty = 0.0
            if occupancy < 0.10 or occupancy > 0.70:
                size_penalty = 10.0
            sage_score = focus - size_penalty
            sage_details = dict(
                focus=focus, leakage=0.0, sa_leakage=sa_leak, entropy=0.0,
                occupancy=occupancy, size_penalty=size_penalty,
                total_score=sage_score, mode="fallback",
            )

    # --- Extract attention maps at all resolutions (for grid visualisation) ---
    with latency_timeit(latency_logger, "step2.attn.aggregate_all_res", device):
        all_res: Dict[str, torch.Tensor] = {}
        for target_spatial, label in [(64, "8x8"), (256, "16x16"), (1024, "32x32"), (4096, "64x64")]:
            A = _aggregate_attention_at_resolution(storing_processors, target_spatial, keyword_indices)
            if A is not None:
                all_res[label] = A

    return sage_score, sage_details, all_res, edited_t


def _run_ip2p_unified_batch(
    rendered_rgbs: List[torch.Tensor],
    roi_masks_2d: List[torch.Tensor],
    ip2p_pipe,
    prompt: str,
    batch_size: int = 4,
    lambda_leak: float = 1.5,
    lambda_ent: float = 2.0,
    entropy_thresh: float = 0.97,
    num_steps: int = 20,
    seed: Optional[int] = None,
    guidance_scale: float = 7.5,
    image_guidance_scale: float = 1.5,
    latency_logger: Optional[LatencyLogger] = None,
    device: str = "cuda",
) -> List[Tuple[float, Dict, Dict[str, torch.Tensor], Optional[torch.Tensor]]]:
    """Batched IP2P: process multiple views per UNet forward pass.

    Groups inputs into mini-batches of *batch_size* and runs them through
    the IP2P pipeline in a single call, dramatically reducing per-view
    overhead (encoding, scheduling, etc.) and improving GPU utilisation.

    Returns a list (one entry per input image) of
        (sage_score, sage_details, all_res_attn_dict, edited_tensor).
    """
    from PIL import Image as PILImage
    from torchvision.transforms import ToPILImage, ToTensor

    N_total = len(rendered_rgbs)
    all_results: List[Optional[Tuple]] = [None] * N_total

    keyword_indices = _tokenize_and_find_keyword_indices(ip2p_pipe, prompt)

    for batch_start in range(0, N_total, batch_size):
        batch_end = min(batch_start + batch_size, N_total)
        B = batch_end - batch_start

        pil_images = []
        for idx in range(batch_start, batch_end):
            pil_img = ToPILImage()(rendered_rgbs[idx].cpu().clamp(0, 1))
            pil_img = pil_img.resize((512, 512), PILImage.BICUBIC)
            pil_images.append(pil_img)

        # Hook attention processors
        original_processors: Dict = {}
        storing_processors: Dict = {}
        sa_storing_processors: Dict = {}
        for name, mod in ip2p_pipe.unet.named_modules():
            if hasattr(mod, "processor"):
                if name.endswith(".attn2"):
                    original_processors[name] = mod.processor
                    sp = _StoringAttnProcessor()
                    storing_processors[name] = sp
                    mod.set_processor(sp)
                elif name.endswith(".attn1"):
                    original_processors[name] = mod.processor
                    sp = _StoringSAOnlySmallProcessor(target_spatial=64)
                    sa_storing_processors[name] = sp
                    mod.set_processor(sp)

        generator = None
        if seed is not None:
            exec_device = getattr(ip2p_pipe, "_execution_device", None) or ip2p_pipe.unet.device
            generator = torch.Generator(device=str(exec_device)).manual_seed(seed)

        with latency_timeit(latency_logger, f"step2.ip2p_batch_{batch_start}", device):
            with torch.no_grad():
                result = ip2p_pipe(
                    prompt=[prompt] * B,
                    image=pil_images,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    image_guidance_scale=image_guidance_scale,
                    output_type="pil",
                    generator=generator,
                )

        # Restore processors
        for name, mod in ip2p_pipe.unet.named_modules():
            if name in original_processors:
                mod.set_processor(original_processors[name])

        edited_tensors: List[Optional[torch.Tensor]] = []
        if result and result.images:
            for img in result.images:
                edited_tensors.append(ToTensor()(img))
        while len(edited_tensors) < B:
            edited_tensors.append(None)

        # Per-image CA maps at all resolutions
        per_image_all_res: List[Dict[str, torch.Tensor]] = [{} for _ in range(B)]
        for target_spatial, label in [(64, "8x8"), (256, "16x16"), (1024, "32x32"), (4096, "64x64")]:
            maps = _aggregate_attention_at_resolution_batched(
                storing_processors, target_spatial, keyword_indices, B,
            )
            for b in range(B):
                if maps[b] is not None:
                    per_image_all_res[b][label] = maps[b]

        A_16_list = _aggregate_attention_at_resolution_batched(
            storing_processors, 256, keyword_indices, B,
        )
        batch_masks = [roi_masks_2d[batch_start + b] for b in range(B)]
        sa_leak_list = _compute_sa_propagation_leakage_batched(
            sa_storing_processors, batch_masks, B,
        )

        for b in range(B):
            idx = batch_start + b
            A_16 = A_16_list[b]
            sa_leak = sa_leak_list[b]
            mask_2d = roi_masks_2d[idx]

            if A_16 is not None:
                sage_score, sage_details = _compute_sage_score(
                    A_16, mask_2d,
                    lambda_leak=lambda_leak,
                    lambda_ent=lambda_ent,
                    entropy_thresh=entropy_thresh,
                    sa_leakage=sa_leak,
                )
                sage_details["mode"] = "sage"
            else:
                M = mask_2d.float()
                occupancy = float(M.mean())
                optimal_ratio = 0.30
                focus = max(0.0, 1.0 - abs(occupancy - optimal_ratio) / optimal_ratio)
                size_penalty = 0.0
                if occupancy < 0.10 or occupancy > 0.70:
                    size_penalty = 10.0
                sage_score = focus - size_penalty
                sage_details = dict(
                    focus=focus, leakage=0.0, sa_leakage=sa_leak, entropy=0.0,
                    occupancy=occupancy, size_penalty=size_penalty,
                    total_score=sage_score, mode="fallback",
                )

            all_results[idx] = (sage_score, sage_details, per_image_all_res[b], edited_tensors[b])

    return all_results


def _compute_sage_score(
    A_spatial: torch.Tensor,
    roi_mask_2d: torch.Tensor,
    lambda_leak: float = 1.5,
    lambda_ent: float = 2.0,
    entropy_thresh: float = 0.97,
    occupancy_lo: float = 0.02,
    occupancy_hi: float = 0.70,
    size_penalty_val: float = 1.0,
    sa_leakage: float = 0.0,
    lambda_sa: float = 5.0,
) -> Tuple[float, Dict]:
    """
    SAGE v2 scoring with Contrast + Thresholded Precision-Recall + SA Leakage.

    Cross-attention shows *where the text directs edits*, but actual edit signal
    propagates through self-attention. SA leakage measures how much BG pixels
    attend to ROI pixels, predicting unwanted background changes.

    Computes:
      S_total = contrast * precision_f1 - λ₁·leakage_top90 - λ_sa·sa_leakage - Penalty_size

    Args:
        A_spatial: (H_a, W_a) raw cross-attention map (will be resized to mask).
        roi_mask_2d: (1, H, W) binary ROI mask.
        lambda_leak: weight for CA leakage penalty.
        lambda_ent: (unused, kept for API compatibility).
        entropy_thresh: (unused, kept for API compatibility).
        occupancy_lo: minimum acceptable occupancy ratio.
        occupancy_hi: maximum acceptable occupancy ratio.
        size_penalty_val: penalty value for out-of-range occupancy.
        sa_leakage: self-attention propagation leakage score (0-1).
        lambda_sa: weight for SA leakage penalty (default 2.0).

    Returns:
        (total_score, details_dict) where details_dict contains individual metrics.
    """
    M = roi_mask_2d.float().squeeze(0)  # (H, W)
    h_m, w_m = M.shape

    # Resize attention map to mask resolution
    A_resized = (
        torch.nn.functional.interpolate(
            A_spatial[None, None].float(), size=(h_m, w_m), mode="bilinear"
        )
        .squeeze()
    )
    A = A_resized.to(M.device)
    # Normalize to [0, 1]
    A = (A - A.min()) / (A.max() - A.min() + 1e-8)

    bg = 1.0 - M

    # --- 1. Contrast Ratio ---
    # Measures how much stronger attention is inside ROI vs outside.
    # +1 = all attention on ROI, 0 = uniform, -1 = all on background
    roi_mean = float((A * M).sum() / (M.sum() + 1e-8))
    bg_mean = float((A * bg).sum() / (bg.sum() + 1e-8))
    contrast = (roi_mean - bg_mean) / (roi_mean + bg_mean + 1e-8)

    # --- 2. Thresholded Precision-Recall (F1) ---
    # Binarize attention at top-25% to find "strongly edited" regions,
    # then compute F1 overlap with ROI mask.
    A_flat = A.flatten()
    threshold = float(A_flat.quantile(0.75))
    A_thresh = (A >= threshold).float()

    precision = float((A_thresh * M).sum() / (A_thresh.sum() + 1e-8))
    recall = float((A_thresh * M).sum() / (M.sum() + 1e-8))
    precision_f1 = 2.0 * precision * recall / (precision + recall + 1e-8)

    # --- 3. Strong Leakage (top-90th percentile on background) ---
    # Instead of mean background attention, capture the strongest leak.
    bg_attention = A[bg > 0.5]
    if bg_attention.numel() > 0:
        leakage_top90 = float(bg_attention.quantile(0.9))
    else:
        leakage_top90 = 0.0

    # --- 4. Occupancy Penalty ---
    occupancy = float(M.mean())
    size_penalty = 0.0
    if occupancy < occupancy_lo or occupancy > occupancy_hi:
        size_penalty = size_penalty_val

    # --- Legacy metrics for logging compatibility ---
    focus = float((A * M).sum() / (A.sum() + 1e-8))

    # --- Final Score ---
    total_score = (
        contrast * precision_f1
        - (lambda_leak * leakage_top90)
        - (lambda_sa * sa_leakage)
        - size_penalty
    )

    details = dict(
        focus=focus,
        contrast=contrast,
        precision_f1=precision_f1,
        leakage=leakage_top90,
        sa_leakage=sa_leakage,
        entropy=0.0,  # kept for API compatibility
        occupancy=occupancy,
        size_penalty=size_penalty,
        total_score=total_score,
    )
    return total_score, details


def _compute_editability_score(
    rendered_rgb: torch.Tensor,
    roi_mask_2d: torch.Tensor,
    ip2p_pipe,
    prompt: str,
    lambda_leak: float = 1.5,
    lambda_ent: float = 2.0,
    entropy_thresh: float = 0.97,
    num_steps: int = 20,
    seed: Optional[int] = None,
) -> Tuple[float, Dict]:
    """
    SAGE-Probing editability score.

    With IP2P: extracts cross-attention, computes SAGE score
      S = Focus - λ₁·Leakage - λ₂·Entropy - Penalty_size
    Without IP2P: falls back to occupancy-based heuristic.

    Returns:
        (score, details_dict)
    """
    M = roi_mask_2d.float()  # (1, H, W)
    occupancy = float(M.mean())

    if ip2p_pipe is None:
        # Heuristic fallback: prefer occupancy in [0.15, 0.50] range
        optimal_ratio = 0.30
        focus = max(0.0, 1.0 - abs(occupancy - optimal_ratio) / optimal_ratio)
        size_penalty = 0.0
        if occupancy < 0.10 or occupancy > 0.70:
            size_penalty = 10.0
        score = focus - size_penalty
        details = dict(
            focus=focus, leakage=0.0, sa_leakage=0.0, precision_f1=0.0,
            contrast=0.0, entropy=0.0,
            occupancy=occupancy, size_penalty=size_penalty,
            total_score=score, mode="heuristic",
        )
        return score, details

    # --- IP2P attention path ---
    A_spatial = _extract_attention_map(
        rendered_rgb, ip2p_pipe, prompt, num_steps, seed=seed
    )

    if A_spatial is None:
        # Fallback if attention extraction failed
        optimal_ratio = 0.30
        focus = max(0.0, 1.0 - abs(occupancy - optimal_ratio) / optimal_ratio)
        size_penalty = 0.0
        if occupancy < 0.10 or occupancy > 0.70:
            size_penalty = 10.0
        score = focus - size_penalty
        details = dict(
            focus=focus, leakage=0.0, entropy=0.0,
            occupancy=occupancy, size_penalty=size_penalty,
            total_score=score, mode="fallback",
        )
        return score, details

    score, details = _compute_sage_score(
        A_spatial, roi_mask_2d,
        lambda_leak=lambda_leak,
        lambda_ent=lambda_ent,
        entropy_thresh=entropy_thresh,
    )
    details["mode"] = "sage"
    return score, details


def _save_attention_grid(
    grid_data: Dict[
        float,
        Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor]],
    ],
    save_path: str,
    res_labels: Optional[List[str]] = None,
    roi_masks: Optional[Dict[float, torch.Tensor]] = None,
    sa_leakage_scores: Optional[Dict[float, float]] = None,
) -> None:
    """Save a distance×resolution attention map grid as an image.

    Args:
        grid_data: ``{multiplier: (rgb_cpu_3HW, {"8x8": A, ...}, edited_cpu or None)}``
        save_path: output image path.
        res_labels: resolution labels to include (default all four).
        roi_masks: ``{multiplier: (1, H, W) binary mask}`` for contrast/pF1 overlay.
        sa_leakage_scores: ``{multiplier: float}`` SA propagation leakage per view.
    """
    from PIL import Image as PILImage, ImageDraw
    from torchvision.transforms import ToPILImage

    if res_labels is None:
        res_labels = ["8x8", "16x16", "32x32", "64x64"]

    multipliers = sorted(grid_data.keys())
    cell = 256
    label_w = 100
    header_h = 30
    has_edited = any(
        len(grid_data[mult]) > 2 and grid_data[mult][2] is not None
        for mult in multipliers
    )
    n_rows = len(res_labels) + 1 + (1 if has_edited else 0)  # +1 render, +1 optional edited

    canvas = PILImage.new(
        "RGB",
        (label_w + len(multipliers) * cell, header_h + n_rows * cell),
        (0, 0, 0),
    )
    draw = ImageDraw.Draw(canvas)

    # Column headers
    for ci, mult in enumerate(multipliers):
        x = label_w + ci * cell + cell // 2 - 20
        draw.text((x, 5), f"{mult:.1f}x", fill="white")

    # Row 0: rendered images
    draw.text((5, header_h + cell // 2 - 8), "Render", fill="white")
    for ci, mult in enumerate(multipliers):
        rgb_t, _ = grid_data[mult][0], grid_data[mult][1]
        rgb_rs = torch.nn.functional.interpolate(
            rgb_t.unsqueeze(0), size=(cell, cell), mode="bilinear",
        ).squeeze(0)
        canvas.paste(ToPILImage()(rgb_rs.clamp(0, 1)), (label_w + ci * cell, header_h))
        # SA leakage label on render row
        if sa_leakage_scores is not None and mult in sa_leakage_scores:
            sa_val = sa_leakage_scores[mult]
            ImageDraw.Draw(canvas).text(
                (label_w + ci * cell + 5, header_h + 5),
                f"SA={sa_val:.3f}",
                fill="yellow",
            )

    # Attention rows
    for ri, res_label in enumerate(res_labels):
        y = header_h + (ri + 1) * cell
        draw.text((5, y + cell // 2 - 8), res_label, fill="white")

        for ci, mult in enumerate(multipliers):
            _, attn_dict = grid_data[mult][0], grid_data[mult][1]
            A = attn_dict.get(res_label)
            if A is None:
                continue
            A_norm = (A - A.min()) / (A.max() - A.min() + 1e-8)
            A_vis = torch.nn.functional.interpolate(
                A_norm[None, None].float(), size=(cell, cell), mode="bilinear",
            ).squeeze()
            heatmap = torch.stack([A_vis, A_vis * 0.5, torch.zeros_like(A_vis)], dim=0)
            canvas.paste(ToPILImage()(heatmap.clamp(0, 1)), (label_w + ci * cell, y))

            # Compute per-cell metrics for overlay
            raw_ent = compute_entropy(A_norm)
            max_ent = math.log(A.numel()) + 1e-8
            label_text = f"H={raw_ent / max_ent:.3f}"

            # If ROI mask available, compute contrast and pF1
            if roi_masks is not None and mult in roi_masks:
                M_cell = roi_masks[mult].float().squeeze(0)
                h_m, w_m = M_cell.shape
                A_rs = torch.nn.functional.interpolate(
                    A_norm[None, None].float(), size=(h_m, w_m), mode="bilinear",
                ).squeeze()
                bg_cell = 1.0 - M_cell
                roi_m = float((A_rs * M_cell).sum() / (M_cell.sum() + 1e-8))
                bg_m = float((A_rs * bg_cell).sum() / (bg_cell.sum() + 1e-8))
                ctr = (roi_m - bg_m) / (roi_m + bg_m + 1e-8)
                # Thresholded precision-recall F1
                thr = float(A_rs.flatten().quantile(0.75))
                A_thr = (A_rs >= thr).float()
                prec = float((A_thr * M_cell).sum() / (A_thr.sum() + 1e-8))
                rec = float((A_thr * M_cell).sum() / (M_cell.sum() + 1e-8))
                pf1 = 2.0 * prec * rec / (prec + rec + 1e-8)
                label_text = f"C={ctr:.2f} F1={pf1:.2f}"

            ImageDraw.Draw(canvas).text(
                (label_w + ci * cell + 5, y + 5),
                label_text,
                fill="white",
            )

    # Last row: IP2P edited images
    if has_edited:
        y_edit = header_h + (len(res_labels) + 1) * cell
        draw.text((5, y_edit + cell // 2 - 8), "Edited", fill="white")
        for ci, mult in enumerate(multipliers):
            tup = grid_data[mult]
            if len(tup) > 2 and tup[2] is not None:
                edited_t = tup[2]
                edited_rs = torch.nn.functional.interpolate(
                    edited_t.unsqueeze(0), size=(cell, cell), mode="bilinear",
                ).squeeze(0)
                canvas.paste(
                    ToPILImage()(edited_rs.clamp(0, 1)),
                    (label_w + ci * cell, int(y_edit)),
                )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    canvas.save(save_path)
    print(f"[Step2] Attention grid saved to {save_path}")


@torch.no_grad()
def scale_probing(
    gaussians: GaussianModel,
    roi_info: Dict,
    pipe_params,
    background: torch.Tensor,
    fovy: float,
    h: int,
    w: int,
    roi_mask: Optional[torch.Tensor],
    ip2p_pipe=None,
    edit_prompt: str = "",
    distance_multipliers: Optional[List[float]] = None,
    lambda_leak: float = 1.5,
    lambda_ent: float = 2.0,
    entropy_thresh: float = 0.97,
    override_opacity: Optional[torch.Tensor] = None,
    save_attn_grid: Optional[str] = None,
    device: str = "cuda",
    ip2p_guidance_scale: float = 7.5,
    ip2p_image_guidance_scale: float = 1.5,
    ip2p_num_inference_steps: int = 20,
    ip2p_batch_size: int = 1,
    latency_logger: Optional[LatencyLogger] = None,
) -> float:
    """
    SAGE-Probing: find the optimal viewing distance d* by rendering pilot
    views at several distances along v_front and scoring each with the
    Sharpness-Aware Guided Editability metric.

    Score = Contrast · PrecisionF1 - λ₁·Leakage_top90 - Penalty_size

    When ip2p_batch_size > 1, multiple views are processed in a single
    IP2P forward pass for significantly faster execution.

    Returns the optimal distance d*.
    """
    if distance_multipliers is None:
        distance_multipliers = [1.5, 2.0, 2.5, 3.0, 3.5]

    center = roi_info["center"]
    v_front = roi_info["v_front"]
    eigenvalues = roi_info["eigenvalues"]
    r_obj = float(np.sqrt(np.abs(eigenvalues[0])))
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    print(f"[Step2] r_obj (sqrt(lambda_max)) = {r_obj:.4f}")
    print(f"[Step2] Candidate distances: {[f'{m}x r_obj = {m * r_obj:.4f}' for m in distance_multipliers]}")

    best_d = distance_multipliers[1] * r_obj if len(distance_multipliers) > 1 else distance_multipliers[0] * r_obj
    best_score = -float("inf")
    results = []

    attn_grid_data: Dict[float, Tuple[torch.Tensor, Dict[str, torch.Tensor]]] = {}
    attn_grid_masks: Dict[float, torch.Tensor] = {}
    attn_grid_sa_scores: Dict[float, float] = {}

    # --- Phase 1: Pre-render all views and ROI masks ---
    rendered_rgbs: List[torch.Tensor] = []
    rendered_masks: List[torch.Tensor] = []
    for i, mult in enumerate(distance_multipliers):
        cand_prefix = f"step2.cand_{i}"
        d = mult * r_obj
        eye = center + v_front * d
        with latency_timeit(latency_logger, f"{cand_prefix}.make_camera", device):
            cam = _make_camera(eye, center, world_up, fovy, h, w, uid=i, device=device)
        with latency_timeit(latency_logger, f"{cand_prefix}.render", device):
            out = render(cam, gaussians, pipe_params, background, override_opacity=override_opacity)
            rgb = out["render"]
        if roi_mask is not None:
            with latency_timeit(latency_logger, f"{cand_prefix}.project_roi_mask", device):
                mask_2d = _project_roi_mask(
                    gaussians, roi_mask, cam, pipe_params, background,
                    override_opacity=override_opacity, device=device,
                )
        else:
            mask_2d = torch.ones(1, h, w, device=device)
        rendered_rgbs.append(rgb)
        rendered_masks.append(mask_2d)

    per_view_seed = 5

    # --- Phase 2: IP2P scoring ---
    if ip2p_pipe is not None and ip2p_batch_size > 1:
        print(f"[Step2] Batched IP2P (batch_size={ip2p_batch_size}, {len(distance_multipliers)} views)")
        with latency_timeit(latency_logger, "step2.ip2p_batch", device):
            batch_results = _run_ip2p_unified_batch(
                rendered_rgbs, rendered_masks, ip2p_pipe, edit_prompt,
                batch_size=ip2p_batch_size,
                lambda_leak=lambda_leak, lambda_ent=lambda_ent,
                entropy_thresh=entropy_thresh,
                num_steps=ip2p_num_inference_steps, seed=per_view_seed,
                guidance_scale=ip2p_guidance_scale,
                image_guidance_scale=ip2p_image_guidance_scale,
                latency_logger=latency_logger, device=device,
            )
        for i, mult in enumerate(distance_multipliers):
            d = mult * r_obj
            score, details, all_res, edited_t = batch_results[i]
            results.append((mult, d, score, details))
            if save_attn_grid:
                attn_grid_data[mult] = (rendered_rgbs[i].cpu(), all_res, edited_t)
                attn_grid_masks[mult] = rendered_masks[i].cpu()
                if 'sa_leakage' in details:
                    attn_grid_sa_scores[mult] = details['sa_leakage']
    else:
        for i, mult in enumerate(distance_multipliers):
            d = mult * r_obj
            rgb, mask_2d = rendered_rgbs[i], rendered_masks[i]
            if ip2p_pipe is not None:
                cand_prefix = f"step2.cand_{i}"
                with latency_timeit(latency_logger, f"{cand_prefix}.ip2p_unified", device):
                    score, details, all_res, edited_t = _run_ip2p_unified(
                        rgb, mask_2d, ip2p_pipe, edit_prompt,
                        lambda_leak=lambda_leak, lambda_ent=lambda_ent,
                        entropy_thresh=entropy_thresh,
                        num_steps=ip2p_num_inference_steps, seed=per_view_seed,
                        guidance_scale=ip2p_guidance_scale,
                        image_guidance_scale=ip2p_image_guidance_scale,
                        latency_logger=latency_logger, device=device,
                    )
                if save_attn_grid:
                    attn_grid_data[mult] = (rgb.cpu(), all_res, edited_t)
                    attn_grid_masks[mult] = mask_2d.cpu()
                    if 'sa_leakage' in details:
                        attn_grid_sa_scores[mult] = details['sa_leakage']
            else:
                score, details = _compute_editability_score(
                    rgb, mask_2d, None, edit_prompt,
                    lambda_leak=lambda_leak, lambda_ent=lambda_ent,
                    entropy_thresh=entropy_thresh, seed=per_view_seed,
                )
            results.append((mult, d, score, details))

    # --- Phase 3: Two-phase SAGE selection ---
    # Phase 3a: SA Containment Filter
    #   Compute mean(sa_leak) across all candidates; keep only those with sa_leak <= mean.
    #   This eliminates views where self-attention spreads the edit signal to the background.
    # Phase 3b: Attention Quality Ranking
    #   Among filtered candidates, pick the one with highest F1 * (1 - ca_leak).
    #   F1 measures how well the high-attention region covers the ROI.
    #   (1 - ca_leak) penalises strong background cross-attention activation.

    sa_vals = [
        details.get("sa_leakage", 0.0) for _, _, _, details in results
    ]
    sa_threshold = sum(sa_vals) / len(sa_vals) if sa_vals else 0.0

    for mult, d, score, details in results:
        contrast_str = f"  contrast={details['contrast']:.3f}" if 'contrast' in details else ""
        pf1_str = f"  pF1={details['precision_f1']:.3f}" if 'precision_f1' in details else ""
        sa_str = f"  sa_leak={details['sa_leakage']:.3f}" if 'sa_leakage' in details else ""
        ca_str = f"  ca_leak={details['leakage']:.3f}" if 'leakage' in details else ""
        new_score = details.get("precision_f1", 0.0) * (1.0 - details.get("leakage", 0.0))
        passed = details.get("sa_leakage", 0.0) <= sa_threshold
        print(
            f"[Step2]  d={d:.4f} ({mult}x) | "
            f"focus={details['focus']:.3f}"
            f"{contrast_str}{pf1_str}{sa_str}{ca_str}"
            f"  occ={details['occupancy']:.3f} | "
            f"F1*(1-ca)={new_score:.4f}"
            f"{'  [pass]' if passed else '  [filtered]'}"
        )

    filtered = [
        (mult, d, details)
        for mult, d, _, details in results
        if details.get("sa_leakage", 0.0) <= sa_threshold
    ]
    if not filtered:
        filtered = [(mult, d, details) for mult, d, _, details in results]

    def _sage_quality(details: Dict) -> float:
        return details.get("precision_f1", 0.0) * (1.0 - details.get("leakage", 0.0))

    best_mult, best_d, best_details = max(filtered, key=lambda x: _sage_quality(x[2]))
    best_score = _sage_quality(best_details)

    print(
        f"[Step2] SA filter: mean(sa)={sa_threshold:.3f}, "
        f"{len(filtered)}/{len(results)} views passed"
    )
    print(
        f"[Step2] Best distance d*={best_d:.4f} "
        f"(mult={best_d / r_obj:.2f}x, "
        f"F1*(1-ca)={best_score:.4f})"
    )

    if save_attn_grid and attn_grid_data:
        _save_attention_grid(
            attn_grid_data, save_attn_grid,
            roi_masks=attn_grid_masks,
            sa_leakage_scores=attn_grid_sa_scores,
        )

    return best_d


@torch.no_grad()
def _project_roi_mask(
    gaussians: GaussianModel,
    roi_mask_3d: torch.Tensor,
    cam: Simple_Camera,
    pipe_params,
    background: torch.Tensor,
    override_opacity: Optional[torch.Tensor] = None,
    device: str = "cuda",
) -> torch.Tensor:
    """Render ROI Gaussians as a solid-colour overlay to get a 2D projection mask."""
    N = gaussians.get_xyz.shape[0]
    roi_mask_3d = roi_mask_3d.to(gaussians.get_xyz.device).bool()

    # Render with override_color: ROI = white, rest = black
    colors = torch.zeros(N, 3, device=device)
    colors[roi_mask_3d] = 1.0

    bg_black = torch.zeros(3, device=device)
    out = render(cam, gaussians, pipe_params, bg_black, override_color=colors, override_opacity=override_opacity)
    mask_rgb = out["render"]  # (3, H, W)
    mask_2d = mask_rgb.mean(dim=0, keepdim=True)  # (1, H, W)
    mask_2d = (mask_2d > 0.1).float()
    return mask_2d


# ===================================================================
# Step 3: Fibonacci Manifold Sampling
# ===================================================================

def fibonacci_sphere_samples(n: int) -> np.ndarray:
    """
    Generate *n* nearly-uniform points on the unit sphere using the
    Fibonacci / golden-angle lattice.  Returns (n, 3).
    """
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    points = []
    for i in range(n):
        z = 1.0 - (2.0 * i) / (n - 1) if n > 1 else 0.0
        radius = math.sqrt(max(0.0, 1.0 - z * z))
        theta = golden_angle * i
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        points.append([x, y, z])
    return np.array(points, dtype=np.float32)


def fibonacci_camera_candidates(
    center: np.ndarray,
    distance: float,
    n_candidates: int = 150,
    world_up: np.ndarray = None,
    fovy: float = 1.0,
    h: int = 512,
    w: int = 512,
    hemisphere_only: bool = False,
    colmap_cam_centers: np.ndarray = None,
    cone_half_angle_deg: float = 90.0,
    cone_axis_direction: Optional[np.ndarray] = None,
    device: str = "cuda",
) -> List[Simple_Camera]:
    """
    Place *n_candidates* cameras on a sphere of radius *distance* centred
    on *center*, each looking at the centre.

    If cone filtering is used (via *cone_axis_direction* or *colmap_cam_centers*),
    restrict candidates to a cone. Prefer *cone_axis_direction* (e.g. v_front)
    so Step 3 aligns with the chosen front; else use mean COLMAP direction.
    """
    if world_up is None:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    directions = fibonacci_sphere_samples(n_candidates)

    if hemisphere_only:
        # Keep only directions with positive y (above ground)
        up_axis = _normalize(world_up)
        dots = directions @ up_axis
        directions = directions[dots > -0.1]  # allow slight below-horizon

    # Constrain to cone: use v_front (cone_axis_direction) when provided so
    # left/right bias follows chosen front; else fall back to COLMAP mean direction.
    if cone_axis_direction is not None:
        cone_axis = _normalize(np.asarray(cone_axis_direction, dtype=np.float32))
        cos_threshold = math.cos(math.radians(cone_half_angle_deg))
        dots = directions @ cone_axis
        directions = directions[dots > cos_threshold]
        print(f"[Step3] Cone filter: {len(directions)} candidates within "
              f"{cone_half_angle_deg}° of v_front")
    elif colmap_cam_centers is not None:
        # Mean direction from center → COLMAP cameras (legacy: can bias left/right)
        mean_cam_dir = _normalize(
            (colmap_cam_centers - center[None, :]).mean(axis=0)
        )
        cos_threshold = math.cos(math.radians(cone_half_angle_deg))
        dots = directions @ mean_cam_dir
        directions = directions[dots > cos_threshold]
        print(f"[Step3] Cone filter: {len(directions)} candidates within "
              f"{cone_half_angle_deg}° of COLMAP mean direction")

    cameras = []
    for i, d in enumerate(directions):
        eye = center + d * distance
        cam = _make_camera(eye, center, world_up, fovy, h, w, uid=i, device=device)
        cameras.append(cam)

    print(f"[Step3] Generated {len(cameras)} Fibonacci candidates (d={distance:.4f})")
    return cameras


# ===================================================================
# Step 4: Energy-based Candidate Scoring
# ===================================================================

@torch.no_grad()
def score_candidates(
    cameras: List[Simple_Camera],
    gaussians: GaussianModel,
    roi_mask: Optional[torch.Tensor],
    roi_info: Dict,
    pipe_params,
    background: torch.Tensor,
    w_vis: float = 0.6,
    w_can: float = 0.4,
    override_opacity: Optional[torch.Tensor] = None,
    device: str = "cuda",
    latency_logger: Optional[LatencyLogger] = None,
) -> List[Tuple[int, float, float, float]]:
    """
    Score each candidate camera.
    Returns list of (index, total_energy, S_vis, S_can).
    """
    center = roi_info["center"]
    v_front = roi_info["v_front"]
    v1 = roi_info["v1"]  # longest axis
    v2 = roi_info["v2"]  # second axis (side)

    results = []
    for idx, cam in enumerate(cameras):
        with latency_timeit(latency_logger, "step4.score_candidates.render", device):
            out = render(cam, gaussians, pipe_params, background, override_opacity=override_opacity)
            depth = out["depth_3dgs"]  # (1, H, W)

        # --- Visibility Score ---
        if roi_mask is not None:
            with latency_timeit(latency_logger, "step4.score_candidates.project_roi_mask", device):
                mask_2d = _project_roi_mask(
                    gaussians,
                    roi_mask,
                    cam,
                    pipe_params,
                    background,
                    override_opacity=override_opacity,
                    device=device,
                )
            with latency_timeit(latency_logger, "step4.score_candidates.visibility_score", device):
                mask_area = float(mask_2d.sum())
                total_area = float(mask_2d.numel())
                S_vis = mask_area / (total_area + 1e-8)
        else:
            S_vis = 1.0

        # --- Canonical Score ---
        with latency_timeit(latency_logger, "step4.score_candidates.canonical_score", device):
            cam_center = cam.camera_center.cpu().numpy()  # (3,)
            view_dir = _normalize(center - cam_center)

            # Reward alignment with v_front or side axis v2
            cos_front = abs(float(np.dot(view_dir, v_front)))
            cos_side = abs(float(np.dot(view_dir, v2)))
            S_can = max(cos_front, cos_side * 0.8)

        with latency_timeit(latency_logger, "step4.score_candidates.combine_energy", device):
            energy = w_vis * S_vis + w_can * S_can
        results.append((idx, energy, S_vis, S_can))

        if (idx + 1) % 50 == 0:
            print(f"[Step4] Scored {idx + 1}/{len(cameras)} candidates")

    results.sort(key=lambda x: x[1], reverse=True)
    print(
        f"[Step4] Top-5 energies: "
        f"{[(r[0], f'{r[1]:.4f}') for r in results[:5]]}"
    )
    return results


# ===================================================================
# Step 5: Diversity-aware Final Selection  (FPS)
# ===================================================================

def diversity_selection(
    cameras: List[Simple_Camera],
    scored: List[Tuple[int, float, float, float]],
    center: np.ndarray,
    n_select: int = 20,
    top_fraction: float = 0.20,
    diversity_x_weight: float = 0.0,
    diversity_y_variance_weight: float = 0.0,
    latency_logger: Optional[LatencyLogger] = None,
    device: str = "cuda",
) -> List[int]:
    """
    Select *n_select* cameras from the top-scoring pool using
    Energy-weighted FPS. diversity_x_weight adds azimuth separation bonus;
    diversity_y_variance_weight > 0 penalizes elevation spread (lower variance in y).
    """
    with latency_timeit(latency_logger, "step5.diversity_selection.prepare_pool", device):
        n_pool = max(int(len(scored) * top_fraction), n_select)
        pool = scored[:n_pool]
        pool_indices = [s[0] for s in pool]
        pool_energies = {s[0]: s[1] for s in pool}

    with latency_timeit(latency_logger, "step5.diversity_selection.view_dirs", device):
        view_dirs = {}
        azimuths = {}
        for ci in pool_indices:
            cc = cameras[ci].camera_center.cpu().numpy()
            v = _normalize(cc - center)
            view_dirs[ci] = v
            vx, vz = float(v[0]), float(v[2])
            azimuths[ci] = math.atan2(vx, vz)

    # Normalise pool energies to [0, 1] so the diversity term and energy term
    # are on comparable scales and diversity_x_weight actually works.
    energy_vals = list(pool_energies.values())
    min_energy = min(energy_vals)
    max_energy = max(energy_vals)
    energy_range = max_energy - min_energy + 1e-8
    pool_energies_norm = {ci: (e - min_energy) / energy_range for ci, e in pool_energies.items()}

    selected = []
    remaining = set(pool_indices)
    first = pool_indices[0]
    selected.append(first)
    remaining.discard(first)

    with latency_timeit(latency_logger, "step5.diversity_selection.fps_loop", device):
        while len(selected) < n_select and remaining:
            best_idx, best_score = None, -1e9
            for ci in remaining:
                min_ang = 1e9
                min_dphi = 1e9
                min_dy = 1e9
                for si in selected:
                    cos_sim = float(np.dot(view_dirs[ci], view_dirs[si]))
                    ang = math.acos(np.clip(cos_sim, -1.0, 1.0))
                    min_ang = min(min_ang, ang)
                    phi_ci, phi_si = azimuths[ci], azimuths[si]
                    dphi = abs(phi_ci - phi_si)
                    if dphi > math.pi:
                        dphi = 2 * math.pi - dphi
                    min_dphi = min(min_dphi, dphi)
                    dy = abs(float(view_dirs[ci][1]) - float(view_dirs[si][1]))
                    min_dy = min(min_dy, dy)
                diversity_score = min_ang + diversity_x_weight * min_dphi - diversity_y_variance_weight * min_dy
                # Additive combination: diversity and energy are independent terms so
                # diversity_x_weight is not suppressed by low-energy cameras on the far side.
                combined = diversity_score + pool_energies_norm[ci]
                if combined > best_score:
                    best_score = combined
                    best_idx = ci
            if best_idx is not None:
                selected.append(best_idx)
                remaining.discard(best_idx)
            else:
                break

    print(f"[Step5] Selected {len(selected)} diverse views from pool of {n_pool}")
    return selected


# ===================================================================
# ROI Mask via LangSAM  (optional)
# ===================================================================

@torch.no_grad()
def compute_roi_mask_from_segmentation(
    gaussians: GaussianModel,
    prompt: str,
    pipe_params,
    background: torch.Tensor,
    cam_centers: np.ndarray,
    cam_forwards: np.ndarray,
    fovy: float,
    h: int,
    w: int,
    n_views: int = 8,
    threshold: float = 0.3,
    override_opacity: Optional[torch.Tensor] = None,
    device: str = "cuda",
    latency_logger: Optional[LatencyLogger] = None,
) -> torch.Tensor:
    """
    Render a few views, run LangSAM segmentation, and back-project
    to create a per-Gaussian ROI mask.
    """
    from threestudio.utils.sam import LangSAMTextSegmentor

    # Segmentor init
    with latency_timeit(latency_logger, "step0.seg_init", device):
        segmentor = LangSAMTextSegmentor()
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    N_gauss = gaussians.get_xyz.shape[0]
    accum = torch.zeros(N_gauss, device=device)
    count = torch.zeros(N_gauss, device=device)

    # Pick up to n_views actual COLMAP cameras (their real positions & view directions)
    num_cams = cam_centers.shape[0]
    if num_cams == 0:
        print("[ROI] WARNING: No COLMAP cameras available for segmentation.")
        return None

    if num_cams <= n_views:
        indices = list(range(num_cams))
    else:
        # Evenly sample n_views indices across the COLMAP trajectory
        indices = np.linspace(0, num_cams - 1, n_views, dtype=int).tolist()

    for j, cam_idx in enumerate(indices):
        view_prefix = "step0.views"

        # Camera + render
        with latency_timeit(latency_logger, f"{view_prefix}.render", device):
            eye = cam_centers[cam_idx].astype(np.float32)
            fwd = _normalize(cam_forwards[cam_idx].astype(np.float32))
            target = eye + fwd

            cam = _make_camera(eye, target, world_up, fovy, h, w, uid=1000 + j, device=device)
            out = render(cam, gaussians, pipe_params, background, override_opacity=override_opacity)
            rgb = out["render"]  # (3, H, W)

        # LangSAM inference
        with latency_timeit(latency_logger, f"{view_prefix}.sam", device):
            # LangSAM expects (B, H, W, C)
            img_bhwc = rgb.permute(1, 2, 0).unsqueeze(0)  # (1, H, W, 3)
            mask_2d = segmentor(img_bhwc, prompt)  # (1, 1, H, W)
            mask_2d = mask_2d.squeeze(0)  # (1, H, W)

        if mask_2d.sum() < 10:
            continue

        # Back-project + accumulate
        with latency_timeit(latency_logger, f"{view_prefix}.backproject", device):
            # Back-project: project each Gaussian's 3D centre to 2D pixel
            # coordinates and look up the mask value at that location.
            xyz = gaussians.get_xyz.detach()  # (N, 3)
            R_c2w = torch.tensor(cam.R, device=xyz.device, dtype=torch.float32)
            T_vec = torch.tensor(cam.T, device=xyz.device, dtype=torch.float32)

            # W2C transform: x_cam = R_c2w^T @ x_world + T
            xyz_cam = xyz @ R_c2w + T_vec[None, :]

            z = xyz_cam[:, 2]
            valid = z > 0.01

            fx = float(fov2focal(cam.FoVx, w))
            fy_val = float(fov2focal(cam.FoVy, h))

            px = xyz_cam[:, 0] * fx / z + w * 0.5
            py = xyz_cam[:, 1] * fy_val / z + h * 0.5

            in_bounds = valid & (px >= 0) & (px < w) & (py >= 0) & (py < h)
            ib_idx = torch.where(in_bounds)[0]

            mask_hw = mask_2d.squeeze().to(xyz.device).float()  # (H, W)
            px_ib = px[ib_idx].long().clamp(0, w - 1)
            py_ib = py[ib_idx].long().clamp(0, h - 1)
            mask_vals = mask_hw[py_ib, px_ib]

            accum[ib_idx] += mask_vals
            count[ib_idx] += 1.0

    # Normalise
    count = count.clamp(min=1.0)
    scores = accum / count
    roi_mask = scores > threshold

    n_roi = int(roi_mask.sum().item())
    print(f"[ROI] Segmented {n_roi}/{N_gauss} Gaussians as ROI (prompt='{prompt}')")
    if n_roi == 0:
        print(f"[ROI] WARNING: No Gaussians matched prompt '{prompt}'. Falling back to all Gaussians.")
        return None
    return roi_mask


# ===================================================================
# Reusable pipeline for gs_load / threestudio
# ===================================================================

def run_generate_by_lens_pipeline(
    gaussians: GaussianModel,
    cam_centers: np.ndarray,
    cam_forwards: np.ndarray,
    fovy: float,
    h: int,
    w: int,
    roi_mask: Optional[torch.Tensor] = None,
    pipe_params=None,
    background: Optional[torch.Tensor] = None,
    seg_prompt: str = "",
    edit_prompt: str = "",
    use_ip2p_scoring: bool = False,
    ip2p_pipe=None,
    distance_multipliers: Optional[List[float]] = None,
    n_candidates: int = 150,
    n_select: int = 20,
    hemisphere_only: bool = False,
    cone_half_angle_deg: float = 90.0,
    w_vis: float = 0.6,
    w_can: float = 0.4,
    top_fraction: float = 0.20,
    diversity_x_weight: float = 0.0,
    diversity_y_variance_weight: float = 0.0,
    lambda_leak: float = 1.5,
    lambda_ent: float = 2.0,
    entropy_thresh: float = 0.97,
    override_opacity: Optional[torch.Tensor] = None,
    device: str = "cuda",
    v_front_method: str = "colmap_mean",
    ip2p_batch_size: int = 1,
) -> List[Simple_Camera]:
    """
    Run the full Generate-by-Lens pipeline. Returns list of Simple_Camera.

    Used by threestudio gs_load when edit_view_selection_strategy == "lens".
    """
    from argparse import Namespace
    from gaussiansplatting.arguments import PipelineParams
    from argparse import ArgumentParser

    if pipe_params is None:
        parser = ArgumentParser()
        pp = PipelineParams(parser)  # add args once; avoid re-calling which causes conflicts
        args = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
        pipe_params = pp.extract(args)

    if background is None:
        background = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)

    # ROI mask via segmentation if needed
    if seg_prompt and roi_mask is None and gaussians is not None:
        roi_mask = compute_roi_mask_from_segmentation(
            gaussians, seg_prompt, pipe_params, background,
            cam_centers, cam_forwards, fovy, h, w,
            override_opacity=override_opacity, device=device,
        )

    # Step 1: ROI Intrinsic Analysis
    roi_info = roi_intrinsic_analysis(
        gaussians, roi_mask, cam_forwards,
        cam_centers=cam_centers,
        v_front_method=v_front_method,
    )
    colmap_dists = np.linalg.norm(cam_centers - roi_info["center"][None, :], axis=1)
    colmap_median_dist = float(np.median(colmap_dists))
    roi_info["colmap_median_dist"] = colmap_median_dist
    if roi_info["object_size"] > colmap_median_dist:
        roi_info["object_size"] = colmap_median_dist

    # Step 2: Scale Probing
    dist_mults = distance_multipliers or [1.5, 2.0, 2.5, 3.0, 3.5]
    optimal_distance = scale_probing(
        gaussians, roi_info, pipe_params, background, fovy, h, w, roi_mask,
        ip2p_pipe=ip2p_pipe, edit_prompt=edit_prompt,
        distance_multipliers=dist_mults,
        lambda_leak=lambda_leak, lambda_ent=lambda_ent, entropy_thresh=entropy_thresh,
        override_opacity=override_opacity, device=device,
        ip2p_batch_size=ip2p_batch_size,
    )

    # Step 3: Fibonacci Manifold Sampling (cone axis = v_front so candidates align with chosen front)
    candidates = fibonacci_camera_candidates(
        center=roi_info["center"],
        distance=optimal_distance,
        n_candidates=n_candidates,
        fovy=fovy, h=h, w=w,
        hemisphere_only=hemisphere_only,
        colmap_cam_centers=cam_centers,
        cone_half_angle_deg=cone_half_angle_deg,
        cone_axis_direction=roi_info["v_front"],
        device=device,
    )

    # Step 4: Energy-based Scoring (with actual rendering)
    scored = score_candidates(
        candidates, gaussians, roi_mask, roi_info, pipe_params, background,
        w_vis=w_vis, w_can=w_can,
        override_opacity=override_opacity, device=device,
    )

    # Step 5: Diversity-aware Selection
    selected_indices = diversity_selection(
        candidates, scored, roi_info["center"],
        n_select=n_select, top_fraction=top_fraction,
        diversity_x_weight=diversity_x_weight,
        diversity_y_variance_weight=diversity_y_variance_weight,
        device=device,
    )

    final_cameras = [candidates[i] for i in selected_indices]
    center = roi_info["center"]

    def _azimuth(cam):
        cc = cam.camera_center.cpu().numpy()
        diff = cc - center
        return math.atan2(diff[2], diff[0])

    final_cameras.sort(key=_azimuth)
    return final_cameras


def simple_camera_to_c2w(simple_cam: Simple_Camera) -> np.ndarray:
    """Convert Simple_Camera to 4x4 c2w matrix. R is R_c2w, T = -(R^T @ cam_center)."""
    R = simple_cam.R
    cam_center = (-R @ np.array(simple_cam.T, dtype=np.float32)).astype(np.float32)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R
    c2w[:3, 3] = cam_center
    return c2w


# ===================================================================
# COLMAP export  (shared with render_orbit_from_colmap)
# ===================================================================

def save_cameras_to_colmap(
    cameras: List[Simple_Camera],
    output_dir: str,
    colmap_cameras: dict,
    fmt: str = "txt",
):
    """Write selected cameras in COLMAP format for downstream use."""
    from render_orbit_from_colmap import save_orbit_cameras_to_colmap

    source_cam = list(colmap_cameras.values())[0]
    save_orbit_cameras_to_colmap(
        cameras, output_dir, camera_id=1, format=fmt, source_camera=source_cam
    )


# ===================================================================
# Main
# ===================================================================

def main():
    parser = ArgumentParser(
        description="Generate-by-Lens: intelligent camera view selection for 3DGS editing"
    )
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    # Data paths
    parser.add_argument(
        "--ply_path", type=str, required=True,
        help="Path to the Gaussian .ply file (e.g. point_cloud.ply)",
    )
    parser.add_argument(
        "--colmap_path", type=str, required=True,
        help="Dataset root containing sparse/0",
    )

    # ROI specification
    parser.add_argument(
        "--seg_prompt", type=str, default="",
        help="Text prompt for LangSAM segmentation (e.g. 'a person'). "
             "If empty, all Gaussians are used as ROI.",
    )
    parser.add_argument(
        "--edit_prompt", type=str, default="",
        help="IP2P edit instruction for editability scoring",
    )

    # Pipeline control
    parser.add_argument(
        "--n_candidates", type=int, default=150,
        help="Number of Fibonacci sphere candidates (Step 3)",
    )
    parser.add_argument(
        "--n_select", type=int, default=20,
        help="Number of final views to select (Step 5)",
    )
    parser.add_argument(
        "--hemisphere_only", action="store_true",
        help="Restrict candidates to upper hemisphere",
    )
    parser.add_argument(
        "--cone_half_angle_deg",
        type=float,
        default=90.0,
        help="Half-angle (in degrees) of the COLMAP view cone used to filter Fibonacci candidates (Step 3)",
    )
    parser.add_argument(
        "--v_front_method",
        type=str,
        default="colmap_mean",
        choices=["colmap_mean", "scene_center"],
        help="How to compute v_front: colmap_mean = mean COLMAP view direction (default); scene_center = from ROI toward scene center (reduces left/right bias for off-center objects)",
    )
    parser.add_argument(
        "--use_ip2p_scoring", action="store_true",
        help="Use IP2P attention for editability scoring (slower, requires model download)",
    )
    parser.add_argument(
        "--distance_multipliers", type=str, default="1.5,2.0,2.5,3.0,3.5",
        help="Comma-separated distance multipliers for scale probing (× r_obj)",
    )

    # Scoring weights
    parser.add_argument("--w_vis", type=float, default=0.6, help="Visibility weight")
    parser.add_argument("--w_can", type=float, default=0.4, help="Canonical alignment weight")
    parser.add_argument("--top_fraction", type=float, default=0.20, help="Top fraction of candidates for FPS")
    parser.add_argument("--diversity_x_weight", type=float, default=0.0,
                        help="Extra weight for azimuth (x-axis) diversity in Step 5 (default: 0)")
    parser.add_argument("--diversity_y_variance_weight", type=float, default=0.0,
                        help="Penalize elevation spread to lower variance in y in Step 5 (default: 0)")

    # SAGE-Probing hyperparameters
    parser.add_argument("--lambda_leak", type=float, default=1.5,
                        help="Leakage penalty weight λ₁ in SAGE score (default: 1.5)")
    parser.add_argument("--lambda_ent", type=float, default=2.0,
                        help="Entropy ratio penalty weight λ₂ in SAGE score (default: 2.0)")
    parser.add_argument("--entropy_thresh", type=float, default=0.97,
                        help="Hard entropy ratio threshold τ (0-1) — views above this are marked unsafe (default: 0.97)")

    # Rendering
    parser.add_argument("--render_width", type=int, default=512)
    parser.add_argument("--render_height", type=int, default=512)
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="CUDA device to use (e.g. cuda, cuda:0, cuda:2). Default: cuda",
    )
    parser.add_argument(
        "--gpu", type=int, default=None, metavar="N",
        help="GPU index to use (e.g. 0, 1, 2). Overrides --device with cuda:N.",
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=7.5,
        help="IP2P text guidance scale (default 7.5). Lower preserves structure more.",
    )
    parser.add_argument(
        "--image_guidance_scale", type=float, default=1.5,
        help="IP2P image guidance scale (default 1.5). Higher preserves input image more.",
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=20,
        help="IP2P diffusion steps used for attention/editing (default 20). Lower is faster.",
    )
    parser.add_argument(
        "--ip2p_batch_size", type=int, default=1,
        help="Batch size for IP2P scale probing (default 1). Higher values process "
             "multiple distance candidates in a single UNet forward pass, reducing "
             "wall time roughly proportionally. Set to number of distance_multipliers "
             "for maximum speed. Requires more VRAM (~proportional to batch size).",
    )

    # Output
    parser.add_argument("--out_dir", type=str, default="output/lens")
    parser.add_argument("--video_path", type=str, default="output/lens.mp4")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--save_colmap", type=str, default=None,
        help="Save selected cameras in COLMAP format at this directory",
    )
    parser.add_argument(
        "--colmap_format", type=str, default="txt", choices=["txt", "bin"],
    )
    parser.add_argument(
        "--save_attn_grid", type=str, default=None,
        help="Save distance×resolution attention map grid image (e.g. output/attn_grid.jpg)",
    )
    parser.add_argument(
        "--visualize_roi",
        action="store_true",
        help="Overlay segmented ROI Gaussians in red on the rendered frames",
    )
    parser.add_argument(
        "--min_opacity",
        type=float,
        default=0.0,
        help="Prune Gaussians with opacity below this for rendering (0 = no pruning). Reduces floaters.",
    )
    parser.add_argument(
        "--prune_z_bottom_percent",
        type=float,
        default=0.0,
        metavar="P",
        help="Prune the bottom P%% of Gaussians by z value (e.g. 0.01 = remove bottom 0.01%%). 0 = disable.",
    )
    parser.add_argument(
        "--prune_y_top_percent",
        type=float,
        default=0.0,
        metavar="P",
        help="Prune the top P%% of Gaussians by y value (e.g. 0.4 = remove top 0.4%%). 0 = disable.",
    )
    parser.add_argument(
        "--prune_x_both_percent",
        type=float,
        default=0.0,
        metavar="P",
        help="Prune the top and bottom P%% of Gaussians by x value each (e.g. 3 = remove top 3%% and bottom 3%%). 0 = disable.",
    )
    parser.add_argument(
        "--save_pruned_ply",
        type=str,
        default=None,
        help="After pruning (opacity and/or below COLMAP z), save the pruned Gaussian model to this path (e.g. output/pruned.ply).",
    )

    args = get_combined_args(parser)
    safe_state(args.quiet)

    # ---------------------------------------------------------------
    # Latency logger (writes summary.txt similar to training script)
    # ---------------------------------------------------------------
    latency_logger: Optional[LatencyLogger] = None
    try:
        # Place latency log next to video_path (or under out_dir if no video)
        base_dir = os.path.dirname(args.video_path) if getattr(args, "video_path", None) else args.out_dir
        base_dir = base_dir or "."
        latency_dir = os.path.join(os.path.abspath(base_dir), "latency")
        latency_logger = LatencyLogger(latency_dir)
    except Exception as e:  # pragma: no cover - best-effort logging only
        print(f"[latency] WARNING: could not initialize latency logger: {e}")
        latency_logger = None

    device = getattr(args, "device", "cuda") or "cuda"
    if getattr(args, "gpu", None) is not None:
        # CUDA_VISIBLE_DEVICES was set at startup so only one GPU is visible; use it.
        device = "cuda"
    elif device.startswith("cuda:") and ":" in device:
        # Same when --device cuda:N was parsed and we set CUDA_VISIBLE_DEVICES.
        device = "cuda"
    print(f"[lens] Using device: {device}")

    # Ensure ModelParams defaults exist for extract() compatibility
    # (sentinel=True sets all defaults to None, which get_combined_args filters out)
    _model_defaults = {"source_path": "", "model_path": "", "sh_degree": 3,
                       "images": "images", "resolution": -1,
                       "white_background": False, "data_device": device, "eval": False}
    for k, v in _model_defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    # ---------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------
    if latency_logger:
        with latency_timeit(latency_logger, "load_data", device):
            with latency_timeit(latency_logger, "load_data.gaussians", device):
                gaussians = load_gaussians(args.ply_path, model.extract(args).sh_degree)

            with latency_timeit(latency_logger, "load_data.colmap_prior", device):
                cam_centers, cam_forwards, fovy, colmap_cameras = load_colmap_prior(
                    args.colmap_path
                )

            with latency_timeit(latency_logger, "load_data.pipeline_extract", device):
                bg_color = [1, 1, 1] if model.extract(args).white_background else [0, 0, 0]
                background = torch.tensor(bg_color, dtype=torch.float32, device=device)
                pipe_params = pipeline.extract(args)
    else:
        gaussians = load_gaussians(args.ply_path, model.extract(args).sh_degree)

        cam_centers, cam_forwards, fovy, colmap_cameras = load_colmap_prior(
            args.colmap_path
        )

        bg_color = [1, 1, 1] if model.extract(args).white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=device)
        pipe_params = pipeline.extract(args)

    h = int(args.render_height)
    w = int(args.render_width)

    # ---------------------------------------------------------------
    # Opacity pruning for rendering (reduces floaters)
    # ---------------------------------------------------------------
    render_opacity_override = None
    if getattr(args, "min_opacity", 0) > 0:
        op = gaussians.get_opacity
        render_opacity_override = torch.where(
            op >= args.min_opacity, op, torch.zeros_like(op)
        ).clone()
        n_keep = int((render_opacity_override > 0).sum().item())
        print(
            f"[Prune] Rendering with opacity >= {args.min_opacity}: "
            f"{n_keep}/{op.shape[0]} Gaussians"
        )

    # ---------------------------------------------------------------
    # Prune by z (bottom P%), y (top P%), x (top & bottom P% each) in-place, then optionally save
    # ---------------------------------------------------------------
    prune_z_pct = getattr(args, "prune_z_bottom_percent", 0.0)
    prune_y_pct = getattr(args, "prune_y_top_percent", 0.0)
    prune_x_pct = getattr(args, "prune_x_both_percent", 0.0)
    if prune_z_pct > 0 or prune_y_pct > 0 or prune_x_pct > 0:
        xyz = gaussians.get_xyz.detach()
        n_pts = xyz.shape[0]
        device = xyz.device
        keep_mask = torch.ones(n_pts, dtype=torch.bool, device=device)
        if prune_z_pct > 0:
            z = xyz[:, 2]
            k_z = max(0, int(round(n_pts * (prune_z_pct / 100.0))))
            if k_z > 0:
                _, idx_smallest_z = torch.topk(z, k_z, largest=False)
                keep_mask[idx_smallest_z] = False
        if prune_y_pct > 0:
            y = xyz[:, 1]
            k_y = max(0, int(round(n_pts * (prune_y_pct / 100.0))))
            if k_y > 0:
                _, idx_largest_y = torch.topk(y, k_y, largest=True)
                keep_mask[idx_largest_y] = False
        if prune_x_pct > 0:
            x = xyz[:, 0]
            k_x = max(0, int(round(n_pts * (prune_x_pct / 100.0))))
            if k_x > 0:
                _, idx_smallest_x = torch.topk(x, k_x, largest=False)
                _, idx_largest_x = torch.topk(x, k_x, largest=True)
                keep_mask[idx_smallest_x] = False
                keep_mask[idx_largest_x] = False
        n_remove = (~keep_mask).sum().item()
        if n_remove > 0:
            n_removed = _prune_gaussians_by_mask(gaussians, keep_mask)
            msg = []
            if prune_z_pct > 0:
                msg.append(f"z bottom {prune_z_pct}%")
            if prune_y_pct > 0:
                msg.append(f"y top {prune_y_pct}%")
            if prune_x_pct > 0:
                msg.append(f"x top & bottom {prune_x_pct}% each")
            print(
                f"[Prune] Removed {n_removed} Gaussians ({', '.join(msg)}). "
                f"Remaining: {gaussians.get_xyz.shape[0]}"
            )
        else:
            print("[Prune] No points to remove (no change).")
    if getattr(args, "save_pruned_ply", None):
        save_path = args.save_pruned_ply
        os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
        gaussians.save_ply(save_path)
        print(f"[Prune] Saved pruned Gaussians to {save_path}")

    # ---------------------------------------------------------------
    # ROI mask (optional segmentation)
    # ---------------------------------------------------------------
    roi_mask = None
    if args.seg_prompt:
        if latency_logger:
            with latency_timeit(latency_logger, "step0", device):
                with latency_timeit(latency_logger, "step0.compute_roi_mask", device):
                    roi_mask = compute_roi_mask_from_segmentation(
                        gaussians,
                        args.seg_prompt,
                        pipe_params,
                        background,
                        cam_centers,
                        cam_forwards,
                        fovy,
                        h,
                        w,
                        override_opacity=render_opacity_override,
                        device=device,
                        latency_logger=latency_logger,
                    )
        else:
            roi_mask = compute_roi_mask_from_segmentation(
                gaussians,
                args.seg_prompt,
                pipe_params,
                background,
                cam_centers,
                cam_forwards,
                fovy,
                h,
                w,
                override_opacity=render_opacity_override,
                device=device,
            )

    # ---------------------------------------------------------------
    # Step 1: ROI Intrinsic Analysis
    # ---------------------------------------------------------------
    print("\n========== Step 1: ROI Intrinsic Analysis ==========")
    if latency_logger:
        with latency_timeit(latency_logger, "step1", device):
            with latency_timeit(latency_logger, "step1.roi_intrinsic_analysis", device):
                roi_info = roi_intrinsic_analysis(
                    gaussians, roi_mask, cam_forwards,
                    cam_centers=cam_centers,
                    v_front_method=getattr(args, "v_front_method", "colmap_mean"),
                )
    else:
        roi_info = roi_intrinsic_analysis(
            gaussians, roi_mask, cam_forwards,
            cam_centers=cam_centers,
            v_front_method=getattr(args, "v_front_method", "colmap_mean"),
        )

    # Use COLMAP median distance to ROI center as a reference
    colmap_dists = np.linalg.norm(cam_centers - roi_info["center"][None, :], axis=1)
    colmap_median_dist = float(np.median(colmap_dists))
    roi_info["colmap_median_dist"] = colmap_median_dist
    print(f"[Step1] COLMAP median distance to ROI center: {colmap_median_dist:.4f}")

    # Cap object_size by COLMAP median distance to avoid overly far cameras
    if roi_info["object_size"] > colmap_median_dist:
        print(f"[Step1] Capping object_size {roi_info['object_size']:.4f} → "
              f"{colmap_median_dist:.4f} (COLMAP reference)")
        roi_info["object_size"] = colmap_median_dist

    # ---------------------------------------------------------------
    # Step 2: Scale Probing
    # ---------------------------------------------------------------
    print("\n========== Step 2: SAGE-Probing (Sharpness-Aware Scale Probing) ==========")
    ip2p_pipe = None
    if args.use_ip2p_scoring and args.edit_prompt:
        from diffusers import StableDiffusionInstructPix2PixPipeline

        print("[Step2] Loading IP2P model for SAGE attention-based scoring ...")
        ip2p_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to(device)

    dist_mults = [float(x) for x in args.distance_multipliers.split(",")]
    if latency_logger:
        with latency_timeit(latency_logger, "step2", device):
            with latency_timeit(latency_logger, "step2.scale_probing", device):
                optimal_distance = scale_probing(
                    gaussians,
                    roi_info,
                    pipe_params,
                    background,
                    fovy,
                    h,
                    w,
                    roi_mask,
                    ip2p_pipe=ip2p_pipe,
                    edit_prompt=args.edit_prompt,
                    distance_multipliers=dist_mults,
                    lambda_leak=args.lambda_leak,
                    lambda_ent=args.lambda_ent,
                    entropy_thresh=args.entropy_thresh,
                    override_opacity=render_opacity_override,
                    save_attn_grid=args.save_attn_grid,
                    device=device,
                    ip2p_guidance_scale=getattr(args, "guidance_scale", 7.5),
                    ip2p_image_guidance_scale=getattr(args, "image_guidance_scale", 1.5),
                    ip2p_num_inference_steps=getattr(args, "num_inference_steps", 20),
                    ip2p_batch_size=getattr(args, "ip2p_batch_size", 1),
                    latency_logger=latency_logger,
                )
    else:
        optimal_distance = scale_probing(
            gaussians,
            roi_info,
            pipe_params,
            background,
            fovy,
            h,
            w,
            roi_mask,
            ip2p_pipe=ip2p_pipe,
            edit_prompt=args.edit_prompt,
            distance_multipliers=dist_mults,
            lambda_leak=args.lambda_leak,
            lambda_ent=args.lambda_ent,
            entropy_thresh=args.entropy_thresh,
            override_opacity=render_opacity_override,
            save_attn_grid=args.save_attn_grid,
            device=device,
            ip2p_guidance_scale=getattr(args, "guidance_scale", 7.5),
            ip2p_image_guidance_scale=getattr(args, "image_guidance_scale", 1.5),
            ip2p_num_inference_steps=getattr(args, "num_inference_steps", 20),
            ip2p_batch_size=getattr(args, "ip2p_batch_size", 1),
            latency_logger=None,
        )

    # Free IP2P if loaded
    if ip2p_pipe is not None:
        del ip2p_pipe
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # Step 3: Fibonacci Manifold Sampling
    # ---------------------------------------------------------------
    print("\n========== Step 3: Fibonacci Manifold Sampling ==========")
    if latency_logger:
        with latency_timeit(latency_logger, "step3", device):
            with latency_timeit(latency_logger, "step3.fibonacci_sampling", device):
                candidates = fibonacci_camera_candidates(
                    center=roi_info["center"],
                    distance=optimal_distance,
                    n_candidates=int(args.n_candidates),
                    fovy=fovy,
                    h=h,
                    w=w,
                    hemisphere_only=args.hemisphere_only,
                    colmap_cam_centers=cam_centers,
                    cone_half_angle_deg=args.cone_half_angle_deg,
                    cone_axis_direction=roi_info["v_front"],
                    device=device,
                )
    else:
        candidates = fibonacci_camera_candidates(
            center=roi_info["center"],
            distance=optimal_distance,
            n_candidates=int(args.n_candidates),
            fovy=fovy,
            h=h,
            w=w,
            hemisphere_only=args.hemisphere_only,
            colmap_cam_centers=cam_centers,
            cone_half_angle_deg=args.cone_half_angle_deg,
            cone_axis_direction=roi_info["v_front"],
            device=device,
        )

    # ---------------------------------------------------------------
    # Step 4: Energy-based Candidate Scoring
    # ---------------------------------------------------------------
    print("\n========== Step 4: Energy-based Scoring ==========")
    if latency_logger:
        with latency_timeit(latency_logger, "step4", device):
            with latency_timeit(latency_logger, "step4.score_candidates", device):
                scored = score_candidates(
                    candidates,
                    gaussians,
                    roi_mask,
                    roi_info,
                    pipe_params,
                    background,
                    w_vis=args.w_vis,
                    w_can=args.w_can,
                    override_opacity=render_opacity_override,
                    device=device,
                    latency_logger=latency_logger,
                )
    else:
        scored = score_candidates(
            candidates,
            gaussians,
            roi_mask,
            roi_info,
            pipe_params,
            background,
            w_vis=args.w_vis,
            w_can=args.w_can,
            override_opacity=render_opacity_override,
            device=device,
        )

    # ---------------------------------------------------------------
    # Step 5: Diversity-aware Selection
    # ---------------------------------------------------------------
    print("\n========== Step 5: Diversity-aware Selection ==========")
    if latency_logger:
        with latency_timeit(latency_logger, "step5", device):
            with latency_timeit(latency_logger, "step5.diversity_selection", device):
                selected_indices = diversity_selection(
                    candidates,
                    scored,
                    roi_info["center"],
                    n_select=int(args.n_select),
                    top_fraction=args.top_fraction,
                    diversity_x_weight=getattr(args, "diversity_x_weight", 0.0),
                    diversity_y_variance_weight=getattr(args, "diversity_y_variance_weight", 0.0),
                    latency_logger=latency_logger,
                    device=device,
                )
    else:
        selected_indices = diversity_selection(
            candidates,
            scored,
            roi_info["center"],
            n_select=int(args.n_select),
            top_fraction=args.top_fraction,
            diversity_x_weight=getattr(args, "diversity_x_weight", 0.0),
            diversity_y_variance_weight=getattr(args, "diversity_y_variance_weight", 0.0),
            device=device,
        )

    final_cameras = [candidates[i] for i in selected_indices]

    # ---------------------------------------------------------------
    # Sort final cameras by azimuth for smooth video
    # ---------------------------------------------------------------
    center = roi_info["center"]

    def _azimuth(cam):
        cc = cam.camera_center.cpu().numpy()
        diff = cc - center
        return math.atan2(diff[2], diff[0])

    final_cameras.sort(key=_azimuth)

    # ---------------------------------------------------------------
    # Render & save
    # ---------------------------------------------------------------
    print(f"\n========== Rendering {len(final_cameras)} selected views ==========")
    os.makedirs(args.out_dir, exist_ok=True)
    frames = []

    def _render_selected_views():
        nonlocal frames
        for i, cam in enumerate(final_cameras):
            out = render(cam, gaussians, pipe_params, background, override_opacity=render_opacity_override)
            rgb = out["render"]

            # Optionally overlay segmented ROI Gaussians in red
            if args.visualize_roi and roi_mask is not None:
                N = gaussians.get_xyz.shape[0]
                roi_mask_3d = roi_mask.to(gaussians.get_xyz.device).bool()
                colors = torch.zeros(N, 3, device=gaussians.get_xyz.device)
                colors[roi_mask_3d] = torch.tensor([1.0, 0.0, 0.0], device=colors.device)

                # Render ROI-only pass (red on black) and use it as an alpha mask
                out_roi = render(
                    cam,
                    gaussians,
                    pipe_params,
                    torch.zeros(3, device=background.device),
                    override_color=colors,
                    override_opacity=render_opacity_override,
                )
                roi_rgb = out_roi["render"]  # (3, H, W)
                roi_mask_2d = roi_rgb.mean(dim=0, keepdim=True)  # (1, H, W), ~1 in ROI, 0 elsewhere

                alpha = 0.6
                red = torch.tensor([1.0, 0.0, 0.0], device=rgb.device)[:, None, None]
                rgb = rgb * (1.0 - alpha * roi_mask_2d) + red * (alpha * roi_mask_2d)

            out_path = os.path.join(args.out_dir, f"{i:05d}.png")
            torchvision.utils.save_image(rgb, out_path)
            frame_np = (
                (rgb.clamp(0.0, 1.0) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
            )
            frames.append(frame_np)

            if (i + 1) % 5 == 0:
                print(f"  Rendered {i + 1}/{len(final_cameras)}")

    if latency_logger:
        with latency_timeit(latency_logger, "render_selected_views", device):
            _render_selected_views()
    else:
        _render_selected_views()

  
    # Save concatenated PNG of all generated views (horizontal strip)
    if frames:
        concat_path = os.path.join(args.out_dir, "generated_views.png")
        try:
            if latency_logger:
                with latency_timeit(latency_logger, "save_generated_views_image", device):
                    concat_img = np.concatenate(frames, axis=1)
                    imageio.imwrite(concat_path, concat_img)
            else:
                concat_img = np.concatenate(frames, axis=1)
                imageio.imwrite(concat_path, concat_img)
            print(f"Concatenated views image saved to {concat_path}")
        except Exception as e:  # pragma: no cover - best-effort visualisation
            print(f"[WARN] Failed to save concatenated views image: {e}")
    
    # Save video
    if args.video_path and frames:
        if latency_logger:
            with latency_timeit(latency_logger, "save_video", device):
                os.makedirs(os.path.dirname(args.video_path) or ".", exist_ok=True)
                print(f"Writing video to {args.video_path} ({len(frames)} frames, fps={args.fps})")
                imageio.mimsave(args.video_path, frames, fps=int(args.fps))
        else:
            os.makedirs(os.path.dirname(args.video_path) or ".", exist_ok=True)
            print(f"Writing video to {args.video_path} ({len(frames)} frames, fps={args.fps})")
            imageio.mimsave(args.video_path, frames, fps=int(args.fps))

    # Save COLMAP
    if args.save_colmap:
        if latency_logger:
            with latency_timeit(latency_logger, "save_colmap", device):
                save_cameras_to_colmap(
                    final_cameras, args.save_colmap, colmap_cameras, fmt=args.colmap_format
                )
                print(f"COLMAP cameras saved to {args.save_colmap}/sparse/0/")
        else:
            save_cameras_to_colmap(
                final_cameras, args.save_colmap, colmap_cameras, fmt=args.colmap_format
            )
            print(f"COLMAP cameras saved to {args.save_colmap}/sparse/0/")

    # Latency summary
    if latency_logger is not None:
        latency_logger.write_summary()
        print(f"[latency] Summary written to {latency_logger.base_dir}/summary.txt")

    print("\n[Done] Generate-by-Lens complete.")
    print(f"  Frames : {args.out_dir}/")
    print(f"  Video  : {args.video_path}")


if __name__ == "__main__":
    main()
