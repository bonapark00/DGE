#!/usr/bin/env python3
"""
Generate-by-Lens: Intelligent camera view generation for 3D Gaussian Splatting editing.

This script selects optimal camera views for IP2P-based editing by analysing the
3D Gaussian distribution of the ROI (Region of Interest).

Pipeline:
  Step 1 – ROI Intrinsic Analysis   (Weighted PCA on ROI Gaussians)
  Step 2 – Scale Probing            (find optimal viewing distance via editability score)
  Step 3 – Fibonacci Manifold Sampling (uniform candidate views on a sphere)
  Step 4 – Energy-based Scoring     (visibility + canonical alignment)
  Step 5 – Diversity-aware Selection (Farthest Point Sampling on top candidates)

Outputs:
  - PNG frames in --out_dir
  - MP4 video  at --video_path
  - (Optional) COLMAP cameras via --save_colmap
"""

# Compatibility patch for huggingface_hub  (must be first)
import hf_hub_patch  # noqa: E402, F401

import collections
import math
import os
from argparse import ArgumentParser
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch
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
from gaussiansplatting.utils.graphics_utils import focal2fov, fov2focal, getWorld2View2

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
    """Camera-to-world 4x4 (camera looks along +Z towards *target*)."""
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
) -> Dict:
    """
    Perform weighted PCA on ROI Gaussians to find the object centre,
    principal axes, and a robust *front* direction.

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

    # Front direction: project mean camera forward onto span(v2, v3) or use v3
    mean_fwd = _normalize(cam_forwards.mean(axis=0))
    # Remove component along v1 (longest axis, e.g. vertical for a standing person)
    proj_on_v1 = np.dot(mean_fwd, v1) * v1
    v_front = _normalize(mean_fwd - proj_on_v1)
    if np.linalg.norm(v_front) < 1e-6:
        v_front = v3.copy()

    center_np = center.cpu().numpy()

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
        data_device="cuda",
        qvec=None,
    )


def _compute_editability_score(
    rendered_rgb: torch.Tensor,
    roi_mask_2d: torch.Tensor,
    ip2p_pipe,
    prompt: str,
    lam: float = 0.5,
    num_steps: int = 5,
) -> float:
    """
    Run a fast IP2P forward (few steps) and compare cross-attention to ROI mask.
    Returns editability score  S_edit = Focus - λ * Leakage.

    If ip2p_pipe is None, fall back to a simple mask-coverage heuristic.
    """
    M = roi_mask_2d.float()  # (1, H, W)
    mask_ratio = float(M.sum() / (M.numel() + 1e-8))

    if ip2p_pipe is None:
        # Heuristic: prefer views where mask covers 15-50 % of pixels
        optimal_ratio = 0.30
        focus = max(0.0, 1.0 - abs(mask_ratio - optimal_ratio) / optimal_ratio)
        leakage = 0.0
        return focus - lam * leakage

    # --- Full IP2P attention path ---
    from PIL import Image as PILImage
    from torchvision.transforms import ToPILImage, ToTensor

    rgb_pil = ToPILImage()(rendered_rgb.cpu().clamp(0, 1))
    # Resize to 512 for IP2P
    rgb_pil = rgb_pil.resize((512, 512), PILImage.BICUBIC)

    # Hook to capture cross-attention maps
    attn_maps = []

    def _attn_hook(module, input, output):
        # output is a tuple; attention weights are the second element
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            attn_maps.append(output[1].detach().cpu())

    hooks = []
    for name, mod in ip2p_pipe.unet.named_modules():
        if "attn2" in name and hasattr(mod, "forward"):
            hooks.append(mod.register_forward_hook(_attn_hook))

    with torch.no_grad():
        _ = ip2p_pipe(
            prompt=prompt,
            image=rgb_pil,
            num_inference_steps=num_steps,
            guidance_scale=7.5,
            image_guidance_scale=1.5,
            output_type="pil",
        )

    for hk in hooks:
        hk.remove()

    if not attn_maps:
        # Fallback if no attention captured
        optimal_ratio = 0.30
        focus = max(0.0, 1.0 - abs(mask_ratio - optimal_ratio) / optimal_ratio)
        return focus

    # Average attention over all captured maps and resize to mask resolution
    A_avg = torch.stack(attn_maps).mean(dim=0)  # rough average
    # Flatten spatial dims -> (H', W')
    h_m, w_m = M.shape[1], M.shape[2]
    side = int(math.sqrt(A_avg.shape[-2]))
    if side * side == A_avg.shape[-2]:
        A_spatial = A_avg.mean(dim=(0, 1))[: side * side, :].mean(dim=-1).view(side, side)
        A_spatial = (
            torch.nn.functional.interpolate(
                A_spatial[None, None].float(), size=(h_m, w_m), mode="bilinear"
            )
            .squeeze()
        )
    else:
        # Fallback
        A_spatial = torch.ones(h_m, w_m)

    A = A_spatial.to(M.device)
    A = (A - A.min()) / (A.max() - A.min() + 1e-8)  # normalize 0-1

    focus = float((A * M.squeeze()).sum() / (A.sum() + 1e-8))
    leakage = float((A * (1.0 - M.squeeze())).sum() / ((1.0 - M.squeeze()).sum() + 1e-8))
    return focus - lam * leakage


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
    lam: float = 0.5,
) -> float:
    """
    Find the optimal viewing distance d* by rendering pilot views at
    several distances along v_front and scoring each.
    """
    if distance_multipliers is None:
        distance_multipliers = [0.5, 1.0, 1.5, 2.0, 2.5]

    center = roi_info["center"]
    v_front = roi_info["v_front"]
    obj_size = roi_info["object_size"]
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    best_d, best_score = obj_size, -1e9
    scores = []

    for i, mult in enumerate(distance_multipliers):
        d = mult * obj_size
        eye = center + v_front * d
        cam = _make_camera(eye, center, world_up, fovy, h, w, uid=i)

        out = render(cam, gaussians, pipe_params, background)
        rgb = out["render"]  # (3, H, W)
        depth = out["depth_3dgs"]  # (1, H, W)

        # Project ROI mask to 2D via depth comparison
        if roi_mask is not None:
            mask_2d = _project_roi_mask(
                gaussians, roi_mask, cam, pipe_params, background
            )
        else:
            # Use centre-distance heuristic
            mask_2d = torch.ones(1, h, w, device="cuda")

        score = _compute_editability_score(
            rgb, mask_2d, ip2p_pipe, edit_prompt, lam=lam
        )
        scores.append((mult, d, score))
        if score > best_score:
            best_score = score
            best_d = d

    print(f"[Step2] Scale probing scores: {[(m, f'{s:.4f}') for m, _, s in scores]}")
    print(f"[Step2] Best distance d*={best_d:.4f} (mult={best_d / obj_size:.2f}x)")
    return best_d


@torch.no_grad()
def _project_roi_mask(
    gaussians: GaussianModel,
    roi_mask_3d: torch.Tensor,
    cam: Simple_Camera,
    pipe_params,
    background: torch.Tensor,
) -> torch.Tensor:
    """Render ROI Gaussians as a solid-colour overlay to get a 2D projection mask."""
    N = gaussians.get_xyz.shape[0]
    roi_mask_3d = roi_mask_3d.to(gaussians.get_xyz.device).bool()

    # Render with override_color: ROI = white, rest = black
    colors = torch.zeros(N, 3, device="cuda")
    colors[roi_mask_3d] = 1.0

    bg_black = torch.zeros(3, device="cuda")
    out = render(cam, gaussians, pipe_params, bg_black, override_color=colors)
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
) -> List[Simple_Camera]:
    """
    Place *n_candidates* cameras on a sphere of radius *distance* centred
    on *center*, each looking at the centre.

    If *colmap_cam_centers* is provided, restrict candidates to a cone
    around the mean COLMAP viewing direction (from cameras → center).
    """
    if world_up is None:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    directions = fibonacci_sphere_samples(n_candidates)

    if hemisphere_only:
        # Keep only directions with positive y (above ground)
        up_axis = _normalize(world_up)
        dots = directions @ up_axis
        directions = directions[dots > -0.1]  # allow slight below-horizon

    # Constrain to cone around mean COLMAP viewing direction
    if colmap_cam_centers is not None:
        # Mean direction from center → COLMAP cameras
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
        cam = _make_camera(eye, center, world_up, fovy, h, w, uid=i)
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
        out = render(cam, gaussians, pipe_params, background)
        depth = out["depth_3dgs"]  # (1, H, W)

        # --- Visibility Score ---
        if roi_mask is not None:
            mask_2d = _project_roi_mask(
                gaussians, roi_mask, cam, pipe_params, background
            )
            mask_area = float(mask_2d.sum())
            total_area = float(mask_2d.numel())
            S_vis = mask_area / (total_area + 1e-8)
        else:
            S_vis = 1.0

        # --- Canonical Score ---
        cam_center = cam.camera_center.cpu().numpy()  # (3,)
        view_dir = _normalize(center - cam_center)

        # Reward alignment with v_front or side axis v2
        cos_front = abs(float(np.dot(view_dir, v_front)))
        cos_side = abs(float(np.dot(view_dir, v2)))
        S_can = max(cos_front, cos_side * 0.8)

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
) -> List[int]:
    """
    Select *n_select* cameras from the top-scoring pool using
    Farthest Point Sampling (angular distance on the view sphere).
    """
    # Filter to top fraction
    n_pool = max(int(len(scored) * top_fraction), n_select)
    pool = scored[:n_pool]  # already sorted descending by energy
    pool_indices = [s[0] for s in pool]
    pool_energies = {s[0]: s[1] for s in pool}

    # Compute unit view directions for pool cameras
    view_dirs = {}
    for ci in pool_indices:
        cc = cameras[ci].camera_center.cpu().numpy()
        view_dirs[ci] = _normalize(cc - center)

    selected = []
    remaining = set(pool_indices)

    # Start with highest-energy camera
    first = pool_indices[0]
    selected.append(first)
    remaining.discard(first)

    while len(selected) < n_select and remaining:
        best_idx, best_score = None, -1e9
        for ci in remaining:
            # Min angular distance to any already-selected view
            min_ang = 1e9
            for si in selected:
                cos_sim = float(np.dot(view_dirs[ci], view_dirs[si]))
                ang = math.acos(np.clip(cos_sim, -1.0, 1.0))
                min_ang = min(min_ang, ang)
            # Combined: diversity (angular distance) * energy
            combined = min_ang * pool_energies[ci]
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
) -> torch.Tensor:
    """
    Render a few views, run LangSAM segmentation, and back-project
    to create a per-Gaussian ROI mask.
    """
    from threestudio.utils.sam import LangSAMTextSegmentor

    segmentor = LangSAMTextSegmentor()
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    N_gauss = gaussians.get_xyz.shape[0]
    accum = torch.zeros(N_gauss, device="cuda")
    count = torch.zeros(N_gauss, device="cuda")

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
        eye = cam_centers[cam_idx].astype(np.float32)
        fwd = _normalize(cam_forwards[cam_idx].astype(np.float32))
        target = eye + fwd

        cam = _make_camera(eye, target, world_up, fovy, h, w, uid=1000 + j)
        out = render(cam, gaussians, pipe_params, background)
        rgb = out["render"]  # (3, H, W)

        # LangSAM expects (B, H, W, C)
        img_bhwc = rgb.permute(1, 2, 0).unsqueeze(0)  # (1, H, W, 3)
        mask_2d = segmentor(img_bhwc, prompt)  # (1, 1, H, W)
        mask_2d = mask_2d.squeeze(0)  # (1, H, W)

        if mask_2d.sum() < 10:
            continue

        # Back-project: render with per-Gaussian colours = index
        # Use override_color trick: render with white for ROI estimation
        vis_filter = out["visibility_filter"]  # (N,) bool
        accum[vis_filter] += mask_2d.mean().item()
        count[vis_filter] += 1.0

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
        "--use_ip2p_scoring", action="store_true",
        help="Use IP2P attention for editability scoring (slower, requires model download)",
    )
    parser.add_argument(
        "--distance_multipliers", type=str, default="0.5,1.0,1.5,2.0,2.5",
        help="Comma-separated distance multipliers for scale probing",
    )

    # Scoring weights
    parser.add_argument("--w_vis", type=float, default=0.6, help="Visibility weight")
    parser.add_argument("--w_can", type=float, default=0.4, help="Canonical alignment weight")
    parser.add_argument("--top_fraction", type=float, default=0.20, help="Top fraction of candidates for FPS")
    parser.add_argument("--lambda_leak", type=float, default=0.5, help="Leakage penalty in editability score")

    # Rendering
    parser.add_argument("--render_width", type=int, default=512)
    parser.add_argument("--render_height", type=int, default=512)

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
        "--visualize_roi",
        action="store_true",
        help="Overlay segmented ROI Gaussians in red on the rendered frames",
    )

    args = get_combined_args(parser)
    safe_state(args.quiet)

    # Ensure ModelParams defaults exist for extract() compatibility
    # (sentinel=True sets all defaults to None, which get_combined_args filters out)
    _model_defaults = {"source_path": "", "model_path": "", "sh_degree": 3,
                       "images": "images", "resolution": -1,
                       "white_background": False, "data_device": "cuda", "eval": False}
    for k, v in _model_defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    # ---------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------
    gaussians = load_gaussians(args.ply_path, model.extract(args).sh_degree)

    cam_centers, cam_forwards, fovy, colmap_cameras = load_colmap_prior(
        args.colmap_path
    )

    bg_color = [1, 1, 1] if model.extract(args).white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pipe_params = pipeline.extract(args)

    h = int(args.render_height)
    w = int(args.render_width)

    # ---------------------------------------------------------------
    # ROI mask (optional segmentation)
    # ---------------------------------------------------------------
    roi_mask = None
    if args.seg_prompt:
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
        )

    # ---------------------------------------------------------------
    # Step 1: ROI Intrinsic Analysis
    # ---------------------------------------------------------------
    print("\n========== Step 1: ROI Intrinsic Analysis ==========")
    roi_info = roi_intrinsic_analysis(gaussians, roi_mask, cam_forwards)

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
    print("\n========== Step 2: Scale Probing ==========")
    ip2p_pipe = None
    if args.use_ip2p_scoring and args.edit_prompt:
        from diffusers import StableDiffusionInstructPix2PixPipeline

        print("[Step2] Loading IP2P model for attention-based scoring ...")
        ip2p_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to("cuda")

    dist_mults = [float(x) for x in args.distance_multipliers.split(",")]
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
        lam=args.lambda_leak,
    )

    # Free IP2P if loaded
    if ip2p_pipe is not None:
        del ip2p_pipe
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # Step 3: Fibonacci Manifold Sampling
    # ---------------------------------------------------------------
    print("\n========== Step 3: Fibonacci Manifold Sampling ==========")
    candidates = fibonacci_camera_candidates(
        center=roi_info["center"],
        distance=optimal_distance,
        n_candidates=int(args.n_candidates),
        fovy=fovy,
        h=h,
        w=w,
        hemisphere_only=args.hemisphere_only,
        colmap_cam_centers=cam_centers,
        cone_half_angle_deg=90.0,
    )

    # ---------------------------------------------------------------
    # Step 4: Energy-based Candidate Scoring
    # ---------------------------------------------------------------
    print("\n========== Step 4: Energy-based Scoring ==========")
    scored = score_candidates(
        candidates,
        gaussians,
        roi_mask,
        roi_info,
        pipe_params,
        background,
        w_vis=args.w_vis,
        w_can=args.w_can,
    )

    # ---------------------------------------------------------------
    # Step 5: Diversity-aware Selection
    # ---------------------------------------------------------------
    print("\n========== Step 5: Diversity-aware Selection ==========")
    selected_indices = diversity_selection(
        candidates,
        scored,
        roi_info["center"],
        n_select=int(args.n_select),
        top_fraction=args.top_fraction,
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

    for i, cam in enumerate(final_cameras):
        out = render(cam, gaussians, pipe_params, background)
        rgb = out["render"]

        # Optionally overlay segmented ROI Gaussians in red
        if args.visualize_roi and roi_mask is not None:
            N = gaussians.get_xyz.shape[0]
            roi_mask_3d = roi_mask.to(gaussians.get_xyz.device).bool()
            colors = torch.zeros(N, 3, device=gaussians.get_xyz.device)
            colors[roi_mask_3d] = torch.tensor([1.0, 0.0, 0.0], device=colors.device)

            # Render ROI-only pass (red on black) and use it as an alpha mask
            out_roi = render(cam, gaussians, pipe_params, torch.zeros(3, device=background.device), override_color=colors)
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

    # Save video
    if args.video_path and frames:
        os.makedirs(os.path.dirname(args.video_path) or ".", exist_ok=True)
        print(f"Writing video to {args.video_path} ({len(frames)} frames, fps={args.fps})")
        imageio.mimsave(args.video_path, frames, fps=int(args.fps))

    # Save COLMAP
    if args.save_colmap:
        save_cameras_to_colmap(
            final_cameras, args.save_colmap, colmap_cameras, fmt=args.colmap_format
        )
        print(f"COLMAP cameras saved to {args.save_colmap}/sparse/0/")

    print("\n[Done] Generate-by-Lens complete.")
    print(f"  Frames : {args.out_dir}/")
    print(f"  Video  : {args.video_path}")


if __name__ == "__main__":
    main()
