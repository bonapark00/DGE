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
        --seg_prompt "red sweater" \
        --edit_prompt "Change the red sweater into a leather jacket" \
        --use_ip2p_scoring \
        --save_colmap output/lens_colmap \
        --video_path output/lens_claude.mp4 \
        --distance_multipliers "2.0, 2.5, 3.0, 4.0, 5.0, 6.0" \
        --visualize_roi \
        --n_select 20 \
        --save_attn_grid output/attn_grid.jpg

"""


# Compatibility patch for huggingface_hub  (must be first)
import hf_hub_patch  # noqa: E402, F401

import collections
import math
import os
from argparse import ArgumentParser
from contextlib import contextmanager
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

    # Front direction: direction from object centre toward cameras.
    # cam_forwards points camera→scene, so negate to get scene→camera.
    mean_fwd = _normalize(cam_forwards.mean(axis=0))
    # Remove component along v1 (longest axis, e.g. vertical for a standing person)
    proj_on_v1 = np.dot(mean_fwd, v1) * v1
    v_front = _normalize(-(mean_fwd - proj_on_v1))
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
        storing_processors = {}
        for name, mod in ip2p_pipe.unet.named_modules():
            if name.endswith(".attn2") and hasattr(mod, "processor"):
                original_processors[name] = mod.processor
                sp = _StoringAttnProcessor()
                storing_processors[name] = sp
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

    return storing_processors, keyword_indices, edited_t


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
    storing_processors, keyword_indices, _ = _run_ip2p_and_collect(
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
    storing_processors, keyword_indices, _ = _run_ip2p_and_collect(
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
        storing_processors, keyword_indices, edited_t = _run_ip2p_and_collect(
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

    # --- Compute SAGE score ---
    with latency_timeit(latency_logger, "step2.sage_score", device):
        if A_16 is not None:
            sage_score, sage_details = _compute_sage_score(
                A_16, roi_mask_2d,
                lambda_leak=lambda_leak,
                lambda_ent=lambda_ent,
                entropy_thresh=entropy_thresh,
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
                focus=focus, leakage=0.0, entropy=0.0,
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


def _compute_sage_score(
    A_spatial: torch.Tensor,
    roi_mask_2d: torch.Tensor,
    lambda_leak: float = 1.5,
    lambda_ent: float = 2.0,
    entropy_thresh: float = 0.97,
    occupancy_lo: float = 0.10,
    occupancy_hi: float = 0.70,
    size_penalty_val: float = 10.0,
) -> Tuple[float, Dict]:
    """
    SAGE (Sharpness-Aware Guided Editability) scoring.

    Computes:
      S_total = S_focus - λ₁·S_leakage - λ₂·H_ratio(A) - Penalty_size

    Entropy is expressed as a ratio of the theoretical maximum (0 = perfectly
    sharp, 1 = uniform). Hard constraint: H_ratio > τ → score = -inf.

    Args:
        A_spatial: (H_a, W_a) raw attention map (will be resized to mask).
        roi_mask_2d: (1, H, W) binary ROI mask.
        lambda_leak: weight for leakage penalty.
        lambda_ent: weight for entropy ratio penalty (default 2.0).
        entropy_thresh: hard threshold for entropy ratio filtering (default 0.97).
        occupancy_lo: minimum acceptable occupancy ratio.
        occupancy_hi: maximum acceptable occupancy ratio.
        size_penalty_val: penalty value for out-of-range occupancy.

    Returns:
        (total_score, details_dict) where details_dict contains individual metrics.
    """
    M = roi_mask_2d.float().squeeze(0)  # (H, W)
    h_m, w_m = M.shape

    # --- 3. Entropy (Sharpness) — compute as ratio on NATIVE resolution ---
    # Compute on the original attention map and normalise by the theoretical
    # maximum entropy at that resolution so the value lies in [0, 1].
    # This makes the threshold resolution-independent (default 0.97 means
    # "reject views whose attention is >97% of maximum entropy").
    A_native = A_spatial.float()
    A_native_norm = (A_native - A_native.min()) / (A_native.max() - A_native.min() + 1e-8)
    raw_entropy = compute_entropy(A_native_norm)
    max_entropy = math.log(A_native.numel()) + 1e-8
    entropy = raw_entropy / max_entropy  # ratio in [0, 1]

    # Resize attention map to mask resolution for focus/leakage computation
    A_resized = (
        torch.nn.functional.interpolate(
            A_spatial[None, None].float(), size=(h_m, w_m), mode="bilinear"
        )
        .squeeze()
    )
    A = A_resized.to(M.device)
    # Normalize to [0, 1]
    A = (A - A.min()) / (A.max() - A.min() + 1e-8)

    # --- 1. Focus Score ---
    focus = float((A * M).sum() / (A.sum() + 1e-8))

    # --- 2. Leakage Score ---
    bg = 1.0 - M
    leakage = float((A * bg).sum() / (bg.sum() + 1e-8))

    # --- 4. Occupancy Penalty ---
    occupancy = float(M.mean())
    size_penalty = 0.0
    if occupancy < occupancy_lo or occupancy > occupancy_hi:
        size_penalty = size_penalty_val

    # --- Hard Filtering ---
    if entropy > entropy_thresh:
        total_score = -float("inf")
    else:
        total_score = focus - (lambda_leak * leakage) - (lambda_ent * entropy) - size_penalty

    details = dict(
        focus=focus,
        leakage=leakage,
        entropy=entropy,
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
            focus=focus, leakage=0.0, entropy=0.0,
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
) -> None:
    """Save a distance×resolution attention map grid as an image.

    Args:
        grid_data: ``{multiplier: (rgb_cpu_3HW, {"8x8": A, ...}, edited_cpu or None)}``
        save_path: output image path.
        res_labels: resolution labels to include (default all four).
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

            # Entropy ratio label
            raw_ent = compute_entropy(A_norm)
            max_ent = math.log(A.numel()) + 1e-8
            ImageDraw.Draw(canvas).text(
                (label_w + ci * cell + 5, y + 5),
                f"H={raw_ent / max_ent:.3f}",
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
    latency_logger: Optional[LatencyLogger] = None,
) -> float:
    """
    SAGE-Probing: find the optimal viewing distance d* by rendering pilot
    views at several distances along v_front and scoring each with the
    Sharpness-Aware Guided Editability metric.

    Score = Focus - λ₁·Leakage - λ₂·Entropy(A) - Penalty_size
    Hard constraint: views with H(A) > τ_entropy are marked unsafe (-inf).

    Returns the optimal distance d*.
    """
    if distance_multipliers is None:
        distance_multipliers = [1.5, 2.0, 2.5, 3.0, 3.5]

    center = roi_info["center"]
    v_front = roi_info["v_front"]
    eigenvalues = roi_info["eigenvalues"]
    # r_obj = sqrt(lambda_max) — use the largest eigenvalue as object radius
    r_obj = float(np.sqrt(np.abs(eigenvalues[0])))
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    print(f"[Step2] r_obj (sqrt(lambda_max)) = {r_obj:.4f}")
    print(f"[Step2] Candidate distances: {[f'{m}x r_obj = {m * r_obj:.4f}' for m in distance_multipliers]}")

    best_d = distance_multipliers[1] * r_obj  # default fallback
    best_score = -float("inf")
    results = []

    # Collect data for attention grid visualisation
    attn_grid_data: Dict[float, Tuple[torch.Tensor, Dict[str, torch.Tensor]]] = {}

    for i, mult in enumerate(distance_multipliers):
        cand_prefix = f"step2.cand_{i}"
        d = mult * r_obj
        eye = center + v_front * d
        with latency_timeit(latency_logger, f"{cand_prefix}.make_camera", device):
            cam = _make_camera(eye, center, world_up, fovy, h, w, uid=i, device=device)

        with latency_timeit(latency_logger, f"{cand_prefix}.render", device):
            out = render(
                cam, gaussians, pipe_params, background, override_opacity=override_opacity
            )
            rgb = out["render"]  # (3, H, W)

        # Project ROI mask to 2D
        if roi_mask is not None:
            with latency_timeit(latency_logger, f"{cand_prefix}.project_roi_mask", device):
                mask_2d = _project_roi_mask(
                    gaussians, roi_mask, cam, pipe_params, background,
                    override_opacity=override_opacity,
                    device=device,
                )
        else:
            mask_2d = torch.ones(1, h, w, device=device)

        # Use a deterministic seed per candidate so that the SAGE
        # score entropy and the attention-grid entropy match.
        per_view_seed = 1337 + i
        per_view_seed = 5 # 5가 x4에서 잘됐음

        if ip2p_pipe is not None:
            # Single unified IP2P pass: SAGE score + all-res attention + edited image
            with latency_timeit(latency_logger, f"{cand_prefix}.ip2p_unified", device):
                score, details, all_res, edited_t = _run_ip2p_unified(
                    rgb, mask_2d, ip2p_pipe, edit_prompt,
                    lambda_leak=lambda_leak,
                    lambda_ent=lambda_ent,
                    entropy_thresh=entropy_thresh,
                    num_steps=ip2p_num_inference_steps,
                    seed=per_view_seed,
                    guidance_scale=ip2p_guidance_scale,
                    image_guidance_scale=ip2p_image_guidance_scale,
                    latency_logger=latency_logger,
                    device=device,
                )
            if save_attn_grid:
                attn_grid_data[mult] = (rgb.cpu(), all_res, edited_t)
        else:
            score, details = _compute_editability_score(
                rgb, mask_2d, None, edit_prompt,
                lambda_leak=lambda_leak,
                lambda_ent=lambda_ent,
                entropy_thresh=entropy_thresh,
                seed=per_view_seed,
            )
        results.append((mult, d, score, details))

        status = "UNSAFE" if score == -float("inf") else f"{score:.4f}"
        print(
            f"[Step2]  d={d:.4f} ({mult}x) | "
            f"focus={details['focus']:.3f}  leak={details['leakage']:.3f}  "
            f"entropy={details['entropy']:.3f}  occ={details['occupancy']:.3f}  "
            f"penalty={details['size_penalty']:.1f} | "
            f"S_total={status}"
        )

        if score > best_score:
            best_score = score
            best_d = d

    if best_score == -float("inf"):
        # All views were unsafe — fall back to median multiplier
        fallback_mult = distance_multipliers[len(distance_multipliers) // 2]
        best_d = fallback_mult * r_obj
        print(f"[Step2] WARNING: All views unsafe. Falling back to {fallback_mult}x → d*={best_d:.4f}")
    else:
        print(f"[Step2] Best distance d*={best_d:.4f} (mult={best_d / r_obj:.2f}x, score={best_score:.4f})")

    # Save attention grid
    if save_attn_grid and attn_grid_data:
        _save_attention_grid(attn_grid_data, save_attn_grid)

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
    device: str = "cuda",
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
    latency_logger: Optional[LatencyLogger] = None,
    device: str = "cuda",
) -> List[int]:
    """
    Select *n_select* cameras from the top-scoring pool using
    Farthest Point Sampling (angular distance on the view sphere).
    """
    with latency_timeit(latency_logger, "step5.diversity_selection.prepare_pool", device):
        # Filter to top fraction
        n_pool = max(int(len(scored) * top_fraction), n_select)
        pool = scored[:n_pool]  # already sorted descending by energy
        pool_indices = [s[0] for s in pool]
        pool_energies = {s[0]: s[1] for s in pool}

    with latency_timeit(latency_logger, "step5.diversity_selection.view_dirs", device):
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

    with latency_timeit(latency_logger, "step5.diversity_selection.fps_loop", device):
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
        "--distance_multipliers", type=str, default="1.5,2.0,2.5,3.0,3.5",
        help="Comma-separated distance multipliers for scale probing (× r_obj)",
    )

    # Scoring weights
    parser.add_argument("--w_vis", type=float, default=0.6, help="Visibility weight")
    parser.add_argument("--w_can", type=float, default=0.4, help="Canonical alignment weight")
    parser.add_argument("--top_fraction", type=float, default=0.20, help="Top fraction of candidates for FPS")

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
                roi_info = roi_intrinsic_analysis(gaussians, roi_mask, cam_forwards)
    else:
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
                    cone_half_angle_deg=90.0,
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
            cone_half_angle_deg=90.0,
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
