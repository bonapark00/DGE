from dataclasses import dataclass
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionInstructPix2PixPipeline
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm
import math
import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, parse_version
from threestudio.utils.typing import *


from threestudio.utils.dge_utils import register_pivotal, register_store_kf_attn_output, register_batch_idx, register_cams, register_epipolar_constrains, register_extended_attention, register_normal_attention, register_extended_attention, make_dge_block, isinstance_str, compute_epipolar_constrains, register_normal_attn_flag, save_epipolar_constraints_image, register_gp_cache, unregister_gp_cache, build_gaussian_provenance_cache, register_anchor_3d_cache, unregister_anchor_3d_cache, set_unet_latency_prefix, register_latency_logger
from collections import defaultdict
from contextlib import nullcontext
from typing import List, Optional, Dict, Any, Tuple
import random


def _get_valid_token_indices(pipe, prompt: str) -> List[int]:
    """Valid content token indices (vcedit-style): skip BOS, EOS, padding."""
    tokenizer = pipe.tokenizer
    tokens = tokenizer(prompt, return_tensors="pt", padding=False)
    input_ids = tokens["input_ids"][0].tolist()
    bos_id = getattr(tokenizer, "bos_token_id", 49406)
    eos_id = getattr(tokenizer, "eos_token_id", 49407)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    indices = []
    for i, tid in enumerate(input_ids):
        if tid in (bos_id, eos_id, pad_id):
            continue
        indices.append(i)
    return indices if indices else list(range(1, min(len(input_ids) - 1, 10)))


class _FastAttnRenderer:
    """Pre-built rasterizer for repeated renders with different override colors.

    Avoids per-call overhead of screenspace_points allocation,
    rasterizer construction, and gaussian property fetching.
    """
    __slots__ = ('rasterizer', 'means3D', 'means2D', 'opacity',
                 'scales', 'rotations', 'cov3D_precomp')

    def __init__(self, cam, gaussian, gs_pipe, bg):
        from diff_gaussian_rasterization import (
            GaussianRasterizationSettings, GaussianRasterizer,
        )
        tanfovx = math.tan(cam.FoVx * 0.5)
        tanfovy = math.tan(cam.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(cam.image_height),
            image_width=int(cam.image_width),
            tanfovx=tanfovx, tanfovy=tanfovy,
            bg=bg, scale_modifier=1.0,
            viewmatrix=cam.world_view_transform,
            projmatrix=cam.full_proj_transform,
            sh_degree=gaussian.active_sh_degree,
            campos=cam.camera_center,
            prefiltered=False, debug=False,
        )
        self.rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        self.means3D = gaussian.get_xyz.float()
        self.means2D = torch.zeros_like(self.means3D)
        self.opacity = gaussian.get_opacity.float()
        if gs_pipe.compute_cov3D_python:
            self.cov3D_precomp = gaussian.get_covariance(1.0)
            self.scales = None
            self.rotations = None
        else:
            self.cov3D_precomp = None
            self.scales = gaussian.get_scaling.float()
            self.rotations = gaussian.get_rotation.float()

    def render_color(self, color: torch.Tensor) -> torch.Tensor:
        """Render with override_color [N, 3], returns [3, H, W]."""
        rendered, _, _ = self.rasterizer(
            means3D=self.means3D,
            means2D=self.means2D,
            shs=None,
            colors_precomp=color,
            opacities=self.opacity,
            scales=self.scales,
            rotations=self.rotations,
            cov3D_precomp=self.cov3D_precomp,
        )
        return rendered


def _per_step_build_consistent_maps(
    store_proc,
    n_source: int,
    source_cams: list,
    target_indices: List[int],
    all_cams: list,
    gaussian,
    gs_pipe,
    device,
    attn_len: int,
    target_resolutions: Tuple[int, ...] = (32 * 32, 64 * 64),
    latency_logger=None,
    latency_prefix: str = "",
) -> Tuple[List[int], Dict[int, Dict[int, torch.Tensor]]]:
    """Build consistent cross-attn maps from pivotal-forward stored attention.

    Inverse-renders attention maps from source (pivotal) cameras to 3D via
    Gaussian splatting, then re-renders to every target view.

    Returns:
        collected_resolutions: list of spatial-resolution keys (e.g. 1024, 4096).
        M_con_by_view_res: {global_view_idx: {res: Tensor[H, W, attn_len]}}.
    """
    _lp = latency_prefix

    def _cam_at_res(cam, h, w):
        if hasattr(cam, "HW_scale"):
            return cam.HW_scale(h, w)
        from gaussiansplatting.scene.cameras import MiniCam
        return MiniCam(
            w, h, cam.FoVy, cam.FoVx,
            getattr(cam, "znear", 0.01), getattr(cam, "zfar", 100.0),
            cam.world_view_transform, cam.full_proj_transform,
        )

    with latency_logger.timeit(f"{_lp}.extract_maps") if latency_logger else nullcontext():
        _allowed = set(target_resolutions)
        key_attn_by_res: Dict[int, torch.Tensor] = {}
        for res, map_list in store_proc.maps.items():
            if not map_list or res not in _allowed:
                continue
            stacked = torch.stack(map_list, dim=0)
            if stacked.ndim == 4:
                L, batch_heads, Hw, C = stacked.shape
                num_heads = batch_heads // (n_source * 3)
                stacked = stacked.view(L, n_source * 3, num_heads, Hw, C).mean(dim=(0, 2))
            else:
                stacked = stacked.mean(dim=(0, 2))
            if stacked.shape[0] >= 3 * n_source:
                stacked = stacked[:n_source]
            key_attn_by_res[res] = stacked.float().to(device)

    if not key_attn_by_res:
        return [], {}

    # Aggregate token channels → 1 for faster 3D rendering (expand back after)
    for _res in key_attn_by_res:
        key_attn_by_res[_res] = key_attn_by_res[_res].mean(dim=-1, keepdim=True)
    _render_attn_len = 1

    collected_res = list(key_attn_by_res.keys())
    N = gaussian.get_xyz.shape[0]
    bg = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)

    with latency_logger.timeit(f"{_lp}.inverse_render_2d_to_3d") if latency_logger else nullcontext():
        M_3d_by_res: Dict[int, torch.Tensor] = {}
        for res in key_attn_by_res:
            side = int(res ** 0.5)
            weights = torch.zeros(N, _render_attn_len, device=device, dtype=torch.float32)
            weights_cnt = torch.zeros(N, device=device, dtype=torch.int32)
            for v_idx, cam in enumerate(source_cams):
                cam_low = _cam_at_res(cam, side, side)
                M_v = key_attn_by_res[res][v_idx]
                if M_v.dim() == 2:
                    M_v = M_v.view(side, side, _render_attn_len)
                for c in range(_render_attn_len):
                    img_w = M_v[:, :, c].unsqueeze(0)
                    gaussian.apply_weights(cam_low, weights[:, c:c + 1], weights_cnt, img_w)
            M_3d_by_res[res] = weights / (weights_cnt.unsqueeze(1).float().clamp(min=1) + 1e-7)

    with latency_logger.timeit(f"{_lp}.render_3d_to_2d") if latency_logger else nullcontext():
        M_con: Dict[int, Dict[int, torch.Tensor]] = {}
        for global_idx in target_indices:
            M_con[global_idx] = {}
            cam = all_cams[global_idx]
            for res in M_3d_by_res:
                side = int(res ** 0.5)
                M_3d = M_3d_by_res[res]
                cam_low = _cam_at_res(cam, side, side)
                renderer = _FastAttnRenderer(cam_low, gaussian, gs_pipe, bg)
                maps_c = []
                for c in range(_render_attn_len):
                    color_c = M_3d[:, c].unsqueeze(-1).expand(-1, 3)
                    maps_c.append(renderer.render_color(color_c)[0])
                stacked = torch.stack(maps_c, dim=-1)  # [H, W, 1]
                M_con[global_idx][res] = stacked.expand(-1, -1, attn_len).contiguous()

    return collected_res, M_con


def _build_target_batches(
    n_target: int,
    camera_batch_size: int,
    t_step_value: int,
    neighbor_threshold: int,
    late_mode: str,
    sliding_stride: int,
    step_index: int,
) -> List[List[int]]:
    """Build batch groups based on timestep (no pivotal selection).

    Early timesteps (t_step >= threshold): neighboring sequential batches.
    Late timesteps (t_step < threshold): random shuffle or timestep-shifted contiguous batches.
    """
    if t_step_value >= neighbor_threshold:
        batches: List[List[int]] = []
        for start in range(0, n_target, camera_batch_size):
            end = min(start + camera_batch_size, n_target)
            batches.append(list(range(start, end)))
    else:
        if late_mode == "sliding_window":
            stride = sliding_stride if sliding_stride > 0 else max(1, camera_batch_size // 2)
            offset = (step_index * stride) % n_target if n_target > 0 else 0
            indices = list(range(n_target))
            indices = indices[offset:] + indices[:offset]
            batches = []
            for start in range(0, n_target, camera_batch_size):
                end = min(start + camera_batch_size, n_target)
                batches.append(indices[start:end])
        else:  # random
            all_indices = list(range(n_target))
            random.shuffle(all_indices)
            batches = []
            for start in range(0, n_target, camera_batch_size):
                end = min(start + camera_batch_size, n_target)
                batches.append(all_indices[start:end])
    return batches


def _select_pivotals_for_batches(
    batches: List[List[int]],
    canonical_scores: List[float],
    key_count: List[int],
    is_early: bool,
) -> List[int]:
    """Select key/pivotal view per batch using canonical-first + progressive coverage.

    Early (is_early=True): pick the most canonical (frontal) view per batch.
    Late (is_early=False): pick the least-used view (progressive coverage),
        breaking ties by highest canonical score.
    """
    pivotal_per_batch = []
    for batch in batches:
        if is_early:
            best = max(batch, key=lambda i: canonical_scores[i])
        else:
            min_count = min(key_count[i] for i in batch)
            candidates = [i for i in batch if key_count[i] == min_count]
            best = max(candidates, key=lambda i: canonical_scores[i])
        pivotal_per_batch.append(best)
        key_count[best] += 1
    return pivotal_per_batch


class CrossAttentionStoreProcessor:
    """Stores cross-attention probs at any spatial resolution (H*W) where cross-attn runs, valid tokens only."""

    def __init__(self, valid_token_indices: List[int], target_resolutions: Tuple[int, ...] = (32 * 32, 64 * 64)):
        self.valid_token_indices = valid_token_indices
        self.attn_len = len(valid_token_indices)
        self.target_resolutions = target_resolutions
        # Allow any resolution: store at every spatial size the UNet actually produces
        self.maps: Dict[int, List[torch.Tensor]] = defaultdict(list)

    def reset(self):
        self.maps.clear()

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
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
        seq_spatial = hidden_states.shape[1]
        _should_store = seq_spatial in self.target_resolutions
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
        if _should_store:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            probs = attention_probs.detach()
            valid = [i for i in self.valid_token_indices if i < probs.shape[-1]]
            if valid:
                self.maps[seq_spatial].append(probs[:, :, valid])
            hidden_states = torch.bmm(attention_probs, value)
        else:
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


class ConsistentCrossAttnProcessor:
    """Uses precomputed consistent cross-attention map when set on attn module.

    The consistent_map has shape [n_views, Hw, attn_len] (one map per view).
    During the UNet forward with CFG the batch dim is n_views * 3 (text / image / uncond).
    We repeat the map for the 3 CFG conditions and expand over attention heads so that
    the bmm with the head-expanded value tensor works correctly.
    """

    def __init__(self, backup_processor=None):
        self.backup_processor = backup_processor

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        data = getattr(attn, "_consistent_attn_map_current", None)
        if data is None:
            if self.backup_processor is not None:
                return self.backup_processor(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
            return self._default_forward(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        elif input_ndim != 3:
            if self.backup_processor is not None:
                return self.backup_processor(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
            return self._default_forward(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        batch_size, sequence_length, inner_dim = hidden_states.shape
        attn_len = data.get("attn_len")
        consistent_map = data.get(sequence_length)
        if attn_len is None or consistent_map is None:
            if self.backup_processor is not None:
                return self.backup_processor(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
            return self._default_forward(attn, hidden_states, encoder_hidden_states, attention_mask, temb)

        n_map = consistent_map.shape[0]
        # Expand map for CFG: [n_views] -> [n_views * 3]
        if n_map == batch_size:
            cmap = consistent_map
        elif n_map * 3 == batch_size:
            cmap = consistent_map.repeat(3, 1, 1)
            if not getattr(self, "_logged_cfg_expand", False):
                print(f"[ConsistentCrossAttn] APPLIED: n_map={n_map} -> batch={batch_size} (CFG 3x), seq={sequence_length}, attn_len={attn_len}, heads={attn.heads}")
                self._logged_cfg_expand = True
        else:
            if self.backup_processor is not None:
                return self.backup_processor(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
            return self._default_forward(attn, hidden_states, encoder_hidden_states, attention_mask, temb)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        if encoder_hidden_states is not None and attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        value = attn.head_to_batch_dim(value)                    # [B*H, seq, head_dim]
        value_valid = value[:, :attn_len]                        # [B*H, attn_len, head_dim]

        # Expand map for attention heads: [B, Hw, attn_len] -> [B*H, Hw, attn_len]
        num_heads = attn.heads
        cmap = cmap.to(value.device).to(value.dtype)
        cmap = cmap.unsqueeze(1).expand(-1, num_heads, -1, -1)   # [B, H, Hw, attn_len]
        cmap = cmap.reshape(batch_size * num_heads, sequence_length, attn_len)

        hidden_states = torch.bmm(cmap, value_valid)             # [B*H, Hw, head_dim]
        hidden_states = attn.batch_to_head_dim(hidden_states)    # [B, Hw, inner_dim]
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, inner_dim, int(sequence_length ** 0.5), int(sequence_length ** 0.5))
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

    def _default_forward(self, attn, hidden_states, encoder_hidden_states, attention_mask, temb):
        if getattr(self, "backup_processor", None) is not None:
            return self.backup_processor(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        raise RuntimeError("ConsistentCrossAttnProcessor has no backup and no consistent map set")


# Latency timeit hierarchy: single source of truth (no overlap with DGE.edit_multiview.*)
EDIT_MULTIVIEW_PREFIX = "edit_multiview.guidance_batch"
EDIT_ALL_VIEW_PREFIX = "edit_all_view.guidance_batch"
#
# Hierarchy when use_multiview_path (DGE calls guidance with use_multiview=True):
#   edit_multiview                          (DGE: whole edit_multiview(); summary shows this name)
#   └─ guidance_batch                       (DGE: self.guidance() call; full name edit_multiview.guidance_batch)
#       ├─ encode_images                    (__call__)
#       ├─ encode_cond_images
#       ├─ text_embeddings
#       ├─ edit_latents_multiview           (__call__ wraps edit_latents_multiview(); children below)
#       │   ├─ setup
#       │   ├─ valid_token_indices
#       │   ├─ install_store_processor
#       │   ├─ key_view_denoise_loop
#       │   │   ├─ init
#       │   │   ├─ per_step_setup
#       │   │   ├─ forward_unet
#       │   │   ├─ guidance_and_step
#       │   │   └─ finalize
#       │   ├─ restore_attn2_processors
#       │   ├─ build_key_cross_attn_by_res
#       │   ├─ inverse_render_2d_to_3d
#       │   ├─ render_consistent_maps
#       │   ├─ install_consistent_processor
#       │   ├─ noise_and_init_latents
#       │   ├─ target_denoise_loop
#       │   │   ├─ per_timestep_setup
#       │   │   ├─ pivotal_forward
#       │   │   ├─ batch_prep
#       │   │   ├─ batch_forward
#       │   │   └─ merge_and_step
#       │   └─ restore_attn2_final
#       └─ decode_latents                   (__call__)
#
# Hierarchy when not multiview (edit_all_view path): EDIT_ALL_VIEW_PREFIX.* and edit_latents.*


@threestudio.register("dge-guidance")
class DGEGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        cache_dir: Optional[str] = None
        ddim_scheduler_name_or_path: str = "CompVis/stable-diffusion-v1-4"
        ip2p_name_or_path: str = "timbrooks/instruct-pix2pix"

        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 7.5
        condition_scale: float = 1.5
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True
        fixed_size: int = -1

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        diffusion_steps: int = 20
        use_sds: bool = False
        use_sds_dge: bool = False  # True: DGE SDS (epipolar, pivotal); False: vanilla SDS
        camera_batch_size: int = 5
        # --- Adaptive batching for target denoise loop ---
        # "fixed": sequential chunks (original), "adaptive": timestep-dependent grouping
        target_batch_strategy: str = "fixed"
        # t_step >= threshold -> neighbor batches; t_step < threshold -> late_mode batches
        target_batch_neighbor_threshold: int = 500
        # Late-timestep mode: "random" (shuffled non-overlapping) or "sliding_window" (overlapping)
        target_batch_late_mode: str = "sliding_window"
        # Stride for sliding_window; -1 = camera_batch_size // 2
        target_batch_sliding_stride: int = -1
        # Key selection per batch: "random", "canonical_progressive", or "fixed"
        # canonical_progressive: early=most-frontal key, late=least-used key (progressive coverage)
        # fixed: deterministic local index per batch (e.g. first view in each batch)
        target_key_selection_mode: str = "random"
        edit_view_selection_strategy: str = ""
        # Feature injection in edit_latents_multiview: "similarity" (cosine + gather) or "3d_anchor" (3DGS-based canonical tokens).
        # Implementation also supports "none" (no cross-view feature injection, only extended/consistent attention).
        feature_injection_mode: str = "similarity"
        # For 3d_anchor: blend h_out = (1 - injection_lambda) * h_sa + injection_lambda * F(v,p). h_sa uses current hidden_states.
        injection_lambda: float = 0.5
        # 3d_anchor only: "blend" = λ*(F-h)+h; "gather" = similarity-like: pivot self-attn + 3D-GS remap gather + residual (no λ).
        injection_3d_anchor_style: str = "blend"
        # Target-view denoise loop: if True, use extended attention after warmup (default behavior);
        # if False, always use normal self-attention for target loop.
        target_use_extended_attention: bool = True
        # Per-step cross-attn consistency: at each early denoising step in the target loop,
        # collect cross-attn from the pivotal forward, inverse-render to 3D, re-render to
        # all target views, and replace attn2 output with the consistent map.
        # Skips the Phase-1 (between-phases) cross-attn pipeline when True.
        per_step_cross_attn_consistency: bool = False
        # Apply per-step consistency only when t_step >= this value (high = early denoising).
        per_step_cross_attn_t_start: int = 500
        # Spatial resolutions (H*W) whose pivotal cross-attention maps are explicitly stored
        # and passed through the 3D consistency pipeline. Non-listed resolutions use the
        # fast fused attention path only.
        per_step_cross_attn_resolutions: Tuple[int, ...] = (32 * 32, 64 * 64)

    def configure(self, preloaded_pipe=None, **kwargs) -> None:
        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
            "cache_dir": self.cfg.cache_dir,
        }

        if preloaded_pipe is not None:
            threestudio.info("Reusing shared InstructPix2Pix pipeline (from lens/dm)")
            self.pipe = preloaded_pipe
            if self.pipe.device != self.device:
                self.pipe = self.pipe.to(self.device)
        else:
            threestudio.info("Loading InstructPix2Pix ...")
            self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                self.cfg.ip2p_name_or_path, **pipe_kwargs
            ).to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.ddim_scheduler_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
            cache_dir=self.cfg.cache_dir,
        )
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None

        threestudio.info(f"Loaded InstructPix2Pix!")
        for _, module in self.unet.named_modules():
            if isinstance_str(module, "BasicTransformerBlock"):
                make_block_fn = make_dge_block 
                module.__class__ = make_block_fn(module.__class__)
                # Something needed for older versions of diffusers
                if not hasattr(module, "use_ada_layer_norm_zero"):
                    module.use_ada_layer_norm = False
                    module.use_ada_layer_norm_zero = False
        register_extended_attention(self)
        # Required for both edit_latents and compute_grad_sds; edit_latents overrides as needed
        register_normal_attn_flag(self.unet, False)

    
    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        cross_attention_kwargs: Optional[dict] = None,
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        kwargs = {}
        if cross_attention_kwargs is not None:
            kwargs["cross_attention_kwargs"] = cross_attention_kwargs
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            **kwargs,
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 H W"]
    ) -> Float[Tensor, "B 4 DH DW"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_cond_images(
        self, imgs: Float[Tensor, "B 3 H W"]
    ) -> Float[Tensor, "B 4 DH DW"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.mode()
        uncond_image_latents = torch.zeros_like(latents)
        latents = torch.cat([latents, latents, uncond_image_latents], dim=0)
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self, latents: Float[Tensor, "B 4 DH DW"]
    ) -> Float[Tensor, "B 3 H W"]:
        input_dtype = latents.dtype
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def use_normal_unet(self):
        # print("use normal unet")
        register_normal_attention(self)
        register_normal_attn_flag(self.unet, True)

    def edit_latents(
        self,
        text_embeddings: Float[Tensor, "BB 77 768"],
        latents: Float[Tensor, "B 4 DH DW"],
        image_cond_latents: Float[Tensor, "B 4 DH DW"],
        t: Int[Tensor, "B"],
        cams=None,
        latency_logger=None,
        gp_cache=None,
        key_cam_indices=None,
        latency_prefix: Optional[str] = None,
    ) -> Float[Tensor, "B 4 DH DW"]:

        self.scheduler.config.num_train_timesteps = t.item() if len(t.shape) < 1 else t[0].item()
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)

        current_H = image_cond_latents.shape[2]
        current_W = image_cond_latents.shape[3]

        camera_batch_size = self.cfg.camera_batch_size
        print("Start editing images...")

        # Base prefix for non-multiview edit_latents path; overridden by caller
        _base = latency_prefix or EDIT_ALL_VIEW_PREFIX
        _p = f"{_base}.edit_latents"

        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents = self.scheduler.add_noise(latents, noise, t)

            # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
            positive_text_embedding, negative_text_embedding, _ = text_embeddings.chunk(3)
            split_image_cond_latents, _, zero_image_cond_latents = image_cond_latents.chunk(3)

            with latency_logger.timeit(f"{_p}.diffusion_loop") if latency_logger else nullcontext():
                for t in self.scheduler.timesteps:
                    with latency_logger.timeit(f"{_p}.diffusion_loop.timestep_setup") if latency_logger else nullcontext():
                        if t < 100:
                            self.use_normal_unet()
                        else:
                            register_normal_attn_flag(self.unet, False)

                    with torch.no_grad():
                        # pred noise
                        noise_pred_text = []
                        noise_pred_image = []
                        noise_pred_uncond = []

                        with latency_logger.timeit(f"{_p}.diffusion_loop.pivotal_setup") if latency_logger else nullcontext():
                            if self.cfg.edit_view_selection_strategy == "manual-20":
                                pivotal_idx = torch.tensor([2, 7, 12, 17])
                            elif self.cfg.edit_view_selection_strategy == "manual-15":
                                pivotal_idx = torch.tensor([2, 7, 12])
                            else:
                                pivotal_idx = torch.randint(
                                    camera_batch_size, (len(latents) // camera_batch_size,)
                                ) + torch.arange(0, len(latents), camera_batch_size)
                            register_pivotal(self.unet, True)

                            key_cams = [cams[cam_pivotal_idx] for cam_pivotal_idx in pivotal_idx.tolist()]
                            latent_model_input = torch.cat([latents[pivotal_idx]] * 3)
                            pivot_text_embeddings = torch.cat(
                                [
                                    positive_text_embedding[pivotal_idx],
                                    negative_text_embedding[pivotal_idx],
                                    negative_text_embedding[pivotal_idx],
                                ],
                                dim=0,
                            )
                            pivot_image_cond_latetns = torch.cat(
                                [
                                    split_image_cond_latents[pivotal_idx],
                                    split_image_cond_latents[pivotal_idx],
                                    zero_image_cond_latents[pivotal_idx],
                                ],
                                dim=0,
                            )
                            latent_model_input = torch.cat([latent_model_input, pivot_image_cond_latetns], dim=1)

                        with latency_logger.timeit(f"{_p}.diffusion_loop.pivotal_forward") if latency_logger else nullcontext():
                            self.forward_unet(latent_model_input, t, encoder_hidden_states=pivot_text_embeddings)
                            register_pivotal(self.unet, False)

                        with latency_logger.timeit(f"{_p}.diffusion_loop.batch_processing") if latency_logger else nullcontext():
                            for i, b in enumerate(range(0, len(latents), camera_batch_size)):
                                with latency_logger.timeit(
                                    f"{_p}.diffusion_loop.batch_processing.register_ops"
                                ) if latency_logger else nullcontext():
                                    with latency_logger.timeit(
                                        f"{_p}.diffusion_loop.batch_processing.register_ops.register_batch_idx"
                                    ) if latency_logger else nullcontext():
                                        register_batch_idx(self.unet, i)

                                    with latency_logger.timeit(
                                        f"{_p}.diffusion_loop.batch_processing.register_ops.register_cams"
                                    ) if latency_logger else nullcontext():
                                        register_cams(
                                            self.unet,
                                            cams[b : b + camera_batch_size],
                                            pivotal_idx[i] % camera_batch_size,
                                            key_cams,
                                        )

                                    with latency_logger.timeit(
                                        f"{_p}.diffusion_loop.batch_processing.register_ops.compute_epipolar_constrains"
                                    ) if latency_logger else nullcontext():
                                        if gp_cache is not None:
                                            # Version B: skip dense epipolar computation entirely.
                                            # Register the stacked gaussian-provenance cache for all
                                            # cameras in this batch (indexed b .. b+camera_batch_size-1).
                                            batch_end = min(b + camera_batch_size, len(cams))
                                            register_gp_cache(self.unet, gp_cache, b, batch_end)
                                        else:
                                            epipolar_constrains = {}
                                            # Create directory for saving epipolar constraint images in save_dir
                                            epipolar_images_dir = os.path.join(
                                                self.save_dir, "epipolar_constraints_images"
                                            )

                                            # Warmup: run first epipolar compute once to avoid cam_0 including CUDA init time
                                            if torch.cuda.is_available() and key_cams:
                                                _ = compute_epipolar_constrains(
                                                    key_cams[0],
                                                    cams[b],
                                                    current_H=current_H // 1,
                                                    current_W=current_W // 1,
                                                    downsample_factor=1,
                                                )
                                                torch.cuda.synchronize()

                                            for down_sample_factor in [1, 2, 4, 8]:
                                                with latency_logger.timeit(
                                                    f"{_p}.diffusion_loop.batch_processing.register_ops.compute_epipolar_constrains.downsample_{down_sample_factor}"
                                                ) if latency_logger else nullcontext():
                                                    H = current_H // down_sample_factor
                                                    W = current_W // down_sample_factor
                                                    epipolar_constrains[H * W] = []
                                                    for cam_idx, cam in enumerate(
                                                        cams[b : b + camera_batch_size]
                                                    ):
                                                        with latency_logger.timeit(
                                                            f"{_p}.diffusion_loop.batch_processing.register_ops.compute_epipolar_constrains.downsample_{down_sample_factor}.cam_{cam_idx}"
                                                        ) if latency_logger else nullcontext():
                                                            cam_epipolar_constrains = []
                                                            for key_cam_idx, key_cam in enumerate(key_cams):
                                                                # Pass downsample_factor to the function
                                                                epipolar_constraint = compute_epipolar_constrains(
                                                                    key_cam,
                                                                    cam,
                                                                    current_H=H,
                                                                    current_W=W,
                                                                    downsample_factor=down_sample_factor,
                                                                )
                                                                cam_epipolar_constrains.append(epipolar_constraint)

                                                                ## Save epipolar constraints as image for visualization
                                                                # save_epipolar_constraints_image(
                                                                #     epipolar_constraint,
                                                                #     H, W,
                                                                #     epipolar_images_dir,
                                                                #     cam_idx,
                                                                #     key_cam_idx,
                                                                #     down_sample_factor
                                                                # )
                                                            epipolar_constrains[H * W].append(
                                                                torch.stack(cam_epipolar_constrains, dim=0)
                                                            )
                                                    epipolar_constrains[H * W] = torch.stack(
                                                        epipolar_constrains[H * W], dim=0
                                                    )

                                            with latency_logger.timeit(
                                                f"{_p}.diffusion_loop.batch_processing.register_ops.register_epipolar_constrains"
                                            ) if latency_logger else nullcontext():
                                                register_epipolar_constrains(self.unet, epipolar_constrains)

                                with latency_logger.timeit(
                                    f"{_p}.diffusion_loop.batch_processing.prepare_input"
                                ) if latency_logger else nullcontext():
                                    batch_model_input = torch.cat(
                                        [latents[b : b + camera_batch_size]] * 3
                                    )
                                    batch_text_embeddings = torch.cat(
                                        [
                                            positive_text_embedding[b : b + camera_batch_size],
                                            negative_text_embedding[b : b + camera_batch_size],
                                            negative_text_embedding[b : b + camera_batch_size],
                                        ],
                                        dim=0,
                                    )
                                    batch_image_cond_latents = torch.cat(
                                        [
                                            split_image_cond_latents[b : b + camera_batch_size],
                                            split_image_cond_latents[b : b + camera_batch_size],
                                            zero_image_cond_latents[b : b + camera_batch_size],
                                        ],
                                        dim=0,
                                    )
                                    batch_model_input = torch.cat(
                                        [batch_model_input, batch_image_cond_latents], dim=1
                                    )

                                with latency_logger.timeit(
                                    f"{_p}.diffusion_loop.batch_processing.unet_forward"
                                ) if latency_logger else nullcontext():
                                    batch_noise_pred = self.forward_unet(
                                        batch_model_input, t, encoder_hidden_states=batch_text_embeddings
                                    )
                                    (
                                        batch_noise_pred_text,
                                        batch_noise_pred_image,
                                        batch_noise_pred_uncond,
                                    ) = batch_noise_pred.chunk(3)
                                    noise_pred_text.append(batch_noise_pred_text)
                                    noise_pred_image.append(batch_noise_pred_image)
                                    noise_pred_uncond.append(batch_noise_pred_uncond)

                        with latency_logger.timeit(
                            f"{_p}.diffusion_loop.concat_outputs"
                        ) if latency_logger else nullcontext():
                            noise_pred_text = torch.cat(noise_pred_text, dim=0)
                            noise_pred_image = torch.cat(noise_pred_image, dim=0)
                            noise_pred_uncond = torch.cat(noise_pred_uncond, dim=0)

                        with latency_logger.timeit(
                            f"{_p}.diffusion_loop.guidance_calc"
                        ) if latency_logger else nullcontext():
                            # perform classifier-free guidance
                            noise_pred = (
                                noise_pred_uncond
                                + self.cfg.guidance_scale * (noise_pred_text - noise_pred_image)
                                + self.cfg.condition_scale * (noise_pred_image - noise_pred_uncond)
                            )

                        with latency_logger.timeit(
                            f"{_p}.diffusion_loop.scheduler_step"
                        ) if latency_logger else nullcontext():
                            # get previous sample, continue loop
                            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        print("Editing finished.")
        return latents

    def edit_latents_multiview(
        self,
        text_embeddings: Float[Tensor, "BB 77 768"],
        latents: Float[Tensor, "B 4 DH DW"],
        image_cond_latents: Float[Tensor, "B 4 DH DW"],
        t: Int[Tensor, "B"],
        cams: list,
        gaussian=None,
        pipe=None,
        prompt_text: str = "",
        latency_logger=None,
        # key_view_camera_ids: Optional[List[int]] = None,
        feature_injection_mode: Optional[str] = None,
        injection_lambda: Optional[float] = None,
        injection_3d_anchor_style: Optional[str] = None,
        key_selection_strategy: Optional[str] = None,
        num_key_views: Optional[int] = None,
        latency_prefix: Optional[str] = None,
    ) -> Float[Tensor, "B 4 DH DW"]:
        """
        Multiview edit: all views are denoised with optional extended self-attention (attn1)
        and optional per-step cross-attention consistency (attn2).
        """
        # Base prefix for multiview path; overridden by caller so it nests under training_step_all.*
        _base = latency_prefix or EDIT_MULTIVIEW_PREFIX
        _p = f"{_base}.edit_latents_multiview"  # so summary shows guidance_batch -> edit_latents_multiview -> setup, key_view_denoise_loop, ...
        _feature_injection_mode = feature_injection_mode if feature_injection_mode is not None else self.cfg.feature_injection_mode
        _injection_lambda = injection_lambda if injection_lambda is not None else self.cfg.injection_lambda
        _injection_3d_anchor_style = injection_3d_anchor_style if injection_3d_anchor_style is not None else self.cfg.injection_3d_anchor_style
        _per_step_mode = bool(getattr(self.cfg, "per_step_cross_attn_consistency", False))
        _per_step_t_start = int(getattr(self.cfg, "per_step_cross_attn_t_start", 500))
        with latency_logger.timeit(f"{_p}.setup") if latency_logger else nullcontext():
            self.scheduler.config.num_train_timesteps = t.item() if len(t.shape) < 1 else t[0].item()
            self.scheduler.set_timesteps(self.cfg.diffusion_steps)
            current_H = image_cond_latents.shape[2]
            current_W = image_cond_latents.shape[3]
            camera_batch_size = self.cfg.camera_batch_size
            device = latents.device
            n_views = latents.shape[0]
            # Key indices: from strategy when provided (uniform / uniform_random / lens_fps), else use passed key_indices (e.g. manual)
            if (
                key_selection_strategy is not None
                and key_selection_strategy not in ("manual",)
                and num_key_views is not None
            ):
                _n_key = min(num_key_views, n_views)
                if key_selection_strategy == "lens_fps" and gaussian is not None:
                    from threestudio.data.gs_load import select_key_views_by_lens_fps
                    key_indices = select_key_views_by_lens_fps(
                        gaussian, cams, n_key=_n_key,
                        top_fraction=0.20, w_vis=0.6, w_can=0.4, device=str(device),
                    )
                    key_indices = [int(i) for i in key_indices]
                elif key_selection_strategy == "uniform_random":
                    segment_size = n_views / _n_key
                    key_indices = []
                    for i in range(_n_key):
                        start = int(i * segment_size)
                        end = min(int((i + 1) * segment_size), n_views) - 1
                        if end < start:
                            end = start
                        key_indices.append(random.randint(start, end))
                    key_indices = sorted(key_indices)
                    key_indices = [int(i) for i in key_indices]
                elif key_selection_strategy == "manual":
                    pass
                else:
                    key_indices = torch.linspace(0, n_views - 1, _n_key, dtype=torch.long, device=device).tolist()
                    key_indices = [int(i) for i in key_indices]
            elif key_indices is None or len(key_indices) == 0:
                raise ValueError("edit_latents_multiview requires key_indices or (key_selection_strategy and num_key_views)")


            target_resolutions = tuple(
                getattr(
                    self.cfg,
                    "per_step_cross_attn_resolutions",
                    (32 * 32, 64 * 64),
                )
            )
            # Cache module lists once to avoid repeated named_modules() traversal in the loop
            _dge_blocks = [(n, m) for n, m in self.unet.named_modules()
                           if isinstance_str(m, "BasicTransformerBlock")]
            _attn2_modules = [(n, m) for n, m in self.unet.named_modules()
                              if n.endswith(".attn2") and hasattr(m, "processor")]
            # Propagate feature-injection disable flag into DGE blocks when requested.
            _disable_injection = (_feature_injection_mode == "none")
            for _, mod in _dge_blocks:
                setattr(mod, "disable_feature_injection", _disable_injection)

        with latency_logger.timeit(f"{_p}.valid_token_indices") if latency_logger else nullcontext():
            valid_indices = _get_valid_token_indices(self.pipe, prompt_text)
        attn_len = len(valid_indices)
        if attn_len == 0:
            return self.edit_latents(
                text_embeddings,
                latents,
                image_cond_latents,
                t,
                cams,
                latency_logger=latency_logger,
                latency_prefix=_base,
            )

        original_attn2_processors = {}

        positive_text_embedding, negative_text_embedding, _ = text_embeddings.chunk(3)
        split_image_cond_latents, _, zero_image_cond_latents = image_cond_latents.chunk(3)
        
        
        
        key_cams = [cams[i] for i in key_indices]
        n_key = len(key_indices)
        # if key_view_camera_ids is not None:
        #     print(f"[edit_latents_multiview] key view indices (sorted pos): {key_indices}, actual camera IDs: {key_view_camera_ids}")
        # else:
        #     print(f"[edit_latents_multiview] key view indices: {key_indices}")

        ## 2.2에서 cross attention map 수집에 쓰이는 view 개수 20개로 늘리고 싶으면 이렇게 하면 됨!
        # key_indices = [_ for _ in range(20)]
        # key_cams = [cams[_] for _ in key_indices]
        # n_key = len(key_indices)
        print(f"[edit_latents_multiview.key_denoise_loop] key view indices: {key_indices}, actual camera IDs: {[getattr(cam, 'id', None) for cam in key_cams]}")
        print(f"[edit_latents_multiview.key_denoise_loop] number of key views: {n_key}")


        # Phase 1 (key denoise loop) removed — set up UNet state for target loop directly.
        register_store_kf_attn_output(self.unet, True)

        # Key indices: from strategy when provided (uniform / uniform_random / lens_fps), else use passed key_indices (e.g. manual)
        if (
            key_selection_strategy is not None
            and key_selection_strategy not in ("manual",)
            and num_key_views is not None
        ):
            _n_key = min(num_key_views, n_views)
            if key_selection_strategy == "lens_fps" and gaussian is not None:
                from threestudio.data.gs_load import select_key_views_by_lens_fps
                key_indices = select_key_views_by_lens_fps(
                    gaussian, cams, n_key=_n_key,
                    top_fraction=0.20, w_vis=0.6, w_can=0.4, device=str(device),
                )
                key_indices = [int(i) for i in key_indices]
            elif key_selection_strategy == "uniform_random":
                segment_size = n_views / _n_key
                key_indices = []
                for i in range(_n_key):
                    start = int(i * segment_size)
                    end = min(int((i + 1) * segment_size), n_views) - 1
                    if end < start:
                        end = start
                    key_indices.append(random.randint(start, end))
                key_indices = sorted(key_indices)
                key_indices = [int(i) for i in key_indices]
            elif key_selection_strategy == "manual":
                pass
            else:
                key_indices = torch.linspace(0, n_views - 1, _n_key, dtype=torch.long, device=device).tolist()
                key_indices = [int(i) for i in key_indices]
        elif key_indices is None or len(key_indices) == 0:
            raise ValueError("edit_latents_multiview requires key_indices or (key_selection_strategy and num_key_views)")

        print(f"[edit_latents_multiview.target_denoise_loop] key view indices: {key_indices}, actual camera IDs: {key_cams} number of key views: {n_key}")
        # print(f"[edit_latents_multiview.target_denoise_loop] number of target views: {n_target}")

        # ------------------------------------------------------------------
        # Phase 2. Target-view preparation (after key_denoise_loop is done)
        #   - install ConsistentCrossAttnProcessor on attn2 blocks
        #   - add noise to latents and (optionally) fix key-view latents
        #   - decide target view indices and gather their conditions
        #   - (optionally) build 3D-anchor cache for "3d_anchor" mode
        # ------------------------------------------------------------------
        # Install ConsistentCrossAttnProcessor only when per-step cross-attn consistency is active.
        # When disabled (pure per-view IP2P or extended-attn only), keep original attn2 processors.
        if _per_step_mode:
            with latency_logger.timeit(f"{_p}.install_consistent_processor") if latency_logger else nullcontext():
                for name, mod in self.unet.named_modules():
                    if name.endswith(".attn2") and hasattr(mod, "processor"):
                        original_attn2_processors[name] = mod.processor
                        mod.processor = ConsistentCrossAttnProcessor(backup_processor=mod.processor)

        with latency_logger.timeit(f"{_p}.noise_and_init_latents") if latency_logger else nullcontext():
            noise = torch.randn_like(latents)
            latents = self.scheduler.add_noise(latents, noise, t)
            positive_text_embedding, negative_text_embedding, _ = text_embeddings.chunk(3)
            split_image_cond_latents, _, zero_image_cond_latents = image_cond_latents.chunk(3)
            # Denoise all views (no Phase 1 key-view freezing)
            target_indices = list(range(n_views))
            n_target = len(target_indices)
            target_cams = [cams[i] for i in target_indices]
            target_split_cond = split_image_cond_latents[target_indices]
            target_zero_cond = zero_image_cond_latents[target_indices]
            target_pos_emb = positive_text_embedding[target_indices]
            target_neg_emb = negative_text_embedding[target_indices]

        # Build 3D-anchor cache once when using 3d_anchor feature injection.
        # This also happens after key_denoise_loop and before target_denoise_loop.
        # Build 3D-anchor cache once when using 3d_anchor feature injection
        anchor_3d_cache = None
        if _feature_injection_mode == "3d_anchor":
            with latency_logger.timeit(f"{_p}.build_anchor_3d_cache") if latency_logger else nullcontext():
                scales_anchor = [(int(r ** 0.5), int(r ** 0.5)) for r in target_resolutions]
                gp_full = build_gaussian_provenance_cache(
                    gaussian, cams, key_indices, scales_anchor,
                    K=2, M_half=1, vis_eps=0.05, alpha_tau=0.4,
                )
                anchor_3d_cache = {
                    "pix2g_id": gp_full["pix2g_id"],
                    "pix2g_w": gp_full["pix2g_w"],
                    "g2uv": gp_full["g2uv"],
                    "g_vis": gp_full["g_vis"],
                    "g2uv_all": gp_full.get("g2uv_all", {}),
                }

        # ------------------------------------------------------------------
        # Precompute canonical scores for canonical_progressive key selection.
        # canonical_score[i] = cosine(view_dir_i, mean_view_dir): higher = more frontal/representative.
        # ------------------------------------------------------------------
        _key_mode = getattr(self.cfg, "target_key_selection_mode", "random")
        _use_canonical_progressive = (_key_mode == "canonical_progressive")
        if _use_canonical_progressive and gaussian is not None:
            with torch.no_grad():
                _obj_center = gaussian.get_xyz.mean(dim=0)
                _cam_pos = torch.stack([c.camera_center for c in target_cams])
                _view_dirs = F.normalize(_obj_center.unsqueeze(0) - _cam_pos, dim=1)
                _mean_dir = F.normalize(_view_dirs.mean(dim=0, keepdim=True), dim=1)
                _canonical_scores = (_view_dirs * _mean_dir).sum(dim=1).cpu().tolist()
            _key_count = [0] * n_target
        else:
            _canonical_scores = None
            _key_count = None

        # ------------------------------------------------------------------
        # Phase 3. Target-view denoise loop
        #   - run denoising only on target views (keys are fixed if skipped)
        #   - use consistent cross-attention and (optionally) 3D-anchor
        # ------------------------------------------------------------------
        with latency_logger.timeit(f"{_p}.target_denoise_loop") if latency_logger else nullcontext():
            # latents_target: views to denoise, shaped [n_target, 4, H, W]
            latents_target = latents[target_indices]
            use_normal_attn_target = True
            # Ensure UNet starts in a well-defined state before the target loop.
            # When target_use_extended_attention=False we must guarantee normal-attn from step 0.
            if not self.cfg.target_use_extended_attention:
                self.use_normal_unet()
            for step_index, t_step in enumerate(self.scheduler.timesteps):
                with latency_logger.timeit(f"{_p}.target_denoise_loop.per_timestep_setup") if latency_logger else nullcontext():
                    if self.cfg.target_use_extended_attention:
                        # Warmup with normal attention, then switch to extended (current default behavior).
                        if t_step < 100:
                            if not use_normal_attn_target:
                                self.use_normal_unet()
                                use_normal_attn_target = True
                        else:
                            if use_normal_attn_target:
                                register_normal_attn_flag(self.unet, False)
                                use_normal_attn_target = False
                    else:
                        # Always use normal self-attention in target loop.
                        if not use_normal_attn_target:
                            self.use_normal_unet()
                            use_normal_attn_target = True
                    if self.cfg.target_batch_strategy == "adaptive":
                        batches = _build_target_batches(
                            n_target,
                            camera_batch_size,
                            t_step.item(),
                            self.cfg.target_batch_neighbor_threshold,
                            self.cfg.target_batch_late_mode,
                            self.cfg.target_batch_sliding_stride,
                            step_index,
                        )
                    else:
                        batches = [
                            list(range(b, min(b + camera_batch_size, n_target)))
                            for b in range(0, n_target, camera_batch_size)
                        ]
                    if _use_canonical_progressive and _canonical_scores is not None and _key_count is not None:
                        _is_early = (t_step.item() >= self.cfg.target_batch_neighbor_threshold)
                        pivotal_per_batch = _select_pivotals_for_batches(
                            batches, _canonical_scores, _key_count, _is_early,
                        )
                    elif _key_mode == "fixed":
                        # Deterministic choice: first local index in each batch
                        pivotal_per_batch = [batch[0] for batch in batches if batch]
                    else:
                        pivotal_per_batch = [random.choice(batch) for batch in batches]
                    if step_index == 0:
                        print(f"[adaptive_batch] strategy={self.cfg.target_batch_strategy} "
                              f"key_mode={self.cfg.target_key_selection_mode} "
                              f"threshold={self.cfg.target_batch_neighbor_threshold} "
                              f"late_mode={self.cfg.target_batch_late_mode} "
                              f"stride_cfg={self.cfg.target_batch_sliding_stride} "
                              f"t_step={t_step.item()} n_target={n_target} bs={camera_batch_size}")
                        print(f"[adaptive_batch] batches={batches}")
                        print(f"[adaptive_batch] pivotals={pivotal_per_batch} "
                              f"canonical_progressive={_use_canonical_progressive}")
                # Pivotal forward is needed when:
                #   (a) per_step_cross_attn_consistency is active (_per_step_mode), OR
                #   (b) extended attention is active (use_normal_attn_target=False) —
                #       DGEBlock.forward requires pivotal_pass / pivot_hidden_states / batch_idx etc.
                # In pure normal-attn + no per-step mode we skip entirely.
                _step_M_con: Optional[Dict[int, Dict[int, torch.Tensor]]] = None
                _step_collected_res: List[int] = []
                _need_pivotal = _per_step_mode or (not use_normal_attn_target)
                if _need_pivotal:
                    pivotal_idx = torch.tensor(pivotal_per_batch, device=device, dtype=torch.long)
                    register_pivotal(self.unet, True)
                    key_cams_batch = [target_cams[i] for i in pivotal_per_batch]
                    latent_model_input = torch.cat([latents_target[pivotal_idx]] * 3)
                    pivot_text_embeddings = torch.cat([
                        target_pos_emb[pivotal_idx], target_neg_emb[pivotal_idx], target_neg_emb[pivotal_idx]
                    ], dim=0)
                    pivot_image_cond_latents = torch.cat([
                        target_split_cond[pivotal_idx], target_split_cond[pivotal_idx], target_zero_cond[pivotal_idx]
                    ], dim=0)
                    latent_model_input = torch.cat([latent_model_input, pivot_image_cond_latents], dim=1)

                    # Per-step cross-attn: swap attn2 to store processor before pivotal forward
                    _step_store_proc = None
                    _step_saved_procs: Dict[str, Any] = {}
                    if _per_step_mode and t_step.item() >= _per_step_t_start:
                        _step_store_proc = CrossAttentionStoreProcessor(valid_indices, target_resolutions)
                        for _name, _mod in _attn2_modules:
                            _step_saved_procs[_name] = _mod.processor
                            _mod.processor = _step_store_proc

                    ## 이게 있어야지 퀄리티가 좋음.!!
                    with latency_logger.timeit(f"{_p}.target_denoise_loop.pivotal_forward") if latency_logger else nullcontext():
                        if latency_logger:
                            set_unet_latency_prefix(f"{_p}.target_denoise_loop.pivotal_forward.unet_forward")
                        try:
                            with latency_logger.timeit(f"{_p}.target_denoise_loop.pivotal_forward.unet_forward") if latency_logger else nullcontext():
                                self.forward_unet(latent_model_input, t_step.unsqueeze(0).expand(len(pivotal_idx) * 3).to(device), encoder_hidden_states=pivot_text_embeddings)
                        finally:
                            if latency_logger:
                                set_unet_latency_prefix(None)
                        register_pivotal(self.unet, False)
                    if _step_store_proc is not None:
                        _psc_prefix = f"{_p}.target_denoise_loop.per_step_consistent_maps"
                        with latency_logger.timeit(_psc_prefix) if latency_logger else nullcontext():
                            with latency_logger.timeit(f"{_psc_prefix}.restore_processors") if latency_logger else nullcontext():
                                for _name, _mod in _attn2_modules:
                                    if _name in _step_saved_procs:
                                        _mod.processor = _step_saved_procs[_name]
                            _step_collected_res, _step_M_con = _per_step_build_consistent_maps(
                                _step_store_proc,
                                len(pivotal_per_batch),
                                key_cams_batch,
                                target_indices,
                                cams,
                                gaussian,
                                pipe,
                                device,
                                attn_len,
                                target_resolutions=target_resolutions,
                                latency_logger=latency_logger,
                                latency_prefix=_psc_prefix,
                            )
                        if step_index == 0:
                            print(f"[per_step_cross_attn] t={t_step.item()} built maps for "
                                  f"{len(target_indices)} views from {len(pivotal_per_batch)} pivotals, "
                                  f"resolutions={_step_collected_res}")

                with latency_logger.timeit(f"{_p}.target_denoise_loop.accum_init") if latency_logger else nullcontext():
                    noise_pred_accum = torch.zeros_like(latents_target)
                    noise_pred_count = torch.zeros(n_target, device=device)
                for batch_idx, batch_local in enumerate(batches):
                    batch_indices_t = torch.tensor(batch_local, device=device, dtype=torch.long)
                    batch_target_indices = [target_indices[i] for i in batch_local]
                    with latency_logger.timeit(f"{_p}.target_denoise_loop.batch_prep") if latency_logger else nullcontext():
                        if _per_step_mode:
                            _active_M_con = _step_M_con
                            _active_res = _step_collected_res
                            data = {"attn_len": attn_len}
                            for res in _active_res:
                                maps_batch = []
                                for i in batch_target_indices:
                                    if i in _active_M_con and res in _active_M_con.get(i, {}):
                                        m = _active_M_con[i][res]
                                        maps_batch.append(m.reshape(-1, attn_len))
                                if maps_batch:
                                    data[res] = torch.stack(maps_batch, dim=0).to(device)
                            _use_consistent = data.get("attn_len") and any(k in data for k in _active_res)
                            _attn_map_val = data if _use_consistent else None
                            for _, mod in _attn2_modules:
                                setattr(mod, "_consistent_attn_map_current", _attn_map_val)
                        if _need_pivotal:
                            if _feature_injection_mode == "3d_anchor" and anchor_3d_cache is not None:
                                _pivot_global = target_indices[pivotal_per_batch[batch_idx]] if batch_idx < len(pivotal_per_batch) else target_indices[0]
                                with latency_logger.timeit(f"{_p}.target_denoise_loop.register_anchor_3d_cache") if latency_logger else nullcontext():
                                    register_anchor_3d_cache(
                                        self.unet, anchor_3d_cache,
                                        batch_view_indices=batch_target_indices,
                                        injection_lambda=_injection_lambda,
                                        pivot_view_index=_pivot_global,
                                        injection_3d_anchor_style=_injection_3d_anchor_style,
                                    )
                            _pivot_in_batch = batch_local.index(pivotal_per_batch[batch_idx])
                            for _, mod in _dge_blocks:
                                setattr(mod, "batch_idx", batch_idx)
                                setattr(mod, "cams", [target_cams[j] for j in batch_local])
                                setattr(mod, "pivot_this_batch", _pivot_in_batch)
                                setattr(mod, "key_cams", key_cams_batch)
                                setattr(mod, "epipolar_constrains", {})
                        batch_model_input = torch.cat([latents_target[batch_indices_t]] * 3)
                        batch_text_embeddings = torch.cat([
                            target_pos_emb[batch_indices_t], target_neg_emb[batch_indices_t], target_neg_emb[batch_indices_t]
                        ], dim=0)
                        batch_image_cond_latents = torch.cat([
                            target_split_cond[batch_indices_t], target_split_cond[batch_indices_t], target_zero_cond[batch_indices_t]
                        ], dim=0)
                        batch_model_input = torch.cat([batch_model_input, batch_image_cond_latents], dim=1)
                    with latency_logger.timeit(f"{_p}.target_denoise_loop.batch_forward") if latency_logger else nullcontext():
                        if latency_logger:
                            set_unet_latency_prefix(f"{_p}.target_denoise_loop.batch_forward.unet_forward")
                        try:
                            with latency_logger.timeit(f"{_p}.target_denoise_loop.batch_forward.unet_forward") if latency_logger else nullcontext():
                                batch_noise_pred = self.forward_unet(batch_model_input, t_step.unsqueeze(0).expand(len(batch_local) * 3).to(device), encoder_hidden_states=batch_text_embeddings)
                            torch.cuda.synchronize()
                        finally:
                            if latency_logger:
                                set_unet_latency_prefix(None)
                    if _feature_injection_mode == "3d_anchor":
                        with latency_logger.timeit(f"{_p}.target_denoise_loop.unregister_anchor_3d_cache") if latency_logger else nullcontext():
                            unregister_anchor_3d_cache(self.unet)
                    with latency_logger.timeit(f"{_p}.target_denoise_loop.cfg_and_accum") if latency_logger else nullcontext():
                        batch_noise_pred_text, batch_noise_pred_image, batch_noise_pred_uncond = batch_noise_pred.chunk(3)
                        batch_combined = (
                            batch_noise_pred_uncond
                            + self.cfg.guidance_scale * (batch_noise_pred_text - batch_noise_pred_image)
                            + self.cfg.condition_scale * (batch_noise_pred_image - batch_noise_pred_uncond)
                        )
                        for j, local_idx in enumerate(batch_local):
                            noise_pred_accum[local_idx] += batch_combined[j]
                            noise_pred_count[local_idx] += 1
                with latency_logger.timeit(f"{_p}.target_denoise_loop.merge_and_step") if latency_logger else nullcontext():
                    for _, mod in _attn2_modules:
                        setattr(mod, "_consistent_attn_map_current", None)
                    noise_pred = noise_pred_accum / noise_pred_count.view(-1, 1, 1, 1).clamp(min=1)
                    latents_target = self.scheduler.step(noise_pred, t_step, latents_target).prev_sample
                    torch.cuda.synchronize()
            with latency_logger.timeit(f"{_p}.target_denoise_loop.write_back") if latency_logger else nullcontext():
                latents[target_indices] = latents_target

        with latency_logger.timeit(f"{_p}.restore_attn2_final") if latency_logger else nullcontext():
            for name, mod in self.unet.named_modules():
                if name.endswith(".attn2") and name in original_attn2_processors:
                    mod.processor = original_attn2_processors[name]
        print("Multiview editing finished.")
        return latents

    def compute_grad_sds(
        self,
        text_embeddings: Float[Tensor, "BB 77 768"],
        latents: Float[Tensor, "B 4 DH DW"],
        image_cond_latents: Float[Tensor, "B 4 DH DW"],
        t: Int[Tensor, "B"],
    ) -> Float[Tensor, "B 4 DH DW"]:
        """Vanilla SDS: pure score distillation, no epipolar/pivotal."""
        noise = torch.randn_like(latents)
        latents = self.scheduler.add_noise(latents, noise, t)
        positive_text_embedding, negative_text_embedding, _ = text_embeddings.chunk(3)
        split_image_cond_latents, _, zero_image_cond_latents = image_cond_latents.chunk(3)

        # Vanilla SDS uses standard attention (no DGE extended attn)
        self.use_normal_unet()
        register_pivotal(self.unet, False)

        with torch.no_grad():
            latent_model_input = torch.cat([latents] * 3)
            batch_text_embeddings = torch.cat([
                positive_text_embedding, negative_text_embedding, negative_text_embedding
            ], dim=0)
            batch_image_cond_latents = torch.cat([
                split_image_cond_latents, split_image_cond_latents, zero_image_cond_latents
            ], dim=0)
            latent_model_input = torch.cat([latent_model_input, batch_image_cond_latents], dim=1)
            noise_pred = self.forward_unet(latent_model_input, t, encoder_hidden_states=batch_text_embeddings)
            noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)

            noise_pred = (
                noise_pred_uncond
                + self.cfg.guidance_scale * (noise_pred_text - noise_pred_image)
                + self.cfg.condition_scale * (noise_pred_image - noise_pred_uncond)
            )

        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        grad = w * (noise_pred - noise)
        return grad

    def compute_grad_sds_dge(
        self,
        text_embeddings: Float[Tensor, "BB 77 768"],
        latents: Float[Tensor, "B 4 DH DW"],
        image_cond_latents: Float[Tensor, "B 4 DH DW"],
        t: Int[Tensor, "B"],
        cams,
    ) -> Float[Tensor, "B 4 DH DW"]:
        """DGE SDS: with epipolar constraints, pivotal pass, multi-view consistency."""
        noise = torch.randn_like(latents)
        latents = self.scheduler.add_noise(latents, noise, t)
        positive_text_embedding, negative_text_embedding, _ = text_embeddings.chunk(3)
        split_image_cond_latents, _, zero_image_cond_latents = image_cond_latents.chunk(3)
        current_H = image_cond_latents.shape[2]
        current_W = image_cond_latents.shape[3]
        camera_batch_size = self.cfg.camera_batch_size
        effective_camera_batch_size = min(camera_batch_size, len(latents))

        with torch.no_grad():
            noise_pred_text = []
            noise_pred_image = []
            noise_pred_uncond = []
            pivotal_idx = torch.randint(0, effective_camera_batch_size, (len(latents) // effective_camera_batch_size,), device=latents.device) + torch.arange(0, len(latents), effective_camera_batch_size, device=latents.device)
            register_pivotal(self.unet, True)

            latent_model_input = torch.cat([latents[pivotal_idx]] * 3)
            pivot_text_embeddings = torch.cat([positive_text_embedding[pivotal_idx], negative_text_embedding[pivotal_idx], negative_text_embedding[pivotal_idx]], dim=0)
            pivot_image_cond_latetns = torch.cat([split_image_cond_latents[pivotal_idx], split_image_cond_latents[pivotal_idx], zero_image_cond_latents[pivotal_idx]], dim=0)
            latent_model_input = torch.cat([latent_model_input, pivot_image_cond_latetns], dim=1)

            key_cams = [cams[i] for i in pivotal_idx.cpu().tolist()]
            self.forward_unet(latent_model_input, t, encoder_hidden_states=pivot_text_embeddings)
            register_pivotal(self.unet, False)

            for i, b in enumerate(range(0, len(latents), effective_camera_batch_size)):
                register_batch_idx(self.unet, i)
                register_cams(self.unet, cams[b:b + effective_camera_batch_size], pivotal_idx[i].item() % effective_camera_batch_size, key_cams)

                epipolar_constrains = {}
                for down_sample_factor in [1, 2, 4, 8]:
                    H = current_H // down_sample_factor
                    W = current_W // down_sample_factor
                    epipolar_constrains[H * W] = []
                    for cam in cams[b:b + effective_camera_batch_size]:
                        cam_epipolar_constrains = []
                        for key_cam in key_cams:
                            cam_epipolar_constrains.append(compute_epipolar_constrains(key_cam, cam, current_H=H, current_W=W))
                        epipolar_constrains[H * W].append(torch.stack(cam_epipolar_constrains, dim=0))
                    epipolar_constrains[H * W] = torch.stack(epipolar_constrains[H * W], dim=0)
                register_epipolar_constrains(self.unet, epipolar_constrains)

                batch_model_input = torch.cat([latents[b:b + effective_camera_batch_size]] * 3)
                batch_text_embeddings = torch.cat([positive_text_embedding[b:b + effective_camera_batch_size], negative_text_embedding[b:b + effective_camera_batch_size], negative_text_embedding[b:b + effective_camera_batch_size]], dim=0)
                batch_image_cond_latents = torch.cat([split_image_cond_latents[b:b + effective_camera_batch_size], split_image_cond_latents[b:b + effective_camera_batch_size], zero_image_cond_latents[b:b + effective_camera_batch_size]], dim=0)
                batch_model_input = torch.cat([batch_model_input, batch_image_cond_latents], dim=1)
                batch_noise_pred = self.forward_unet(batch_model_input, t, encoder_hidden_states=batch_text_embeddings)
                batch_noise_pred_text, batch_noise_pred_image, batch_noise_pred_uncond = batch_noise_pred.chunk(3)
                noise_pred_text.append(batch_noise_pred_text)
                noise_pred_image.append(batch_noise_pred_image)
                noise_pred_uncond.append(batch_noise_pred_uncond)

            noise_pred_text = torch.cat(noise_pred_text, dim=0)
            noise_pred_image = torch.cat(noise_pred_image, dim=0)
            noise_pred_uncond = torch.cat(noise_pred_uncond, dim=0)

            noise_pred = (
                noise_pred_uncond
                + self.cfg.guidance_scale * (noise_pred_text - noise_pred_image)
                + self.cfg.condition_scale * (noise_pred_image - noise_pred_uncond)
            )

        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        grad = w * (noise_pred - noise)
        return grad
    



    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"], # images
        cond_rgb: Float[Tensor, "B H W C"], # original_frames
        prompt_utils: PromptProcessorOutput, # prompt_processor(text prompts)
        gaussians = None,
        cams= None,
        render=None,
        pipe=None,
        background=None,
        latency_logger=None,
        **kwargs,
    ):
        if not self.cfg.use_sds or self.cfg.use_sds_dge:
            assert cams is not None, "cams is required for dge guidance (edit_latents or use_sds_dge)"
        batch_size, H, W, _ = rgb.shape
        factor = 512 / max(W, H)
        factor = math.ceil(min(W, H) * factor / 64) * 64 / min(W, H)

        width = int((W * factor) // 64) * 64
        height = int((H * factor) // 64) * 64
        rgb_BCHW = rgb.permute(0, 3, 1, 2)

        RH, RW = height, width

        rgb_BCHW_HW8 = F.interpolate(
            rgb_BCHW, (RH, RW), mode="bilinear", align_corners=False
        )

        _kv = kwargs.get("key_indices", None)
        _strat = kwargs.get("key_selection_strategy", None)
        _nkv = kwargs.get("num_key_views", None)
        use_multiview_path = (
            kwargs.get("use_multiview", False)
            and kwargs.get("gaussian", None) is not None
            and kwargs.get("pipe", None) is not None
            and (_kv is not None and len(_kv) > 0 or _strat is not None and _nkv is not None)
        )
        # Optional override so DGE system can anchor hierarchy under training_step_all.*
        _override_prefix = kwargs.get("latency_prefix", None)
        if _override_prefix is not None:
            _prefix = _override_prefix
        else:
            _prefix = EDIT_MULTIVIEW_PREFIX if use_multiview_path else EDIT_ALL_VIEW_PREFIX

        # So that DGE blocks (make_dge_block) can record latency under the correct hierarchy
        if latency_logger is not None:
            register_latency_logger(self.unet, latency_logger)

        with latency_logger.timeit(f"{_prefix}.encode_images") if latency_logger else nullcontext():
            latents = self.encode_images(rgb_BCHW_HW8)

        cond_rgb_BCHW = cond_rgb.permute(0, 3, 1, 2)
        cond_rgb_BCHW_HW8 = F.interpolate(
            cond_rgb_BCHW,
            (RH, RW),
            mode="bilinear",
            align_corners=False,
        )

        with latency_logger.timeit(f"{_prefix}.encode_cond_images") if latency_logger else nullcontext():
            cond_latents = self.encode_cond_images(cond_rgb_BCHW_HW8)

        temp = torch.zeros(batch_size).to(rgb.device)

        with latency_logger.timeit(f"{_prefix}.text_embeddings") if latency_logger else nullcontext():
            text_embeddings = prompt_utils.get_text_embeddings(temp, temp, temp, False)
            
        positive_text_embeddings, negative_text_embeddings = text_embeddings.chunk(2)
        text_embeddings = torch.cat(
            [positive_text_embeddings, negative_text_embeddings, negative_text_embeddings], dim=0)  # [positive, negative, negative]

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.max_step - 1,
            self.max_step,
            [1],
            dtype=torch.long,
            device=self.device,
        ).repeat(batch_size)

        if self.cfg.use_sds:
            with latency_logger.timeit(f"{_prefix}.compute_grad_sds") if latency_logger else nullcontext():
                if self.cfg.use_sds_dge:
                    grad = self.compute_grad_sds_dge(text_embeddings, latents, cond_latents, t, cams)
                else:
                    grad = self.compute_grad_sds(text_embeddings, latents, cond_latents, t)
            grad = torch.nan_to_num(grad)
            if self.grad_clip_val is not None:
                grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
            target = (latents - grad).detach()
            loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
            return {
                "loss_sds": loss_sds,
                "grad_norm": grad.norm(),
                "min_step": self.min_step,
                "max_step": self.max_step,
            }
        else:
            use_multiview = kwargs.get("use_multiview", False)
            gaussian = kwargs.get("gaussian", None)
            # pipe is also a __call__ parameter; kwargs.get("pipe", None) would be None when passed as pipe=...
            pipe = kwargs.get("pipe", pipe)
            key_indices = kwargs.get("key_indices", None)
            prompt_text = kwargs.get("prompt_text", "") or getattr(prompt_utils, "prompt", "")
            if use_multiview and gaussian is not None and pipe is not None and (
                key_indices is not None and len(key_indices) > 0 or _strat is not None and _nkv is not None
            ):
                key_view_camera_ids = kwargs.get("key_view_camera_ids", None)
                with latency_logger.timeit(f"{_prefix}.edit_latents_multiview") if latency_logger else nullcontext():
                    edit_latents = self.edit_latents_multiview(
                        text_embeddings,
                        latents,
                        cond_latents,
                        t,
                        cams,
                        # key_indices=key_indices,
                        # key_view_camera_ids=key_view_camera_ids,
                        gaussian=gaussian,
                        pipe=pipe,
                        prompt_text=prompt_text,
                        latency_logger=latency_logger,
                        key_selection_strategy=_strat,
                        num_key_views=_nkv,
                        latency_prefix=_prefix,
                    )
            else:
                gp_cache = kwargs.get("gp_cache", None)
                key_cam_indices = kwargs.get("key_cam_indices", None)
                edit_latents = self.edit_latents(
                    text_embeddings,
                    latents,
                    cond_latents,
                    t,
                    cams,
                    latency_logger,
                    gp_cache=gp_cache,
                    key_cam_indices=key_cam_indices,
                    latency_prefix=_prefix,
                )
            with latency_logger.timeit(f"{_prefix}.decode_latents") if latency_logger else nullcontext():
                edit_images = self.decode_latents(edit_latents)
            edit_images = F.interpolate(edit_images, (H, W), mode="bilinear")

            return {"edit_images": edit_images.permute(0, 2, 3, 1)}

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )


