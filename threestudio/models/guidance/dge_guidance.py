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
        # For cross-attn storage we need spatial resolution (query length), not text length
        seq_spatial = hidden_states.shape[1]
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
        # Store at any spatial resolution (UNet may use 64*64, 32*32, 16*16, 8*8, etc. depending on input)
        probs = attention_probs.detach().float()
        valid = [i for i in self.valid_token_indices if i < probs.shape[-1]]
        if valid:
            probs_valid = probs[:, :, valid]
            self.maps[seq_spatial].append(probs_valid)
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


class ConsistentCrossAttnProcessor:
    """Uses precomputed consistent cross-attention map when set on attn module."""

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
        batch_size, sequence_length, channel = hidden_states.shape
        height = width = int(sequence_length ** 0.5) if sequence_length > 0 else 0
        attn_len = data.get("attn_len")
        consistent_map = data.get(sequence_length)
        if attn_len is None or consistent_map is None or consistent_map.shape[0] != batch_size:
            if self.backup_processor is not None:
                return self.backup_processor(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
            return self._default_forward(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        if encoder_hidden_states is not None and attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        value = attn.head_to_batch_dim(value)
        value_valid = value[:, :attn_len]
        consistent_map = consistent_map.to(value.device).to(value.dtype)
        if consistent_map.dim() == 2:
            consistent_map = consistent_map.unsqueeze(0).expand(batch_size, -1, -1)
        hidden_states = torch.bmm(consistent_map, value_valid)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
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
        edit_view_selection_strategy: str = ""
        skip_key_views_in_target_loop: bool = False
        # Feature injection in edit_latents_multiview: "similarity" (cosine + gather) or "3d_anchor" (3DGS-based canonical tokens)
        feature_injection_mode: str = "similarity"
        # For 3d_anchor: blend h_out = (1 - injection_lambda) * h_sa + injection_lambda * F(v,p). h_sa uses current hidden_states.
        injection_lambda: float = 0.5
        # 3d_anchor only: "blend" = λ*(F-h)+h; "gather" = similarity-like: pivot self-attn + 3D-GS remap gather + residual (no λ).
        injection_3d_anchor_style: str = "blend"
        # Key-view denoise loop: if True, use extended (multi-frame) self-attention after warmup; if False, always use normal self-attention.
        key_denoise_use_extended_attention: bool = False

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
        cams= None,
        latency_logger=None,
        gp_cache=None,
        key_cam_indices=None,
    ) -> Float[Tensor, "B 4 DH DW"]:
        
        self.scheduler.config.num_train_timesteps = t.item() if len(t.shape) < 1 else t[0].item()
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)

        current_H = image_cond_latents.shape[2]
        current_W = image_cond_latents.shape[3]

        camera_batch_size = self.cfg.camera_batch_size
        print("Start editing images...")

        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents = self.scheduler.add_noise(latents, noise, t) 

            # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
            positive_text_embedding, negative_text_embedding, _ = text_embeddings.chunk(3)
            split_image_cond_latents, _, zero_image_cond_latents = image_cond_latents.chunk(3)
            
            with latency_logger.timeit("edit_all_view.guidance_batch.edit_latents.diffusion_loop"):
                for t in self.scheduler.timesteps:
                    with latency_logger.timeit("edit_all_view.guidance_batch.edit_latents.diffusion_loop.timestep_setup"):
                        if t < 100:
                            self.use_normal_unet()
                        else:
                            register_normal_attn_flag(self.unet, False)
                        
                    with torch.no_grad():
                        # pred noise
                        noise_pred_text = []
                        noise_pred_image = []
                        noise_pred_uncond = []
                        
                        with latency_logger.timeit("edit_all_view.guidance_batch.edit_latents.diffusion_loop.pivotal_setup"):
                            if self.cfg.edit_view_selection_strategy == "manual-20":
                                pivotal_idx = torch.tensor([2, 7, 12, 17]) # 카메라 uid가 [6, 32, 42, 20] 인 카메라를 가리키는 인덱스
                            elif self.cfg.edit_view_selection_strategy == "manual-15":
                                pivotal_idx = torch.tensor([2, 7, 12]) # 카메라 uid가 [6, 32, 20] 인 카메라를 가리키는 인덱스
                            else:
                                pivotal_idx = torch.randint(camera_batch_size, (len(latents)//camera_batch_size,)) + torch.arange(0, len(latents), camera_batch_size) # ex)  [0, 2, 1, 2] + [0, 5, 10, 15]
                            register_pivotal(self.unet, True)
                            
                            key_cams = [cams[cam_pivotal_idx] for cam_pivotal_idx in pivotal_idx.tolist()]
                            latent_model_input = torch.cat([latents[pivotal_idx]] * 3)
                            pivot_text_embeddings = torch.cat([positive_text_embedding[pivotal_idx], negative_text_embedding[pivotal_idx], negative_text_embedding[pivotal_idx]], dim=0)
                            pivot_image_cond_latetns = torch.cat([split_image_cond_latents[pivotal_idx], split_image_cond_latents[pivotal_idx], zero_image_cond_latents[pivotal_idx]], dim=0)
                            latent_model_input = torch.cat([latent_model_input, pivot_image_cond_latetns], dim=1)

                        with latency_logger.timeit("edit_all_view.guidance_batch.edit_latents.diffusion_loop.pivotal_forward"):
                            self.forward_unet(latent_model_input, t, encoder_hidden_states=pivot_text_embeddings)
                            register_pivotal(self.unet, False)

                        with latency_logger.timeit("edit_all_view.guidance_batch.edit_latents.diffusion_loop.batch_processing"):
                            for i, b in enumerate(range(0, len(latents), camera_batch_size)):
                                with latency_logger.timeit("edit_all_view.guidance_batch.edit_latents.diffusion_loop.batch_processing.register_ops"):
                                    with latency_logger.timeit("edit_all_view.guidance_batch.edit_latents.diffusion_loop.batch_processing.register_ops.register_batch_idx"):
                                        register_batch_idx(self.unet, i)

                                    with latency_logger.timeit("edit_all_view.guidance_batch.edit_latents.diffusion_loop.batch_processing.register_ops.register_cams"):
                                        register_cams(self.unet, cams[b:b+camera_batch_size], pivotal_idx[i] % camera_batch_size, key_cams) 
                                    
                                    with latency_logger.timeit("edit_all_view.guidance_batch.edit_latents.diffusion_loop.batch_processing.register_ops.compute_epipolar_constrains"):
                                        if gp_cache is not None:
                                            # Version B: skip dense epipolar computation entirely.
                                            # Register the stacked gaussian-provenance cache for all
                                            # cameras in this batch (indexed b .. b+camera_batch_size-1).
                                            batch_end = min(b + camera_batch_size, len(cams))
                                            register_gp_cache(self.unet, gp_cache, b, batch_end)
                                        else:
                                            epipolar_constrains = {}
                                            # Create directory for saving epipolar constraint images in save_dir
                                            epipolar_images_dir = os.path.join(self.save_dir, "epipolar_constraints_images")

                                            # Warmup: run first epipolar compute once to avoid cam_0 including CUDA init time
                                            if torch.cuda.is_available() and key_cams:
                                                _ = compute_epipolar_constrains(
                                                    key_cams[0], cams[b], current_H=current_H // 1, current_W=current_W // 1, downsample_factor=1
                                                )
                                                torch.cuda.synchronize()

                                            for down_sample_factor in [1, 2, 4, 8]:
                                                with latency_logger.timeit(f"edit_all_view.guidance_batch.edit_latents.diffusion_loop.batch_processing.register_ops.compute_epipolar_constrains.downsample_{down_sample_factor}"):
                                                    H = current_H // down_sample_factor
                                                    W = current_W // down_sample_factor
                                                    epipolar_constrains[H * W] = []
                                                    for cam_idx, cam in enumerate(cams[b:b + camera_batch_size]):
                                                        with latency_logger.timeit(f"edit_all_view.guidance_batch.edit_latents.diffusion_loop.batch_processing.register_ops.compute_epipolar_constrains.downsample_{down_sample_factor}.cam_{cam_idx}"):
                                                            cam_epipolar_constrains = []
                                                            for key_cam_idx, key_cam in enumerate(key_cams):
                                                                # Pass downsample_factor to the function
                                                                epipolar_constraint = compute_epipolar_constrains(key_cam, cam, current_H=H, current_W=W, downsample_factor=down_sample_factor)
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
                                                            epipolar_constrains[H * W].append(torch.stack(cam_epipolar_constrains, dim=0))
                                                    epipolar_constrains[H * W] = torch.stack(epipolar_constrains[H * W], dim=0)

                                            with latency_logger.timeit("edit_all_view.guidance_batch.edit_latents.diffusion_loop.batch_processing.register_ops.register_epipolar_constrains"):
                                                register_epipolar_constrains(self.unet, epipolar_constrains)

                                with latency_logger.timeit("edit_all_view.guidance_batch.edit_latents.diffusion_loop.batch_processing.prepare_input"):
                                    batch_model_input = torch.cat([latents[b:b + camera_batch_size]] * 3)
                                    batch_text_embeddings = torch.cat([positive_text_embedding[b:b + camera_batch_size], negative_text_embedding[b:b + camera_batch_size], negative_text_embedding[b:b + camera_batch_size]], dim=0)
                                    batch_image_cond_latents = torch.cat([split_image_cond_latents[b:b + camera_batch_size], split_image_cond_latents[b:b + camera_batch_size], zero_image_cond_latents[b:b + camera_batch_size]], dim=0)
                                    batch_model_input = torch.cat([batch_model_input, batch_image_cond_latents], dim=1)

                                with latency_logger.timeit("edit_all_view.guidance_batch.edit_latents.diffusion_loop.batch_processing.unet_forward"):
                                    batch_noise_pred = self.forward_unet(batch_model_input, t, encoder_hidden_states=batch_text_embeddings)
                                    batch_noise_pred_text, batch_noise_pred_image, batch_noise_pred_uncond = batch_noise_pred.chunk(3)
                                    noise_pred_text.append(batch_noise_pred_text)
                                    noise_pred_image.append(batch_noise_pred_image)
                                    noise_pred_uncond.append(batch_noise_pred_uncond)

                        with latency_logger.timeit("edit_all_view.guidance_batch.edit_latents.diffusion_loop.concat_outputs"):
                            noise_pred_text = torch.cat(noise_pred_text, dim=0)
                            noise_pred_image = torch.cat(noise_pred_image, dim=0)
                            noise_pred_uncond = torch.cat(noise_pred_uncond, dim=0)

                        with latency_logger.timeit("edit_all_view.guidance_batch.edit_latents.diffusion_loop.guidance_calc"):
                            # perform classifier-free guidance
                            noise_pred = (
                                noise_pred_uncond
                                + self.cfg.guidance_scale * (noise_pred_text - noise_pred_image)
                                + self.cfg.condition_scale * (noise_pred_image - noise_pred_uncond)
                            )

                        with latency_logger.timeit("edit_all_view.guidance_batch.edit_latents.diffusion_loop.scheduler_step"):
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
        skip_key_views_in_target_loop: bool = True,
        feature_injection_mode: Optional[str] = None,
        injection_lambda: Optional[float] = None,
        injection_3d_anchor_style: Optional[str] = None,
        key_selection_strategy: Optional[str] = None,
        num_key_views: Optional[int] = None,
    ) -> Float[Tensor, "B 4 DH DW"]:
        """
        Multiview edit: key views with cross-view attention; collect cross-attn from key views,
        inverse-render to 3D, render to consistent 2D maps; target views use consistent map in upsampling.

        skip_key_views_in_target_loop: if True, key views are frozen (set to key_edited) and
            only target views are denoised in target_denoise_loop, reducing UNet forwards by ~(n_key/n_views).
            if False, all views (including key views) are denoised in target_denoise_loop (original behavior).
        """
        _p = f"{EDIT_MULTIVIEW_PREFIX}.edit_latents_multiview"  # so summary shows guidance_batch -> edit_latents_multiview -> setup, key_view_denoise_loop, ...
        _feature_injection_mode = feature_injection_mode if feature_injection_mode is not None else self.cfg.feature_injection_mode
        _injection_lambda = injection_lambda if injection_lambda is not None else self.cfg.injection_lambda
        _injection_3d_anchor_style = injection_3d_anchor_style if injection_3d_anchor_style is not None else self.cfg.injection_3d_anchor_style
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


            target_resolutions = (32 * 32, 64 * 64)
            # Cache module lists once to avoid repeated named_modules() traversal in the loop
            _dge_blocks = [(n, m) for n, m in self.unet.named_modules()
                           if isinstance_str(m, "BasicTransformerBlock")]
            _attn2_modules = [(n, m) for n, m in self.unet.named_modules()
                              if n.endswith(".attn2") and hasattr(m, "processor")]

        with latency_logger.timeit(f"{_p}.valid_token_indices") if latency_logger else nullcontext():
            valid_indices = _get_valid_token_indices(self.pipe, prompt_text)
        attn_len = len(valid_indices)
        if attn_len == 0:
            return self.edit_latents(text_embeddings, latents, image_cond_latents, t, cams, latency_logger=latency_logger)

        with latency_logger.timeit(f"{_p}.install_store_processor") if latency_logger else nullcontext():
            storing_processor = CrossAttentionStoreProcessor(valid_indices, target_resolutions)
            original_attn2_processors = {}
            for name, mod in self.unet.named_modules():
                if name.endswith(".attn2") and hasattr(mod, "processor"):
                    original_attn2_processors[name] = mod.processor
                    mod.processor = storing_processor

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
        print(f"[edit_latents_multiview.key_denoise_loop] key view indices: {key_indices}, actual camera IDs: {key_cams}")
        print(f"[edit_latents_multiview.key_denoise_loop] number of key views: {n_key}")


        with torch.no_grad():
            with latency_logger.timeit(f"{_p}.key_view_denoise_loop") if latency_logger else nullcontext():
                # Key denoise never uses kf_attn_output; skip storing to save memory/copy.
                register_store_kf_attn_output(self.unet, False)
                with latency_logger.timeit(f"{_p}.key_view_denoise_loop.init") if latency_logger else nullcontext():
                    noise = torch.randn_like(latents)
                    latents_key = self.scheduler.add_noise(latents[key_indices], noise[key_indices], t[key_indices])
                # Precompute pivot text/cond (unchanged across steps)
                pivot_text = torch.cat([
                    positive_text_embedding[key_indices], negative_text_embedding[key_indices], negative_text_embedding[key_indices]
                ], dim=0)
                pivot_image_cond = torch.cat([
                    split_image_cond_latents[key_indices], split_image_cond_latents[key_indices], zero_image_cond_latents[key_indices]
                ], dim=0)
                # Key loop self-attention strategy:
                # - key_denoise_use_extended_attention = False:
                #     * always use normal self-attention (no frame-extend) for key loop
                # - key_denoise_use_extended_attention = True:
                #     * warmup (<100) with normal self-attn, then switch to extended (like target loop)
                use_normal_attn = True
                # Start key loop in normal-attn mode to have a well-defined initial state.
                self.use_normal_unet()
                for t_step in self.scheduler.timesteps:
                    with latency_logger.timeit(f"{_p}.key_view_denoise_loop.per_step_setup") if latency_logger else nullcontext():
                        if self.cfg.key_denoise_use_extended_attention:
                            # Original behavior: warmup with normal attn, then switch to extended.
                            if t_step < 100:
                                if not use_normal_attn:
                                    self.use_normal_unet()
                                    use_normal_attn = True
                            else:
                                if use_normal_attn:
                                    register_normal_attn_flag(self.unet, False)
                                    use_normal_attn = False
                        else:
                            # Always normal self-attention in key loop: if something toggled it off, restore.
                            if not use_normal_attn:
                                self.use_normal_unet()
                                use_normal_attn = True

                        register_pivotal(self.unet, True)
                        latent_model_input = torch.cat([latents_key] * 3)
                        latent_model_input = torch.cat([latent_model_input, pivot_image_cond], dim=1)
                        t_exp = t_step.unsqueeze(0).expand(n_key * 3).to(device)
                    with latency_logger.timeit(f"{_p}.key_view_denoise_loop.forward_unet") if latency_logger else nullcontext():
                        if latency_logger:
                            set_unet_latency_prefix(f"{_p}.key_view_denoise_loop.forward_unet.unet_forward")
                        try:
                            with latency_logger.timeit(f"{_p}.key_view_denoise_loop.forward_unet.unet_forward") if latency_logger else nullcontext():
                                noise_pred = self.forward_unet(latent_model_input, t_exp, encoder_hidden_states=pivot_text)
                        finally:
                            if latency_logger:
                                set_unet_latency_prefix(None)
                    with latency_logger.timeit(f"{_p}.key_view_denoise_loop.guidance_and_step") if latency_logger else nullcontext():
                        noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                        noise_pred_key = (
                            noise_pred_uncond
                            + self.cfg.guidance_scale * (noise_pred_text - noise_pred_image)
                            + self.cfg.condition_scale * (noise_pred_image - noise_pred_uncond)
                        )
                        latents_key = self.scheduler.step(noise_pred_key, t_step, latents_key).prev_sample
                with latency_logger.timeit(f"{_p}.key_view_denoise_loop.finalize") if latency_logger else nullcontext():
                    register_pivotal(self.unet, False)
                    register_store_kf_attn_output(self.unet, True)  # target phase will use pivotal cache
                    key_edited = latents_key

        with latency_logger.timeit(f"{_p}.restore_attn2_processors") if latency_logger else nullcontext():
            for name, mod in self.unet.named_modules():
                if name in original_attn2_processors:
                    mod.processor = original_attn2_processors[name]

        with latency_logger.timeit(f"{_p}.build_key_cross_attn_by_res") if latency_logger else nullcontext():
            key_cross_attn_by_res: Dict[int, Float[Tensor, "n_key H*W attn_len"]] = {}
            for res, list_maps in storing_processor.maps.items():
                if not list_maps:
                    continue
                stacked = torch.stack(list_maps, dim=0)
                # Stored shape is (batch*heads, Hw, C) per step -> stack gives (L, batch*heads, Hw, C) = 4D
                if stacked.ndim == 4:
                    L, batch_heads, Hw, C = stacked.shape
                    num_heads = batch_heads // (n_key * 3)
                    stacked = stacked.view(L, n_key * 3, num_heads, Hw, C).mean(dim=(0, 2))
                else:
                    T, B, heads, Hw, C = stacked.shape[0], stacked.shape[1], stacked.shape[2], stacked.shape[3], stacked.shape[4]
                    stacked = stacked.mean(dim=(0, 2))
                if stacked.shape[0] >= 3 * n_key:
                    stacked = stacked[n_key:2 * n_key]
                else:
                    stacked = stacked.unsqueeze(0).expand(n_key, -1, -1)
                key_cross_attn_by_res[res] = stacked.float().to(device)
            collected_resolutions = list(key_cross_attn_by_res.keys())

        if not key_cross_attn_by_res:
            print("No key cross-attention found, falling back to single-view editing")
            return self.edit_latents(text_embeddings, latents, image_cond_latents, t, cams, latency_logger=latency_logger)

        def _camera_at_res(cam, h, w):
            if hasattr(cam, "HW_scale"):
                return cam.HW_scale(h, w)
            from gaussiansplatting.scene.cameras import MiniCam
            return MiniCam(w, h, cam.FoVy, cam.FoVx, getattr(cam, "znear", 0.01), getattr(cam, "zfar", 100.0),
                          cam.world_view_transform, cam.full_proj_transform)

        from gaussiansplatting.gaussian_renderer import render as gs_render
        N = gaussian.get_xyz.shape[0]
        bg = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        with latency_logger.timeit(f"{_p}.inverse_render_2d_to_3d") if latency_logger else nullcontext():
            M_3d_by_res: Dict[int, Float[Tensor, "N attn_len"]] = {}
            for res in key_cross_attn_by_res:
                side = int(res ** 0.5)
                H, W = side, side
                weights = torch.zeros(N, attn_len, device=device, dtype=torch.float32)
                weights_cnt = torch.zeros(N, device=device, dtype=torch.int32)
                for v_idx, cam in enumerate(key_cams):
                    cam_low = _camera_at_res(cam, H, W)
                    M_v = key_cross_attn_by_res[res][v_idx]
                    if M_v.dim() == 2:
                        M_v = M_v.view(H, W, attn_len)
                    for c in range(attn_len):
                        img_w = M_v[:, :, c].unsqueeze(0)
                        gaussian.apply_weights(cam_low, weights[:, c:c + 1], weights_cnt, img_w)
                M_3d_by_res[res] = weights / (weights_cnt.unsqueeze(1).float().clamp(min=1) + 1e-7)

        with latency_logger.timeit(f"{_p}.render_consistent_maps") if latency_logger else nullcontext():
            M_con_by_view_res: Dict[int, Dict[int, Float[Tensor, "H W attn_len"]]] = {}
            all_cams_list = cams
            for view_idx in range(n_views):
                M_con_by_view_res[view_idx] = {}
                cam = all_cams_list[view_idx]
                for res in M_3d_by_res:
                    side = int(res ** 0.5)
                    M_3d = M_3d_by_res[res]
                    cam_low = _camera_at_res(cam, side, side)
                    maps_c = []
                    for c in range(attn_len):
                        color_c = M_3d[:, c].unsqueeze(-1).expand(-1, 3)
                        pkg = gs_render(cam_low, gaussian, pipe, bg, override_color=color_c)
                        maps_c.append(pkg["render"][0])
                    M_con_by_view_res[view_idx][res] = torch.stack(maps_c, dim=-1)



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
        with latency_logger.timeit(f"{_p}.install_consistent_processor") if latency_logger else nullcontext():
            consistent_processor = ConsistentCrossAttnProcessor(backup_processor=None)
            for name, mod in self.unet.named_modules():
                if name.endswith(".attn2") and hasattr(mod, "processor"):
                    consistent_processor.backup_processor = mod.processor
                    break
            for name, mod in self.unet.named_modules():
                if name.endswith(".attn2") and hasattr(mod, "processor"):
                    mod.processor = ConsistentCrossAttnProcessor(backup_processor=original_attn2_processors.get(name, mod.processor))

        with latency_logger.timeit(f"{_p}.noise_and_init_latents") if latency_logger else nullcontext():
            noise = torch.randn_like(latents)
            latents = self.scheduler.add_noise(latents, noise, t)
            positive_text_embedding, negative_text_embedding, _ = text_embeddings.chunk(3)
            split_image_cond_latents, _, zero_image_cond_latents = image_cond_latents.chunk(3)
            if skip_key_views_in_target_loop:
                # key views are fully denoised — fix them and only denoise target views
                key_indices_set = set(key_indices)
                target_indices = [i for i in range(n_views) if i not in key_indices_set]
                latents[key_indices] = key_edited
            else:
                # original behavior: denoise all views including key views
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
                scales_anchor = [(int(r ** 0.5), int(r ** 0.5)) for r in collected_resolutions]
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
        # Phase 3. Target-view denoise loop
        #   - run denoising only on target views (keys are fixed if skipped)
        #   - use consistent cross-attention and (optionally) 3D-anchor
        # ------------------------------------------------------------------
        with latency_logger.timeit(f"{_p}.target_denoise_loop") if latency_logger else nullcontext():
            # latents_target: views to denoise, shaped [n_target, 4, H, W]
            latents_target = latents[target_indices]
            use_normal_attn_target = True
            for t_step in self.scheduler.timesteps:
                with latency_logger.timeit(f"{_p}.target_denoise_loop.per_timestep_setup") if latency_logger else nullcontext():
                    if t_step < 100:
                        if not use_normal_attn_target:
                            self.use_normal_unet()
                            use_normal_attn_target = True
                    else:
                        if use_normal_attn_target:
                            register_normal_attn_flag(self.unet, False)
                            use_normal_attn_target = False
                    num_batches = (n_target + camera_batch_size - 1) // camera_batch_size
                    pivotal_idx = torch.randint(camera_batch_size, (num_batches,), device=device) + torch.arange(0, n_target, camera_batch_size, device=device)[:num_batches]
                    pivotal_idx = pivotal_idx.clamp(max=n_target - 1)
                    register_pivotal(self.unet, True)
                    key_cams_batch = [target_cams[i] for i in pivotal_idx.cpu().tolist()]
                    latent_model_input = torch.cat([latents_target[pivotal_idx]] * 3)
                    pivot_text_embeddings = torch.cat([
                        target_pos_emb[pivotal_idx], target_neg_emb[pivotal_idx], target_neg_emb[pivotal_idx]
                    ], dim=0)
                    pivot_image_cond_latents = torch.cat([
                        target_split_cond[pivotal_idx], target_split_cond[pivotal_idx], target_zero_cond[pivotal_idx]
                    ], dim=0)
                    latent_model_input = torch.cat([latent_model_input, pivot_image_cond_latents], dim=1)

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

                noise_pred_text = []
                noise_pred_image = []
                noise_pred_uncond = []
                for b in range(0, n_target, camera_batch_size):
                    batch_end = min(b + camera_batch_size, n_target)
                    batch_target_indices = target_indices[b:batch_end]  # original view indices
                    batch_local_indices = list(range(b, batch_end))     # local indices into latents_target
                    with latency_logger.timeit(f"{_p}.target_denoise_loop.batch_prep") if latency_logger else nullcontext():
                        data = {"attn_len": attn_len}
                        for res in collected_resolutions:
                            maps_batch = []
                            for i in batch_target_indices:
                                if i < len(M_con_by_view_res) and res in M_con_by_view_res.get(i, {}):
                                    m = M_con_by_view_res[i][res]
                                    maps_batch.append(m.reshape(-1, attn_len))
                            if maps_batch:
                                data[res] = torch.stack(maps_batch, dim=0).to(device)
                        _use_consistent = data.get("attn_len") and any(k in data for k in collected_resolutions)
                        _attn_map_val = data if _use_consistent else None
                        for _, mod in _attn2_modules:
                            setattr(mod, "_consistent_attn_map_current", _attn_map_val)
                        if _feature_injection_mode == "3d_anchor" and anchor_3d_cache is not None:
                            _batch_idx = b // camera_batch_size
                            _pivot_global = target_indices[pivotal_idx[_batch_idx].item()] if _batch_idx < len(pivotal_idx) else target_indices[0]
                            with latency_logger.timeit(f"{_p}.target_denoise_loop.register_anchor_3d_cache") if latency_logger else nullcontext():
                                register_anchor_3d_cache(
                                    self.unet, anchor_3d_cache,
                                    batch_view_indices=batch_target_indices,
                                    injection_lambda=_injection_lambda,
                                    pivot_view_index=_pivot_global,
                                    injection_3d_anchor_style=_injection_3d_anchor_style,
                                )
                        _batch_idx = b // camera_batch_size
                        _pivot_this_batch = pivotal_idx[_batch_idx] % camera_batch_size if _batch_idx < len(pivotal_idx) else 0
                        for _, mod in _dge_blocks:
                            setattr(mod, "batch_idx", _batch_idx)
                            setattr(mod, "cams", [target_cams[j] for j in batch_local_indices])
                            setattr(mod, "pivot_this_batch", _pivot_this_batch)
                            setattr(mod, "key_cams", key_cams_batch)
                            setattr(mod, "epipolar_constrains", {})
                        batch_model_input = torch.cat([latents_target[b:batch_end]] * 3)
                        batch_text_embeddings = torch.cat([
                            target_pos_emb[b:batch_end], target_neg_emb[b:batch_end], target_neg_emb[b:batch_end]
                        ], dim=0)
                        batch_image_cond_latents = torch.cat([
                            target_split_cond[b:batch_end], target_split_cond[b:batch_end], target_zero_cond[b:batch_end]
                        ], dim=0)
                        batch_model_input = torch.cat([batch_model_input, batch_image_cond_latents], dim=1)
                    with latency_logger.timeit(f"{_p}.target_denoise_loop.batch_forward") if latency_logger else nullcontext():
                        if latency_logger:
                            set_unet_latency_prefix(f"{_p}.target_denoise_loop.batch_forward.unet_forward")
                        try:
                            with latency_logger.timeit(f"{_p}.target_denoise_loop.batch_forward.unet_forward") if latency_logger else nullcontext():
                                batch_noise_pred = self.forward_unet(batch_model_input, t_step.unsqueeze(0).expand(len(batch_local_indices) * 3).to(device), encoder_hidden_states=batch_text_embeddings)
                        finally:
                            if latency_logger:
                                set_unet_latency_prefix(None)
                    if _feature_injection_mode == "3d_anchor":
                        with latency_logger.timeit(f"{_p}.target_denoise_loop.unregister_anchor_3d_cache") if latency_logger else nullcontext():
                            unregister_anchor_3d_cache(self.unet)
                    batch_noise_pred_text, batch_noise_pred_image, batch_noise_pred_uncond = batch_noise_pred.chunk(3)
                    noise_pred_text.append(batch_noise_pred_text)
                    noise_pred_image.append(batch_noise_pred_image)
                    noise_pred_uncond.append(batch_noise_pred_uncond)
                with latency_logger.timeit(f"{_p}.target_denoise_loop.merge_and_step") if latency_logger else nullcontext():
                    for _, mod in _attn2_modules:
                        setattr(mod, "_consistent_attn_map_current", None)
                    noise_pred_text = torch.cat(noise_pred_text, dim=0)
                    noise_pred_image = torch.cat(noise_pred_image, dim=0)
                    noise_pred_uncond = torch.cat(noise_pred_uncond, dim=0)
                    noise_pred = (
                        noise_pred_uncond
                        + self.cfg.guidance_scale * (noise_pred_text - noise_pred_image)
                        + self.cfg.condition_scale * (noise_pred_image - noise_pred_uncond)
                    )
                    latents_target = self.scheduler.step(noise_pred, t_step, latents_target).prev_sample
            # write denoised views back
            latents[target_indices] = latents_target
            if skip_key_views_in_target_loop:
                latents[key_indices] = key_edited

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
        _prefix = EDIT_MULTIVIEW_PREFIX if use_multiview_path else EDIT_ALL_VIEW_PREFIX

        # So that DGE blocks (make_dge_block) can record latency under the correct hierarchy
        if latency_logger is not None:
            register_latency_logger(self.unet, latency_logger)

        with latency_logger.timeit(f"{_prefix}.encode_images"):
            latents = self.encode_images(rgb_BCHW_HW8)

        cond_rgb_BCHW = cond_rgb.permute(0, 3, 1, 2)
        cond_rgb_BCHW_HW8 = F.interpolate(
            cond_rgb_BCHW,
            (RH, RW),
            mode="bilinear",
            align_corners=False,
        )

        with latency_logger.timeit(f"{_prefix}.encode_cond_images"):
            cond_latents = self.encode_cond_images(cond_rgb_BCHW_HW8)

        temp = torch.zeros(batch_size).to(rgb.device)

        with latency_logger.timeit(f"{_prefix}.text_embeddings"):
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
            with latency_logger.timeit(f"{EDIT_ALL_VIEW_PREFIX}.compute_grad_sds"):
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
            if use_multiview and gaussian is not None and pipe is not None and (key_indices is not None and len(key_indices) > 0 or _strat is not None and _nkv is not None):
                key_view_camera_ids = kwargs.get("key_view_camera_ids", None)
                with latency_logger.timeit(f"{_prefix}.edit_latents_multiview"):
                    edit_latents = self.edit_latents_multiview(
                        text_embeddings, latents, cond_latents, t, cams,
                        # key_indices=key_indices, 
                        # key_view_camera_ids=key_view_camera_ids,
                        gaussian=gaussian, pipe=pipe, prompt_text=prompt_text,
                        latency_logger=latency_logger,
                        skip_key_views_in_target_loop=self.cfg.skip_key_views_in_target_loop,
                        key_selection_strategy=_strat,
                        num_key_views=_nkv,
                    )
            else:
                gp_cache = kwargs.get("gp_cache", None)
                key_cam_indices = kwargs.get("key_cam_indices", None)
                edit_latents = self.edit_latents(
                    text_embeddings, latents, cond_latents, t, cams, latency_logger,
                    gp_cache=gp_cache, key_cam_indices=key_cam_indices,
                )
            with latency_logger.timeit(f"{_prefix}.decode_latents"):
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


