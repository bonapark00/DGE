from dataclasses import dataclass, field
from typing import Optional
import contextlib
import math
import random
import time
from re import T

from PIL import Image, ImageDraw, ImageFont
import PIL.Image as PILImage
from tqdm import tqdm
import cv2
import numpy as np
import sys
import shutil
import torch
import torch.nn.functional as F
import threestudio
import os
from threestudio.systems.base import BaseLift3DSystem

from threestudio.utils.typing import *
from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.scene import GaussianModel

from gaussiansplatting.arguments import (
    PipelineParams,
    OptimizationParams,
)
from omegaconf import OmegaConf

from argparse import ArgumentParser
from threestudio.utils.misc import get_device
from threestudio.utils.perceptual import PerceptualLoss
from threestudio.utils.sam import LangSAMTextSegmentor
from threestudio.utils.latency import LatencyLogger
from threestudio.utils.dge_utils import register_normal_attn_flag


from CLIP.utils.image_utils import img_normalize, clip_normalize
from CLIP.scene.VGG import get_features
import CLIP

clip_model = CLIP.load_model()

def _compute_keep_mask_xyz_percent(
    xyz: torch.Tensor,
    prune_z_bottom_percent: float,
    prune_y_top_percent: float,
    prune_x_both_percent: float,
) -> torch.Tensor:
    """
    Returns a boolean keep_mask (True=keep) using the same percent logic as generate_by_lens.py.
    Percent values are in [0, 100], where 3.0 means 3%.
    """
    n_pts = int(xyz.shape[0])
    device = xyz.device
    keep_mask = torch.ones(n_pts, dtype=torch.bool, device=device)
    if n_pts == 0:
        return keep_mask

    if prune_z_bottom_percent > 0:
        z = xyz[:, 2]
        k_z = max(0, int(round(n_pts * (prune_z_bottom_percent / 100.0))))
        if k_z > 0:
            _, idx_smallest_z = torch.topk(z, k_z, largest=False)
            keep_mask[idx_smallest_z] = False

    if prune_y_top_percent > 0:
        y = xyz[:, 1]
        k_y = max(0, int(round(n_pts * (prune_y_top_percent / 100.0))))
        if k_y > 0:
            _, idx_largest_y = torch.topk(y, k_y, largest=True)
            keep_mask[idx_largest_y] = False

    if prune_x_both_percent > 0:
        x = xyz[:, 0]
        k_x = max(0, int(round(n_pts * (prune_x_both_percent / 100.0))))
        if k_x > 0:
            _, idx_smallest_x = torch.topk(x, k_x, largest=False)
            _, idx_largest_x = torch.topk(x, k_x, largest=True)
            keep_mask[idx_smallest_x] = False
            keep_mask[idx_largest_x] = False

    return keep_mask


def _hard_prune_gaussians_by_mask(gaussians: GaussianModel, keep_mask: torch.Tensor) -> int:
    """
    Hard prune that changes Parameter sizes.
    Safe only if called BEFORE optimizer is created (i.e., before gaussians.training_setup()).
    Returns number removed.
    """
    if keep_mask.dtype != torch.bool:
        keep_mask = keep_mask.bool()
    keep_mask = keep_mask.to(gaussians.get_xyz.device)
    n_before = int(keep_mask.shape[0])
    n_remove = n_before - int(keep_mask.sum().item())
    if n_remove <= 0:
        return 0
    gaussians._xyz = torch.nn.Parameter(
        gaussians._xyz[keep_mask].detach().clone().requires_grad_(True)
    )
    gaussians._features_dc = torch.nn.Parameter(
        gaussians._features_dc[keep_mask].detach().clone().requires_grad_(True)
    )
    gaussians._features_rest = torch.nn.Parameter(
        gaussians._features_rest[keep_mask].detach().clone().requires_grad_(True)
    )
    gaussians._opacity = torch.nn.Parameter(
        gaussians._opacity[keep_mask].detach().clone().requires_grad_(True)
    )
    gaussians._scaling = torch.nn.Parameter(
        gaussians._scaling[keep_mask].detach().clone().requires_grad_(True)
    )
    gaussians._rotation = torch.nn.Parameter(
        gaussians._rotation[keep_mask].detach().clone().requires_grad_(True)
    )
    # Keep internal book-keeping tensors in sync (required by densify code).
    if hasattr(gaussians, "mask") and isinstance(getattr(gaussians, "mask"), torch.Tensor):
        if gaussians.mask.shape[0] == n_before:
            gaussians.mask = gaussians.mask[keep_mask].detach().clone()
    if hasattr(gaussians, "_generation") and isinstance(getattr(gaussians, "_generation"), torch.Tensor):
        if gaussians._generation.shape[0] == n_before:
            gaussians._generation = gaussians._generation[keep_mask].detach().clone()
    # Optional buffers (may exist depending on GaussianModel variant)
    if hasattr(gaussians, "max_radii2D") and gaussians.max_radii2D is not None:
        if gaussians.max_radii2D.shape[0] == n_before:
            gaussians.max_radii2D = gaussians.max_radii2D[keep_mask].detach().clone()
    if hasattr(gaussians, "xyz_gradient_accum") and gaussians.xyz_gradient_accum is not None:
        if gaussians.xyz_gradient_accum.shape[0] == n_before:
            gaussians.xyz_gradient_accum = (
                gaussians.xyz_gradient_accum[keep_mask].detach().clone()
            )
    if hasattr(gaussians, "denom") and gaussians.denom is not None:
        if gaussians.denom.shape[0] == n_before:
            gaussians.denom = gaussians.denom[keep_mask].detach().clone()
    return n_remove


@threestudio.register("dge-system")
class DGE(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        gs_source: str = None

        per_editing_step: int = -1
        edit_begin_step: int = 0
        edit_until_step: int = 4000

        densify_until_iter: int = 4000
        densify_from_iter: int = 0
        densification_interval: int = 100
        max_densify_percent: float = 0.01

        gs_lr_scaler: float = 1
        gs_final_lr_scaler: float = 1            
        color_lr_scaler: float = 1
        opacity_lr_scaler: float = 1
        scaling_lr_scaler: float = 1
        rotation_lr_scaler: float = 1

        # lr
        mask_thres: float = 0.5
        mask_max_ratio: float = 0.9  # Skip views where mask covers > this fraction of image (0~1)
        mask_min_ratio: float = 0.01  # Skip views where mask covers < this fraction (likely failed seg)
        mask_outlier_iqr: float = 1.5  # IQR multiplier for outlier detection; exclude views outside [Q1-k*IQR, Q3+k*IQR]
        # update_mask: view population and count
        mask_view_population: str = "colmap_views"  # "colmap_views" = colmap_cameras_for_mask, "train_cameras" = train_dataset.scene.cameras
        mask_num_views: int = 30  # number of views to use for masking (top by distance from Gaussian center)
        max_grad: float = 1e-7
        min_opacity: float = 0.005
        
        # floater pruning
        prune_floater_at_step: int = -1  # -1: disabled, otherwise prune at this step
        prune_floater_depth_range: Tuple[float, float] = (10.0, 15.0)  # depth range for object
        prune_floater_threshold: float = 1.5  # multiplier: prune points beyond depth_range * threshold

        seg_prompt: str = ""
        target_prompt: str = ""

        # cache
        cache_overwrite: bool = True
        cache_dir: str = ""


        # DDS-lite
        dds_t_range: Tuple[float, float] = (0.02, 0.5)  # narrower t range for stability
        dds_cfg_scale: float = 7.5

        # anchor
        anchor_weight_init: float = 0.1
        anchor_weight_init_g0: float = 1.0
        anchor_weight_multiplier: float = 2
        
        training_args: dict = field(default_factory=dict)

        use_masked_image: bool = False
        local_edit: bool = False

        # guidance 
        camera_update_per_step: int = 500
        added_noise_schedule: List[int] = field(default_factory=[999, 200, 200, 21])    
        
        
        mask_update_at_step: int = 500 ## BONA
        # number of novel views used when updating mask at mask_update_at_step
        mask_update_view_num: int = 10

        # Gaussian-Provenance Sparse Cross-View Attention (Version B)
        use_gaussian_provenance: bool = False  # If True, use GP sparse attention instead of epipolar DGE
        gp_K: int = 2            # top-K gaussians per pixel
        gp_M_half: int = 1       # half-window size → M=(2*M_half+1)^2 neighbours
        gp_vis_eps: float = 0.05 # depth tolerance for visibility test
        gp_alpha_tau: float = 0.4  # scale for alpha gating (top-1 weight / tau)

        # Multiview edit (key-view cross-attn + inverse-render + consistent map for target views)
        use_multiview_edit: bool = False  # If True, use edit_multiview instead of edit_all_view
        # Key view selection: "uniform" = linspace; "uniform_random" = one random per interval; "lens_fps" = LENS Step 5 energy-weighted FPS
        multiview_edit_key_selection_strategy: str = "uniform"

        # Warp-and-Refine (propagate_and_refine_views) settings
        use_warp_refine: bool = False  # If True, use warp-and-refine instead of DGE guidance
        warp_refine_color_fit_steps: int = 100  # Color-only fitting steps per anchor
        warp_refine_anchor_ip2p_steps: int = 50  # IP2P steps for anchor view (stronger edit → increase)
        warp_refine_ip2p_strength: float = 0.75  # Blend: IP2P vs warped for non-anchor (stronger → 0.75~1.0)
        warp_refine_ip2p_steps: int = 20  # IP2P steps for propagate-and-refine views
        warp_refine_image_guidance_scale: float = 1.5  # Lower = stronger edit (e.g. 1.2, 1.0)
        warp_refine_text_guidance_scale: float = 7.5  # Higher = stronger edit (e.g. 9.0, 10.0)
        warp_refine_color_lr: float = 5e-3  # LR for color-only fitting

    cfg: Config

    def configure(self) -> None:
        self.gaussian = GaussianModel(
            sh_degree=0,
            anchor_weight_init_g0=self.cfg.anchor_weight_init_g0,
            anchor_weight_init=self.cfg.anchor_weight_init,
            anchor_weight_multiplier=self.cfg.anchor_weight_multiplier,
        )
        bg_color = [1, 1, 1] if False else [0, 0, 0]
        self.background_tensor = torch.tensor(
            bg_color, dtype=torch.float32, device="cuda"
        )
        self.edit_frames = {}
        self.origin_frames = {}
        self.edit_frames_order = []  # view_sorted 순서를 저장
        self.perceptual_loss = PerceptualLoss().eval().to(get_device())
        self.text_segmentor = LangSAMTextSegmentor().to(get_device())

        if len(self.cfg.cache_dir) > 0:
            print("Using cache directory: ", self.cfg.cache_dir)
            self.cache_dir = os.path.join(self.cfg.cache_dir, "edit_cache")
            os.makedirs(self.cache_dir, exist_ok=True)
        else:
            print("No cache directory provided")
            self.cache_dir = os.path.join(self.cfg.cache_dir, "edit_cache", self.cfg.gs_source.replace("/", "-"))
            os.makedirs(self.cache_dir, exist_ok=True)

    @torch.no_grad()
    def update_mask(self, seg_object=None, save_name="mask") -> None:
        _lat = getattr(self, "_latency_logger", None)
        _time = (lambda _n, **_kw: _lat.timeit(_n, **_kw)) if _lat is not None else lambda _n, **_kw: contextlib.nullcontext()

        train_dataset = self.trainer.datamodule.train_dataset

        with _time("update_mask"):
            with _time("update_mask.setup"):
                # View population: colmap_views or train_cameras (from cfg)
                use_colmap = (
                    self.cfg.mask_view_population == "colmap_views"
                    and hasattr(train_dataset, "colmap_cameras_for_mask")
                    and train_dataset.colmap_cameras_for_mask is not None
                )
                if use_colmap:
                    all_cameras = train_dataset.colmap_cameras_for_mask
                    print("[Lens] update_mask: using Colmap cameras for mask (mask_view_population=colmap_views)")
                else:
                    all_cameras = train_dataset.scene.cameras
                    print("[Lens] update_mask: using train_dataset.scene.cameras (mask_view_population=train_cameras)")

                # Compute Gaussian center from model
                gaussian_center = None
                if hasattr(self, 'gaussian') and self.gaussian is not None:
                    try:
                        xyz = self.gaussian.get_xyz
                        if isinstance(xyz, torch.Tensor):
                            xyz_np = xyz.detach().cpu().numpy()
                        else:
                            xyz_np = np.array(xyz)
                        gaussian_center = np.mean(xyz_np, axis=0).astype(np.float32)
                        print(f"Computed Gaussian center from model: {gaussian_center}")
                    except Exception as e:
                        threestudio.warn(f"Failed to get Gaussian center from model: {e}")

                if gaussian_center is None:
                    cam_centers = []
                    for cam in all_cameras:
                        center = cam.camera_center
                        if isinstance(center, torch.Tensor):
                            center = center.detach().cpu().numpy()
                        cam_centers.append(center)
                    cam_centers = np.array(cam_centers)
                    gaussian_center = np.median(cam_centers, axis=0)
                    print(f"Using median of camera centers as object center: {gaussian_center}")

                gaussian_center_tensor = torch.tensor(gaussian_center, device=get_device(), dtype=torch.float32)
                camera_distances = []
                for idx, cam in enumerate(all_cameras):
                    camera_center = cam.camera_center
                    if isinstance(camera_center, torch.Tensor):
                        camera_center = camera_center.to(get_device())
                    else:
                        camera_center = torch.tensor(camera_center, device=get_device(), dtype=torch.float32)
                    distance = torch.norm(camera_center - gaussian_center_tensor).item()
                    camera_distances.append((idx, distance))
                camera_distances.sort(key=lambda x: x[1], reverse=True)
                n_views = min(self.cfg.mask_num_views, len(camera_distances))
                view_list = [idx for idx, _ in camera_distances[:n_views]]
                print(f"Selected {n_views} cameras with largest distance from Gaussian center (mask_num_views={self.cfg.mask_num_views}): {[f'{idx}(dist={dist:.2f})' for idx, dist in camera_distances[:n_views]]}")

                print(f"View list for segmentation: {view_list}")

                print(f"Segment with prompt: {seg_object}")
                mask_cache_dir = os.path.join(
                    self.cache_dir, seg_object + f"_{save_name}_{len(view_list)}_view"
                )
                gs_mask_path = os.path.join(mask_cache_dir, "gs_mask.pt")

            if (seg_object == self.cfg.target_prompt) or not os.path.exists(gs_mask_path) or self.cfg.cache_overwrite:
                os.makedirs(mask_cache_dir, exist_ok=True)
                weights = torch.zeros_like(self.gaussian._opacity)
                weights_cnt = torch.zeros_like(self.gaussian._opacity, dtype=torch.int32)
                threestudio.info(f"Segmentation with prompt: {seg_object}")

                use_colmap_for_mask = use_colmap

                # Pass 1: collect mask ratios for all views
                collected = []
                with _time("update_mask.pass1"):
                    for id in tqdm(view_list, desc="update_mask pass1"):
                        cur_path = os.path.join(mask_cache_dir, "{:0>4d}.png".format(id))
                        cur_path_viz = os.path.join(
                            mask_cache_dir, "viz_{:0>4d}.png".format(id)
                        )
                        with _time("update_mask.pass1.select_cam"):
                            if use_colmap_for_mask:
                                cur_cam = train_dataset.colmap_cameras_for_mask[id]
                            else:
                                cur_cam = train_dataset.scene.cameras[id]

                        with _time("update_mask.pass1.render", sync_cuda=True):
                            if seg_object == self.cfg.target_prompt:
                                cur_batch = {
                                    "index": id,
                                    "camera": [cur_cam],
                                    "height": train_dataset.height,
                                    "width": train_dataset.width,
                                }
                                out = self(cur_batch)["comp_rgb"]
                                out_to_save = (
                                        out[0].cpu().detach().numpy().clip(0.0, 1.0) * 255.0
                                ).astype(np.uint8)
                                with _time("update_mask.pass1.io"):
                                    out_to_save = cv2.cvtColor(out_to_save, cv2.COLOR_RGB2BGR)
                                    cv2.imwrite(cur_path, out_to_save)
                                    cached_image = cv2.cvtColor(cv2.imread(cur_path), cv2.COLOR_BGR2RGB)
                                with _time("update_mask.pass1.to_tensor"):
                                    image_to_segment = torch.tensor(
                                        cached_image / 255, device="cuda", dtype=torch.float32
                                    )[None]

                            elif seg_object == self.cfg.seg_prompt:
                                if use_colmap_for_mask:
                                    cur_batch = {
                                        "index": id,
                                        "camera": [cur_cam],
                                        "height": train_dataset.height,
                                        "width": train_dataset.width,
                                    }
                                    out = self(cur_batch)["comp_rgb"]
                                    image_to_segment = out.detach().clone()
                                else:
                                    image_to_segment = self.origin_frames[id]

                        with _time("update_mask.pass1.segment", sync_cuda=True):
                            mask = self.text_segmentor(image_to_segment, seg_object)[0].to(get_device())

                        with _time("update_mask.pass1.ratio_filter"):
                            mask_ratio = mask[0].float().mean().item()

                            # Hard bounds: skip extreme failures
                            if mask_ratio > self.cfg.mask_max_ratio:
                                print(f"[update_mask] Skipping view {id}: mask_ratio={mask_ratio:.3f} > mask_max_ratio={self.cfg.mask_max_ratio}")
                                continue
                            if mask_ratio < self.cfg.mask_min_ratio:
                                print(f"[update_mask] Skipping view {id}: mask_ratio={mask_ratio:.3f} < mask_min_ratio={self.cfg.mask_min_ratio}")
                                continue

                        with _time("update_mask.pass1.collect"):
                            collected.append((id, mask, mask_ratio, cur_cam, image_to_segment, cur_path, cur_path_viz))

                # Outlier detection: exclude views with ratio outside [Q1 - k*IQR, Q3 + k*IQR]
                with _time("update_mask.outlier_filter"):
                    if len(collected) >= 3:
                        ratios = np.array([r for _, _, r, _, _, _, _ in collected])
                        q1, q3 = np.percentile(ratios, [25, 75])
                        iqr = q3 - q1
                        k = self.cfg.mask_outlier_iqr
                        low = max(0.0, q1 - k * iqr)
                        high = min(1.0, q3 + k * iqr)
                        inlier_indices = [i for i, (_, _, r, _, _, _, _) in enumerate(collected) if low <= r <= high]
                        outlier_count = len(collected) - len(inlier_indices)
                        if outlier_count > 0:
                            print(f"[update_mask] Outlier filter: Q1={q1:.3f} Q3={q3:.3f} IQR={iqr:.3f} -> [{low:.3f}, {high:.3f}], excluding {outlier_count} views")
                            for i in range(len(collected)):
                                if i not in inlier_indices:
                                    print(f"  - view {collected[i][0]}: ratio={collected[i][2]:.3f} (outlier)")
                            collected = [collected[i] for i in inlier_indices]

                # Pass 2: apply_weights only for inlier views
                with _time("update_mask.pass2"):
                    for id, mask, mask_ratio, cur_cam, image_to_segment, cur_path, cur_path_viz in tqdm(collected, desc="update_mask pass2"):
                        mask_to_save = ( # todo: target_prompt에 대한 마스크는 저장할 필요 없음.
                                mask[0]
                                .cpu()  
                                .detach()[..., None]
                                .repeat(1, 1, 3)
                                .numpy()
                                .clip(0.0, 1.0)
                                * 255.0
                        ).astype(np.uint8)
                        # cv2.imwrite(cur_path, mask_to_save)

                        masked_image = image_to_segment.detach().clone()[0]
                        masked_image[mask[0].bool()] *= 0.3
                        masked_image_to_save = (
                                masked_image.cpu().detach().numpy().clip(0.0, 1.0) * 255.0
                        ).astype(np.uint8)
                        masked_image_to_save = cv2.cvtColor(
                            masked_image_to_save, cv2.COLOR_RGB2BGR
                        )
                        # cv2.imwrite(cur_path_viz, masked_image_to_save)
                        self.gaussian.apply_weights(cur_cam, weights, weights_cnt, mask)

                with _time("update_mask.save_weights"):
                    weights /= weights_cnt + 1e-7

                    selected_mask = weights > self.cfg.mask_thres
                    selected_mask = selected_mask[:, 0]
                    # torch.save(selected_mask, gs_mask_path)
            else:
                with _time("update_mask.load_cache"):
                    print("load cache")
                    mask_cache_dir = os.path.join(
                        self.cache_dir, seg_object + f"_{save_name}_65_view"
                    )
                    for id in tqdm(self.edit_view_index):
                        cur_path = os.path.join(mask_cache_dir, "{:0>4d}.png".format(id))
                        cur_mask = cv2.imread(cur_path)
                        cur_mask = torch.tensor(
                            cur_mask / 255, device="cuda", dtype=torch.float32
                        )[..., 0][None]
                    selected_mask = torch.load(gs_mask_path)

            with _time("update_mask.apply_mask"):
                self.gaussian.set_mask(selected_mask)
                self.gaussian.apply_grad_mask(selected_mask)

    @torch.no_grad()
    def prune_distant_floater_gaussians(self):
        """
        Prune Gaussian points that are too far from editing views.
        If editing views have objects at depth 10-15, prune points beyond that range.
        """
        if not hasattr(self, 'edit_view_index') or len(self.edit_view_index) == 0:
            print("No editing views available for floater pruning")
            return
        
        print(f"Pruning distant floater Gaussians using {len(self.edit_view_index)} editing views...")
        
        # Get Gaussian point positions
        gaussian_xyz = self.gaussian.get_xyz  # (N, 3)
        num_gaussians = gaussian_xyz.shape[0]
        
        # Depth range for object (10-15)
        depth_min, depth_max = self.cfg.prune_floater_depth_range
        depth_threshold = depth_max * self.cfg.prune_floater_threshold  # e.g., 15 * 1.5 = 22.5
        
        # Track which points should be pruned
        # A point is pruned if it's too far from ALL editing views
        # Start with all points marked as "too far" (True = prune)
        prune_mask = torch.ones(num_gaussians, dtype=torch.bool, device=gaussian_xyz.device)
        
        # Find the maximum object depth across all editing views
        max_object_depth_all_views = depth_min
        
        # First pass: find maximum object depth across all views
        for view_idx in self.edit_view_index:
            cam = self.trainer.datamodule.train_dataset.scene.cameras[view_idx]
            
            # Render depth for this view
            render_pkg = render(cam, self.gaussian, self.pipe, self.background_tensor)
            depth_map = render_pkg["depth_3dgs"]  # (H, W)
            
            # Find pixels with depth in the object range (10-15)
            object_depth_mask = (depth_map >= depth_min) & (depth_map <= depth_max)
            
            if object_depth_mask.sum() > 0:
                max_object_depth_view = depth_map[object_depth_mask].max().item()
                max_object_depth_all_views = max(max_object_depth_all_views, max_object_depth_view)
        
        # Calculate threshold based on maximum object depth
        if max_object_depth_all_views > depth_min:
            threshold_distance = max_object_depth_all_views * self.cfg.prune_floater_threshold
        else:
            # Fallback: use depth_max if no object pixels found
            threshold_distance = depth_max * self.cfg.prune_floater_threshold
        
        print(f"Object depth range: [{depth_min}, {depth_max}], Max found: {max_object_depth_all_views:.2f}, Threshold: {threshold_distance:.2f}")
        
        # Second pass: check each editing view and mark points that are within threshold
        for view_idx in self.edit_view_index:
            cam = self.trainer.datamodule.train_dataset.scene.cameras[view_idx]
            
            # Calculate distance from each Gaussian point to camera center
            camera_center = cam.camera_center
            if isinstance(camera_center, torch.Tensor):
                camera_center = camera_center.to(gaussian_xyz.device)
            else:
                camera_center = torch.tensor(camera_center, device=gaussian_xyz.device, dtype=torch.float32)
            
            # Distance from each Gaussian to camera center
            gaussian_to_camera = gaussian_xyz - camera_center[None, :]
            distances = torch.norm(gaussian_to_camera, dim=1)  # (N,)
            
            # Points that are within reasonable distance from this view (keep these)
            # If a point is within threshold from at least one view, keep it
            within_threshold = distances <= threshold_distance
            prune_mask = prune_mask & (~within_threshold)  # Only prune if too far from ALL views
        
        # Final prune mask: points that are too far from ALL editing views
        num_to_prune = prune_mask.sum().item()
        num_total = num_gaussians
        
        if num_to_prune > 0:
            print(f"Pruning {num_to_prune}/{num_total} ({100*num_to_prune/num_total:.2f}%) distant Gaussian points")
            self.gaussian.prune_points(prune_mask)
            torch.cuda.empty_cache()
        else:
            print(f"No distant floater Gaussians found to prune")

    def compute_dds_loss(
        self,
        images: torch.Tensor,       # (B, H, W, C), rendered from current 3DGS
        batch_index: list,           # view indices for origin_frames lookup
    ) -> torch.Tensor:
        """Lightweight DDS loss: single UNet call per view with cosine stabilisation.

        Steps:
            1. Encode rendered image and source image to latent space.
            2. Sample t from a narrow range, add noise.
            3. Batch=3 UNet call: [uncond, target_prompt, source_prompt].
            4. delta = cfg * (eps_tgt - eps_src),  base = eps_tgt - eps_uncond.
            5. L = w(t) * (1 - cos(delta, base)).
        """
        g = self.guidance  # DGEGuidance – owns vae, unet, scheduler, etc.
        device = images.device

        # --- resize & encode -------------------------------------------------
        B, H, W, C = images.shape
        factor = 512 / max(W, H)
        factor = math.ceil(min(W, H) * factor / 64) * 64 / min(W, H)
        rh = int((H * factor) // 64) * 64
        rw = int((W * factor) // 64) * 64

        rgb_BCHW = images.permute(0, 3, 1, 2)
        rgb_rs = F.interpolate(rgb_BCHW, (rh, rw), mode="bilinear", align_corners=False)
        latents = g.encode_images(rgb_rs)  # (B,4,h,w)  – keeps grad

        # source (original) images → deterministic encode (no grad needed)
        src_imgs = torch.cat([self.origin_frames[idx] for idx in batch_index], dim=0)
        src_BCHW = src_imgs.permute(0, 3, 1, 2)
        src_rs = F.interpolate(src_BCHW, (rh, rw), mode="bilinear", align_corners=False)
        with torch.no_grad():
            src_latents = g.encode_images(src_rs)  # (B,4,h,w)

        # IP2P image conditioning: concat source image latents on channel dim
        src_cond = src_rs * 2.0 - 1.0
        with torch.no_grad():
            image_cond_latents = g.vae.encode(
                src_cond.to(g.weights_dtype)
            ).latent_dist.mode().to(latents.dtype)  # (B,4,h,w) deterministic
        zero_image_cond = torch.zeros_like(image_cond_latents)

        # --- sample timestep --------------------------------------------------
        t_lo = int(g.num_train_timesteps * self.cfg.dds_t_range[0])
        t_hi = int(g.num_train_timesteps * self.cfg.dds_t_range[1])
        t = torch.randint(t_lo, max(t_hi, t_lo + 1), (1,), device=device, dtype=torch.long).expand(B)

        noise = torch.randn_like(latents)
        z_t = g.scheduler.add_noise(latents, noise, t)
        # Same noise for source latent so that the unconditioned noise cancels
        z_t_src = g.scheduler.add_noise(src_latents, noise, t)

        # --- text embeddings --------------------------------------------------
        # target prompt embeddings are already in self.prompt_processor
        prompt_utils = self.prompt_processor()
        temp = torch.zeros(B, device=device)
        text_emb = prompt_utils.get_text_embeddings(temp, temp, temp, False)
        # text_emb: (2B, 77, 768) = [target, uncond]
        tgt_emb, uncond_emb = text_emb.chunk(2)  # each (B, 77, 768)

        # source prompt embeddings (lazy-cached)
        if not hasattr(self, '_src_text_emb') or self._src_text_emb is None:
            self._src_text_emb = self._encode_prompt(self.cfg.seg_prompt)
        src_emb = self._src_text_emb.expand(B, -1, -1)  # (B, 77, 768)

        # --- single batched UNet call: [uncond, target, source] ---------------
        # For IP2P: model input = [noisy_latent ; image_cond] on channel dim
        # uncond uses zero image cond; target & source use real image cond
        z_t_batch = torch.cat([z_t, z_t, z_t_src], dim=0)  # (3B,4,h,w)
        img_cond_batch = torch.cat([zero_image_cond, image_cond_latents, image_cond_latents], dim=0)
        model_input = torch.cat([z_t_batch, img_cond_batch], dim=1)  # (3B,8,h,w)
        text_batch = torch.cat([uncond_emb, tgt_emb, src_emb], dim=0)  # (3B,77,768)

        # Ensure normal attention (no DGE extended attn for this lightweight call)
        register_normal_attn_flag(g.unet, True)
        with torch.no_grad():
            eps_pred = g.forward_unet(model_input, t, encoder_hidden_states=text_batch)
        register_normal_attn_flag(g.unet, False)

        eps_uncond, eps_tgt, eps_src = eps_pred.chunk(3)

        # --- DDS gradient with cosine stabilisation --------------------------
        cfg_s = self.cfg.dds_cfg_scale
        delta = cfg_s * (eps_tgt - 0.1 * eps_src)        # directional edit signal
        base = eps_tgt - eps_uncond                 # CFG direction (stabiliser)

        # Cosine weighting: scale gradient by alignment with CFG direction
        delta_flat = delta.reshape(B, -1)
        base_flat = base.reshape(B, -1)
        cos_weight = F.cosine_similarity(delta_flat, base_flat, dim=1)  # (B,)
        cos_weight = cos_weight.clamp(min=0.0).view(B, 1, 1, 1)  # ignore negative

        w = (1 - g.alphas[t]).float().view(B, 1, 1, 1)
        # DDS gradient: w(t) * cos_weight * delta
        grad = (w * cos_weight * delta).detach()
        grad = torch.nan_to_num(grad)

        # SDS-style loss: creates gradient path back to Gaussian params
        target = (latents - grad).detach()
        loss_dds = 0.5 * F.mse_loss(latents, target, reduction="sum") / B

        return loss_dds

    def compute_lite_ism_loss(
        self,
        images: torch.Tensor,       # (B, H, W, C), rendered from current 3DGS
        batch_index: list,           # view indices for origin_frames lookup
    ) -> torch.Tensor:
        """Lite-ISM loss: ISM (Interval Score Matching) adapted for 1-step prediction.

        Compared to DDS:
          - Removes source-prompt branch → 2-batch UNet call (33% faster).
          - Predicts x0 directly via DDIM x0-prediction formula.
          - Loss = MSE(latents, stop_grad(x0_pred)), pulling 3DGS toward
            the diffusion model's one-shot denoised target (strong edit signal).

        Steps:
            1. Encode rendered image z and source image for IP2P conditioning.
            2. Sample t, add noise → z_t.
            3. 2-batch UNet call: [uncond, tgt] with IP2P image cond.
            4. CFG: eps_hat = uncond + cfg_scale * (tgt - uncond).
            5. x0_pred = (z_t - sigma_t * eps_hat) / alpha_t  (DDIM x0 formula).
            6. loss = 0.5 * MSE(latents, stop_grad(x0_pred)) / B.
        """
        g = self.guidance  # DGEGuidance – owns vae, unet, scheduler, etc.
        device = images.device

        # --- resize & encode -------------------------------------------------
        B, H, W, C = images.shape
        factor = 512 / max(W, H)
        factor = math.ceil(min(W, H) * factor / 64) * 64 / min(W, H)
        rh = int((H * factor) // 64) * 64
        rw = int((W * factor) // 64) * 64

        rgb_BCHW = images.permute(0, 3, 1, 2)
        rgb_rs = F.interpolate(rgb_BCHW, (rh, rw), mode="bilinear", align_corners=False)
        latents = g.encode_images(rgb_rs)  # (B,4,h,w) – keeps grad

        # source (original) images for IP2P conditioning (no grad)
        src_imgs = torch.cat([self.origin_frames[idx] for idx in batch_index], dim=0)
        src_BCHW = src_imgs.permute(0, 3, 1, 2)
        src_rs = F.interpolate(src_BCHW, (rh, rw), mode="bilinear", align_corners=False)
        src_cond = src_rs * 2.0 - 1.0
        with torch.no_grad():
            image_cond_latents = g.vae.encode(
                src_cond.to(g.weights_dtype)
            ).latent_dist.mode().to(latents.dtype)  # (B,4,h,w) deterministic
        zero_image_cond = torch.zeros_like(image_cond_latents)

        # --- sample timestep --------------------------------------------------
        t_lo = int(g.num_train_timesteps * self.cfg.dds_t_range[0])
        t_hi = int(g.num_train_timesteps * self.cfg.dds_t_range[1])
        t = torch.randint(t_lo, max(t_hi, t_lo + 1), (1,), device=device, dtype=torch.long).expand(B)

        noise = torch.randn_like(latents)
        z_t = g.scheduler.add_noise(latents, noise, t)

        # --- text embeddings: 2-batch [uncond, tgt] ---------------------------
        prompt_utils = self.prompt_processor()
        temp = torch.zeros(B, device=device)
        text_emb = prompt_utils.get_text_embeddings(temp, temp, temp, False)
        tgt_emb, uncond_emb = text_emb.chunk(2)  # each (B, 77, 768)

        # --- 2-batch UNet call: [uncond, tgt] (no src branch) -----------------
        # IP2P: model input = concat(noisy_latent, image_cond) on channel dim
        z_t_batch = torch.cat([z_t, z_t], dim=0)                                       # (2B,4,h,w)
        img_cond_batch = torch.cat([zero_image_cond, image_cond_latents], dim=0)        # (2B,4,h,w)
        model_input = torch.cat([z_t_batch, img_cond_batch], dim=1)                    # (2B,8,h,w)
        text_batch = torch.cat([uncond_emb, tgt_emb], dim=0)                           # (2B,77,768)

        register_normal_attn_flag(g.unet, True)
        with torch.no_grad():
            eps_pred = g.forward_unet(model_input, t, encoder_hidden_states=text_batch)
        register_normal_attn_flag(g.unet, False)

        eps_uncond, eps_tgt = eps_pred.chunk(2)

        # --- CFG noise prediction ---------------------------------------------
        cfg_s = self.cfg.dds_cfg_scale
        eps_hat = eps_uncond + cfg_s * (eps_tgt - eps_uncond)  # (B,4,h,w)
        eps_hat = torch.nan_to_num(eps_hat)

        # --- x0 prediction (DDIM formula) -------------------------------------
        # alphas_cumprod[t] = alpha_t^2  →  alpha_t = sqrt(alphas[t])
        # sigma_t = sqrt(1 - alphas[t])
        alpha_t = g.alphas[t].float().sqrt().view(B, 1, 1, 1)   # sqrt(ᾱ_t)
        sigma_t = (1.0 - g.alphas[t]).float().sqrt().view(B, 1, 1, 1)  # sqrt(1-ᾱ_t)

        # x0_pred = (z_t - sigma_t * eps_hat) / alpha_t
        x0_pred = (z_t - sigma_t * eps_hat) / (alpha_t + 1e-8)
        x0_pred = x0_pred.detach()  # stop gradient through diffusion model

        # --- ISM loss: 3DGS latents → x0_pred target -------------------------
        loss_ism = 0.5 * F.mse_loss(latents, x0_pred, reduction="sum") / B

        return loss_ism

    @torch.no_grad()
    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode a single prompt string using the guidance pipeline's text encoder.
        Returns (1, 77, 768) text embeddings cached on device."""
        pipe = self.guidance.pipe
        tok = pipe.tokenizer(
            [prompt],
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            emb = pipe.text_encoder(tok.input_ids.to(self.guidance.device))[0]
        return emb  # (1, 77, 768)

    def on_validation_epoch_end(self):
        pass

    def forward(self, batch: Dict[str, Any], renderbackground=None, local=False) -> Dict[str, Any]:
        if renderbackground is None:
            renderbackground = self.background_tensor
        images = []
        depths = []
        semantics = []
        masks = []
        self.viewspace_point_list = []
        self.gaussian.localize = local
        for id, cam in enumerate(batch["camera"]):

            render_pkg = render(cam, self.gaussian, self.pipe, renderbackground)
            image, viewspace_point_tensor, _, radii = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )
            self.viewspace_point_list.append(viewspace_point_tensor)

            if id == 0:
                self.radii = radii
            else:
                self.radii = torch.max(radii, self.radii)

            depth = render_pkg["depth_3dgs"]
            depth = depth.permute(1, 2, 0)

            semantic_map = render(
                cam,
                self.gaussian,
                self.pipe,
                renderbackground,
                override_color=self.gaussian.mask[..., None].float().repeat(1, 3),
            )["render"]
            semantic_map = torch.norm(semantic_map, dim=0)
            semantic_map = semantic_map > 0.8
            semantic_map_viz = image.detach().clone()
            semantic_map_viz = semantic_map_viz.permute(
                1, 2, 0
            )  # 3 512 512 to 512 512 3
            semantic_map_viz[semantic_map] = 0.40 * semantic_map_viz[
                semantic_map
            ] + 0.60 * torch.tensor([1.0, 0.0, 0.0], device="cuda")
            semantic_map_viz = semantic_map_viz.permute(
                2, 0, 1
            )  # 512 512 3 to 3 512 512

            semantics.append(semantic_map_viz)
            masks.append(semantic_map)
            image = image.permute(1, 2, 0)
            images.append(image)
            depths.append(depth)

        self.gaussian.localize = False  # reverse

        images = torch.stack(images, 0)
        depths = torch.stack(depths, 0)
        semantics = torch.stack(semantics, dim=0)
        masks = torch.stack(masks, dim=0)

        render_pkg["semantic"] = semantics
        render_pkg["masks"] = masks
        self.visibility_filter = self.radii > 0.0
        render_pkg["comp_rgb"] = images
        render_pkg["depth"] = depths
        render_pkg["opacity"] = depths / (depths.max() + 1e-5)
        return {
            **render_pkg,
        }

    def render_all_view(self, cache_name):
        cache_dir = os.path.join(self.cache_dir, cache_name)
        os.makedirs(cache_dir, exist_ok=True)
        with torch.no_grad():
            for id in tqdm(range(self.trainer.datamodule.train_dataset.total_view_num)):
                cur_path = os.path.join(cache_dir, "{:0>4d}.png".format(id))
                need_render = not os.path.exists(cur_path) or self.cfg.cache_overwrite
                if need_render:
                    cur_cam = self.trainer.datamodule.train_dataset.scene.cameras[id]
                    cur_batch = {
                        "index": id,
                        "camera": [cur_cam],
                        "height": self.trainer.datamodule.train_dataset.height,
                        "width": self.trainer.datamodule.train_dataset.width,
                    }
                    out = self(cur_batch)["comp_rgb"]
                    out_to_save = (
                            out[0].cpu().detach().numpy().clip(0.0, 1.0) * 255.0
                    ).astype(np.uint8)
                    out_to_save = cv2.cvtColor(out_to_save, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(cur_path, out_to_save)
                img_bgr = cv2.imread(cur_path)
                if img_bgr is None or img_bgr.size == 0:
                    # Corrupted or missing PNG (e.g. parallel write collision): re-render
                    if os.path.exists(cur_path):
                        try:
                            os.remove(cur_path)
                        except OSError:
                            pass
                    cur_cam = self.trainer.datamodule.train_dataset.scene.cameras[id]
                    cur_batch = {
                        "index": id,
                        "camera": [cur_cam],
                        "height": self.trainer.datamodule.train_dataset.height,
                        "width": self.trainer.datamodule.train_dataset.width,
                    }
                    out = self(cur_batch)["comp_rgb"]
                    out_to_save = (
                            out[0].cpu().detach().numpy().clip(0.0, 1.0) * 255.0
                    ).astype(np.uint8)
                    out_to_save = cv2.cvtColor(out_to_save, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(cur_path, out_to_save)
                    img_bgr = cv2.imread(cur_path)
                cached_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                origin_img = torch.tensor(
                    cached_image / 255, device="cuda", dtype=torch.float32
                )[None]  # [1, H, W, 3]
                self.origin_frames[id] = origin_img

                # Also store the rendered origin image on the camera object itself
                # so that downstream code can access it as a camera-attached tensor.
                cur_cam = self.trainer.datamodule.train_dataset.scene.cameras[id]
                img_chw = origin_img[0].permute(2, 0, 1).contiguous()  # [3, H, W]
                device = getattr(cur_cam, "data_device", img_chw.device)
                cur_cam.rendered_image_from_generated_view = img_chw.to(device)

    def on_before_backward(self, loss):
        """Called by Lightning before loss.backward(). Start timer for backward."""
        if hasattr(self, "_latency_logger") and self._latency_logger is not None:
            self._ts_backward_start = time.perf_counter()

    def on_after_backward(self):
        """Called by Lightning after loss.backward(). Record backward duration."""
        if hasattr(self, "_latency_logger") and self._latency_logger is not None and hasattr(self, "_ts_backward_start"):
            self._latency_logger.record("backward", time.perf_counter() - self._ts_backward_start)

    def on_before_optimizer_step(self, optimizer):
        with self._latency_logger.timeit("on_before_optimizer_step"):
            with torch.no_grad():
                if self.true_global_step < self.cfg.densify_until_iter:
                    with self._latency_logger.timeit("on_before_optimizer_step.densification_stats"):
                        viewspace_point_tensor_grad = torch.zeros_like(
                            self.viewspace_point_list[0]
                        )
                        for idx in range(len(self.viewspace_point_list)):
                            viewspace_point_tensor_grad = (
                                    viewspace_point_tensor_grad
                                    + self.viewspace_point_list[idx].grad
                            )
                        # Keep track of max radii in image-space for pruning
                        self.gaussian.max_radii2D[self.visibility_filter] = torch.max(
                            self.gaussian.max_radii2D[self.visibility_filter],
                            self.radii[self.visibility_filter],
                        )
                        self.gaussian.add_densification_stats(
                            viewspace_point_tensor_grad, self.visibility_filter
                        )
                    # Densification
                    if (
                            self.true_global_step >= self.cfg.densify_from_iter
                            and self.true_global_step % self.cfg.densification_interval == 0
                    ):  # 500 100
                        with self._latency_logger.timeit("on_before_optimizer_step.densify_and_prune"):
                            self.gaussian.densify_and_prune(
                                self.cfg.max_grad,
                                self.cfg.max_densify_percent,
                                self.cfg.min_opacity,
                                self.cameras_extent,
                                5,
                            )
        if hasattr(self, "_latency_logger") and self._latency_logger is not None:
            self._ts_optimizer_step_start = time.perf_counter()

    def on_after_optimizer_step(self, optimizer):
        """Called by Lightning after optimizer.step(). Record optimizer_step duration."""
        if hasattr(self, "_latency_logger") and self._latency_logger is not None and hasattr(self, "_ts_optimizer_step_start"):
            self._latency_logger.record("optimizer_step", time.perf_counter() - self._ts_optimizer_step_start)

    def validation_step(self, batch, batch_idx):
        batch["camera"] = [
            self.trainer.datamodule.train_dataset.scene.cameras[idx]
            for idx in batch["index"]
        ]
        out = self(batch)
        for idx in range(len(batch["index"])):
            cam_index = batch["index"][idx].item()
            self.save_image_grid(
                f"it{self.true_global_step}-val/{batch['index'][idx]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": self.origin_frames[cam_index][0],
                            "kwargs": {"data_format": "HWC"},
                        },
                        {
                            "type": "rgb",
                            "img": self.edit_frames[cam_index][0]
                            if cam_index in self.edit_frames
                            else torch.zeros_like(self.origin_frames[cam_index][0]),
                            "kwargs": {"data_format": "HWC"},
                        },
                    ]
                ),
                name=f"validation_step_{idx}",
                step=self.true_global_step,
            )
            self.save_image_grid(
                f"render_it{self.true_global_step}-val/{batch['index'][idx]}.png",
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][idx],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][idx],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                )
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["semantic"][idx].moveaxis(0, -1),
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "semantic" in out
                    else []
                ),
                name=f"validation_step_render_{idx}",
                step=self.true_global_step,
            )

    def test_step(self, batch, batch_idx):
        only_rgb = True  # TODO add depth test step
        bg_color = [1, 1, 1] if False else [0, 0, 0]
        batch["camera"] = [
            self.trainer.datamodule.val_dataset.scene.cameras[batch["index"]]
        ]
        testbackground_tensor = torch.tensor(
            bg_color, dtype=torch.float32, device="cuda"
        )

        out = self(batch, testbackground_tensor)
        if only_rgb:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                ),
                name="test_step",
                step=self.true_global_step,
            )
        else:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": batch["rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb" in batch
                    else []
                )
                + [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                )
                + (
                    [
                        {
                            "type": "grayscale",
                            "img": out["depth"][0],
                            "kwargs": {},
                        }
                    ]
                    if "depth" in out
                    else []
                )
                + [
                    {
                        "type": "grayscale",
                        "img": out["opacity"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ],
                name="test_step",
                step=self.true_global_step,
            )

    def _add_index_to_image(self, img, index):
        """이미지 상단에 인덱스 번호를 추가하는 헬퍼 메서드 (OpenCV 사용)"""
        try:
            # If configured to not draw texts on grids, return the original image.
            cfg = getattr(self, "cfg", None)
            if cfg is not None and hasattr(cfg, "save_image_grid_draw_texts"):
                if not bool(getattr(cfg, "save_image_grid_draw_texts")):
                    return img

            # 원본 형식 저장 (나중에 복원하기 위해)
            original_is_tensor = isinstance(img, torch.Tensor)
            
            # 이미지가 torch tensor인 경우 numpy로 변환
            if isinstance(img, torch.Tensor):
                img_np = img.detach().cpu().numpy()
            else:
                img_np = img.copy()
            
            # 배치 차원이 있으면 제거
            if len(img_np.shape) == 4:
                img_np = img_np[0]
            
            # CHW 형식이면 HWC로 변환
            if len(img_np.shape) == 3 and img_np.shape[0] == 3:
                img_np = img_np.transpose(1, 2, 0)
            
            # 원본 값 범위 저장
            if img_np.dtype != np.uint8:
                if img_np.max() <= 1.0:
                    # 0-1 범위의 float
                    img_for_cv = (img_np * 255.0).astype(np.uint8)
                    value_range = (0.0, 1.0)
                else:
                    # 0-255 범위의 float
                    img_for_cv = np.clip(img_np, 0, 255).astype(np.uint8)
                    value_range = (0.0, 255.0)
            else:
                img_for_cv = img_np.copy()
                value_range = (0.0, 255.0)
            
            # HWC 형식 확인 및 RGB로 변환
            if len(img_for_cv.shape) == 3 and img_for_cv.shape[2] == 3:
                # RGB를 BGR로 변환 (OpenCV는 BGR 사용)
                img_bgr = cv2.cvtColor(img_for_cv, cv2.COLOR_RGB2BGR)
                
                # 텍스트 설정
                text = str(index)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = max(0.7, min(img_bgr.shape[:2]) / 400.0)  # 이미지 크기에 비례
                thickness = max(1, int(font_scale * 2))
                
                # 텍스트 크기 계산
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                
                # 텍스트 위치 (상단 중앙)
                x = (img_bgr.shape[1] - text_width) // 2
                y = text_height + 10
                
                # 텍스트 테두리 그리기 (흰색)
                for dx in [-2, -1, 0, 1, 2]:
                    for dy in [-2, -1, 0, 1, 2]:
                        if dx != 0 or dy != 0:
                            cv2.putText(img_bgr, text, (x + dx, y + dy), font, font_scale, (255, 255, 255), thickness + 1, cv2.LINE_AA)
                
                # 텍스트 그리기 (검은색)
                cv2.putText(img_bgr, text, (x, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
                
                # BGR를 RGB로 다시 변환
                img_with_text = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
                
                # 원본 값 범위로 복원
                if value_range == (0.0, 1.0):
                    img_with_text = img_with_text / 255.0
                elif value_range == (0.0, 255.0) and img_np.dtype == np.uint8:
                    img_with_text = img_with_text.astype(np.uint8)
                
                # 원본이 tensor였으면 tensor로 변환
                if original_is_tensor:
                    img_with_text = torch.from_numpy(img_with_text).to(img.device if hasattr(img, 'device') else 'cpu')
                
                return img_with_text
            else:
                # 형식이 맞지 않으면 원본 반환
                print(f"Warning: Image shape {img_np.shape} is not supported, returning original")
                return img
        except Exception as e:
            import traceback
            print(f"Error in _add_index_to_image: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            print(f"Image type: {type(img)}, shape: {img.shape if hasattr(img, 'shape') else 'N/A'}")
            return img

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=5,
            name="test",
            step=self.true_global_step,
        )
        # Save camera params for origin rendering (used by metrics GT)
        # Use test_dataset to match test render output (it1500-test)
        try:
            test_ds = self.trainer.datamodule.test_dataset
            cameras_list = []
            indices = []
            for i in range(len(test_ds)):
                idx = test_ds.selected_views[i] if isinstance(test_ds.selected_views, (list, tuple)) else int(test_ds.selected_views[i].item())
                cam = test_ds.scene.cameras[idx]
                R = np.array(cam.R, dtype=np.float32) if not isinstance(cam.R, np.ndarray) else cam.R.astype(np.float32)
                T = np.array(cam.T, dtype=np.float32) if not isinstance(cam.T, np.ndarray) else cam.T.astype(np.float32)
                h = int(getattr(cam, "image_height", getattr(cam, "h", 512)))
                w = int(getattr(cam, "image_width", getattr(cam, "w", 512)))
                cameras_list.append({"R": R, "T": T, "FoVx": float(cam.FoVx), "FoVy": float(cam.FoVy), "h": h, "w": w})
                indices.append(int(idx))
            save_path = os.path.join(self.get_save_dir(), "cameras_for_origin.pt")
            torch.save({"cameras": cameras_list, "indices": indices}, save_path)
            print(f"[DGE] Saved {len(cameras_list)} cameras for origin rendering to {save_path}")
        except Exception as e:
            threestudio.warn(f"Failed to save cameras for origin: {e}")
        # save_list = []
        # # view_sorted 순서대로 저장 (순서가 저장되어 있으면 사용, 없으면 view index 오름차순)
        # if len(self.edit_frames_order) > 0:
        #     # view_sorted 순서대로 저장
        #     for index in self.edit_frames_order:
        #         if index in self.edit_frames:
        #             # 이미지에 인덱스 번호 추가
        #             img_with_index = self._add_index_to_image(self.edit_frames[index][0], index)
        #             save_list.append(
        #                 {
        #                     "type": "rgb",
        #                     "img": img_with_index,
        #                     "kwargs": {"data_format": "HWC"},
        #                 },
        #             )
        # else:
        #     # 순서 정보가 없으면 view index 오름차순으로 정렬
        #     for index, image in sorted(self.edit_frames.items(), key=lambda item: item[0]):
        #         # 이미지에 인덱스 번호 추가
        #         img_with_index = self._add_index_to_image(image[0], index)
        #         save_list.append(
        #             {
        #                 "type": "rgb",
        #                 "img": img_with_index,
        #                 "kwargs": {"data_format": "HWC"},
        #             },
        #         )
        # if len(save_list) > 0:
        #     self.save_image_grid(
        #         f"edited_images.png",
        #         save_list,
        #         name="edited_images",
        #         step=self.true_global_step,
        #     )

        save_path = self.get_save_path(f"last.ply")
        print("save_path", save_path)
        self.gaussian.save_ply(save_path)

    def configure_optimizers(self):
        self.parser = ArgumentParser(description="Training script parameters")
        self.edit_view_index = self.trainer.datamodule.train_dataset.edit_view_index
        self.edit_view_num = len(self.edit_view_index)
        opt = OptimizationParams(self.parser, self.trainer.max_steps, self.cfg.gs_lr_scaler, self.cfg.gs_final_lr_scaler, self.cfg.color_lr_scaler,
                                 self.cfg.opacity_lr_scaler, self.cfg.scaling_lr_scaler, self.cfg.rotation_lr_scaler, )
        self.gaussian.load_ply(self.cfg.gs_source)

        # ---------------------------------------------------------------
        # Optional HARD Gaussian pruning (same percent logic as generate_by_lens.py)
        # Read from datamodule config overrides:
        # - data.lens_prune_z_bottom_percent
        # - data.lens_prune_y_top_percent
        # - data.lens_prune_x_both_percent
        # This is placed here (before training_setup/optimizer creation) so it is safe.
        # ---------------------------------------------------------------
        dm_cfg = getattr(getattr(self.trainer, "datamodule", None), "cfg", None)
        # User preference: prune ONLY the bottom z-percentile (no y/x pruning).
        pz = float(getattr(dm_cfg, "lens_prune_z_bottom_percent", 0.0) or 0.0)
        py = 0.0
        px = 0.0
        if pz > 0:
            try:
                xyz = self.gaussian.get_xyz.detach()
                keep_mask = _compute_keep_mask_xyz_percent(xyz, pz, py, px)
                n_removed = _hard_prune_gaussians_by_mask(self.gaussian, keep_mask)
                if n_removed > 0:
                    threestudio.info(
                        f"[DGE prune] Hard-pruned {n_removed} Gaussians "
                        f"(z_bottom={pz}%). "
                        f"Remaining: {self.gaussian.get_xyz.shape[0]}"
                    )
            except Exception as e:
                threestudio.warn(f"[DGE prune] Failed to hard-prune Gaussians: {e}")

        self.gaussian.max_radii2D = torch.zeros(
            (self.gaussian.get_xyz.shape[0]), device="cuda"
        )
        self.cameras_extent = self.trainer.datamodule.train_dataset.scene.cameras_extent
        self.gaussian.spatial_lr_scale = self.cameras_extent
        
        # Set Gaussian model to dataset for MMR view selection
        if hasattr(self.trainer.datamodule.train_dataset, 'gaussian_model'):
            self.trainer.datamodule.train_dataset.gaussian_model = self.gaussian

        self.pipe = PipelineParams(self.parser)
        opt = OmegaConf.create(vars(opt))
        opt.update(self.cfg.training_args)
        self.gaussian.training_setup(opt)

        ret = {
            "optimizer": self.gaussian.optimizer,
        }

        return ret
    
    def edit_all_view(self, original_render_name, cache_name, update_camera=False, global_step=0):
        # if self.true_global_step >= self.cfg.camera_update_per_step * 2:
        #     self.guidance.use_normal_unet()
        
        # self.edited_cams = []
        if update_camera: ## 60개 view 중에서 max_view_num개만 랜덤하게 선택됨.
            with self._latency_logger.timeit("training_step_all.edit_all_view.update_editing_cameras"):
                self.trainer.datamodule.train_dataset.update_editing_cameras(random_seed = global_step + 1)
                self.edit_view_index = self.trainer.datamodule.train_dataset.edit_view_index
                sorted_train_view_list = sorted(self.edit_view_index)
                selected_views = torch.linspace(
                    0, len(sorted_train_view_list) - 1, self.trainer.datamodule.val_dataset.n_views, dtype=torch.int
                )
                self.trainer.datamodule.val_dataset.selected_views = [sorted_train_view_list[idx] for idx in selected_views]
        
        print(f"{self.true_global_step}th step, Camera view index: {self.edit_view_index}")

        self.edit_frames = {}
        self.edit_frames_order = []  # view_sorted 순서 초기화
        cache_dir = os.path.join(self.cache_dir, cache_name)
        original_render_cache_dir = os.path.join(self.cache_dir, original_render_name)
        os.makedirs(cache_dir, exist_ok=True)

        cameras = []
        images = []
        original_frames = []
        t_max_step = self.cfg.added_noise_schedule
        self.guidance.max_step = t_max_step[min(len(t_max_step)-1, self.true_global_step//self.cfg.camera_update_per_step)]
        with torch.no_grad():
            with self._latency_logger.timeit("training_step_all.edit_all_view.collect_cameras"):
                # self.edit_view_index = [_ for _ in range(len(self.trainer.datamodule.train_dataset.scene.cameras))]
                for id in self.edit_view_index:
                    cameras.append(self.trainer.datamodule.train_dataset.scene.cameras[id])

            # if self.cfg.guidance.edit_view_selection_strategy != "random":
            #     sorted_cam_idx = [_ for _ in range(len(cameras))]
            # else:
            sorted_cam_idx = self.sort_the_cameras_idx(cameras) # 카메라 x축 기준으로 정렬된 카메라의 인덱스
                # [19, 7, 16, 0, 1, 3, 11, 5, 17, 13, 8, 10, 15, 12, 18, 4, 14, 2, 6, 9] 인덱스랑은 상관이 없네

            view_sorted = [self.edit_view_index[idx] for idx in sorted_cam_idx]
            cams_sorted = [cameras[idx] for idx in sorted_cam_idx]     
                   
            for id in view_sorted:
                cur_path = os.path.join(cache_dir, "{:0>4d}.png".format(id))
                original_image_path = os.path.join(original_render_cache_dir, "{:0>4d}.png".format(id))
                cur_cam = self.trainer.datamodule.train_dataset.scene.cameras[id]
                cur_batch = {
                    "index": id,
                    "camera": [cur_cam],
                    "height": self.trainer.datamodule.train_dataset.height,
                    "width": self.trainer.datamodule.train_dataset.width,
                }
                with self._latency_logger.timeit("training_step_all.edit_all_view.render_single"):
                    out_pkg = self(cur_batch)
                out = out_pkg["comp_rgb"] ## 이게 forward해서 렌더링 결과 얻는 부분임!
                if self.cfg.use_masked_image:
                    with self._latency_logger.timeit("training_step_all.edit_all_view.apply_mask"):
                        out = out * out_pkg["masks"].unsqueeze(-1)
                images.append(out)
                assert os.path.exists(original_image_path)
                with self._latency_logger.timeit("training_step_all.edit_all_view.load_original"):
                    cached_image = cv2.cvtColor(cv2.imread(original_image_path), cv2.COLOR_BGR2RGB)
                    self.origin_frames[id] = torch.tensor(
                        cached_image / 255, device="cuda", dtype=torch.float32
                    )[None]
                original_frames.append(self.origin_frames[id])
            with self._latency_logger.timeit("training_step_all.edit_all_view.concat_batches"):
                images = torch.cat(images, dim=0) ## view들을 concat하여 배치로 만듦
                original_frames = torch.cat(original_frames, dim=0)

            with self._latency_logger.timeit("training_step_all.edit_all_view.guidance_batch"):
                edited_images = self.guidance(  # DGEGuidance.__call__
                    images,
                    original_frames,
                    self.prompt_processor(),
                    cams=cams_sorted,
                    latency_logger=self._latency_logger,
                    latency_prefix="training_step_all.edit_all_view.guidance_batch",
                )

            with self._latency_logger.timeit("training_step_all.edit_all_view.assign_outputs"):
                # view_sorted 순서를 저장 (나중에 이 순서대로 저장하기 위해)
                self.edit_frames_order = view_sorted.copy()
                for view_index_tmp in range(len(self.edit_view_index)):
                    self.edit_frames[view_sorted[view_index_tmp]] = edited_images['edit_images'][view_index_tmp].unsqueeze(0).detach().clone() # 1 H W C
    
        save_list = []
        # view_sorted 순서대로 저장 (순서가 저장되어 있으면 사용, 없으면 view index 오름차순)
        if len(self.edit_frames_order) > 0:
            # view_sorted 순서대로 저장
            for index in self.edit_frames_order:
                if index in self.edit_frames:
                    # 이미지에 인덱스 번호 추가
                    img_with_index = self._add_index_to_image(self.edit_frames[index][0], index)
                    save_list.append(
                        {
                            "type": "rgb",
                            "img": img_with_index,
                            "kwargs": {"data_format": "HWC"},
                        },
                    )
        else:
            # 순서 정보가 없으면 view index 오름차순으로 정렬
            for index, image in sorted(self.edit_frames.items(), key=lambda item: item[0]):
                # 이미지에 인덱스 번호 추가
                img_with_index = self._add_index_to_image(image[0], index)
                save_list.append(
                    {
                        "type": "rgb",
                        "img": img_with_index,
                        "kwargs": {"data_format": "HWC"},
                    },
                )
        if len(save_list) > 0:
            self.save_image_grid(
                f"edited_images.png",
                save_list,
                name="edited_images",
                step=self.true_global_step,
            )
        print("edited images saved to:", self.get_save_path("edited_images.png"))

    def edit_multiview(self, original_render_name, cache_name, update_camera=False, global_step=0):
        """
        Multiview edit: key views edited with cross-view (pivotal) attention; cross-attention
        from key views is inverse-rendered to 3D and re-rendered to consistent 2D maps;
        target views use these consistent maps in UNet upsampling cross-attention.
        """
        if getattr(self, "pipe", None) is None:
            self.parser = ArgumentParser(description="Training script parameters")
            self.pipe = PipelineParams(self.parser)
        _lat = getattr(self, "_latency_logger", None)
        _edit_wrap = _lat.timeit("training_step_all.edit_multiview") if _lat is not None else contextlib.nullcontext()
        with _edit_wrap:
            if update_camera:
                with self._latency_logger.timeit("training_step_all.edit_multiview.update_editing_cameras"):
                    self.trainer.datamodule.train_dataset.update_editing_cameras(random_seed=global_step + 1)
                    self.edit_view_index = self.trainer.datamodule.train_dataset.edit_view_index
                    sorted_train_view_list = sorted(self.edit_view_index)
                    selected_views = torch.linspace(
                        0, len(sorted_train_view_list) - 1, self.trainer.datamodule.val_dataset.n_views, dtype=torch.int
                    )
                    self.trainer.datamodule.val_dataset.selected_views = [sorted_train_view_list[idx] for idx in selected_views]

            print(f"{self.true_global_step}th step, Camera view index: {self.edit_view_index}")

            self.edit_frames = {}
            self.edit_frames_order = []
            cache_dir = os.path.join(self.cache_dir, cache_name)
            original_render_cache_dir = os.path.join(self.cache_dir, original_render_name)
            os.makedirs(cache_dir, exist_ok=True)

            cameras = []
            for id in self.edit_view_index:
                cameras.append(self.trainer.datamodule.train_dataset.scene.cameras[id])
            sorted_cam_idx = self.sort_the_cameras_idx(cameras)
            view_sorted = [self.edit_view_index[idx] for idx in sorted_cam_idx]
            cams_sorted = [cameras[idx] for idx in sorted_cam_idx]

            camera_batch_size = getattr(self.cfg.guidance, "camera_batch_size", 5)
            n_views = len(view_sorted)
        # Derive number of key views purely from editing cameras and batch size:
            #   num_key_views = max(1, floor(n_views / camera_batch_size))
            num_key_views = max(1, n_views // camera_batch_size)
            key_selection = getattr(self.cfg, "multiview_edit_key_selection_strategy", "uniform")
            # if key_selection == "lens_fps" and self.gaussian is not None and n_views >= num_key_views:
        #     from threestudio.data.gs_load import select_key_views_by_lens_fps
        #     with self._latency_logger.timeit("edit_multiview.key_selection_lens_fps"):
        #         key_indices = select_key_views_by_lens_fps(
        #             self.gaussian,
        #             cams_sorted,
        #             n_key=num_key_views,
        #             top_fraction=0.20,
        #             w_vis=0.6,
        #             w_can=0.4,
        #             device="cuda",
        #         )
        #     key_indices = [int(i) for i in key_indices]
        # elif key_selection == "uniform_random":
        #     # 구간을 num_key_views개로 나누고, 각 구간에서 랜덤으로 하나씩 선택
        #     segment_size = n_views / num_key_views
        #     key_indices = []
        #     for i in range(num_key_views):
        #         start = int(i * segment_size)
        #         end = min(int((i + 1) * segment_size), n_views) - 1
        #         if end < start:
        #             end = start
        #         key_indices.append(random.randint(start, end))
        #     key_indices = sorted(key_indices)
        #     key_indices = [int(i) for i in key_indices]
        # elif key_selection == "manual":
        #     key_indices = [_ for _ in range(20)]
        # else:
        #     key_indices = torch.linspace(0, n_views - 1, num_key_views, dtype=torch.long).tolist()
            #     key_indices = [int(i) for i in key_indices]
            # key_view_camera_ids = [view_sorted[i] for i in key_indices]

            images = []
            original_frames = []
            t_max_step = self.cfg.added_noise_schedule
            self.guidance.max_step = t_max_step[min(len(t_max_step) - 1, self.true_global_step // self.cfg.camera_update_per_step)]
            with torch.no_grad():
                with self._latency_logger.timeit("training_step_all.edit_multiview.render_and_load_originals"):
                    for id in view_sorted:
                        cur_path = os.path.join(cache_dir, "{:0>4d}.png".format(id))
                        original_image_path = os.path.join(original_render_cache_dir, "{:0>4d}.png".format(id))
                        cur_cam = self.trainer.datamodule.train_dataset.scene.cameras[id]
                        cur_batch = {
                            "index": id,
                            "camera": [cur_cam],
                            "height": self.trainer.datamodule.train_dataset.height,
                            "width": self.trainer.datamodule.train_dataset.width,
                        }
                        out_pkg = self(cur_batch)
                        out = out_pkg["comp_rgb"]
                        if self.cfg.use_masked_image:
                            out = out * out_pkg["masks"].unsqueeze(-1)
                        images.append(out)
                        assert os.path.exists(original_image_path)
                        cached_image = cv2.cvtColor(cv2.imread(original_image_path), cv2.COLOR_BGR2RGB)
                        self.origin_frames[id] = torch.tensor(cached_image / 255, device="cuda", dtype=torch.float32)[None]
                        original_frames.append(self.origin_frames[id])
                with self._latency_logger.timeit("training_step_all.edit_multiview.concat_batch"):
                    images = torch.cat(images, dim=0)
                    original_frames = torch.cat(original_frames, dim=0)

                with self._latency_logger.timeit("training_step_all.edit_multiview.guidance_batch"):
                    edited_images = self.guidance(
                        images,
                        original_frames,
                        self.prompt_processor(),
                        cams=cams_sorted,
                        latency_logger=self._latency_logger,
                        use_multiview=True,
                        gaussian=self.gaussian,
                        pipe=self.pipe,
                        # key_indices=key_indices,
                        # key_view_camera_ids=key_view_camera_ids,
                        prompt_text=getattr(self.cfg, "target_prompt", "") or "",
                        key_selection_strategy=key_selection,
                        num_key_views=num_key_views,
                        latency_prefix="training_step_all.edit_multiview.guidance_batch",
                    )

                with self._latency_logger.timeit("training_step_all.edit_multiview.assign_outputs"):
                    self.edit_frames_order = view_sorted.copy()
                    for view_index_tmp in range(len(self.edit_view_index)):
                        self.edit_frames[view_sorted[view_index_tmp]] = edited_images["edit_images"][view_index_tmp].unsqueeze(0).detach().clone()

            with self._latency_logger.timeit("training_step_all.edit_multiview.build_save_list"):
                save_list = []
                if len(self.edit_frames_order) > 0:
                    for index in self.edit_frames_order:
                        if index in self.edit_frames:
                            img_with_index = self._add_index_to_image(self.edit_frames[index][0], index)
                            save_list.append({"type": "rgb", "img": img_with_index, "kwargs": {"data_format": "HWC"}})
                else:
                    for index, image in sorted(self.edit_frames.items(), key=lambda item: item[0]):
                        img_with_index = self._add_index_to_image(image[0], index)
                        save_list.append({"type": "rgb", "img": img_with_index, "kwargs": {"data_format": "HWC"}})
            if len(save_list) > 0:
                with self._latency_logger.timeit("training_step_all.edit_multiview.save_image_grid"):
                    self.save_image_grid("edited_images_multiview.png", save_list, name="edited_images_multiview", step=self.true_global_step)
            threestudio.info("multiview edited images saved to: %s" % self.get_save_path("edited_images_multiview.png"))

            return # editing finished

    @torch.no_grad()
    def _render_single(self, cam) -> torch.Tensor:
        """Render a single camera view. Returns [H, W, C] float32 tensor in [0,1]."""
        render_pkg = render(cam, self.gaussian, self.pipe, self.background_tensor)
        return render_pkg["render"].permute(1, 2, 0)  # [H, W, 3]

    def _color_fit_to_anchor(
        self,
        anchor_cam,
        edited_anchor_image: torch.Tensor,  # [H, W, C] float32 [0,1]
    ):
        """
        Freeze geometry; optimise SH colour params to overfit to edited_anchor_image.
        Saves and restores original SH values afterwards so the caller may proceed
        with the fitted colours.

        Returns: (fitted_features_dc, fitted_features_rest) – detached clones.
        """
        # ------------------------------------------------------------------
        # 1. Save originals
        # ------------------------------------------------------------------
        orig_features_dc   = self.gaussian._features_dc.data.clone()
        orig_features_rest = self.gaussian._features_rest.data.clone()

        # ------------------------------------------------------------------
        # 2. Build a colour-only Adam optimiser (no optimizer state pollution)
        # ------------------------------------------------------------------
        color_lr = self.cfg.warp_refine_color_lr
        color_params = [
            {"params": [self.gaussian._features_dc],   "lr": color_lr,        "name": "f_dc"},
            {"params": [self.gaussian._features_rest],  "lr": color_lr / 20.0, "name": "f_rest"},
        ]
        color_optimizer = torch.optim.Adam(color_params, lr=0.0, eps=1e-15)

        # Target: [1, C, H, W] for easy perceptual loss; also keep HWC for L1
        target_hwc = edited_anchor_image.to(self.gaussian._features_dc.device)
        target_bchw = target_hwc.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]

        # ------------------------------------------------------------------
        # 3. Fitting loop – geometry gradients are not computed (no_grad on others)
        # ------------------------------------------------------------------
        for _ in range(self.cfg.warp_refine_color_fit_steps):
            color_optimizer.zero_grad()

            render_pkg = render(anchor_cam, self.gaussian, self.pipe, self.background_tensor)
            rendered = render_pkg["render"].permute(1, 2, 0)  # [H, W, C]

            loss = F.l1_loss(rendered, target_hwc)
            loss += self.perceptual_loss(
                rendered.permute(2, 0, 1).unsqueeze(0).contiguous(),
                target_bchw.contiguous(),
            ).sum() * 0.1

            loss.backward()

            # Zero out gradients for geometry parameters to be safe
            for pname in ["_xyz", "_scaling", "_rotation", "_opacity"]:
                p = getattr(self.gaussian, pname)
                if p.grad is not None:
                    p.grad.zero_()

            color_optimizer.step()

        # ------------------------------------------------------------------
        # 4. Snapshot fitted colours then restore originals
        # ------------------------------------------------------------------
        fitted_dc   = self.gaussian._features_dc.data.clone()
        fitted_rest = self.gaussian._features_rest.data.clone()

        self.gaussian._features_dc.data.copy_(orig_features_dc)
        self.gaussian._features_rest.data.copy_(orig_features_rest)

        del color_optimizer
        return fitted_dc, fitted_rest

    def propagate_and_refine_views(
        self,
        anchor_cam,
        edited_anchor_image: torch.Tensor,  # [H, W, C] float32 [0,1]
        target_cams: list,
        ip2p_pipe,
        prompt: str,
        seed: Optional[int] = None,
    ) -> list:
        """
        Warp-and-Refine: propagate the edited appearance from anchor_cam to
        target_cams via 3DGS geometry, then refine with vanilla IP2P SDEdit.

        Steps
        -----
        1. Colour-only fitting: overfit SH colours to edited_anchor_image at
           anchor_cam without touching geometry.
        2. Render target views with the fitted colours → warped images.
        3. Restore original colours.
        4. IP2P SDEdit refinement on each warped image (low strength).

        Returns
        -------
        List of [H, W, C] float32 tensors (one per target cam).
        """
        from PIL import Image as PILImage

        # Same seed for all views (deterministic, same edit style across views).
        ip2p_seed = int(seed) if seed is not None else 42

        # ------------------------------------------------------------------ #
        # 1. Colour fitting                                                   #
        # ------------------------------------------------------------------ #
        fitted_dc, fitted_rest = self._color_fit_to_anchor(
            anchor_cam, edited_anchor_image
        )

        # ------------------------------------------------------------------ #
        # 2. Render target views with fitted colours                          #
        # ------------------------------------------------------------------ #
        orig_dc   = self.gaussian._features_dc.data.clone()
        orig_rest = self.gaussian._features_rest.data.clone()

        self.gaussian._features_dc.data.copy_(fitted_dc)
        self.gaussian._features_rest.data.copy_(fitted_rest)

        warped_images = []  # list of [H, W, C] tensors
        with torch.no_grad():
            for cam in target_cams:
                warped = self._render_single(cam)  # [H, W, C]
                warped_images.append(warped)

        # Restore originals immediately
        self.gaussian._features_dc.data.copy_(orig_dc)
        self.gaussian._features_rest.data.copy_(orig_rest)

        # ------------------------------------------------------------------ #
        # 3. IP2P refinement (vanilla InstructPix2Pix, optional blend)       #
        # ------------------------------------------------------------------ #
        strength     = self.cfg.warp_refine_ip2p_strength  # used as blend factor between warped & edited
        num_steps    = self.cfg.warp_refine_ip2p_steps
        img_guidance = self.cfg.warp_refine_image_guidance_scale
        txt_guidance = self.cfg.warp_refine_text_guidance_scale

        refined = []
        # Prepare execution device for diffusers generator
        exec_device = getattr(ip2p_pipe, "_execution_device", None) or ip2p_pipe.unet.device
        with torch.no_grad():
            for idx, warped_hwc in enumerate(warped_images):
                # Convert to PIL for the diffusers pipeline
                warped_np  = (warped_hwc.cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
                warped_pil = PILImage.fromarray(warped_np)
                H, W       = warped_np.shape[:2]

                # Anchor edit as the "original" image condition for IP2P
                anchor_np  = (edited_anchor_image.cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
                anchor_pil = PILImage.fromarray(anchor_np)

                # Same seed for all views
                generator = torch.Generator(device=str(exec_device)).manual_seed(ip2p_seed)
                out_pil = ip2p_pipe(
                    prompt=prompt,
                    image=warped_pil,
                    num_inference_steps=num_steps,
                    image_guidance_scale=img_guidance,
                    guidance_scale=txt_guidance,
                    generator=generator,
                ).images[0]

                # Back to float tensor [H, W, C]
                out_np = np.array(out_pil.resize((W, H))).astype(np.float32) / 255.0
                # Optional SDEdit-style blend with warped input using strength in [0,1]
                if 0.0 < float(strength) < 1.0:
                    warped_f = warped_np.astype(np.float32) / 255.0
                    alpha = float(strength)
                    out_np = alpha * out_np + (1.0 - alpha) * warped_f
                refined.append(torch.from_numpy(out_np).to(warped_hwc.device))

        return refined  # List of [H, W, C] float32 tensors

    def edit_all_view_warp_refine(
        self,
        original_render_name: str,
        cache_name: str,
        ip2p_pipe,
        update_camera: bool = False,
        global_step: int = 0,
    ):
        """
        Alternative to edit_all_view that uses Warp-and-Refine propagation
        instead of the DGE attention-based guidance.

        For each camera-update cycle:
          1. Edit the first (anchor) view with vanilla IP2P.
          2. Propagate & refine to all remaining views via propagate_and_refine_views.
          3. Store results in self.edit_frames.
        """
        if update_camera:
            with self._latency_logger.timeit("training_step_all.edit_all_view_warp_refine.update_editing_cameras"):
                self.trainer.datamodule.train_dataset.update_editing_cameras(
                    random_seed=global_step + 1
                )
                self.edit_view_index = self.trainer.datamodule.train_dataset.edit_view_index
                sorted_train_view_list = sorted(self.edit_view_index)
                selected_views = torch.linspace(
                    0,
                    len(sorted_train_view_list) - 1,
                    self.trainer.datamodule.val_dataset.n_views,
                    dtype=torch.int,
                )
                self.trainer.datamodule.val_dataset.selected_views = [
                    sorted_train_view_list[idx] for idx in selected_views
                ]

        print(f"{self.true_global_step}th step [warp-refine], Camera view index: {self.edit_view_index}")

        self.edit_frames = {}
        self.edit_frames_order = []
        cache_dir               = os.path.join(self.cache_dir, cache_name)
        original_render_cache   = os.path.join(self.cache_dir, original_render_name)
        os.makedirs(cache_dir, exist_ok=True)

        # ------------------------------------------------------------------ #
        # Collect & sort cameras (same ordering as edit_all_view)             #
        # ------------------------------------------------------------------ #
        with self._latency_logger.timeit("training_step_all.edit_all_view_warp_refine.collect_and_sort_cameras"):
            cameras = [
                self.trainer.datamodule.train_dataset.scene.cameras[i]
                for i in self.edit_view_index
            ]
            sorted_cam_idx = self.sort_the_cameras_idx(cameras)
            view_sorted  = [self.edit_view_index[i] for i in sorted_cam_idx]
            cams_sorted  = [cameras[i]              for i in sorted_cam_idx]

        # Reload origin frames
        with self._latency_logger.timeit("training_step_all.edit_all_view_warp_refine.reload_origin_frames"):
            with torch.no_grad():
                for vid in view_sorted:
                    orig_path = os.path.join(original_render_cache, "{:0>4d}.png".format(vid))
                    assert os.path.exists(orig_path), f"Missing origin render: {orig_path}"
                    img_bgr = cv2.imread(orig_path)
                    cached  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    self.origin_frames[vid] = torch.tensor(
                        cached / 255, device="cuda", dtype=torch.float32
                    )[None]

        # ------------------------------------------------------------------ #
        # Anchor view: edit with vanilla IP2P (full-strength)                 #
        # ------------------------------------------------------------------ #
        with self._latency_logger.timeit("training_step_all.edit_all_view_warp_refine.anchor_total"):
            # Use the middle view in the sorted circular ordering as the anchor (\"central\" view)
            mid_idx   = len(view_sorted) // 2
            anchor_vid = view_sorted[mid_idx]
            anchor_cam = cams_sorted[mid_idx]
            print(f"[warp-refine] Anchor view index: {anchor_vid} (mid_idx={mid_idx}, total_views={len(view_sorted)})")

            with self._latency_logger.timeit("training_step_all.edit_all_view_warp_refine.anchor_render"):
                with torch.no_grad():
                    anchor_rendered_hwc = self._render_single(anchor_cam)  # [H, W, C]

            with self._latency_logger.timeit("training_step_all.edit_all_view_warp_refine.anchor_ip2p"):
                anchor_rendered_np  = (anchor_rendered_hwc.cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
                anchor_rendered_pil = PILImage.fromarray(anchor_rendered_np)
                H, W = anchor_rendered_np.shape[:2]

                prompt = self.cfg.target_prompt

                with torch.no_grad():
                    anchor_edited_pil = ip2p_pipe(
                        prompt=prompt,
                        image=anchor_rendered_pil,
                        num_inference_steps=getattr(
                            self.cfg, "warp_refine_anchor_ip2p_steps", 50
                        ),
                        image_guidance_scale=self.cfg.warp_refine_image_guidance_scale,
                        guidance_scale=self.cfg.warp_refine_text_guidance_scale,
                    ).images[0]

                anchor_edited_np  = np.array(anchor_edited_pil.resize((W, H))).astype(np.float32) / 255.0
                anchor_edited_hwc = torch.from_numpy(anchor_edited_np).to("cuda")

            # Store anchor edit
            self.edit_frames[anchor_vid] = anchor_edited_hwc.unsqueeze(0).detach().clone()
            self.edit_frames_order.append(anchor_vid)

        # ------------------------------------------------------------------ #
        # Target views: warp-and-refine                                       #
        # ------------------------------------------------------------------ #
        target_vids  = view_sorted[1:]
        target_cams  = cams_sorted[1:]

        if len(target_cams) > 0:
            with self._latency_logger.timeit("training_step_all.edit_all_view_warp_refine.propagate_and_refine"):
                refined_list = self.propagate_and_refine_views(
                    anchor_cam=anchor_cam,
                    edited_anchor_image=anchor_edited_hwc,
                    target_cams=target_cams,
                    ip2p_pipe=ip2p_pipe,
                    prompt=prompt,
                )

            with self._latency_logger.timeit("training_step_all.edit_all_view_warp_refine.assign_refined"):
                for vid, refined_hwc in zip(target_vids, refined_list):
                    self.edit_frames[vid] = refined_hwc.unsqueeze(0).detach().clone()
                    self.edit_frames_order.append(vid)

        # ------------------------------------------------------------------ #
        # Save grid for inspection                                             #
        # ------------------------------------------------------------------ #
        with self._latency_logger.timeit("training_step_all.edit_all_view_warp_refine.save_grid"):
            save_list = []
            for vid in self.edit_frames_order:
                if vid in self.edit_frames:
                    img_with_idx = self._add_index_to_image(self.edit_frames[vid][0], vid)
                    save_list.append(
                        {"type": "rgb", "img": img_with_idx, "kwargs": {"data_format": "HWC"}}
                    )
            if save_list:
                self.save_image_grid(
                    "edited_images_wr.png", save_list, name="edited_images_wr", step=self.true_global_step
                )
        print("[warp-refine] edited images saved to:", self.get_save_path("edited_images_wr.png"))
        return

    # ---------------------------------------------------------------------- #
    # Version B: Gaussian-Provenance Sparse Cross-View Attention editing      #
    # ---------------------------------------------------------------------- #

    def build_gp_cache(
        self,
        cams_sorted: list,
        key_cam_indices: list,
        scales: list = None,
        K: int = 2,
        M_half: int = 1,
        vis_eps: float = 0.05,
        alpha_tau: float = 0.4,
    ):
        """
        Build and store the gaussian-provenance attention cache used by
        edit_all_view_gaussian_provenance.

        Should be called once before the diffusion loop (or whenever the
        camera set / edit_view_index changes).  The cache is geometry-static
        so it is safe to reuse across diffusion steps as long as Gaussian
        positions are not updated between editing rounds.

        Args:
            cams_sorted      : Sorted list of camera objects (all edit views).
            key_cam_indices  : Indices into cams_sorted that are the key/pivotal
                               views (e.g. [0] or [0, n//2]).
            scales           : List of (H, W) tuples matching UNet latent scales.
                               Defaults to [(64,64), (32,32), (16,16), (8,8)].
            K, M_half, vis_eps, alpha_tau : passed through to
                               build_gaussian_provenance_cache.
        """
        from threestudio.utils.dge_utils import build_gaussian_provenance_cache

        if scales is None:
            scales = [(64, 64), (32, 32), (16, 16), (8, 8)]

        with self._latency_logger.timeit("training_step_all.edit_all_view_gaussian_provenance.build_gp_cache"):
            self._gp_cache = build_gaussian_provenance_cache(
                gaussian         = self.gaussian,
                cams             = cams_sorted,
                key_cam_indices  = key_cam_indices,
                scales           = scales,
                K                = K,
                M_half           = M_half,
                vis_eps          = vis_eps,
                alpha_tau        = alpha_tau,
            )

        print(f"[GP-cache] built for {len(cams_sorted)} views, "
              f"{len(key_cam_indices)} key views, scales={scales}, K={K}, L={K*(2*M_half+1)**2}")

    def edit_all_view_gaussian_provenance(
        self,
        original_render_name: str,
        cache_name: str,
        update_camera: bool = False,
        global_step: int = 0,
        gp_K: int = 2,
        gp_M_half: int = 1,
        gp_vis_eps: float = 0.05,
        gp_alpha_tau: float = 0.4,
        gp_scales: list = None,
    ):
        """
        Alternative to edit_all_view that replaces the O(HW²) epipolar
        similarity search with O(HW·K·M) sparse cross-view attention guided
        by 3DGS gaussian provenance.

        Key differences from edit_all_view:
          - No epipolar constraint precomputation (epipolar_constrains).
          - No dense einsum similarity matrix.
          - Uses gaussian projection to find candidate correspondences.
          - Per-pixel confidence gating via alpha (top-1 gaussian weight).

        The pivotal/key-view pass is identical to edit_all_view: the DGE
        guidance runs its normal extended-attention UNet forward for key views
        (their attention output is cached as kf_attn_output).  For non-key
        views the cached kf_attn_output is gathered with sparse_xview_attn
        using the pre-built idx_map / cand_valid / alpha tensors.
        """
        from threestudio.utils.dge_utils import (
            register_gp_cache,
            unregister_gp_cache,
            sparse_xview_attn,
        )

        if gp_scales is None:
            gp_scales = [(64, 64), (32, 32), (16, 16), (8, 8)]

        # ------------------------------------------------------------------ #
        # Camera update (identical to edit_all_view)                          #
        # ------------------------------------------------------------------ #
        if update_camera:
            with self._latency_logger.timeit("training_step_all.edit_all_view_gaussian_provenance.update_editing_cameras"):
                self.trainer.datamodule.train_dataset.update_editing_cameras(
                    random_seed=global_step + 1
                )
                self.edit_view_index = self.trainer.datamodule.train_dataset.edit_view_index
                sorted_train_view_list = sorted(self.edit_view_index)
                selected_views = torch.linspace(
                    0,
                    len(sorted_train_view_list) - 1,
                    self.trainer.datamodule.val_dataset.n_views,
                    dtype=torch.int,
                )
                self.trainer.datamodule.val_dataset.selected_views = [
                    sorted_train_view_list[idx] for idx in selected_views
                ]

        print(f"{self.true_global_step}th step [gp], Camera view index: {self.edit_view_index}")

        self.edit_frames       = {}
        self.edit_frames_order = []
        cache_dir              = os.path.join(self.cache_dir, cache_name)
        original_render_cache  = os.path.join(self.cache_dir, original_render_name)
        os.makedirs(cache_dir, exist_ok=True)

        # ------------------------------------------------------------------ #
        # Collect & sort cameras                                              #
        # ------------------------------------------------------------------ #
        cameras = [
            self.trainer.datamodule.train_dataset.scene.cameras[i]
            for i in self.edit_view_index
        ]
        sorted_cam_idx = self.sort_the_cameras_idx(cameras)
        view_sorted    = [self.edit_view_index[i] for i in sorted_cam_idx]
        cams_sorted    = [cameras[i]              for i in sorted_cam_idx]

        n_views  = len(cams_sorted)
        # Key view(s): first camera in the sorted order (and optionally the
        # midpoint, mirroring DGE's 1- or 2-key-view logic).
        # Use a single key view to keep it simple; caller can override via
        # gp_K if needed.
        key_cam_indices = [0]
        if n_views > 2:
            key_cam_indices.append(n_views // 2)

        # ------------------------------------------------------------------ #
        # Load max_step schedule (same as edit_all_view)                      #
        # ------------------------------------------------------------------ #
        t_max_step = self.cfg.added_noise_schedule
        self.guidance.max_step = t_max_step[
            min(len(t_max_step) - 1,
                self.true_global_step // self.cfg.camera_update_per_step)
        ]

        # ------------------------------------------------------------------ #
        # Build gaussian-provenance cache (1 per camera-update cycle)        #
        # ------------------------------------------------------------------ #
        with self._latency_logger.timeit("training_step_all.edit_all_view_gaussian_provenance.gp_cache"):
            self.build_gp_cache(
                cams_sorted      = cams_sorted,
                key_cam_indices  = key_cam_indices,
                scales           = gp_scales,
                K                = gp_K,
                M_half           = gp_M_half,
                vis_eps          = gp_vis_eps,
                alpha_tau        = gp_alpha_tau,
            )

        gp_cache = self._gp_cache

        # ------------------------------------------------------------------ #
        # Render all views & load original frames                             #
        # ------------------------------------------------------------------ #
        images         = []
        original_frames = []

        with torch.no_grad():
            with self._latency_logger.timeit("training_step_all.edit_all_view_gaussian_provenance.render_and_load"):
                for id in view_sorted:
                    orig_path = os.path.join(
                        original_render_cache, "{:0>4d}.png".format(id)
                    )
                    cur_cam = self.trainer.datamodule.train_dataset.scene.cameras[id]
                    cur_batch = {
                        "index":  id,
                        "camera": [cur_cam],
                        "height": self.trainer.datamodule.train_dataset.height,
                        "width":  self.trainer.datamodule.train_dataset.width,
                    }
                    out_pkg = self(cur_batch)
                    out     = out_pkg["comp_rgb"]
                    if self.cfg.use_masked_image:
                        out = out * out_pkg["masks"].unsqueeze(-1)
                    images.append(out)

                    assert os.path.exists(orig_path), f"Missing origin render: {orig_path}"
                    cached_image = cv2.cvtColor(cv2.imread(orig_path), cv2.COLOR_BGR2RGB)
                    self.origin_frames[id] = torch.tensor(
                        cached_image / 255, device="cuda", dtype=torch.float32
                    )[None]
                    original_frames.append(self.origin_frames[id])

            images          = torch.cat(images,          dim=0)   # [N, H, W, 3]
            original_frames = torch.cat(original_frames, dim=0)   # [N, H, W, 3]

        # ------------------------------------------------------------------ #
        # Run DGE guidance with GP-cache injected into DGEBlocks              #
        #                                                                      #
        # The guidance.__call__ internally iterates batches and calls         #
        # edit_latents which:                                                  #
        #   1. register_pivotal(True)  → key views → kf_attn_output cached   #
        #   2. register_pivotal(False) → non-key views                        #
        #      → DGEBlock reads gp_idx_map / gp_cand_valid / gp_alpha and    #
        #        calls sparse_xview_attn instead of the dense einsum          #
        #                                                                      #
        # We pass gp_cache through a new keyword so guidance can forward it   #
        # to each DGEBlock via the existing register_* mechanism.             #
        # ------------------------------------------------------------------ #
        with torch.no_grad():
            with self._latency_logger.timeit("training_step_all.edit_all_view_gaussian_provenance.guidance_batch"):
                edited_images = self.guidance(
                    images,
                    original_frames,
                    self.prompt_processor(),
                    cams=cams_sorted,
                    latency_logger=self._latency_logger,
                    gp_cache=gp_cache,
                    key_cam_indices=key_cam_indices,
                    latency_prefix="training_step_all.edit_all_view_gaussian_provenance.guidance_batch",
                )

        # ------------------------------------------------------------------ #
        # Store results (identical to edit_all_view)                          #
        # ------------------------------------------------------------------ #
        with self._latency_logger.timeit("training_step_all.edit_all_view_gaussian_provenance.assign_outputs"):
            self.edit_frames_order = view_sorted.copy()
            for vi, vid in enumerate(view_sorted):
                self.edit_frames[vid] = (
                    edited_images['edit_images'][vi].unsqueeze(0).detach().clone()
                )

        # ------------------------------------------------------------------ #
        # Save grid                                                           #
        # ------------------------------------------------------------------ #
        save_list = []
        for vid in self.edit_frames_order:
            if vid in self.edit_frames:
                img_with_idx = self._add_index_to_image(self.edit_frames[vid][0], vid)
                save_list.append(
                    {"type": "rgb", "img": img_with_idx, "kwargs": {"data_format": "HWC"}}
                )
        if save_list:
            self.save_image_grid(
                "edited_images_gp.png",
                save_list,
                name="edited_images_gp",
                step=self.true_global_step,
            )
        print("[gp] edited images saved to:", self.get_save_path("edited_images_gp.png"))
        a = 1

    # ---------------------------------------------------------------------- #
    # Version C: Multi-View Diffusion Attention Warping (AttentionWarpManager)#
    # ---------------------------------------------------------------------- #

    def edit_all_view_attn_warp(
        self,
        original_render_name: str,
        cache_name: str,
        update_camera: bool = False,
        global_step: int = 0,
        key_view_num: int = 4,
        add_noise_t: int = 500,
        inject_until_t: float = 0.5,
        occlusion_threshold: float = 0.05,
    ):
        """
        Multi-view attention warping alternative to edit_all_view.

        Unlike the epipolar / GP approaches that operate inside the DGEBlock
        on every timestep, this method:

          1. Pre-computes Self-Attention K/V features from N key views using a
             single UNet forward pass per key view  (AttentionWarpManager.
             extract_and_store_features).
          2. For each target view, back-projects the 3-D geometry (via 2DGS
             depth) and blends the key-view K/V features weighted by:
               • Visibility   (occlusion check via depth comparison)
               • Angular sim  (cosine similarity between viewing directions)
               • Confidence   (normal-dot-viewdir if normals are available)
          3. Injects the blended K_merged / V_merged into every UNet
             self-attention layer during denoising via a custom AttnProcessor
             with a timestep-based linear decay schedule.

        Parameters
        ----------
        original_render_name : subdirectory name under self.cache_dir that
            holds the pre-rendered original (unedited) PNG frames.
        cache_name           : subdirectory name for saving edited frames.
        update_camera        : whether to re-sample the editing camera set.
        global_step          : current training step (used for camera update
            random seed and noise schedule).
        key_view_num         : how many key views to use for feature extraction
            (uniformly sampled from the sorted camera set).
        add_noise_t          : forward-process timestep used when running the
            UNet forward pass to extract K/V features.
        inject_until_t       : fraction of denoising steps over which injection
            weight linearly decays from 1→0.
        occlusion_threshold  : relative depth tolerance for the visibility
            check (see AttentionWarpManager).
        """
        from threestudio.utils.attention_warp import AttentionWarpManager
        from gaussiansplatting.gaussian_renderer import render as gs_render

        # ------------------------------------------------------------------ #
        # Camera update (identical to edit_all_view)                          #
        # ------------------------------------------------------------------ #
        if update_camera:
            with self._latency_logger.timeit("training_step_all.edit_aw.update_editing_cameras"):
                self.trainer.datamodule.train_dataset.update_editing_cameras(
                    random_seed=global_step + 1
                )
                self.edit_view_index = (
                    self.trainer.datamodule.train_dataset.edit_view_index
                )
                sorted_train_view_list = sorted(self.edit_view_index)
                selected_views = torch.linspace(
                    0,
                    len(sorted_train_view_list) - 1,
                    self.trainer.datamodule.val_dataset.n_views,
                    dtype=torch.int,
                )
                self.trainer.datamodule.val_dataset.selected_views = [
                    sorted_train_view_list[idx] for idx in selected_views
                ]

        print(
            f"{self.true_global_step}th step [attn-warp], "
            f"Camera view index: {self.edit_view_index}"
        )

        self.edit_frames = {}
        self.edit_frames_order = []
        cache_dir             = os.path.join(self.cache_dir, cache_name)
        original_render_cache = os.path.join(self.cache_dir, original_render_name)
        os.makedirs(cache_dir, exist_ok=True)

        # ------------------------------------------------------------------ #
        # Collect & sort cameras                                              #
        # ------------------------------------------------------------------ #
        with self._latency_logger.timeit("training_step_all.edit_aw.collect_and_sort"):
            cameras = [
                self.trainer.datamodule.train_dataset.scene.cameras[i]
                for i in self.edit_view_index
            ]
            sorted_cam_idx = self.sort_the_cameras_idx(cameras)
            view_sorted  = [self.edit_view_index[i] for i in sorted_cam_idx]
            cams_sorted  = [cameras[i]              for i in sorted_cam_idx]

        n_views = len(cams_sorted)

        # ------------------------------------------------------------------ #
        # Noise schedule (same as edit_all_view)                              #
        # ------------------------------------------------------------------ #
        t_max_step = self.cfg.added_noise_schedule
        self.guidance.max_step = t_max_step[
            min(len(t_max_step) - 1,
                self.true_global_step // self.cfg.camera_update_per_step)
        ]

        # ------------------------------------------------------------------ #
        # Step 1: Render all views + collect depth maps                       #
        # ------------------------------------------------------------------ #
        rendered_images: list = []   # list of (1, H, W, C) tensors  [0,1]
        depth_maps:      list = []   # list of (H_img, W_img) tensors
        original_frames: list = []

        with torch.no_grad():
            with self._latency_logger.timeit("training_step_all.edit_aw.render_all"):
                for vid in view_sorted:
                    cam = self.trainer.datamodule.train_dataset.scene.cameras[vid]
                    cur_batch = {
                        "index":  vid,
                        "camera": [cam],
                        "height": self.trainer.datamodule.train_dataset.height,
                        "width":  self.trainer.datamodule.train_dataset.width,
                    }
                    out_pkg = self(cur_batch)
                    rgb = out_pkg["comp_rgb"]  # (1, H, W, C)
                    if self.cfg.use_masked_image:
                        rgb = rgb * out_pkg["masks"].unsqueeze(-1)
                    rendered_images.append(rgb)

                    # Render depth from 2DGS
                    render_pkg = gs_render(
                        cam, self.gaussian, self.pipe, self.background_tensor
                    )
                    # "depth" key: (1, H, W) or (H, W) depending on version
                    depth_raw = render_pkg.get("depth", render_pkg.get("surf_depth", None))
                    if depth_raw is not None:
                        depth_hw = depth_raw.squeeze()  # (H, W)
                    else:
                        # Fallback: uniform depth (disables geometry-aware blending)
                        H_img = int(cam.image_height)
                        W_img = int(cam.image_width)
                        depth_hw = torch.ones(H_img, W_img, device="cuda")
                    depth_maps.append(depth_hw)

                    # Load original frame
                    orig_path = os.path.join(
                        original_render_cache, "{:0>4d}.png".format(vid)
                    )
                    assert os.path.exists(orig_path), \
                        f"Missing original render: {orig_path}"
                    cached_img = cv2.cvtColor(
                        cv2.imread(orig_path), cv2.COLOR_BGR2RGB
                    )
                    self.origin_frames[vid] = torch.tensor(
                        cached_img / 255, device="cuda", dtype=torch.float32
                    )[None]
                    original_frames.append(self.origin_frames[vid])

        # ------------------------------------------------------------------ #
        # Step 2: Select key views (uniformly sampled from sorted set)        #
        # ------------------------------------------------------------------ #
        key_view_num = min(key_view_num, n_views)
        key_indices  = torch.linspace(0, n_views - 1, key_view_num,
                                      dtype=torch.long).tolist()
        key_indices  = [int(i) for i in key_indices]

        key_images_for_extract = [
            rendered_images[i].permute(0, 3, 1, 2)  # (1, C, H, W)
            for i in key_indices
        ]
        key_cams_for_extract   = [cams_sorted[i] for i in key_indices]
        key_depths_for_extract = [depth_maps[i]  for i in key_indices]

        print(
            f"[attn-warp] Using {key_view_num} key views at sorted indices "
            f"{key_indices} out of {n_views} total views."
        )

        # ------------------------------------------------------------------ #
        # Step 3: Build AttentionWarpManager & extract features               #
        # ------------------------------------------------------------------ #
        with self._latency_logger.timeit("training_step_all.edit_aw.build_manager"):
            diffusion_steps = self.guidance.cfg.diffusion_steps
            aw_manager = AttentionWarpManager(
                unet                   = self.guidance.unet,
                vae                    = self.guidance.vae,
                scheduler              = self.guidance.scheduler,
                weights_dtype          = self.guidance.weights_dtype,
                total_denoising_steps  = diffusion_steps,
                inject_until_t         = inject_until_t,
                occlusion_threshold    = occlusion_threshold,
            )

        with self._latency_logger.timeit("training_step_all.edit_aw.extract_features"):
            # Get text embeddings from the prompt processor
            prompt_utils = self.prompt_processor()
            # text_embeddings shape from DGE: (3*B, 77, 768) – use the first slice
            text_emb_all = prompt_utils.get_text_embeddings(
                elevation_deg=torch.zeros(1),
                azimuth_deg=torch.zeros(1),
                camera_distances=torch.ones(1),
                use_local_text_embeddings=False,
            )  # (3, 77, 768)
            # Use positive text embedding for feature extraction
            text_emb_pos = text_emb_all[0:1]  # (1, 77, 768)

            aw_manager.extract_and_store_features(
                key_images       = key_images_for_extract,
                key_cams         = key_cams_for_extract,
                key_depths       = key_depths_for_extract,
                key_normals      = None,  # normals not used by default
                text_embeddings  = text_emb_pos,
                add_noise_t      = add_noise_t,
            )

        # Register custom AttnProcessors on the UNet
        with self._latency_logger.timeit("training_step_all.edit_aw.apply_to_unet"):
            aw_manager.apply_to_unet()

        # ------------------------------------------------------------------ #
        # Step 4: Denoise each target view individually                       #
        # ------------------------------------------------------------------ #
        images_batched  = torch.cat(rendered_images,  dim=0)  # (N, H, W, C)
        orig_batched    = torch.cat(original_frames,  dim=0)  # (N, H, W, C)

        edited_results = {}  # vid → (1, H, W, C) tensor

        with torch.no_grad():
            for view_i, (vid, tgt_cam, tgt_depth) in enumerate(
                zip(view_sorted, cams_sorted, depth_maps)
            ):
                with self._latency_logger.timeit(f"training_step_all.edit_aw.denoise_view_{view_i}"):
                    # Set the target view for geometry-aware K/V blending
                    aw_manager.set_target_view(tgt_cam, tgt_depth)
                    # Reset per-step decay counters
                    aw_manager.reset_step_counters()

                    # Single-view tensors for guidance
                    single_img = images_batched[view_i:view_i+1]  # (1, H, W, C)
                    single_orig = orig_batched[view_i:view_i+1]   # (1, H, W, C)

                    # Run DGE guidance for this single view
                    # We use use_normal_unet=True (no epipolar) since AttnProcessors
                    # now handle the cross-view injection.
                    from threestudio.utils.dge_utils import (
                        register_normal_attn_flag,
                        register_pivotal,
                    )
                    register_normal_attn_flag(self.guidance.unet, True)
                    register_pivotal(self.guidance.unet, False)

                    edited_out = self.guidance(
                        single_img,
                        single_orig,
                        self.prompt_processor(),
                        cams=[tgt_cam],
                        latency_logger=self._latency_logger,
                        latency_prefix=f"training_step_all.edit_aw.denoise_view_{view_i}.guidance_batch",
                    )
                    edited_results[vid] = (
                        edited_out["edit_images"][0].unsqueeze(0).detach().clone()
                    )

        # Restore original attention processors
        with self._latency_logger.timeit("training_step_all.edit_aw.remove_from_unet"):
            aw_manager.remove_from_unet()
        # Restore normal-attn flag
        register_normal_attn_flag(self.guidance.unet, False)

        # ------------------------------------------------------------------ #
        # Step 5: Store results & save grid                                   #
        # ------------------------------------------------------------------ #
        self.edit_frames_order = view_sorted.copy()
        for vid in view_sorted:
            self.edit_frames[vid] = edited_results[vid]

        save_list = []
        for vid in self.edit_frames_order:
            if vid in self.edit_frames:
                img_with_idx = self._add_index_to_image(
                    self.edit_frames[vid][0], vid
                )
                save_list.append(
                    {
                        "type": "rgb",
                        "img": img_with_idx,
                        "kwargs": {"data_format": "HWC"},
                    }
                )
        if save_list:
            self.save_image_grid(
                "edited_images_aw.png",
                save_list,
                name="edited_images_aw",
                step=self.true_global_step,
            )
        print(
            "[attn-warp] edited images saved to:",
            self.get_save_path("edited_images_aw.png"),
        )

    def sort_the_cameras_idx(self, cams):
        # 각도 기반 원형 정렬 (한 방향으로만, 방향 전환 없이) - 벡터화 최적화
        # 전방 벡터와 카메라 중심 추출 (벡터화)
        forward_vectors = np.array([cam.R[:, 2] for cam in cams])  # (N, 3)
        cams_center_x = np.array([cam.camera_center[0].item() for cam in cams])
        
        # 가장 왼쪽 카메라의 전방 벡터를 기준으로 선택
        most_left_idx = np.argmin(cams_center_x)
        most_left_vector = forward_vectors[most_left_idx]
        
        # 참조 축 생성: 카메라 중심들의 평균 위치를 기준으로
        cam_centers = np.array([cam.camera_center.cpu().numpy() if isinstance(cam.camera_center, torch.Tensor) else cam.camera_center for cam in cams])
        center_mean = cam_centers.mean(axis=0)
        center_to_left = cam_centers[most_left_idx] - center_mean
        center_to_left = center_to_left / (np.linalg.norm(center_to_left) + 1e-8)
        
        # 참조 축: center_to_left와 most_left_vector의 외적 (카메라 배치에 맞는 축)
        reference_axis = np.cross(most_left_vector, center_to_left)
        if np.linalg.norm(reference_axis) < 1e-6:
            # 평행한 경우, 위쪽 방향(Y축) 사용
            up_vector = np.array([0, 1, 0])
            reference_axis = np.cross(most_left_vector, up_vector)
            if np.linalg.norm(reference_axis) < 1e-6:
                # 여전히 평행하면 X축 사용
                reference_axis = np.cross(most_left_vector, np.array([1, 0, 0]))
        reference_axis = reference_axis / (np.linalg.norm(reference_axis) + 1e-8)
        
        # 벡터화된 signed angle 계산 (한 방향으로만)
        # 모든 카메라의 전방 벡터와 기준 벡터의 내적 계산
        dot_products = np.clip(np.dot(forward_vectors, most_left_vector), -1.0, 1.0)  # (N,)
        angles = np.arccos(dot_products)  # (N,)
        
        # 외적 계산 (벡터화)
        # most_left_vector와 각 forward_vector의 외적
        cross_products = np.cross(most_left_vector[None, :], forward_vectors)  # (N, 3)
        signs = np.sign(np.dot(cross_products, reference_axis))  # (N,)
        
        # signed angle 계산 및 정규화
        signed_angles = signs * angles  # (N,)
        normalized_angles = np.where(signed_angles < 0, 2 * np.pi + signed_angles, signed_angles)  # (N,)
        
        # 각도 순서로 정렬
        sorted_cam_idx = np.argsort(normalized_angles).tolist()

        print(f"sorted_cam_idx: {sorted_cam_idx}")


        return sorted_cam_idx

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # latency logger under trial_dir/latency (created in launch.py if not set)
        if not hasattr(self, "_latency_logger") or self._latency_logger is None:
            latency_dir = os.path.join(self.get_save_dir(), "..", "latency")
            latency_dir = os.path.abspath(latency_dir)
            os.makedirs(latency_dir, exist_ok=True)
            self._latency_logger = LatencyLogger(latency_dir)

        with self._latency_logger.timeit("render_all_view"):
            self.render_all_view(cache_name="origin_render")
        
        # 원본이미지 저장
        save_list = []
        for index, image in sorted(
                self.origin_frames.items(), key=lambda item: item[0]
        ):
            # 이미지에 인덱스 번호 추가
            img_with_index = self._add_index_to_image(image[0], index)
            save_list.append(
                {
                    "type": "rgb",
                    "img": img_with_index,
                    "kwargs": {"data_format": "HWC"},
                },
            )
        self.save_image_grid(
            f"origin_images.png",
            save_list,
            name="origin",
            step=self.true_global_step,
        )
        threestudio.info(f"origin_images saved to: {self.get_save_path('origin_images.png')}")


        if len(self.cfg.prompt_processor) > 0:
            self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
                self.cfg.prompt_processor
            )
        dm = getattr(self.trainer, "datamodule", None)
        if self.cfg.loss.lambda_l1 > 0 or self.cfg.loss.lambda_p > 0 or self.cfg.loss.use_sds or self.cfg.loss.lambda_dds > 0 or self.cfg.loss.lambda_ism > 0:
            # Get or load IP2P OUTSIDE of latency measurement (exclude model loading)
            ip2p_pipe = None
            if dm is not None and getattr(dm, "ip2p_pipe", None) is not None:
                ip2p_pipe = dm.ip2p_pipe
            elif self.cfg.guidance_type == "dge-guidance":
                from diffusers import StableDiffusionInstructPix2PixPipeline
                threestudio.info("[DGE] Loading InstructPix2Pix (no shared pipe from lens)...")
                ip2p_path = OmegaConf.select(self.cfg, "guidance.ip2p_name_or_path", default="timbrooks/instruct-pix2pix")
                ip2p_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                    ip2p_path,
                    torch_dtype=torch.float16,
                    safety_checker=None,
                ).to(get_device())
            with self._latency_logger.timeit("guidance_init"):
                self.guidance = threestudio.find(self.cfg.guidance_type)(
                    self.cfg.guidance, preloaded_pipe=ip2p_pipe
                )
                # Set save_dir for guidance to save epipolar constraint images
                self.guidance.save_dir = self.get_save_dir()

        # Load vanilla IP2P for warp-and-refine (separate from DGE guidance)
        if self.cfg.use_warp_refine:
            from diffusers import StableDiffusionInstructPix2PixPipeline
            threestudio.info("[DGE] Loading vanilla InstructPix2Pix for warp-and-refine...")
            ip2p_path = OmegaConf.select(self.cfg, "guidance.ip2p_name_or_path", default="timbrooks/instruct-pix2pix")
            self._warp_refine_ip2p = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                ip2p_path,
                torch_dtype=torch.float16,
                safety_checker=None,
            ).to(get_device())
            
        self.style_direction = CLIP.get_style_embedding(
            clip_model,
            self.cfg.target_prompt,
            None, # self.cfg.style_image,
            self.cfg.seg_prompt # self.cfg.object_prompt
        )

        # if len(self.cfg.seg_prompt) > 0:
        #     self.update_mask(self.cfg.seg_prompt)

    def training_step(self, batch, batch_idx):
        _ts_start = time.perf_counter()



        if self.true_global_step % self.cfg.camera_update_per_step == 0 and self.cfg.use_warp_refine:
            # Warp-and-Refine branch: vanilla IP2P propagation (no DGE attention)
            with self._latency_logger.timeit("training_step_all.edit_all_view_warp_refine"):
                self.edit_all_view_warp_refine(
                    original_render_name='origin_render',
                    cache_name="edited_views_wr",
                    ip2p_pipe=self._warp_refine_ip2p,
                    update_camera=self.true_global_step >= self.cfg.camera_update_per_step,
                    global_step=self.true_global_step,
                )
        elif self.true_global_step % self.cfg.camera_update_per_step == 0 and self.cfg.use_gaussian_provenance and self.cfg.guidance_type == 'dge-guidance' and not self.cfg.loss.use_sds:
            # Version B: Gaussian-Provenance Sparse Cross-View Attention
            with self._latency_logger.timeit("training_step_all.edit_all_view_gaussian_provenance"):
                self.edit_all_view_gaussian_provenance(
                    original_render_name='origin_render',
                    cache_name="edited_views_gp",
                    update_camera=self.true_global_step >= self.cfg.camera_update_per_step,
                    global_step=self.true_global_step,
                    gp_K=self.cfg.gp_K,
                    gp_M_half=self.cfg.gp_M_half,
                    gp_vis_eps=self.cfg.gp_vis_eps,
                    gp_alpha_tau=self.cfg.gp_alpha_tau,
                )
        elif self.true_global_step % self.cfg.camera_update_per_step == 0 and self.cfg.use_multiview_edit and self.cfg.guidance_type == 'dge-guidance' and not self.cfg.loss.use_sds:
            with self._latency_logger.timeit("training_step_all.edit_multiview"):
                self.edit_multiview(
                    original_render_name='origin_render',
                    cache_name="edited_views_multiview",
                    update_camera=self.true_global_step >= self.cfg.camera_update_per_step,
                    global_step=self.true_global_step,
                )
        elif self.true_global_step % self.cfg.camera_update_per_step == 0 and self.cfg.guidance_type == 'dge-guidance' and not self.cfg.loss.use_sds:
            with self._latency_logger.timeit("training_step_all.edit_all_view"):
                self.edit_all_view(original_render_name='origin_render', cache_name="edited_views", update_camera=self.true_global_step >= self.cfg.camera_update_per_step, global_step=self.true_global_step)

        
        if self.true_global_step == 0 and len(self.cfg.seg_prompt) > 0:
            print(f"Update mask with seg prompt: {self.cfg.seg_prompt}")
            self.update_mask(self.cfg.seg_prompt)


        if self.true_global_step == self.cfg.mask_update_at_step and len(self.cfg.target_prompt) > 0:
            print(f"Update mask with target prompt: {self.cfg.target_prompt}")
            self.update_mask(self.cfg.target_prompt)
        


        # Prune distant floater Gaussians
        if self.cfg.prune_floater_at_step >= 0 and self.true_global_step == self.cfg.prune_floater_at_step:
            with self._latency_logger.timeit(f"training_step_all.prune_floater"):
                self.prune_distant_floater_gaussians()

        with self._latency_logger.timeit("training_step_all.lr_update"):
            self.gaussian.update_learning_rate(self.true_global_step)
        batch_index = batch["index"]

        if isinstance(batch_index, int):
            batch_index = [batch_index]
        # if self.cfg.guidance_type == 'dge-guidance': 
        #     for img_index, cur_index in enumerate(batch_index):
        #         if cur_index not in self.edit_frames:
        #             batch_index[img_index] = self.trainer.datamodule.train_dataset.train_view_index[img_index] # 전체 train view

        with self._latency_logger.timeit("training_step_all.render_forward"):
            out = self(batch, local=self.cfg.local_edit)

        images = out["comp_rgb"]
        mask = out["masks"].unsqueeze(-1)
        loss = 0.0
        # nerf2nerf loss
        if self.cfg.loss.lambda_l1 > 0 or self.cfg.loss.lambda_p > 0:
            prompt_utils = self.prompt_processor()
            with self._latency_logger.timeit("training_step_all.collect_gt_images"):
                gt_images = []
                for img_index, cur_index in enumerate(batch_index):
                    # if cur_index not in self.edit_frames:
                    #     # cur_index = self.view_list[0]
                    if cur_index in self.edit_frames:
                        gt_images.append(self.edit_frames[cur_index])

                    else: # CLIP LOSS
                        pass

            loss_dict = {}
            ## L1 + Perceptual loss
            if len(gt_images) > 0: # ground truth image가 있다면 기존의 Loss를 그대로 활용
                gt_images = torch.concatenate(gt_images, dim=0)

                if self.cfg.use_masked_image:
                    print("use masked image")
                    with self._latency_logger.timeit("training_step_all.loss_l1"):
                        loss_dict["loss_l1"] = torch.nn.functional.l1_loss(images * mask, gt_images * mask)
                    with self._latency_logger.timeit("training_step_all.loss_p"):
                        loss_dict["loss_p"] = self.perceptual_loss(
                            (images * mask).permute(0, 3, 1, 2).contiguous(),
                            (gt_images * mask).permute(0, 3, 1, 2).contiguous(),
                        ).sum()
                else:
                    with self._latency_logger.timeit("training_step_all.loss_l1"):
                        loss_dict["loss_l1"] = torch.nn.functional.l1_loss(images, gt_images)
                    with self._latency_logger.timeit("training_step_all.loss_p"):
                        loss_dict["loss_p"] = self.perceptual_loss(
                            images.permute(0, 3, 1, 2).contiguous(),
                            gt_images.permute(0, 3, 1, 2).contiguous(),
                        ).sum()

            ## Lite-ISM loss
            if self.cfg.loss.lambda_ism > 0:
                # Lite-ISM: 2-batch UNet, x0-prediction target, strong edit signal
                with self._latency_logger.timeit("training_step_all.loss_ism"):
                    loss_ism = self.compute_lite_ism_loss(images, batch_index)
                loss_dict["loss_ism"] = loss_ism

            ## DDS-lite loss
            if self.cfg.loss.lambda_dds > 0:
                # DDS-lite: lightweight distillation for views without edit_frames
                with self._latency_logger.timeit("training_step_all.loss_dds"):
                    loss_dds = self.compute_dds_loss(images, batch_index)
                loss_dict["loss_dds"] = loss_dds

            ## Directional CLIP loss
            if self.cfg.loss.lambda_d > 0:
                # Direction CLIP loss
                # images shape: (B, H, W, C) -> (B, C, H, W)로 변환 필요
                # Prepare images for CLIP: apply mask if use_masked_image is True
                with self._latency_logger.timeit("training_step_all.loss_d"):
                    images_clip = images.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
                    gt_images_list = []
                    for idx in batch_index:
                        gt_images_list.append(self.origin_frames[idx])
                    gt_images_clip = torch.concatenate(gt_images_list, dim=0).permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)

                    render_features = clip_model.encode_image(
                        clip_normalize(images_clip))
                    source_features = clip_model.encode_image(
                        clip_normalize(gt_images_clip))
                    # NOTE: add eps to prevent NaN/Inf when norm is near-zero
                    eps = 1e-6
                    render_features = render_features / (
                        render_features.clone().norm(dim=-1, keepdim=True) + eps
                    )

                    img_direction = render_features - source_features
                    img_direction = img_direction / (
                        img_direction.clone().norm(dim=-1, keepdim=True) + eps
                    )

                    # `self.style_direction` may be stored as (D,) or (1, D).
                    # Make it 1D first, then broadcast to (B, D) safely.
                    style_dir = self.style_direction
                    if style_dir.ndim == 2 and style_dir.shape[0] == 1:
                        style_dir = style_dir[0]
                    style_dir = style_dir / (style_dir.norm(dim=-1, keepdim=False) + eps)
                    style_dir = style_dir.unsqueeze(0).expand(render_features.size(0), -1)

                    loss_d = (1 - torch.cosine_similarity(img_direction,
                            style_dir, dim=1)).mean()

                    loss_dict["loss_d"] = loss_d

            # novel views에 대해서 DDS, CLIP 모두 적용하지 않은 경우에만 dummy로 graph 연결 (기존 loss_dict 덮어쓰지 않음)
            if self.cfg.loss.lambda_d <= 0 and len(loss_dict) == 0:
                z = (images * 0).sum()
                loss_dict["loss_l1"] = z
                loss_dict["loss_p"] = z

            with self._latency_logger.timeit("training_step_all.loss_combine"):
                for name, value in loss_dict.items():
                    self.log(f"train/{name}", value)
                    if name.startswith("loss_"):
                        loss += value * self.C(
                            self.cfg.loss[name.replace("loss_", "lambda_")]
                        )


        # sds loss
        if self.cfg.loss.use_sds:
            prompt_utils = self.prompt_processor()
            self.guidance.cfg.use_sds = True
            with self._latency_logger.timeit("training_step_all.guidance_sds"):
                loss_dict = self.guidance(
                    out["comp_rgb"],
                    torch.concatenate(
                        [self.origin_frames[idx] for idx in batch_index], dim=0
                    ),
                    prompt_utils,
                    cams=batch["camera"],
                    latency_logger=self._latency_logger,
                    latency_prefix="training_step_all.guidance_sds",
                )
            loss += loss_dict["loss_sds"] * self.cfg.loss.lambda_sds

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        self._latency_logger.record("training_step_all", time.perf_counter() - _ts_start)
        return {"loss": loss}

    def on_train_end(self) -> None:
        """Called when training ends. Write latency summary."""
        if hasattr(self, '_latency_logger'):
            self._latency_logger.write_summary()
            threestudio.info(f"Latency summary written to {self._latency_logger.base_dir}/summary.txt")
