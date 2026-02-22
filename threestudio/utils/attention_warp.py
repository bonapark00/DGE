"""
AttentionWarpManager
====================
Multi-view Diffusion-Based Attention Warping for 2DGS.

Logic overview
--------------
1. extract_and_store_features()
   - Run a forward diffusion pass through ip2p UNet for each key view.
   - Hook every Self-Attention layer and store K, V tensors per layer per key view.
   - Back-project each pixel's K/V into 3-D world space using the 2DGS depth map
     and the camera's intrinsic / extrinsic matrices.

2. warp_and_blend_attention()
   - For a target view, project the stored 3-D points into the target image plane.
   - For each projected pixel compute a weight  w_i = visibility × angular × confidence.
   - Compute weighted-average  K_merged, V_merged  across all key views.
   - Return merged tensors ready to be injected into the UNet.

3. apply_to_unet()
   - Use diffusers' set_attn_processor() to register a custom AttnProcessor that,
     during the target-view denoising pass, replaces the UNet's own K/V with the
     merged ones (with a timestep-dependent decay factor).

Camera convention (matches gaussiansplatting Camera)
-----------------------------------------------------
* world_view_transform  : 4×4 column-major W2C  (stored as its transpose,
                          so viewpoint_camera.world_view_transform = W2C^T).
  The actual W2C matrix =  world_view_transform.T
* projection_matrix     : 4×4 column-major projection (also stored transposed).
* camera_center         : 3-D world-space camera position.
* FoVx, FoVy           : horizontal / vertical field of view (radians).
* image_width, image_height: pixel dimensions.

Depth map convention
--------------------
depth_map : (H, W) float tensor, values in camera-space Z (positive forward),
            e.g. the "depth" output of gaussiansplatting's render().
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fov_to_focal(fov: float, size: int) -> float:
    """Convert field-of-view (radians) to focal length in pixels."""
    return size / (2.0 * math.tan(fov * 0.5))


def _get_intrinsics(cam, H: int, W: int) -> Tuple[float, float, float, float]:
    """Return (fx, fy, cx, cy) in pixel units for a gaussiansplatting Camera."""
    fx = _fov_to_focal(cam.FoVx, W)
    fy = _fov_to_focal(cam.FoVy, H)
    cx = W / 2.0
    cy = H / 2.0
    return fx, fy, cx, cy


def _backproject_depth(
    depth: torch.Tensor,  # (H, W)
    cam,
    latent_H: int,
    latent_W: int,
) -> torch.Tensor:
    """
    Back-project depth map pixels to 3-D world coordinates.

    The depth map is first bilinearly resized to (latent_H, latent_W) so that
    each spatial position corresponds to one attention token.

    Returns
    -------
    points_world : (latent_H * latent_W, 3)  float32 world-space 3-D points.
    """
    H_orig, W_orig = depth.shape
    device = depth.device

    # Resize depth to latent resolution
    d = F.interpolate(
        depth.unsqueeze(0).unsqueeze(0),  # (1,1,H,W)
        size=(latent_H, latent_W),
        mode="bilinear",
        align_corners=False,
    ).squeeze()  # (latent_H, latent_W)

    fx, fy, cx, cy = _get_intrinsics(cam, H_orig, W_orig)
    # Rescale principal point and focal length to latent resolution
    scale_x = latent_W / W_orig
    scale_y = latent_H / H_orig
    fx = fx * scale_x
    fy = fy * scale_y
    cx = cx * scale_x
    cy = cy * scale_y

    # Pixel grid (latent_H × latent_W)
    u = torch.arange(latent_W, device=device, dtype=torch.float32)
    v = torch.arange(latent_H, device=device, dtype=torch.float32)
    grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")  # (latent_H, latent_W)

    # Unproject to camera space
    z = d  # (latent_H, latent_W)
    x_cam = (grid_u - cx) / fx * z
    y_cam = (grid_v - cy) / fy * z
    # Camera-space points: (latent_H*latent_W, 3)
    pts_cam = torch.stack([x_cam, y_cam, z], dim=-1).reshape(-1, 3)

    # World-space:  P_world = C2W[:3,:3] @ P_cam + C2W[:3,3]
    # world_view_transform is stored as W2C^T  →  W2C = world_view_transform.T
    W2C = cam.world_view_transform.T.float()  # (4,4)
    C2W = torch.linalg.inv(W2C)              # (4,4)
    R_c2w = C2W[:3, :3]  # (3,3)
    t_c2w = C2W[:3, 3]   # (3,)

    pts_world = pts_cam @ R_c2w.T + t_c2w.unsqueeze(0)  # (N, 3)
    return pts_world


def _project_points(
    pts_world: torch.Tensor,  # (N, 3)
    cam,
    latent_H: int,
    latent_W: int,
    H_orig: int,
    W_orig: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project 3-D world points into the (latent) image plane of `cam`.

    Returns
    -------
    uv       : (N, 2) pixel coordinates in latent space (float).
    valid    : (N,)   bool mask – True when the point projects inside the frame
               and in front of the camera.
    """
    device = pts_world.device
    fx, fy, cx, cy = _get_intrinsics(cam, H_orig, W_orig)
    scale_x = latent_W / W_orig
    scale_y = latent_H / H_orig
    fx, fy = fx * scale_x, fy * scale_y
    cx, cy = cx * scale_x, cy * scale_y

    W2C = cam.world_view_transform.T.float()  # (4,4)
    R = W2C[:3, :3]
    t = W2C[:3, 3]
    pts_cam = pts_world @ R.T + t.unsqueeze(0)  # (N, 3)

    z = pts_cam[:, 2]
    valid = z > 1e-4

    u = fx * pts_cam[:, 0] / (z + 1e-8) + cx
    v = fy * pts_cam[:, 1] / (z + 1e-8) + cy
    uv = torch.stack([u, v], dim=-1)  # (N, 2)

    valid = valid & (u >= 0) & (u < latent_W) & (v >= 0) & (v < latent_H)
    return uv, valid


# ---------------------------------------------------------------------------
# Custom AttnProcessor for injection
# ---------------------------------------------------------------------------

class MultiViewWarpAttnProcessor:
    """
    Replaces the UNet self-attention K/V with the geometry-merged multi-view
    K_merged / V_merged during the target-view denoising pass.

    Parameters
    ----------
    manager         : AttentionWarpManager – provides merged K/V on demand.
    layer_name      : str – identifies which UNet layer this processor serves.
    total_steps     : int – total diffusion denoising steps (for decay schedule).
    inject_until_t  : float – fraction of total steps; after this step the
                      injection weight linearly decays to 0.
    """

    def __init__(
        self,
        manager: "AttentionWarpManager",
        layer_name: str,
        total_steps: int = 20,
        inject_until_t: float = 0.5,
    ):
        self.manager = manager
        self.layer_name = layer_name
        self.total_steps = total_steps
        self.inject_until_t = inject_until_t
        self._step_counter = 0

    def reset_step_counter(self):
        self._step_counter = 0

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        is_cross = encoder_hidden_states is not None
        context = encoder_hidden_states if is_cross else hidden_states

        q = attn.head_to_batch_dim(attn.to_q(hidden_states))
        k_self = attn.head_to_batch_dim(attn.to_k(context))
        v_self = attn.head_to_batch_dim(attn.to_v(context))

        # Compute temporal decay weight
        inject_weight = self._compute_inject_weight()

        merged = self.manager.get_merged_kv(self.layer_name)
        if merged is not None and inject_weight > 0.0:
            k_merged, v_merged = merged  # (seq_len, dim)
            B, S, D = k_self.shape

            # Expand merged K/V to batch dimension
            k_m = k_merged.unsqueeze(0).expand(B, -1, -1)
            v_m = v_merged.unsqueeze(0).expand(B, -1, -1)

            # Interpolate between self K/V and merged K/V
            k = (1.0 - inject_weight) * k_self + inject_weight * k_m
            v = (1.0 - inject_weight) * v_self + inject_weight * v_m
        else:
            k = k_self
            v = v_self

        self._step_counter += 1

        attn_probs = attn.get_attention_scores(q, k, attention_mask)
        out = torch.bmm(attn_probs, v)
        out = attn.batch_to_head_dim(out)
        return attn.to_out[0](out)

    def _compute_inject_weight(self) -> float:
        """
        Returns injection strength in [0, 1].
        Starts at 1.0, linearly decays to 0 once `inject_until_t` fraction
        of denoising steps have passed.
        """
        threshold = int(self.total_steps * self.inject_until_t)
        if self._step_counter >= threshold:
            return 0.0
        return 1.0 - self._step_counter / max(threshold, 1)


# ---------------------------------------------------------------------------
# Main manager class
# ---------------------------------------------------------------------------

class AttentionWarpManager:
    """
    Manages multi-view attention extraction, 3-D blending, and UNet injection.

    Workflow
    --------
    1. Call extract_and_store_features(key_views) once before each denoising run.
    2. Call set_target_view(tgt_cam, tgt_depth) to prepare for a specific target.
    3. Call apply_to_unet(unet) to register processors that blend K/V on-the-fly.
    4. Run the denoising loop as usual.
    5. Optionally call remove_from_unet(unet) to restore original processors.
    """

    def __init__(
        self,
        unet,                          # ip2p UNet
        vae,                           # ip2p VAE (for encoding key images)
        scheduler,                     # DDIMScheduler
        weights_dtype: torch.dtype = torch.float16,
        total_denoising_steps: int = 20,
        inject_until_t: float = 0.5,
        occlusion_threshold: float = 0.05,  # relative depth tolerance for visibility check
    ):
        self.unet = unet
        self.vae = vae
        self.scheduler = scheduler
        self.weights_dtype = weights_dtype
        self.total_denoising_steps = total_denoising_steps
        self.inject_until_t = inject_until_t
        self.occlusion_threshold = occlusion_threshold

        # Storage populated by extract_and_store_features
        # key_features[layer_name] = list of dicts per key view:
        #   {"k": Tensor(seq, dim), "v": Tensor(seq, dim),
        #    "pts_world": Tensor(seq, 3), "cam": Camera,
        #    "depth": Tensor(H,W), "normal": Tensor(H,W,3) or None}
        self.key_features: Dict[str, List[dict]] = {}
        self.key_cams: List = []
        self.key_depths: List[torch.Tensor] = []
        self.key_normals: List[Optional[torch.Tensor]] = []

        # Target-view info set by set_target_view
        self.tgt_cam = None
        self.tgt_depth: Optional[torch.Tensor] = None

        # Cached merged K/V per layer for the current target view
        self._merged_kv: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        # Track original attn processors so we can restore them
        self._original_processors: Dict = {}

        # Hook handles for K/V extraction
        self._hooks: List = []

        # Temporary buffers during a single forward pass (extraction mode)
        self._current_kv_buf: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_and_store_features(
        self,
        key_images: List[torch.Tensor],   # list of (1, 3, H, W) tensors in [0,1]
        key_cams: List,                   # list of gaussiansplatting Camera objects
        key_depths: List[torch.Tensor],   # list of (H, W) depth tensors
        key_normals: Optional[List[Optional[torch.Tensor]]] = None,
        # list of (H, W, 3) normal tensors or None per view
        text_embeddings: Optional[torch.Tensor] = None,
        # (1, 77, 768) – if None, use zero embeddings
        add_noise_t: int = 500,           # timestep for forward noising
    ) -> None:
        """
        Run a single UNet forward pass per key view with forward-process noise
        and capture Self-Attention K / V per layer.  Also back-projects each
        token to a 3-D world point using the corresponding depth map.

        Parameters
        ----------
        key_images  : rendered key-view images (1,3,H,W) in [0,1].
        key_cams    : gaussiansplatting Camera objects for each key view.
        key_depths  : 2DGS depth maps (H_img, W_img) per key view.
        key_normals : optional per-pixel normal maps (H, W, 3) per key view.
        text_embeddings : (optionally) text conditioning tensor from ip2p.
        add_noise_t : how much noise to add before the forward pass (controls
                      which diffusion feature level is captured).
        """
        assert len(key_images) == len(key_cams) == len(key_depths)
        self.key_cams = key_cams
        self.key_depths = key_depths
        self.key_normals = key_normals or [None] * len(key_cams)
        self.key_features = {}  # reset

        device = next(self.unet.parameters()).device

        for view_idx, (img, cam, depth) in enumerate(
            zip(key_images, key_cams, key_depths)
        ):
            # --- Encode image to latent ---
            img_bchw = img.to(device=device, dtype=self.weights_dtype)
            img_bchw = img_bchw * 2.0 - 1.0
            latent = self.vae.encode(img_bchw).latent_dist.sample()
            latent = latent * self.vae.config.scaling_factor
            latent = latent.to(self.weights_dtype)

            latent_H, latent_W = latent.shape[2], latent.shape[3]
            H_img = int(cam.image_height)
            W_img = int(cam.image_width)

            # --- Add noise (forward process) ---
            t_tensor = torch.tensor([add_noise_t], device=device, dtype=torch.long)
            noise = torch.randn_like(latent)
            noisy_latent = self.scheduler.add_noise(latent, noise, t_tensor)

            # For ip2p the model input is (noisy_latent | image_cond_latent)
            # We only need features; pass zero image-cond to keep it simple.
            image_cond = torch.zeros_like(latent)
            model_input = torch.cat([noisy_latent, image_cond], dim=1)

            # --- Text embeddings ---
            if text_embeddings is None:
                enc_hs = torch.zeros(
                    1, 77, self.unet.config.cross_attention_dim,
                    device=device, dtype=self.weights_dtype
                )
            else:
                enc_hs = text_embeddings.to(device=device, dtype=self.weights_dtype)

            # --- Register hooks to capture K/V ---
            kv_buf: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

            def _make_hook(layer_name: str):
                def _hook(module, args, kwargs_hook, output):
                    # args[0] = hidden_states (the input x to self-attn)
                    x = args[0] if args else kwargs_hook.get("hidden_states")
                    if x is None:
                        return
                    with torch.no_grad():
                        enc = x  # self-attention → no encoder_hidden_states
                        k = module.to_k(enc).detach().float()
                        v = module.to_v(enc).detach().float()
                    # k, v shape: (1, seq_len, inner_dim)
                    kv_buf[layer_name] = (k.squeeze(0), v.squeeze(0))
                return _hook

            hooks = []
            for name, module in self.unet.named_modules():
                if _is_self_attn(module):
                    h = module.register_forward_hook(
                        _make_hook(name), with_kwargs=True
                    )
                    hooks.append(h)

            # --- Forward pass ---
            self.unet(
                model_input,
                t_tensor,
                encoder_hidden_states=enc_hs,
            )

            for h in hooks:
                h.remove()

            # --- Back-project each token to 3-D world space ---
            pts_world = _backproject_depth(depth.float(), cam, latent_H, latent_W)
            # pts_world: (latent_H * latent_W, 3)

            for layer_name, (k, v) in kv_buf.items():
                if layer_name not in self.key_features:
                    self.key_features[layer_name] = []
                self.key_features[layer_name].append({
                    "k": k,             # (seq, inner_dim)
                    "v": v,             # (seq, inner_dim)
                    "pts_world": pts_world,
                    "cam": cam,
                    "depth": depth,
                    "normal": self.key_normals[view_idx],
                    "H_img": H_img,
                    "W_img": W_img,
                    "latent_H": latent_H,
                    "latent_W": latent_W,
                })

    def set_target_view(
        self,
        tgt_cam,
        tgt_depth: torch.Tensor,  # (H, W)
    ) -> None:
        """
        Set the target camera and depth map.  Triggers computation of
        warp_and_blend_attention() for all layers and caches the result.
        """
        self.tgt_cam = tgt_cam
        self.tgt_depth = tgt_depth
        self._merged_kv = {}  # clear cache

        for layer_name in self.key_features:
            k_m, v_m = self.warp_and_blend_attention(layer_name)
            if k_m is not None:
                self._merged_kv[layer_name] = (k_m, v_m)

    def get_merged_kv(
        self, layer_name: str
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return pre-computed merged (K, V) for a layer, or None."""
        return self._merged_kv.get(layer_name, None)

    @torch.no_grad()
    def warp_and_blend_attention(
        self, layer_name: str
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        For the current target view, project all stored 3-D key-view points into
        the target frame and compute a weighted blend of their K / V features.

        Weight  w_i = visibility × angular_similarity × confidence

        Returns
        -------
        k_merged : (tgt_seq_len, inner_dim)  or None if no key views available.
        v_merged : same shape.
        """
        views = self.key_features.get(layer_name)
        if not views or self.tgt_cam is None:
            return None, None

        device = views[0]["k"].device
        tgt_cam = self.tgt_cam
        tgt_depth = self.tgt_depth.to(device).float() if self.tgt_depth is not None else None

        latent_H = views[0]["latent_H"]
        latent_W = views[0]["latent_W"]
        tgt_seq_len = latent_H * latent_W
        inner_dim = views[0]["k"].shape[-1]

        # Target back-projected world points (for visibility check)
        tgt_pts_world: Optional[torch.Tensor] = None
        if tgt_depth is not None:
            tgt_pts_world = _backproject_depth(tgt_depth, tgt_cam, latent_H, latent_W)

        # Direction from target camera to scene (per pixel)
        tgt_center = tgt_cam.camera_center.float().to(device)  # (3,)

        accumulated_k = torch.zeros(tgt_seq_len, inner_dim, device=device)
        accumulated_v = torch.zeros(tgt_seq_len, inner_dim, device=device)
        accumulated_w = torch.zeros(tgt_seq_len, 1, device=device)

        for view_info in views:
            key_cam = view_info["cam"]
            key_depth = view_info["depth"].to(device).float()
            key_pts_world = view_info["pts_world"].to(device)  # (key_seq, 3)
            key_k = view_info["k"].to(device)  # (key_seq, dim)
            key_v = view_info["v"].to(device)  # (key_seq, dim)
            key_normal = view_info["normal"]
            H_img = view_info["H_img"]
            W_img = view_info["W_img"]

            # ---- 1. Project key-view world points into target image plane ----
            uv_in_tgt, valid = _project_points(
                key_pts_world, tgt_cam, latent_H, latent_W, H_img, W_img
            )
            # uv_in_tgt: (key_seq, 2),  valid: (key_seq,) bool

            if valid.sum() == 0:
                continue

            # Integer pixel indices in the target latent map
            u_idx = uv_in_tgt[:, 0].long().clamp(0, latent_W - 1)
            v_idx = uv_in_tgt[:, 1].long().clamp(0, latent_H - 1)
            tgt_token_idx = v_idx * latent_W + u_idx  # (key_seq,)

            # ---- 2a. Visibility weight – compare projected depth vs target depth ----
            w_vis = torch.ones(len(key_pts_world), device=device)
            if tgt_pts_world is not None:
                # Depth of the 3-D key point as seen from the target camera
                W2C_tgt = tgt_cam.world_view_transform.T.float().to(device)
                pts_in_tgt_cam = key_pts_world @ W2C_tgt[:3, :3].T + W2C_tgt[:3, 3]
                proj_depth = pts_in_tgt_cam[:, 2]  # (key_seq,)

                # Depth of the target surface at the same pixel
                tgt_surface_depth = tgt_pts_world[tgt_token_idx, 2].clamp(min=1e-4)
                rel_diff = (proj_depth - tgt_surface_depth).abs() / tgt_surface_depth
                w_vis = (rel_diff < self.occlusion_threshold).float()  # 1 if visible

            # ---- 2b. Angular similarity weight ----
            key_center = key_cam.camera_center.float().to(device)  # (3,)
            # Direction vectors from each camera to each 3-D point
            tgt_dir = F.normalize(key_pts_world - tgt_center.unsqueeze(0), dim=-1)  # (N,3)
            key_dir = F.normalize(key_pts_world - key_center.unsqueeze(0), dim=-1)  # (N,3)
            w_ang = (tgt_dir * key_dir).sum(dim=-1).clamp(0.0, 1.0)  # cosine ∈ [0,1]

            # ---- 2c. Confidence weight – key-view normal vs key view direction ----
            w_conf = torch.ones(len(key_pts_world), device=device)
            if key_normal is not None:
                # Resize normal map to latent resolution
                n = key_normal.float().to(device)  # (H, W, 3)
                n_lat = F.interpolate(
                    n.permute(2, 0, 1).unsqueeze(0),  # (1,3,H,W)
                    size=(latent_H, latent_W),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze().permute(1, 2, 0).reshape(-1, 3)  # (key_seq, 3)
                n_lat = F.normalize(n_lat, dim=-1)
                # dot(normal, -view_dir) – normals pointing toward camera are reliable
                w_conf = (n_lat * (-key_dir)).sum(dim=-1).clamp(0.0, 1.0)

            # ---- 3. Combined per-token weight ----
            w = (w_vis * w_ang * w_conf).unsqueeze(-1)  # (key_seq, 1)
            w = w * valid.float().unsqueeze(-1)

            # ---- 4. Scatter into target token space (nearest-neighbour) ----
            tgt_tok = tgt_token_idx.clamp(0, tgt_seq_len - 1)

            accumulated_k.scatter_add_(0, tgt_tok.unsqueeze(-1).expand_as(key_k), w * key_k)
            accumulated_v.scatter_add_(0, tgt_tok.unsqueeze(-1).expand_as(key_v), w * key_v)
            accumulated_w.scatter_add_(0, tgt_tok, w)

        # Normalise
        denom = accumulated_w.clamp(min=1e-8)
        k_merged = accumulated_k / denom
        v_merged = accumulated_v / denom

        return k_merged, v_merged

    def apply_to_unet(self, unet=None) -> None:
        """
        Register MultiViewWarpAttnProcessor on every self-attention layer of
        the ip2p UNet using diffusers' set_attn_processor().

        Saves the original processors so they can be restored later.
        """
        unet = unet or self.unet
        self._original_processors = {}

        new_processors: Dict = {}
        for name, module in unet.named_modules():
            if _is_self_attn(module):
                self._original_processors[name] = module.processor
                proc = MultiViewWarpAttnProcessor(
                    manager=self,
                    layer_name=name,
                    total_steps=self.total_denoising_steps,
                    inject_until_t=self.inject_until_t,
                )
                new_processors[name] = proc

        unet.set_attn_processor(new_processors)

    def remove_from_unet(self, unet=None) -> None:
        """Restore the original attention processors."""
        unet = unet or self.unet
        if self._original_processors:
            unet.set_attn_processor(self._original_processors)
            self._original_processors = {}

    def reset_step_counters(self) -> None:
        """Reset denoising-step counters in all registered processors."""
        for _, module in self.unet.named_modules():
            if hasattr(module, "processor") and isinstance(
                module.processor, MultiViewWarpAttnProcessor
            ):
                module.processor.reset_step_counter()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _is_self_attn(module) -> bool:
    """
    Heuristic: the module is a self-attention projection layer if it
    has both `to_k` and `to_v` submodules (diffusers Attention class).
    We filter out cross-attention by checking that the K projection input
    dimension equals the Q projection input dimension (same as `dim`).
    """
    return (
        hasattr(module, "to_k")
        and hasattr(module, "to_v")
        and hasattr(module, "to_q")
        and hasattr(module, "to_out")
        and hasattr(module, "heads")
        # Exclude cross-attention: in diffusers, self-attn has to_k weight
        # with in_features == to_q in_features.
        and module.to_k.weight.shape[1] == module.to_q.weight.shape[1]
    )
