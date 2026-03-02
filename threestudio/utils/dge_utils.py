from typing import Type, Optional
import threading
import torch
import os
from contextlib import nullcontext

# Latency hierarchy: when running under edit_multiview, unet timers must use the same
# tree (edit_multiview.*) so they appear in the summary. Call set_unet_latency_prefix()
# before each forward_unet; make_dge_block uses this or the default below.
_UNET_LATENCY_PREFIX = threading.local()
DEFAULT_UNET_LATENCY_PREFIX = "edit_all_view.guidance_batch.edit_latents.diffusion_loop.batch_processing.unet_forward"


def get_unet_latency_prefix() -> Optional[str]:
    return getattr(_UNET_LATENCY_PREFIX, "value", None)


def set_unet_latency_prefix(prefix: Optional[str]) -> None:
    _UNET_LATENCY_PREFIX.value = prefix
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from PIL import Image
import torch
import yaml
import math

from gaussiansplatting.utils.graphics_utils import get_fundamental_matrix_with_H
import torchvision.transforms as T
from torchvision.io import read_video,write_video
import os
import random
import numpy as np
from torchvision.io import write_video
from kornia.geometry.transform import remap

def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.
    
    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    
    return False


def batch_cosine_sim(x, y):
    if type(x) is list:
        x = torch.cat(x, dim=0)
    if type(y) is list:
        y = torch.cat(y, dim=0)
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)
    similarity = x @ y.T
    return similarity


def resize_bool_tensor(bool_tensor, size):
    """
    Resizes a boolean tensor to a new size using nearest neighbor interpolation.
    """
    # Convert boolean tensor to float
    H_new, W_new = size
    tensor_float = bool_tensor.float()

    # Resize using nearest interpolation
    resized_float = torch.nn.functional.interpolate(tensor_float, size=(H_new, W_new), mode='nearest')

    # Convert back to boolean
    resized_bool = resized_float > 0.5
    return resized_bool

def point_to_line_dist(points, lines):
    """
    Calculate the distance from points to lines in 2D.
    points: Nx3
    lines: Mx3

    return distance: NxM
    """
    numerator = torch.abs(lines @ points.T)
    denominator = torch.linalg.norm(lines[:,:2], dim=1, keepdim=True)
    return numerator / denominator

def save_video_frames(video_path, img_size=(512,512)):
    video, _, _ = read_video(video_path, output_format="TCHW")
    # rotate video -90 degree if video is .mov format. this is a weird bug in torchvision
    if video_path.endswith('.mov'):
        video = T.functional.rotate(video, -90)
    video_name = Path(video_path).stem
    os.makedirs(f'data/{video_name}', exist_ok=True)
    for i in range(len(video)):
        ind = str(i).zfill(5)
        image = T.ToPILImage()(video[i])
        image_resized = image.resize((img_size),  resample=Image.Resampling.LANCZOS)
        image_resized.save(f'data/{video_name}/{ind}.png')

def add_dict_to_yaml_file(file_path, key, value):
    data = {}

    # If the file already exists, load its contents into the data dictionary
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)

    # Add or update the key-value pair
    data[key] = value

    # Save the data back to the YAML file
    with open(file_path, 'w') as file:
        yaml.dump(data, file)
        
def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.
    
    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    
    return False


def batch_cosine_sim(x, y):
    if type(x) is list:
        x = torch.cat(x, dim=0)
    if type(y) is list:
        y = torch.cat(y, dim=0)
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)
    similarity = x @ y.T
    return similarity


def load_imgs(data_path, n_frames, device='cuda', pil=False):
    imgs = []
    pils = []
    for i in range(n_frames):
        img_path = os.path.join(data_path, "%05d.jpg" % i)
        if not os.path.exists(img_path):
            img_path = os.path.join(data_path, "%05d.png" % i)
        img_pil = Image.open(img_path)
        pils.append(img_pil)
        img = T.ToTensor()(img_pil).unsqueeze(0)
        imgs.append(img)
    if pil:
        return torch.cat(imgs).to(device), pils
    return torch.cat(imgs).to(device)


def save_video(raw_frames, save_path, fps=10):
    video_codec = "libx264"
    video_options = {
        "crf": "18",  # Constant Rate Factor (lower value = higher quality, 18 is a good balance)
        "preset": "slow",  # Encoding preset (e.g., ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
    }

    frames = (raw_frames * 255).to(torch.uint8).cpu().permute(0, 2, 3, 1)
    write_video(save_path, frames, fps=fps, video_codec=video_codec, options=video_options)


def compute_epipolar_constrains(cam1, cam2, current_H=64, current_W=64, downsample_factor=8):
    n_frames = 1
    sequence_length = current_W * current_H
    
    # return torch.zeros(sequence_length, sequence_length, dtype=torch.bool).cuda()

    # if downsample_factor != 8:
    #     # Return zeros for scales 1, 2, 4
    #     return torch.zeros(sequence_length, sequence_length, dtype=torch.bool).cuda()
    
    # Only compute epipolar constraints for scale 8
    fundamental_matrix_1 = []
    
    fundamental_matrix_1.append(get_fundamental_matrix_with_H(cam1, cam2, current_H, current_W))
    fundamental_matrix_1 = torch.stack(fundamental_matrix_1, dim=0)

    x = torch.arange(current_W)
    y = torch.arange(current_H)
    x, y = torch.meshgrid(x, y, indexing='xy')
    x = x.reshape(-1)
    y = y.reshape(-1)
    heto_cam2 = torch.stack([x, y, torch.ones(size=(len(x),))], dim=1).view(-1, 3).cuda()
    heto_cam1 = torch.stack([x, y, torch.ones(size=(len(x),))], dim=1).view(-1, 3).cuda()
    # epipolar_line: n_frames X seq_len,  3
    line1 = (heto_cam2.unsqueeze(0).repeat(n_frames, 1, 1) @ fundamental_matrix_1.cuda()).view(-1, 3)
    
    distance1 = point_to_line_dist(heto_cam1, line1)

    
    idx1_epipolar = distance1 > 1 # sequence_length x sequence_lengths

    return idx1_epipolar

def save_epipolar_constraints_image(epipolar_constraints, H, W, save_path, cam_idx, key_cam_idx, downsample_factor):
    """
    Save epipolar constraints as an image for visualization
    
    Args:
        epipolar_constraints: Boolean tensor of shape (H*W, H*W)
        H, W: Height and width of the image
        save_path: Directory to save the image
        cam_idx: Camera index
        key_cam_idx: Key camera index  
        downsample_factor: Downsample factor used
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Convert boolean tensor to numpy array
    epipolar_np = epipolar_constraints.cpu().numpy()
    
    # For visualization, we'll show the constraint map for a specific pixel
    # Let's use the center pixel as reference
    center_idx = (H // 2) * W + (W // 2)
    constraint_map = epipolar_np[center_idx, :].reshape(H, W)
    
    # Create the image
    plt.figure(figsize=(10, 10))
    plt.imshow(constraint_map, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(f'Epipolar Constraints Map\nCamera {cam_idx} -> Key Camera {key_cam_idx}\nDownsample: {downsample_factor} (H={H}, W={W})')
    plt.xlabel('Width')
    plt.ylabel('Height')
    
    # Save the image
    filename = f'epipolar_cam{cam_idx}_key{key_cam_idx}_downsample{downsample_factor}_H{H}W{W}.png'
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved epipolar constraints image: {filepath}")
    
    return filepath

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def register_epipolar_constrains(diffusion_model, epipolar_constrains):
    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "epipolar_constrains", epipolar_constrains)

def register_cams(diffusion_model, cams, pivot_this_batch, key_cams):
    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "cams", cams)
            setattr(module, "pivot_this_batch", pivot_this_batch)
            setattr(module, "key_cams", key_cams)

def register_pivotal(diffusion_model, is_pivotal):
    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "pivotal_pass", is_pivotal)


def register_store_kf_attn_output(diffusion_model, store: bool):
    """When False, pivotal pass does not write kf_attn_output (e.g. key denoise in edit_multiview)."""
    for _, module in diffusion_model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "store_kf_attn_output", store)


def register_batch_idx(diffusion_model, batch_idx):
    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "batch_idx", batch_idx)


def register_t(diffusion_model, t):

    for _, module in diffusion_model.named_modules():
    # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "t", t)


def register_normal_attention(model):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out
        def forward(x, encoder_hidden_states=None, attention_mask=None):
            # assert encoder_hidden_states is None 
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            q = self.to_q(x)
            k = self.to_k(encoder_hidden_states)
            v = self.to_v(encoder_hidden_states)

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.head_to_batch_dim(q)
            key = self.head_to_batch_dim(k)
            value = self.head_to_batch_dim(v)

            attention_probs = self.get_attention_scores(query, key)
            hidden_states = torch.bmm(attention_probs, value)
            out = self.batch_to_head_dim(hidden_states)

            return to_out(out)

        return forward

    for _, module in model.unet.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            module.attn1.normal_attn = sa_forward(module.attn1)
            module.use_normal_attn = True

def register_normal_attn_flag(diffusion_model, use_normal_attn):
    for _, module in diffusion_model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "use_normal_attn", use_normal_attn)
            # Also propagate to attn1 so that sa_forward (register_extended_attention)
            # can fall back to normal attention for non-3-batch layouts (e.g. Lite-ISM).
            if hasattr(module, "attn1"):
                setattr(module.attn1, "use_normal_attn", use_normal_attn)

def register_latency_logger(diffusion_model, latency_logger):
    for _, module in diffusion_model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "latency_logger", latency_logger)

def register_extended_attention(model):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out
        def forward(x, encoder_hidden_states=None, attention_mask=None):
            # Fall back to normal attention when flagged (e.g. 2-batch Lite-ISM calls)
            if getattr(self, "use_normal_attn", False):
                is_cross = encoder_hidden_states is not None
                enc = encoder_hidden_states if is_cross else x
                q = self.head_to_batch_dim(self.to_q(x))
                k = self.head_to_batch_dim(self.to_k(enc))
                v = self.head_to_batch_dim(self.to_v(enc))
                attn_probs = self.get_attention_scores(q, k, attention_mask)
                out = self.batch_to_head_dim(torch.bmm(attn_probs, v))
                return to_out(out)

            assert encoder_hidden_states is None
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            n_frames = batch_size // 3
            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            q = self.to_q(x)
            k = self.to_k(encoder_hidden_states)
            v = self.to_v(encoder_hidden_states)
            
            k_text = k[:n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            k_image = k[n_frames: 2*n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            k_uncond = k[2*n_frames:].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)

            v_text = v[:n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            v_image = v[n_frames:2*n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            v_uncond = v[2*n_frames:].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)

            q_text = self.head_to_batch_dim(q[:n_frames])
            q_image = self.head_to_batch_dim(q[n_frames: 2*n_frames])
            q_uncond = self.head_to_batch_dim(q[2 * n_frames:])

            k_text = self.head_to_batch_dim(k_text)
            k_image = self.head_to_batch_dim(k_image)
            k_uncond = self.head_to_batch_dim(k_uncond)

            
            v_text = self.head_to_batch_dim(v_text)
            v_image = self.head_to_batch_dim(v_image)
            v_uncond = self.head_to_batch_dim(v_uncond)

            out_text = []
            out_image = []
            out_uncond = []

            q_text = q_text.view(n_frames, h, sequence_length, dim // h)
            k_text = k_text.view(n_frames, h, sequence_length * n_frames, dim // h)
            v_text = v_text.view(n_frames, h, sequence_length * n_frames, dim // h)

            q_image = q_image.view(n_frames, h, sequence_length, dim // h)
            k_image = k_image.view(n_frames, h, sequence_length * n_frames, dim // h)
            v_image = v_image.view(n_frames, h, sequence_length * n_frames, dim // h)

            q_uncond = q_uncond.view(n_frames, h, sequence_length, dim // h)
            k_uncond = k_uncond.view(n_frames, h, sequence_length * n_frames, dim // h)
            v_uncond = v_uncond.view(n_frames, h, sequence_length * n_frames, dim // h)

            for j in range(h):
                sim_text = torch.bmm(q_text[:, j], k_text[:, j].transpose(-1, -2)) * self.scale
                sim_image = torch.bmm(q_image[:, j], k_image[:, j].transpose(-1, -2)) * self.scale
                sim_uncond = torch.bmm(q_uncond[:, j], k_uncond[:, j].transpose(-1, -2)) * self.scale
                
                out_text.append(torch.bmm(sim_text.softmax(dim=-1), v_text[:, j]))
                out_image.append(torch.bmm(sim_image.softmax(dim=-1), v_image[:, j]))
                out_uncond.append(torch.bmm(sim_uncond.softmax(dim=-1), v_uncond[:, j]))

            out_text = torch.cat(out_text, dim=0).view(h, n_frames, sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)
            out_image = torch.cat(out_image, dim=0).view(h, n_frames,sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)
            out_uncond = torch.cat(out_uncond, dim=0).view(h, n_frames,sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)

            out = torch.cat([out_text, out_image, out_uncond], dim=0)
            out = self.batch_to_head_dim(out)

            return to_out(out)

        return forward

    for _, module in model.unet.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            module.attn1.forward = sa_forward(module.attn1)


def compute_camera_distance(cams, key_cams):
    cam_centers = [cam.camera_center for cam in cams]
    key_cam_centers = [cam.camera_center for cam in key_cams] 
    cam_centers = torch.stack(cam_centers).cuda()
    key_cam_centers = torch.stack(key_cam_centers).cuda()
    cam_distance = torch.cdist(cam_centers, key_cam_centers)

    return cam_distance   

def make_dge_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:

    class DGEBlock(block_class):
        def forward(
            self,
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            timestep=None,
            cross_attention_kwargs=None,
            class_labels=None,
        ) -> torch.Tensor:
            # When flagged, behave like the original transformer block.
            # This is required for non-DGE batch layouts (e.g. 2B for Lite-ISM).
            if getattr(self, "use_normal_attn", False):
                return block_class.forward(
                    self,
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                )
            
            # Initialize latency logger if available (prefix set by caller so hierarchy matches edit_multiview vs edit_all_view)
            latency_logger = getattr(self, 'latency_logger', None)
            _latency_base = get_unet_latency_prefix() or DEFAULT_UNET_LATENCY_PREFIX

            with latency_logger.timeit(f'{_latency_base}.dge_block.init') if latency_logger else nullcontext():
                batch_size, sequence_length, dim = hidden_states.shape
                n_frames = batch_size // 3
                hidden_states = hidden_states.view(3, n_frames, sequence_length, dim)

                if self.use_ada_layer_norm:
                    norm_hidden_states = self.norm1(hidden_states, timestep)
                elif self.use_ada_layer_norm_zero:
                    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                        hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                    )
                else:
                    norm_hidden_states = self.norm1(hidden_states)
            
                norm_hidden_states = norm_hidden_states.view(3, n_frames, sequence_length, dim)
                
                ## key view의 경우에는 pivot_hidden_states로 저장해서 나중에 써먹기
                if self.pivotal_pass: 
                    self.pivot_hidden_states = norm_hidden_states

            if not self.use_normal_attn: 
                
                if self.pivotal_pass:
                    with latency_logger.timeit(f'{_latency_base}.dge_block.pivotal_pass') if latency_logger else nullcontext():
                        self.pivot_hidden_states = norm_hidden_states

                ## key view가 아닌 경우
                else:
                    # -------------------------------------------------------- #
                    # Version B (GP): camera_distance / sim / epipolar 전부 스킵 #
                    # -------------------------------------------------------- #
                    if getattr(self, "use_gp_attn", False):
                        # sparse_xview_attn은 feature_injection 단계에서 처리.
                        # closest_cam은 필요 없지만 self_attention 단계에서
                        # kf_attn_output gather에 쓰이므로 dummy 값 설정.
                        batch_idxs = [self.batch_idx]
                        closest_cam = torch.zeros(n_frames, 1, dtype=torch.long, device=hidden_states.device)
                        cam_distance_min = (torch.zeros(n_frames, 1, device=hidden_states.device),
                                            closest_cam)
                        idx1 = None
                        idx2 = None

                    # -------------------------------------------------------- #
                    # Version C (3D Anchor): 3DGS-based canonical token injection #
                    # -------------------------------------------------------- #
                    elif getattr(self, "use_3d_anchor_attn", False):
                        batch_idxs = [self.batch_idx]
                        closest_cam = torch.zeros(n_frames, 1, dtype=torch.long, device=hidden_states.device)
                        cam_distance_min = (torch.zeros(n_frames, 1, device=hidden_states.device),
                                            closest_cam)
                        idx1 = None
                        idx2 = None

                    # -------------------------------------------------------- #
                    # Version A (Epipolar): 기존 로직 그대로                    #
                    # -------------------------------------------------------- #
                    else:
                        with latency_logger.timeit(f'{_latency_base}.dge_block.camera_distance') if latency_logger else nullcontext():
                            batch_idxs = [self.batch_idx]
                            if self.batch_idx > 0:
                                batch_idxs.append(self.batch_idx - 1)
                            idx1 = []
                            idx2 = []
                            ## 가장 가까운 카메라 찾기
                            cam_distance = compute_camera_distance(self.cams, self.key_cams)
                            cam_distance_min = cam_distance.sort(dim=-1)
                            closest_cam = cam_distance_min[1][:,:len(batch_idxs)]
                            closest_cam_pivot_hidden_states = self.pivot_hidden_states[1][closest_cam]

                        with latency_logger.timeit(f'{_latency_base}.dge_block.spatio_temporal_attention') if latency_logger else nullcontext():
                            sim = torch.einsum('bld,bcsd->bcls', norm_hidden_states[1] / norm_hidden_states[1].norm(dim=-1, keepdim=True), closest_cam_pivot_hidden_states / closest_cam_pivot_hidden_states.norm(dim=-1, keepdim=True)).squeeze()

                        if len(batch_idxs) == 2:
                            sim1, sim2 = sim.chunk(2, dim=1)
                            sim1 = sim1.reshape(-1, sequence_length)
                            sim2 = sim2.reshape(-1, sequence_length)
                            sim1_max = sim1.max(dim=-1)
                            sim2_max = sim2.max(dim=-1)
                            idx1.append(sim1_max[1])
                            idx2.append(sim2_max[1])

                        else:
                            sim = sim.reshape(-1, sequence_length)
                            sim_max = sim.max(dim=-1)
                            idx1.append(sim_max[1])

                        with latency_logger.timeit(f'{_latency_base}.dge_block.epipolar_constraints') if latency_logger else nullcontext():
                            if sequence_length in getattr(self, "epipolar_constrains", {}):
                                if len(batch_idxs) == 2:
                                    idx1 = []
                                    idx2 = []
                                    pivot_this_batch = self.pivot_this_batch

                                    ## EPIPOLAR CONSTRAINT가 활용되는 부분
                                    idx1_epipolar, idx2_epipolar = self.epipolar_constrains[sequence_length].gather(dim=1, index=closest_cam[:, :, None, None].expand(-1, -1, self.epipolar_constrains[sequence_length].shape[2], self.epipolar_constrains[sequence_length].shape[3])).cuda().chunk(2, dim=1)
                                    idx1_epipolar = idx1_epipolar.reshape(n_frames, sequence_length, sequence_length)

                                    idx1_epipolar[pivot_this_batch, ...] = False
                                    idx2_epipolar = idx2_epipolar.reshape(n_frames, sequence_length, sequence_length)

                                    idx1_epipolar = idx1_epipolar.reshape(n_frames * sequence_length, sequence_length)
                                    idx2_epipolar = idx2_epipolar.reshape(n_frames * sequence_length, sequence_length)
                                    idx2_sum = idx2_epipolar.sum(dim=-1)
                                    idx1_sum = idx1_epipolar.sum(dim=-1)

                                    idx1_epipolar[idx1_sum == sequence_length, :] = False
                                    idx2_epipolar[idx2_sum == sequence_length, :] = False
                                    sim1[idx1_epipolar] = 0
                                    sim2[idx2_epipolar] = 0

                                    sim1_max = sim1.max(dim=-1)
                                    sim2_max = sim2.max(dim=-1)
                                    idx1.append(sim1_max[1])
                                    idx2.append(sim2_max[1])


                                else:
                                    idx1 = []
                                    pivot_this_batch = self.pivot_this_batch

                                    # 마스크 불러오기
                                    idx1_epipolar = self.epipolar_constrains[sequence_length].gather(dim=1, index=closest_cam[:, :, None, None].expand(-1, -1, self.epipolar_constrains[sequence_length].shape[2], self.epipolar_constrains[sequence_length].shape[3])).cuda()

                                    idx1_epipolar = idx1_epipolar.view(n_frames, -1, sequence_length)
                                    idx1_epipolar[pivot_this_batch, ...] = False

                                    idx1_epipolar = idx1_epipolar.view(n_frames * sequence_length, sequence_length)
                                    idx1_sum = idx1_epipolar.sum(dim=-1)
                                    idx1_epipolar[idx1_sum == sequence_length, :] = False
                                    sim[idx1_epipolar] = 0 # geometry 적으로 맞지 않는 픽셀의 similarity 값을 0으로 만듦.
                                    sim_max = sim.max(dim=-1) # 각 픽셀에서 가장 유사한 key-view 픽셀 index 를 찾음.
                                    idx1.append(sim_max[1]) # 이 index 를 기준으로 나중에 attention 결과를 gather 함.
                            else:
                                # No epipolar constraints (e.g. multiview target_denoise_loop with epipolar disabled): use similarity max only.
                                if len(batch_idxs) == 2:
                                    idx1 = [sim1_max[1]]
                                    idx2 = [sim2_max[1]]
                                else:
                                    idx1 = [sim_max[1]]

                        idx1 = torch.stack(idx1 * 3, dim=0) # 3, n_frames * seq_len
                        idx1 = idx1.squeeze(1)

                        if len(batch_idxs) == 2:
                            idx2 = torch.stack(idx2 * 3, dim=0) # 3, n_frames * seq_len
                            idx2 = idx2.squeeze(1)

                            
            
            # 1. Self-Attention
            with latency_logger.timeit(f'{_latency_base}.dge_block.self_attention') if latency_logger else nullcontext():
                cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
                if self.use_normal_attn:
                    # print("use normal attn")
                    self.attn_output = self.attn1.normal_attn(
                            norm_hidden_states.view(batch_size, sequence_length, dim),
                            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                            **cross_attention_kwargs,
                        )         
                else:
                    # print("use extend attn")
                    if self.pivotal_pass:
                        # norm_hidden_states.shape = 3, n_frames * seq_len, dim
                        self.attn_output = self.attn1(
                                norm_hidden_states.view(batch_size, sequence_length, dim),
                                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                                **cross_attention_kwargs,
                            )
                        # Only cache for target-phase pivotal pass; skip in key denoise (edit_multiview).
                        if getattr(self, "store_kf_attn_output", True):
                            self.kf_attn_output = self.attn_output

                    else:
                        batch_kf_size, _, _ = self.kf_attn_output.shape
                        if getattr(self, "use_gp_attn", False):
                            # GP 경로: closest_cam gather 없이 kf_attn_output 그대로 보존
                            # (feature_injection 단계에서 sparse_xview_attn으로 처리)
                            self.attn_output = self.kf_attn_output
                        elif getattr(self, "use_3d_anchor_attn", False):
                            # 3D anchor: "gather" = pivot self-attn as-is (like similarity); "blend" = filled in feature_injection
                            if getattr(self, "injection_3d_anchor_style", "blend") == "gather":
                                n_frames_attn = batch_size // 3
                                pivot_attn = self.kf_attn_output.view(3, -1, sequence_length, dim)[:, 0]  # [3, L, D]
                                self.attn_output = pivot_attn.unsqueeze(1).expand(3, n_frames_attn, sequence_length, dim)
                            else:
                                self.attn_output = torch.zeros_like(norm_hidden_states.view(batch_size, sequence_length, dim))
                        else:
                            self.attn_output = self.kf_attn_output.view(3, batch_kf_size // 3, sequence_length, dim)[:,
                                            closest_cam]

            if self.use_ada_layer_norm_zero:
                self.n = gate_msa.unsqueeze(1) * self.attn_output

            # gather values from attn_output, using idx as indices, and get a tensor of shape 3, n_frames, seq_len, dim
            with latency_logger.timeit(f'{_latency_base}.dge_block.feature_injection') if latency_logger else nullcontext():
                # Optionally disable all feature-injection logic (keep self-attn output as-is).
                if getattr(self, "disable_feature_injection", False):
                    attn_output = self.attn_output
                elif not self.use_normal_attn:
                    if not self.pivotal_pass:
                        # ---------------------------------------------------------- #
                        # Version C: 3D-Anchor canonical token injection               #
                        # t_j = Σ_k w_k φ_k(proj(k,j)) / Σw_k;  F(v,p) = Σ_a t_j;   #
                        # h_out = (1-λ)*h + λ*F  =>  attn_output = λ*(F - h)        #
                        # ---------------------------------------------------------- #
                        if getattr(self, "use_3d_anchor_attn", False):
                            with latency_logger.timeit(f'{_latency_base}.dge_block.feature_injection.3d_anchor') if latency_logger else nullcontext():
                                device = hidden_states.device
                                dtype = hidden_states.dtype
                                H = W = int(sequence_length ** 0.5)
                                hw_key = (H, W)
                                pix2g_id = getattr(self, "anchor_3d_pix2g_id", {}).get(hw_key, None)
                                pix2g_w = getattr(self, "anchor_3d_pix2g_w", {}).get(hw_key, None)
                                g2uv_list = getattr(self, "anchor_3d_g2uv", {}).get(hw_key, [])
                                g_vis_list = getattr(self, "anchor_3d_g_vis", {}).get(hw_key, [])
                                lam = getattr(self, "injection_lambda", 0.5)
                                style_3d = getattr(self, "injection_3d_anchor_style", "blend")

                                if pix2g_id is not None and len(g2uv_list) > 0:
                                    pivot_view_index = getattr(self, "anchor_3d_pivot_view_index", None)
                                    g2uv_all = getattr(self, "anchor_3d_g2uv_all", {}).get(hw_key, None)

                                    if style_3d == "gather" and g2uv_all is not None and pivot_view_index is not None:
                                        # Similarity-like: remap by 3D GS (top-1 Gaussian -> pivot pixel), then gather; residual = h + attn_output
                                        pix2g_id = pix2g_id.to(device)   # [n_frames, L, K]
                                        g2uv_pivot = g2uv_all[pivot_view_index].to(device)   # [N_g, 2]
                                        j_star = pix2g_id[:, :, 0]   # [n_frames, L]
                                        uv_pivot = g2uv_pivot[j_star]   # [n_frames, L, 2]
                                        u = uv_pivot[..., 0].clamp(0, W - 1).long()
                                        v = uv_pivot[..., 1].clamp(0, H - 1).long()
                                        idx_3d = (v * W + u).clamp(0, sequence_length - 1)   # [n_frames, L]
                                        idx_gather = idx_3d.unsqueeze(0).expand(3, -1, -1).unsqueeze(-1).expand(-1, -1, -1, dim)
                                        attn_output = self.attn_output.gather(dim=2, index=idx_gather)
                                        attn_output = attn_output.reshape(batch_size, sequence_length, dim)
                                        if attn_output.dtype != self.norm2.weight.dtype:
                                            attn_output = attn_output.to(self.norm2.weight.dtype)
                                    else:
                                        # blend: t_j, F(v,p), attn_output = λ*(F - h)
                                        kf_src = self.kf_attn_output.view(3, -1, sequence_length, dim)[1]  # [n_key, L, D]
                                        n_key = kf_src.shape[0]
                                        N_g = g2uv_list[0].shape[0]

                                        if g2uv_all is not None and pivot_view_index is not None:
                                            g2uv_pivot = g2uv_all[pivot_view_index].to(device)
                                            u = g2uv_pivot[:, 0].clamp(0, W - 1).long()
                                            v = g2uv_pivot[:, 1].clamp(0, H - 1).long()
                                            linear = (v * W + u).clamp(0, sequence_length - 1)
                                            t_j = kf_src[0].index_select(0, linear).to(dtype)
                                        else:
                                            t_j = torch.zeros(N_g, dim, device=device, dtype=dtype)
                                            w_sum = torch.zeros(N_g, 1, device=device, dtype=dtype)
                                            for k in range(n_key):
                                                g2uv_k = g2uv_list[k].to(device)
                                                g_vis_k = g_vis_list[k].to(device).float().unsqueeze(1)
                                                u = g2uv_k[:, 0].clamp(0, W - 1).long()
                                                v = g2uv_k[:, 1].clamp(0, H - 1).long()
                                                linear = (v * W + u).clamp(0, sequence_length - 1)
                                                phi_k_j = kf_src[k].index_select(0, linear)
                                                t_j = t_j + g_vis_k * phi_k_j.to(dtype)
                                                w_sum = w_sum + g_vis_k
                                            t_j = t_j / w_sum.clamp(min=1e-8)

                                        pix2g_id = pix2g_id.to(device)   # [n_frames, HW, K]
                                        pix2g_w = pix2g_w.to(device)     # [n_frames, HW, K]
                                        gathered = t_j[pix2g_id]         # [n_frames, L, K, D]
                                        w = pix2g_w.unsqueeze(-1)        # [n_frames, L, K, 1]
                                        F_vp = (w * gathered).sum(dim=2) / pix2g_w.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # [n_frames, L, D]

                                        F_3 = F_vp.unsqueeze(0).expand(3, -1, -1, -1)  # [3, n_frames, L, D]
                                        h_flat = hidden_states.view(3, n_frames, sequence_length, dim)
                                        attn_output = (lam * (F_3 - h_flat)).reshape(batch_size, sequence_length, dim)
                                        if attn_output.dtype != self.norm2.weight.dtype:
                                            attn_output = attn_output.to(self.norm2.weight.dtype)
                                else:
                                    attn_output = torch.zeros(batch_size, sequence_length, dim, device=device, dtype=hidden_states.dtype)

                        # ---------------------------------------------------------- #
                        # Version B: Gaussian-Provenance Sparse Cross-View Attention  #
                        # ---------------------------------------------------------- #
                        elif getattr(self, "use_gp_attn", False):
                            # kf_attn_output: [3*n_key, sequence_length, dim]
                            # Use the first key view's cached attn output as source
                            kf_src = self.kf_attn_output   # [3*n_key, seq_len, dim]
                            kf_n   = kf_src.shape[0] // 3
                            # Take the first key for source tokens (shape [3, 1, seq_len, dim])
                            kf_first = kf_src.view(3, kf_n, sequence_length, dim)[:, 0]  # [3, seq_len, dim]

                            # Query: current view tokens [3, n_frames, seq_len, dim]
                            q_3d   = norm_hidden_states.view(3, n_frames, sequence_length, dim)

                            # Build flat [3*n_frames, seq_len, dim] for query and source
                            q_flat  = q_3d.reshape(3 * n_frames, sequence_length, dim)
                            # Repeat key-view src across n_frames: [3*n_frames, seq_len, dim]
                            kv_flat = kf_first.unsqueeze(1).expand(-1, n_frames, -1, -1).reshape(3 * n_frames, sequence_length, dim)

                            # Load pre-built candidate maps from cache
                            # gp_idx_map / gp_cand_valid / gp_alpha are dicts keyed by (H, W)
                            hw_key  = (int(sequence_length ** 0.5), int(sequence_length ** 0.5))
                            idx_map    = getattr(self, "gp_idx_map",    {}).get(hw_key, None)
                            cand_valid = getattr(self, "gp_cand_valid", {}).get(hw_key, None)
                            alpha      = getattr(self, "gp_alpha",      {}).get(hw_key, None)

                            if idx_map is not None:
                                device = q_flat.device
                                idx_map    = idx_map.to(device)       # [n_frames, seq_len, L]
                                cand_valid = cand_valid.to(device)    # [n_frames, seq_len, L]
                                if alpha is not None:
                                    alpha = alpha.to(device)          # [n_frames, seq_len, 1]

                                # Expand across the 3 CFG branches
                                idx_3    = idx_map.unsqueeze(0).expand(3, -1, -1, -1).reshape(3 * n_frames, sequence_length, -1)
                                valid_3  = cand_valid.unsqueeze(0).expand(3, -1, -1, -1).reshape(3 * n_frames, sequence_length, -1)
                                alpha_3  = None
                                if alpha is not None:
                                    alpha_3 = alpha.unsqueeze(0).expand(3, -1, -1, -1).reshape(3 * n_frames, sequence_length, 1)

                                inj = sparse_xview_attn(q_flat, kv_flat, kv_flat, idx_3, valid_3, alpha=alpha_3)
                                attn_output = inj.reshape(batch_size, sequence_length, dim).half()
                            else:
                                # Fallback: copy key-view features directly (no sparse attn)
                                attn_output = kv_flat.reshape(batch_size, sequence_length, dim).half()

                        # ---------------------------------------------------------- #
                        # Original Version A: Epipolar dense similarity              #
                        # ---------------------------------------------------------- #
                        elif len(batch_idxs) == 2:
                            attn_1, attn_2 = self.attn_output[:, :, 0], self.attn_output[:, :, 1]
                            idx1 = idx1.view(3, n_frames, sequence_length)
                            idx2 = idx2.view(3, n_frames, sequence_length)
                            attn_output1 = attn_1.gather(dim=2, index=idx1.unsqueeze(-1).repeat(1, 1, 1, dim))
                            attn_output2 = attn_2.gather(dim=2, index=idx2.unsqueeze(-1).repeat(1, 1, 1, dim))
                            d1 = cam_distance_min[0][:,0]
                            d2 = cam_distance_min[0][:,1]
                            w1 = d2 / (d1 + d2)
                            w1 = torch.sigmoid(w1)
                            w1 = w1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(3, 1, sequence_length, dim)
                            attn_output1 = attn_output1.view(3, n_frames, sequence_length, dim)
                            attn_output2 = attn_output2.view(3, n_frames, sequence_length, dim)
                            attn_output = w1 * attn_output1 + (1 - w1) * attn_output2
                            attn_output = attn_output.reshape(
                                batch_size, sequence_length, dim).half()
                        elif idx1 is not None:
                            idx1 = idx1.view(3, n_frames, sequence_length)
                            attn_output = self.attn_output[:,:,0].gather(dim=2, index=idx1.unsqueeze(-1).repeat(1, 1, 1, dim))
                            attn_output = attn_output.reshape(batch_size, sequence_length, dim).half()
                        else:
                            attn_output = self.attn_output.reshape(batch_size, sequence_length, dim).to(hidden_states.dtype)
                    else:
                        attn_output = self.attn_output
                else:
                    attn_output = self.attn_output
            
            
            with latency_logger.timeit(f'{_latency_base}.dge_block.residual_connection') if latency_logger else nullcontext():
                hidden_states = hidden_states.reshape(batch_size, sequence_length, dim)  # 3 * n_frames, seq_len, dim
                hidden_states =  attn_output + hidden_states
                hidden_states = hidden_states.to(self.norm2.weight.dtype)
            
            if self.attn2 is not None:
                with latency_logger.timeit(f'{_latency_base}.dge_block.cross_attention') if latency_logger else nullcontext():
                    norm_hidden_states = (
                        self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                    )

                    # 2. Cross-Attention
                    attn_output = self.attn2(
                        norm_hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=encoder_attention_mask,
                        **cross_attention_kwargs,
                    )
                    hidden_states = attn_output + hidden_states

            # 3. Feed-forward
            with latency_logger.timeit(f'{_latency_base}.dge_block.feed_forward') if latency_logger else nullcontext():
                norm_hidden_states = self.norm3(hidden_states)

                if self.use_ada_layer_norm_zero:
                    norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

                ff_output = self.ff(norm_hidden_states)

                if self.use_ada_layer_norm_zero:
                    ff_output = gate_mlp.unsqueeze(1) * ff_output

                hidden_states = ff_output + hidden_states # [12, 4096, 320] + [12, 4096, 320]

            return hidden_states # [12, 4096, 320]

    return DGEBlock


# ---------------------------------------------------------------------------
# Version B: Gaussian-Provenance Sparse Cross-View Attention helpers
# ---------------------------------------------------------------------------

def gather_hw(ksrc: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gather key-view tokens for each query pixel using candidate indices.

    Args:
        ksrc : [B, HW, C]   – source (key-view) token map
        idx  : [B, HW, L]   – candidate linear indices in [0, HW)
    Returns:
              [B, HW, L, C] – gathered candidate tokens
    """
    B, HW, C = ksrc.shape
    L = idx.shape[-1]
    device = ksrc.device

    # Shift per-batch so every element addresses its own HW block
    offset = torch.arange(B, device=device).view(B, 1, 1) * HW  # [B,1,1]
    idx_g  = (idx + offset).reshape(-1)                          # [B*HW*L]

    flat = ksrc.reshape(B * HW, C)     # [B*HW, C]
    out  = flat[idx_g]                 # [B*HW*L, C]
    return out.view(B, HW, L, C)


def sparse_xview_attn(
    q:          torch.Tensor,            # [B, HW, C]
    ksrc:       torch.Tensor,            # [B, HW, C]  (key-view tokens)
    vsrc:       torch.Tensor,            # [B, HW, C]
    idx_map:    torch.Tensor,            # [B, HW, L]  int32
    cand_valid: torch.Tensor,            # [B, HW, L]  bool
    alpha:      torch.Tensor = None,     # [B, HW, 1]  or None
) -> torch.Tensor:
    """
    Sparse cross-view attention over L gaussian-provenance candidate positions.

    Returns: [B, HW, C] injection tensor (same shape as q).
    """
    k = gather_hw(ksrc, idx_map)   # [B, HW, L, C]
    v = gather_hw(vsrc, idx_map)   # [B, HW, L, C]

    # Cosine-scaled dot-product attention over L candidates
    scale  = math.sqrt(q.shape[-1])
    qn     = torch.nn.functional.normalize(q, dim=-1)    # [B, HW, C]
    kn     = torch.nn.functional.normalize(k, dim=-1)    # [B, HW, L, C]

    # score[b, p, l] = dot(q[b,p], k[b,p,l]) / scale
    score = (qn.unsqueeze(2) * kn).sum(dim=-1) / scale   # [B, HW, L]

    # Mask invalid candidates (out-of-bounds, occluded, no gaussian)
    # Use finfo.min so float16 is safe (float16 cannot represent -1e9)
    score = score.masked_fill(~cand_valid, torch.finfo(score.dtype).min)

    w   = torch.softmax(score, dim=-1)                    # [B, HW, L]
    out = (w.unsqueeze(-1) * v).sum(dim=2)                # [B, HW, C]

    if alpha is not None:
        out = out * alpha   # confidence gating

    return out


def build_gaussian_provenance_cache(
    gaussian,
    cams,
    key_cam_indices: list,
    scales: list,
    K: int = 2,
    M_half: int = 1,
    vis_eps: float = 0.05,
    alpha_tau: float = 0.4,
):
    """
    Pre-compute pixel→gaussian provenance and gaussian→key-view projection
    caches for Version B sparse cross-view attention.

    Args:
        gaussian        : GaussianModel with .get_xyz [N_g, 3]
        cams            : List of all camera objects (sorted, non-key views first)
        key_cam_indices : Indices into `cams` that are the pivotal/key views
        scales          : List of (H, W) tuples for each UNet attention scale
                          e.g. [(64,64), (32,32), (16,16), (8,8)]
        K               : Number of top gaussians per pixel
        M_half          : Half-size of neighbourhood window (M=(2*M_half+1)^2)
        vis_eps         : Depth tolerance for visibility test (in scene units)
        alpha_tau       : Scale factor for alpha gating (alpha = w_top1 / tau)

    Returns dict with keys:
        'pix2g_id'  : {(H,W): [n_views, HW, K]  int32}
        'pix2g_w'   : {(H,W): [n_views, HW, K]  float32}
        'g2uv'      : {ki: {(H,W): [N_g, 2]      int16}}
        'g_vis'     : {ki: {(H,W): [N_g]         bool}}
        'idx_map'   : {vi: {(H,W): [HW, L]       int32}}
        'cand_valid': {vi: {(H,W): [HW, L]       bool}}
        'alpha'     : {vi: {(H,W): [HW, 1]       float16}}
        'closest_key': [n_views]  int  (index into key_cam_indices)
    """
    device = 'cuda'
    n_views    = len(cams)
    n_keys     = len(key_cam_indices)
    key_cams   = [cams[ki] for ki in key_cam_indices]

    # gaussian positions [N_g, 3]
    xyz = gaussian.get_xyz.detach()   # [N_g, 3]
    N_g = xyz.shape[0]

    # Neighbourhood offsets [M, 2] (dy, dx)
    M_win = 2 * M_half + 1
    offs  = [(dy, dx)
             for dy in range(-M_half, M_half + 1)
             for dx in range(-M_half, M_half + 1)]
    offsets = torch.tensor(offs, device=device, dtype=torch.int32)  # [M, 2]
    M_total  = offsets.shape[0]
    L = K * M_total

    # Closest key view for each non-key view (by camera distance)
    cam_centers     = torch.stack([c.camera_center for c in cams], dim=0).to(device)     # [n_views, 3]
    key_cam_centers = torch.stack([c.camera_center for c in key_cams], dim=0).to(device) # [n_keys,  3]
    dists           = torch.cdist(cam_centers, key_cam_centers)   # [n_views, n_keys]
    closest_key_idx = dists.argmin(dim=-1).cpu().tolist()          # [n_views]

    # Output containers
    pix2g_id_cache   = {}
    pix2g_w_cache    = {}
    g2uv_cache       = {ki: {} for ki in range(n_keys)}
    g_vis_cache      = {ki: {} for ki in range(n_keys)}
    idx_map_cache    = {}
    cand_valid_cache = {}
    alpha_cache      = {}
    g2uv_all_cache   = {}

    for (H, W) in scales:
        HW = H * W

        # ------------------------------------------------------------------
        # Step 1: For each view, render at (H, W) and extract top-K gaussian
        #         provenance via alpha-compositing weights.
        #         We approximate provenance from the depth map and gaussian
        #         opacity/alpha rendered at the query view.
        # ------------------------------------------------------------------
        # We use the rasterizer's rendered alpha (depth proxy) per gaussian.
        # Since the CUDA rasterizer doesn't expose per-pixel gaussian ids by
        # default, we use a lightweight soft-assignment: project all gaussians,
        # find the K nearest in projected-pixel space weighted by opacity.

        # Project all gaussians into every view at this scale
        # Result: [N_g, 2] pixel coords + [N_g] depth
        all_proj = []   # one entry per view: (uv [N_g,2], z [N_g])
        all_depth_map = []  # rendered depth [HW] per view

        for vi, cam in enumerate(cams):
            # Project gaussian centres to this camera
            # world_view_transform: 4x4  (col-major, world→view)
            W_mat = cam.world_view_transform.to(device)   # [4, 4]
            ones  = torch.ones(N_g, 1, device=device)
            xyz_h = torch.cat([xyz, ones], dim=1)         # [N_g, 4]
            xyz_c = xyz_h @ W_mat                         # [N_g, 4]  (view space, col-major)

            z_c   = xyz_c[:, 2]                           # [N_g]

            # Project to NDC then to pixel
            full_proj = cam.full_proj_transform.to(device)  # [4, 4]
            p         = xyz_h @ full_proj                   # [N_g, 4]
            p_ndc     = p[:, :2] / (p[:, 3:4].clamp(min=1e-6))  # [N_g, 2]

            # NDC [-1,1] → pixel [0, W) and [0, H)
            u = ((p_ndc[:, 0] + 1.0) * 0.5 * W).long()   # [N_g]
            v = ((p_ndc[:, 1] + 1.0) * 0.5 * H).long()   # [N_g]

            all_proj.append((u, v, z_c))

        # g2uv for ALL views (for 3d_anchor: pivot view projection)
        u_all = torch.stack([all_proj[vi][0].clamp(0, W - 1) for vi in range(n_views)], dim=0)
        v_all = torch.stack([all_proj[vi][1].clamp(0, H - 1) for vi in range(n_views)], dim=0)
        g2uv_all_cache[(H, W)] = torch.stack([u_all, v_all], dim=-1).to(torch.int16)  # [n_views, N_g, 2]

        # ------------------------------------------------------------------
        # Step 2: For each key view, compute gaussian projection + visibility
        # ------------------------------------------------------------------
        for ki, (kv_idx, key_cam) in enumerate(zip(key_cam_indices, key_cams)):
            u_k, v_k, z_k = all_proj[kv_idx]

            # Inbounds mask
            inbounds = (u_k >= 0) & (u_k < W) & (v_k >= 0) & (v_k < H) & (z_k > 0)

            # Build a z-buffer for the key view to test visibility
            # Use the rendered depth map at this scale (approximation via
            # nearest-gaussian depth rendering)
            zmap = torch.full((HW,), float('inf'), device=device, dtype=z_k.dtype)

            in_idx = inbounds.nonzero(as_tuple=False).squeeze(1)
            if in_idx.numel() > 0:
                pix_lin = (v_k[in_idx] * W + u_k[in_idx]).clamp(0, HW - 1)
                # scatter min-z into zmap
                zmap.scatter_reduce_(0, pix_lin, z_k[in_idx], reduce='amin', include_self=True)

            # Visibility: gaussian is visible if its z ≈ z-buffer at its projected pixel
            g_vis_map = torch.zeros(N_g, dtype=torch.bool, device=device)
            if in_idx.numel() > 0:
                pix_z_ref = zmap[pix_lin]
                vis_mask  = (z_k[in_idx] - pix_z_ref).abs() < vis_eps * pix_z_ref.clamp(min=1.0)
                g_vis_map[in_idx[vis_mask]] = True

            # Clamp pixel coords
            u_k_cl = u_k.clamp(0, W - 1)
            v_k_cl = v_k.clamp(0, H - 1)
            g2uv   = torch.stack([u_k_cl, v_k_cl], dim=-1).to(torch.int16)  # [N_g, 2]

            g2uv_cache[ki][(H, W)]  = g2uv
            g_vis_cache[ki][(H, W)] = g_vis_map

        # ------------------------------------------------------------------
        # Step 3: For each non-key view, build pix2g (top-K by proximity)
        #         and then build idx_map / cand_valid / alpha
        # ------------------------------------------------------------------
        # We need pix2g_id per view.  Since we have no per-pixel gaussian id
        # from the rasterizer, we use a proximity-based assignment:
        # for each pixel (u,v) find the K gaussians whose projection is
        # closest (in pixel distance) AND inbounds in this view.

        # Pre-build projected pixel coords for all views
        all_uv_inbounds = []
        for vi in range(n_views):
            u_v, v_v, z_v = all_proj[vi]
            ib = (u_v >= 0) & (u_v < W) & (v_v >= 0) & (v_v < H) & (z_v > 0)
            all_uv_inbounds.append((u_v, v_v, ib))

        pix2g_id_scale = []
        pix2g_w_scale  = []

        for vi in range(n_views):
            u_v, v_v, ib = all_uv_inbounds[vi]

            # Build query pixel grid
            pu = torch.arange(W, device=device).view(1, W).expand(H, W).reshape(HW).float()
            pv = torch.arange(H, device=device).view(H, 1).expand(H, W).reshape(HW).float()

            # For each valid gaussian, compute distance to each pixel
            in_idx = ib.nonzero(as_tuple=False).squeeze(1)

            if in_idx.numel() >= K:
                gu = u_v[in_idx].float()   # [n_valid]
                gv = v_v[in_idx].float()   # [n_valid]

                # Chunked distance to avoid OOM: [HW, n_valid]
                chunk = 256
                topk_ids  = []
                topk_dists = []
                for start in range(0, HW, chunk):
                    end = min(start + chunk, HW)
                    d2 = ((pu[start:end, None] - gu[None, :]) ** 2 +
                          (pv[start:end, None] - gv[None, :]) ** 2)  # [chunk, n_valid]
                    tk = min(K, d2.shape[1])
                    dist_k, idx_k = d2.topk(tk, dim=-1, largest=False)  # [chunk, K]
                    topk_ids.append(in_idx[idx_k])
                    topk_dists.append(dist_k)

                topk_ids   = torch.cat(topk_ids,   dim=0)   # [HW, K]
                topk_dists = torch.cat(topk_dists, dim=0)   # [HW, K]

                # Pad to K if fewer inbounds gaussians
                pad_needed = K - topk_ids.shape[1]
                if pad_needed > 0:
                    topk_ids   = torch.cat([topk_ids,   topk_ids[:, :1].expand(-1, pad_needed)], dim=1)
                    topk_dists = torch.cat([topk_dists, topk_dists[:, :1].expand(-1, pad_needed) + 1e6], dim=1)

                # Soft weights: inverse distance (normalised)
                w = 1.0 / (topk_dists + 1.0)
                w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            else:
                # Fallback: repeat gaussian 0
                topk_ids   = torch.zeros(HW, K, device=device, dtype=torch.long)
                w          = torch.full((HW, K), 1.0 / K, device=device)

            pix2g_id_scale.append(topk_ids.int())   # [HW, K]
            pix2g_w_scale.append(w.float())          # [HW, K]

        pix2g_id_cache[(H, W)] = torch.stack(pix2g_id_scale, dim=0)  # [n_views, HW, K]
        pix2g_w_cache[(H, W)]  = torch.stack(pix2g_w_scale,  dim=0)  # [n_views, HW, K]

        # ------------------------------------------------------------------
        # Step 4: Build idx_map / cand_valid / alpha per non-key view
        # ------------------------------------------------------------------
        for vi in range(n_views):
            ki_closest = closest_key_idx[vi]
            g2uv   = g2uv_cache[ki_closest][(H, W)].long()   # [N_g, 2]
            g_vis  = g_vis_cache[ki_closest][(H, W)]          # [N_g] bool

            p2g_id = pix2g_id_cache[(H, W)][vi].long()        # [HW, K]
            p2g_w  = pix2g_w_cache[(H, W)][vi]                # [HW, K]

            # candidate uv from gaussian projections [HW, K, 2]
            cand_uv = g2uv[p2g_id]                            # [HW, K, 2]

            # Neighbourhood expansion [HW, K, M, 2]
            cand_uv_exp = cand_uv.unsqueeze(2) + offsets[None, None, :, :]  # [HW,K,M,2]

            # Clamp and compute linear index
            u_cand = cand_uv_exp[..., 0].clamp(0, W - 1)      # [HW, K, M]
            v_cand = cand_uv_exp[..., 1].clamp(0, H - 1)      # [HW, K, M]
            lin    = (v_cand * W + u_cand).int()               # [HW, K, M]

            # Validity: gaussian visible AND original coord in-bounds AND z>0
            u_orig  = cand_uv[..., 0]                          # [HW, K]
            v_orig  = cand_uv[..., 1]                          # [HW, K]
            g_inbounds = ((u_orig >= 0) & (u_orig < W) &
                          (v_orig >= 0) & (v_orig < H))        # [HW, K]
            g_visible  = g_vis[p2g_id]                         # [HW, K]
            base_valid = g_inbounds & g_visible                # [HW, K]

            # Expand validity to [HW, K, M] then flatten to [HW, L]
            valid_exp = base_valid.unsqueeze(2).expand(-1, -1, M_total)  # [HW,K,M]

            idx_flat   = lin.reshape(HW, L)
            valid_flat = valid_exp.reshape(HW, L)

            # Fallback for pixels with no valid candidates
            no_cand = ~valid_flat.any(dim=-1)   # [HW]
            idx_flat[no_cand, :]   = 0
            valid_flat[no_cand, :] = False

            # Alpha: based on top-1 gaussian weight / tau
            w_top1 = p2g_w[:, 0]                               # [HW]
            alpha  = (w_top1 / alpha_tau).clamp(0.0, 1.0).unsqueeze(-1).to(torch.float16)  # [HW,1]
            # Pixels with no candidate get alpha=0
            alpha[no_cand, :] = 0.0

            if vi not in idx_map_cache:
                idx_map_cache[vi]    = {}
                cand_valid_cache[vi] = {}
                alpha_cache[vi]      = {}

            idx_map_cache[vi][(H, W)]    = idx_flat.cpu()
            cand_valid_cache[vi][(H, W)] = valid_flat.cpu()
            alpha_cache[vi][(H, W)]      = alpha.cpu()

    return {
        'pix2g_id':    pix2g_id_cache,
        'pix2g_w':     pix2g_w_cache,
        'g2uv':        g2uv_cache,
        'g_vis':       g_vis_cache,
        'g2uv_all':    g2uv_all_cache,
        'idx_map':     idx_map_cache,
        'cand_valid':  cand_valid_cache,
        'alpha':       alpha_cache,
        'closest_key': closest_key_idx,
    }


def register_gp_cache(diffusion_model, gp_cache, batch_start: int, batch_end: int):
    """
    Register stacked gaussian-provenance idx_map/cand_valid/alpha for views
    [batch_start, batch_end) onto every BasicTransformerBlock.

    The tensors are stacked along dim-0 so that DGEBlock sees
      idx_map    : [n_frames, HW, L]
      cand_valid : [n_frames, HW, L]
      alpha      : [n_frames, HW, 1]
    keyed by (H, W) as before.
    """
    # Collect scale keys from first view that has cache
    first_vi = batch_start
    scale_keys = list(gp_cache['idx_map'].get(first_vi, {}).keys())

    stacked_idx   = {}
    stacked_valid = {}
    stacked_alpha = {}

    for hw in scale_keys:
        idx_list   = []
        valid_list = []
        alpha_list = []
        for vi in range(batch_start, batch_end):
            idx_map_vi    = gp_cache['idx_map'].get(vi, {}).get(hw, None)
            cand_valid_vi = gp_cache['cand_valid'].get(vi, {}).get(hw, None)
            alpha_vi      = gp_cache['alpha'].get(vi, {}).get(hw, None)
            if idx_map_vi is not None:
                idx_list.append(idx_map_vi)
                valid_list.append(cand_valid_vi)
                if alpha_vi is not None:
                    alpha_list.append(alpha_vi)
        if idx_list:
            stacked_idx[hw]   = torch.stack(idx_list,   dim=0)   # [n_frames, HW, L]
            stacked_valid[hw] = torch.stack(valid_list, dim=0)   # [n_frames, HW, L]
            if alpha_list:
                stacked_alpha[hw] = torch.stack(alpha_list, dim=0)  # [n_frames, HW, 1]

    for _, module in diffusion_model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "gp_idx_map",    stacked_idx)
            setattr(module, "gp_cand_valid", stacked_valid)
            setattr(module, "gp_alpha",      stacked_alpha)
            setattr(module, "use_gp_attn",   True)


def unregister_gp_cache(diffusion_model):
    """Clear gaussian-provenance cache attributes from all transformer blocks."""
    for _, module in diffusion_model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "use_gp_attn",   False)
            setattr(module, "gp_idx_map",    {})
            setattr(module, "gp_cand_valid", {})
            setattr(module, "gp_alpha",      {})


def register_anchor_3d_cache(
    diffusion_model,
    anchor_3d_cache: dict,
    batch_view_indices: list,
    injection_lambda: float = 0.5,
    pivot_view_index: Optional[int] = None,
    injection_3d_anchor_style: str = "blend",
):
    """
    Register 3D-anchor feature injection cache for views in this batch.
    anchor_3d_cache must have: pix2g_id, pix2g_w (per-scale [n_views, HW, K]),
    g2uv, g_vis (per key view, per scale). batch_view_indices: global view indices for this batch.
    """
    pix2g_id_cache = anchor_3d_cache["pix2g_id"]
    pix2g_w_cache = anchor_3d_cache["pix2g_w"]
    g2uv_cache = anchor_3d_cache["g2uv"]
    g_vis_cache = anchor_3d_cache["g_vis"]
    g2uv_all = anchor_3d_cache.get("g2uv_all", {})

    stacked_pix2g_id = {}
    stacked_pix2g_w = {}
    for hw in list(pix2g_id_cache.keys()):
        idx = batch_view_indices
        stacked_pix2g_id[hw] = pix2g_id_cache[hw][idx]
        stacked_pix2g_w[hw] = pix2g_w_cache[hw][idx]

    g2uv_list = {hw: [g2uv_cache[ki][hw] for ki in range(len(g2uv_cache))] for hw in pix2g_id_cache.keys()}
    g_vis_list = {hw: [g_vis_cache[ki][hw] for ki in range(len(g_vis_cache))] for hw in pix2g_id_cache.keys()}

    for _, module in diffusion_model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "anchor_3d_pix2g_id", stacked_pix2g_id)
            setattr(module, "anchor_3d_pix2g_w", stacked_pix2g_w)
            setattr(module, "anchor_3d_g2uv", g2uv_list)
            setattr(module, "anchor_3d_g_vis", g_vis_list)
            setattr(module, "anchor_3d_g2uv_all", g2uv_all)
            setattr(module, "anchor_3d_pivot_view_index", pivot_view_index)
            setattr(module, "use_3d_anchor_attn", True)
            setattr(module, "injection_lambda", injection_lambda)
            setattr(module, "injection_3d_anchor_style", injection_3d_anchor_style)


def unregister_anchor_3d_cache(diffusion_model):
    """Clear 3D-anchor cache and flag from all transformer blocks."""
    for _, module in diffusion_model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "use_3d_anchor_attn", False)
            setattr(module, "anchor_3d_pix2g_id", {})
            setattr(module, "anchor_3d_pix2g_w", {})
            setattr(module, "anchor_3d_g2uv", {})
            setattr(module, "anchor_3d_g_vis", {})
            setattr(module, "anchor_3d_g2uv_all", {})
            setattr(module, "anchor_3d_pivot_view_index", None)
            setattr(module, "injection_lambda", 0.5)
            setattr(module, "injection_3d_anchor_style", "blend")
