from typing import Type
import torch
import os
from contextlib import nullcontext
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
            
            # Initialize latency logger if available
            latency_logger = getattr(self, 'latency_logger', None)
            
            with latency_logger.timeit('edit_all_view.guidance_batch.edit_latents.diffusion_loop.batch_processing.unet_forward.dge_block.init') if latency_logger else nullcontext():
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
                    with latency_logger.timeit('edit_all_view.guidance_batch.edit_latents.diffusion_loop.batch_processing.unet_forward.dge_block.pivotal_pass') if latency_logger else nullcontext():
                        self.pivot_hidden_states = norm_hidden_states

                ## key view가 아닌 경우에는 가장 가까운 카메라 찾기
                else:
                    with latency_logger.timeit('edit_all_view.guidance_batch.edit_latents.diffusion_loop.batch_processing.unet_forward.dge_block.camera_distance') if latency_logger else nullcontext():
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
                    
                    with latency_logger.timeit('edit_all_view.guidance_batch.edit_latents.diffusion_loop.batch_processing.unet_forward.dge_block.spatio_temporal_attention') if latency_logger else nullcontext():
                        sim = torch.einsum('bld,bcsd->bcls', norm_hidden_states[1] / norm_hidden_states[1].norm(dim=-1, keepdim=True), closest_cam_pivot_hidden_states / closest_cam_pivot_hidden_states.norm(dim=-1, keepdim=True)).squeeze()
                        
                    if len(batch_idxs) == 2:
                        sim1, sim2 = sim.chunk(2, dim=1)
                        sim1 = sim1.view(-1, sequence_length)
                        sim2 = sim2.view(-1, sequence_length)
                        sim1_max = sim1.max(dim=-1)
                        sim2_max = sim2.max(dim=-1)
                        idx1.append(sim1_max[1])
                        idx2.append(sim2_max[1])

                    else:
                        sim = sim.view(-1, sequence_length)
                        sim_max = sim.max(dim=-1)
                        idx1.append(sim_max[1])

                    with latency_logger.timeit('edit_all_view.guidance_batch.edit_latents.diffusion_loop.batch_processing.unet_forward.dge_block.epipolar_constraints') if latency_logger else nullcontext():
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
                            
                    idx1 = torch.stack(idx1 * 3, dim=0) # 3, n_frames * seq_len
                    idx1 = idx1.squeeze(1)


                    if len(batch_idxs) == 2:
                        idx2 = torch.stack(idx2 * 3, dim=0) # 3, n_frames * seq_len
                        idx2 = idx2.squeeze(1)

                            
            
            # 1. Self-Attention
            with latency_logger.timeit('edit_all_view.guidance_batch.edit_latents.diffusion_loop.batch_processing.unet_forward.dge_block.self_attention') if latency_logger else nullcontext():
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
                        # 3, n_frames * seq_len, dim - > 3 * n_frames, seq_len, dim
                        self.kf_attn_output = self.attn_output

                    else:
                        batch_kf_size, _, _ = self.kf_attn_output.shape
                        self.attn_output = self.kf_attn_output.view(3, batch_kf_size // 3, sequence_length, dim)[:,
                                        closest_cam]

            if self.use_ada_layer_norm_zero:
                self.n = gate_msa.unsqueeze(1) * self.attn_output

            # gather values from attn_output, using idx as indices, and get a tensor of shape 3, n_frames, seq_len, dim
            with latency_logger.timeit('edit_all_view.guidance_batch.edit_latents.diffusion_loop.batch_processing.unet_forward.dge_block.feature_injection') if latency_logger else nullcontext():
                if not self.use_normal_attn:
                    if not self.pivotal_pass:
                        if len(batch_idxs) == 2:
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
                        else:
                            idx1 = idx1.view(3, n_frames, sequence_length)
                            attn_output = self.attn_output[:,:,0].gather(dim=2, index=idx1.unsqueeze(-1).repeat(1, 1, 1, dim))
                            attn_output = attn_output.reshape(batch_size, sequence_length, dim).half()                       
                    else:
                        attn_output = self.attn_output
                else:
                    attn_output = self.attn_output
            
            
            with latency_logger.timeit('edit_all_view.guidance_batch.edit_latents.diffusion_loop.batch_processing.unet_forward.dge_block.residual_connection') if latency_logger else nullcontext():
                hidden_states = hidden_states.reshape(batch_size, sequence_length, dim)  # 3 * n_frames, seq_len, dim
                hidden_states = attn_output + hidden_states
                hidden_states = hidden_states.to(self.norm2.weight.dtype)
            
            if self.attn2 is not None:
                with latency_logger.timeit('edit_all_view.guidance_batch.edit_latents.diffusion_loop.batch_processing.unet_forward.dge_block.cross_attention') if latency_logger else nullcontext():
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
            with latency_logger.timeit('edit_all_view.guidance_batch.edit_latents.diffusion_loop.batch_processing.unet_forward.dge_block.feed_forward') if latency_logger else nullcontext():
                norm_hidden_states = self.norm3(hidden_states)

                if self.use_ada_layer_norm_zero:
                    norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

                ff_output = self.ff(norm_hidden_states)

                if self.use_ada_layer_norm_zero:
                    ff_output = gate_mlp.unsqueeze(1) * ff_output

                hidden_states = ff_output + hidden_states

            return hidden_states

    return DGEBlock

