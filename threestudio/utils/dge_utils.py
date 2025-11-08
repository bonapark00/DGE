from typing import Type
import torch
import os

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


def compute_epipolar_constrains(cam1, cam2, current_H=64, current_W=64):
    """
    두 카메라 간의 epipolar constraint를 계산
    epipolar geometry에 맞지 않는 픽셀 쌍을 마스크로 반환
    """
    n_frames = 1
    sequence_length = current_W * current_H
    
    # 두 카메라 간의 fundamental matrix 계산
    fundamental_matrix_1 = []
    fundamental_matrix_1.append(get_fundamental_matrix_with_H(cam1, cam2, current_H, current_W))
    fundamental_matrix_1 = torch.stack(fundamental_matrix_1, dim=0)

    # 이미지의 모든 픽셀 위치 생성
    x = torch.arange(current_W)
    y = torch.arange(current_H)
    x, y = torch.meshgrid(x, y, indexing='xy')
    x = x.reshape(-1)
    y = y.reshape(-1)
    
    # Homogeneous 좌표로 변환 [x, y, 1]
    heto_cam2 = torch.stack([x, y, torch.ones(size=(len(x),))], dim=1).view(-1, 3).cuda()
    heto_cam1 = torch.stack([x, y, torch.ones(size=(len(x),))], dim=1).view(-1, 3).cuda()
    
    # cam2의 각 픽셀에 대해 cam1에서의 epipolar line 계산
    line1 = (heto_cam2.unsqueeze(0).repeat(n_frames, 1, 1) @ fundamental_matrix_1.cuda()).view(-1, 3)
    
    # cam1의 각 픽셀에서 epipolar line까지의 거리 계산
    distance1 = point_to_line_dist(heto_cam1, line1)

    # 거리가 1보다 큰 경우 = epipolar constraint를 위반하는 픽셀 쌍
    idx1_epipolar = distance1 > 1  # shape: (sequence_length, sequence_length)

    return idx1_epipolar

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def register_epipolar_constrains(diffusion_model, epipolar_constrains):
    """UNet의 모든 BasicTransformerBlock에 epipolar constraint 등록"""
    for _, module in diffusion_model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "epipolar_constrains", epipolar_constrains)

def register_cams(diffusion_model, cams, pivot_this_batch, key_cams):
    """UNet의 모든 BasicTransformerBlock에 카메라 정보 등록"""
    for _, module in diffusion_model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "cams", cams)
            setattr(module, "pivot_this_batch", pivot_this_batch)
            setattr(module, "key_cams", key_cams)

def register_pivotal(diffusion_model, is_pivotal):
    """현재 pass가 pivotal (key camera) pass인지 설정"""
    for _, module in diffusion_model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "pivotal_pass", is_pivotal)
            
def register_batch_idx(diffusion_model, batch_idx):
    """현재 batch index 등록"""
    for _, module in diffusion_model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "batch_idx", batch_idx)


def register_t(diffusion_model, t):

    for _, module in diffusion_model.named_modules():
    # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "t", t)


def register_normal_attention(model):
    """표준 attention을 사용하도록 설정 (multi-view 처리 없음)"""
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out
        def forward(x, encoder_hidden_states=None, attention_mask=None):
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

def register_extended_attention(model):
    """
    Multi-view 간 attention을 수행하는 extended attention 등록
    text, image, uncond 각각에 대해 모든 view 간 attention 계산
    """
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
            n_frames = batch_size // 3  # text, image, uncond 각각 n_frames개
            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            q = self.to_q(x)
            k = self.to_k(encoder_hidden_states)
            v = self.to_v(encoder_hidden_states)
            
            # 각 view의 key/value를 모든 view에 대해 반복
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

            # Multi-head attention을 위해 reshape
            q_text = q_text.view(n_frames, h, sequence_length, dim // h)
            k_text = k_text.view(n_frames, h, sequence_length * n_frames, dim // h)
            v_text = v_text.view(n_frames, h, sequence_length * n_frames, dim // h)

            q_image = q_image.view(n_frames, h, sequence_length, dim // h)
            k_image = k_image.view(n_frames, h, sequence_length * n_frames, dim // h)
            v_image = v_image.view(n_frames, h, sequence_length * n_frames, dim // h)

            q_uncond = q_uncond.view(n_frames, h, sequence_length, dim // h)
            k_uncond = k_uncond.view(n_frames, h, sequence_length * n_frames, dim // h)
            v_uncond = v_uncond.view(n_frames, h, sequence_length * n_frames, dim // h)

            # 각 head별로 attention 계산
            for j in range(h):
                sim_text = torch.bmm(q_text[:, j], k_text[:, j].transpose(-1, -2)) * self.scale
                sim_image = torch.bmm(q_image[:, j], k_image[:, j].transpose(-1, -2)) * self.scale
                sim_uncond = torch.bmm(q_uncond[:, j], k_uncond[:, j].transpose(-1, -2)) * self.scale
                
                out_text.append(torch.bmm(sim_text.softmax(dim=-1), v_text[:, j]))
                out_image.append(torch.bmm(sim_image.softmax(dim=-1), v_image[:, j]))
                out_uncond.append(torch.bmm(sim_uncond.softmax(dim=-1), v_uncond[:, j]))

            # 결과를 concat하여 원래 shape로 복원
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
    """카메라 중심점 간의 거리 계산 (가장 가까운 key camera 찾기 위해 사용)"""
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
            
            batch_size, sequence_length, dim = hidden_states.shape
            n_frames = batch_size // 3  # text, image, uncond 각각 n_frames개
            hidden_states = hidden_states.view(3, n_frames, sequence_length, dim)

            # Layer normalization
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)
        
            norm_hidden_states = norm_hidden_states.view(3, n_frames, sequence_length, dim)
            
            # Pivotal pass (key camera)인 경우 hidden states 저장
            if self.pivotal_pass:
                self.pivot_hidden_states = norm_hidden_states
                
            if not self.use_normal_attn:
                if self.pivotal_pass:
                    self.pivot_hidden_states = norm_hidden_states
                else:
                    # Non-pivotal pass: 가장 가까운 key camera 찾기
                    batch_idxs = [self.batch_idx]
                    if self.batch_idx > 0:
                        batch_idxs.append(self.batch_idx - 1)  # 이전 batch도 고려
                    idx1 = []
                    idx2 = []
                    
                    # 카메라 간 거리 계산하여 가장 가까운 key camera 찾기
                    cam_distance = compute_camera_distance(self.cams, self.key_cams)
                    cam_distance_min = cam_distance.sort(dim=-1)
                    closest_cam = cam_distance_min[1][:,:len(batch_idxs)]
                    closest_cam_pivot_hidden_states = self.pivot_hidden_states[1][closest_cam]
                    
                    # 현재 view와 key camera view 간의 similarity 계산
                    sim = torch.einsum('bld,bcsd->bcls', norm_hidden_states[1] / norm_hidden_states[1].norm(dim=-1, keepdim=True), closest_cam_pivot_hidden_states / closest_cam_pivot_hidden_states.norm(dim=-1, keepdim=True)).squeeze()
                        
                    # Epipolar constraint 적용 전 초기 similarity 기반 인덱스 찾기
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

                    # Epipolar constraint 적용하여 geometry에 맞지 않는 similarity 제거
                    if len(batch_idxs) == 2:
                        idx1 = []
                        idx2 = []
                        pivot_this_batch = self.pivot_this_batch
                        
                        # 가장 가까운 key camera에 대한 epipolar constraint 가져오기
                        idx1_epipolar, idx2_epipolar = self.epipolar_constrains[sequence_length].gather(dim=1, index=closest_cam[:, :, None, None].expand(-1, -1, self.epipolar_constrains[sequence_length].shape[2], self.epipolar_constrains[sequence_length].shape[3])).cuda().chunk(2, dim=1)
                        idx1_epipolar = idx1_epipolar.reshape(n_frames, sequence_length, sequence_length)
    
                        # Pivot camera 자체는 constraint 적용 안 함
                        idx1_epipolar[pivot_this_batch, ...] = False
                        idx2_epipolar = idx2_epipolar.reshape(n_frames, sequence_length, sequence_length)

                        idx1_epipolar = idx1_epipolar.reshape(n_frames * sequence_length, sequence_length)
                        idx2_epipolar = idx2_epipolar.reshape(n_frames * sequence_length, sequence_length)
                        idx2_sum = idx2_epipolar.sum(dim=-1)
                        idx1_sum = idx1_epipolar.sum(dim=-1)

                        # 모든 픽셀이 constraint에 걸리는 경우는 제외
                        idx1_epipolar[idx1_sum == sequence_length, :] = False
                        idx2_epipolar[idx2_sum == sequence_length, :] = False
                        
                        # Epipolar constraint를 위반하는 similarity를 0으로 설정
                        sim1[idx1_epipolar] = 0
                        sim2[idx2_epipolar] = 0

                        # Constraint 적용 후 가장 유사한 픽셀 인덱스 찾기
                        sim1_max = sim1.max(dim=-1)
                        sim2_max = sim2.max(dim=-1)
                        idx1.append(sim1_max[1])
                        idx2.append(sim2_max[1])
                    else:
                        idx1 = []
                        pivot_this_batch = self.pivot_this_batch

                        # Epipolar constraint 마스크 가져오기
                        idx1_epipolar = self.epipolar_constrains[sequence_length].gather(dim=1, index=closest_cam[:, :, None, None].expand(-1, -1, self.epipolar_constrains[sequence_length].shape[2], self.epipolar_constrains[sequence_length].shape[3])).cuda()

                        idx1_epipolar = idx1_epipolar.view(n_frames, -1, sequence_length)
                        idx1_epipolar[pivot_this_batch, ...] = False

                        idx1_epipolar = idx1_epipolar.view(n_frames * sequence_length, sequence_length)
                        idx1_sum = idx1_epipolar.sum(dim=-1)
                        idx1_epipolar[idx1_sum == sequence_length, :] = False
                        
                        # Geometry에 맞지 않는 similarity를 0으로 설정
                        sim[idx1_epipolar] = 0
                        sim_max = sim.max(dim=-1)
                        idx1.append(sim_max[1])
                            
                    idx1 = torch.stack(idx1 * 3, dim=0) # 3, n_frames * seq_len
                    idx1 = idx1.squeeze(1)


                    if len(batch_idxs) == 2:
                        idx2 = torch.stack(idx2 * 3, dim=0) # 3, n_frames * seq_len
                        idx2 = idx2.squeeze(1)

                            
            
            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            
            # Timestep < 100: use_normal_attn = True
            #   - 일반 attention 사용
            #   - Multi-view 처리 없음
            #   - 초기 denoising 단계에서 사용
            # Timestep >= 100: use_normal_attn = False
            #   - Extended attention 사용
            #   - Multi-view 처리 및 epipolar constraint 적용
            #   - 후반부에서 multi-view 일관성 확보
            
            if self.use_normal_attn:
                # 일반 attention 사용 (multi-view 처리 없음)
                self.attn_output = self.attn1.normal_attn(
                        norm_hidden_states.view(batch_size, sequence_length, dim),
                        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                        **cross_attention_kwargs,
                    )         
            else:
                # Extended attention 사용 (multi-view 처리)
                if self.pivotal_pass:
                    # Key camera의 attention 계산 및 저장
                    self.attn_output = self.attn1(
                            norm_hidden_states.view(batch_size, sequence_length, dim),
                            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                            **cross_attention_kwargs,
                        )
                    self.kf_attn_output = self.attn_output  # 나중에 다른 view에서 사용하기 위해 저장

                else:
                    # Non-pivotal pass: 가장 가까운 key camera의 attention을 가져옴
                    batch_kf_size, _, _ = self.kf_attn_output.shape
                    self.attn_output = self.kf_attn_output.view(3, batch_kf_size // 3, sequence_length, dim)[:, closest_cam]

            if self.use_ada_layer_norm_zero:
                self.n = gate_msa.unsqueeze(1) * self.attn_output

            # Epipolar constraint를 만족하는 픽셀의 attention 값만 gather
            if not self.use_normal_attn:
                if not self.pivotal_pass:
                    if len(batch_idxs) == 2:
                        # 두 개의 가까운 key camera가 있는 경우
                        attn_1, attn_2 = self.attn_output[:, :, 0], self.attn_output[:, :, 1]
                        idx1 = idx1.view(3, n_frames, sequence_length)
                        idx2 = idx2.view(3, n_frames, sequence_length)
                        
                        # Epipolar constraint를 만족하는 픽셀의 attention 값만 선택
                        attn_output1 = attn_1.gather(dim=2, index=idx1.unsqueeze(-1).repeat(1, 1, 1, dim))
                        attn_output2 = attn_2.gather(dim=2, index=idx2.unsqueeze(-1).repeat(1, 1, 1, dim))
                        
                        # 카메라 거리에 따라 가중 평균
                        d1 = cam_distance_min[0][:,0]
                        d2 = cam_distance_min[0][:,1]
                        w1 = d2 / (d1 + d2)
                        w1 = torch.sigmoid(w1)
                        w1 = w1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(3, 1, sequence_length, dim)
                        
                        attn_output1 = attn_output1.view(3, n_frames, sequence_length, dim)
                        attn_output2 = attn_output2.view(3, n_frames, sequence_length, dim)
                        attn_output = w1 * attn_output1 + (1 - w1) * attn_output2
                        attn_output = attn_output.reshape(batch_size, sequence_length, dim).half()
                    else:
                        # 하나의 가까운 key camera만 있는 경우
                        idx1 = idx1.view(3, n_frames, sequence_length)
                        attn_output = self.attn_output[:,:,0].gather(dim=2, index=idx1.unsqueeze(-1).repeat(1, 1, 1, dim))
                        attn_output = attn_output.reshape(batch_size, sequence_length, dim).half()                       
                else:
                    attn_output = self.attn_output
            else:
                attn_output = self.attn_output
            
            # Residual connection: attention output을 원본 hidden states에 더함
            hidden_states = hidden_states.reshape(batch_size, sequence_length, dim)
            hidden_states = attn_output + hidden_states
            hidden_states = hidden_states.to(self.norm2.weight.dtype)
            # 2. Cross-Attention (텍스트 조건과의 attention)
            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )

                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states  # Residual connection

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            hidden_states = ff_output + hidden_states  # Residual connection

            return hidden_states

    return DGEBlock

