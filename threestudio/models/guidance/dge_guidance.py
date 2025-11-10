from dataclasses import dataclass

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


from threestudio.utils.dge_utils import register_pivotal, register_batch_idx, register_cams, register_epipolar_constrains, register_extended_attention, register_normal_attention, register_extended_attention, make_dge_block, isinstance_str, compute_epipolar_constrains, register_normal_attn_flag

@threestudio.register("dge-guidance")
class DGEGuidance(BaseObject):
    """
    DGE (Diffusion Guidance for Editing) Guidance 클래스
    InstructPix2Pix 모델을 사용하여 이미지 편집을 수행하는 guidance 모듈
    """
    @dataclass
    class Config(BaseObject.Config):
        """설정 클래스: DGE Guidance의 하이퍼파라미터 정의"""
        cache_dir: Optional[str] = None  # 모델 캐시 디렉토리 경로
        ddim_scheduler_name_or_path: str = "CompVis/stable-diffusion-v1-4"  # DDIM 스케줄러 모델 경로
        ip2p_name_or_path: str = "timbrooks/instruct-pix2pix"  # InstructPix2Pix 모델 경로

        enable_memory_efficient_attention: bool = False  # 메모리 효율적인 attention 활성화 여부
        enable_sequential_cpu_offload: bool = False  # 순차적 CPU 오프로드 활성화 여부
        enable_attention_slicing: bool = False  # attention slicing 활성화 여부
        enable_channels_last_format: bool = False  # channels last 메모리 포맷 사용 여부
        guidance_scale: float = 7.5  # classifier-free guidance 스케일
        condition_scale: float = 1.5  # 조건 이미지 guidance 스케일
        grad_clip: Optional[
            Any
        ] = None  # 그래디언트 클리핑 값 (field(default_factory=lambda: [0, 2.0, 8.0, 1000]))
        half_precision_weights: bool = True  # float16 가중치 사용 여부
        fixed_size: int = -1  # 고정 이미지 크기 (-1이면 동적)

        min_step_percent: float = 0.02  # 최소 timestep 비율 (전체의 2%)
        max_step_percent: float = 0.98  # 최대 timestep 비율 (전체의 98%)
        diffusion_steps: int = 20  # diffusion denoising 단계 수
        use_sds: bool = False  # Score Distillation Sampling 사용 여부
        camera_batch_size: int = 5  # 카메라 배치 크기

    cfg: Config

    def configure(self) -> None:
        """
        모델 초기화 및 설정
        InstructPix2Pix 파이프라인과 DDIM 스케줄러를 로드하고 설정
        """
        threestudio.info(f"Loading InstructPix2Pix ...")

        # 가중치 데이터 타입 설정 (float16 또는 float32)
        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        # 파이프라인 초기화 인자
        pipe_kwargs = {
            "safety_checker": None,  # 안전 검사기 비활성화
            "feature_extractor": None,  # 특징 추출기 비활성화
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
            "cache_dir": self.cfg.cache_dir,
        }

        # InstructPix2Pix 파이프라인 로드
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.cfg.ip2p_name_or_path, **pipe_kwargs
        ).to(self.device)
        
        # DDIM 스케줄러 로드 및 설정
        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.ddim_scheduler_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
            cache_dir=self.cfg.cache_dir,
        )
        # diffusion 단계 수 설정 (20단계로 denoising 수행)
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)

        # 메모리 효율적인 attention 활성화
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

        # 순차적 CPU 오프로드 활성화 (메모리 절약)
        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        # Attention slicing 활성화 (메모리 절약)
        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        # Channels last 메모리 포맷 사용 (성능 최적화)
        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        # VAE와 UNet 모델 추출 및 평가 모드로 설정
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()

        # 모델 파라미터의 그래디언트 계산 비활성화 (추론 모드)
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        # 학습 시 사용된 timestep 수 저장
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # 기본값으로 최소/최대 step 설정

        # 누적 알파 값 저장 (노이즈 스케줄)
        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None

        threestudio.info(f"Loaded InstructPix2Pix!")
        
        # UNet의 모든 BasicTransformerBlock을 DGE 블록으로 변환
        for _, module in self.unet.named_modules():
            if isinstance_str(module, "BasicTransformerBlock"):
                make_block_fn = make_dge_block 
                module.__class__ = make_block_fn(module.__class__)
                # 구버전 diffusers 호환성을 위한 설정
                if not hasattr(module, "use_ada_layer_norm_zero"):
                    module.use_ada_layer_norm = False
                    module.use_ada_layer_norm_zero = False
        
        # 확장된 attention 메커니즘 등록
        register_extended_attention(self)

    
    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        """
        최소/최대 timestep 설정
        너무 높거나 낮은 노이즈 레벨을 피하기 위해 timestep 범위 제한
        """
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        """
        UNet을 통한 노이즈 예측
        Args:
            latents: 잠재 공간 이미지
            t: timestep
            encoder_hidden_states: 텍스트 임베딩
        Returns:
            예측된 노이즈
        """
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 H W"]
    ) -> Float[Tensor, "B 4 DH DW"]:
        """
        이미지를 VAE를 통해 잠재 공간으로 인코딩
        Args:
            imgs: 입력 이미지 [배치, 3채널, 높이, 너비]
        Returns:
            잠재 공간 표현 [배치, 4채널, 높이/8, 너비/8]
        """
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0  # [0, 1] -> [-1, 1] 정규화
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_cond_images(
        self, imgs: Float[Tensor, "B 3 H W"]
    ) -> Float[Tensor, "B 4 DH DW"]:
        """
        조건 이미지를 VAE를 통해 인코딩 (classifier-free guidance용)
        positive, positive, negative(zeros) 세 가지 버전 생성
        Args:
            imgs: 조건 이미지
        Returns:
            [positive, positive, negative] 형태의 잠재 공간 표현
        """
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0  # [0, 1] -> [-1, 1] 정규화
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.mode()  # 샘플링 대신 모드 사용
        uncond_image_latents = torch.zeros_like(latents)  # negative 조건 (zeros)
        latents = torch.cat([latents, latents, uncond_image_latents], dim=0)
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self, latents: Float[Tensor, "B 4 DH DW"]
    ) -> Float[Tensor, "B 3 H W"]:
        """
        잠재 공간 표현을 VAE를 통해 이미지로 디코딩
        Args:
            latents: 잠재 공간 표현
        Returns:
            복원된 이미지 [배치, 3채널, 높이, 너비]
        """
        input_dtype = latents.dtype
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)  # [-1, 1] -> [0, 1] 정규화
        return image.to(input_dtype)

    def use_normal_unet(self):
        """
        일반 UNet attention 메커니즘 사용
        timestep이 작을 때 (거의 완성된 상태) 일반 attention 사용
        """
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
    ) -> Float[Tensor, "B 4 DH DW"]:
        """
        잠재 공간 이미지를 편집 (denoising 과정)
        Args:
            text_embeddings: 텍스트 임베딩 [positive, negative, negative]
            latents: 편집할 잠재 공간 이미지
            image_cond_latents: 조건 이미지 잠재 공간 표현
            t: 초기 timestep
            cams: 카메라 파라미터 리스트
        Returns:
            편집된 잠재 공간 이미지
        """
        
        # 스케줄러의 최대 timestep을 입력 timestep으로 설정
        self.scheduler.config.num_train_timesteps = t.item() if len(t.shape) < 1 else t[0].item()
        # diffusion 단계 수 설정 (20단계)
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)

        current_H = image_cond_latents.shape[2]  # 현재 높이
        current_W = image_cond_latents.shape[3]  # 현재 너비

        camera_batch_size = self.cfg.camera_batch_size # 5
        print("Start editing images...")

        with torch.no_grad():
            # 노이즈 추가 (diffusion 과정 시작)
            noise = torch.randn_like(latents)
            latents = self.scheduler.add_noise(latents, noise, t) 

            # 코드 일부는 huggingface diffusers의 InstructPix2Pix 파이프라인에서 가져옴
            # https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
            positive_text_embedding, negative_text_embedding, _ = text_embeddings.chunk(3)
            split_image_cond_latents, _, zero_image_cond_latents = image_cond_latents.chunk(3)
            
            # Denoising 루프: timestep이 큰 값(932, 883, ...)에서 작은 값(50, 1)까지 역순으로 진행
            # timestep: 932, 883, 834, ..., 50, 1까지 약 49씩 작아지는 20개의 숫자
            for t in self.scheduler.timesteps:
                # timestep에 따라 다른 attention 메커니즘 사용
                if t < 100:
                    # timestep이 작을 때 (거의 완성된 상태): 일반 UNet 사용
                    self.use_normal_unet()
                else:
                    # timestep이 클 때 (노이즈 많은 상태): 확장된 attention 사용
                    register_normal_attn_flag(self.unet, False)
                    
                with torch.no_grad():
                    # 노이즈 예측을 위한 리스트 초기화
                    noise_pred_text = []
                    noise_pred_image = []
                    noise_pred_uncond = []
                    
                    # 각 카메라 배치에서 pivotal(중심) 이미지 인덱스 랜덤 선택
                    pivotal_idx = torch.randint(camera_batch_size, (len(latents)//camera_batch_size,)) + torch.arange(0, len(latents), camera_batch_size) 
                    register_pivotal(self.unet, True)  # pivotal 모드 활성화
                    
                    ## PIVOT 연산
                    # Pivotal 카메라 선택
                    key_cams = [cams[cam_pivotal_idx] for cam_pivotal_idx in pivotal_idx.tolist()] # camera_batch_size개의 카메라
                    
                    # Pivotal 이미지에 대한 입력 준비 (positive, negative, negative 3개 복사)
                    latent_model_input = torch.cat([latents[pivotal_idx]] * 3)
                    pivot_text_embeddings = torch.cat([positive_text_embedding[pivotal_idx], negative_text_embedding[pivotal_idx], negative_text_embedding[pivotal_idx]], dim=0)
                    pivot_image_cond_latetns = torch.cat([split_image_cond_latents[pivotal_idx], split_image_cond_latents[pivotal_idx], zero_image_cond_latents[pivotal_idx]], dim=0)
                    latent_model_input = torch.cat([latent_model_input, pivot_image_cond_latetns], dim=1)

                    # Pivotal 이미지에 대한 UNet forward pass (attention 메커니즘 초기화)
                    self.forward_unet(latent_model_input, t, encoder_hidden_states=pivot_text_embeddings)
                    register_pivotal(self.unet, False)  # pivotal 모드 비활성화

                    ## NON-PIVOT 연산
                    # 각 카메라 배치에 대해 처리
                    for i, b in enumerate(range(0, len(latents), camera_batch_size)):
                        register_batch_idx(self.unet, i)  # 현재 배치 인덱스 등록
                        # 현재 배치의 카메라 정보 등록
                        register_cams(self.unet, cams[b:b + camera_batch_size], pivotal_idx[i] % camera_batch_size, key_cams) 
                        
                        # Epipolar constraint 계산 (다양한 해상도에서)
                        epipolar_constrains = {}
                        for down_sample_factor in [1, 2, 4, 8]:  # 원본, 1/2, 1/4, 1/8 해상도
                            H = current_H // down_sample_factor
                            W = current_W // down_sample_factor
                            epipolar_constrains[H * W] = []
                            # 각 카메라에 대해 pivotal 카메라와의 epipolar constraint 계산
                            for cam in cams[b:b + camera_batch_size]:
                                cam_epipolar_constrains = []
                                for key_cam in key_cams:
                                    cam_epipolar_constrains.append(compute_epipolar_constrains(key_cam, cam, current_H=H, current_W=W))
                                epipolar_constrains[H * W].append(torch.stack(cam_epipolar_constrains, dim=0))
                            epipolar_constrains[H * W] = torch.stack(epipolar_constrains[H * W], dim=0)
                        register_epipolar_constrains(self.unet, epipolar_constrains)  # epipolar constraint 등록

                        # 배치 입력 준비 (positive, negative, negative 3개 복사)
                        batch_model_input = torch.cat([latents[b:b + camera_batch_size]] * 3)
                        batch_text_embeddings = torch.cat([positive_text_embedding[b:b + camera_batch_size], negative_text_embedding[b:b + camera_batch_size], negative_text_embedding[b:b + camera_batch_size]], dim=0)
                        batch_image_cond_latents = torch.cat([split_image_cond_latents[b:b + camera_batch_size], split_image_cond_latents[b:b + camera_batch_size], zero_image_cond_latents[b:b + camera_batch_size]], dim=0)
                        batch_model_input = torch.cat([batch_model_input, batch_image_cond_latents], dim=1)

                        # UNet을 통한 노이즈 예측
                        batch_noise_pred = self.forward_unet(batch_model_input, t, encoder_hidden_states=batch_text_embeddings)
                        batch_noise_pred_text, batch_noise_pred_image, batch_noise_pred_uncond = batch_noise_pred.chunk(3)
                        noise_pred_text.append(batch_noise_pred_text)
                        noise_pred_image.append(batch_noise_pred_image)
                        noise_pred_uncond.append(batch_noise_pred_uncond)

                    # 모든 배치의 예측 결과 합치기
                    noise_pred_text = torch.cat(noise_pred_text, dim=0)
                    noise_pred_image = torch.cat(noise_pred_image, dim=0)
                    noise_pred_uncond = torch.cat(noise_pred_uncond, dim=0)

                    # Classifier-free guidance 수행
                    # 최종 노이즈 예측 = uncond + guidance_scale * (text - image) + condition_scale * (image - uncond)
                    noise_pred = (
                        noise_pred_uncond
                        + self.cfg.guidance_scale * (noise_pred_text - noise_pred_image)
                        + self.cfg.condition_scale * (noise_pred_image - noise_pred_uncond)
                    )

                    # 스케줄러를 사용하여 이전 샘플로 업데이트 (denoising 한 단계 진행)
                    latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                    
        print("Editing finished.")
        return latents

    def compute_grad_sds(
        self,
        text_embeddings: Float[Tensor, "BB 77 768"],
        latents: Float[Tensor, "B 4 DH DW"],
        image_cond_latents: Float[Tensor, "B 4 DH DW"],
        t: Int[Tensor, "B"],
        cams= None,
    ):
        """
        Score Distillation Sampling (SDS)를 위한 그래디언트 계산
        Args:
            text_embeddings: 텍스트 임베딩
            latents: 잠재 공간 이미지
            image_cond_latents: 조건 이미지 잠재 공간 표현
            t: timestep
            cams: 카메라 파라미터 리스트
        Returns:
            SDS 그래디언트
        """
        # 노이즈 추가
        noise = torch.randn_like(latents)
        latents = self.scheduler.add_noise(latents, noise, t) 
        positive_text_embedding, negative_text_embedding, _ = text_embeddings.chunk(3)
        split_image_cond_latents, _, zero_image_cond_latents = image_cond_latents.chunk(3)
        current_H = image_cond_latents.shape[2]
        current_W = image_cond_latents.shape[3]
        camera_batch_size = self.cfg.camera_batch_size
        
        with torch.no_grad():
            noise_pred_text = []
            noise_pred_image = []
            noise_pred_uncond = []
            # 각 카메라 배치에서 pivotal 이미지 인덱스 랜덤 선택
            pivotal_idx = torch.randint(camera_batch_size, (len(latents)//camera_batch_size,)) + torch.arange(0,len(latents),camera_batch_size) 
            print(pivotal_idx)
            register_pivotal(self.unet, True)  # pivotal 모드 활성화

            # Pivotal 이미지에 대한 입력 준비
            latent_model_input = torch.cat([latents[pivotal_idx]] * 3)
            pivot_text_embeddings = torch.cat([positive_text_embedding[pivotal_idx], negative_text_embedding[pivotal_idx], negative_text_embedding[pivotal_idx]], dim=0)
            pivot_image_cond_latetns = torch.cat([split_image_cond_latents[pivotal_idx], split_image_cond_latents[pivotal_idx], zero_image_cond_latents[pivotal_idx]], dim=0)
            latent_model_input = torch.cat([latent_model_input, pivot_image_cond_latetns], dim=1)
            
            key_cams = cams[pivotal_idx]
            # Pivotal 이미지에 대한 UNet forward pass
            self.forward_unet(latent_model_input, t, encoder_hidden_states=pivot_text_embeddings)
            register_pivotal(self.unet, False)  # pivotal 모드 비활성화


            # 각 카메라 배치에 대해 처리
            for i, b in enumerate(range(0, len(latents), camera_batch_size)):
                register_batch_idx(self.unet, i)  # 현재 배치 인덱스 등록
                # 현재 배치의 카메라 정보 등록
                register_cams(self.unet, cams[b:b + camera_batch_size], pivotal_idx[i] % camera_batch_size, key_cams) 
                
                # Epipolar constraint 계산 (다양한 해상도에서)
                epipolar_constrains = {}
                for down_sample_factor in [1, 2, 4, 8]:  # 원본, 1/2, 1/4, 1/8 해상도
                    H = current_H // down_sample_factor
                    W = current_W // down_sample_factor
                    epipolar_constrains[H * W] = []
                    # 각 카메라에 대해 pivotal 카메라와의 epipolar constraint 계산
                    for cam in cams[b:b + camera_batch_size]:
                        cam_epipolar_constrains = []
                        for key_cam in key_cams:
                            cam_epipolar_constrains.append(compute_epipolar_constrains(key_cam, cam, current_H=H, current_W=W))
                        epipolar_constrains[H * W].append(torch.stack(cam_epipolar_constrains, dim=0))
                    epipolar_constrains[H * W] = torch.stack(epipolar_constrains[H * W], dim=0)
                register_epipolar_constrains(self.unet, epipolar_constrains)  # epipolar constraint 등록

                # 배치 입력 준비 (positive, negative, negative 3개 복사)
                batch_model_input = torch.cat([latents[b:b + camera_batch_size]] * 3)
                batch_text_embeddings = torch.cat([positive_text_embedding[b:b + camera_batch_size], negative_text_embedding[b:b + camera_batch_size], negative_text_embedding[b:b + camera_batch_size]], dim=0)
                batch_image_cond_latents = torch.cat([split_image_cond_latents[b:b + camera_batch_size], split_image_cond_latents[b:b + camera_batch_size], zero_image_cond_latents[b:b + camera_batch_size]], dim=0)
                batch_model_input = torch.cat([batch_model_input, batch_image_cond_latents], dim=1)
                
                # UNet을 통한 노이즈 예측
                batch_noise_pred = self.forward_unet(batch_model_input, t, encoder_hidden_states=batch_text_embeddings)
                batch_noise_pred_text, batch_noise_pred_image, batch_noise_pred_uncond = batch_noise_pred.chunk(3)
                noise_pred_text.append(batch_noise_pred_text)
                noise_pred_image.append(batch_noise_pred_image)
                noise_pred_uncond.append(batch_noise_pred_uncond)

            # 모든 배치의 예측 결과 합치기
            noise_pred_text = torch.cat(noise_pred_text, dim=0)
            noise_pred_image = torch.cat(noise_pred_image, dim=0)
            noise_pred_uncond = torch.cat(noise_pred_uncond, dim=0)

            # Classifier-free guidance 수행
            noise_pred = (
                noise_pred_uncond
                + self.cfg.guidance_scale * (noise_pred_text - noise_pred_image)
                + self.cfg.condition_scale * (noise_pred_image - noise_pred_uncond)
            )

        # SDS 그래디언트 계산: 가중치 * (예측 노이즈 - 실제 노이즈)
        # 가중치는 timestep에 따라 결정됨 (큰 timestep일수록 큰 가중치)
        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        grad = w * (noise_pred - noise)
        return grad
    



    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        cond_rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        gaussians = None,
        cams= None,
        render=None,
        pipe=None,
        background=None,
        **kwargs,
    ):
        """
        메인 호출 함수: 이미지 편집 수행
        Args:
            rgb: 입력 RGB 이미지 [배치, 높이, 너비, 3채널]
            cond_rgb: 조건 RGB 이미지
            prompt_utils: 프롬프트 처리 유틸리티
            cams: 카메라 파라미터 리스트 (필수)
        Returns:
            편집된 이미지 또는 SDS 손실
        """
        assert cams is not None, "cams is required for dge guidance"
        batch_size, H, W, _ = rgb.shape # batch_size: max_view_num
        
        # 이미지 크기를 64의 배수로 조정 (VAE 요구사항)
        factor = 512 / max(W, H)
        factor = math.ceil(min(W, H) * factor / 64) * 64 / min(W, H)

        width = int((W * factor) // 64) * 64
        height = int((H * factor) // 64) * 64
        rgb_BCHW = rgb.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        RH, RW = height, width

        # 이미지 리사이즈 및 인코딩
        rgb_BCHW_HW8 = F.interpolate(
            rgb_BCHW, (RH, RW), mode="bilinear", align_corners=False
        )
        latents = self.encode_images(rgb_BCHW_HW8)
        
        # 조건 이미지 리사이즈 및 인코딩
        cond_rgb_BCHW = cond_rgb.permute(0, 3, 1, 2)
        cond_rgb_BCHW_HW8 = F.interpolate(
            cond_rgb_BCHW,
            (RH, RW),
            mode="bilinear",
            align_corners=False,
        )
        cond_latents = self.encode_cond_images(cond_rgb_BCHW_HW8)

        # 텍스트 임베딩 생성
        temp = torch.zeros(batch_size).to(rgb.device)
        text_embeddings = prompt_utils.get_text_embeddings(temp, temp, temp, False)
        positive_text_embeddings, negative_text_embeddings = text_embeddings.chunk(2)
        text_embeddings = torch.cat(
            [positive_text_embeddings, negative_text_embeddings, negative_text_embeddings], dim=0)  # [positive, negative, negative]

        # Timestep 랜덤 샘플링: 너무 높거나 낮은 노이즈 레벨을 피하기 위해 범위 제한
        # timestep ~ U(max_step-1, max_step) ≈ U(0.98 * num_train_timesteps, 0.98 * num_train_timesteps)
        t = torch.randint(
            self.max_step - 1,
            self.max_step,
            [1],
            dtype=torch.long,
            device=self.device,
        ).repeat(batch_size)

        if self.cfg.use_sds:
            # Score Distillation Sampling 모드: 그래디언트 계산 및 손실 반환
            grad = self.compute_grad_sds(text_embeddings, latents, cond_latents, t, cams)
            grad = torch.nan_to_num(grad)  # NaN 값 처리
            if self.grad_clip_val is not None:
                grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)  # 그래디언트 클리핑
            target = (latents - grad).detach()
            loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
            return {
                "loss_sds": loss_sds,
                "grad_norm": grad.norm(),
                "min_step": self.min_step,
                "max_step": self.max_step,
            }
        else:
            # 이미지 편집 모드: denoising을 통한 이미지 편집 수행
            edit_latents = self.edit_latents(text_embeddings, latents, cond_latents, t, cams)
            edit_images = self.decode_latents(edit_latents)
            # 원본 해상도로 복원
            edit_images = F.interpolate(edit_images, (H, W), mode="bilinear")

            return {"edit_images": edit_images.permute(0, 2, 3, 1)}  # [B, C, H, W] -> [B, H, W, C]

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        """
        학습 단계마다 호출되는 업데이트 함수
        그래디언트 클리핑 값과 timestep 범위를 동적으로 조정
        참고: Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        http://arxiv.org/abs/2303.15413
        """
        # 그래디언트 클리핑 값 업데이트 (학습 단계에 따라 동적 조정)
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        # 최소/최대 timestep 범위 업데이트 (학습 단계에 따라 동적 조정)
        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )


