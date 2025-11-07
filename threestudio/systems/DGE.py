from dataclasses import dataclass, field
import random
from re import T

from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np
import sys
import shutil
import torch
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


from CLIP.utils.image_utils import img_normalize, clip_normalize
from CLIP.scene.VGG import get_features
import CLIP

clip_model = CLIP.load_model()


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
        max_grad: float = 1e-7
        min_opacity: float = 0.005

        seg_prompt: str = ""
        target_prompt: str = ""

        # cache
        cache_overwrite: bool = True
        cache_dir: str = ""


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
        
        if seg_object == self.cfg.target_prompt:
            # select 20 views not in self.view_list
            all_views = set(range(0, 60))
            candidates = list(all_views - set(self.edit_view_index))
            if len(candidates) < 20:
                raise ValueError(f"Not enough views outside self.view_list to sample 20 views (got {len(candidates)}).")
            view_list = random.sample(candidates, 20)


        elif seg_object == self.cfg.seg_prompt:
            # view_list = self.view_list
            view_list = random.sample(range(0, 60), 30)

        # view_list = [_ for _ in range(0, 65)]

        print(f"View list: {view_list}")

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


            for id in tqdm(view_list):
                cur_path = os.path.join(mask_cache_dir, "{:0>4d}.png".format(id))
                cur_path_viz = os.path.join(
                    mask_cache_dir, "viz_{:0>4d}.png".format(id)
                )
                cur_cam = self.trainer.datamodule.train_dataset.scene.cameras[id]

                if seg_object == self.cfg.target_prompt:
                    # image_to_segment = self.edit_frames[id]

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
                    cached_image = cv2.cvtColor(cv2.imread(cur_path), cv2.COLOR_BGR2RGB)
                    image_to_segment = torch.tensor(
                        cached_image / 255, device="cuda", dtype=torch.float32
                    )[None]

                elif seg_object == self.cfg.seg_prompt:
                    image_to_segment = self.origin_frames[id]

                mask = self.text_segmentor(image_to_segment, seg_object)[0].to(get_device())

                mask_to_save = ( # todo: target_prompt에 대한 마스크는 저장할 필요 없음.
                        mask[0]
                        .cpu()  
                        .detach()[..., None]
                        .repeat(1, 1, 3)
                        .numpy()
                        .clip(0.0, 1.0)
                        * 255.0
                ).astype(np.uint8)
                cv2.imwrite(cur_path, mask_to_save)

                masked_image = image_to_segment.detach().clone()[0]
                masked_image[mask[0].bool()] *= 0.3
                masked_image_to_save = (
                        masked_image.cpu().detach().numpy().clip(0.0, 1.0) * 255.0
                ).astype(np.uint8)
                masked_image_to_save = cv2.cvtColor(
                    masked_image_to_save, cv2.COLOR_RGB2BGR
                )
                cv2.imwrite(cur_path_viz, masked_image_to_save)
                self.gaussian.apply_weights(cur_cam, weights, weights_cnt, mask)

            weights /= weights_cnt + 1e-7

            selected_mask = weights > self.cfg.mask_thres
            selected_mask = selected_mask[:, 0]
            torch.save(selected_mask, gs_mask_path)
        else:
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

        self.gaussian.set_mask(selected_mask)
        self.gaussian.apply_grad_mask(selected_mask)

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
                if not os.path.exists(cur_path) or self.cfg.cache_overwrite:
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
                cached_image = cv2.cvtColor(cv2.imread(cur_path), cv2.COLOR_BGR2RGB)
                self.origin_frames[id] = torch.tensor(
                    cached_image / 255, device="cuda", dtype=torch.float32
                )[None]

    def on_before_optimizer_step(self, optimizer):
        with torch.no_grad():
            if self.true_global_step < self.cfg.densify_until_iter:
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
                    self.gaussian.densify_and_prune(
                        self.cfg.max_grad,
                        self.cfg.max_densify_percent,
                        self.cfg.min_opacity,
                        self.cameras_extent,
                        5,
                    )

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
        save_list = []
        for index, image in sorted(self.edit_frames.items(), key=lambda item: item[0]):
            save_list.append(
                {
                    "type": "rgb",
                    "img": image[0],
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
        save_list = []
        for index, image in sorted(
                self.origin_frames.items(), key=lambda item: item[0]
        ):
            save_list.append(
                {
                    "type": "rgb",
                    "img": image[0],
                    "kwargs": {"data_format": "HWC"},
                },
            )
        self.save_image_grid(
            f"origin_images.png",
            save_list,
            name="origin",
            step=self.true_global_step,
        )

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
        self.gaussian.max_radii2D = torch.zeros(
            (self.gaussian.get_xyz.shape[0]), device="cuda"
        )
        self.cameras_extent = self.trainer.datamodule.train_dataset.scene.cameras_extent
        self.gaussian.spatial_lr_scale = self.cameras_extent

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
            with self._latency_logger.timeit("edit_all_view.update_editing_cameras"):
                self.trainer.datamodule.train_dataset.update_editing_cameras(random_seed = global_step + 1)
                self.edit_view_index = self.trainer.datamodule.train_dataset.edit_view_index
                sorted_train_view_list = sorted(self.edit_view_index)
                selected_views = torch.linspace(
                    0, len(sorted_train_view_list) - 1, self.trainer.datamodule.val_dataset.n_views, dtype=torch.int
                )
                self.trainer.datamodule.val_dataset.selected_views = [sorted_train_view_list[idx] for idx in selected_views]
        
        print(f"{self.true_global_step}th step, Camera view index: {self.edit_view_index}")

        self.edit_frames = {}
        cache_dir = os.path.join(self.cache_dir, cache_name)
        original_render_cache_dir = os.path.join(self.cache_dir, original_render_name)
        os.makedirs(cache_dir, exist_ok=True)

        cameras = []
        images = []
        original_frames = []
        t_max_step = self.cfg.added_noise_schedule
        self.guidance.max_step = t_max_step[min(len(t_max_step)-1, self.true_global_step//self.cfg.camera_update_per_step)]
        with torch.no_grad():
            with self._latency_logger.timeit("edit_all_view.collect_cameras"):
                for id in self.edit_view_index:
                    cameras.append(self.trainer.datamodule.train_dataset.scene.cameras[id])
            with self._latency_logger.timeit("edit_all_view.sort_cameras"):
                sorted_cam_idx = self.sort_the_cameras_idx(cameras)
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
                with self._latency_logger.timeit("edit_all_view.render_single"):
                    out_pkg = self(cur_batch)
                out = out_pkg["comp_rgb"] ## 이게 forward해서 렌더링 결과 얻는 부분임!
                if self.cfg.use_masked_image:
                    with self._latency_logger.timeit("edit_all_view.apply_mask"):
                        out = out * out_pkg["masks"].unsqueeze(-1)
                images.append(out)
                assert os.path.exists(original_image_path)
                with self._latency_logger.timeit("edit_all_view.load_original"):
                    cached_image = cv2.cvtColor(cv2.imread(original_image_path), cv2.COLOR_BGR2RGB)
                    self.origin_frames[id] = torch.tensor(
                        cached_image / 255, device="cuda", dtype=torch.float32
                    )[None]
                original_frames.append(self.origin_frames[id])
            with self._latency_logger.timeit("edit_all_view.concat_batches"):
                images = torch.cat(images, dim=0) ## view들을 concat하여 배치로 만듦
                original_frames = torch.cat(original_frames, dim=0)

            with self._latency_logger.timeit("edit_all_view.guidance_batch"):
                edited_images = self.guidance( ## DGEGuidance.__call__ 함수 호출
                    images, ## 편집대상(latents): training 되고 있는 3dgs에서 렌더한 이미지. 
                    original_frames, ## 이미지 조건(latents): 원본 이미지(Edit 전의 GT)
                    self.prompt_processor(), ## 텍스트 조건(text_embeddings): 편집 프롬프트
                    cams = cams_sorted,
                    latency_logger = self._latency_logger
                )

            with self._latency_logger.timeit("edit_all_view.assign_outputs"):
                for view_index_tmp in range(len(self.edit_view_index)):
                    self.edit_frames[view_sorted[view_index_tmp]] = edited_images['edit_images'][view_index_tmp].unsqueeze(0).detach().clone() # 1 H W C
    
    def sort_the_cameras_idx(self, cams):
        foward_vectos = [cam.R[:, 2] for cam in cams]
        foward_vectos = np.array(foward_vectos)
        cams_center_x = np.array([cam.camera_center[0].item() for cam in cams])
        most_left_vecotr = foward_vectos[np.argmin(cams_center_x)]
        distances = [np.arccos(np.clip(np.dot(most_left_vecotr, cam.R[:, 2]), 0, 1)) for cam in cams]
        sorted_cams = [cam for _, cam in sorted(zip(distances, cams), key=lambda pair: pair[0])]
        reference_axis = np.cross(most_left_vecotr, sorted_cams[1].R[:, 2])
        distances_with_sign = [np.arccos(np.dot(most_left_vecotr, cam.R[:, 2])) if np.dot(reference_axis,  np.cross(most_left_vecotr, cam.R[:, 2])) >= 0 else 2 * np.pi - np.arccos(np.dot(most_left_vecotr, cam.R[:, 2])) for cam in cams]
        
        sorted_cam_idx = [idx for _, idx in sorted(zip(distances_with_sign, range(len(cams))), key=lambda pair: pair[0])]

        return sorted_cam_idx

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # latency logger under trial_dir/latency
        latency_dir = os.path.join(self.get_save_dir(), "..", "latency")
        latency_dir = os.path.abspath(latency_dir)
        os.makedirs(latency_dir, exist_ok=True)
        self._latency_logger = LatencyLogger(latency_dir)

        with self._latency_logger.timeit("render_all_view"):
            self.render_all_view(cache_name="origin_render")

        if len(self.cfg.seg_prompt) > 0:
            with self._latency_logger.timeit("update_mask"):
                self.update_mask(self.cfg.seg_prompt)

        if len(self.cfg.prompt_processor) > 0:
            self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
                self.cfg.prompt_processor
            )
        if self.cfg.loss.lambda_l1 > 0 or self.cfg.loss.lambda_p > 0 or self.cfg.loss.use_sds:
            with self._latency_logger.timeit("guidance_init"):
                self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
                # Set save_dir for guidance to save epipolar constraint images
                self.guidance.save_dir = self.get_save_dir()
            
        self.style_direction = CLIP.get_style_embedding(
            clip_model,
            self.cfg.target_prompt,
            None, # self.cfg.style_image,
            self.cfg.seg_prompt # self.cfg.object_prompt
        )

    def training_step(self, batch, batch_idx):
        if self.true_global_step % self.cfg.camera_update_per_step == 0 and self.cfg.guidance_type == 'dge-guidance' and not self.cfg.loss.use_sds:
            with self._latency_logger.timeit("edit_all_view"):
                self.edit_all_view(original_render_name='origin_render', cache_name="edited_views", update_camera=self.true_global_step >= self.cfg.camera_update_per_step, global_step=self.true_global_step) 
        
        if self.true_global_step == self.cfg.mask_update_at_step and len(self.cfg.target_prompt) > 0:
            with self._latency_logger.timeit(f"update_mask at step {self.true_global_step}"):
                print(f"Update mask with prompt: {self.cfg.target_prompt}")
                self.update_mask(self.cfg.target_prompt)

        self.gaussian.update_learning_rate(self.true_global_step)
        batch_index = batch["index"]

        if isinstance(batch_index, int):
            batch_index = [batch_index]
        if self.cfg.guidance_type == 'dge-guidance': 
            for img_index, cur_index in enumerate(batch_index):
                if cur_index not in self.edit_frames:
                    batch_index[img_index] = self.trainer.datamodule.train_dataset.train_view_index[img_index] # 전체 train view

        with self._latency_logger.timeit("render_forward"):
            out = self(batch, local=self.cfg.local_edit)

        images = out["comp_rgb"]
        mask = out["masks"].unsqueeze(-1)
        loss = 0.0
        # nerf2nerf loss
        if self.cfg.loss.lambda_l1 > 0 or self.cfg.loss.lambda_p > 0:
            prompt_utils = self.prompt_processor()
            gt_images = []
            for img_index, cur_index in enumerate(batch_index):
                # if cur_index not in self.edit_frames:
                #     # cur_index = self.view_list[0]
                if cur_index in self.edit_frames:
                    gt_images.append(self.edit_frames[cur_index])

                else: # CLIP LOSS
                    pass


                # if (cur_index not in self.edit_frames or ( # 만약에 cur_index가 edit_frames에 없다면 그때 즉시 guidance 통과해서 이미지를 에디팅 해주는거임.
                #     # edited_frames: dict{view_index: image} 형태로 저장됨.
                #         self.cfg.per_editing_step > 0
                #         and self.cfg.edit_begin_step
                #         < self.global_step
                #         < self.cfg.edit_until_step
                #         and self.global_step % self.cfg.per_editing_step == 0
                # )) and 'dge' not in str(self.cfg.guidance_type) and not self.cfg.loss.use_sds:
                #     print(self.cfg.guidance_type)
                #     with self._latency_logger.timeit("guidance_edit_single"): ## 여기 절대로 통과 안됨!
                #         result = self.guidance(
                #             images[img_index][None],
                #             self.origin_frames[cur_index],
                #             prompt_utils,
                #         )
                #     self.edit_frames[cur_index] = result["edit_images"].detach().clone()

            if len(gt_images) > 0:
                gt_images = torch.concatenate(gt_images, dim=0)

                if self.cfg.use_masked_image:
                    print("use masked image")
                    loss_dict = {
                    "loss_l1": torch.nn.functional.l1_loss(images * mask, gt_images * mask),
                    "loss_p": self.perceptual_loss(
                        (images * mask).permute(0, 3, 1, 2).contiguous(),
                        (gt_images * mask ).permute(0, 3, 1, 2).contiguous(),
                    ).sum(),
                    }
                else:
                    loss_dict = {
                        "loss_l1": torch.nn.functional.l1_loss(images, gt_images),
                        "loss_p": self.perceptual_loss(
                            images.permute(0, 3, 1, 2).contiguous(),
                            gt_images.permute(0, 3, 1, 2).contiguous(),
                        ).sum(),
                    }

                
            else:
                # Direction CLIP loss
                # images shape: (B, H, W, C) -> (B, C, H, W)로 변환 필요
                images_clip = images.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
                gt_images_clip = self.origin_frames[batch_index[0]].permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
                render_features = clip_model.encode_image(
                    clip_normalize(images_clip))
                source_features = clip_model.encode_image(
                    clip_normalize(gt_images_clip))
                render_features /= (render_features.clone().norm(dim=-1, keepdim=True))

                img_direction = (render_features-source_features)
                img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)

                loss_d = (1 - torch.cosine_similarity(img_direction,
                        self.style_direction.repeat(render_features.size(0), 1), dim=1)).mean()
                
                loss_dict = {"loss_d": loss_d}

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
            with self._latency_logger.timeit("guidance_sds"):
                loss_dict = self.guidance(
                    out["comp_rgb"],
                    torch.concatenate(
                        [self.origin_frames[idx] for idx in batch_index], dim=0
                    ),
                    prompt_utils)  
            loss += loss_dict["loss_sds"] * self.cfg.loss.lambda_sds 

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))
    
        return {"loss": loss}

    def on_train_end(self) -> None:
        """Called when training ends. Write latency summary."""
        if hasattr(self, '_latency_logger'):
            self._latency_logger.write_summary()
            threestudio.info(f"Latency summary written to {self._latency_logger.base_dir}/summary.txt")
