import bisect
import random
import os
from dataclasses import dataclass, field

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

import threestudio
from threestudio import register
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured

from threestudio.utils.typing import *
from threestudio.utils.sam import LangSAMTextSegmentor
from threestudio.utils.misc import get_device
import numpy as np
from plyfile import PlyData


def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))


def convert_camera_to_world_transform(transform):
    converted_transform = transform.clone()

    converted_transform[:, 2] *= -1

    converted_transform[[0, 2], :] = converted_transform[[2, 0], :]

    return converted_transform


def circle_poses(
    device, radius=torch.tensor([3.2]), theta=torch.tensor([60]), phi=torch.tensor([0])
):
    theta = theta / 180 * np.pi
    phi = phi / 180 * np.pi

    centers = torch.stack(
        [
            radius * torch.sin(theta) * torch.sin(phi),
            radius * torch.cos(theta),
            radius * torch.sin(theta) * torch.cos(phi),
        ],
        dim=-1,
    )  # [B, 3]

    # lookat
    forward_vector = safe_normalize(centers)
    up_vector = (
        torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(len(centers), 1)
    )
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = (
        torch.eye(4, dtype=torch.float, device=device)
        .unsqueeze(0)
        .repeat(len(centers), 1, 1)
    )
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses


trans_t = lambda t: torch.Tensor(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]]
).float()

rot_phi = lambda phi: torch.Tensor(
    [
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1],
    ]
).float()

rot_theta = lambda th: torch.Tensor(
    [
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1],
    ]
).float()


def rodrigues_mat_to_rot(R):
    eps = 1e-16
    trc = np.trace(R)
    trc2 = (trc - 1.0) / 2.0
    s = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if (1 - trc2 * trc2) >= eps:
        tHeta = np.arccos(trc2)
        tHetaf = tHeta / (2 * (np.sin(tHeta)))
    else:
        tHeta = np.real(np.arccos(trc2))
        tHetaf = 0.5 / (1 - tHeta / 6)
    omega = tHetaf * s
    return omega


def rodrigues_rot_to_mat(r):
    wx, wy, wz = r
    theta = np.sqrt(wx * wx + wy * wy + wz * wz)
    a = np.cos(theta)
    b = (1 - np.cos(theta)) / (theta * theta)
    c = np.sin(theta) / theta
    R = np.zeros([3, 3])
    R[0, 0] = a + b * (wx * wx)
    R[0, 1] = b * wx * wy - c * wz
    R[0, 2] = b * wx * wz + c * wy
    R[1, 0] = b * wx * wy + c * wz
    R[1, 1] = a + b * (wy * wy)
    R[1, 2] = b * wy * wz - c * wx
    R[2, 0] = b * wx * wz - c * wy
    R[2, 1] = b * wz * wy + c * wx
    R[2, 2] = a + b * (wz * wz)
    return R


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = (
        torch.Tensor(
            np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        )
        @ c2w
    )
    return c2w


def convert_camera_pose(camera_pose):
    # Clone the tensor to avoid in-place operations
    colmap_pose = camera_pose.clone()

    # Extract rotation and translation components
    rotation = colmap_pose[:, :3, :3]
    translation = colmap_pose[:, :3, 3]

    # Change rotation orientation
    rotation[:, 0, :] *= -1
    rotation[:, 1, :] *= -1

    # Change translation position
    translation[:, 0] *= -1
    translation[:, 1] *= -1

    return colmap_pose


def convert_camera_pose(camera_pose):
    # Clone the tensor to avoid in-place operations
    colmap_pose = camera_pose.clone()

    # Extract rotation and translation components
    rotation = colmap_pose[:, :3, :3]
    translation = colmap_pose[:, :3, 3]

    # Change rotation orientation
    rotation[:, 0, :] *= -1
    rotation[:, 1, :] *= -1

    # Change translation position
    translation[:, 0] *= -1
    translation[:, 1] *= -1

    return colmap_pose


@dataclass
class GSLoadDataModuleConfig:
    # height, width, and batch_size should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    source: str = None
    height: Any = 512
    width: Any = 512
    batch_size: Any = 1
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    eval_height: int = -1
    eval_width: int = -1
    eval_batch_size: int = 1
    max_view_num: int = 60
    max_edit_view_num: int = 15
    edit_view_selection_strategy: str = "quadrant"

    
    n_val_views: int = 8
    n_test_views: int = 120
    elevation_range: Tuple[float, float] = (-10, 45)
    elevation_view_num: int = 2
    azimuth_range: Tuple[float, float] = (-180, 180)
    azimuth_view_num: int = 8
    camera_distance_range: Tuple[float, float] = (4.0, 6.0)
    fovy_range: Tuple[float, float] = (
        40,
        70,
    )  # in degrees, in vertical direction (along height)
    camera_perturb: float = 0.0
    center_perturb: float = 0.0
    up_perturb: float = 0.0
    light_position_perturb: float = 1.0
    light_distance_range: Tuple[float, float] = (0.8, 1.5)
    eval_elevation_deg: float = 15.0
    eval_camera_distance: float = 6.0
    eval_fovy_deg: float = 70.0
    light_sample_strategy: str = "dreamfusion"
    batch_uniform_azimuth: bool = True
    progressive_until: int = 0 
    use_original_resolution: bool = False # use the original resolution of the image or center crop the image
    
    # MMR view selection hyperparameters
    mmr_object_center: Optional[Tuple[float, float, float]] = None  # Object center in world coordinates, if None will be estimated
    mmr_representative_view_idx: int = 0  # Representative view index v_r for segmentation
    mmr_target_coverage: float = 0.3  # Target coverage γ* (desired object coverage ratio)
    mmr_sigma_d: float = 0.7  # Depth matching scale parameter
    mmr_sigma_h: float = 0.3  # Height matching scale parameter
    
    mmr_alpha_d: float = 1.0  # Weight for depth score
    mmr_alpha_h: float = 0.0  # Weight for height score
    
    mmr_sigma_c: float = 1.0  # Position similarity scale parameter
    mmr_sigma_theta: float = 0.5  # Direction similarity scale parameter (in radians)
    mmr_lambda: float = 0.7  # Relevance-diversity trade-off (0=only diversity, 1=only relevance)
    mmr_seg_prompt: str = ""  # Text prompt for segmentation (e.g., "a lego bulldozer"). If empty, will use gt_alpha_mask if available
    mmr_use_gt_mask: bool = True  # If True and mmr_seg_prompt is empty, use gt_alpha_mask from camera. If False, assume full image as object (fallback)


class GSLoadIterableDataset(IterableDataset, Updateable):
    def __init__(self, cfg, scene, system_seg_prompt: Optional[str] = None, gaussian_model=None) -> None:
        super().__init__()
        self.cfg: GSLoadDataModuleConfig = cfg
        self.scene = scene
        self.system_seg_prompt = system_seg_prompt  # System's seg_prompt, if available
        self.gaussian_model = gaussian_model  # GaussianModel instance, if available
        self.total_view_num = len(self.scene.cameras)
        random.seed(0)  # make sure same views


        # self.train_view_index = random.sample(
        #     range(0, self.total_view_num),
        #     min(self.total_view_num, self.cfg.max_view_num),
        # )
        # self.train_view_index_stack = self.train_view_index.copy()
        
        # self.edit_view_index = random.sample(
        #     self.train_view_index,
        #     min(len(self.train_view_index), self.cfg.max_edit_view_num),
        # )
        # self.edit_view_index_stack = self.edit_view_index.copy()


        if self.cfg.edit_view_selection_strategy == "quadrant":
            self.edit_view_index = self._select_cameras_by_quadrants(
                range(0, self.total_view_num),
                self.cfg.max_edit_view_num
            )
        elif self.cfg.edit_view_selection_strategy == "row":
            # row 전략: 기존 카메라 선택
            self.edit_view_index = self._select_cameras_by_rows(
                range(0, self.total_view_num),
                self.cfg.max_edit_view_num
            )
        elif self.cfg.edit_view_selection_strategy == "row-generate": # 안돼~
            # row-generate 전략: 새로운 카메라 뷰 생성 (4개 row, 각 5개씩)
            # 함수 내부에서 이미 scene.cameras에 추가하고 인덱스를 반환함
            self.generated_cameras, self.edit_view_index = self._generate_cameras_by_rows()
            self.total_view_num = len(self.scene.cameras)

        elif self.cfg.edit_view_selection_strategy == "spherical":
            self.generated_cameras, self.edit_view_index = self._generate_spherical_novel_cameras()
            self.total_view_num = len(self.scene.cameras)
        elif self.cfg.edit_view_selection_strategy == "depth":
            self.edit_view_index = self._select_cameras_by_depth()
            self.total_view_num = len(self.scene.cameras)
        elif self.cfg.edit_view_selection_strategy == "y-axis":
            self.edit_view_index = self._select_cameras_by_y_axis(self.cfg.max_edit_view_num)
            self.total_view_num = len(self.scene.cameras)
        elif self.cfg.edit_view_selection_strategy == "mmr":
            self.edit_view_index = self._select_cameras_by_mmr(self.cfg.max_edit_view_num)
            self.total_view_num = len(self.scene.cameras)
        elif self.cfg.edit_view_selection_strategy == "region-aware-only":
            self.edit_view_index = self._select_cameras_by_region_aware_only(self.cfg.max_edit_view_num)
        elif self.cfg.edit_view_selection_strategy == "manual-20":
            self.edit_view_index = [10, 7, 6, 50, 3, 37, 35, 32, 30, 29, 40, 41, 42, 45, 47, 16, 19, 20, 21, 24]
        elif self.cfg.edit_view_selection_strategy == "manual-15":
            # self.edit_view_index = [10, 7, 6, 50, 3, 37, 35, 32, 30, 29, 16, 19, 20, 21, 24]
            self.edit_view_index = [10, 7, 6, 50, 3, 40, 41, 42, 45, 47, 16, 19, 20, 21, 24]

        elif self.cfg.edit_view_selection_strategy == "random":
            self.edit_view_index = random.sample(
                range(0, self.total_view_num),
                self.cfg.max_edit_view_num
            )
        else:
            raise ValueError(f"Invalid edit view selection strategy: {self.cfg.edit_view_selection_strategy}")
        self.edit_view_index_stack = self.edit_view_index.copy()

        # train_view_index는 edit_view_index를 포함하고 max_view_num - max_edit_view_num 만큼을 샘플해가지고 합치는거 하고 싶어
        add_n = self.cfg.max_view_num - self.cfg.max_edit_view_num
        rest = list(set(range(self.total_view_num)) - set(self.edit_view_index))
        self.train_view_index = self.edit_view_index + random.sample(rest, add_n) if add_n > 0 else self.edit_view_index
        self.train_view_index_stack = self.train_view_index.copy()


        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        self.batch_sizes: List[int] = (
            [self.cfg.batch_size] # 1
            if isinstance(self.cfg.batch_size, int)
            else self.cfg.batch_size
        )
        assert len(self.heights) == len(self.widths) == len(self.batch_sizes)
        self.resolution_milestones: List[int]
        if (
            len(self.heights) == 1
            and len(self.widths) == 1
            and len(self.batch_sizes) == 1
        ):
            if len(self.cfg.resolution_milestones) > 0:
                threestudio.warn(
                    "Ignoring resolution_milestones since height and width are not changing"
                )
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones

        self.height: int = self.heights[0]
        self.width: int = self.widths[0]
        self.batch_size: int = self.batch_sizes[0]

    def _select_cameras_by_quadrants(self, candidate_indices, num_cameras):
        """
        Forward facing scene에 대해 x, y 축 기준으로 2x2 = 4개 구획으로 나눠서
        각 구획에서 균등하게 카메라를 선택
        
        Args:
            candidate_indices: 선택 가능한 카메라 인덱스 리스트 또는 range 객체
            num_cameras: 선택할 총 카메라 수
        
        Returns:
            선택된 카메라 인덱스 리스트
        """
        # range 객체를 리스트로 변환
        candidate_indices = list(candidate_indices)
        
        if len(candidate_indices) == 0:
            return []
        
        # 모든 카메라의 중심 위치 가져오기
        cam_centers = []
        for idx in candidate_indices:
            cam = self.scene.cameras[idx]
            center = cam.camera_center
            # torch.Tensor를 numpy로 변환
            if isinstance(center, torch.Tensor):
                center = center.detach().cpu().numpy()
            cam_centers.append(center)
        
        cam_centers = np.array(cam_centers)  # shape: (N, 3)
        
        # x, y 좌표의 중앙값 계산 (z는 무시)
        median_x = np.median(cam_centers[:, 0])
        median_y = np.median(cam_centers[:, 1])
        
        # 4개 구획으로 분류
        # 구획 0: x < median_x, y < median_y (왼쪽 아래)
        # 구획 1: x >= median_x, y < median_y (오른쪽 아래)
        # 구획 2: x < median_x, y >= median_y (왼쪽 위)
        # 구획 3: x >= median_x, y >= median_y (오른쪽 위)
        quadrants = {
            0: [],  # 왼쪽 아래
            1: [],  # 오른쪽 아래
            2: [],  # 왼쪽 위
            3: [],  # 오른쪽 위
        }
        
        for idx, center in zip(candidate_indices, cam_centers):
            x, y = center[0], center[1]
            if x < median_x and y < median_y:
                quadrants[0].append(idx)
            elif x >= median_x and y < median_y:
                quadrants[1].append(idx)
            elif x < median_x and y >= median_y:
                quadrants[2].append(idx)
            else:  # x >= median_x and y >= median_y
                quadrants[3].append(idx)
        
        # 각 구획에서 선택할 카메라 수 계산
        cameras_per_quadrant = num_cameras // 4
        remainder = num_cameras % 4
        
        selected_indices = []
        
        # 각 구획에서 균등하게 선택
        for quad_idx in range(4):
            quadrant_candidates = quadrants[quad_idx]
            if len(quadrant_candidates) == 0:
                continue
            
            # 나머지가 있으면 처음 4개 구획에 1개씩 추가
            num_to_select = cameras_per_quadrant + (1 if quad_idx < remainder else 0)
            num_to_select = min(num_to_select, len(quadrant_candidates))
            
            if num_to_select > 0:
                selected = random.sample(quadrant_candidates, num_to_select)
                selected_indices.extend(selected)
        
        return selected_indices

    def _select_cameras_by_farthest(self, candidate_indices, num_cameras: int):
        """
        기존 self.scene.cameras 중에서 camera_center를 기준으로
        farthest point sampling(FPS)으로 num_cameras개를 선택.

        Args:
            candidate_indices: 선택 가능한 카메라 인덱스 리스트 또는 range 객체
            num_cameras: 선택할 카메라 개수

        Returns:
            선택된 카메라 인덱스 리스트
        """
        import numpy as np
        import torch

        candidate_indices = list(candidate_indices)
        if len(candidate_indices) == 0:
            return []
        if len(candidate_indices) <= num_cameras:
            return candidate_indices

        # camera_center 모으기
        cam_centers = []
        for idx in candidate_indices:
            cam = self.scene.cameras[idx]
            center = cam.camera_center
            if isinstance(center, torch.Tensor):
                center = center.detach().cpu().numpy()
            cam_centers.append(center)
        cam_centers = np.array(cam_centers, dtype=np.float32)  # (N, 3)

        N = cam_centers.shape[0]
        K = min(num_cameras, N)

        # 초기점: 전체 center의 평균에서 가장 먼 카메라
        mean_center = cam_centers.mean(axis=0, keepdims=True)  # (1, 3)
        dists_to_mean = np.linalg.norm(cam_centers - mean_center, axis=1)  # (N,)
        first_idx_local = int(np.argmax(dists_to_mean))

        selected_local_indices = [first_idx_local]

        # 각 포인트가 현재 선택 집합과 가지는 최소 거리
        min_dists = np.linalg.norm(
            cam_centers - cam_centers[first_idx_local:first_idx_local + 1], axis=1
        )  # (N,)

        for _ in range(1, K):
            # 아직 선택되지 않은 것들 중에서 min_dists가 가장 큰 것 선택
            # 이미 선택된 인덱스는 -1로 마킹해서 다시 뽑히지 않도록 처리
            min_dists[selected_local_indices] = -1.0
            next_idx_local = int(np.argmax(min_dists))
            selected_local_indices.append(next_idx_local)

            # 새로 선택된 포인트와의 거리로 min_dists 업데이트
            new_dists = np.linalg.norm(
                cam_centers - cam_centers[next_idx_local:next_idx_local + 1], axis=1
            )
            # 아직 선택되지 않은 위치에 대해서만 최소 거리 갱신
            mask = min_dists >= 0.0
            min_dists[mask] = np.minimum(min_dists[mask], new_dists[mask])

        # local index -> 원래 scene 카메라 인덱스로 매핑
        selected_indices = [candidate_indices[i] for i in selected_local_indices]
        return selected_indices

    def _select_cameras_by_rows(self, candidate_indices, num_cameras):
        """
        y축으로 4등분해서 각 row마다 균등한 개수의 카메라를 선택
        
        Args:
            candidate_indices: 선택 가능한 카메라 인덱스 리스트 또는 range 객체
            num_cameras: 선택할 총 카메라 수
        
        Returns:
            선택된 카메라 인덱스 리스트
        """
        # range 객체를 리스트로 변환
        candidate_indices = list(candidate_indices)
        
        if len(candidate_indices) == 0:
            return []
        
        # 모든 카메라의 중심 위치 가져오기
        cam_centers = []
        for idx in candidate_indices:
            cam = self.scene.cameras[idx]
            center = cam.camera_center
            # torch.Tensor를 numpy로 변환
            if isinstance(center, torch.Tensor):
                center = center.detach().cpu().numpy()
            cam_centers.append(center)
        
        cam_centers = np.array(cam_centers)  # shape: (N, 3)
        
        # y 좌표의 최소값과 최대값 계산
        min_y = np.min(cam_centers[:, 1])
        max_y = np.max(cam_centers[:, 1])

        # y축 기준으로 정렬한 인덱스들
        sorted_indices = np.argsort(cam_centers[:, 1])
        sorted_candidate_indices = [candidate_indices[i] for i in sorted_indices]
        print(f"sorted_candidate_indices: {sorted_candidate_indices}")
        print(f"cam_centers: {cam_centers}")
        
        # y축을 4등분하는 경계값 계산
        y_range = max_y - min_y
        y_boundary_1 = min_y + y_range * 0.25  # 25% 지점
        y_boundary_2 = min_y + y_range * 0.5   # 50% 지점 (중앙)
        y_boundary_3 = min_y + y_range * 0.75  # 75% 지점
        
        # 4개 row로 분류
        # row 0: y < y_boundary_1 (가장 아래)
        # row 1: y_boundary_1 <= y < y_boundary_2
        # row 2: y_boundary_2 <= y < y_boundary_3
        # row 3: y >= y_boundary_3 (가장 위)
        rows = {
            0: [],  # 가장 아래
            1: [],  # 아래쪽 중간
            2: [],  # 위쪽 중간
            3: [],  # 가장 위
        }
        
        for idx, center in zip(candidate_indices, cam_centers):
            y = center[1]
            if y < y_boundary_1:
                rows[0].append(idx)
            elif y < y_boundary_2:
                rows[1].append(idx)
            elif y < y_boundary_3:
                rows[2].append(idx)
            else:  # y >= y_boundary_3
                rows[3].append(idx)
        
        # 각 row에서 선택할 카메라 수 계산
        cameras_per_row = num_cameras // 4
        remainder = num_cameras % 4
        
        selected_indices = []
        
        # 각 row에서 균등하게 선택
        for row_idx in range(4):
            row_candidates = rows[row_idx]
            if len(row_candidates) == 0:
                continue
            
            # 나머지가 있으면 처음 4개 row에 1개씩 추가
            num_to_select = cameras_per_row + (1 if row_idx < remainder else 0)
            num_to_select = min(num_to_select, len(row_candidates))
            
            if num_to_select > 0:
                selected = random.sample(row_candidates, num_to_select)
                selected.sort()  # 각 row별로 정렬
                selected_indices.extend(selected)
        
        return selected_indices

    def _generate_cameras_by_rows(self):
        """
        4개 row × 각 5개의 카메라(총 20개)를 기존 카메라 분포를 기준으로 생성.
        - y축을 값 기준으로 4등분해서 row를 나눔
        - 각 row 안에서 기존 카메라의 x 범위에서 5개 위치를 샘플
        - 회전(R)은 해당 row의 '대표 카메라'에서 그대로 가져오고, 위치(translation)만 바꿈
        => GS/Colmap 좌표계 convention을 유지하므로 검정 화면 문제를 피함
        """
        from gaussiansplatting.scene.cameras import C2W_Camera
        import numpy as np
        import torch

        # 1. 기존 카메라 인덱스와 center 수집
        indices = list(range(len(self.scene.cameras)))
        centers = []
        for idx in indices:
            c = self.scene.cameras[idx].camera_center
            if isinstance(c, torch.Tensor):
                c = c.detach().cpu().numpy()
            centers.append(c)
        centers = np.array(centers)  # (N, 3)

        # 2. y축으로 4등분
        min_y = np.min(centers[:, 1])
        max_y = np.max(centers[:, 1])
        y_range = max_y - min_y
        y_b1 = min_y + 0.25 * y_range
        y_b2 = min_y + 0.50 * y_range
        y_b3 = min_y + 0.75 * y_range

        rows = {0: [], 1: [], 2: [], 3: []}  # 각 row에 카메라 인덱스 저장
        for idx, c in zip(indices, centers):
            y = c[1]
            if y < y_b1:
                rows[0].append(idx)
            elif y < y_b2:
                rows[1].append(idx)
            elif y < y_b3:
                rows[2].append(idx)
            else:
                rows[3].append(idx)

        generated_cameras = []

        height = self.scene.cameras[0].image_height
        width = self.scene.cameras[0].image_width
        fovy = self.scene.cameras[0].FoVy

        # 3. 각 row마다 5개씩 생성
        for r in range(4):
            row_idx_list = rows[r]
            if len(row_idx_list) == 0:
                continue

            row_centers = centers[row_idx_list]  # (Nr, 3)

            # x 범위, y/z 대표값
            x_min, x_max = row_centers[:, 0].min(), row_centers[:, 0].max()
            y_med = np.median(row_centers[:, 1])
            z_med = np.median(row_centers[:, 2])

            x_samples = np.linspace(x_min, x_max, 5)

            # 이 row의 '대표 카메라' 하나 선택 (중앙 인덱스)
            ref_idx = row_idx_list[len(row_idx_list) // 2]
            ref_cam = self.scene.cameras[ref_idx]

            # ref_cam 의 world_view_transform 을 이용해 c2w 추출
            # (Graphdeco convention: c2w = inv(world_view_transform^T))
            ref_wv = ref_cam.world_view_transform  # 4x4
            ref_c2w = torch.inverse(ref_wv.T).detach().cpu().numpy()

            # ref 카메라의 기존 center (검증용)
            ref_center = ref_c2w[:3, 3].copy()

            for x in x_samples:
                new_c2w = ref_c2w.copy()
                new_c2w[:3, 3] = np.array([x, y_med, z_med], dtype=np.float32)

                c2w_tensor = torch.from_numpy(new_c2w).float()

                new_cam = C2W_Camera(
                    c2w=c2w_tensor,
                    FoVy=fovy,
                    height=height,
                    width=width,
                    data_device="cuda",
                )
                generated_cameras.append(new_cam)

        # 4. scene 에 append 하고 인덱스 반환
        start_idx = len(self.scene.cameras)
        self.scene.cameras.extend(generated_cameras)
        end_idx = len(self.scene.cameras)

        camera_indices = list(range(start_idx, end_idx))
        return generated_cameras, camera_indices


    def estimate_radius_from_cameras(self, quantile: float = 0.8) -> float:
        """
        기존 scene.cameras의 camera_center 분포에서 적당한 반지름을 추정.
        - 너무 작은 반지름이면 오ブ젝트에 너무 가까워서 깨질 수 있어서
        상위 quantile 쪽 거리(예: 80% 지점)를 사용.
        """
        cam_centers = []
        for cam in self.scene.cameras:
            c = cam.camera_center
            if isinstance(c, torch.Tensor):
                c = c.detach().cpu().numpy()
            cam_centers.append(c)
        cam_centers = np.array(cam_centers)  # (N,3)

        # 원점으로부터의 거리
        dists = np.linalg.norm(cam_centers, axis=1)
        radius = float(np.quantile(dists, quantile))  # 예: 80% 지점
        return radius

    def get_gaussian_center_from_model(self) -> Optional[np.ndarray]:
        """
        가우시안 모델 객체에서 직접 xyz 좌표를 읽어서 중심 좌표를 계산.
        
        Returns:
            가우시안들의 중심 좌표 (3D numpy array) 또는 None
        """
        if self.gaussian_model is None:
            return None
        
        try:
            # get_xyz는 property로 정의되어 있음
            xyz = self.gaussian_model.get_xyz  # (N, 3) tensor
            
            if isinstance(xyz, torch.Tensor):
                xyz_np = xyz.detach().cpu().numpy()
            else:
                xyz_np = np.array(xyz)
            
            # 중심 좌표 계산 (평균)
            gaussian_center = np.mean(xyz_np, axis=0)
            threestudio.info(f"Computed Gaussian center from model: {gaussian_center}")
            return gaussian_center.astype(np.float32)
        except Exception as e:
            threestudio.warn(f"Failed to get Gaussian center from model: {e}")
            return None

    def get_gaussian_center_from_ply(self) -> Optional[np.ndarray]:
        """
        PLY 파일에서 가우시안 위치를 읽어서 중심 좌표를 계산.
        
        Returns:
            가우시안들의 중심 좌표 (3D numpy array) 또는 None
        """
        source_path = self.cfg.source
        
        # 1. 학습된 가우시안 모델의 PLY 파일 찾기 시도
        # 일반적으로 model_path/point_cloud/iteration_X/point_cloud.ply에 있음
        # 하지만 model_path를 모르므로, source_path의 상위 디렉토리나 일반적인 위치를 확인
        possible_ply_paths = []
        
        # 학습된 모델 경로 시도 (일반적인 구조)
        if os.path.exists(source_path):
            parent_dir = os.path.dirname(source_path)
            # output/xxx/point_cloud/iteration_XXX/point_cloud.ply 형태를 찾기
            if os.path.exists(parent_dir):
                for item in os.listdir(parent_dir):
                    potential_model_path = os.path.join(parent_dir, item, "point_cloud")
                    if os.path.exists(potential_model_path):
                        # 최신 iteration 찾기
                        iterations = []
                        for iter_dir in os.listdir(potential_model_path):
                            if iter_dir.startswith("iteration_"):
                                try:
                                    iter_num = int(iter_dir.split("_")[1])
                                    iterations.append((iter_num, iter_dir))
                                except:
                                    pass
                        if iterations:
                            latest_iter = max(iterations, key=lambda x: x[0])[1]
                            ply_path = os.path.join(potential_model_path, latest_iter, "point_cloud.ply")
                            if os.path.exists(ply_path):
                                possible_ply_paths.append(ply_path)
        
        # 2. 초기 point cloud 파일 시도
        initial_ply_path = os.path.join(source_path, "sparse", "0", "points3D.ply")
        if os.path.exists(initial_ply_path):
            possible_ply_paths.append(initial_ply_path)
        
        # 3. 다른 일반적인 위치들
        other_paths = [
            os.path.join(source_path, "points3d.ply"),
            os.path.join(source_path, "sparse", "points3D.ply"),
        ]
        for path in other_paths:
            if os.path.exists(path):
                possible_ply_paths.append(path)
        
        # PLY 파일 읽기 시도
        for ply_path in possible_ply_paths:
            try:
                plydata = PlyData.read(ply_path)
                vertices = plydata['vertex']
                
                # 가우시안 위치 추출
                xyz = np.stack([
                    np.asarray(vertices['x']),
                    np.asarray(vertices['y']),
                    np.asarray(vertices['z'])
                ], axis=1)
                
                # 중심 좌표 계산 (평균)
                gaussian_center = np.mean(xyz, axis=0)
                threestudio.info(f"Loaded Gaussian center from PLY file: {ply_path}")
                return gaussian_center.astype(np.float32)
            except Exception as e:
                threestudio.warn(f"Failed to load PLY file {ply_path}: {e}")
                continue
        
        # 모든 시도 실패 시 None 반환
        threestudio.warn("Could not find any PLY file to compute Gaussian center. Falling back to camera center median.")
        return None


    def _generate_spherical_novel_cameras(self,
        n_azimuth: int = 24,
        elevations: list[float] = [0.0, 15.0],
        radius: float | None = None,
        device: str = "cuda",
    ):
        """
        3DGS 좌표계 기준 spherical novel view들을 생성.
        - azimuth: 0~360도를 균일하게 샘플
        - elevations: 여러 고도(각도)에서 링을 여러 개 생성
        - radius: None이면 기존 카메라들에서 추정
        """
        if radius is None:
            radius = self.estimate_radius_from_cameras()

        height = self.scene.cameras[0].image_height
        width = self.scene.cameras[0].image_width
        fovy = self.scene.cameras[0].FoVy

        from gaussiansplatting.scene.cameras import C2W_Camera
        novel_cams: list[C2W_Camera] = []

        for phi in elevations:            # 고도
            for i in range(n_azimuth):    # 방위각
                theta = 360.0 * i / n_azimuth  # [0, 360)
                c2w = pose_spherical(theta, phi, radius)  # 이미 정의된 함수 사용
                c2w = c2w.to(device)

                cam = C2W_Camera(
                    c2w=c2w,
                    FoVy=fovy,
                    height=height,
                    width=width,
                    data_device=device,
                )
                novel_cams.append(cam)

        start_idx = len(self.scene.cameras)
        self.scene.cameras.extend(novel_cams)
        end_idx = len(self.scene.cameras)
        camera_indices = list(range(start_idx, end_idx))
        return novel_cams, camera_indices

    def _select_cameras_by_depth(self):
        """
        Select cameras based on distance from source_masked_center.
        Returns the 20 cameras with the largest distance.
        
        Returns:
            List of selected camera indices
        """
        device = "cuda"
        source_masked_center = torch.tensor([1.4750, 2.2077, 7.0923], device=device)
        cam_centers = []
        for cam in self.scene.cameras:
            c = cam.camera_center
            if isinstance(c, torch.Tensor):
                c = c.detach().to(device)
            else:
                c = torch.tensor(c, device=device, dtype=torch.float32)
            cam_centers.append(c)
        cam_centers = torch.stack(cam_centers, dim=0)  # (N,3)

        dists = torch.norm(cam_centers - source_masked_center[None, :], dim=1)
        sorted_indices = torch.argsort(dists).cpu().numpy()
        selected_indices = sorted_indices[-20:]
        return list(selected_indices)

    def _select_cameras_by_y_axis(self, num_cameras):
        """
        Select cameras based on y-axis value (highest y values).
        
        Args:
            num_cameras: Number of cameras to select (default: 20)
        
        Returns:
            List of selected camera indices
        """
        device = "cuda"
        cam_centers = []
        for cam in self.scene.cameras:
            c = cam.camera_center
            if isinstance(c, torch.Tensor):
                c = c.detach().to(device)
            else:
                c = torch.tensor(c, device=device, dtype=torch.float32)
            cam_centers.append(c)
        cam_centers = torch.stack(cam_centers, dim=0)  # (N,3)

        # Select by y-axis value (highest y values)
        y_values = cam_centers[:, 1]  # y-axis is index 1
        sorted_indices = torch.argsort(y_values, descending=True).cpu().numpy()
        selected_indices = sorted_indices[:num_cameras]
        
        return list(selected_indices)

    def _load_camera_image(self, cam, cam_idx: int = 0):
        """
        Load image from camera object.
        Supports both Camera (with original_image) and Simple_Camera (with image_name).
        
        Args:
            cam: Camera object (Camera, Simple_Camera, or C2W_Camera)
            cam_idx: Camera index for logging purposes
        
        Returns:
            torch.Tensor: Image tensor in format (1, H, W, C) normalized to [0, 1], or None if failed
        """
        # Try Camera class with original_image
        if hasattr(cam, 'original_image'):
            image_to_segment = cam.original_image.permute(1, 2, 0).unsqueeze(0)  # (1, H, W, C)
            if image_to_segment.max() > 1.0:
                image_to_segment = image_to_segment / 255.0
            image_to_segment = image_to_segment.clamp(0.0, 1.0).to(get_device())
            return image_to_segment
        
        # Try Simple_Camera with image_name
        elif hasattr(cam, 'image_name'):
            from PIL import Image
            from gaussiansplatting.utils.general_utils import PILtoTorch
            
            # Construct image path from source path
            source_path = self.cfg.source
            images_folder = os.path.join(source_path, "images")
            if not os.path.exists(images_folder):
                # Try alternative paths
                images_folder = source_path
            
            # Find image file (try common extensions)
            image_name = cam.image_name
            image_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                potential_path = os.path.join(images_folder, image_name + ext)
                if os.path.exists(potential_path):
                    image_path = potential_path
                    break
                # Also try with image_name as full filename
                potential_path = os.path.join(images_folder, image_name)
                if os.path.exists(potential_path):
                    image_path = potential_path
                    break
            
            if image_path and os.path.exists(image_path):
                try:
                    pil_image = Image.open(image_path)
                    # Resize to camera resolution
                    h = cam.image_height if hasattr(cam, 'image_height') else 512
                    w = cam.image_width if hasattr(cam, 'image_width') else 512
                    resized_image_rgb = PILtoTorch(pil_image, (w, h))
                    gt_image = resized_image_rgb[:3, ...]  # (C, H, W)
                    image_to_segment = gt_image.permute(1, 2, 0).unsqueeze(0)  # (1, H, W, C)
                    
                    if image_to_segment.max() > 1.0:
                        image_to_segment = image_to_segment / 255.0
                    image_to_segment = image_to_segment.clamp(0.0, 1.0).to(get_device())
                    return image_to_segment
                except Exception as e:
                    threestudio.warn(f"Failed to load image from {image_path}: {e}")
            else:
                threestudio.warn(f"Could not find image file for camera {cam_idx} with image_name: {image_name}")
        
        return None

    def _compute_region_aware_scores(self, candidate_indices, cam_centers, cam_centers_np):
        """
        Compute region-aware relevance scores for cameras.
        This is the shared logic used by both MMR and region-aware-only selection.
        
        Args:
            candidate_indices: List of camera indices
            cam_centers: Tensor of camera centers (N, 3)
            cam_centers_np: Numpy array of camera centers (N, 3)
        
        Returns:
            relevance_scores: Normalized relevance scores (N,)
        """
        import numpy as np
        import torch
        
        device = cam_centers.device if isinstance(cam_centers, torch.Tensor) else "cuda"
        
        # Step 1: Single-view segmentation from representative view v_r
        rep_view_idx = self.cfg.mmr_representative_view_idx
        if rep_view_idx >= len(candidate_indices):
            rep_view_idx = 0  # Fallback to first view
        
        rep_cam = self.scene.cameras[rep_view_idx]
        
        # Get segmentation mask (A_obj area)
        A_obj = 0.0
        A_img = 1.0
        
        # Determine which segmentation prompt to use
        seg_prompt = None
        if self.cfg.mmr_seg_prompt and len(self.cfg.mmr_seg_prompt.strip()) > 0:
            seg_prompt = self.cfg.mmr_seg_prompt
        elif self.system_seg_prompt and len(self.system_seg_prompt.strip()) > 0:
            seg_prompt = self.system_seg_prompt
        
        # Try text-based segmentation first if prompt is provided
        mask_np = None
        if seg_prompt:
            try:
                # Load image from camera using common function
                image_to_segment = self._load_camera_image(rep_cam, rep_view_idx)
                
                if image_to_segment is not None:
                    # Perform text-based segmentation
                    text_segmentor = LangSAMTextSegmentor().to(get_device())
                    with torch.no_grad():
                        mask_tensor = text_segmentor(image_to_segment, seg_prompt)[0]
                    
                    # Convert mask to numpy
                    mask_np = mask_tensor[0].detach().cpu().numpy()  # (H, W)
                    
                    threestudio.info(f"Text-based segmentation successful for prompt: {seg_prompt}")
                else:
                    threestudio.warn(f"Could not load image for camera {rep_view_idx}, falling back to gt_alpha_mask or full image")
            except Exception as e:
                threestudio.warn(f"Text-based segmentation failed: {e}, falling back to gt_alpha_mask or full image")
        
        # Fallback to gt_alpha_mask if text segmentation failed or not attempted
        if mask_np is None and self.cfg.mmr_use_gt_mask and hasattr(rep_cam, 'gt_alpha_mask') and rep_cam.gt_alpha_mask is not None:
            mask = rep_cam.gt_alpha_mask
            if isinstance(mask, torch.Tensor):
                mask_np = mask.detach().cpu().numpy()
            else:
                mask_np = np.array(mask)
            mask_np = mask_np.squeeze()
        
        # Process mask if we have one
        if mask_np is not None:
            if mask_np.ndim > 2:
                mask_np = mask_np.squeeze()
            
            A_obj = float(np.sum(mask_np > 0.5))
            if hasattr(rep_cam, 'image_height') and hasattr(rep_cam, 'image_width'):
                A_img = float(rep_cam.image_height * rep_cam.image_width)
            else:
                A_img = float(mask_np.size)
            
            if A_obj > 0 and mask_np.ndim == 2:
                y_coords, x_coords = np.where(mask_np > 0.5)
                if len(y_coords) > 0:
                    obj_centroid_y = float(np.mean(y_coords))
                    obj_mask_height = float(np.max(y_coords) - np.min(y_coords)) if len(y_coords) > 1 else 1.0
                else:
                    obj_centroid_y = mask_np.shape[0] / 2.0
                    obj_mask_height = mask_np.shape[0]
            else:
                obj_centroid_y = mask_np.shape[0] / 2.0 if mask_np.ndim >= 1 else 0.5
                obj_mask_height = mask_np.shape[0] if mask_np.ndim >= 1 else 1.0
        else:
            # Fallback: assume full image as object
            if hasattr(rep_cam, 'image_height') and hasattr(rep_cam, 'image_width'):
                A_img = float(rep_cam.image_height * rep_cam.image_width)
                A_obj = A_img
                obj_centroid_y = float(rep_cam.image_height) / 2.0
                obj_mask_height = float(rep_cam.image_height)
            else:
                A_img = 1.0
                A_obj = 1.0
                obj_centroid_y = 0.5
                obj_mask_height = 1.0
        
        # Calculate coverage ratio γ_r
        gamma_r = A_obj / A_img if A_img > 0 else 0.0
        
        # Estimate or use provided object center
        if self.cfg.mmr_object_center is None:
            gaussian_center = self.get_gaussian_center_from_model()
            if gaussian_center is None:
                gaussian_center = self.get_gaussian_center_from_ply()
            
            if gaussian_center is not None:
                object_center = gaussian_center
            else:
                object_center = np.median(cam_centers_np, axis=0)
        else:
            object_center = np.array(self.cfg.mmr_object_center, dtype=np.float32)
        
        object_center_tensor = torch.tensor(object_center, device=device, dtype=torch.float32)
        
        # Calculate d_r: distance from representative view camera center to object center
        rep_cam_center = cam_centers[rep_view_idx]
        d_r = abs(torch.norm(rep_cam_center - object_center_tensor).item())
        
        # Calculate κ = γ_r * d_r
        kappa = abs(gamma_r * d_r)
        
        # Compute target depth and height
        target_coverage = self.cfg.mmr_target_coverage
        if target_coverage > 0:
            target_depth = abs(kappa / target_coverage)
        else:
            target_depth = d_r
        
        # Map object centroid y-coordinate to camera height
        if hasattr(rep_cam, 'image_height'):
            img_height = float(rep_cam.image_height)
            normalized_y = obj_centroid_y / img_height if img_height > 0 else 0.5
            y_min = float(np.min(cam_centers_np[:, 1]))
            y_max = float(np.max(cam_centers_np[:, 1]))
            y_range = y_max - y_min if y_max > y_min else 1.0
            target_height = y_min + normalized_y * y_range
        else:
            target_height = float(np.median(cam_centers_np[:, 1]))
        
        # Compute depth to object center for all cameras
        depths = torch.norm(cam_centers - object_center_tensor.unsqueeze(0), dim=1)
        depths_np = np.abs(depths.detach().cpu().numpy())
        
        # Compute camera height (y coordinate)
        heights = cam_centers[:, 1].detach().cpu().numpy()
        
        # Compute depth/height matching scores
        sigma_d = self.cfg.mmr_sigma_d
        sigma_h = self.cfg.mmr_sigma_h
        
        depth_scores = np.exp(-np.abs(depths_np - target_depth) / sigma_d)
        height_scores = np.exp(-np.abs(heights - target_height) / sigma_h)
        
        # Combine into region-aware relevance score
        alpha_d = self.cfg.mmr_alpha_d
        alpha_h = self.cfg.mmr_alpha_h
        relevance_scores = alpha_d * depth_scores + alpha_h * height_scores
        
        # Normalize to [0, 1]
        if relevance_scores.max() > relevance_scores.min():
            relevance_scores = (relevance_scores - relevance_scores.min()) / (relevance_scores.max() - relevance_scores.min())
        
        return relevance_scores

    def _select_cameras_by_mmr(self, num_cameras: int):
        """
        Region-Aware and Redundancy-Aware (MMR) View Selection.
        
        Selects cameras based on:
        1. Region-aware scores (depth and height matching)
        2. Redundancy-aware diversity (position and direction similarity)
        
        Args:
            num_cameras: Number of cameras to select (budget M)
        
        Returns:
            List of selected camera indices
        """
        import numpy as np
        import torch
        
        candidate_indices = list(range(self.total_view_num))
        if len(candidate_indices) == 0:
            return []
        if len(candidate_indices) <= num_cameras:
            return candidate_indices
        
        device = "cuda"
        
        # Extract camera centers and viewing directions
        cam_centers = []
        viewing_directions = []
        for idx in candidate_indices:
            cam = self.scene.cameras[idx]
            center = cam.camera_center
            if isinstance(center, torch.Tensor):
                center = center.detach().to(device)
            else:
                center = torch.tensor(center, device=device, dtype=torch.float32)
            cam_centers.append(center)
            
            # Extract viewing direction from camera (forward vector)
            # In camera coordinate system, forward is typically -z
            # We need to transform to world coordinates using c2w
            forward = None
            if hasattr(cam, 'c2w'):
                # C2W_Camera has direct c2w attribute
                c2w = cam.c2w
                if isinstance(c2w, torch.Tensor):
                    forward = c2w[:3, 2].to(device)
                else:
                    forward = torch.tensor(c2w[:3, 2], device=device, dtype=torch.float32)
            elif hasattr(cam, 'world_view_transform'):
                # Standard camera: compute c2w from world_view_transform
                wv = cam.world_view_transform  # 4x4
                if isinstance(wv, torch.Tensor):
                    wv = wv.to(device)
                else:
                    wv = torch.tensor(wv, device=device, dtype=torch.float32)
                c2w = torch.inverse(wv.T)
                # Forward vector in world coordinates (third column of rotation matrix)
                forward = c2w[:3, 2]
            
            if forward is None:
                # Fallback: use direction from center to origin (looking at origin)
                forward = -safe_normalize(center.unsqueeze(0)).squeeze(0)
            
            # Normalize viewing direction
            forward_normalized = safe_normalize(forward.unsqueeze(0)).squeeze(0)
            viewing_directions.append(forward_normalized)
        
        cam_centers = torch.stack(cam_centers, dim=0)  # (N, 3)
        viewing_directions = torch.stack(viewing_directions, dim=0)  # (N, 3)
        
        # Convert to numpy for easier computation
        cam_centers_np = cam_centers.detach().cpu().numpy()
        viewing_directions_np = viewing_directions.detach().cpu().numpy()
        
        # Compute region-aware relevance scores (shared logic)
        relevance_scores = self._compute_region_aware_scores(candidate_indices, cam_centers, cam_centers_np)
        
        # Step 2: Define view similarity for redundancy measurement
        sigma_c = self.cfg.mmr_sigma_c
        sigma_theta = self.cfg.mmr_sigma_theta
        
        def compute_similarity(i, j):
            # Position similarity
            pos_diff = np.linalg.norm(cam_centers_np[i] - cam_centers_np[j])
            pos_sim = np.exp(-(pos_diff ** 2) / (2 * sigma_c ** 2))
            
            # Direction similarity using angular distance
            dot_product = np.clip(np.dot(viewing_directions_np[i], viewing_directions_np[j]), -1.0, 1.0)
            theta_ij = np.arccos(dot_product)
            dir_sim = np.exp(-(theta_ij ** 2) / (2 * sigma_theta ** 2))
            
            return pos_sim * dir_sim
        
        # Step 3: MMR Greedy selection
        # 3.1. Initialize with highest relevance score
        selected_indices = [int(np.argmax(relevance_scores))]
        
        # 3.2. Initialize redundancy cache
        redundancy_cache = np.zeros(len(candidate_indices))
        for i in range(len(candidate_indices)):
            if i not in selected_indices:
                redundancy_cache[i] = compute_similarity(i, selected_indices[0])
        
        # 3.3. Greedy selection
        lambda_param = self.cfg.mmr_lambda
        
        while len(selected_indices) < num_cameras:
            best_score = -np.inf
            best_idx = None
            
            for i in range(len(candidate_indices)):
                if i in selected_indices:
                    continue
                
                # MMR marginal score: λ * relevance - (1-λ) * redundancy
                mmr_score = lambda_param * relevance_scores[i] - (1 - lambda_param) * redundancy_cache[i]
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            if best_idx is None:
                break
            
            selected_indices.append(best_idx)
            
            # Update redundancy cache efficiently
            for i in range(len(candidate_indices)):
                if i not in selected_indices:
                    sim = compute_similarity(i, best_idx)
                    redundancy_cache[i] = max(redundancy_cache[i], sim)
        
        return selected_indices

    def _select_cameras_by_region_aware_only(self, num_cameras: int):
        """
        Region-Aware View Selection (without MMR diversity).
        
        Selects cameras based only on region-aware scores (depth and height matching).
        This is a simpler version that doesn't consider redundancy/diversity.
        
        Args:
            num_cameras: Number of cameras to select (budget M)
        
        Returns:
            List of selected camera indices
        """
        import numpy as np
        import torch
        
        candidate_indices = list(range(self.total_view_num))
        if len(candidate_indices) == 0:
            return []
        if len(candidate_indices) <= num_cameras:
            return candidate_indices
        
        device = "cuda"
        
        # Extract camera centers
        cam_centers = []
        for idx in candidate_indices:
            cam = self.scene.cameras[idx]
            center = cam.camera_center
            if isinstance(center, torch.Tensor):
                center = center.detach().to(device)
            else:
                center = torch.tensor(center, device=device, dtype=torch.float32)
            cam_centers.append(center)
        
        cam_centers = torch.stack(cam_centers, dim=0)  # (N, 3)
        
        # Convert to numpy for easier computation
        cam_centers_np = cam_centers.detach().cpu().numpy()
        
        # Compute region-aware relevance scores (shared logic)
        relevance_scores = self._compute_region_aware_scores(candidate_indices, cam_centers, cam_centers_np)
        
        # Select cameras with highest relevance scores (simple top-k selection)
        sorted_indices = np.argsort(relevance_scores)[::-1]  # Descending order
        selected_indices = sorted_indices[:num_cameras].tolist()
        
        return selected_indices


    def collate(self, batch) -> Dict[str, Any]:
        cam_list = []
        index_list = []
        for _ in range(self.batch_size):
            if not self.train_view_index_stack:
                self.train_view_index_stack = self.train_view_index.copy()

            view_index = random.choice(self.train_view_index_stack) # 하나의 뷰 인덱스 번호 선택
            self.train_view_index_stack.remove(view_index)
            cam_list.append(self.scene.cameras[view_index])
            index_list.append(view_index)

        return {
            "index": index_list, # 예시: [5, 12, 8] - 카메라 인덱스 번호들
            "camera": cam_list, # 예시: [Camera(...), Camera(...), Camera(...)] - 실제 카메라 객체들
            "height": self.height,
            "width": self.width,
        }

    # def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
    #     size_ind = bisect.bisect_right(self.resolution_milestones, global_step) - 1
    #     self.height = self.heights[size_ind]
    #     self.width = self.widths[size_ind]
    #     self.batch_size = self.batch_sizes[size_ind]
    #     # self.directions_unit_focal = self.directions_unit_focals[size_ind]
    #     threestudio.debug(
    #         f"Training height: {self.height}, width: {self.width}, batch_size: {self.batch_size}"
    #     )
    #     # progressive view
    #     self.progressive_view(global_step)

    def update_editing_cameras(self, random_seed: int = 0): # TODO: edit_view_index 먼저 고르고 train_view_index는 나머지 카메라 중에서 고르기
        random.seed(random_seed)

        self.train_view_index = random.sample(
            range(0, self.total_view_num),
            min(self.total_view_num, self.cfg.max_view_num),
        )
        self.train_view_index_stack = self.train_view_index.copy()

        # Forward facing scene에 대해 x, y 축 기준으로 2x2 = 4개 구획으로 나눠서 선택
        self.edit_view_index = self._select_cameras_by_quadrants(
            range(0, self.total_view_num),
            min(len(self.train_view_index), self.cfg.max_edit_view_num)
        )
        self.edit_view_index_stack = self.edit_view_index.copy()


    def __iter__(self):
        while True:
            yield {}



class GSLoadDataset(Dataset):
    def __init__(self, cfg, split, scene, train_view_list=None) -> None:
        super().__init__()
        self.cfg: GSLoadDataModuleConfig = cfg
        self.split = split
        self.scene = scene
        self.total_view_num = len(self.scene.cameras)

        if split == "val":
            self.n_views = self.cfg.n_val_views
            self.h = self.cfg.height
            self.w = self.cfg.width
        else:
            self.n_views = self.total_view_num
            self.h = self.cfg.eval_height if self.cfg.eval_height > 0 else self.scene.cameras[0].image_height
            self.w = self.cfg.eval_width if self.cfg.eval_width > 0 else self.scene.cameras[0].image_width

        if train_view_list is None:
            self.selected_views = torch.linspace(
                0, self.total_view_num - 1, self.n_views, dtype=torch.int
            )
        else:
            train_view_list = sorted(train_view_list)
            self.selected_views = torch.linspace(
                0, len(train_view_list) - 1, self.n_views, dtype=torch.int
            )
            self.selected_views = [train_view_list[idx] for idx in self.selected_views]

    def __len__(self):
        return self.n_views

    def __getitem__(self, index):
        return {
            "index": self.selected_views[index] if self.split == "val" else index,
            "height": self.h,
            "width": self.w,
        }

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.h, "width": self.w})
        return batch


@register("gs-load")
class GS_load(pl.LightningDataModule):
    cfg: GSLoadDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        from gaussiansplatting.scene.camera_scene import CamScene

        super().__init__()
        self.cfg = parse_structured(GSLoadDataModuleConfig, cfg)
        if self.cfg.use_original_resolution:
            self.cfg.height = self.cfg.eval_height
            self.cfg.width = self.cfg.eval_width
        
        self.train_scene = CamScene( # 전체 카메라를 로드
            self.cfg.source, h=self.cfg.height, w=self.cfg.width # Colmap_HW -> readColmapSceneInfo_hw
        )
        self.eval_scene = CamScene(
            self.cfg.source, h=self.cfg.eval_height, w=self.cfg.eval_width # Colmap -> readColmapSceneInfo
        )

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = GSLoadIterableDataset(self.cfg, self.train_scene)

        if stage in [None, "fit", "validate"]:
            self.val_dataset = GSLoadDataset(
                self.cfg, "val", self.eval_scene, self.train_dataset.edit_view_index
            )
        if stage in [None, "test", "predict"]:
            self.test_dataset = GSLoadDataset(self.cfg, "test", self.eval_scene)

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            # very important to disable multi-processing if you want to change self attributes at runtime!
            # (for example setting self.width and self.height in update_step)
            num_workers=0,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset,
            batch_size=1,
            collate_fn=self.val_dataset.collate,
        )
        # return self.general_loader(self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )
