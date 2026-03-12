#!/usr/bin/env python3
"""
WandB Sweep for DGE camera-selection hyperparameter search.

Sweep target metric: CLIP directional similarity (maximize)
Search space:
  - lambda_d:                    [1.0,  5.0, 10.0]
  - edit_view_selection_strategy: [lens, random]
  - lambda_ism:                  [0.0, 0.00001, 0.0001]
  - task:                        [smile, mustache, leather_jacket, mondigliani, pixar, clown]  (prompt combos from TASKS)

Usage:
  # 1) Create sweep (run once)
  python script/camera-selection/3d-ovs/sweep_3d_ovs_v1.py --create

  # 2) Run agent(s) – each agent picks a config and runs train+metrics
  python script/camera-selection/3d-ovs/sweep_3d_ovs_v1.py --agent --sweep_id <SWEEP_ID>

  # Use specific GPU(s): single GPU or comma-separated for multi-GPU per run
  python script/camera-selection/3d-ovs/sweep_3d_ovs_v1.py --agent --sweep_id mt4ttadh --gpu 0
  python script/camera-selection/3d-ovs/sweep_3d_ovs_v1.py --agent --sweep_id <ID> --gpu 0,1,2

  # Run multiple agents in parallel (one process per GPU, single command): ## 이거로!!!
  python script/camera-selection/3d-ovs/sweep_3d_ovs_v1.py --agent --sweep_id <ID> --gpus 0,1,2

  # Or manually in separate terminals:
  CUDA_VISIBLE_DEVICES=0 python script/camera-selection/3d-ovs/sweep_3d_ovs_v1.py --agent --sweep_id <ID> --gpu 0
  CUDA_VISIBLE_DEVICES=1 python script/camera-selection/3d-ovs/sweep_3d_ovs_v1.py --agent --sweep_id <ID> --gpu 1

  # Or do both in one shot (sequential, single machine):
  python script/camera-selection/3d`-ovs/sweep_3d_ovs_v1.py --create --agent
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import wandb

# ==========================
# Fixed configuration (same as launch_and_metrics.py)
# ==========================

CONFIG = "configs/dge_camera-selection.yaml"
GPU = "0,1,2,3"  # default; override with --gpu (e.g. "0" or "0,1,2" for multi-GPU per run)

DATA_TYPE = "3d-ovs"
DATA_NAME = "covered_desk"  # default; may be overridden per TASK via DATA_NAME
DATA_SOURCE_ROOT = "/data/users/jaeyeonpark/dataset"
GS_SOURCE_ROOT = "/data/users/jaeyeonpark/3dgs-trained"
RENDER_SUBDIR = "colmap_render_full"
GUIDANCE_SCALE = "12.5"
MASK_THRES = "0.6"
MAX_VIEW_NUM = "25"
MAX_EDIT_VIEW_NUM = "20"
CAMERA_UPDATE_PER_STEP = "1500"
MASK_UPDATE_AT_STEP = "600"
MASK_MAX_RATIO = "0.6"
MASK_MIN_RATIO = "0.01"
MASK_OUTLIER_IQR = "1.5"
MASK_UPDATE_VIEW_NUM = "30"
PRUNE_FLOATER_AT_STEP = "-1"
MAX_STEPS = "1500"

# (max_view_num, max_edit_view_num) 쌍 조합 — sweep에서 view_config로 선택
VIEW_CONFIG_PAIRS = [
    ("20", "20"),
    ("15", "15"),
    ("10", "10"),
    ("5", "5"),
]

LAMBDA_D_DEFAULT = "0.0"
LAMBDA_DDS = "0.0"
DDS_T_RANGE = "0.02,0.5"
DDS_CFG_SCALE = "7.5"
USE_SDS = False
USE_SDS_DGE = False
LAMBDA_SDS = "0.0"

FEATURE_INJECTION_MODE = "similarity" # `"similarity"`(내 논문) , `"3d_anchor"` | 3d_anchor 시 `injection_3d_anchor_style` (`"blend"` | `"gather"`).
MULTIVIEW_EDIT_KEY_SELECTION_STRATEGY = "lens_fps"
USE_MULTIVIEW_EDIT_DEFAULT = True
USE_GAUSSIAN_PROVENANCE_DEFAULT = False

LENS_USE_IP2P_SCORING = "true"
LENS_ENTROPY_THRESH = "0.97"
LENS_IP2P_STEPS = "5"
LENS_IP2P_GUIDANCE_SCALE = "12.5"
LENS_IP2P_IMAGE_GUIDANCE_SCALE = "1.5"
LENS_DISTANCE_MULTIPLIERS = "2.0,2.5,3.0,4.0,5.0,6.0"
LENS_CONE_HALF_ANGLE_DEG = "60"
LENS_LAMBDA_LEAK = "1.5"
LENS_LAMBDA_ENT = "15.0"
LENS_HEMISPHERE_ONLY = "true"
LENS_V_FRONT_METHOD = "scene_center"
LENS_N_CANDIDATES = "900"
LENS_DIVERSITY_X_WEIGHT = "30.0"
LENS_DIVERSITY_Y_VARIANCE_WEIGHT = "3.0"
LENS_IP2P_BATCH_SIZE = "2"
CAMERA_BATCH_SIZE = "5"

LENS_DISTANCE_MULTIPLIERS_DEFAULT = "2.0,2.5,3.0,4.0,5.0,6.0"

EDIT_VIEW_SELECTION_STRATEGY_DEFAULT = "lens"


INTERVAL = 1
DEVICE = "cuda"
LAMBDA_ISM_DEFAULT = 0.0
USE_WARP_REFINE_DEFAULT = False
WARP_REFINE_COLOR_FIT_STEPS_DEFAULT = 100 

# Extended-attention & target batch defaults (launch 설정과 동일하게 맞춤)
SAVE_IMAGE_GRID_DRAW_TEXTS = "false"
TARGET_USE_EXTENDED_ATTENTION = "true"
PER_STEP_CROSS_ATTN_CONSISTENCY = "true"
PER_STEP_CROSS_ATTN_T_START = "500"
PER_STEP_CROSS_ATTN_RESOLUTIONS = "1024"  # passed as [1024]
TARGET_BATCH_STRATEGY = "adaptive"
TARGET_BATCH_NEIGHBOR_THRESHOLD = "900"
TARGET_BATCH_LATE_MODE = "sliding_window"
TARGET_BATCH_SLIDING_STRIDE = "5"
TARGET_KEY_SELECTION_MODE = "canonical_progressive"

# Mask view population defaults
MASK_VIEW_POPULATION = "train_cameras"
MASK_NUM_VIEWS = "30"

# 출력 루트: 여기 아래에 sweep/3d-ovs/... 가 생성됨 (dge.yaml exp_root_dir 오버라이드용)
EXP_ROOT_DIR = "/data/users/jaeyeonpark/DGE-ours-outputs"

WANDB_PROJECT = "dge-ours-3d-ovs"
WANDB_SWEEP_NAME = f"sweep/{DATA_TYPE}"

# ==========================
# Prompt combinations for 3d-ovs; sweep can vary over these via "task" parameter
# (각 TASK는 launch_and_metrics.py의 PROMPT / SEG_PROMPT / TARGET_PROMPT 조합을 반영)
# ==========================

# STYLE_SOURCE_PROMPT = 편집 전(원본), STYLE_TARGET_PROMPT = 편집 후(목표)
TASKS = [
    ## covered_desk
    # 1) Change the shaving razor into an apple
    # {
    #     "name": "razor_to_apple",
    #     "DATA_NAME": "covered_desk",
    #     "PROMPT": "Change the shaving razor into an apple",
    #     "SEG_PROMPT": "shaving razor",
    #     "TARGET_PROMPT": "apple",
    #     "STYLE_SOURCE_PROMPT": "shaving razor",
    #     "STYLE_TARGET_PROMPT": "apple",
    # },
    # 2) Change the shampoo bottle into an apple
    # {
    #     "name": "bottle_to_apple",
    #     "DATA_NAME": "covered_desk",
    #     "PROMPT": "Change the shampoo bottle into an apple",
    #     "SEG_PROMPT": "shampoo bottle",
    #     "TARGET_PROMPT": "apple",
    #     "STYLE_SOURCE_PROMPT": "shampoo bottle",
    #     "STYLE_TARGET_PROMPT": "apple",
    # },
    # # 3) Give the pooh a pair of pants
    # {
    #     "name": "pooh_pants",
    #     "DATA_NAME": "covered_desk",
    #     "PROMPT": "Give the pooh a pair of pants",
    #     "SEG_PROMPT": "pooh",
    #     "TARGET_PROMPT": "pants",
    #     "STYLE_SOURCE_PROMPT": "pooh",
    #     "STYLE_TARGET_PROMPT": "pooh wearing pants",
    # },
    # 4) Make the pooh look like a penguin
    # {
    #     "name": "pooh_penguin",
    #     "DATA_NAME": "covered_desk",
    #     "PROMPT": "Make the pooh look like a penguin",
    #     "SEG_PROMPT": "pooh",
    #     "LENS_SEG_PROMPT": "pooh",
    #     "LENS_DISTANCE_MULTIPLIERS": "4.0",
    #     "TARGET_PROMPT": "penguin",
    #     "STYLE_SOURCE_PROMPT": "pooh",
    #     "STYLE_TARGET_PROMPT": "penguin",
    # },
    # # 5) Change the red sweater into a leather jacket
    # {
    #     "name": "sweater_to_leather_jacket",
    #     "DATA_NAME": "covered_desk",
    #     "PROMPT": "Change the red sweater into a leather jacket",
    #     "SEG_PROMPT": "red sweater of the pooh",
    #     "LENS_SEG_PROMPT": "red sweater of the pooh",
    #     "LENS_DISTANCE_MULTIPLIERS": "4.0",
    #     "TARGET_PROMPT": "leather jacket of the pooh",
    #     "STYLE_SOURCE_PROMPT": "red sweater of the pooh",
    #     "STYLE_TARGET_PROMPT": "leather jacket of the pooh",
    # },
    # # 6) Make the pooh wear sunglasses on his face
    # {
    #     "name": "pooh_sunglasses",
    #     "DATA_NAME": "covered_desk",
    #     "PROMPT": "Make the pooh wear sunglasses on his face",
    #     "SEG_PROMPT": " pooh",
    #     "LENS_SEG_PROMPT": "pooh",
    #     "LENS_DISTANCE_MULTIPLIERS": "7.0",
    #     "TARGET_PROMPT": "sunglasses",
    #     "STYLE_SOURCE_PROMPT": "head of the pooh",
    #     "STYLE_TARGET_PROMPT": "pooh wearing sunglasses",
    # },
    # # 7) Change the pooh's sweater color to blue
    # {
    #     "name": "sweater_blue",
    #     "DATA_NAME": "covered_desk",
    #     "PROMPT": "Change the pooh's sweater color to blue",
    #     "SEG_PROMPT": "red sweater of the pooh",
    #     "LENS_SEG_PROMPT": "red sweater of the pooh",
    #     "LENS_DISTANCE_MULTIPLIERS": "4.0",
    #     "TARGET_PROMPT": "blue sweater of the pooh",
    #     "STYLE_SOURCE_PROMPT": "red sweater of the pooh",
    #     "STYLE_TARGET_PROMPT": "blue sweater of the pooh",
    # },
    # # 8) Add flower pattern to the pooh's sweater
    # {
    #     "name": "sweater_flower",
    #     "DATA_NAME": "covered_desk",
    #     "PROMPT": "Add flower pattern to the pooh's sweater",
    #     "SEG_PROMPT": "red sweater of the pooh",
    #     "LENS_SEG_PROMPT": "red sweater of the pooh",
    #     "LENS_DISTANCE_MULTIPLIERS": "6.0",
    #     "TARGET_PROMPT": "sweater of the pooh with flower pattern",
    #     "STYLE_SOURCE_PROMPT": "red sweater of the pooh",
    #     "STYLE_TARGET_PROMPT": "sweater of the pooh with flower pattern",
    # },
    # # 9) Make the pooh look like a panda
    # {
    #     "name": "pooh_panda",
    #     "DATA_NAME": "covered_desk",
    #     "PROMPT": "Make the pooh look like a panda",
    #     "SEG_PROMPT": "pooh",
    #     "TARGET_PROMPT": "panda",
    #     "STYLE_SOURCE_PROMPT": "pooh",
    #     "STYLE_TARGET_PROMPT": "panda",
    # },
    # # 10) Make the pooh look like a robot
    # {
    #     "name": "pooh_robot",
    #     "DATA_NAME": "covered_desk",
    #     "PROMPT": "Make the pooh look like a robot",
    #     "SEG_PROMPT": "pooh",
    #     "TARGET_PROMPT": "robot",
    #     "STYLE_SOURCE_PROMPT": "pooh",
    #     "STYLE_TARGET_PROMPT": "robot",
    # },


    # ## blue_sofa
    # # 1) Change the plush toy's color to pink
    # {
    #     "name": "plush_pink",
    #     "DATA_NAME": "blue_sofa",
    #     "PROMPT": "Change the plush toy's color to pink",
    #     "SEG_PROMPT": "yellow plush toy",
    #     "TARGET_PROMPT": "pink plush toy",
    #     "STYLE_SOURCE_PROMPT": "yellow plush toy",
    #     "STYLE_TARGET_PROMPT": "pink plush toy",
    # },
    # # 2) Make the plush toy wear a tiny hat
    # # {
    # #     "name": "plush_hat",
    # #     "DATA_NAME": "blue_sofa",
    # #     "PROMPT": "Make the plush toy wear a tiny hat",
    # #     "SEG_PROMPT": "head of the plush toy",
    # #     "TARGET_PROMPT": "head of the plush toy wearing a tiny party hat",
    # #     "STYLE_SOURCE_PROMPT": "head of the plush toy",
    # #     "STYLE_TARGET_PROMPT": "head of the plush toy wearing a tiny party hat",
    # # },
    # # 3) Change the sunglasses to red frames
    # {
    #     "name": "glasses_red",
    #     "DATA_NAME": "blue_sofa",
    #     "PROMPT": "Change the sunglasses to red frames",
    #     "SEG_PROMPT": "black frames of the sunglasses",
    #     "LENS_SEG_PROMPT": "sunglasses",
    #     "LENS_DISTANCE_MULTIPLIERS": "5.0",
    #     "TARGET_PROMPT": "bright red frames of the sunglasses",
    #     "STYLE_SOURCE_PROMPT": "black frames of the sunglasses",
    #     "STYLE_TARGET_PROMPT": "bright red frames of the sunglasses",
    # },
    # # 4) Replace the JBL speaker with a vintage radio
    # {
    #     "name": "speaker_vintage",
    #     "DATA_NAME": "blue_sofa",
    #     "PROMPT": "Replace the JBL speaker with a vintage radio",
    #     "SEG_PROMPT": "grey JBL speaker",
    #     "TARGET_PROMPT": "vintage wooden radio",
    #     "STYLE_SOURCE_PROMPT": "grey JBL speaker",
    #     "STYLE_TARGET_PROMPT": "vintage wooden radio",
    # },
    # # 5) Change the perfume liquid color to blue
    # {
    #     "name": "perfume_blue",
    #     "DATA_NAME": "blue_sofa",
    #     "PROMPT": "Change the perfume liquid color to blue",
    #     "SEG_PROMPT": "yellow liquid inside the perfume bottle",
    #     "LENS_SEG_PROMPT": "the perfume bottle",
    #     "LENS_DISTANCE_MULTIPLIERS": "5.0",
    #     "TARGET_PROMPT": "ocean blue liquid inside the perfume bottle",
    #     "STYLE_SOURCE_PROMPT": "yellow liquid inside the perfume bottle",
    #     "STYLE_TARGET_PROMPT": "ocean blue liquid inside the perfume bottle",
    # },
    # # 6) Add a digital clock display to the remote controller
    # # {
    # #     "name": "remote_digital",
    # #     "DATA_NAME": "blue_sofa",
    # #     "PROMPT": "Make the remote controller screen glow neon green",
    # #     "SEG_PROMPT": "screen of the white remote controller",
    # #     "TARGET_PROMPT": "glowing neon green screen of the remote controller",
    # #     "STYLE_SOURCE_PROMPT": "screen of the white remote controller",
    # #     "STYLE_TARGET_PROMPT": "glowing neon green screen of the remote controller",
    # # },
    # # 7) Turn the plush toy into a tiger
    # # {
    # #     "name": "plush_tiger",
    # #     "DATA_NAME": "blue_sofa",
    # #     "PROMPT": "Change the plush toy's pattern to tiger stripes",
    # #     "SEG_PROMPT": "plush toy",
    # #     "TARGET_PROMPT": "tiger striped plush toy",
    # #     "STYLE_SOURCE_PROMPT": "plush toy",
    # #     "STYLE_TARGET_PROMPT": "tiger striped plush toy",
    # # },
    # # 8) Change the speaker color to gold
    # {
    #     "name": "speaker_gold",
    #     "DATA_NAME": "blue_sofa",
    #     "PROMPT": "Change the speaker to a shiny gold texture",
    #     "SEG_PROMPT": "grey speaker body",
    #     "LENS_SEG_PROMPT": "grey speaker body",
    #     "LENS_DISTANCE_MULTIPLIERS": "2.5",
    #     "TARGET_PROMPT": "shiny metallic gold speaker body",
    #     "STYLE_SOURCE_PROMPT": "grey speaker body",
    #     "STYLE_TARGET_PROMPT": "shiny metallic gold speaker body",
    # },

    # # 9) Make the sunglasses look like aviator glasses
    # {
    #     "name": "glasses_aviator",
    #     "DATA_NAME": "blue_sofa",
    #     "PROMPT": "Change the sunglasses style to gold-rimmed aviators",
    #     "SEG_PROMPT": "black sunglasses",
    #     "LENS_SEG_PROMPT": "sunglasses",
    #     "LENS_DISTANCE_MULTIPLIERS": "5.0",
    #     "TARGET_PROMPT": "gold-rimmed aviator sunglasses",
    #     "STYLE_SOURCE_PROMPT": "black sunglasses",
    #     "STYLE_TARGET_PROMPT": "gold-rimmed aviator sunglasses",
    # },
    # # 10) Change the perfume bottle cap to silver
    # {
    #     "name": "perfume_silver_cap",
    #     "DATA_NAME": "blue_sofa",
    #     "PROMPT": "Change the perfume bottle cap to silver",
    #     "LENS_SEG_PROMPT": "the perfume bottle",
    #     "LENS_DISTANCE_MULTIPLIERS": "5.0",
    #     "SEG_PROMPT": "black cap of the perfume bottle",
    #     "TARGET_PROMPT": "polished silver cap of the perfume bottle",
    #     "STYLE_SOURCE_PROMPT": "black cap of the perfume bottle",
    #     "STYLE_TARGET_PROMPT": "polished silver cap of the perfume bottle",
    # },

    # ## room
    # # 1) Change the rubber chicken's color to red
    # {
    #     "name": "chicken_red",
    #     "DATA_NAME": "room",
    #     "PROMPT": "Change the rubber chicken's color to red",
    #     "SEG_PROMPT": "yellow rubber chicken",
    #     "LENS_SEG_PROMPT": "yellow rubber chicken",
    #     "LENS_DISTANCE_MULTIPLIERS": "1.4",
    #     "TARGET_PROMPT": "red rubber chicken",
    #     "STYLE_SOURCE_PROMPT": "yellow rubber chicken",
    #     "STYLE_TARGET_PROMPT": "red rubber chicken",
    # },
    # # 2) Make the rabbit figure white
    # # {
    # #     "name": "rabbit_white",
    # #     "DATA_NAME": "room",
    # #     "PROMPT": "Make the rabbit figure white",
    # #     "SEG_PROMPT": "grey rabbit figure",
    # #     "TARGET_PROMPT": "white rabbit figure",
    # #     "STYLE_SOURCE_PROMPT": "grey rabbit figure",
    # #     "STYLE_TARGET_PROMPT": "white rabbit figure",
    # # },
    # # 3) Change the dinosaur figure to green
    # {
    #     "name": "dino_green",
    #     "DATA_NAME": "room",
    #     "PROMPT": "Change the dinosaur figure to green",
    #     "SEG_PROMPT": "brown dinosaur figure",
    #     "LENS_SEG_PROMPT": "brown dinosaur figure",
    #     "LENS_DISTANCE_MULTIPLIERS": "4.0",
    #     "TARGET_PROMPT": "green dinosaur figure",
    #     "STYLE_SOURCE_PROMPT": "brown dinosaur figure",
    #     "STYLE_TARGET_PROMPT": "green dinosaur figure",
    # },
    # # 4) Replace the baseball with a tennis ball
    # {
    #     "name": "ball_tennis",
    #     "DATA_NAME": "room",
    #     "PROMPT": "Replace the baseball with a tennis ball",
    #     "SEG_PROMPT": "white baseball",
    #     "LENS_SEG_PROMPT": "white baseball",
    #     "LENS_DISTANCE_MULTIPLIERS": "6.0",
    #     "TARGET_PROMPT": "yellow tennis ball",
    #     "STYLE_SOURCE_PROMPT": "white baseball",
    #     "STYLE_TARGET_PROMPT": "yellow tennis ball",
    # },
    # # 5) Change the basket material to wood
    # {
    #     "name": "basket_wood",
    #     "DATA_NAME": "room",
    #     "PROMPT": "Change the basket material to dark wood",
    #     "SEG_PROMPT": "woven basket",
    #     "TARGET_PROMPT": "dark wooden basket",
    #     "STYLE_SOURCE_PROMPT": "woven basket",
    #     "STYLE_TARGET_PROMPT": "dark wooden basket",
    # },
    # # 6) Add a tiny bow tie to the rubber chicken
    # {
    #     "name": "chicken_bow_tie",
    #     "DATA_NAME": "room",
    #     "PROMPT": "Add a tiny blue bow tie to the rubber chicken's neck",
    #     "SEG_PROMPT": "neck of the yellow rubber chicken",
    #     "LENS_SEG_PROMPT": "yellow rubber chicken",
    #     "LENS_DISTANCE_MULTIPLIERS": "1.5",
    #     "TARGET_PROMPT": "yellow rubber chicken wearing a tiny blue bow tie",
    #     "STYLE_SOURCE_PROMPT": "neck of the yellow rubber chicken", # neck 들어가는게 잘돼
    #     "STYLE_TARGET_PROMPT": "yellow rubber chicken wearing a tiny blue bow tie",
    # },
    # # 7) Give the rabbit figure sunglasses
    # {
    #     "name": "rabbit_sunglasses",
    #     "DATA_NAME": "room",
    #     "PROMPT": "Give the rabbit figure a pair of small sunglasses",
    #     "SEG_PROMPT": "face of the grey rabbit figure",
    #     "LENS_SEG_PROMPT": "face of the grey rabbit figure",
    #     "LENS_DISTANCE_MULTIPLIERS": "1.6",
    #     "TARGET_PROMPT": "grey rabbit figure wearing small sunglasses",
    #     "STYLE_SOURCE_PROMPT": "face of the grey rabbit figure",
    #     "STYLE_TARGET_PROMPT": "grey rabbit figure wearing small sunglasses",
    # },
    # # 8) Make the dinosaur figure look like it's made of metal
    # {
    #     "name": "dino_metal",
    #     "DATA_NAME": "room",
    #     "PROMPT": "Make the dinosaur figure look like it's made of shiny metal",
    #     "SEG_PROMPT": "brown dinosaur figure",
    #     "LENS_SEG_PROMPT": "brown dinosaur figure",
    #     "LENS_DISTANCE_MULTIPLIERS": "4.0",
    #     "TARGET_PROMPT": "shiny metallic dinosaur figure",
    #     "STYLE_SOURCE_PROMPT": "brown dinosaur figure",
    #     "STYLE_TARGET_PROMPT": "shiny metallic dinosaur figure",
    # },
    # # 9) Change the baseball to a golden ball
    {
        "name": "ball_gold",
        "DATA_NAME": "room",
        "PROMPT": "Change the baseball to a solid golden ball",
        "SEG_PROMPT": "white baseball",
        "LENS_SEG_PROMPT": "white baseball",
        "LENS_DISTANCE_MULTIPLIERS": "20.0", # 20 이상 다 굿
        "TARGET_PROMPT": "solid golden ball",
        "STYLE_SOURCE_PROMPT": "white baseball",
        "STYLE_TARGET_PROMPT": "solid golden ball",
    },
    # # 10) Fill the basket with apples
    # {
    #     "name": "basket_apples",
    #     "DATA_NAME": "room",
    #     "PROMPT": "Fill the empty space in the basket with red apples",
    #     "SEG_PROMPT": "inside of the woven basket",
    #     "TARGET_PROMPT": "woven basket filled with red apples",
    #     "STYLE_SOURCE_PROMPT": "inside of the woven basket",
    #     "STYLE_TARGET_PROMPT": "woven basket filled with red apples",
    # },
]
TASKS_BY_NAME = {t["name"]: t for t in TASKS if "name" in t}

# ==========================
# Sweep configuration
# ==========================

SWEEP_CONFIG = {
    "name": WANDB_SWEEP_NAME,
    "method": "grid",
    "metric": {
        "name": "clip_dir_similarity",
        "goal": "maximize",
    },
    "parameters": {
        # 필요에 따라 아래 주석을 풀어 sweep 공간을 확장할 수 있음
        # "lambda_d": {
        #     "values": [0.0, 5.0, 10.0],
        # },
        "edit_view_selection_strategy": {
            "values": ["lens"],
        },
        # "lambda_ism": {
        #     "values": [0.0, 0.0001],
        # },
        # (max_view_num, max_edit_view_num) 쌍 — "25_20" 형식
        "view_config": {
            "values": [f"{m}_{e}" for m, e in VIEW_CONFIG_PAIRS],
        },
        # Warp-refine edit strength (0~1, blend between warped input & IP2P output).
        # "warp_refine_ip2p_strength": {
        #     "values": [0.5, 0.75, 1.0],
        # },
        # Lens sampling distance multipliers around COLMAP mean direction
        # "lens_distance_multipliers": {
        #     "values": [
        #         "2.0,2.5,3.0,4.0,5.0,6.0",   # 기본값 (launch_and_metrics.py)
        #         "1.5,2.0,2.5,3.0,3.5,4.0",   # 더 근접한 뷰 위주
        #         "3.0,4.0,5.0,6.0",           # 더 먼 뷰 위주
        #     ],
        # },
        # Cone constraint (in degrees) around COLMAP mean direction
        # "lens_cone_half_angle_deg": {
        #     "values": [45, 60, 75],
        # },
        # Warp-refine 사용 여부 (True: warp-refine 브랜치, False: 기존 DGE guidance)
        # "use_warp_refine": {
        #     "values": [True, False],
        #     # "values": [False],
        # },
        "task": {
            "values": [t["name"] for t in TASKS],
        },
    },
}

# ==========================
# Helpers
# ==========================


def get_root_dir() -> Path:
    """
    DGE repo root를 찾는다.

    - 우선 환경변수 DGE_ROOT가 설정되어 있으면 그 경로를 사용
    - 아니면 현재 파일에서 위로 올라가며 launch.py와 configs 디렉토리를 찾음
    - 그래도 못 찾으면 기존 heuristic (parent.parent.parent.parent) 사용
    """
    env_root = os.environ.get("DGE_ROOT")
    if env_root:
        p = Path(env_root).expanduser().absolute()
        if p.exists():
            return p

    cur = Path(__file__).absolute()
    # 너무 많이 안 올라가도록 depth 제한
    for _ in range(10):
        if (cur / "launch.py").exists() and (cur / "configs").exists():
            return cur.absolute()
        if cur.parent == cur:
            break
        cur = cur.parent

    # Fallback: 예전 구조 가정
    return Path(__file__).parent.parent.parent.parent.absolute()


def find_save_directory(launch_output: str, name: str) -> Optional[Path]:
    """Find save directory from launch.py stdout. 로그 파싱만 사용 (여러 프로세스 시 fallback은 잘못된 trial 반환 가능)."""
    prefix = "Test results saved to "
    if prefix not in launch_output:
        # 로그에 없으면 fallback (단일 프로세스용)
        exp_root = Path(EXP_ROOT_DIR)
        for exp_dir in [exp_root / name, exp_root / name / str(MAX_VIEW_NUM)]:
            if not exp_dir.exists():
                continue
            trial_saves = []
            for d in exp_dir.iterdir():
                if not d.is_dir():
                    continue
                if "@" in d.name:
                    save = d / "save"
                    if save.exists():
                        trial_saves.append((save, save.stat().st_mtime))
                    continue
                for sub in d.iterdir():
                    if sub.is_dir() and "@" in sub.name:
                        save = sub / "save"
                        if save.exists():
                            trial_saves.append((save, save.stat().st_mtime))
                        break
            if trial_saves:
                trial_saves.sort(key=lambda x: x[1], reverse=True)
                return trial_saves[0][0]
        return None

    # 로그에 있으면 반드시 파싱해서 사용 (fallback 사용 안 함)
    # tqdm 출력이 [INFO] 앞에 붙어서 한 줄로 합쳐지는 경우를 처리:
    #   "100%|██████| 65/65 [00:09<00:00, 7.17it/s][INFO] Test results saved to /path/to/save"
    for pattern in [
        r"Test results saved to\s+(/[^\s\r\n]+)",  # 절대경로 (공백/줄바꿈 전까지)
        r"Test results saved to\s+(.+?)(?:\r?\n|$)",  # \r\n, \n 처리
        r"Test results saved to\s+(.+?/save)(?:\s|[\r\n]|$)",  # /save로 끝나는 경로
        r"Test results saved to\s+(.+)",
    ]:
        matches = re.findall(pattern, launch_output)
        if matches:
            raw = matches[-1].strip().rstrip("\r")  # tqdm \r 등 제거
            p = Path(raw)
            if p.exists():
                return p
    # 모든 패턴 실패 시, 마지막으로 절대경로를 직접 추출 시도
    # (tqdm \r로 인해 줄이 덮어쓰여 regex가 실패하는 경우)
    abs_pattern = r"Test results saved to\s+(/\S+)"
    abs_matches = re.findall(abs_pattern, launch_output)
    if abs_matches:
        raw = abs_matches[-1].strip().rstrip("\r")
        p = Path(raw)
        if p.exists():
            return p
        # 경로가 존재하지 않더라도, /save로 끝나면 반환 (아직 생성 중일 수 있음)
        if raw.endswith("/save") or "/save" in raw:
            print(f"[Sweep] Path does not exist yet, but returning anyway: {p}")
            return p
    print("[Sweep] Found 'Test results saved to' in output but could not extract valid path.")
    print(f"[Sweep] Attempted to match in output (last 500 chars): {launch_output[-500:]}")
    return None

def find_render_directory(save_dir: Path) -> Optional[Path]:
    """Find it*-test render dir; search save_dir, parent, and recursively."""
    # 1) save_dir/it{MAX_STEPS}-test
    render_dir = save_dir / f"it{MAX_STEPS}-test"
    if render_dir.exists() and list(render_dir.glob("*.png")):
        return render_dir
    # 2) save_dir/it*-test (any step)
    test_dirs = sorted(save_dir.glob("it*-test"), key=lambda p: p.stat().st_mtime, reverse=True)
    for d in test_dirs:
        if d.is_dir() and list(d.glob("*.png")):
            return d
    # 3) trial_dir/it*-test (save_dir parent)
    parent = save_dir.parent
    test_dirs = sorted(parent.glob("it*-test"), key=lambda p: p.stat().st_mtime, reverse=True)
    for d in test_dirs:
        if d.is_dir() and list(d.glob("*.png")):
            return d
    # 4) recursive under save_dir
    for d in save_dir.rglob("it*-test"):
        if d.is_dir() and list(d.glob("*.png")):
            return d
    # 5) save_dir itself has .png (flat structure)
    if list(save_dir.glob("*.png")):
        return save_dir
    # 6) search entire trial tree (save_dir.parent and above)
    for parent in [save_dir.parent, save_dir.parent.parent]:
        if parent.exists():
            for d in parent.rglob("it*-test"):
                if d.is_dir() and list(d.glob("*.png")):
                    return d
    return None


def parse_metrics(output: str) -> dict:
    """Parse metrics.py stdout into a dict.

    Matches only lines that start with the metric name (after optional
    carriage-returns from tqdm) to avoid false matches inside progress bars.
    """
    result = {}
    # Use ^ with MULTILINE so we only match at the start of a line.
    # Also allow optional leading \r (tqdm sometimes emits \r before the line).
    patterns = {
        # tqdm progress lines look like "CLIP directional consistency:  26%|..."
        # Real result lines end with just a number followed by end-of-line.
        # \s*$ ensures no trailing % or | after the number.
        "clip_dir_consistency": r"^(?:\r)?CLIP directional consistency:\s*([-\d.]+)\s*$",
        "clip_f_scaled":        r"^(?:\r)?CLIP_F \(scaled\):\s*([-\d.]+)\s*$",
        "clip_score":           r"^(?:\r)?CLIP Score:\s*([-\d.]+)\s*$",
        "clip_dir_similarity":  r"^(?:\r)?CLIP directional similarity:\s*([-\d.]+)\s*$",
    }
    for key, pat in patterns.items():
        m = re.search(pat, output, re.MULTILINE)
        if m:
            result[key] = float(m.group(1))
    return result


# ==========================
# Agent function (called per sweep run)
# ==========================

def train_and_evaluate():
    """Single sweep run: pick config from wandb, train, evaluate, log metric."""
    run = wandb.init()
    cfg = run.config

    # GPU: from env (set by main() when --gpu is passed) or default
    gpu = os.environ.get("SWEEP_GPU", GPU)

    # getattr(..., default)로 읽어서, SWEEP_CONFIG에서 해당 파라미터를
    # 주석 처리해도 여기서 KeyError 없이 동작하도록 한다.
    lambda_d = getattr(cfg, "lambda_d", float(LAMBDA_D_DEFAULT))
    strategy = getattr(cfg, "edit_view_selection_strategy", EDIT_VIEW_SELECTION_STRATEGY_DEFAULT)
    lambda_ism = getattr(cfg, "lambda_ism", LAMBDA_ISM_DEFAULT)
    warp_refine_color_fit_steps = getattr(
        cfg, "warp_refine_color_fit_steps", WARP_REFINE_COLOR_FIT_STEPS_DEFAULT
    )
    use_warp_refine = getattr(cfg, "use_warp_refine", USE_WARP_REFINE_DEFAULT)
    task_name = getattr(cfg, "task", TASKS[0]["name"])
    task = TASKS_BY_NAME[task_name]
    prompt = task["PROMPT"]
    seg_prompt = task["SEG_PROMPT"]
    target_prompt = task["TARGET_PROMPT"]
    style_target_prompt = task.get("STYLE_TARGET_PROMPT", "")
    style_source_prompt = task.get("STYLE_SOURCE_PROMPT", "a Photo")

    # view_config: "25_20" → max_view_num=25, max_edit_view_num=20
    view_config_str = getattr(cfg, "view_config", f"{MAX_VIEW_NUM}_{MAX_EDIT_VIEW_NUM}")
    max_view_num, max_edit_view_num = view_config_str.split("_")

    data_name = task.get("DATA_NAME", DATA_NAME)
    data_source = f"{DATA_SOURCE_ROOT}/{DATA_TYPE}/{data_name}/"
    gs_source = f"{GS_SOURCE_ROOT}/{DATA_TYPE}/{data_name}/point_cloud/iteration_30000/point_cloud.ply"
    gt_dir = f"{GS_SOURCE_ROOT}/{DATA_TYPE}/{data_name}/{RENDER_SUBDIR}"

    name = (
        f"sweep/{DATA_TYPE}/{data_name}/view{view_config_str}"
        f"/lambda_d{lambda_d}/{strategy}/lambda_ism{lambda_ism}/cf{warp_refine_color_fit_steps}/wr{use_warp_refine}/{task_name}"
    )

    root_dir = get_root_dir()
    os.chdir(root_dir)

    # ---- Build launch command ----
    launch_cmd = [
        "python", "launch.py",
        "--config", CONFIG,
        "--train",
        "--gpu", gpu,
        f"trainer.max_steps={MAX_STEPS}",
        f"system.prompt_processor.prompt={prompt}",
        f"data.source={data_source}",
        f"system.guidance.guidance_scale={GUIDANCE_SCALE}",
        f"system.gs_source={gs_source}",
        f"system.seg_prompt={seg_prompt}",
        f"data.mmr_seg_prompt={seg_prompt}",
        f"system.target_prompt={target_prompt}",
        f"system.guidance.target_use_extended_attention={TARGET_USE_EXTENDED_ATTENTION}",
        f"system.guidance.feature_injection_mode={FEATURE_INJECTION_MODE}",
        f"system.guidance.per_step_cross_attn_consistency={PER_STEP_CROSS_ATTN_CONSISTENCY}",
        f"system.guidance.per_step_cross_attn_t_start={PER_STEP_CROSS_ATTN_T_START}",
        f"system.guidance.per_step_cross_attn_resolutions=[{PER_STEP_CROSS_ATTN_RESOLUTIONS}]",
        f"system.guidance.target_batch_strategy={TARGET_BATCH_STRATEGY}",
        f"system.guidance.target_batch_neighbor_threshold={TARGET_BATCH_NEIGHBOR_THRESHOLD}",
        f"system.guidance.target_batch_late_mode={TARGET_BATCH_LATE_MODE}",
        f"system.guidance.target_batch_sliding_stride={TARGET_BATCH_SLIDING_STRIDE}",
        f"system.guidance.target_key_selection_mode={TARGET_KEY_SELECTION_MODE}",
        f"system.mask_thres={MASK_THRES}",
        f"system.mask_max_ratio={MASK_MAX_RATIO}",
        f"system.mask_min_ratio={MASK_MIN_RATIO}",
        f"system.mask_outlier_iqr={MASK_OUTLIER_IQR}",
        f"system.loss.lambda_d={lambda_d}",
        f"system.loss.lambda_dds={LAMBDA_DDS}",
        f"system.dds_t_range=[{DDS_T_RANGE}]",
        f"system.dds_cfg_scale={DDS_CFG_SCALE}",
        f"system.loss.lambda_ism={lambda_ism}",
        f"system.loss.use_sds={str(USE_SDS).lower()}",
        f"system.guidance.use_sds_dge={str(USE_SDS_DGE).lower()}",
        f"system.loss.lambda_sds={LAMBDA_SDS}",
        f"system.warp_refine_color_fit_steps={warp_refine_color_fit_steps}",
        f"system.use_warp_refine={str(use_warp_refine).lower()}",
        f"system.save_image_grid_draw_texts={SAVE_IMAGE_GRID_DRAW_TEXTS}",
        f"data.max_view_num={max_view_num}",
        f"data.max_edit_view_num={max_edit_view_num}",
        f"system.multiview_edit_key_selection_strategy={MULTIVIEW_EDIT_KEY_SELECTION_STRATEGY}",
        f"system.use_multiview_edit={str(USE_MULTIVIEW_EDIT_DEFAULT).lower()}",
        f"system.use_gaussian_provenance={str(USE_GAUSSIAN_PROVENANCE_DEFAULT).lower()}",
        f"data.edit_view_selection_strategy={strategy}",
        f"data.lens_ply_path={gs_source}",
        f"data.lens_seg_prompt={seg_prompt}",
        f"data.lens_edit_prompt={prompt}",
        f"data.lens_use_ip2p_scoring={LENS_USE_IP2P_SCORING}",
        f"data.lens_entropy_thresh={LENS_ENTROPY_THRESH}",
        f"data.lens_ip2p_steps={LENS_IP2P_STEPS}",
        f"data.lens_ip2p_guidance_scale={LENS_IP2P_GUIDANCE_SCALE}",
        f"data.lens_ip2p_image_guidance_scale={LENS_IP2P_IMAGE_GUIDANCE_SCALE}",
        f"data.lens_distance_multipliers={LENS_DISTANCE_MULTIPLIERS}",
        f"data.lens_cone_half_angle_deg={LENS_CONE_HALF_ANGLE_DEG}",
        f"data.lens_lambda_leak={LENS_LAMBDA_LEAK}",
        f"data.lens_lambda_ent={LENS_LAMBDA_ENT}",
        f"data.lens_hemisphere_only={LENS_HEMISPHERE_ONLY}",
        f"data.lens_v_front_method={LENS_V_FRONT_METHOD}",
        f"data.lens_n_candidates={LENS_N_CANDIDATES}",
        f"data.lens_diversity_x_weight={LENS_DIVERSITY_X_WEIGHT}",
        f"data.lens_diversity_y_weight={LENS_DIVERSITY_Y_VARIANCE_WEIGHT}",
        f"data.lens_ip2p_batch_size={LENS_IP2P_BATCH_SIZE}",
        f"system.guidance.camera_batch_size={CAMERA_BATCH_SIZE}",
        f"system.guidance.edit_view_selection_strategy={strategy}",
        f"system.camera_update_per_step={CAMERA_UPDATE_PER_STEP}",
        f"system.mask_update_at_step={MASK_UPDATE_AT_STEP}",
        f"system.mask_update_view_num={MASK_UPDATE_VIEW_NUM}",
        f"system.mask_view_population={MASK_VIEW_POPULATION}",
        f"system.mask_num_views={MASK_NUM_VIEWS}",
        f"system.prune_floater_at_step={PRUNE_FLOATER_AT_STEP}",
        f"exp_root_dir={EXP_ROOT_DIR}",
        f"name={name}",
        # Disable inner wandb logging to avoid nested runs
        "system.loggers.wandb.enable=false",
    ]

    print(f"\n[Sweep] Running: task={task_name}, lambda_d={lambda_d}, strategy={strategy}, lambda_ism={lambda_ism}")
    print(f"[Sweep] name={name}\n")

    # ---- Run training ----
    launch_output = ""
    proc = subprocess.Popen(
        launch_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=root_dir,
        env=os.environ.copy(),
    )
    lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    launch_output = "".join(lines)

    if proc.returncode != 0:
        print("[Sweep] Training failed.")
        wandb.log({"clip_dir_similarity": float("nan")})
        run.finish(exit_code=1)
        return

    # ---- Find render directory ----
    save_dir = find_save_directory(launch_output, name)
    if not save_dir:
        print("[Sweep] Could not find save directory.")
        wandb.log({"clip_dir_similarity": float("nan")})
        run.finish(exit_code=1)
        return

    render_dir = find_render_directory(save_dir)
    if not render_dir:
        print(f"[Sweep] Could not find render dir in {save_dir}")
        wandb.log({"clip_dir_similarity": float("nan")})
        run.finish(exit_code=1)
        return

    print(f"[Sweep] Render dir: {render_dir}")

    # ---- Run metrics ----
    # GT dir must match this run's strategy (lens vs random); path must exist with pre-generated origin renders
    gt_dir = gt_dir
    # Use first GPU in list for metrics (e.g. "0,1" -> cuda:0)
    gpu_id = gpu.split(",")[0].strip()
    device = f"cuda:{gpu_id}"
    metrics_cmd = [
        "python", "metrics.py",
        "--gt", gt_dir,
        "--render", str(render_dir),
        "--device", device,
        "--interval", str(INTERVAL),
        "--style_source_prompt", style_source_prompt,
        "--style_target_prompt", style_target_prompt,
    ]

    metrics_proc = subprocess.Popen(
        metrics_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=root_dir,
        env=os.environ.copy(),
    )
    metrics_lines = []
    for line in metrics_proc.stdout:
        print(line, end="", flush=True)
        metrics_lines.append(line)
    metrics_proc.wait()
    metrics_output = "".join(metrics_lines)

    if metrics_proc.returncode != 0:
        print("[Sweep] Metrics failed.")
        wandb.log({"clip_dir_similarity": float("nan")})
        run.finish(exit_code=1)
        return

    # ---- Parse and log metrics ----
    metrics = parse_metrics(metrics_output)
    print(f"\n[Sweep] Metrics: {metrics}")

    if not metrics:
        print("[Sweep] Could not parse any metrics from output.")
        wandb.log({"clip_dir_similarity": float("nan")})
    else:
        wandb.log(metrics)

    run.finish()


# ==========================
# Entry point
# ==========================
def main():
    parser = argparse.ArgumentParser(description="WandB Sweep for DGE camera-selection")
    parser.add_argument("--create", action="store_true", help="Create a new sweep")
    parser.add_argument("--agent", action="store_true", help="Start a sweep agent")
    parser.add_argument("--sweep_id", type=str, default=None, help="Sweep ID")
    parser.add_argument("--count", type=int, default=None, help="Max number of runs per agent")
    parser.add_argument("--gpu", type=str, default=None, help="GPU id for this specific agent")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPUs (e.g., '0,1,2,3') to run parallel agents")
    args = parser.parse_args()

    sweep_id = args.sweep_id

    if args.create:
        sweep_id = wandb.sweep(SWEEP_CONFIG, project=WANDB_PROJECT)
        print(f"\n[Created] Sweep ID: {sweep_id}")

    if args.agent:
        if not sweep_id:
            print("Error: --sweep_id is required.")
            sys.exit(1)

        # 핵심: 여러 GPU를 한 번에 넣었을 경우 (예: --gpus 0,1,2,3)
        if args.gpus:
            gpu_list = [s.strip() for s in args.gpus.split(",") if s.strip()]
            print(f"🚀 Launching {len(gpu_list)} agents on GPUs: {gpu_list}")
            
            processes = []
            for g in gpu_list:
                env = os.environ.copy()
                # 해당 프로세스는 오직 지정된 GPU만 보이게 설정
                env["CUDA_VISIBLE_DEVICES"] = g
                env["SWEEP_GPU"] = "0" # 컨테이너/프로세스 내부에서는 0번으로 접근
                
                # 자기 자신을 실행하되, --gpus 대신 단일 --gpu 인자를 넣어 재귀 방지
                cmd = [
                    sys.executable, "-u", sys.argv[0],
                    "--agent",
                    "--sweep_id", sweep_id,
                    "--gpu", "0" 
                ]
                if args.count:
                    cmd += ["--count", str(args.count)]
                
                p = subprocess.Popen(cmd, env=env)
                processes.append(p)
            
            # 모든 에이전트가 끝날 때까지 대기
            for p in processes:
                p.wait()
            sys.exit(0)

        # 단일 에이전트 실행부 (위의 Popen에 의해 각 GPU별로 실행됨)
        gpu_to_use = args.gpu if args.gpu else "0"
        os.environ["SWEEP_GPU"] = gpu_to_use
        
        print(f"✅ Agent started on Physical GPU {os.environ.get('CUDA_VISIBLE_DEVICES', 'Unknown')}")
        
        wandb.agent(
            sweep_id,
            function=train_and_evaluate,
            project=WANDB_PROJECT,
            count=args.count,
        )

if __name__ == "__main__":
    main()