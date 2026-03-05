#!/usr/bin/env python3
"""
WandB Sweep for DGE camera-selection hyperparameter search.

이 스크립트는 `DGE-camera-selection/script/camera-selection/3d-ovs/sweep.py`
와 동일한 로직을 현재 레포 구조에 맞게 옮겨온 것입니다.

Sweep target metric: CLIP directional similarity (maximize)
Search space:
  - lambda_d:                    [1.0,  5.0, 10.0]
  - edit_view_selection_strategy: [lens, random]
  - task:                        [다양한 3d-ovs 편집 task들]  (prompt combos from TASKS)

Usage:
  # 1) Create sweep (run once)
  python script/sweep_in2n.py --create

  # 2) Run agent(s) – each agent picks a config and runs train+metrics

  # Run multiple agents in parallel (one process per GPU, single command)
  python script/sweep_in2n.py --agent --sweep_id 6o0wf0zu --gpus 0,1,2
  
  python script/sweep_in2n.py --agent --sweep_id 0r1nt2az --gpu 0
  python script/sweep_in2n.py --agent --sweep_id 6o0wf0zu --gpu 1
  python script/sweep_in2n.py --agent --sweep_id 6o0wf0zu --gpu 2
  python script/sweep_in2n.py --agent --sweep_id 6o0wf0zu --gpu 3

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
# Fixed configuration
# ==========================

CONFIG = "configs/dge_camera-selection.yaml"
GPU = "0,1,2,3"  # default; override with --gpu (e.g. "0" or "0,1,2" for multi-GPU per run)

DATA_TYPE = "in2n-GSEditor"
DATA_NAME = "covered_desk"  # default; may be overridden per TASK via DATA_NAME
DATA_SOURCE_ROOT = "/data/users/jaeyeonpark/dataset"
GS_SOURCE_ROOT = "/data/users/jaeyeonpark/3dgs-trained"
RENDER_SUBDIR = "colmap_render_full"

GUIDANCE_SCALE = "12.5"
MASK_THRES = "0.6"
MASK_MAX_RATIO = "0.6"
MASK_MIN_RATIO = "0.01"
MASK_OUTLIER_IQR = "1.5"
MAX_VIEW_NUM = "25"
MAX_EDIT_VIEW_NUM = "20"

# (max_view_num, max_edit_view_num) 쌍 조합 — sweep에서 view_config로 선택
VIEW_CONFIG_PAIRS = [
    ("20", "20"),
    ("15", "15"),
    ("10", "10"),
    ("5", "5"),
]
CAMERA_UPDATE_PER_STEP = "1500"
MASK_UPDATE_AT_STEP = "600"
MASK_UPDATE_VIEW_NUM = "30"
PRUNE_FLOATER_AT_STEP = "-1"
MAX_STEPS = "1500"

# Loss / guidance (3d-ovs sweep와 동일 기본값)
LAMBDA_D_DEFAULT = "0.0"
LAMBDA_DDS = "0.0"
DDS_T_RANGE = "0.02,0.5"
DDS_CFG_SCALE = "7.5"
USE_SDS = False
USE_SDS_DGE = False
LAMBDA_SDS = "0.0"
LAMBDA_ISM_DEFAULT = 0.0

# Multiview edit (3d-ovs sweep와 동일 기본값)
MULTIVIEW_EDIT_KEY_SELECTION_STRATEGY = "lens_fps"
USE_MULTIVIEW_EDIT_DEFAULT = True
USE_GAUSSIAN_PROVENANCE_DEFAULT = False

# Warp refine (3d-ovs sweep와 동일 기본값)
USE_WARP_REFINE_DEFAULT = False
WARP_REFINE_COLOR_FIT_STEPS_DEFAULT = 100

INTERVAL = 1
DEVICE = "cuda"

WANDB_PROJECT = "dge-ours-in2n"
WANDB_SWEEP_NAME = f"sweep/{DATA_TYPE}"

# ==========================
# Lens camera-selection defaults (multiview-in2n와 동일한 설정)
# ==========================

# face: lens, bear: random
EDIT_VIEW_SELECTION_STRATEGY_BY_DATA_NAME = {
    "face": "lens",
    "bear": "random",
}

LENS_USE_IP2P_SCORING = "true"
LENS_ENTROPY_THRESH = "0.97"
LENS_IP2P_STEPS = "5"
LENS_LAMBDA_LEAK = "1.5"
LENS_LAMBDA_ENT = "2.0"
LENS_IP2P_GUIDANCE_SCALE = "7.5"
LENS_IP2P_IMAGE_GUIDANCE_SCALE = "1.5"
LENS_HEMISPHERE_ONLY = "true"
LENS_CONE_HALF_ANGLE_DEG = "60.0"  # 60: man
LENS_V_FRONT_METHOD = "scene_center"
LENS_N_CANDIDATES = "900"
LENS_DIVERSITY_X_WEIGHT = "30.0"
LENS_DIVERSITY_Y_VARIANCE_WEIGHT = "3.0"
LENS_IP2P_BATCH_SIZE = "2"
CAMERA_BATCH_SIZE = "5"

# face에 맞춘 기본값
LENS_DISTANCE_MULTIPLIERS_DEFAULT = "3.0"

# Extended-attention & target batch defaults (to match launch.json)
SAVE_IMAGE_GRID_DRAW_TEXTS = "false"
TARGET_USE_EXTENDED_ATTENTION = "true"
FEATURE_INJECTION_MODE = "similarity"
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

# 출력 루트: 여기 아래에 sweep/3d-ovs/... 가 생성됨 (dge.yaml exp_root_dir 오버라이드)
EXP_ROOT_DIR = "/data/users/jaeyeonpark/DGE-ours-outputs"

# ==========================
# TASK 정의 (원본과 동일)
# ==========================

# STYLE_SOURCE_PROMPT = 편집 전(원본), STYLE_TARGET_PROMPT = 편집 후(목표)
TASKS = [

    # # 0) Turn the man into a clown
    # {
    #     "name": "man_to_clown",
    #     "DATA_NAME": "face",
    #     "PROMPT": "Turn the man's face into a clown",
    #     "SEG_PROMPT": "a face",
    #     "TARGET_PROMPT": "a face of the clown",
    #     "STYLE_SOURCE_PROMPT": "a face of the man",
    #     "STYLE_TARGET_PROMPT": "a face of the clown",
    # },
    # 2) Change the man's hair color to dark brown
    {
        "name": "hair_dark_brown",
        "DATA_NAME": "face",
        "PROMPT": "Change his hair color to dark brown",
        "SEG_PROMPT": "hair",
        "LENS_SEG_PROMPT": "a face",
        "TARGET_PROMPT": "the man's dark brown hair",
        "STYLE_SOURCE_PROMPT": "man with natural light brown hair",
        "STYLE_TARGET_PROMPT": "man with dark brown hair",
    },
    # 3) Make the man's mouth smile
    {
        "name": "mouth_smile",
        "DATA_NAME": "face",
        "PROMPT": "Make his mouth smile",
        "SEG_PROMPT": "the man's mouth",
        "LENS_SEG_PROMPT": "a face",
        "TARGET_PROMPT": "the man's mouth in a smiling pose",
        "STYLE_SOURCE_PROMPT": "man with neutral, closed mouth",
        "STYLE_TARGET_PROMPT": "man with open, smiling mouth",
    },
    # 4) Make him wear sunglasses
    # {
    #     "name": "wear_sunglasses",
    #     "DATA_NAME": "face",
    #     "PROMPT": "Make him wear sunglasses on his face",
    #     "SEG_PROMPT": "a face",
    #     "TARGET_PROMPT": "a face of the man with dark sunglasses",
    #     "STYLE_SOURCE_PROMPT": "a face of the man",
    #     "STYLE_TARGET_PROMPT": "a face of the man with dark sunglasses",
    # },
    # 5) Make his ear like an elf's ear
    {
        "name": "ear_elf",
        "DATA_NAME": "face",
        "PROMPT": "Make his ear like an elf's ear",
        "SEG_PROMPT": "a face",
        "TARGET_PROMPT": "man's pointed, elf-like right ear",
        "STYLE_SOURCE_PROMPT": "man's rounded, normal human ear",
        "STYLE_TARGET_PROMPT": "man with pointed, elf-like human ear",
    },
    # 6) Change the Patagonia logo to a simple tree graphic
    {
        "name": "logo_change_tree",
        "DATA_NAME": "face",
        "PROMPT": "Change the Patagonia patch to a simple tree graphic patch",
        "SEG_PROMPT": "Patagonia logo patch on the fleece pocket",
        "LENS_SEG_PROMPT": "a fleece jacket",
        "LENS_CONE_HALF_ANGLE_DEG": "90.0",  # 60: man
        "LENS_DIVERSITY_X_WEIGHT": "30.0",
        "LENS_DIVERSITY_Y_VARIANCE_WEIGHT": "5.0",
        "TARGET_PROMPT": "a patch with a simple tree graphic on the fleece pocket",
        "STYLE_SOURCE_PROMPT": "man wearing jacket with a patch with the text 'PATAGONIA' and a mountain range",
        "STYLE_TARGET_PROMPT": "man wearing jacket with a patch with a simple, stylized tree graphic and no text",
    },
    # 7) Add a small silver nose stud piercing
    {
        "name": "nose_stud_silver",
        "DATA_NAME": "face",
        "PROMPT": "Add a small silver stud piercing to his nose",
        "SEG_PROMPT": "nose",
        "LENS_SEG_PROMPT": "a face",
        "LENS_DISTANCE_MULTIPLIERS": "2.0",
        "TARGET_PROMPT": "a nostril with a small silver stud piercing",
        "STYLE_SOURCE_PROMPT": "man with plain skin of the man's nose",
        "STYLE_TARGET_PROMPT": "man with skin with a small, glinting silver stud piercing",
    },
    # 8) Change the zipper pull to a bright red color
    {
        "name": "zipper_pull_red",
        "DATA_NAME": "face",
        "PROMPT": "Change the zipper pull to a bright red color",
        "SEG_PROMPT": "metallic zipper pull of the main zipper",
        "LENS_SEG_PROMPT": "a fleece jacket",
        "LENS_DISTANCE_MULTIPLIERS": "2.5",
        "TARGET_PROMPT": "bright red colored zipper pull",
        "STYLE_SOURCE_PROMPT": "man wearing jacket with dull, metallic zipper pull",
        "STYLE_TARGET_PROMPT": "man wearing jacket with vibrant, bright red zipper pull",
    },
    # 9) Change the material of the jacket to a denim jacket
    {
        "name": "jacket_denim",
        "DATA_NAME": "face",
        "PROMPT": "Change the jacket material to blue denim",
        "SEG_PROMPT": "entire fleece jacket",
        "LENS_SEG_PROMPT": "a fleece jacket",
        "LENS_DISTANCE_MULTIPLIERS": "3.0",
        "LENS_CONE_HALF_ANGLE_DEG": "90.0",
        "LENS_DIVERSITY_Y_VARIANCE_WEIGHT": "5.0",
        "TARGET_PROMPT": "a blue denim jacket",
        "STYLE_SOURCE_PROMPT": "man wearing jacket with textured grey speckled fleece fabric",
        "STYLE_TARGET_PROMPT": "man wearing jacket with classic blue denim twill fabric",
    },
    # 10) Add a graphic of a compass to the sleeve
    # {
    #     "name": "sleeve_compass",
    #     "DATA_NAME": "face",
    #     "PROMPT": "Add a graphic of a compass to the left sleeve of his fleece",
    #     "SEG_PROMPT": "fleece fabric of the left sleeve",
    #     "TARGET_PROMPT": "fleece sleeve with a small black compass graphic added",
    #     "STYLE_SOURCE_PROMPT": "man wearing jacket with plain grey speckled fleece fabric",
    #     "STYLE_TARGET_PROMPT": "man wearing jacket with plain grey fleece fabric with a detailed compass graphic",
    # },
    # 11) Change his hair style to a short, cropped style
    {
        "name": "hair_style_short",
        "DATA_NAME": "face",
        "PROMPT": "Change his hair style to a short, cropped look",
        "SEG_PROMPT": "hair",
        "LENS_SEG_PROMPT": "a face",
        "LENS_DISTANCE_MULTIPLIERS": "3.5",
        "LENS_CONE_HALF_ANGLE_DEG": "90.0",
        "LENS_DIVERSITY_Y_VARIANCE_WEIGHT": "5.0",
        "TARGET_PROMPT": "the man with a short, cropped haircut",
        "STYLE_SOURCE_PROMPT": "man with natural curly, wavy light brown hair",
        "STYLE_TARGET_PROMPT": "man with short, closely cropped light brown hair",
    },
    # 12) Change the color of his eyes to blue
    {
        "name": "eyes_blue",
        "DATA_NAME": "face",
        "PROMPT": "Change his eye color to blue",
        "SEG_PROMPT": "irises of the man's visible eyes",
        "LENS_SEG_PROMPT": "a face",
        "LENS_DISTANCE_MULTIPLIERS": "3.0",
        "TARGET_PROMPT": "eyes with blue irises",
        "STYLE_SOURCE_PROMPT": "man with brown eye irises",
        "STYLE_TARGET_PROMPT": "man with blue eye irises",
    },
    # 13) Add text to the chest pocket saying 'STAFF'
    {
        "name": "chest_text_staff",
        "DATA_NAME": "face",
        "PROMPT": "Add text that says 'STAFF' to the fleece chest pocket patch, below the logo",
        "SEG_PROMPT": "bottom area of the fleece chest pocket patch",
        "LENS_SEG_PROMPT": "a fleece jacket",
        "LENS_DISTANCE_MULTIPLIERS": "3.0",
        "LENS_CONE_HALF_ANGLE_DEG": "90.0",
        "LENS_DIVERSITY_Y_VARIANCE_WEIGHT": "5.0",
        "TARGET_PROMPT": "chest pocket patch with text 'STAFF' in block letters added",
        "STYLE_SOURCE_PROMPT": "man wearing jacket with plain patch surface",
        "STYLE_TARGET_PROMPT": "man wearing jacket with patch surface with detailed block text 'STAFF'",
    },
    # 14) Add a realistic-looking tattoo of an anchor to his neck
    # {
    #     "name": "neck_tattoo_anchor",
    #     "DATA_NAME": "face",
    #     "PROMPT": "Add a realistic-looking anchor tattoo to his neck",
    #     "SEG_PROMPT": "man's neck skin",
    #     "TARGET_PROMPT": "a neck with a detailed, small black anchor tattoo",
    #     "STYLE_SOURCE_PROMPT": "man with plain skin of the man's neck",
    #     "STYLE_TARGET_PROMPT": "man with skin with a detailed, black anchor tattoo that looks realistic",
    # },
    # 15) Replace the white wall on the right with a large, city-view window
    # {
    #     "name": "background_window_city",
    #     "DATA_NAME": "face",
    #     "PROMPT": "Replace the white wall on the right with a large window looking out at a city",
    #     "SEG_PROMPT": "a face",
    #     "TARGET_PROMPT": "a large window with a detailed city skyline view",
    #     "STYLE_SOURCE_PROMPT": "man standing in front of a flat, plain white painted surface",
    #     "STYLE_TARGET_PROMPT": "man standing in front of a detailed window view with city buildings and sky",
    # },

    ## bear
    # # 1) Change the yellow face markings to red
    # {
    #     "name": "markings_red",
    #     "DATA_NAME": "bear",
    #     "PROMPT": "Change the yellow face markings to red",
    #     "SEG_PROMPT": "markings on the bear's face",
    #     "TARGET_PROMPT": "red markings on the bear's face",
    #     "STYLE_SOURCE_PROMPT": "bear with yellow markings",
    #     "STYLE_TARGET_PROMPT": "bear with red markings",
    # },
    # # 2) Add a small plaid scarf around the neck area
    # {
    #     "name": "neck_scarf",
    #     "DATA_NAME": "bear",
    #     "PROMPT": "Add a small plaid scarf around the bear's neck",
    #     "SEG_PROMPT": "neck area of the bear statue",
    #     "TARGET_PROMPT": "bear statue wearing a plaid scarf",
    #     "STYLE_SOURCE_PROMPT": "bear statue with plain stone neck",
    #     "STYLE_TARGET_PROMPT": "bear statue with stone neck with a woven plaid scarf",
    # },
    # # 3) Put a pair of sunglasses on the bear's head
    # {
    #     "name": "wear_sunglasses",
    #     "DATA_NAME": "bear",
    #     "PROMPT": "Put a pair of sunglasses on the bear's head",
    #     "SEG_PROMPT": "head of the bear statue",
    #     "TARGET_PROMPT": "bear statue wearing sunglasses",
    #     "STYLE_SOURCE_PROMPT": "bare head of the bear statue",
    #     "STYLE_TARGET_PROMPT": "bear statue with head with a pair of dark sunglasses",
    # },
    # # 4) Change the material of the entire statue to bronze
    # {
    #     "name": "statue_bronze",
    #     "DATA_NAME": "bear",
    #     "PROMPT": "Change the material of the bear statue to bronze",
    #     "SEG_PROMPT": "entire bear statue",
    #     "TARGET_PROMPT": "bronze bear statue",
    #     "STYLE_SOURCE_PROMPT": "bear statue with grey stone texture",
    #     "STYLE_TARGET_PROMPT": "bear statue with aged bronze metal texture",
    # },
    # # 5) Change the material of the entire statue to clear ice
    # {
    #     "name": "statue_ice",
    #     "DATA_NAME": "bear",
    #     "PROMPT": "Change the material of the bear statue to clear ice",
    #     "SEG_PROMPT": "entire bear statue",
    #     "TARGET_PROMPT": "clear ice bear statue",
    #     "STYLE_SOURCE_PROMPT": "bear statue with grey stone texture",
    #     "STYLE_TARGET_PROMPT": "bear statue with transparent ice texture with reflections",
    # },
    # # 6) Make the facial expression of the head look much angrier
    # {
    #     "name": "expression_angry",
    #     "DATA_NAME": "bear",
    #     "PROMPT": "Make the facial expression of the bear much angrier",
    #     "SEG_PROMPT": "head of the bear statue",
    #     "TARGET_PROMPT": "angry-faced bear statue",
    #     "STYLE_SOURCE_PROMPT": "bear statue with neutral, open-mouthed expression",
    #     "STYLE_TARGET_PROMPT": "bear statue with furrowed brow and snarling teeth",
    # },
    # # 7) Replace the rock pedestal under the statue with a pile of gold bars
    # {
    #     "name": "pedestal_gold_bars",
    #     "DATA_NAME": "bear",
    #     "PROMPT": "Replace the rock pedestal with a pile of gold bars",
    #     "SEG_PROMPT": "rock pedestal under the bear statue",
    #     "TARGET_PROMPT": "pile of gold bars under the bear statue",
    #     "STYLE_SOURCE_PROMPT": "bear statue with grey rock texture",
    #     "STYLE_TARGET_PROMPT": "bear statue with stack of shiny gold bars",
    # },
    # # 8) Cover the entire statue with a layer of fuzzy green moss
    # # {
    # #     "name": "statue_mossy",
    # #     "DATA_NAME": "bear",
    # #     "PROMPT": "Cover the bear statue with a layer of fuzzy green moss",
    # #     "SEG_PROMPT": "entire bear statue",
    # #     "TARGET_PROMPT": "bear statue covered in green moss",
    # #     "STYLE_SOURCE_PROMPT": "bear statue with clean stone texture",
    # #     "STYLE_TARGET_PROMPT": "bear statue with mossy green fuzzy texture",
    # # },
    # # 9) Add a small, fabric backpack to the back area of the bear
    # {
    #     "name": "back_backpack",
    #     "DATA_NAME": "bear",
    #     "PROMPT": "Add a small backpack to the bear's back",
    #     "SEG_PROMPT": "back area of the bear statue",
    #     "TARGET_PROMPT": "bear statue with a small backpack",
    #     "STYLE_SOURCE_PROMPT": "bear statue with plain stone back",
    #     "STYLE_TARGET_PROMPT": "bear statue with fleece back with a small fabric backpack",
    # },
    # # 10) Replace the background plants with a detailed city skyline
    # # {
    # #     "name": "background_city",
    # #     "DATA_NAME": "bear",
    # #     "PROMPT": "Replace the background plants with a city skyline",
    # #     "SEG_PROMPT": "plants and dirt background ground",
    # #     "TARGET_PROMPT": "background showing a city skyline and concrete pavement",
    # #     "STYLE_SOURCE_PROMPT": "background with soil, leaves, and green foliage",
    # #     "STYLE_TARGET_PROMPT": "background with pavement, buildings, and clear city view",
    # # },
    # # 11) Change the color of the front paws to solid gold
    # {
    #     "name": "paws_gold",
    #     "DATA_NAME": "bear",
    #     "PROMPT": "Change the color of the bear's front paws to solid gold",
    #     "SEG_PROMPT": "front paws of the bear statue",
    #     "TARGET_PROMPT": "front paws colored in solid gold",
    #     "STYLE_SOURCE_PROMPT": "bear statue with grey stone color",
    #     "STYLE_TARGET_PROMPT": "bear statue with metallic gold color",
    # },
    # # 12) Add clear text that says 'GRIZZLY' below the statue on the rock face
    # # {
    # #     "name": "text_grizzly",
    # #     "DATA_NAME": "bear",
    # #     "PROMPT": "Add text that says 'GRIZZLY' below the statue on the rock",
    # #     "SEG_PROMPT": "flat surface of the rock pedestal",
    # #     "TARGET_PROMPT": "rock pedestal with 'GRIZZLY' text",
    # #     "STYLE_SOURCE_PROMPT": "rock pedestal with plain rock surface",
    # #     "STYLE_TARGET_PROMPT": "rock pedestal with rock surface with detailed black block text",
    # # },
]

TASKS_BY_NAME = {t["name"]: t for t in TASKS}

# ==========================
# Sweep configuration
# ==========================

SWEEP_CONFIG = {
    "name": WANDB_SWEEP_NAME,
    "method": "grid",  # grid = task × steps × view_config 조합
    "metric": {
        "name": "clip_dir_similarity",
        "goal": "maximize",
    },
    "parameters": {
        # dge.yaml에 실제로 존재하는 설정만 sweep에서 변경
        "task": {
            "values": [t["name"] for t in TASKS],
        },

        # (max_view_num, max_edit_view_num) 쌍 — "25_20" 형식
        "view_config": {
            "values": [f"{m}_{e}" for m, e in VIEW_CONFIG_PAIRS],
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
    - 그래도 못 찾으면 기존 heuristic (parent.parent) 사용
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

    # Fallback: 현재 구조 가정 (script/ 바로 위가 repo root)
    return Path(__file__).parent.parent.absolute()

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

    방금 만든 루트 `metrics.py`의 출력 형식을 그대로 따라 파싱한다.
    tqdm progress line과 실제 결과 라인을 구분하기 위해 ^와 MULTILINE을 사용.
    """
    result = {}
    patterns = {
        "clip_dir_consistency": r"^(?:\r)?CLIP directional consistency:\s*([-\d.]+)\s*$",
        "clip_f_scaled": r"^(?:\r)?CLIP_F \(scaled\):\s*([-\d.]+)\s*$",
        "clip_score": r"^(?:\r)?CLIP Score:\s*([-\d.]+)\s*$",
        "clip_dir_similarity": r"^(?:\r)?CLIP directional similarity:\s*([-\d.]+)\s*$",
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

    gpu = os.environ.get("SWEEP_GPU", GPU)
    task_name = getattr(cfg, "task", TASKS[0]["name"])
    task = TASKS_BY_NAME[task_name]
    prompt = task["PROMPT"]
    seg_prompt = task["SEG_PROMPT"]
    target_prompt = task["TARGET_PROMPT"]
    style_target_prompt = task.get("STYLE_TARGET_PROMPT", "")
    style_source_prompt = task.get("STYLE_SOURCE_PROMPT", "a Photo")
    # Per-task overrides for lens prompts / hyperparameters (fallback to global defaults)
    task_lens_edit_prompt = task.get("LENS_EDIT_PROMPT", prompt)
    task_lens_seg_prompt = task.get("LENS_SEG_PROMPT", seg_prompt)
    task_lens_distance_multipliers = task.get("LENS_DISTANCE_MULTIPLIERS", LENS_DISTANCE_MULTIPLIERS_DEFAULT)
    task_lens_cone_half_angle = task.get("LENS_CONE_HALF_ANGLE_DEG", LENS_CONE_HALF_ANGLE_DEG)
    task_lens_v_front_method = task.get("LENS_V_FRONT_METHOD", LENS_V_FRONT_METHOD)
    task_lens_diversity_x_weight = task.get("LENS_DIVERSITY_X_WEIGHT", LENS_DIVERSITY_X_WEIGHT)
    task_lens_diversity_y_variance_weight = task.get("LENS_DIVERSITY_Y_VARIANCE_WEIGHT", LENS_DIVERSITY_Y_VARIANCE_WEIGHT)

    data_name = task.get("DATA_NAME", DATA_NAME)
    data_source = f"{DATA_SOURCE_ROOT}/{DATA_TYPE}/{data_name}/"
    gs_source = (
        f"{GS_SOURCE_ROOT}/{DATA_TYPE}/{data_name}/point_cloud/iteration_30000/point_cloud.ply"
    )
    gt_dir = f"{GS_SOURCE_ROOT}/{DATA_TYPE}/{data_name}/{RENDER_SUBDIR}"

    # face: lens, bear: random
    edit_view_strategy = EDIT_VIEW_SELECTION_STRATEGY_BY_DATA_NAME.get(
        data_name, "lens"
    )

    # # steps: system.camera_update_per_step에만 사용 (trainer.max_steps는 고정)
    # steps = int(getattr(cfg, "steps", int(CAMERA_UPDATE_PER_STEP)))

    # view_config: "25_20" → max_view_num=25, max_edit_view_num=20
    view_config_str = getattr(cfg, "view_config", "25_20")
    max_view_num, max_edit_view_num = view_config_str.split("_")

    name = f"sweep/{DATA_TYPE}/{data_name}/view{view_config_str}/{task_name}"

    root_dir = get_root_dir()
    os.chdir(root_dir)

    # ---- Build launch command ----
    launch_cmd = [
        "python",
        "launch.py",
        "--train",
        "--config",
        CONFIG,
        "--gpu",
        gpu,
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
        f"system.loss.lambda_d={LAMBDA_D_DEFAULT}",
        f"system.loss.lambda_dds={LAMBDA_DDS}",
        f"system.dds_t_range=[{DDS_T_RANGE}]",
        f"system.dds_cfg_scale={DDS_CFG_SCALE}",
        f"system.loss.lambda_ism={LAMBDA_ISM_DEFAULT}",
        f"system.loss.use_sds={str(USE_SDS).lower()}",
        f"system.guidance.use_sds_dge={str(USE_SDS_DGE).lower()}",
        f"system.loss.lambda_sds={LAMBDA_SDS}",
        f"system.warp_refine_color_fit_steps={WARP_REFINE_COLOR_FIT_STEPS_DEFAULT}",
        f"system.use_warp_refine={str(USE_WARP_REFINE_DEFAULT).lower()}",
        f"system.save_image_grid_draw_texts={SAVE_IMAGE_GRID_DRAW_TEXTS}",
        f"data.max_view_num={max_view_num}",
        f"data.max_edit_view_num={max_edit_view_num}",
        f"system.multiview_edit_key_selection_strategy={MULTIVIEW_EDIT_KEY_SELECTION_STRATEGY}",
        f"system.use_multiview_edit={str(USE_MULTIVIEW_EDIT_DEFAULT).lower()}",
        f"system.use_gaussian_provenance={str(USE_GAUSSIAN_PROVENANCE_DEFAULT).lower()}",
        # f"system.camera_update_per_step={steps}",
        f"system.mask_update_at_step={MASK_UPDATE_AT_STEP}",
        f"system.mask_update_view_num={MASK_UPDATE_VIEW_NUM}",
        f"system.mask_view_population={MASK_VIEW_POPULATION}",
        f"system.mask_num_views={MASK_NUM_VIEWS}",
        f"system.prune_floater_at_step={PRUNE_FLOATER_AT_STEP}",
        f"exp_root_dir={EXP_ROOT_DIR}",
        f"name={name}",
        f"data.edit_view_selection_strategy={edit_view_strategy}",
        f"system.guidance.edit_view_selection_strategy={edit_view_strategy}",
        # Disable inner wandb logging to avoid nested runs
        "system.loggers.wandb.enable=false",
    ]

    if edit_view_strategy == "lens":
        launch_cmd.extend([
            f"data.lens_ply_path={gs_source}",
            f"data.lens_seg_prompt={task_lens_seg_prompt}",
            f"data.lens_edit_prompt={task_lens_edit_prompt}",
            f"data.lens_use_ip2p_scoring={LENS_USE_IP2P_SCORING}",
            f"data.lens_entropy_thresh={LENS_ENTROPY_THRESH}",
            f"data.lens_ip2p_steps={LENS_IP2P_STEPS}",
            f"data.lens_lambda_leak={LENS_LAMBDA_LEAK}",
            f"data.lens_lambda_ent={LENS_LAMBDA_ENT}",
            f"data.lens_ip2p_guidance_scale={LENS_IP2P_GUIDANCE_SCALE}",
            f"data.lens_ip2p_image_guidance_scale={LENS_IP2P_IMAGE_GUIDANCE_SCALE}",
            f"data.lens_hemisphere_only={LENS_HEMISPHERE_ONLY}",
            f"data.lens_cone_half_angle_deg={task_lens_cone_half_angle}",
            f"data.lens_v_front_method={task_lens_v_front_method}",
            f"data.lens_n_candidates={LENS_N_CANDIDATES}",
            f"data.lens_diversity_x_weight={task_lens_diversity_x_weight}",
            f"data.lens_diversity_y_variance_weight={task_lens_diversity_y_variance_weight}",
            f"data.lens_ip2p_batch_size={LENS_IP2P_BATCH_SIZE}",
            f"data.lens_distance_multipliers={task_lens_distance_multipliers}",
            f"system.guidance.camera_batch_size={CAMERA_BATCH_SIZE}",
        ])

    print(f"\n[Sweep] Running: task={task_name}, edit_view_strategy={edit_view_strategy}, view_config={view_config_str}")
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

    # ---- Run metrics (uses the metrics.py we just added at repo root) ----
    gpu_id = gpu.split(",")[0].strip()
    device = f"cuda:{gpu_id}" if DEVICE == "cuda" else DEVICE
    metrics_cmd = [
        "python",
        "metrics.py",
        "--gt",
        gt_dir,
        "--render",
        str(render_dir),
        "--device",
        device,
        "--interval",
        str(INTERVAL),
        "--style_source_prompt",
        style_source_prompt,
        "--style_target_prompt",
        style_target_prompt,
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
    parser = argparse.ArgumentParser(
        description="WandB Sweep for DGE camera-selection"
    )
    parser.add_argument("--create", action="store_true", help="Create a new sweep")
    parser.add_argument("--agent", action="store_true", help="Start a sweep agent")
    parser.add_argument("--sweep_id", type=str, default=None, help="Sweep ID")
    parser.add_argument(
        "--count", type=int, default=None, help="Max number of runs per agent"
    )
    parser.add_argument(
        "--gpu", type=str, default=None, help="GPU id for this specific agent"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPUs (e.g., '0,1,2,3') to run parallel agents",
    )
    args = parser.parse_args()

    sweep_id = args.sweep_id

    if args.create:
        sweep_id = wandb.sweep(SWEEP_CONFIG, project=WANDB_PROJECT)
        print(f"\n[Created] Sweep ID: {sweep_id}")

    if args.agent:
        if not sweep_id:
            print("Error: --sweep_id is required.")
            sys.exit(1)

        # 여러 GPU를 한 번에 넣었을 경우 (예: --gpus 0,1,2,3)
        if args.gpus:
            gpu_list = [s.strip() for s in args.gpus.split(",") if s.strip()]
            print(f"Launching {len(gpu_list)} agents on GPUs: {gpu_list}")

            processes = []
            for g in gpu_list:
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = g
                env["SWEEP_GPU"] = "0"  # 프로세스 내부에서는 0번으로 접근

                cmd = [
                    sys.executable,
                    "-u",
                    sys.argv[0],
                    "--agent",
                    "--sweep_id",
                    sweep_id,
                    "--gpu",
                    "0",
                ]
                if args.count:
                    cmd += ["--count", str(args.count)]

                p = subprocess.Popen(cmd, env=env)
                processes.append(p)

            for p in processes:
                p.wait()
            sys.exit(0)

        # 단일 에이전트 실행부
        gpu_to_use = args.gpu if args.gpu else "0"
        os.environ["SWEEP_GPU"] = gpu_to_use

        print(
            f"Agent started on Physical GPU {os.environ.get('CUDA_VISIBLE_DEVICES', 'Unknown')}"
        )

        wandb.agent(
            sweep_id,
            function=train_and_evaluate,
            project=WANDB_PROJECT,
            count=args.count,
        )


if __name__ == "__main__":
    main()

