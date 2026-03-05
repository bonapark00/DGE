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
  python script/camera-selection/in2n/sweep.py --create

  # 2) Run agent(s) – each agent picks a config and runs train+metrics
  python script/camera-selection/in2n/sweep.py --agent --sweep_id <SWEEP_ID>

  # Use specific GPU(s): single GPU or comma-separated for multi-GPU per run
  python script/camera-selection/in2n/sweep.py --agent --sweep_id <ID> --gpu 0
  python script/camera-selection/in2n/sweep.py --agent --sweep_id <ID> --gpu 0,1,2

  # Run multiple agents in parallel (one process per GPU, single command): ## 이거로!!!
  python script/camera-selection/in2n/sweep.py --agent --sweep_id <ID> --gpus 0,1,2

  # Or manually in separate terminals:
  CUDA_VISIBLE_DEVICES=0 python script/camera-selection/in2n/sweep.py --agent --sweep_id <ID> --gpu 0
  CUDA_VISIBLE_DEVICES=1 python script/camera-selection/in2n/sweep.py --agent --sweep_id <ID> --gpu 1

  # Or do both in one shot (sequential, single machine):
  python script/camera-selection/in2n/sweep.py --create --agent
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
MASK_UPDATE_VIEW_NUM = "5"
PRUNE_FLOATER_AT_STEP = "-1"
MAX_STEPS = "1500"

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
LENS_IP2P_STEPS = "5"
LENS_IP2P_GUIDANCE_SCALE = "12.5"
LENS_IP2P_IMAGE_GUIDANCE_SCALE = "1.5"
LENS_DISTANCE_MULTIPLIERS = "2.0,2.5,3.0,4.0,5.0,6.0"
LENS_CONE_HALF_ANGLE_DEG = "60"
LENS_LAMBDA_LEAK = "1.5"
LENS_LAMBDA_ENT = "15.0"

EDIT_VIEW_SELECTION_STRATEGY_DEFAULT = "lens"


INTERVAL = 1
DEVICE = "cuda"
LAMBDA_ISM_DEFAULT = 0.0001
USE_WARP_REFINE_DEFAULT = True
WARP_REFINE_COLOR_FIT_STEPS_DEFAULT = 100 

WANDB_PROJECT = "dge-orig"
WANDB_SWEEP_NAME = f"sweep-2/ours/{DATA_TYPE}"

# ==========================
# Prompt combinations for 3d-ovs; sweep can vary over these via "task" parameter
# (각 TASK는 launch_and_metrics.py의 PROMPT / SEG_PROMPT / TARGET_PROMPT 조합을 반영)
# ==========================

# STYLE_SOURCE_PROMPT = 편집 전(원본), STYLE_TARGET_PROMPT = 편집 후(목표)
TASKS = [
# 1) Change the red flower into a sunflower
    {
        "name": "flower_to_sunflower",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Change the red flower into a sunflower",
        "SEG_PROMPT": "red flower",
        "TARGET_PROMPT": "sunflower",
        "STYLE_SOURCE_PROMPT": "red flower",
        "STYLE_TARGET_PROMPT": "sunflower",
    },
    # 2) Change the shampoo bottle into a wine bottle
    {
        "name": "bottle_to_wine",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Change the shampoo bottle into a wine bottle",
        "SEG_PROMPT": "shampoo bottle",
        "TARGET_PROMPT": "wine bottle",
        "STYLE_SOURCE_PROMPT": "shampoo bottle",
        "STYLE_TARGET_PROMPT": "wine bottle",
    },
    # 3) Make the pooh wear a black hat
    {
        "name": "pooh_hat",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Make the pooh wear a black hat",
        "SEG_PROMPT": "head of the pooh",
        "TARGET_PROMPT": "black hat",
        "STYLE_SOURCE_PROMPT": "head of the pooh",
        "STYLE_TARGET_PROMPT": "pooh wearing a black hat",
    },
    # 4) Change the shaving razor into a toy car
    {
        "name": "razor_to_car",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Change the shaving razor into a toy car",
        "SEG_PROMPT": "shaving razor",
        "TARGET_PROMPT": "toy car",
        "STYLE_SOURCE_PROMPT": "shaving razor",
        "STYLE_TARGET_PROMPT": "toy car",
    },
    # 5) Change the jam jar into a honey pot
    {
        "name": "jar_to_honey",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Change the jam jar into a honey pot",
        "SEG_PROMPT": "small jar on the right",
        "TARGET_PROMPT": "honey pot",
        "STYLE_SOURCE_PROMPT": "small jar on the right",
        "STYLE_TARGET_PROMPT": "honey pot",
    },
    # 6) Change the pooh's sweater color to green
    {
        "name": "sweater_green",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Change the pooh's sweater color to green",
        "SEG_PROMPT": "red sweater of the pooh",
        "TARGET_PROMPT": "green sweater of the pooh",
        "STYLE_SOURCE_PROMPT": "red sweater of the pooh",
        "STYLE_TARGET_PROMPT": "green sweater of the pooh",
    },
    # 7) Make the pooh look like a tiger
    {
        "name": "pooh_tigger",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Make the pooh look like a tiger",
        "SEG_PROMPT": "pooh",
        "TARGET_PROMPT": "tiger",
        "STYLE_SOURCE_PROMPT": "pooh",
        "STYLE_TARGET_PROMPT": "tiger with orange and black stripes",
    },
    # 8) Replace the red flower with a blue butterfly
    {
        "name": "flower_to_butterfly",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Replace the red flower with a blue butterfly",
        "SEG_PROMPT": "red flower",
        "TARGET_PROMPT": "blue butterfly",
        "STYLE_SOURCE_PROMPT": "red flower",
        "STYLE_TARGET_PROMPT": "blue butterfly",
    },
    # 9) Put a red bowtie on the pooh
    {
        "name": "pooh_bowtie",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Put a red bowtie on the pooh",
        "SEG_PROMPT": "neck of the pooh",
        "TARGET_PROMPT": "red bowtie",
        "STYLE_SOURCE_PROMPT": "neck of the pooh",
        "STYLE_TARGET_PROMPT": "pooh wearing a red bowtie",
    },
    # 10) Change the shampoo bottle into a cactus
    {
        "name": "bottle_to_cactus",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Change the shampoo bottle into a cactus",
        "SEG_PROMPT": "shampoo bottle",
        "TARGET_PROMPT": "cactus",
        "STYLE_SOURCE_PROMPT": "shampoo bottle",
        "STYLE_TARGET_PROMPT": "cactus in a terracotta pot",
    },
    # 11) Add a pair of headphones to the pooh
    {
        "name": "pooh_headphones",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Add a pair of headphones to the pooh",
        "SEG_PROMPT": "head of the pooh",
        "TARGET_PROMPT": "headphones",
        "STYLE_SOURCE_PROMPT": "head of the pooh",
        "STYLE_TARGET_PROMPT": "pooh wearing modern headphones",
    },
    # 12) Change the shaving razor into a banana
    {
        "name": "razor_to_banana",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Change the shaving razor into a banana",
        "SEG_PROMPT": "shaving razor",
        "TARGET_PROMPT": "banana",
        "STYLE_SOURCE_PROMPT": "shaving razor",
        "STYLE_TARGET_PROMPT": "ripe yellow banana",
    },
    # 13) Make the pooh wear sunglasses
    {
        "name": "pooh_sunglasses",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Make the pooh wear sunglasses",
        "SEG_PROMPT": "face of the pooh",
        "TARGET_PROMPT": "sunglasses",
        "STYLE_SOURCE_PROMPT": "face of the pooh",
        "STYLE_TARGET_PROMPT": "pooh wearing cool sunglasses",
    },
    # 14) Change the red sweater into a tuxedo
    {
        "name": "sweater_to_tuxedo",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Change the red sweater into a tuxedo",
        "SEG_PROMPT": "red sweater of the pooh",
        "TARGET_PROMPT": "tuxedo",
        "STYLE_SOURCE_PROMPT": "red sweater of the pooh",
        "STYLE_TARGET_PROMPT": "pooh wearing a formal tuxedo",
    },
    # 15) Change the jam jar into a gold trophy
    {
        "name": "jar_to_trophy",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Change the jam jar into a gold trophy",
        "SEG_PROMPT": "small jar",
        "TARGET_PROMPT": "gold trophy",
        "STYLE_SOURCE_PROMPT": "small jar",
        "STYLE_TARGET_PROMPT": "shining gold trophy",
    },
## blue_sofa_extended
    # 11) Change the sofa material to brown leather
    {
        "name": "sofa_leather",
        "DATA_NAME": "blue_sofa",
        "PROMPT": "Change the blue sofa to a brown leather texture",
        "SEG_PROMPT": "blue sofa background",
        "TARGET_PROMPT": "rustic brown leather sofa background",
        "STYLE_SOURCE_PROMPT": "blue sofa background",
        "STYLE_TARGET_PROMPT": "rustic brown leather sofa background",
    },
    # 12) Turn the plush toy into a panda
    {
        "name": "plush_panda",
        "DATA_NAME": "blue_sofa",
        "PROMPT": "Turn the plush toy into a black and white panda",
        "SEG_PROMPT": "yellow plush toy",
        "TARGET_PROMPT": "black and white panda plush toy",
        "STYLE_SOURCE_PROMPT": "yellow plush toy",
        "STYLE_TARGET_PROMPT": "black and white panda plush toy",
    },
    # 13) Change sunglasses lenses to purple gradient
    {
        "name": "glasses_purple_lens",
        "DATA_NAME": "blue_sofa",
        "PROMPT": "Change the sunglasses lenses to a purple gradient tint",
        "SEG_PROMPT": "dark lenses of the sunglasses",
        "TARGET_PROMPT": "semi-transparent purple gradient lenses",
        "STYLE_SOURCE_PROMPT": "dark lenses of the sunglasses",
        "STYLE_TARGET_PROMPT": "semi-transparent purple gradient lenses",
    },
    # 14) Change the remote controller body to matte black
    {
        "name": "remote_black",
        "DATA_NAME": "blue_sofa",
        "PROMPT": "Change the remote controller body to sleek matte black",
        "SEG_PROMPT": "white body of the remote controller",
        "TARGET_PROMPT": "sleek matte black body of the remote controller",
        "STYLE_SOURCE_PROMPT": "white body of the remote controller",
        "STYLE_TARGET_PROMPT": "sleek matte black body of the remote controller",
    },
    # 15) Make the perfume bottle look like frosted glass
    {
        "name": "perfume_frosted",
        "DATA_NAME": "blue_sofa",
        "PROMPT": "Change the perfume bottle to frosted glass material",
        "SEG_PROMPT": "clear glass perfume bottle",
        "TARGET_PROMPT": "semi-transparent frosted glass perfume bottle",
        "STYLE_SOURCE_PROMPT": "clear glass perfume bottle",
        "STYLE_TARGET_PROMPT": "semi-transparent frosted glass perfume bottle",
    },
    # 16) Change the JBL logo to a music note icon
    {
        "name": "speaker_logo_note",
        "DATA_NAME": "blue_sofa",
        "PROMPT": "Replace the logo on the speaker with a musical note icon",
        "SEG_PROMPT": "JBL logo on the speaker",
        "TARGET_PROMPT": "embossed musical note icon on the speaker",
        "STYLE_SOURCE_PROMPT": "JBL logo on the speaker",
        "STYLE_TARGET_PROMPT": "embossed musical note icon on the speaker",
    },
    # 17) Add a red bowtie to the plush toy
    {
        "name": "plush_bowtie",
        "DATA_NAME": "blue_sofa",
        "PROMPT": "Add a small red bowtie to the plush toy",
        "SEG_PROMPT": "neck area of the plush toy",
        "TARGET_PROMPT": "a small red silk bowtie on the plush toy",
        "STYLE_SOURCE_PROMPT": "neck area of the plush toy",
        "STYLE_TARGET_PROMPT": "a small red silk bowtie on the plush toy",
    },
    # 18) Change the speaker fabric to camouflage pattern
    {
        "name": "speaker_camouflage",
        "DATA_NAME": "blue_sofa",
        "PROMPT": "Change the speaker's fabric to a camouflage pattern",
        "SEG_PROMPT": "grey fabric of the speaker",
        "TARGET_PROMPT": "green and brown camouflage fabric",
        "STYLE_SOURCE_PROMPT": "grey fabric of the speaker",
        "STYLE_TARGET_PROMPT": "green and brown camouflage fabric",
    },
    # 19) Make the sunglasses look like cyberpunk visors
    {
        "name": "glasses_cyberpunk",
        "DATA_NAME": "blue_sofa",
        "PROMPT": "Change the sunglasses into glowing cyberpunk visors",
        "SEG_PROMPT": "black sunglasses",
        "TARGET_PROMPT": "futuristic glowing cyberpunk visor glasses",
        "STYLE_SOURCE_PROMPT": "black sunglasses",
        "STYLE_TARGET_PROMPT": "futuristic glowing cyberpunk visor glasses",
    },
    # 20) Change the liquid inside the bottle to glowing lava
    {
        "name": "perfume_lava",
        "DATA_NAME": "blue_sofa",
        "PROMPT": "Make the liquid inside the bottle glow like molten lava",
        "SEG_PROMPT": "yellow liquid inside the perfume bottle",
        "TARGET_PROMPT": "glowing orange and red molten lava liquid",
        "STYLE_SOURCE_PROMPT": "yellow liquid inside the perfume bottle",
        "STYLE_TARGET_PROMPT": "glowing orange and red molten lava liquid",
    },
   ## room_extended
    # 11) Change the dinosaur figure's color to blue
    {
        "name": "dino_blue",
        "DATA_NAME": "room",
        "PROMPT": "Change the dinosaur figure's color to blue",
        "SEG_PROMPT": "brown dinosaur figure",
        "TARGET_PROMPT": "vibrant blue dinosaur figure",
        "STYLE_SOURCE_PROMPT": "brown dinosaur figure",
        "STYLE_TARGET_PROMPT": "vibrant blue dinosaur figure",
    },
    # 12) Change the rabbit figure to a rough stone texture
    {
        "name": "rabbit_stone",
        "DATA_NAME": "room",
        "PROMPT": "Change the rabbit figure to a rough stone texture",
        "SEG_PROMPT": "grey rabbit figure",
        "TARGET_PROMPT": "rough stone carved rabbit figure",
        "STYLE_SOURCE_PROMPT": "grey rabbit figure",
        "STYLE_TARGET_PROMPT": "rough stone carved rabbit figure",
    },
    # 13) Replace the baseball with a small potted succulent plant
    {
        "name": "ball_succulent",
        "DATA_NAME": "room",
        "PROMPT": "Replace the baseball with a small potted succulent plant",
        "SEG_PROMPT": "white baseball",
        "TARGET_PROMPT": "small potted succulent plant",
        "STYLE_SOURCE_PROMPT": "white baseball",
        "STYLE_TARGET_PROMPT": "small potted succulent plant",
    },
    # 14) Add a tiny, delicate flower behind the rabbit figure's ear
    {
        "name": "rabbit_flower",
        "DATA_NAME": "room",
        "PROMPT": "Add a tiny, delicate flower behind the rabbit figure's ear",
        "SEG_PROMPT": "head of the grey rabbit figure",
        "TARGET_PROMPT": "grey rabbit figure wearing a tiny flower behind its ear",
        "STYLE_SOURCE_PROMPT": "head of the grey rabbit figure",
        "STYLE_TARGET_PROMPT": "grey rabbit figure wearing a tiny flower behind its ear",
    },
    # 15) Make the dinosaur figure's eyes glow with a red light
    {
        "name": "dino_eyes_glow",
        "DATA_NAME": "room",
        "PROMPT": "Make the dinosaur figure's eyes glow with a red light",
        "SEG_PROMPT": "eyes of the brown dinosaur figure",
        "TARGET_PROMPT": "brown dinosaur figure with glowing red eyes",
        "STYLE_SOURCE_PROMPT": "eyes of the brown dinosaur figure",
        "STYLE_TARGET_PROMPT": "brown dinosaur figure with glowing red eyes",
    },
    # 16) Change the chicken's red wattle and comb to polished wood
    {
        "name": "chicken_parts_wood",
        "DATA_NAME": "room",
        "PROMPT": "Change the chicken's red wattle and comb to polished wood",
        "SEG_PROMPT": "red wattle and comb of the rubber chicken",
        "TARGET_PROMPT": "red wattle and comb of the rubber chicken, now made of polished wood",
        "STYLE_SOURCE_PROMPT": "red wattle and comb of the rubber chicken",
        "STYLE_TARGET_PROMPT": "red wattle and comb of the rubber chicken, now made of polished wood",
    },
    # 17) Fill the empty space in the basket with colorful glass marbles
    {
        "name": "basket_marbles",
        "DATA_NAME": "room",
        "PROMPT": "Fill the empty space in the basket with colorful glass marbles",
        "SEG_PROMPT": "inside of the woven basket",
        "TARGET_PROMPT": "woven basket filled with colorful glass marbles",
        "STYLE_SOURCE_PROMPT": "inside of the woven basket",
        "STYLE_TARGET_PROMPT": "woven basket filled with colorful glass marbles",
    },
    # 18) Replace the rubber chicken with a small, stylized decorative skull
    {
        "name": "chicken_skull",
        "DATA_NAME": "room",
        "PROMPT": "Replace the rubber chicken with a small, stylized, decorative skull",
        "SEG_PROMPT": "yellow rubber chicken",
        "TARGET_PROMPT": "small, stylized, decorative skull",
        "STYLE_SOURCE_PROMPT": "yellow rubber chicken",
        "STYLE_TARGET_PROMPT": "small, stylized, decorative skull",
    },
    # 19) Make the baseball look like it's crudely carved from wood
    {
        "name": "ball_carved",
        "DATA_NAME": "room",
        "PROMPT": "Make the baseball look like it's crudely carved from a single piece of wood",
        "SEG_PROMPT": "white baseball",
        "TARGET_PROMPT": "a ball crudely carved from wood to look like a baseball",
        "STYLE_SOURCE_PROMPT": "white baseball",
        "STYLE_TARGET_PROMPT": "a ball crudely carved from wood to look like a baseball",
    },
    # 20) Add a tiny magnifying glass resting on the floor next to the dinosaur figure
    {
        "name": "dino_magnifying_glass",
        "DATA_NAME": "room",
        "PROMPT": "Add a tiny magnifying glass resting on the floor next to the dinosaur figure",
        "SEG_PROMPT": "floor area next to the dinosaur",
        "TARGET_PROMPT": "floor area next to the dinosaur containing a tiny magnifying glass",
        "STYLE_SOURCE_PROMPT": "floor area next to the dinosaur",
        "STYLE_TARGET_PROMPT": "floor area next to the dinosaur containing a tiny magnifying glass",
    },
]
TASKS_BY_NAME = {t["name"]: t for t in TASKS}

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
        # "edit_view_selection_strategy": {
        #     "values": ["lens", "random"],
        # },
        # "lambda_ism": {
        #     "values": [0.0, 0.0001],
        # },
        # Warp-refine edit strength (0~1, blend between warped input & IP2P output).
        # "warp_refine_ip2p_strength": {
        #     "values": [0.5, 0.75, 1.0],
        # },
        # Lens sampling distance multipliers around COLMAP mean direction
        "lens_distance_multipliers": {
            "values": [
                "2.0,2.5,3.0,4.0,5.0,6.0",   # 기본값 (launch_and_metrics.py)
                "1.5,2.0,2.5,3.0,3.5,4.0",   # 더 근접한 뷰 위주
                "3.0,4.0,5.0,6.0",           # 더 먼 뷰 위주
            ],
        },
        # Cone constraint (in degrees) around COLMAP mean direction
        "lens_cone_half_angle_deg": {
            "values": [45, 60, 75],
        },
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
    """Find save directory from launch.py stdout."""
    pattern = r"Test results saved to (.+)"
    matches = re.findall(pattern, launch_output)
    if matches:
        p = Path(matches[-1].strip())
        if p.exists():
            return p

    # Fallback: latest trial dir
    exp_root = Path("/data/users/jaeyeonpark/DGE-outputs")
    for exp_dir in [exp_root / name, exp_root / name / str(MAX_VIEW_NUM)]:
        if exp_dir.exists():
            trial_dirs = sorted(
                [d for d in exp_dir.iterdir() if d.is_dir() and "@" in d.name],
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )
            if trial_dirs:
                save = trial_dirs[0] / "save"
                if save.exists():
                    return save
    return None


def find_render_directory(save_dir: Path) -> Optional[Path]:
    render_dir = save_dir / f"it{MAX_STEPS}-test"
    if render_dir.exists():
        return render_dir
    test_dirs = list(save_dir.glob("it*-test"))
    return test_dirs[0] if test_dirs else None


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

    data_name = task.get("DATA_NAME", DATA_NAME)
    data_source = f"{DATA_SOURCE_ROOT}/{DATA_TYPE}/{data_name}/"
    gs_source = f"{GS_SOURCE_ROOT}/{DATA_TYPE}/{data_name}/point_cloud/iteration_30000/point_cloud.ply"
    gt_dir = f"{GS_SOURCE_ROOT}/{DATA_TYPE}/{data_name}/{RENDER_SUBDIR}"

    name = (
        f"sweep-2/ours/{DATA_TYPE}/{data_name}"
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
        f"data.max_view_num={MAX_VIEW_NUM}",
        f"data.max_edit_view_num={MAX_EDIT_VIEW_NUM}",
        f"system.multiview_edit_key_selection_strategy={MULTIVIEW_EDIT_KEY_SELECTION_STRATEGY}",
        f"system.use_multiview_edit={str(USE_MULTIVIEW_EDIT_DEFAULT).lower()}",
        f"system.use_gaussian_provenance={str(USE_GAUSSIAN_PROVENANCE_DEFAULT).lower()}",
        f"data.edit_view_selection_strategy={strategy}",
        f"data.lens_ply_path={gs_source}",
        f"data.lens_seg_prompt={seg_prompt}",
        f"data.lens_edit_prompt={prompt}",
        f"data.lens_use_ip2p_scoring={LENS_USE_IP2P_SCORING}",
        f"data.lens_ip2p_steps={LENS_IP2P_STEPS}",
        f"data.lens_ip2p_guidance_scale={LENS_IP2P_GUIDANCE_SCALE}",
        f"data.lens_ip2p_image_guidance_scale={LENS_IP2P_IMAGE_GUIDANCE_SCALE}",
        f"data.lens_distance_multipliers={LENS_DISTANCE_MULTIPLIERS}",
        f"data.lens_cone_half_angle_deg={LENS_CONE_HALF_ANGLE_DEG}",
        f"data.lens_lambda_leak={LENS_LAMBDA_LEAK}",
        f"data.lens_lambda_ent={LENS_LAMBDA_ENT}",
        f"system.guidance.edit_view_selection_strategy={strategy}",
        f"system.camera_update_per_step={CAMERA_UPDATE_PER_STEP}",
        f"system.mask_update_at_step={MASK_UPDATE_AT_STEP}",
        f"system.mask_update_view_num={MASK_UPDATE_VIEW_NUM}",
        f"system.prune_floater_at_step={PRUNE_FLOATER_AT_STEP}",
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