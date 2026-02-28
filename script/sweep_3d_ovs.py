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
  python script/sweep_3d_ovs.py --create

  # 2) Run agent(s) – each agent picks a config and runs train+metrics
  python script/sweep_3d_ovs.py --agent --sweep_id <SWEEP_ID>



  # Run multiple agents in parallel (one process per GPU, single command)
  python script/sweep_3d_ovs.py --agent --sweep_id <ID> --gpus 0,1,2

  # Or manually in separate terminals:
  CUDA_VISIBLE_DEVICES=0 python script/sweep_3d_ovs.py --agent --sweep_id <ID> --gpu 0
  CUDA_VISIBLE_DEVICES=1 python script/sweep_3d_ovs.py --agent --sweep_id <ID> --gpu 1

  # Or do both in one shot (sequential, single machine):
  python script/sweep_3d_ovs.py --create --agent
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

CONFIG = "configs/dge.yaml"
GPU = "0,1,2,3"  # default; override with --gpu (e.g. "0" or "0,1,2" for multi-GPU per run)

DATA_TYPE = "3d-ovs"
DATA_NAME = "covered_desk"  # default; may be overridden per TASK via DATA_NAME
DATA_SOURCE_ROOT = "/data/users/jaeyeonpark/dataset"
GS_SOURCE_ROOT = "/data/users/jaeyeonpark/3dgs-trained"
RENDER_SUBDIR = "colmap_render_full"

GUIDANCE_SCALE = "12.5"
MASK_THRES = "0.6"
MAX_VIEW_NUM = "25"
CAMERA_UPDATE_PER_STEP = "1500"
MAX_STEPS = "1500"

INTERVAL = 1
DEVICE = "cuda"

WANDB_PROJECT = "dge-orig"
WANDB_SWEEP_NAME = f"sweep/{DATA_TYPE}"

# 출력 루트: 여기 아래에 sweep/3d-ovs/... 가 생성됨 (dge.yaml exp_root_dir 오버라이드)
EXP_ROOT_DIR = "/data/users/jaeyeonpark/DGE-outputs"

# ==========================
# TASK 정의 (원본과 동일)
# ==========================

# STYLE_SOURCE_PROMPT = 편집 전(원본), STYLE_TARGET_PROMPT = 편집 후(목표)
TASKS = [
    ## covered_desk
    # 1) Change the shaving razor into an apple
    {
        "name": "razor_to_apple",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Change the shaving razor into an apple",
        "SEG_PROMPT": "shaving razor",
        "TARGET_PROMPT": "apple",
        "STYLE_SOURCE_PROMPT": "shaving razor",
        "STYLE_TARGET_PROMPT": "apple",
    },
    # 2) Change the shampoo bottle into an apple
    {
        "name": "bottle_to_apple",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Change the shampoo bottle into an apple",
        "SEG_PROMPT": "shampoo bottle",
        "TARGET_PROMPT": "apple",
        "STYLE_SOURCE_PROMPT": "shampoo bottle",
        "STYLE_TARGET_PROMPT": "apple",
    },
    # 3) Give the pooh a pair of pants
    {
        "name": "pooh_pants",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Give the pooh a pair of pants",
        "SEG_PROMPT": "pooh",
        "TARGET_PROMPT": "pants",
        "STYLE_SOURCE_PROMPT": "pooh",
        "STYLE_TARGET_PROMPT": "pooh wearing pants",
    },
    # 4) Make the pooh look like a penguin
    {
        "name": "pooh_penguin",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Make the pooh look like a penguin",
        "SEG_PROMPT": "pooh",
        "TARGET_PROMPT": "penguin",
        "STYLE_SOURCE_PROMPT": "pooh",
        "STYLE_TARGET_PROMPT": "penguin",
    },
    # 5) Change the red sweater into a leather jacket
    {
        "name": "sweater_to_leather_jacket",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Change the red sweater into a leather jacket",
        "SEG_PROMPT": "red sweater of the pooh",
        "TARGET_PROMPT": "leather jacket of the pooh",
        "STYLE_SOURCE_PROMPT": "red sweater of the pooh",
        "STYLE_TARGET_PROMPT": "leather jacket of the pooh",
    },
    # 6) Make the pooh wear sunglasses on his face
    {
        "name": "pooh_sunglasses",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Make the pooh wear sunglasses on his face",
        "SEG_PROMPT": "head of the pooh",
        "TARGET_PROMPT": "sunglasses",
        "STYLE_SOURCE_PROMPT": "head of the pooh",
        "STYLE_TARGET_PROMPT": "pooh wearing sunglasses",
    },
    # 7) Change the pooh's sweater color to blue
    {
        "name": "sweater_blue",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Change the pooh's sweater color to blue",
        "SEG_PROMPT": "red sweater of the pooh",
        "TARGET_PROMPT": "blue sweater of the pooh",
        "STYLE_SOURCE_PROMPT": "red sweater of the pooh",
        "STYLE_TARGET_PROMPT": "blue sweater of the pooh",
    },
    # 8) Add flower pattern to the pooh's sweater
    {
        "name": "sweater_flower",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Add flower pattern to the pooh's sweater",
        "SEG_PROMPT": "red sweater of the pooh",
        "TARGET_PROMPT": "sweater of the pooh with flower pattern",
        "STYLE_SOURCE_PROMPT": "red sweater of the pooh",
        "STYLE_TARGET_PROMPT": "sweater of the pooh with flower pattern",
    },
    # 9) Make the pooh look like a panda
    {
        "name": "pooh_panda",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Make the pooh look like a panda",
        "SEG_PROMPT": "pooh",
        "TARGET_PROMPT": "panda",
        "STYLE_SOURCE_PROMPT": "pooh",
        "STYLE_TARGET_PROMPT": "panda",
    },
    # 10) Make the pooh look like a robot
    {
        "name": "pooh_robot",
        "DATA_NAME": "covered_desk",
        "PROMPT": "Make the pooh look like a robot",
        "SEG_PROMPT": "pooh",
        "TARGET_PROMPT": "robot",
        "STYLE_SOURCE_PROMPT": "pooh",
        "STYLE_TARGET_PROMPT": "robot",
    },
    ## blue_sofa
    # 1) Change the plush toy's color to pink
    {
        "name": "plush_pink",
        "DATA_NAME": "blue_sofa",
        "PROMPT": "Change the plush toy's color to pink",
        "SEG_PROMPT": "yellow plush toy",
        "TARGET_PROMPT": "pink plush toy",
        "STYLE_SOURCE_PROMPT": "yellow plush toy",
        "STYLE_TARGET_PROMPT": "pink plush toy",
    },
    # 2) Make the plush toy wear a tiny hat
    {
        "name": "plush_hat",
        "DATA_NAME": "blue_sofa",
        "PROMPT": "Make the plush toy wear a tiny hat",
        "SEG_PROMPT": "head of the plush toy",
        "TARGET_PROMPT": "head of the plush toy wearing a tiny party hat",
        "STYLE_SOURCE_PROMPT": "head of the plush toy",
        "STYLE_TARGET_PROMPT": "head of the plush toy wearing a tiny party hat",
    },
    # 3) Change the sunglasses to red frames
    {
        "name": "glasses_red",
        "DATA_NAME": "blue_sofa",
        "PROMPT": "Change the sunglasses to red frames",
        "SEG_PROMPT": "black frames of the sunglasses",
        "TARGET_PROMPT": "bright red frames of the sunglasses",
        "STYLE_SOURCE_PROMPT": "black frames of the sunglasses",
        "STYLE_TARGET_PROMPT": "bright red frames of the sunglasses",
    },
    # 4) Replace the JBL speaker with a vintage radio
    {
        "name": "speaker_vintage",
        "DATA_NAME": "blue_sofa",
        "PROMPT": "Replace the JBL speaker with a vintage radio",
        "SEG_PROMPT": "grey JBL speaker",
        "TARGET_PROMPT": "vintage wooden radio",
        "STYLE_SOURCE_PROMPT": "grey JBL speaker",
        "STYLE_TARGET_PROMPT": "vintage wooden radio",
    },
    # 5) Change the perfume liquid color to blue
    {
        "name": "perfume_blue",
        "DATA_NAME": "blue_sofa",
        "PROMPT": "Change the perfume liquid color to blue",
        "SEG_PROMPT": "yellow liquid inside the perfume bottle",
        "TARGET_PROMPT": "ocean blue liquid inside the perfume bottle",
        "STYLE_SOURCE_PROMPT": "yellow liquid inside the perfume bottle",
        "STYLE_TARGET_PROMPT": "ocean blue liquid inside the perfume bottle",
    },
    # 6) Add a digital clock display to the remote controller
    {
        "name": "remote_digital",
        "DATA_NAME": "blue_sofa",
        "PROMPT": "Make the remote controller screen glow neon green",
        "SEG_PROMPT": "screen of the white remote controller",
        "TARGET_PROMPT": "glowing neon green screen of the remote controller",
        "STYLE_SOURCE_PROMPT": "screen of the white remote controller",
        "STYLE_TARGET_PROMPT": "glowing neon green screen of the remote controller",
    },
    # 7) Turn the plush toy into a tiger
    {
        "name": "plush_tiger",
        "DATA_NAME": "blue_sofa",
        "PROMPT": "Change the plush toy's pattern to tiger stripes",
        "SEG_PROMPT": "plush toy",
        "TARGET_PROMPT": "tiger striped plush toy",
        "STYLE_SOURCE_PROMPT": "plush toy",
        "STYLE_TARGET_PROMPT": "tiger striped plush toy",
    },
    # 8) Change the speaker color to gold
    {
        "name": "speaker_gold",
        "DATA_NAME": "blue_sofa",
        "PROMPT": "Change the speaker to a shiny gold texture",
        "SEG_PROMPT": "grey speaker body",
        "TARGET_PROMPT": "shiny metallic gold speaker body",
        "STYLE_SOURCE_PROMPT": "grey speaker body",
        "STYLE_TARGET_PROMPT": "shiny metallic gold speaker body",
    },
    # 9) Make the sunglasses look like aviator glasses
    {
        "name": "glasses_aviator",
        "DATA_NAME": "blue_sofa",
        "PROMPT": "Change the sunglasses style to gold-rimmed aviators",
        "SEG_PROMPT": "black sunglasses",
        "TARGET_PROMPT": "gold-rimmed aviator sunglasses",
        "STYLE_SOURCE_PROMPT": "black sunglasses",
        "STYLE_TARGET_PROMPT": "gold-rimmed aviator sunglasses",
    },
    # 10) Change the perfume bottle cap to silver
    {
        "name": "perfume_silver_cap",
        "DATA_NAME": "blue_sofa",
        "PROMPT": "Change the perfume bottle cap to silver",
        "SEG_PROMPT": "black cap of the perfume bottle",
        "TARGET_PROMPT": "polished silver cap of the perfume bottle",
        "STYLE_SOURCE_PROMPT": "black cap of the perfume bottle",
        "STYLE_TARGET_PROMPT": "polished silver cap of the perfume bottle",
    },
    ## room
    # 1) Change the rubber chicken's color to red
    {
        "name": "chicken_red",
        "DATA_NAME": "room",
        "PROMPT": "Change the rubber chicken's color to red",
        "SEG_PROMPT": "yellow rubber chicken",
        "TARGET_PROMPT": "red rubber chicken",
        "STYLE_SOURCE_PROMPT": "yellow rubber chicken",
        "STYLE_TARGET_PROMPT": "red rubber chicken",
    },
    # 2) Make the rabbit figure white
    {
        "name": "rabbit_white",
        "DATA_NAME": "room",
        "PROMPT": "Make the rabbit figure white",
        "SEG_PROMPT": "grey rabbit figure",
        "TARGET_PROMPT": "white rabbit figure",
        "STYLE_SOURCE_PROMPT": "grey rabbit figure",
        "STYLE_TARGET_PROMPT": "white rabbit figure",
    },
    # 3) Change the dinosaur figure to green
    {
        "name": "dino_green",
        "DATA_NAME": "room",
        "PROMPT": "Change the dinosaur figure to green",
        "SEG_PROMPT": "brown dinosaur figure",
        "TARGET_PROMPT": "green dinosaur figure",
        "STYLE_SOURCE_PROMPT": "brown dinosaur figure",
        "STYLE_TARGET_PROMPT": "green dinosaur figure",
    },
    # 4) Replace the baseball with a tennis ball
    {
        "name": "ball_tennis",
        "DATA_NAME": "room",
        "PROMPT": "Replace the baseball with a tennis ball",
        "SEG_PROMPT": "white baseball",
        "TARGET_PROMPT": "yellow tennis ball",
        "STYLE_SOURCE_PROMPT": "white baseball",
        "STYLE_TARGET_PROMPT": "yellow tennis ball",
    },
    # 5) Change the basket material to wood
    {
        "name": "basket_wood",
        "DATA_NAME": "room",
        "PROMPT": "Change the basket material to dark wood",
        "SEG_PROMPT": "woven basket",
        "TARGET_PROMPT": "dark wooden basket",
        "STYLE_SOURCE_PROMPT": "woven basket",
        "STYLE_TARGET_PROMPT": "dark wooden basket",
    },
    # 6) Add a tiny bow tie to the rubber chicken
    {
        "name": "chicken_bow_tie",
        "DATA_NAME": "room",
        "PROMPT": "Add a tiny blue bow tie to the rubber chicken's neck",
        "SEG_PROMPT": "neck of the yellow rubber chicken",
        "TARGET_PROMPT": "yellow rubber chicken wearing a tiny blue bow tie",
        "STYLE_SOURCE_PROMPT": "neck of the yellow rubber chicken",
        "STYLE_TARGET_PROMPT": "yellow rubber chicken wearing a tiny blue bow tie",
    },
    # 7) Give the rabbit figure sunglasses
    {
        "name": "rabbit_sunglasses",
        "DATA_NAME": "room",
        "PROMPT": "Give the rabbit figure a pair of small sunglasses",
        "SEG_PROMPT": "face of the grey rabbit figure",
        "TARGET_PROMPT": "grey rabbit figure wearing small sunglasses",
        "STYLE_SOURCE_PROMPT": "face of the grey rabbit figure",
        "STYLE_TARGET_PROMPT": "grey rabbit figure wearing small sunglasses",
    },
    # 8) Make the dinosaur figure look like it's made of metal
    {
        "name": "dino_metal",
        "DATA_NAME": "room",
        "PROMPT": "Make the dinosaur figure look like it's made of shiny metal",
        "SEG_PROMPT": "brown dinosaur figure",
        "TARGET_PROMPT": "shiny metallic dinosaur figure",
        "STYLE_SOURCE_PROMPT": "brown dinosaur figure",
        "STYLE_TARGET_PROMPT": "shiny metallic dinosaur figure",
    },
    # 9) Change the baseball to a golden ball
    {
        "name": "ball_gold",
        "DATA_NAME": "room",
        "PROMPT": "Change the baseball to a solid golden ball",
        "SEG_PROMPT": "white baseball",
        "TARGET_PROMPT": "solid golden ball",
        "STYLE_SOURCE_PROMPT": "white baseball",
        "STYLE_TARGET_PROMPT": "solid golden ball",
    },
    # 10) Fill the basket with apples
    {
        "name": "basket_apples",
        "DATA_NAME": "room",
        "PROMPT": "Fill the empty space in the basket with red apples",
        "SEG_PROMPT": "inside of the woven basket",
        "TARGET_PROMPT": "woven basket filled with red apples",
        "STYLE_SOURCE_PROMPT": "inside of the woven basket",
        "STYLE_TARGET_PROMPT": "woven basket filled with red apples",
    },
]

TASKS_BY_NAME = {t["name"]: t for t in TASKS}

# ==========================
# Sweep configuration
# ==========================

SWEEP_CONFIG = {
    "name": WANDB_SWEEP_NAME,
    "method": "grid",  # grid = 30 task × 2 steps = 60 run 한 번씩 돌리고 끝. bayes는 같은 조합 반복 제안함.
    "metric": {
        "name": "clip_dir_similarity",
        "goal": "maximize",
    },
    "parameters": {
        # dge.yaml에 실제로 존재하는 설정만 sweep에서 변경
        "task": {
            "values": [t["name"] for t in TASKS],
        },
        # 학습 길이 및 카메라 업데이트 주기 (동일 값으로 사용)
        "steps": {
            "values": [500, 1500],
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
    """Find save directory from launch.py stdout."""
    # 1) log 안의 "Test results saved to ..." 경로 그대로 사용
    pattern = r"Test results saved to (.+)"
    matches = re.findall(pattern, launch_output)
    if matches:
        p = Path(matches[-1].strip())
        if p.exists():
            return p

    # 2) fallback: EXP_ROOT_DIR 기준으로 최신 trial 검색
    exp_root = Path(EXP_ROOT_DIR)
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

    data_name = task.get("DATA_NAME", DATA_NAME)
    data_source = f"{DATA_SOURCE_ROOT}/{DATA_TYPE}/{data_name}/"
    gs_source = (
        f"{GS_SOURCE_ROOT}/{DATA_TYPE}/{data_name}/point_cloud/iteration_30000/point_cloud.ply"
    )
    gt_dir = f"{GS_SOURCE_ROOT}/{DATA_TYPE}/{data_name}/{RENDER_SUBDIR}"

    # steps: system.camera_update_per_step에만 사용 (trainer.max_steps는 고정)
    steps = int(getattr(cfg, "steps", int(CAMERA_UPDATE_PER_STEP)))

    name = f"sweep/{DATA_TYPE}/{data_name}/camstep{steps}/{task_name}"

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
        f"system.mask_thres={MASK_THRES}",
        f"data.max_view_num={MAX_VIEW_NUM}",
        f"system.seg_prompt={seg_prompt}",
        f"system.camera_update_per_step={steps}",
        f"exp_root_dir={EXP_ROOT_DIR}",
        f"name={name}",
        # Disable inner wandb logging to avoid nested runs
        "system.loggers.wandb.enable=false",
    ]

    print(f"\n[Sweep] Running: task={task_name}")
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

