#!/usr/bin/env bash
# Run generate_by_lens.py with preset arguments.
# Edit the variables below and run: ./run_generate_by_lens.sh

set -e

# ---------- config (edit these) ----------
# GPU: device string (e.g. cuda, cuda:0, cuda:2). Default cuda uses device 0.
DEVICE="cuda:2"

DATA_TYPE="in2n-GSEditor"
DATA_NAME="face"

# PLY_PATH="/data/users/jaeyeonpark/3dgs-trained/3d-ovs/covered_desk/point_cloud/iteration_30000/point_cloud.ply"
PLY_PATH="/data/users/jaeyeonpark/3dgs-trained/GSEditor-in2n/${DATA_NAME}/point_cloud/iteration_30000/point_cloud.ply"

# COLMAP_PATH="/data/users/jaeyeonpark/dataset/3d-ovs/covered_desk"
COLMAP_PATH="/data/users/jaeyeonpark/dataset/${DATA_TYPE}/${DATA_NAME}"


# SEG_PROMPT="the man's face"
SEG_PROMPT="human head"
# SEG_PROMPT="fleece jacket"


EDIT_PROMPT="Make his mouth smile"
# EDIT_PROMPT="Change the fleece jacket's color into blue"
# EDIT_PROMPT="Change the fleece jacket into a leather jacket"
# EDIT_PROMPT="Make the shampoo look like a rectangle"
# EDIT_PROMPT="Change the shampoo bottle's color into blue" ## 전체가 바뀌진 하지만 보존됨
# EDIT_PROMPT="Change the bear's t-shirt color to blue"
# EDIT_PROMPT="Change the pooh look like a panda"
# EDIT_PROMPT="Change the red sweater into a leather jacket"
# EDIT_PROMPT="Make the man wear black sunglasses"
# EDIT_PROMPT="Give him a mustache"

SAVE_COLMAP="output/lens_colmap"
VIDEO_PATH="output/lens_claude_${DATA_NAME}.mp4"
# DISTANCE_MULTIPLIERS는 마지막 값만 쓰입니다. 여러 후보를 사용할 땐 주석으로 관리하세요.
DISTANCE_MULTIPLIERS="2.0, 2.5, 3.0, 3.5, 4.0, 4.5"
# DISTANCE_MULTIPLIERS="3.0, 4.0, 5.0, 6.0, 7.0, 8.0"
# DISTANCE_MULTIPLIERS="5.0, 6.0, 7.0, 8.0, 9.0, 10.0"
# DISTANCE_MULTIPLIERS="2.0, 2.5, 3.0, 4.0, 5.0, 6.0"

N_SELECT="20"
# COLMAP cone half-angle (degrees) for Fibonacci candidate filtering
CONE_HALF_ANGLE_DEG="40.0"
# Bash에서는 f-string이 안 되므로 변수치환은 아래처럼 사용
SAVE_ATTN_GRID="output/attn_grid_${DATA_NAME}.png"

# IP2P (same as run_instruct_pix2pix.sh). Lower guidance_scale = preserve structure more.
GUIDANCE_SCALE="4.0"           # 7.5 default; try 4.0 for less change
IMAGE_GUIDANCE_SCALE="1.5"     # 1.5 default; higher = preserve input more
NUM_INFERENCE_STEPS="5"       # lower = faster (e.g. 10, 5)

# SAGE-Probing hyperparameters (see generate_by_lens.py --lambda_leak/--lambda_ent)
LAMBDA_LEAK="1.5"
LAMBDA_ENT="10.0"

# optional flags: set to non-empty to enable (e.g. "1" or "yes")
USE_IP2P_SCORING="1"
VISUALIZE_ROI="1"
# ----------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Force process onto the chosen GPU (avoids device being overridden by other config)
if [[ "$DEVICE" =~ ^cuda:([0-9]+)$ ]]; then
    export CUDA_VISIBLE_DEVICES="${BASH_REMATCH[1]}"
    DEVICE_FOR_PY="cuda"
else
    DEVICE_FOR_PY="${DEVICE:-cuda}"
fi

args=(
  --device "$DEVICE_FOR_PY"
  --ply_path "$PLY_PATH"
  --colmap_path "$COLMAP_PATH"
  --seg_prompt "$SEG_PROMPT"
  --edit_prompt "$EDIT_PROMPT"
  --guidance_scale "$GUIDANCE_SCALE"
  --image_guidance_scale "$IMAGE_GUIDANCE_SCALE"
  --num_inference_steps "$NUM_INFERENCE_STEPS"
  --save_colmap "$SAVE_COLMAP"
  --video_path "$VIDEO_PATH"
  --distance_multipliers "$DISTANCE_MULTIPLIERS"
  --n_select "$N_SELECT"
  --save_attn_grid "$SAVE_ATTN_GRID"
  --cone_half_angle_deg "$CONE_HALF_ANGLE_DEG"
  --lambda_leak "$LAMBDA_LEAK"
  --lambda_ent "$LAMBDA_ENT"
)
[[ -n "$USE_IP2P_SCORING" ]] && args+=(--use_ip2p_scoring)
[[ -n "$VISUALIZE_ROI" ]] && args+=(--visualize_roi)

python generate_by_lens.py "${args[@]}"

