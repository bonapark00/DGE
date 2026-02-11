#!/usr/bin/env bash
# Run generate_by_lens.py with preset arguments.
# Edit the variables below and run: ./run_generate_by_lens.sh

set -e

# ---------- config (edit these) ----------
# GPU: device string (e.g. cuda, cuda:0, cuda:2). Default cuda uses device 0.
DEVICE="cuda:3"

PLY_PATH="/data/users/jaeyeonpark/3dgs-trained/3d-ovs/covered_desk/point_cloud/iteration_30000/point_cloud.ply"
COLMAP_PATH="/data/users/jaeyeonpark/dataset/3d-ovs/covered_desk"

SEG_PROMPT="shampoo bottle"
SEG_PROMPT="t-shirt"
SEG_PROMPT="pooh"
SEG_PROMPT="red sweater"

EDIT_PROMPT="Make the shampoo look like a rectangle"
EDIT_PROMPT="Change the shampoo bottle's color into blue" ## 전체가 바뀌진 하지만 보존됨
EDIT_PROMPT="Change the bear's t-shirt color to blue"
EDIT_PROMPT="Change the pooh look like a panda"
EDIT_PROMPT="Change the red sweater into a leather jacket"


SAVE_COLMAP="output/lens_colmap"
VIDEO_PATH="output/lens_claude.mp4"
DISTANCE_MULTIPLIERS="3.0, 4.0, 5.0, 6.0, 7.0, 8.0"
DISTANCE_MULTIPLIERS="5.0, 6.0, 7.0, 8.0, 9.0, 10.0"
DISTANCE_MULTIPLIERS="2.0, 2.5, 3.0, 4.0, 5.0, 6.0"
N_SELECT="20"
SAVE_ATTN_GRID="output/attn_grid.jpg"

# IP2P (same as run_instruct_pix2pix.sh). Lower guidance_scale = preserve structure more.
GUIDANCE_SCALE="4.0"           # 7.5 default; try 4.0 for less change
IMAGE_GUIDANCE_SCALE="1.5"     # 1.5 default; higher = preserve input more
NUM_INFERENCE_STEPS="5"       # lower = faster (e.g. 10, 5)

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
)
[[ -n "$USE_IP2P_SCORING" ]] && args+=(--use_ip2p_scoring)
[[ -n "$VISUALIZE_ROI" ]] && args+=(--visualize_roi)

python generate_by_lens.py "${args[@]}"

