#!/usr/bin/env bash
# Run InstructPix2Pix on a single image with a prompt.
# Edit the variables below and run: ./run_instruct_pix2pix.sh

set -e

# ---------- config (edit these) ----------
IMAGE="/data/users/jaeyeonpark/DGE-outputs/origin_render/0028.png"
IMAGE="/data/users/jaeyeonpark/DGE-outputs/origin_render/0053.png"
IMAGE="/data/users/jaeyeonpark/dataset/mip-360/kitchen/images/DSCF0656.JPG"
IMAGE="/data/users/jaeyeonpark/DGE-outputs/origin_render/0042.png"
IMAGE="/data/users/jaeyeonpark/dataset/in2n-GSEditor/bear/images/frame_00017.jpg"
IMAGE="/data/users/jaeyeonpark/dataset/mip-360/bicycle/images/_DSC8679.JPG"

PROMPT="Turn his ear into a cat's ear"
PROMPT="Add pearl earring to his ear"
PROMPT="Add rounded earring to his ear"
PROMPT="Turn his eye colors into red"
PROMPT="Make his right ear like an elf's ear"
PROMPT="Make the bear look like a "
PROMPT="Turn the dozer into green"
PROMPT="Make his mouth smile"
PROMPT="Give the bear a pair of sunglasses"
PROMPT="Turn the grass into flowers"

OUTPUT="./editing_image.png"   # leave empty for auto: <input_stem>_edited.<ext>

# optional
GUIDANCE_SCALE="7.5"
NUM_STEPS="20"
IMAGE_GUIDANCE_SCALE="1.5"
DEVICE="cuda:2"
SEED=""     # leave empty for random
# -----------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

args=(
  --image "$IMAGE"
  --prompt "$PROMPT"
  --guidance_scale "$GUIDANCE_SCALE"
  --num_inference_steps "$NUM_STEPS"
  --image_guidance_scale "$IMAGE_GUIDANCE_SCALE"
  --device "$DEVICE"
)
[[ -n "$OUTPUT" ]] && args+=(--output "$OUTPUT")
[[ -n "$SEED" ]] && args+=(--seed "$SEED")

python run_instruct_pix2pix.py "${args[@]}"
