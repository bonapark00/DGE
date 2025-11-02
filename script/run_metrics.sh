#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES=3

# ==========================
# User-configurable settings
# ==========================

# Required: set these paths
GT_DIR="/data/users/jaeyeonpark/DGE-outputs/edit_cache/origin_render/"            # e.g., /working/style-transfer/DGE/outputs/.../save/render_it1500-val
RENDER_DIR="/data/users/jaeyeonpark/DGE-outputs/dge/Turn_the_man_into_a_clown@20251029-190325/save/it1500-test/"    # e.g., /working/style-transfer/DGE/outputs/.../save/render_it1500-val

# Optional: choose ONE of the following for style condition
STYLE_PROMPT="a clown"        # leave empty "" to disable
STYLE_IMAGE=""                # set to an image path to use style image instead of text

# Misc
INTERVAL=1                     # temporal interval k for consistency metrics
DEVICE="cuda"                 # cuda or cpu

# ==========================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d "$GT_DIR" ]]; then
  echo "GT_DIR not found: $GT_DIR" >&2; exit 1
fi
if [[ ! -d "$RENDER_DIR" ]]; then
  echo "RENDER_DIR not found: $RENDER_DIR" >&2; exit 1
fi

# If both provided, prefer STYLE_IMAGE
if [[ -n "$STYLE_PROMPT" && -n "$STYLE_IMAGE" ]]; then
  echo "Both STYLE_PROMPT and STYLE_IMAGE set; using STYLE_IMAGE." >&2
  STYLE_PROMPT=""
fi

CMD=(python metrics.py --gt "$GT_DIR" --render "$RENDER_DIR" --device "$DEVICE" --interval "$INTERVAL")

if [[ -n "$STYLE_IMAGE" ]]; then
  CMD+=(--style_image "$STYLE_IMAGE")
elif [[ -n "$STYLE_PROMPT" ]]; then
  CMD+=(--style_prompt "$STYLE_PROMPT")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"


