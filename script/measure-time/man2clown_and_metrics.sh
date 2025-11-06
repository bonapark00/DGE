#!/usr/bin/env bash

set -euo pipefail

# ==========================
# User-configurable settings
# ==========================

# Training config
CONFIG="configs/dge.yaml"                 # or configs/dge_view5.yaml
GPU="0"
MAX_STEPS="1500"

# Task-specific overrides (edit as needed)
PROMPT="Turn the man into a clown"
DATA_SOURCE="/working/style-transfer/VcEdit/gs_data/face/"
GS_SOURCE="/working/style-transfer/VcEdit/gs_data/trained_gs_models/face/point_cloud.ply"
GUIDANCE_SCALE="12.5"

# Optional: local editing (set empty to disable)
SEG_PROMPT=""         # e.g., "man"
TARGET_PROMPT=""      # e.g., "clown"
MASK_THRES="0.5"

# Output discovery and metrics
EXP_ROOT_DIR="/data/users/jaeyeonpark/DGE-outputs"   # must match YAML exp_root_dir
NAME="dge"                                            # must match YAML name
TAG_OVERRIDE=""                                       # if set, use this tag (and discover latest ${TAG}@timestamp)
RENDER_DIR_OVERRIDE=""                                # if set, skip discovery and use this directly

# Style condition for metrics (choose one or leave both empty)
STYLE_PROMPT="a clown"
STYLE_IMAGE=""

# Ground-truth directory for metrics (by default uses cache_dir/edit_cache/origin_render)
GT_DIR="/data/users/jaeyeonpark/DGE-outputs/edit_cache/origin_render/"

# Misc
INTERVAL="1"
DEVICE="cuda"

# ==========================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/3] Launch training..."

TRAIN_CMD=(python launch.py --config "$CONFIG" --train --gpu "$GPU" \
  trainer.max_steps="$MAX_STEPS" \
  system.prompt_processor.prompt="$PROMPT" \
  data.source="$DATA_SOURCE" \
  system.guidance.guidance_scale="$GUIDANCE_SCALE" \
  system.gs_source="$GS_SOURCE")

if [[ -n "$SEG_PROMPT" ]]; then
  TRAIN_CMD+=(system.seg_prompt="$SEG_PROMPT")
fi
if [[ -n "$TARGET_PROMPT" ]]; then
  TRAIN_CMD+=(system.target_prompt="$TARGET_PROMPT")
fi
if [[ -n "$MASK_THRES" ]]; then
  TRAIN_CMD+=(system.mask_thres="$MASK_THRES")
fi
if [[ -n "$TAG_OVERRIDE" ]]; then
  TRAIN_CMD+=(tag="$TAG_OVERRIDE")
fi

echo "Running: ${TRAIN_CMD[*]}"
"${TRAIN_CMD[@]}"

echo "[2/3] Discover render directory..."

if [[ -n "$RENDER_DIR_OVERRIDE" ]]; then
  RENDER_DIR="$RENDER_DIR_OVERRIDE"
else
  EXP_DIR="$EXP_ROOT_DIR/$NAME"
  if [[ ! -d "$EXP_DIR" ]]; then
    echo "EXP_DIR not found: $EXP_DIR" >&2; exit 1
  fi
  if [[ -n "$TAG_OVERRIDE" ]]; then
    # pick the latest trial_dir for the given TAG (tag@timestamp)
    TRIAL_DIR="$(ls -td "$EXP_DIR/$TAG_OVERRIDE"@* 2>/dev/null | head -1 || true)"
    if [[ -z "$TRIAL_DIR" ]]; then
      echo "No trial_dir found for tag '$TAG_OVERRIDE' under $EXP_DIR" >&2; exit 1
    fi
  else
    # pick the latest trial_dir regardless of tag
    TRIAL_DIR="$(ls -td "$EXP_DIR"/* 2>/dev/null | head -1 || true)"
    if [[ -z "$TRIAL_DIR" ]]; then
      echo "No trial_dir found under $EXP_DIR" >&2; exit 1
    fi
  fi

  RENDER_DIR="$TRIAL_DIR/save/it${MAX_STEPS}-test/"
fi

echo "Resolved RENDER_DIR: $RENDER_DIR"

if [[ ! -d "$RENDER_DIR" ]]; then
  echo "RENDER_DIR not found: $RENDER_DIR" >&2; exit 1
fi
if [[ ! -d "$GT_DIR" ]]; then
  echo "GT_DIR not found: $GT_DIR" >&2; exit 1
fi

if [[ -n "$STYLE_PROMPT" && -n "$STYLE_IMAGE" ]]; then
  echo "Both STYLE_PROMPT and STYLE_IMAGE set; using STYLE_IMAGE." >&2
  STYLE_PROMPT=""
fi

echo "[3/3] Run metrics..."
METRICS_CMD=(python metrics.py --gt "$GT_DIR" --render "$RENDER_DIR" --device "$DEVICE" --interval "$INTERVAL")
if [[ -n "$STYLE_IMAGE" ]]; then
  METRICS_CMD+=(--style_image "$STYLE_IMAGE")
elif [[ -n "$STYLE_PROMPT" ]]; then
  METRICS_CMD+=(--style_prompt "$STYLE_PROMPT")
fi

echo "Running: ${METRICS_CMD[*]}"
"${METRICS_CMD[@]}"

echo "Done."


