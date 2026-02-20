#!/usr/bin/env bash

set -euo pipefail

# ==========================
# User-configurable settings
# ==========================

# Training config (edit as needed)
CONFIG="configs/dge_camera-selection.yaml"
GPU="7"
MAX_STEPS="1500"

# Task-specific overrides (edit as needed)
# PROMPT="Turn the man into a clown"
# PROMPT="Give him a checkered jacket"
PROMPT="Turn him into spider man with a mask"
DATA_SOURCE="/working/style-transfer/VcEdit/gs_data/face/"
GS_SOURCE="/working/style-transfer/VcEdit/gs_data/trained_gs_models/face/point_cloud.ply"
GUIDANCE_SCALE="12.5"
# SEG_PROMPT="man"
SEG_PROMPT="A man with curly hair in a grey jacket"
MMR_SEG_PROMPT="A mans's face" # for text-based segmentation

# TARGET_PROMPT="clown"
TARGET_PROMPT="A man with curly hair in a checkered cloth"
# TARGET_PROMPT="A spider man with a mask and curly hair"
MASK_THRES="0.6"
LAMBDA_D="10.0"
MAX_VIEW_NUM="25"
MAX_EDIT_VIEW_NUM="20"
EDIT_VIEW_SELECTION_STRATEGY="random"  # row, quadrant, manual-20, manual-15, random
CAMERA_UPDATE_PER_STEP="1500"
MASK_UPDATE_AT_STEP="-1"
PRUNE_FLOATER_AT_STEP="-1"  # -1: disabled, otherwise prune at this step
NAME="camera-selection/wo-MaskUpdate/iter1/lambda_d${LAMBDA_D}/${EDIT_VIEW_SELECTION_STRATEGY}"

# Metrics config
GT_DIR="/data/users/jaeyeonpark/DGE-outputs/edit_cache/origin_render/"
STYLE_PROMPT=$SEG_PROMPT       # leave empty "" to disable
STYLE_IMAGE=""     # set to an image path to use style image instead of text
OBJECT_PROMPT="$TARGET_PROMPT"        # default: "a Photo"
INTERVAL=1                 # temporal interval k for consistency metrics
DEVICE="cuda"              # cuda or cpu

# ==========================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

echo "=========================================="
echo "[1/3] Launch training..."
echo "=========================================="

# Build launch.py command
LAUNCH_CMD=(python launch.py \
    --config "$CONFIG" \
    --train \
    --gpu "$GPU" \
    trainer.max_steps="$MAX_STEPS" \
    system.prompt_processor.prompt="$PROMPT" \
    data.source="$DATA_SOURCE" \
    system.guidance.guidance_scale="$GUIDANCE_SCALE" \
    system.gs_source="$GS_SOURCE" \
    system.seg_prompt="$SEG_PROMPT" \
    system.target_prompt="$TARGET_PROMPT" \
    system.mask_thres="$MASK_THRES" \
    system.loss.lambda_d="$LAMBDA_D" \
    data.max_view_num="$MAX_VIEW_NUM" \
    data.max_edit_view_num="$MAX_EDIT_VIEW_NUM" \
    data.edit_view_selection_strategy="$EDIT_VIEW_SELECTION_STRATEGY" \
    system.guidance.edit_view_selection_strategy="$EDIT_VIEW_SELECTION_STRATEGY" \
    system.camera_update_per_step="$CAMERA_UPDATE_PER_STEP" \
    system.mask_update_at_step="$MASK_UPDATE_AT_STEP" \
    system.prune_floater_at_step="$PRUNE_FLOATER_AT_STEP" \
    data.mmr_seg_prompt="$MMR_SEG_PROMPT" \
    name="$NAME")

echo "Running: ${LAUNCH_CMD[*]}"
echo ""

# Run launch.py and capture output
LAUNCH_OUTPUT=$(mktemp)
if "${LAUNCH_CMD[@]}" 2>&1 | tee "$LAUNCH_OUTPUT"; then
    echo ""
    echo "Training completed successfully."
else
    echo ""
    echo "Training failed. Check the output above." >&2
    rm -f "$LAUNCH_OUTPUT"
    exit 1
fi

echo ""
echo "=========================================="
echo "[2/3] Discover render directory from output..."
echo "=========================================="

# Parse the output to find the save directory
# Look for line: "[INFO] Test results saved to ..." or "Test results saved to ..."
SAVE_DIR=""

# Look for any line containing "Test results saved to"
MATCHED_LINE=$(grep "Test results saved to" "$LAUNCH_OUTPUT" | tail -1 || true)

if [[ -n "$MATCHED_LINE" ]]; then
    # Extract path after "Test results saved to " using sed
    # This handles both "[INFO] Test results saved to ..." and "Test results saved to ..."
    SAVE_DIR=$(echo "$MATCHED_LINE" | sed -n 's/.*Test results saved to //p' | sed 's/[[:space:]]*$//')
    
    # Debug output
    echo "Found line: $MATCHED_LINE"
    echo "Extracted path: $SAVE_DIR"
    
    # Verify the directory exists
    if [[ ! -d "$SAVE_DIR" ]]; then
        echo "Warning: Extracted path does not exist: $SAVE_DIR" >&2
        SAVE_DIR=""
    fi
fi

rm -f "$LAUNCH_OUTPUT"

# Fallback: try to find the latest directory based on config
if [[ -z "$SAVE_DIR" || ! -d "$SAVE_DIR" ]]; then
    echo "Warning: Could not find 'Test results saved to' message in output." >&2
    echo "Attempting to find latest directory manually..." >&2
    
    EXP_ROOT_DIR="/data/users/jaeyeonpark/DGE-outputs"
    # The path structure is: $EXP_ROOT_DIR/$NAME/$MAX_VIEW_NUM/*@*/save
    EXP_DIR="$EXP_ROOT_DIR/$NAME/$MAX_VIEW_NUM"
    
    if [[ -d "$EXP_DIR" ]]; then
        # Find the latest trial directory (sorted by modification time)
        # Try find first (more reliable), fallback to ls
        LATEST_TRIAL=$(find "$EXP_DIR" -maxdepth 1 -type d -name "*@*" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || \
                      ls -td "$EXP_DIR"/*@* 2>/dev/null | head -1 || true)
        if [[ -n "$LATEST_TRIAL" && -d "$LATEST_TRIAL/save" ]]; then
            SAVE_DIR="$LATEST_TRIAL/save"
            echo "Found directory via fallback: $SAVE_DIR"
        fi
    else
        # Try without MAX_VIEW_NUM in case structure is different
        EXP_DIR="$EXP_ROOT_DIR/$NAME"
        if [[ -d "$EXP_DIR" ]]; then
            LATEST_TRIAL=$(find "$EXP_DIR" -maxdepth 1 -type d -name "*@*" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || \
                          ls -td "$EXP_DIR"/*@* 2>/dev/null | head -1 || true)
            if [[ -n "$LATEST_TRIAL" && -d "$LATEST_TRIAL/save" ]]; then
                SAVE_DIR="$LATEST_TRIAL/save"
                echo "Found directory via fallback (without view_num): $SAVE_DIR"
            fi
        fi
    fi
fi

if [[ -z "$SAVE_DIR" || ! -d "$SAVE_DIR" ]]; then
    echo "Error: Could not find save directory." >&2
    echo "Expected pattern: [INFO] Test results saved to <path>" >&2
    exit 1
fi

echo "Found save directory: $SAVE_DIR"

# Find the test render directory (it{max_steps}-test)
RENDER_DIR=""
if [[ -d "$SAVE_DIR/it${MAX_STEPS}-test" ]]; then
    RENDER_DIR="$SAVE_DIR/it${MAX_STEPS}-test"
else
    # Try to find any it*-test directory
    TEST_DIRS=("$SAVE_DIR"/it*-test)
    if [[ -e "${TEST_DIRS[0]}" ]]; then
        RENDER_DIR="${TEST_DIRS[0]}"
    else
        echo "Error: Could not find test render directory in $SAVE_DIR" >&2
        echo "Expected: $SAVE_DIR/it${MAX_STEPS}-test or similar" >&2
        exit 1
    fi
fi

echo "Using render directory: $RENDER_DIR"
echo ""

# Validate directories
if [[ ! -d "$GT_DIR" ]]; then
    echo "Error: GT_DIR not found: $GT_DIR" >&2
    exit 1
fi
if [[ ! -d "$RENDER_DIR" ]]; then
    echo "Error: RENDER_DIR not found: $RENDER_DIR" >&2
    exit 1
fi

echo "=========================================="
echo "[3/3] Run metrics..."
echo "=========================================="

# If both provided, prefer STYLE_IMAGE
if [[ -n "$STYLE_PROMPT" && -n "$STYLE_IMAGE" ]]; then
    echo "Both STYLE_PROMPT and STYLE_IMAGE set; using STYLE_IMAGE." >&2
    STYLE_PROMPT=""
fi

METRICS_CMD=(python metrics.py \
    --gt "$GT_DIR" \
    --render "$RENDER_DIR" \
    --device "$DEVICE" \
    --interval "$INTERVAL" \
    --object_prompt "$OBJECT_PROMPT")

if [[ -n "$STYLE_IMAGE" ]]; then
    METRICS_CMD+=(--style_image "$STYLE_IMAGE")
elif [[ -n "$STYLE_PROMPT" ]]; then
    METRICS_CMD+=(--style_prompt "$STYLE_PROMPT")
fi

echo "Running: ${METRICS_CMD[*]}"
echo ""
"${METRICS_CMD[@]}"

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="

