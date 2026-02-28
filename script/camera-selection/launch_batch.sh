#!/usr/bin/env bash

set -euo pipefail

# ==========================
# User-configurable settings
# ==========================

# Number of repetitions
NUM_RUNS=5

# Training config (edit as needed)
CONFIG="configs/dge_camera-selection.yaml"
GPU="4"
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
# TARGET_PROMPT="clown"
TARGET_PROMPT="A man with curly hair in a checkered cloth"
TARGET_PROMPT="A spider man with a mask and curly hair"
MASK_THRES="0.6"
LAMBDA_D="10.0"
MAX_VIEW_NUM="25"
MAX_EDIT_VIEW_NUM="20"
EDIT_VIEW_SELECTION_STRATEGY="manual-20"  # row, quadrant, manual-20, manual-15
CAMERA_UPDATE_PER_STEP="1500"
MASK_UPDATE_AT_STEP="-1"
NAME="camera-selection/wo-MaskUpdate/iter1/lambda_d${LAMBDA_D}/${EDIT_VIEW_SELECTION_STRATEGY}"

# Log directory for batch runs
BATCH_LOG_DIR="nohups/batch_runs"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BATCH_LOG_BASE="${BATCH_LOG_DIR}/${TIMESTAMP}"

# ==========================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

# Create log directory
mkdir -p "$BATCH_LOG_BASE"

echo "=========================================="
echo "Batch Launch Configuration"
echo "=========================================="
echo "Number of runs: $NUM_RUNS"
echo "Config: $CONFIG"
echo "GPU: $GPU"
echo "Max steps: $MAX_STEPS"
echo "Name: $NAME"
echo "Log directory: $BATCH_LOG_BASE"
echo "=========================================="
echo ""

# Track results
SUCCESS_COUNT=0
FAIL_COUNT=0
declare -a FAILED_RUNS=()

# Run each iteration
for ((run=1; run<=NUM_RUNS; run++)); do
    echo ""
    echo "=========================================="
    echo "[Run $run/$NUM_RUNS] Starting..."
    echo "=========================================="
    echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Log file for this run
    RUN_LOG="${BATCH_LOG_BASE}/run_${run}.log"
    
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
        name="$NAME")
    
    echo "Running: ${LAUNCH_CMD[*]}"
    echo ""
    
    # Run the script and capture output
    if "${LAUNCH_CMD[@]}" 2>&1 | tee "$RUN_LOG"; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo ""
        echo "[Run $run/$NUM_RUNS] ✓ Completed successfully"
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_RUNS+=("$run")
        echo ""
        echo "[Run $run/$NUM_RUNS] ✗ Failed"
    fi
    
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Add a small delay between runs (optional)
    if [[ $run -lt $NUM_RUNS ]]; then
        echo "Waiting 5 seconds before next run..."
        sleep 5
    fi
done

echo ""
echo "=========================================="
echo "Batch Run Summary"
echo "=========================================="
echo "Total runs: $NUM_RUNS"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"
echo ""

if [[ $FAIL_COUNT -gt 0 ]]; then
    echo "Failed runs: ${FAILED_RUNS[*]}"
    echo ""
    echo "Check logs in: $BATCH_LOG_BASE"
    exit 1
else
    echo "All runs completed successfully!"
    echo ""
    echo "All logs saved in: $BATCH_LOG_BASE"
    exit 0
fi

