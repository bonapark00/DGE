#!/usr/bin/env bash

set -euo pipefail

# GPU 리스트 (0-7, 4와 5 제외)
GPUS=(0 1 2 3 6 7)
# GPUS=(6 7)

# max_view_num 리스트
# MAX_VIEW_NUMS=(5 10 15 20)
MAX_VIEW_NUMS=(25)
MAX_EDIT_VIEW_NUM=20

# Base script parameters
CONFIG="configs/dge_clip-loss.yaml"
SOURCE="/working/style-transfer/VcEdit/gs_data/face/"
GS_SOURCE="/working/style-transfer/VcEdit/gs_data/trained_gs_models/face/point_cloud.ply"
PROMPT="Turn the man into a clown"
SEG_PROMPT="man"
TARGET_PROMPT="clown"
MASK_THRES=0.6
GUIDANCE_SCALE=12.5
MAX_STEPS=1500


CAMERA_UPDATE_PER_STEP=1500 # 1500(iter1), 500(iter3)
MASK_UPDATE_AT_STEP=-1 # -1(wo-MaskUpdate) or 300(w-MaskUpdate)

# Lambdas to run in parallel for each max_view_num
# LAMBDA_LIST=(0.0 5.0 10.0)
LAMBDA_LIST=(0.5 0.7 1.0 2.0 3.0 4.0)
# LAMBDA_LIST=(30.0 40.0 50.0 100.0 150.0 180.0)

# Base name prefix for experiment outputs
NAME_PREFIX="clip-loss/wo-MaskUpdate"

echo "GPUs: ${GPUS[*]}"
echo "max_view_nums: ${MAX_VIEW_NUMS[*]}"
echo "lambda_list: ${LAMBDA_LIST[*]}"
echo ""

# Track running jobs per GPU
declare -A GPU_PIDS=()
# Track all started PIDs in current view_num batch
declare -a STARTED_PIDS=()

# Function to check if GPU has a running job
is_gpu_busy() {
    local gpu=$1
    if [ -n "${GPU_PIDS[$gpu]:-}" ]; then
        # Check if process is still running
        if kill -0 "${GPU_PIDS[$gpu]}" 2>/dev/null; then
            return 0  # GPU is busy
        else
            unset GPU_PIDS[$gpu]  # Process finished, clear PID
        fi
    fi
    return 1  # GPU is free
}

# Function to wait for a GPU to become available
wait_for_gpu() {
    local gpu=$1
    while is_gpu_busy "$gpu"; do
        sleep 1
    done
}

# Function to run a single job
run_job() {
    local gpu=$1
    local max_view_num=$2
    local lambda_val=$3
    local timestamp=$(date +%Y%m%d-%H%M%S)
    
    echo "[${timestamp}] Starting job: GPU=${gpu}, max_view_num=${max_view_num}, lambda_d=${lambda_val}"
    
    # Create log directory if it doesn't exist
    local name="${NAME_PREFIX}/iter$((MAX_STEPS/CAMERA_UPDATE_PER_STEP))/lambda_d${lambda_val}"
    local log_dir="nohups/${name}/lambda_d${lambda_val}"
    mkdir -p "$log_dir"
    
    python launch.py \
        --config "${CONFIG}" \
        --train --gpu "${gpu}" \
        trainer.max_steps="${MAX_STEPS}" \
        system.prompt_processor.prompt="${PROMPT}" \
        data.source="${SOURCE}" \
        system.guidance.guidance_scale="${GUIDANCE_SCALE}" \
        system.gs_source="${GS_SOURCE}" \
        system.seg_prompt="${SEG_PROMPT}" \
        system.mask_thres="${MASK_THRES}" \
        data.max_view_num="${max_view_num}" \
        data.max_edit_view_num="${MAX_EDIT_VIEW_NUM}" \
        system.loss.lambda_d="${lambda_val}" \
        system.camera_update_per_step="${CAMERA_UPDATE_PER_STEP}" \
        system.mask_update_at_step="${MASK_UPDATE_AT_STEP}" \
        system.target_prompt="${TARGET_PROMPT}" \
        name="${name}" \
        > "${log_dir}/gpu${gpu}_view${max_view_num}.log" 2>&1 &
    
    local pid=$!
    GPU_PIDS[$gpu]=$pid
    STARTED_PIDS+=("$pid")
    echo "[${timestamp}] Job started: GPU=${gpu}, max_view_num=${max_view_num}, lambda_d=${lambda_val}"
    echo "[${timestamp}] PID: ${pid}"
}

run_for_view_num() {
    local max_view_num=$1

    # Build jobs for this max_view_num (one job per lambda)
    ALL_JOBS=()
    for lambda_idx in "${!LAMBDA_LIST[@]}"; do
        lambda_val="${LAMBDA_LIST[$lambda_idx]}"
        gpu="${GPUS[$lambda_idx]}"
        ALL_JOBS+=("${gpu}:${lambda_val}")
    done

    echo "Starting max_view_num=${max_view_num} with ${#ALL_JOBS[@]} jobs (lambdas)..."
    STARTED_PIDS=()
    GPU_PIDS=()

    job_count=0
    for job in "${ALL_JOBS[@]}"; do
        IFS=':' read -r gpu lambda_val <<< "$job"
        wait_for_gpu "$gpu"
        run_job "$gpu" "$max_view_num" "$lambda_val"
        job_count=$((job_count + 1))
        if [ $job_count -lt ${#ALL_JOBS[@]} ]; then
            sleep 3
        fi
    done

    echo "All ${#ALL_JOBS[@]} jobs for max_view_num=${max_view_num} queued. Waiting to finish..."
    for pid in "${STARTED_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            wait "$pid" || true
        fi
    done
    echo "All jobs for max_view_num=${max_view_num} completed."
}

# Run sequential batches for each max_view_num value
for view_num in "${MAX_VIEW_NUMS[@]}"; do
    run_for_view_num "$view_num"
done

echo ""
echo "Monitor logs in: nohups/${NAME_PREFIX}/iter*/lambda_d*/gpu*_view*.log"
echo ""
echo "To check running processes:"
echo "  ps aux | grep 'python launch.py'"
echo ""
echo "To check GPU usage:"
echo "  nvidia-smi"

