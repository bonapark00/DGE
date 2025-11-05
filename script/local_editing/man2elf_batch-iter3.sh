#!/usr/bin/env bash

set -euo pipefail

# GPU 리스트 (0-7, 4와 5 제외)
GPUS=(0 1 2 3 6 7)
# GPUS=(6 7)

# max_view_num 리스트
# MAX_VIEW_NUMS=(5 10 15 20)
MAX_VIEW_NUMS=(5 10 15 20)

# Base script parameters
CONFIG="configs/dge_view5.yaml"
SOURCE="/working/style-transfer/VcEdit/gs_data/face/"
GS_SOURCE="/working/style-transfer/VcEdit/gs_data/trained_gs_models/face/point_cloud.ply"
PROMPT="Turn the man into an elf"
SEG_PROMPT="man"
MASK_THRES=0.6
GUIDANCE_SCALE=12.5
MAX_STEPS=1500
TARGET_PROMPT="an elf"


CAMERA_UPDATE_PER_STEP=500
MASK_UPDATE_AT_STEP=300 # -1 or 300

# Lambdas to iterate through (will run sequentially)
LAMBDA_LIST=(0.0 5.0 10.0 15.0 20.0 25.0)

# Base name prefix for experiment outputs (NAME will be set per-lambda once)
NAME_PREFIX="clip-loss/w-MaskUpdate"


declare -a ALL_JOBS=()

echo "Total jobs: ${#ALL_JOBS[@]}"
echo "GPUs: ${GPUS[*]}"
echo "max_view_nums: ${MAX_VIEW_NUMS[*]}"
echo ""

# Track running jobs per GPU
declare -A GPU_PIDS=()
# Track all started PIDs in current lambda batch
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
    local timestamp=$(date +%Y%m%d-%H%M%S)
    
    echo "[${timestamp}] Starting job: GPU=${gpu}, max_view_num=${max_view_num}"
    
    # Create log directory if it doesn't exist
    log_dir="nohups/${NAME}/lambda_d${LAMBDA_D}"
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
        system.loss.lambda_d="${LAMBDA_D}" \
        system.camera_update_per_step="${CAMERA_UPDATE_PER_STEP}" \
        system.mask_update_at_step="${MASK_UPDATE_AT_STEP}" \
        system.target_prompt="${TARGET_PROMPT}" \
        name="${NAME}" \
        > "${log_dir}/gpu${gpu}_view${max_view_num}.log" 2>&1 &
    
    local pid=$!
    GPU_PIDS[$gpu]=$pid
    STARTED_PIDS+=("$pid")
    echo "[${timestamp}] Job started: GPU=${gpu}, max_view_num=${max_view_num}"
    echo "[${timestamp}] PID: ${pid}"
}

run_for_lambda() {
    local lambda_val=$1
    LAMBDA_D=$lambda_val
    NAME="${NAME_PREFIX}/iter$((MAX_STEPS/CAMERA_UPDATE_PER_STEP))/lambda_d${LAMBDA_D}"

    # Build jobs for this lambda
    ALL_JOBS=()
    for view_idx in "${!MAX_VIEW_NUMS[@]}"; do
        max_view_num="${MAX_VIEW_NUMS[$view_idx]}"
        gpu="${GPUS[$view_idx]}"
        ALL_JOBS+=("${gpu}:${max_view_num}")
    done

    echo "Starting lambda_d=${LAMBDA_D} with ${#ALL_JOBS[@]} jobs..."
    STARTED_PIDS=()
    GPU_PIDS=()

    job_count=0
    for job in "${ALL_JOBS[@]}"; do
        IFS=':' read -r gpu max_view_num <<< "$job"
        wait_for_gpu "$gpu"
        run_job "$gpu" "$max_view_num"
        job_count=$((job_count + 1))
        if [ $job_count -lt ${#ALL_JOBS[@]} ]; then
            sleep 3
        fi
    done

    echo "All ${#ALL_JOBS[@]} jobs for lambda_d=${LAMBDA_D} queued. Waiting to finish..."
    for pid in "${STARTED_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            wait "$pid" || true
        fi
    done
    echo "All jobs for lambda_d=${LAMBDA_D} completed."
}

# Run sequential batches for each lambda value
for lam in "${LAMBDA_LIST[@]}"; do
    run_for_lambda "$lam"
done

echo ""
echo "Monitor logs in: nohups/${NAME}/lambda_d${LAMBDA_D}/gpu*_view*.log"
echo ""
echo "To check running processes:"
echo "  ps aux | grep 'python launch.py'"
echo ""
echo "To check GPU usage:"
echo "  nvidia-smi"

