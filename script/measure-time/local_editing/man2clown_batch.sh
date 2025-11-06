#!/usr/bin/env bash

set -euo pipefail

# GPU 리스트 (0-7, 4와 5 제외)
# GPUS=(0 1 2 3 6 7)
GPUS=(6 7)

# max_view_num 리스트
# MAX_VIEW_NUMS=(5 10 15 20)
MAX_VIEW_NUMS=(8 12)

# Base script parameters
CONFIG="configs/dge_view5.yaml"
SOURCE="/working/style-transfer/VcEdit/gs_data/face/"
GS_SOURCE="/working/style-transfer/VcEdit/gs_data/trained_gs_models/face/point_cloud.ply"
PROMPT="Turn the man into a clown"
SEG_PROMPT="man"
MASK_THRES=0.6
GUIDANCE_SCALE=12.5
MAX_STEPS=1500

# Create job combinations: max_view_num당 한 번씩만 실행
# GPU 0->5, GPU 1->10, GPU 2->15, GPU 3->20 (첫 4개 GPU만 사용)
declare -a ALL_JOBS=()

# max_view_num 개수만큼만 GPU에 할당 (각 max_view_num은 한 번씩만)
for view_idx in "${!MAX_VIEW_NUMS[@]}"; do
    max_view_num="${MAX_VIEW_NUMS[$view_idx]}"
    # GPU는 순서대로 할당 (0, 1, 2, 3)
    gpu="${GPUS[$view_idx]}"
    ALL_JOBS+=("${gpu}:${max_view_num}")
done

echo "Total jobs: ${#ALL_JOBS[@]}"
echo "GPUs: ${GPUS[*]}"
echo "max_view_nums: ${MAX_VIEW_NUMS[*]}"
echo ""

# Track running jobs per GPU
declare -A GPU_PIDS=()

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
        > "nohups/man2clown_gpu${gpu}_view${max_view_num}.log" 2>&1 &
    
    local pid=$!
    GPU_PIDS[$gpu]=$pid
    echo "[${timestamp}] Job started: GPU=${gpu}, max_view_num=${max_view_num}"
    echo "[${timestamp}] PID: ${pid}"
}

# Launch jobs: one per GPU, with 3-second intervals between starts
job_count=0
for job in "${ALL_JOBS[@]}"; do
    IFS=':' read -r gpu max_view_num <<< "$job"
    
    # Wait for this GPU to be available (only one job per GPU at a time)
    wait_for_gpu "$gpu"
    
    run_job "$gpu" "$max_view_num"
    
    job_count=$((job_count + 1))
    
    # Wait 3 seconds before starting next job (except for the last one)
    if [ $job_count -lt ${#ALL_JOBS[@]} ]; then
        sleep 3
    fi
done

echo ""
echo "All ${#ALL_JOBS[@]} jobs have been queued."
echo "Monitor logs in: nohups/man2clown_gpu*_view*.log"
echo ""
echo "To check running processes:"
echo "  ps aux | grep 'python launch.py'"
echo ""
echo "To check GPU usage:"
echo "  nvidia-smi"

