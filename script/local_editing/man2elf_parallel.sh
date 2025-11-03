#!/usr/bin/env bash

set -euo pipefail

# ==========================
# Configuration
# ==========================

# Lambda_d values to test
LAMBDA_D_VALUES=(0.0 1.0 2.5 5.0 7.5 10.0 12.5 15.0)

# GPU configuration
MAX_GPUS=4  # GPU 0-3
AVAILABLE_GPUS=(0 1 2 3)

# Base configuration (same as man2elf.sh)
CONFIG="configs/dge_view5.yaml"
BASE_ARGS=(
    --config "$CONFIG"
    --train
    trainer.max_steps=1500
    system.prompt_processor.prompt="Turn the man into an elf"
    data.source="/working/style-transfer/VcEdit/gs_data/face/"
    system.guidance.guidance_scale=12.5
    system.gs_source="/working/style-transfer/VcEdit/gs_data/trained_gs_models/face/point_cloud.ply"
    system.seg_prompt="man"
    system.target_prompt="an elf"
    system.mask_thres=0.3
)

# Metrics configuration
GT_DIR="/data/users/jaeyeonpark/DGE-outputs/edit_cache/origin_render/"
STYLE_PROMPT="an elf"
EXP_ROOT_DIR="/data/users/jaeyeonpark/DGE-outputs"  # config에서 설정된 exp_root_dir
EXP_NAME="dge"  # config에서 설정된 name

# ==========================
# Helper functions
# ==========================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

# Check if GPU is available (memory usage < 500MB means likely free)
is_gpu_available() {
    local gpu=$1
    local mem_used=$(nvidia-smi --id=$gpu --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{print $1}')
    if [[ -z "$mem_used" ]]; then
        echo "no"
        return
    fi
    # If memory usage is less than 500MB, consider it available
    if [[ $mem_used -lt 500 ]]; then
        echo "yes"
    else
        echo "no"
    fi
}

# Find the latest render directory for a given experiment
find_render_dir() {
    local lambda_d=$1
    local exp_root="$EXP_ROOT_DIR"
    local exp_name="$EXP_NAME"
    local max_views=20  # from config
    
    # Path structure: {exp_root}/{exp_name}/{max_views}/{tag}@{timestamp}/save/it{max_steps}-test/
    local view_dir="$exp_root/$exp_name/$max_views"
    
    if [[ ! -d "$view_dir" ]]; then
        echo "Warning: View directory not found: $view_dir" >&2
        return 1
    fi
    
    # Find the most recent trial directory matching the prompt
    local tag_pattern="Turn_the_man_into_an_elf"
    local latest_dir=""
    local latest_time=0
    
    while IFS= read -r dir; do
        if [[ "$(basename "$dir")" == *"$tag_pattern"* ]]; then
            local dir_time=$(stat -c %Y "$dir" 2>/dev/null || stat -f %m "$dir" 2>/dev/null || echo 0)
            if [[ $dir_time -gt $latest_time ]]; then
                latest_time=$dir_time
                latest_dir="$dir"
            fi
        fi
    done < <(find "$view_dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null)
    
    if [[ -z "$latest_dir" ]] || [[ ! -d "$latest_dir" ]]; then
        echo "Warning: Could not find trial directory for lambda_d=$lambda_d" >&2
        return 1
    fi
    
    # Find render directory: {trial_dir}/save/it{max_steps}-test/
    local max_steps=1500
    local render_dir="$latest_dir/save/it${max_steps}-test"
    
    if [[ -d "$render_dir" ]]; then
        echo "$render_dir"
        return 0
    else
        echo "Warning: Render directory not found: $render_dir" >&2
        return 1
    fi
}

# Run metrics for a completed experiment
run_metrics() {
    local lambda_d=$1
    local render_dir=$2
    
    if [[ ! -d "$GT_DIR" ]]; then
        echo "GT_DIR not found: $GT_DIR, skipping metrics" >&2
        return 1
    fi
    
    if [[ ! -d "$render_dir" ]]; then
        echo "RENDER_DIR not found: $render_dir, skipping metrics" >&2
        return 1
    fi
    
    echo "Running metrics for lambda_d=$lambda_d..."
    CMD=(python metrics.py --gt "$GT_DIR" --render "$render_dir" --device "cuda" --interval 1)
    
    if [[ -n "$STYLE_PROMPT" ]]; then
        CMD+=(--style_prompt "$STYLE_PROMPT")
    fi
    
    echo "Command: ${CMD[*]}"
    "${CMD[@]}" || {
        echo "Metrics failed for lambda_d=$lambda_d" >&2
        return 1
    }
    
    echo "Metrics completed for lambda_d=$lambda_d"
}

# Run training on a specific GPU
run_training() {
    local gpu=$1
    local lambda_d=$2
    local log_file="nohups/lambda_d_${lambda_d}_gpu${gpu}.log"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [GPU $gpu] Starting training with lambda_d=$lambda_d (log: $log_file)"
    
    mkdir -p nohups
    
    # Run training
    # IMPORTANT: Set CUDA_VISIBLE_DEVICES in the environment before running
    # This ensures each process only sees its assigned GPU
    env CUDA_VISIBLE_DEVICES=$gpu python launch.py \
        "${BASE_ARGS[@]}" \
        --gpu "$gpu" \
        system.loss.lambda_d="$lambda_d" \
        > "$log_file" 2>&1
    
    local exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [GPU $gpu] Training completed for lambda_d=$lambda_d"
        
        # Wait a bit for filesystem to sync
        sleep 10
        
        # Try to find render directory (may take a few attempts)
        local render_dir=""
        local attempts=0
        while [[ $attempts -lt 5 ]]; do
            if render_dir=$(find_render_dir "$lambda_d" 2>/dev/null); then
                break
            fi
            ((attempts++))
            echo "[GPU $gpu] Waiting for render directory (attempt $attempts/5)..."
            sleep 10
        done
        
        if [[ -n "$render_dir" ]]; then
            run_metrics "$lambda_d" "$render_dir"
        else
            echo "[GPU $gpu] Warning: Could not find render directory for lambda_d=$lambda_d after multiple attempts" >&2
        fi
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [GPU $gpu] Training failed for lambda_d=$lambda_d (exit code: $exit_code)" >&2
    fi
    
    return $exit_code
}

# ==========================
# Main execution
# ==========================

echo "Starting parallel training with lambda_d values: ${LAMBDA_D_VALUES[*]}"
echo "Using GPUs: ${AVAILABLE_GPUS[*]}"
echo ""
echo "Status: The script will continuously check for available GPUs and assign"
echo "        the next lambda_d value to each available GPU in a loop."
echo ""

# Track running processes: gpu -> lambda_d
declare -A gpu_jobs
declare -A gpu_pids

# Track completed experiments
declare -A completed

# Initialize
for gpu in "${AVAILABLE_GPUS[@]}"; do
    gpu_jobs[$gpu]=""
    gpu_pids[$gpu]=""
done

# Process all lambda_d values
lambda_idx=0
total_lambdas=${#LAMBDA_D_VALUES[@]}

while [[ $lambda_idx -lt $total_lambdas ]]; do
    # Check for completed jobs
    for gpu in "${AVAILABLE_GPUS[@]}"; do
        if [[ -n "${gpu_pids[$gpu]:-}" ]]; then
            if ! kill -0 "${gpu_pids[$gpu]}" 2>/dev/null; then
                # Process completed
                lambda_d="${gpu_jobs[$gpu]}"
                wait "${gpu_pids[$gpu]}"
                exit_code=$?
                
                echo "[GPU $gpu] Job completed for lambda_d=$lambda_d (exit: $exit_code)"
                
                gpu_jobs[$gpu]=""
                gpu_pids[$gpu]=""
                completed[$lambda_d]=1
            fi
        fi
    done
    
    # Assign new jobs to available GPUs
    # IMPORTANT: Process all available GPUs in the same iteration
    for gpu in "${AVAILABLE_GPUS[@]}"; do
        # Check if GPU is available (no PID or PID doesn't exist)
        gpu_available=false
        if [[ -z "${gpu_pids[$gpu]:-}" ]]; then
            gpu_available=true
        elif ! ps -p "${gpu_pids[$gpu]}" > /dev/null 2>&1; then
            # PID exists but process is dead
            gpu_available=true
            gpu_pids[$gpu]=""
            gpu_jobs[$gpu]=""
        fi
        
        if [[ "$gpu_available" == true ]]; then
            # GPU is available, find next unassigned lambda_d
            found_unassigned=false
            temp_idx=$lambda_idx
            while [[ $temp_idx -lt $total_lambdas ]]; do
                lambda_d="${LAMBDA_D_VALUES[$temp_idx]}"
                
                # Skip if already completed or assigned to another GPU
                if [[ -z "${completed[$lambda_d]:-}" ]]; then
                    # Check if this lambda_d is already assigned to another GPU
                    already_assigned=false
                    for check_gpu in "${AVAILABLE_GPUS[@]}"; do
                        if [[ "${gpu_jobs[$check_gpu]:-}" == "$lambda_d" ]] && \
                           [[ -n "${gpu_pids[$check_gpu]:-}" ]] && \
                           ps -p "${gpu_pids[$check_gpu]}" > /dev/null 2>&1; then
                            already_assigned=true
                            break
                        fi
                    done
                    
                    if [[ "$already_assigned" == false ]]; then
                        # Start new job on this GPU
                        (
                            run_training "$gpu" "$lambda_d"
                        ) &
                        
                        gpu_pids[$gpu]=$!
                        gpu_jobs[$gpu]=$lambda_d
                        
                        echo "[$(date '+%H:%M:%S')] [GPU $gpu] Assigned lambda_d=$lambda_d (PID: ${gpu_pids[$gpu]}, idx: $temp_idx)"
                        
                        # Update lambda_idx only for the first unassigned lambda_d we found
                        if [[ "$found_unassigned" == false ]]; then
                            lambda_idx=$((temp_idx + 1))
                            found_unassigned=true
                        fi
                        break
                    fi
                fi
                ((temp_idx++))
            done
            
            if [[ "$found_unassigned" == false ]]; then
                # No more unassigned lambda_d values
                break
            fi
        fi
    done
    
    # If all GPUs busy and still have work, wait
    all_busy=true
    assigned_count=0
    for gpu in "${AVAILABLE_GPUS[@]}"; do
        if [[ -n "${gpu_pids[$gpu]:-}" ]] && ps -p "${gpu_pids[$gpu]}" > /dev/null 2>&1; then
            assigned_count=$((assigned_count + 1))
        else
            all_busy=false
        fi
    done
    
    # Show current status every loop iteration (for terminal monitoring)
    echo "[$(date '+%H:%M:%S')] [Status] Progress: $lambda_idx/$total_lambdas lambda_d values processed | Active GPUs: $assigned_count/$MAX_GPUS"
    
    # Show which GPU is running which lambda_d
    for gpu in "${AVAILABLE_GPUS[@]}"; do
        if [[ -n "${gpu_pids[$gpu]:-}" ]] && ps -p "${gpu_pids[$gpu]}" > /dev/null 2>&1; then
            lambda_d_running="${gpu_jobs[$gpu]}"
            pid="${gpu_pids[$gpu]}"
            # Check if log file exists and get last line (training progress)
            log_file="nohups/lambda_d_${lambda_d_running}_gpu${gpu}.log"
            if [[ -f "$log_file" ]]; then
                last_line=$(tail -n 1 "$log_file" 2>/dev/null | grep -o "Epoch [0-9]*:.*" | head -1 || echo "Initializing...")
                echo "  └─ GPU $gpu (PID: $pid): lambda_d=$lambda_d_running | $last_line"
            else
                echo "  └─ GPU $gpu (PID: $pid): lambda_d=$lambda_d_running | Log file not created yet"
            fi
        elif [[ -z "${gpu_pids[$gpu]:-}" ]]; then
            echo "  └─ GPU $gpu: Available (no job assigned)"
        fi
    done
    
    if [[ $all_busy == true ]] && [[ $lambda_idx -lt $total_lambdas ]]; then
        echo "All GPUs busy, waiting 30 seconds..."
        sleep 30
    else
        sleep 5
    fi
done

# Wait for all remaining jobs
echo ""
echo "All experiments queued. Waiting for completion..."
for gpu in "${AVAILABLE_GPUS[@]}"; do
    if [[ -n "${gpu_pids[$gpu]:-}" ]]; then
        if ps -p "${gpu_pids[$gpu]}" > /dev/null 2>&1; then
            lambda_d="${gpu_jobs[$gpu]}"
            echo "[GPU $gpu] Waiting for lambda_d=$lambda_d to complete..."
            wait "${gpu_pids[$gpu]}"
            echo "[GPU $gpu] lambda_d=$lambda_d completed"
            
            # Run metrics if not already done
            if [[ -z "${completed[$lambda_d]:-}" ]]; then
                render_dir=""
                if render_dir=$(find_render_dir "$lambda_d"); then
                    run_metrics "$lambda_d" "$render_dir"
                fi
            fi
        fi
    fi
done

echo ""
echo "All experiments completed!"

