
#!/usr/bin/env bash
set -euo pipefail

# ==========================
# User-configurable settings
# ==========================

# Base directory base (lambda and max_view_num will be appended)
BASE_DIR_BASE="/data/users/jaeyeonpark/DGE-outputs/clip-loss/w-MaskUpdate/iter1/"

# List of max_view_num values to process
MAX_VIEW_NUMS=(5 10 15 20)
# MAX_VIEW_NUMS=(5 10)
# MAX_VIEW_NUMS=(15 20)

# GPUs to use (mapped per max_view_num, round-robin if needed)
GPUS=(0 1 2 3 6 7)


# Lambda values to iterate sequentially
LAMBDA_LIST=(0.0 5.0 10.0 15.0 20.0 25.0)
# LAMBDA_LIST=(5.0 10.0)


# Ground truth directory (same for all experiments)5
GT_DIR="/data/users/jaeyeonpark/DGE-outputs/edit_cache/origin_render/"

# Optional: choose ONE of the following for style condition
STYLE_PROMPT="an elf"        # leave empty "" to disable
STYLE_IMAGE=""                # set to an image path to use style image instead of text
# Object prompt for CLIP direction similarity
OBJECT_PROMPT="man"      # default: "a Photo"

# Optional: filter experiment folders by name pattern (leave empty to process all)
# Example: "Turn_the_man_into_an_elf" to only process elf experiments
# Example: "clown" to only process clown experiments
FOLDER_FILTER="elf"


# Misc
INTERVAL=1                     # temporal interval k for consistency metrics
DEVICE="cuda"                 # cuda or cpu

# Max steps to look for (will try to find it{max_steps}-test directories)
# If not found, will try to find any it*-test directory
MAX_STEPS=1500


# Optional: output summary file
SUMMARY_FILE=""               # leave empty to disable summary, or set path like "batch_metrics_summary.txt"

# ==========================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d "$GT_DIR" ]]; then
  echo "GT_DIR not found: $GT_DIR" >&2; exit 1
fi

# If both provided, prefer STYLE_IMAGE
if [[ -n "$STYLE_PROMPT" && -n "$STYLE_IMAGE" ]]; then
  echo "Both STYLE_PROMPT and STYLE_IMAGE set; using STYLE_IMAGE." >&2
  STYLE_PROMPT=""
fi


# Arrays to track results across all lambdas (for overall summary)
TOTAL_SUCCESSFUL=()
TOTAL_FAILED=()
TOTAL_SKIPPED=()

# Function to process a single max_view_num on a specific GPU
run_for_max_view() {
  local max_view_num="$1"
  local gpu="$2"

  BASE_DIR="${BASE_DIR_PREFIX}${max_view_num}/"

  echo "=========================================="
  echo "Processing max_view_num: $max_view_num (GPU=$gpu)"
  echo "BASE_DIR: $BASE_DIR"
  echo "=========================================="

  if [[ ! -d "$BASE_DIR" ]]; then
    echo "  WARNING: BASE_DIR not found: $BASE_DIR, skipping..." >&2
    echo ""
    return 0
  fi

  # Find all experiment folders
  if [[ -n "$FOLDER_FILTER" ]]; then
    EXP_FOLDERS=("$BASE_DIR"*"$FOLDER_FILTER"*)
  else
    EXP_FOLDERS=("$BASE_DIR"*/)
  fi

  # Filter out non-directories and sort
  EXP_FOLDERS=($(printf '%s\n' "${EXP_FOLDERS[@]}" | grep -v '^$' | sort))

  if [[ ${#EXP_FOLDERS[@]} -eq 0 ]]; then
    echo "  No experiment folders found in $BASE_DIR"
    echo ""
    return 0
  fi

  echo "  Found ${#EXP_FOLDERS[@]} experiment folder(s)"
  echo ""

  SUCCESSFUL=()
  FAILED=()
  SKIPPED=()

  for exp_folder in "${EXP_FOLDERS[@]}"; do
    exp_folder="${exp_folder%/}"
    exp_name=$(basename "$exp_folder")

    echo "=========================================="
    echo "Processing: $exp_name"
    echo "=========================================="

    # Try to find render directory
    render_dir="$exp_folder/save/it${MAX_STEPS}-test"
    if [[ ! -d "$render_dir" ]]; then
      found_dirs=("$exp_folder/save"/it*-test)
      if [[ -e "${found_dirs[0]}" ]]; then
        render_dir="${found_dirs[0]}"
        echo "  Using found render dir: $render_dir"
      else
        echo "  WARNING: No render directory found in $exp_folder/save/"
        SKIPPED+=("$exp_name")
        echo ""
        continue
      fi
    fi

    # Extract latency if available
    latency_summary="$exp_folder/latency/summary.txt"
    latency_time=""
    if [[ -f "$latency_summary" ]]; then
      latency_line=$(head -n 1 "$latency_summary")
      if echo "$latency_line" | grep -q "Total Time:"; then
        latency_time=$(echo "$latency_line" | grep -oE "Total Time: [0-9.]+s" | grep -oE "[0-9.]+s")
      fi
    fi

    METRICS_CMD=(python metrics.py --gt "$GT_DIR" --render "$render_dir" --device "$DEVICE" --interval "$INTERVAL" --object_prompt "$OBJECT_PROMPT")
    if [[ -n "$STYLE_IMAGE" ]]; then
      METRICS_CMD+=(--style_image "$STYLE_IMAGE")
    elif [[ -n "$STYLE_PROMPT" ]]; then
      METRICS_CMD+=(--style_prompt "$STYLE_PROMPT")
    fi

    echo "  Running on GPU $gpu: ${METRICS_CMD[*]}"
    if [[ -n "$latency_time" ]]; then
      echo "  Latency: $latency_time"
    fi
    echo ""

    # Run with per-job GPU assignment
    if CUDA_VISIBLE_DEVICES="$gpu" "${METRICS_CMD[@]}"; then
      SUCCESSFUL+=("$exp_name")
      if [[ -n "$latency_time" ]]; then
        echo "  ✓ Success (Latency: $latency_time)"
      else
        echo "  ✓ Success"
      fi
    else
      FAILED+=("$exp_name")
      if [[ -n "$latency_time" ]]; then
        echo "  ✗ Failed (Latency: $latency_time)"
      else
        echo "  ✗ Failed"
      fi
    fi

    # Read metrics from eval_clip.txt written by metrics.py and print alongside latency
    trial_dir=""
    if [[ "$(basename "$(dirname "$render_dir")")" == "save" ]]; then
      trial_dir="$(dirname "$(dirname "$render_dir")")"
    else
      trial_dir="$(dirname "$render_dir")"
    fi
    eval_file="$trial_dir/eval_clip.txt"
    if [[ -f "$eval_file" ]]; then
      clip_dir_consistency=$(grep -E "^CLIP directional consistency:" "$eval_file" | sed 's/.*: //')
      clip_f_scaled=$(grep -E "^CLIP_F \(scaled\):" "$eval_file" | sed 's/.*: //')
      clip_score_val=$(grep -E "^CLIP Score:" "$eval_file" | sed 's/.*: //')
      clip_dir_similarity=$(grep -E "^CLIP directional similarity:" "$eval_file" | sed 's/.*: //')
      echo "  Metrics:"
      [[ -n "$latency_time" ]] && echo "    - Latency: $latency_time"
      [[ -n "$clip_dir_consistency" ]] && echo "    - CLIP directional consistency: $clip_dir_consistency"
      [[ -n "$clip_f_scaled" ]] && echo "    - CLIP_F (scaled): $clip_f_scaled"
      [[ -n "$clip_score_val" ]] && echo "    - CLIP Score: $clip_score_val"
      [[ -n "$clip_dir_similarity" ]] && echo "    - CLIP directional similarity: $clip_dir_similarity"
    else
      echo "  Metrics file not found: $eval_file"
    fi

    echo ""
  done

  echo "=========================================="
  echo "Summary for max_view_num=$max_view_num (GPU=$gpu)"
  echo "=========================================="
  echo "Total folders processed: ${#EXP_FOLDERS[@]}"
  echo "Successful: ${#SUCCESSFUL[@]}"
  echo "Failed: ${#FAILED[@]}"
  echo "Skipped (no render dir): ${#SKIPPED[@]}"
  echo ""

  if [[ ${#SUCCESSFUL[@]} -gt 0 ]]; then
    echo "Successful experiments:"
    for exp in "${SUCCESSFUL[@]}"; do
      echo "  ✓ $exp"
      ALL_SUCCESSFUL+=("[$max_view_num] $exp")
    done
    echo ""
  fi
  if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Failed experiments:"
    for exp in "${FAILED[@]}"; do
      echo "  ✗ $exp"
      ALL_FAILED+=("[$max_view_num] $exp")
    done
    echo ""
  fi
  if [[ ${#SKIPPED[@]} -gt 0 ]]; then
    echo "Skipped experiments (no render dir):"
    for exp in "${SKIPPED[@]}"; do
      echo "  ⊘ $exp"
      ALL_SKIPPED+=("[$max_view_num] $exp")
    done
    echo ""
  fi
}

# Sequentially process each lambda value
for lam in "${LAMBDA_LIST[@]}"; do
  echo ""
  echo "########## Processing lambda_d=${lam} ##########"
  BASE_DIR_PREFIX="${BASE_DIR_BASE}lambda_d${lam}/"

  # Reset per-lambda accumulators
  ALL_SUCCESSFUL=()
  ALL_FAILED=()
  ALL_SKIPPED=()

  # Launch one background job per max_view_num, each on a different GPU
  for idx in "${!MAX_VIEW_NUMS[@]}"; do
    max_view_num="${MAX_VIEW_NUMS[$idx]}"
    gpu_index=$(( idx % ${#GPUS[@]} ))
    gpu="${GPUS[$gpu_index]}"
    run_for_max_view "$max_view_num" "$gpu" &
  done

  # Wait for this lambda batch to finish
  wait

  # Append to totals
  TOTAL_SUCCESSFUL+=("${ALL_SUCCESSFUL[@]}")
  TOTAL_FAILED+=("${ALL_FAILED[@]}")
  TOTAL_SKIPPED+=("${ALL_SKIPPED[@]}")

  # Per-lambda summary
  echo "=========================================="
  echo "LAMBDA SUMMARY (lambda_d=${lam})"
  echo "=========================================="
  echo "Total successful: ${#ALL_SUCCESSFUL[@]}"
  echo "Total failed: ${#ALL_FAILED[@]}"
  echo "Total skipped: ${#ALL_SKIPPED[@]}"
  echo ""
done

# Overall summary across lambdas
echo "=========================================="
echo "OVERALL BATCH PROCESSING SUMMARY (all lambdas)"
echo "=========================================="
echo "Total successful: ${#TOTAL_SUCCESSFUL[@]}"
echo "Total failed: ${#TOTAL_FAILED[@]}"
echo "Total skipped: ${#TOTAL_SKIPPED[@]}"
echo ""

if [[ ${#TOTAL_SUCCESSFUL[@]} -gt 0 ]]; then
  echo "All successful experiments:"
  for exp in "${TOTAL_SUCCESSFUL[@]}"; do
    echo "  ✓ $exp"
  done
  echo ""
fi

if [[ ${#TOTAL_FAILED[@]} -gt 0 ]]; then
  echo "All failed experiments:"
  for exp in "${TOTAL_FAILED[@]}"; do
    echo "  ✗ $exp"
  done
  echo ""
fi

if [[ ${#TOTAL_SKIPPED[@]} -gt 0 ]]; then
  echo "All skipped experiments (no render dir):"
  for exp in "${TOTAL_SKIPPED[@]}"; do
    echo "  ⊘ $exp"
  done
  echo ""
fi

echo "Done."

