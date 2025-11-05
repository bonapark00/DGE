#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES=3

# ==========================
# User-configurable settings
# ==========================

# Base directory prefix (max_view_num will be appended)
BASE_DIR_PREFIX="/data/users/jaeyeonpark/DGE-outputs/dge/"

# List of max_view_num values to process
MAX_VIEW_NUMS=(5 8 10 15 20)

# Ground truth directory (same for all experiments)
GT_DIR="/data/users/jaeyeonpark/DGE-outputs/edit_cache/origin_render/"

# Optional: choose ONE of the following for style condition
STYLE_PROMPT="clown"        # leave empty "" to disable
STYLE_IMAGE=""                # set to an image path to use style image instead of text
# Object prompt for CLIP direction similarity
OBJECT_PROMPT="man"      # default: "a Photo"

# Optional: filter experiment folders by name pattern (leave empty to process all)
# Example: "Turn_the_man_into_an_elf" to only process elf experiments
# Example: "clown" to only process clown experiments
FOLDER_FILTER="clown"


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

# Arrays to track results across all max_view_num values
ALL_SUCCESSFUL=()
ALL_FAILED=()
ALL_SKIPPED=()

# Process each max_view_num
for max_view_num in "${MAX_VIEW_NUMS[@]}"; do
  BASE_DIR="${BASE_DIR_PREFIX}${max_view_num}/"
  
  echo "=========================================="
  echo "Processing max_view_num: $max_view_num"
  echo "BASE_DIR: $BASE_DIR"
  echo "=========================================="
  
  if [[ ! -d "$BASE_DIR" ]]; then
    echo "  WARNING: BASE_DIR not found: $BASE_DIR, skipping..." >&2
    echo ""
    continue
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
    continue
  fi
  
  echo "  Found ${#EXP_FOLDERS[@]} experiment folder(s)"
  echo ""
  
  # Arrays to track results for this max_view_num
  SUCCESSFUL=()
  FAILED=()
  SKIPPED=()
  
  # Process each folder
  for exp_folder in "${EXP_FOLDERS[@]}"; do
  exp_folder="${exp_folder%/}"  # Remove trailing slash
  exp_name=$(basename "$exp_folder")
  
  echo "=========================================="
  echo "Processing: $exp_name"
  echo "=========================================="
  
  # Try to find render directory
  # First try: save/it{MAX_STEPS}-test/
  render_dir="$exp_folder/save/it${MAX_STEPS}-test"
  
  # If not found, try to find any it*-test directory
  if [[ ! -d "$render_dir" ]]; then
    # Find any it*-test directory
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
  
  # Build metrics command
  METRICS_CMD=(python metrics.py --gt "$GT_DIR" --render "$render_dir" --device "$DEVICE" --interval "$INTERVAL" --object_prompt "$OBJECT_PROMPT")
  
  if [[ -n "$STYLE_IMAGE" ]]; then
    METRICS_CMD+=(--style_image "$STYLE_IMAGE")
  elif [[ -n "$STYLE_PROMPT" ]]; then
    METRICS_CMD+=(--style_prompt "$STYLE_PROMPT")
  fi
  
  echo "  Running: ${METRICS_CMD[*]}"
  echo ""
  
  # Run metrics and capture result
  if "${METRICS_CMD[@]}"; then
    SUCCESSFUL+=("$exp_name")
    echo "  ✓ Success"
  else
    FAILED+=("$exp_name")
    echo "  ✗ Failed"
  fi
  
  echo ""
  done
  
  # Print summary for this max_view_num
  echo "=========================================="
  echo "Summary for max_view_num=$max_view_num"
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
  
  # Save summary to file if requested (per max_view_num)
  if [[ -n "$SUMMARY_FILE" ]]; then
    summary_path="$BASE_DIR/$SUMMARY_FILE"
    {
      echo "Batch Metrics Summary - max_view_num=$max_view_num"
      echo "Generated: $(date)"
      echo "=========================================="
      echo ""
      echo "Total folders processed: ${#EXP_FOLDERS[@]}"
      echo "Successful: ${#SUCCESSFUL[@]}"
      echo "Failed: ${#FAILED[@]}"
      echo "Skipped: ${#SKIPPED[@]}"
      echo ""
      if [[ ${#SUCCESSFUL[@]} -gt 0 ]]; then
        echo "Successful experiments:"
        for exp in "${SUCCESSFUL[@]}"; do
          echo "  ✓ $exp"
        done
        echo ""
      fi
      if [[ ${#FAILED[@]} -gt 0 ]]; then
        echo "Failed experiments:"
        for exp in "${FAILED[@]}"; do
          echo "  ✗ $exp"
        done
        echo ""
      fi
      if [[ ${#SKIPPED[@]} -gt 0 ]]; then
        echo "Skipped experiments:"
        for exp in "${SKIPPED[@]}"; do
          echo "  ⊘ $exp"
        done
      fi
    } > "$summary_path"
    echo "Summary saved to: $summary_path"
    echo ""
  fi
done

# Print overall summary
echo "=========================================="
echo "OVERALL BATCH PROCESSING SUMMARY"
echo "=========================================="
echo "Total successful: ${#ALL_SUCCESSFUL[@]}"
echo "Total failed: ${#ALL_FAILED[@]}"
echo "Total skipped: ${#ALL_SKIPPED[@]}"
echo ""

if [[ ${#ALL_SUCCESSFUL[@]} -gt 0 ]]; then
  echo "All successful experiments:"
  for exp in "${ALL_SUCCESSFUL[@]}"; do
    echo "  ✓ $exp"
  done
  echo ""
fi

if [[ ${#ALL_FAILED[@]} -gt 0 ]]; then
  echo "All failed experiments:"
  for exp in "${ALL_FAILED[@]}"; do
    echo "  ✗ $exp"
  done
  echo ""
fi

if [[ ${#ALL_SKIPPED[@]} -gt 0 ]]; then
  echo "All skipped experiments (no render dir):"
  for exp in "${ALL_SKIPPED[@]}"; do
    echo "  ⊘ $exp"
  done
  echo ""
fi

echo "Done."

