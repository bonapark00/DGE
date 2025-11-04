#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES=3

# ==========================
# User-configurable settings
# ==========================

# Base directory containing experiment folders
BASE_DIR="/data/users/jaeyeonpark/DGE-outputs/dge/20/"

# Ground truth directory (same for all experiments)
GT_DIR="/data/users/jaeyeonpark/DGE-outputs/edit_cache/origin_render/"

# Optional: choose ONE of the following for style condition
STYLE_PROMPT="an elf"        # leave empty "" to disable
STYLE_IMAGE=""                # set to an image path to use style image instead of text

# Misc
INTERVAL=1                     # temporal interval k for consistency metrics
DEVICE="cuda"                 # cuda or cpu

# Max steps to look for (will try to find it{max_steps}-test directories)
# If not found, will try to find any it*-test directory
MAX_STEPS=1500

# Optional: filter experiment folders by name pattern (leave empty to process all)
# Example: "Turn_the_man_into_an_elf" to only process elf experiments
FOLDER_FILTER=""

# Optional: output summary file
SUMMARY_FILE=""               # leave empty to disable summary, or set path like "batch_metrics_summary.txt"

# ==========================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d "$BASE_DIR" ]]; then
  echo "BASE_DIR not found: $BASE_DIR" >&2; exit 1
fi

if [[ ! -d "$GT_DIR" ]]; then
  echo "GT_DIR not found: $GT_DIR" >&2; exit 1
fi

# If both provided, prefer STYLE_IMAGE
if [[ -n "$STYLE_PROMPT" && -n "$STYLE_IMAGE" ]]; then
  echo "Both STYLE_PROMPT and STYLE_IMAGE set; using STYLE_IMAGE." >&2
  STYLE_PROMPT=""
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
  echo "No experiment folders found in $BASE_DIR" >&2; exit 1
fi

echo "Found ${#EXP_FOLDERS[@]} experiment folder(s)"
echo ""

# Arrays to track results
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
  METRICS_CMD=(python metrics.py --gt "$GT_DIR" --render "$render_dir" --device "$DEVICE" --interval "$INTERVAL")
  
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

# Print summary
echo "=========================================="
echo "BATCH PROCESSING SUMMARY"
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
  echo "Skipped experiments (no render dir):"
  for exp in "${SKIPPED[@]}"; do
    echo "  ⊘ $exp"
  done
  echo ""
fi

# Save summary to file if requested
if [[ -n "$SUMMARY_FILE" ]]; then
  summary_path="$BASE_DIR/$SUMMARY_FILE"
  {
    echo "Batch Metrics Summary"
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
fi

echo "Done."

