#!/usr/bin/env bash

set -euo pipefail

# ==========================
# User-configurable settings
# ==========================

# Base directory containing experiment results
# Example: /data/users/jaeyeonpark/DGE-outputs/camera-selection/wo-MaskUpdate/iter1/lambda_d10.0/manual-20/25
# BASE_DIR="${1:-/data/users/jaeyeonpark/DGE-outputs/camera-selection/w-MaskUpdate/iter1/lambda_d0.0/manual-20/20}"
BASE_DIR="${1:-/data/users/jaeyeonpark/DGE-outputs/camera-selection/w-MaskUpdate/iter1/lambda_d10.0/manual-20/25}"

# ==========================

if [[ ! -d "$BASE_DIR" ]]; then
    echo "Error: Directory not found: $BASE_DIR" >&2
    echo "Usage: $0 [BASE_DIR]" >&2
    exit 1
fi

echo "=========================================="
echo "Collecting Metrics from eval_clip.txt"
echo "=========================================="
echo "Base directory: $BASE_DIR"
echo ""

# Arrays to store metrics
declare -a clip_dir_consistency=()
declare -a clip_f_scaled=()
declare -a clip_score=()
declare -a clip_dir_similarity=()
declare -a experiment_names=()

# Find all eval_clip.txt files
EVAL_FILES=("$BASE_DIR"/*/eval_clip.txt)

if [[ ! -e "${EVAL_FILES[0]}" ]]; then
    echo "Error: No eval_clip.txt files found in $BASE_DIR" >&2
    exit 1
fi

echo "Found ${#EVAL_FILES[@]} eval_clip.txt file(s)"
echo ""

# Process each file
for eval_file in "${EVAL_FILES[@]}"; do
    if [[ ! -f "$eval_file" ]]; then
        continue
    fi
    
    # Extract experiment name from path
    exp_name=$(basename "$(dirname "$eval_file")")
    experiment_names+=("$exp_name")
    
    echo "Processing: $exp_name"
    
    # Extract metrics using grep and awk
    # CLIP directional consistency: 9.8359375
    dir_cons=$(grep -E "^CLIP directional consistency:" "$eval_file" | awk '{print $4}' || echo "")
    # CLIP_F (scaled): 100.625
    f_scaled=$(grep -E "^CLIP_F \(scaled\):" "$eval_file" | awk '{print $4}' || echo "")
    # CLIP Score: 28.125
    score=$(grep -E "^CLIP Score:" "$eval_file" | awk '{print $3}' || echo "")
    # CLIP directional similarity: 26.703125
    dir_sim=$(grep -E "^CLIP directional similarity:" "$eval_file" | awk '{print $4}' || echo "")
    
    # Store values (convert empty to 0 for calculation)
    if [[ -n "$dir_cons" ]]; then
        clip_dir_consistency+=("$dir_cons")
        echo "  CLIP directional consistency: $dir_cons"
    else
        clip_dir_consistency+=("0")
        echo "  CLIP directional consistency: (not found)"
    fi
    
    if [[ -n "$f_scaled" ]]; then
        clip_f_scaled+=("$f_scaled")
        echo "  CLIP_F (scaled): $f_scaled"
    else
        clip_f_scaled+=("0")
        echo "  CLIP_F (scaled): (not found)"
    fi
    
    if [[ -n "$score" ]]; then
        clip_score+=("$score")
        echo "  CLIP Score: $score"
    else
        clip_score+=("0")
        echo "  CLIP Score: (not found)"
    fi
    
    if [[ -n "$dir_sim" ]]; then
        clip_dir_similarity+=("$dir_sim")
        echo "  CLIP directional similarity: $dir_sim"
    else
        clip_dir_similarity+=("0")
        echo "  CLIP directional similarity: (not found)"
    fi
    
    echo ""
done

# Calculate averages using awk (no bc dependency)
calculate_average() {
    local values=("$@")
    local count=${#values[@]}
    
    if [[ $count -eq 0 ]]; then
        echo "0"
        return
    fi
    
    # Use awk for floating point arithmetic
    printf '%s\n' "${values[@]}" | awk '{
        sum += $1
        count++
    }
    END {
        if (count > 0) {
            printf "%.6f\n", sum / count
        } else {
            printf "0\n"
        }
    }'
}

# Calculate averages
avg_dir_cons=$(calculate_average "${clip_dir_consistency[@]}")
avg_f_scaled=$(calculate_average "${clip_f_scaled[@]}")
avg_score=$(calculate_average "${clip_score[@]}")
avg_dir_sim=$(calculate_average "${clip_dir_similarity[@]}")

# Print summary table
echo "=========================================="
echo "Summary"
echo "=========================================="
printf "%-40s %10s\n" "Metric" "Average"
echo "------------------------------------------"
printf "%-40s %10.6f\n" "CLIP directional consistency" "$avg_dir_cons"
printf "%-40s %10.6f\n" "CLIP_F (scaled)" "$avg_f_scaled"
printf "%-40s %10.6f\n" "CLIP Score" "$avg_score"
printf "%-40s %10.6f\n" "CLIP directional similarity" "$avg_dir_sim"
echo "------------------------------------------"
printf "%-40s %10d\n" "Total experiments" "${#experiment_names[@]}"
echo ""

# Print detailed results table
echo "=========================================="
echo "Detailed Results"
echo "=========================================="
printf "%-50s %15s %15s %15s %15s\n" "Experiment" "Dir Cons" "CLIP_F" "Score" "Dir Sim"
echo "------------------------------------------------------------------------------------------------------------------------"

for i in "${!experiment_names[@]}"; do
    exp_name="${experiment_names[$i]}"
    # Truncate name if too long
    if [[ ${#exp_name} -gt 48 ]]; then
        exp_name="${exp_name:0:45}..."
    fi
    printf "%-50s %15.6f %15.6f %15.6f %15.6f\n" \
        "$exp_name" \
        "${clip_dir_consistency[$i]}" \
        "${clip_f_scaled[$i]}" \
        "${clip_score[$i]}" \
        "${clip_dir_similarity[$i]}"
done

echo "------------------------------------------------------------------------------------------------------------------------"
printf "%-50s %15.6f %15.6f %15.6f %15.6f\n" \
    "AVERAGE" \
    "$avg_dir_cons" \
    "$avg_f_scaled" \
    "$avg_score" \
    "$avg_dir_sim"
echo ""
