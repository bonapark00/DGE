#!/usr/bin/env bash

set -euo pipefail

# ==========================
# User-configurable settings
# ==========================

# Root directory that contains experiment outputs
# Example structure assumed:
#   ${BASE_ROOT}/clip-loss/wo-MaskUpdate/${ITER}/lambda_d${LAMBDA}/${VIEW}/{EXPERIMENT}/
BASE_ROOT="/data/users/jaeyeonpark/DGE-outputs"

# Sub-tree under BASE_ROOT to search in (e.g., clip-loss/wo-MaskUpdate)
SUB_TREE="clip-loss/w-MaskUpdate"

# Iterations and lambdas to scan (lists)
ITERS=(iter1 iter3)
LAMBDAS=(lambda_d0.0 lambda_d5.0 lambda_d10.0)

# Optional: views to scan under each (leave empty to scan all numeric folders found)
VIEWS=(5 10 15 20)

# Optional filter for experiment folder names (empty = all)
FOLDER_FILTER="clown"

# ==========================

ROOT_PATH="${BASE_ROOT}/${SUB_TREE}"

if [[ ! -d "${ROOT_PATH}" ]]; then
  echo "Root path not found: ${ROOT_PATH}" >&2
  exit 1
fi

pad() { printf "%s" "$1"; }

print_header() {
  echo "=========================================="
  echo "Metrics Summary"
  echo "Root: ${ROOT_PATH}"
  echo "Iters: ${ITERS[*]}"
  echo "Lambdas: ${LAMBDAS[*]}"
  echo "Views: ${VIEWS[*]}"
  echo "Filter: ${FOLDER_FILTER:-<none>}"
  echo "=========================================="
}

print_row_header() {
  printf "%-8s | %-12s | %-4s | %-8s | %-26s | %-12s\n" \
    "Iter" "Lambda" "View" "Latency" "CLIP directional similarity" "CLIP Score"
  printf -- "%.0s-" {1..90}; echo ""
}

extract_latency() {
  local dir="$1"  # experiment folder path
  local f="${dir}/latency/summary.txt"
  if [[ -f "$f" ]]; then
    local line
    line=$(head -n 1 "$f" 2>/dev/null || true)
    if echo "$line" | grep -q "Total Time:"; then
      echo "$line" | grep -oE "Total Time: [0-9.]+s" | grep -oE "[0-9.]+s" || true
      return
    fi
  fi
  echo "-"
}

extract_clip_vals() {
  local dir="$1"  # experiment folder path
  local f="${dir}/eval_clip.txt"
  local ds cs
  if [[ -f "$f" ]]; then
    ds=$(grep -E "^CLIP directional similarity:" "$f" | sed 's/.*: //') || ds="-"
    cs=$(grep -E "^CLIP Score:" "$f" | sed 's/.*: //') || cs="-"
  else
    ds="-"; cs="-"
  fi
  printf "%s|%s" "$ds" "$cs"
}

print_header
print_row_header

for iter in "${ITERS[@]}"; do
  for lambda_dir in "${LAMBDAS[@]}"; do
    base_iter_lambda="${ROOT_PATH}/${iter}/${lambda_dir}"
    if [[ ! -d "$base_iter_lambda" ]]; then
      echo "WARN: missing ${base_iter_lambda}" >&2
      continue
    fi

    for view in "${VIEWS[@]}"; do
      view_dir="${base_iter_lambda}/${view}"
      if [[ ! -d "$view_dir" ]]; then
        continue
      fi

      # pick experiment folders under the view directory
      if [[ -n "$FOLDER_FILTER" ]]; then
        exp_folders=("${view_dir}"/*"${FOLDER_FILTER}"*)
      else
        exp_folders=("${view_dir}"/*)
      fi

      # choose the latest experiment folder if multiple exist
      latest_exp=""
      for f in "${exp_folders[@]}"; do
        [[ -d "$f" ]] || continue
        latest_exp="$f"
      done

      if [[ -z "$latest_exp" ]]; then
        continue
      fi

      latency=$(extract_latency "$latest_exp")
      clip_vals=$(extract_clip_vals "$latest_exp")
      clip_dir_sim=${clip_vals%%|*}
      clip_score=${clip_vals##*|}

      printf "%-8s | %-12s | %-4s | %-8s | %-26s | %-12s\n" \
        "$iter" "$lambda_dir" "$view" "$latency" "$clip_dir_sim" "$clip_score"
    done
  done
done

