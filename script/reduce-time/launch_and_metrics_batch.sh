#!/usr/bin/env bash

set -euo pipefail

# ==========================
# User-configurable settings
# ==========================

# Number of repetitions
NUM_RUNS=5

# Base script to run (relative to this script's directory)
BASE_SCRIPT="launch_and_metrics.sh"

# Log directory for batch runs
BATCH_LOG_DIR="nohups/batch_runs"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BATCH_LOG_BASE="${BATCH_LOG_DIR}/${TIMESTAMP}"

# ==========================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT_PATH="${SCRIPT_DIR}/${BASE_SCRIPT}"

# Check if base script exists
if [[ ! -f "$BASE_SCRIPT_PATH" ]]; then
    echo "Error: Base script not found: $BASE_SCRIPT_PATH" >&2
    exit 1
fi

# Make base script executable
chmod +x "$BASE_SCRIPT_PATH"

# Create log directory
mkdir -p "$BATCH_LOG_BASE"

echo "=========================================="
echo "Batch Run Configuration"
echo "=========================================="
echo "Base script: $BASE_SCRIPT_PATH"
echo "Number of runs: $NUM_RUNS"
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
    
    # Run the script and capture output
    if "$BASE_SCRIPT_PATH" 2>&1 | tee "$RUN_LOG"; then
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

