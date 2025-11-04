#!/usr/bin/env bash
set -euo pipefail

# ==========================
# Configuration
# ==========================

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
    system.target_prompt="elf"
    system.mask_thres=0.5
)

# GPU & Lambda_d pairs
GPU_IDS=(0 1 2 3)
# LAMBDA_D_VALUES=(0.0 1.0 5.0 7.5)
LAMBDA_D_VALUES=(10.0 15.0 20.0 25.0)

# ==========================
# Run processes in parallel
# ==========================

for i in "${!GPU_IDS[@]}"; do
    GPU="${GPU_IDS[$i]}"
    LAMBDA_D="${LAMBDA_D_VALUES[$i]}"
    
    echo ">>> Running on GPU $GPU with lambda_d=$LAMBDA_D"
    
    CUDA_VISIBLE_DEVICES=$GPU \
    python launch.py \
        "${BASE_ARGS[@]}" \
        system.loss.lambda_d=$LAMBDA_D \
        --gpu $GPU \
        > "nohups/gpu${GPU}_lambda${LAMBDA_D}.log" 2>&1 &
done

wait
echo "✅ All runs completed!"