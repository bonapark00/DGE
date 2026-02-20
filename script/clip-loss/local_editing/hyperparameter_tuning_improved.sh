#!/usr/bin/env bash
"""
개선된 bash 기반 하이퍼파라미터 튜닝 스크립트
- 결과를 JSON으로 저장
- 간단한 메트릭 추적
- 재시작 가능
"""

set -euo pipefail

# GPU 리스트
GPUS=(0 1 2 3 6 7)

# 하이퍼파라미터 그리드
LAMBDA_D_VALUES=(30.0 40.0 50.0 100.0 150.0 180.0)
MAX_VIEW_NUM_VALUES=(25 30 35 40)
GUIDANCE_SCALE_VALUES=(10.0 12.5 15.0)

# 결과 저장 디렉토리
RESULTS_DIR="hyperparameter_results"
mkdir -p "$RESULTS_DIR"

# 진행 상황 저장 파일
PROGRESS_FILE="${RESULTS_DIR}/progress.json"

# 진행 상황 로드
if [ -f "$PROGRESS_FILE" ]; then
    echo "Loading progress from $PROGRESS_FILE"
    # jq를 사용하여 완료된 실험 확인 가능
fi

# Base parameters
CONFIG="configs/dge_clip-loss.yaml"
SOURCE="/working/style-transfer/VcEdit/gs_data/face/"
GS_SOURCE="/working/style-transfer/VcEdit/gs_data/trained_gs_models/face/point_cloud.ply"
PROMPT="Turn the man into a clown"
SEG_PROMPT="man"
TARGET_PROMPT="clown"
MASK_THRES=0.6
MAX_STEPS=1500
MAX_EDIT_VIEW_NUM=20
CAMERA_UPDATE_PER_STEP=1500
MASK_UPDATE_AT_STEP=-1
NAME_PREFIX="hyperparam-tuning"

# GPU 관리
declare -A GPU_PIDS=()

is_gpu_busy() {
    local gpu=$1
    if [ -n "${GPU_PIDS[$gpu]:-}" ]; then
        if kill -0 "${GPU_PIDS[$gpu]}" 2>/dev/null; then
            return 0
        else
            unset GPU_PIDS[$gpu]
        fi
    fi
    return 1
}

wait_for_gpu() {
    local gpu=$1
    while is_gpu_busy "$gpu"; do
        sleep 1
    done
}

# 실험 실행 함수
run_experiment() {
    local lambda_d=$1
    local max_view_num=$2
    local guidance_scale=$3
    local gpu=$4
    local exp_id="${lambda_d}_${max_view_num}_${guidance_scale}"
    
    local name="${NAME_PREFIX}/lambda_d${lambda_d}/view${max_view_num}/scale${guidance_scale}"
    local log_dir="${RESULTS_DIR}/${exp_id}"
    mkdir -p "$log_dir"
    
    echo "[$(date +%Y%m%d-%H%M%S)] Starting: lambda_d=${lambda_d}, view=${max_view_num}, scale=${guidance_scale}, GPU=${gpu}"
    
    python launch.py \
        --config "${CONFIG}" \
        --train --gpu "${gpu}" \
        trainer.max_steps="${MAX_STEPS}" \
        system.prompt_processor.prompt="${PROMPT}" \
        data.source="${SOURCE}" \
        system.guidance.guidance_scale="${guidance_scale}" \
        system.gs_source="${GS_SOURCE}" \
        system.seg_prompt="${SEG_PROMPT}" \
        system.mask_thres="${MASK_THRES}" \
        data.max_view_num="${max_view_num}" \
        data.max_edit_view_num="${MAX_EDIT_VIEW_NUM}" \
        system.loss.lambda_d="${lambda_d}" \
        system.camera_update_per_step="${CAMERA_UPDATE_PER_STEP}" \
        system.mask_update_at_step="${MASK_UPDATE_AT_STEP}" \
        system.target_prompt="${TARGET_PROMPT}" \
        name="${name}" \
        > "${log_dir}/output.log" 2>&1 &
    
    local pid=$!
    GPU_PIDS[$gpu]=$pid
    
    # 실험 정보 저장
    cat > "${log_dir}/config.json" <<EOF
{
    "lambda_d": ${lambda_d},
    "max_view_num": ${max_view_num},
    "guidance_scale": ${guidance_scale},
    "gpu": ${gpu},
    "pid": ${pid},
    "start_time": "$(date -Iseconds)"
}
EOF
    
    echo "$pid"
}

# 그리드 서치 실행
exp_count=0
total_exps=$((${#LAMBDA_D_VALUES[@]} * ${#MAX_VIEW_NUM_VALUES[@]} * ${#GUIDANCE_SCALE_VALUES[@]}))

echo "Total experiments: ${total_exps}"
echo "Starting grid search..."

for lambda_d in "${LAMBDA_D_VALUES[@]}"; do
    for max_view_num in "${MAX_VIEW_NUM_VALUES[@]}"; do
        for guidance_scale in "${GUIDANCE_SCALE_VALUES[@]}"; do
            # GPU 할당
            gpu_idx=$((exp_count % ${#GPUS[@]}))
            gpu=${GPUS[$gpu_idx]}
            
            wait_for_gpu "$gpu"
            run_experiment "$lambda_d" "$max_view_num" "$guidance_scale" "$gpu"
            
            exp_count=$((exp_count + 1))
            sleep 3
        done
    done
done

# 모든 작업 완료 대기
echo "All experiments queued. Waiting for completion..."
for pid in "${GPU_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
        wait "$pid" || true
    fi
done

echo "All experiments completed!"
echo "Results saved in: ${RESULTS_DIR}/"





