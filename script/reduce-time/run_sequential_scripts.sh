#!/usr/bin/env bash
set -euo pipefail

# 실행할 스크립트 목록 (절대/상대 경로 모두 가능)
SCRIPTS=(
  "/working/style-transfer/DGE/script/local_editing/man2clown_batch.sh"
  "/working/style-transfer/DGE/script/run_metrics_batch_clown.sh"
  "/working/style-transfer/DGE/script/local_editing/man2elf_batch.sh"
  "/working/style-transfer/DGE/script/run_metrics_batch_elf.sh" 
  "/working/style-transfer/DGE/script/local_editing/man2elf_batch-iter3.sh"
  "/working/style-transfer/DGE/script/run_metrics_batch_elf-iter3.sh"
)

INTERVAL_SEC=$((30 * 60))  # 10분

# 공통 실행 옵션: 백그라운드가 아니라 순차 실행
for script in "${SCRIPTS[@]}"; do
  if [[ ! -x "$script" ]]; then
    echo "Make executable: $script"
    chmod +x "$script"
  fi

  ts=$(date +%Y-%m-%dT%H:%M:%S)
  echo "[$ts] Running: $script"
  bash "$script"

  # 마지막 항목 이후에는 대기하지 않음
  if [[ "$script" != "${SCRIPTS[-1]}" ]]; then
    echo "Waiting ${INTERVAL_SEC}s before next..."
    sleep "${INTERVAL_SEC}"
  fi
done

echo "All scripts completed."