#!/usr/bin/env python3
"""
Optuna를 사용한 하이퍼파라미터 튜닝
사용법: python hyperparameter_tuning_optuna.py
"""

import optuna
import subprocess
import os
import json
from pathlib import Path

# GPU 풀 관리
GPUS = [0, 1, 2, 3, 6, 7]
gpu_queue = []

def objective(trial):
    """하나의 trial 실행"""
    
    # 하이퍼파라미터 제안
    lambda_d = trial.suggest_categorical("lambda_d", [30.0, 40.0, 50.0, 100.0, 150.0, 180.0])
    max_view_num = trial.suggest_categorical("max_view_num", [25, 30, 35, 40])
    guidance_scale = trial.suggest_float("guidance_scale", 10.0, 15.0, step=2.5)
    
    # GPU 할당 (간단한 방식)
    gpu_id = trial.number % len(GPUS)
    
    # 실행 디렉토리 생성
    output_dir = f"outputs/optuna_trial_{trial.number}"
    os.makedirs(output_dir, exist_ok=True)
    
    # launch.py 실행
    cmd = [
        "python", "launch.py",
        "--config", "configs/dge_clip-loss.yaml",
        "--train", "--gpu", str(GPUS[gpu_id]),
        f"trainer.max_steps=1500",
        f"system.prompt_processor.prompt=Turn the man into a clown",
        f"data.source=/working/style-transfer/VcEdit/gs_data/face/",
        f"system.guidance.guidance_scale={guidance_scale}",
        f"system.gs_source=/working/style-transfer/VcEdit/gs_data/trained_gs_models/face/point_cloud.ply",
        f"system.seg_prompt=man",
        f"system.mask_thres=0.6",
        f"data.max_view_num={max_view_num}",
        f"data.max_edit_view_num=20",
        f"system.loss.lambda_d={lambda_d}",
        f"system.camera_update_per_step=1500",
        f"system.mask_update_at_step=-1",
        f"system.target_prompt=clown",
        f"name=optuna-trial-{trial.number}",
    ]
    
    # 실행
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # 메트릭 파싱 (실제로는 로그 파일에서 파싱해야 함)
    # 예시: val_loss = parse_metric_from_log(output_dir)
    val_loss = 0.0  # 실제로는 계산 필요
    
    return val_loss

def main():
    # Study 생성
    study = optuna.create_study(
        study_name="lambda_d_tuning",
        direction="minimize",  # 또는 "maximize"
        storage="sqlite:///optuna.db",  # 결과 저장
        load_if_exists=True
    )
    
    # 최적화 실행
    study.optimize(objective, n_trials=20)
    
    # 결과 출력
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print(f"  Params: {trial.params}")
    
    # 결과 시각화
    try:
        import optuna.visualization as vis
        fig = vis.plot_optimization_history(study)
        fig.write_html("optuna_history.html")
    except:
        pass

if __name__ == "__main__":
    main()





