#!/usr/bin/env python3
"""
W&B Sweep을 사용한 하이퍼파라미터 튜닝
사용법: wandb sweep hyperparameter_tuning_wandb.py
"""

import wandb
import subprocess
import os

# Sweep 설정
sweep_config = {
    "method": "grid",  # 또는 "random", "bayes"
    "metric": {
        "name": "val_loss",  # 최적화할 메트릭
        "goal": "minimize"
    },
    "parameters": {
        "lambda_d": {
            "values": [30.0, 40.0, 50.0, 100.0, 150.0, 180.0]
        },
        # "max_view_num": {
        #     "values": [25, 30, 35, 40]
        # },
        # "guidance_scale": {
        #     "values": [10.0, 12.5, 15.0]
        # }
    }
}

def train():
    # W&B run 초기화
    run = wandb.init()
    
    # 하이퍼파라미터 가져오기
    lambda_d = wandb.config.lambda_d
    max_view_num = wandb.config.max_view_num
    guidance_scale = wandb.config.guidance_scale
    
    # GPU 할당 (간단한 라운드로빈)
    gpu_id = run.id % 8  # 0-7 GPU
    
    # launch.py 실행
    cmd = [
        "python", "launch.py",
        "--config", "configs/dge_clip-loss.yaml",
        "--train", "--gpu", str(gpu_id),
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
        f"name=wandb-sweep/lambda_d{lambda_d}_view{max_view_num}",
    ]
    
    # 실행
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # 결과 로깅 (실제로는 메트릭을 파싱해야 함)
    # wandb.log({"val_loss": parse_loss_from_output(result.stdout)})

if __name__ == "__main__":
    train()





