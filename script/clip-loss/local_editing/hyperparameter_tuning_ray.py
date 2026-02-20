#!/usr/bin/env python3
"""
Ray Tune을 사용한 하이퍼파라미터 튜닝
사용법: python hyperparameter_tuning_ray.py
"""

from ray import tune
from ray.tune import CLIReporter
import subprocess
import os

def train_fn(config):
    """하나의 trial 실행 함수"""
    
    lambda_d = config["lambda_d"]
    max_view_num = config["max_view_num"]
    guidance_scale = config["guidance_scale"]
    gpu_id = config.get("gpu_id", 0)
    
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
        f"name=ray-tune/{tune.get_trial_id()}",
    ]
    
    # 실행
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # 메트릭 리포팅 (실제로는 파싱 필요)
    # tune.report(val_loss=parse_loss(result.stdout))

def main():
    # 검색 공간 정의
    config = {
        "lambda_d": tune.choice([30.0, 40.0, 50.0, 100.0, 150.0, 180.0]),
        "max_view_num": tune.choice([25, 30, 35, 40]),
        "guidance_scale": tune.choice([10.0, 12.5, 15.0]),
        "gpu_id": tune.choice([0, 1, 2, 3, 6, 7]),
    }
    
    # 리포터 설정
    reporter = CLIReporter(metric_columns=["val_loss", "training_iteration"])
    
    # Tune 실행
    analysis = tune.run(
        train_fn,
        config=config,
        num_samples=20,  # trial 수
        resources_per_trial={"gpu": 1},
        reporter=reporter,
        local_dir="./ray_results",
    )
    
    # 최적 결과 출력
    print("Best config:", analysis.best_config)
    print("Best result:", analysis.best_result)

if __name__ == "__main__":
    main()





