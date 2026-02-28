# 하이퍼파라미터 튜닝 가이드

## 추천 프레임워크 비교

### 1. **Weights & Biases (W&B) - 추천** ⭐
**장점:**
- 이미 코드베이스에 설정되어 있음
- 실험 추적 + 하이퍼파라미터 스윕 통합
- 웹 UI로 결과 비교 용이
- 자동으로 메트릭 추적

**단점:**
- 외부 서비스 의존 (또는 self-hosted 필요)

**설정:**
```bash
# 1. W&B 설치
pip install wandb

# 2. 로그인
wandb login

# 3. Sweep 생성
wandb sweep hyperparameter_tuning_wandb.py

# 4. Agent 실행 (여러 GPU에서)
wandb agent <sweep-id>
```

### 2. **Optuna** 
**장점:**
- 베이지안 최적화로 효율적 탐색
- 로컬 실행 가능
- 시각화 도구 내장

**단점:**
- 메트릭 파싱 로직 직접 구현 필요

**설정:**
```bash
pip install optuna optuna-dashboard

# 실행
python hyperparameter_tuning_optuna.py

# 대시보드
optuna-dashboard sqlite:///optuna.db
```

### 3. **Ray Tune**
**장점:**
- 분산 실행 최적화
- GPU 자동 관리
- 다양한 검색 알고리즘

**단점:**
- 설정이 복잡할 수 있음
- 메모리 사용량 높음

**설정:**
```bash
pip install ray[tune]

# 실행
python hyperparameter_tuning_ray.py
```

### 4. **개선된 Bash 스크립트**
**장점:**
- 현재 워크플로우와 완벽 호환
- 추가 의존성 없음
- 즉시 사용 가능

**단점:**
- 자동 최적화 없음 (그리드 서치만)
- 메트릭 추적 수동

**사용:**
```bash
chmod +x hyperparameter_tuning_improved.sh
./hyperparameter_tuning_improved.sh
```

## 현재 스크립트 개선 제안

현재 `man2clown_batch.sh`를 다음과 같이 개선할 수 있습니다:

1. **결과 저장**: JSON 형식으로 실험 결과 저장
2. **메트릭 파싱**: 로그에서 loss 값 추출
3. **재시작 기능**: 완료된 실험 건너뛰기
4. **결과 요약**: 최적 하이퍼파라미터 자동 찾기

## 추천 워크플로우

1. **빠른 탐색**: 개선된 bash 스크립트로 그리드 서치
2. **정밀 튜닝**: W&B Sweep으로 베이지안 최적화
3. **결과 분석**: W&B 대시보드에서 비교

## 예시: W&B Sweep 설정

```yaml
# sweep_config.yaml
program: launch.py
method: bayes
metric:
  name: val_loss
  goal: minimize
parameters:
  system.loss.lambda_d:
    distribution: categorical
    values: [30.0, 40.0, 50.0, 100.0, 150.0, 180.0]
  data.max_view_num:
    distribution: categorical  
    values: [25, 30, 35, 40]
  system.guidance.guidance_scale:
    distribution: uniform
    min: 10.0
    max: 15.0
```

## 다음 단계

1. 어떤 프레임워크를 사용할지 결정
2. 메트릭 파싱 로직 추가 (로그에서 loss 추출)
3. 결과 비교 스크립트 작성





