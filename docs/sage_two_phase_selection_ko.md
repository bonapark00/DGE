## 1. 문제 정의

본 연구의 목표는 텍스트 기반 이미지 편집 모델(예: InstructPix2Pix)을 활용하여 3D 장면의 특정 객체(Region of Interest, ROI)를 편집할 때, **여러 카메라 거리 후보 중에서 가장 국소적으로 편집이 잘 되는 뷰를 자동으로 선택**하는 것이다.

구체적으로, Gaussian Splatting으로 표현된 3D 장면과 LangSAM 기반의 3D ROI 마스크가 주어졌다고 하자. 카메라 전방 방향 벡터를 따라 여러 거리 d_i 에서 2D 뷰를 렌더링하고, 각 뷰에 대해 다음 정보를 얻는다.

- 2D ROI 마스크 M_i (0/1 바이너리 마스크, 크기 H×W)
- 텍스트 조건 하에서 InstructPix2Pix UNet의 **cross-attention 맵** A_i
- UNet self-attention으로부터 추출한 **self-attention leakage** 척도 sa_i (ROI → 배경 전파량)

이들 신호를 이용해, **원하는 객체만 잘 바뀌고 주변 배경은 최대한 유지되는 카메라 거리 d\*** 를 선택하는 것이 문제의 핵심이다.

---

## 2. 기존 SAGE 기반 접근과 한계

### 2.1 기존 점수식 개요

이전 버전의 SAGE(SHarpness-Aware Guided Editability) 스코어는 대략 다음과 같은 형태를 가진다.

> S_i = C_i * F1_i − λ_ca * L_CA_i − λ_sa * sa_i − SizePenalty_i

- **C_i**: ROI와 배경 간 cross-attention의 대비(contrast ratio)  
- **F1_i**: 상위 attention 영역과 ROI 마스크 사이의 정밀도–재현율 F1 점수  
- **L_CA_i**: 배경 픽셀에서의 상위 90% cross-attention(강한 background leakage)  
- **sa_i**: self-attention을 통한 ROI→배경 전파량(self-attention leakage)  
- **SizePenalty_i**: ROI 점유율이 너무 작거나 클 때 부과되는 패널티

직관적으로는,  
**“ROI 쪽으로 집중된 attention은 보상하고, 배경으로 새어 나가는 attention과 부적절한 크기의 ROI는 벌점 준다”** 는 설계이다.

### 2.2 Self-attention leakage와 ROI 비율의 강한 상관

실험적으로, 각 거리에서의 ROI 점유율 occ_i = mean(M_i) 와  
self-attention leakage sa_i 사이의 상관계수를 계산해 보면, 거의 항상

> corr(sa_i, occ_i) ≈ 0.96

수준으로 매우 높음을 확인할 수 있었다.

- ROI가 화면에서 차지하는 비율이 커질수록(occ_i 증가)  
  self-attention으로 배경 픽셀이 ROI 픽셀을 참조하는 정도 sa_i 도 기계적으로 증가한다.
- 즉, sa_i 는 “편집이 배경으로 번지는 정도”뿐 아니라, **ROI 크기 자체**에 강하게 묶여 있다.

이 상태에서 occ_i 와 sa_i 를 동시에 점수식에 포함시키면,

- ROI가 적절히 크게 잡힌 뷰도, “크다 = leakage 크다”는 이유로 이중 벌점을 받게 되고,
- 실제로는 **작은 객체인데 근거리에서 장면 전체가 함께 바뀌는 나쁜 leakage**와  
  **얼굴 전체를 잘 담아서 편집하기 좋은 큰 ROI**를 구분하기 어려워진다.

### 2.3 단일 스칼라 점수의 구조적 한계

여러 장면(작은 피규어, 사람 얼굴, 중간 크기 물체 등)에 대해 사용자가 선호하는 거리를 수집하고,

> S_i = C_i^a * F1_i^b * (1 − sa_i)^k

형태(혹은 이와 유사한 곱셈 형태)에 다양한 가중치를 부여해 보았으나, 아래와 같은 구조적 문제가 드러났다.

- **작은 ROI**(예: 방 한켠의 작은 피규어):
  - 근거리에서 edit가 장면 전체로 쉽게 전파되므로, leakage에 매우 민감해야 한다.  
  - → k 가 크게 필요
- **큰 ROI**(예: 얼굴, 스피커 등):
  - 일정 수준 근거리에서 edit가 가장 잘 드러나므로, leakage를 지나치게 강하게 벌점 주면  
    항상 지나치게 먼 뷰가 선택된다.  
  - → k 가 작게 필요

두 요구사항을 동시에 만족하는 (a, b, k) 조합이 존재하지 않음을,  
부등식 수준의 분석을 통해 확인할 수 있었다.  
즉, **하나의 스칼라 점수와 고정 하이퍼파라미터로는 상충하는 선호를 모두 만족시키기 어렵다**는 것이 한계이다.

---

## 3. 제안 방법: 2-Phase SAGE Selection

이러한 한계를 해결하기 위해, 본 연구에서는 SAGE 점수를

- **Phase 1: Self-attention 기반 “안전성 필터”**,  
- **Phase 2: Cross-attention 기반 “편집 품질 순위”**

의 두 단계로 분리하는 **2-Phase SAGE Selection**을 제안한다.

핵심 아이디어는 다음과 같다.

- self-attention leakage sa_i 는 “편집 신호가 ROI 밖으로 퍼지는 위험도”를 나타내지만,  
  절대값보다는 **동일 장면·동일 프롬프트에서의 상대적인 크기**가 더 중요하다.
- cross-attention 기반 품질 지표(F1_i, L_CA_i) 는  
  **SA leakage가 과도하지 않은 후보에 대해서만** 신뢰할 수 있는 품질 척도이다.

### 3.1 Phase 1: Self-Attention 기반 Containment Filter

동일 장면·동일 편집 프롬프트에 대해 여러 거리 d_i 에서 self-attention leakage sa_i 를 계산한다.

1. 후보 집합 { i = 1, …, N } 에 대해 평균 leakage를 구한다.
   - sa_bar = (1/N) * Σ_i sa_i
2. **안전한(safe) 후보 집합**을
   - I_safe = { i | sa_i ≤ sa_bar }
   로 정의한다.
   - self-attention leakage가 평균보다 큰 뷰는  
     “편집 신호가 ROI를 넘어 배경에 과도하게 전파되는 뷰”로 보고 1차적으로 제거한다.
3. 극단적인 경우 I_safe 가 공집합이면, 필터를 적용하지 않고 전체 후보를 사용한다.

이는 sa_leak에 대해 **절대 임계값**을 튜닝하는 대신,  
각 probing run 내에서 상대적으로 “위험한” 뷰만 제거하는 **adaptive filtering**으로 볼 수 있다.

### 3.2 Phase 2: Cross-Attention 기반 품질 순위

Phase 1을 통과한 안전한 후보들에 대해서는, cross-attention 품질에 기반한 점수로 순위를 매긴다.

각 후보 i 에 대해:

- ROI 마스크 M_i 와 정규화된 attention 맵 A_i 가 주어지고,
- 상위 attention 영역(예: 상위 25% quantile)과 ROI 사이의 precision–recall F1 점수 F1_i,
- ROI 밖 배경 픽셀에서 상위 90% attention 크기 L_CA_i (background cross-attention leakage)를 계산한다.

이때 **최종 품질 점수**는

> Q_i = F1_i * (1 − L_CA_i)

로 정의한다.

- F1_i 가 클수록, “강한 attention 영역이 ROI 전체를 잘 덮고 있음”.
- (1 − L_CA_i) 가 클수록, “강한 attention이 배경에 덜 분포”함.

최종 선택되는 후보는

> i\* = argmax_{i ∈ I_safe} Q_i  
> d\* = d_{i\*}

와 같이 정의된다.

결과적으로,

- **Phase 1**은 self-attention 관점에서 “편집이 과도하게 퍼지는 뷰”를 제거하는 **안전성 필터**,
- **Phase 2**는 cross-attention 관점에서 “ROI에 잘 붙고 배경은 최소로 건드리는 뷰”를 선택하는 **품질 순위 단계**로 해석할 수 있다.

---

## 4. 구현 관점 요약

코드 상에서는 다음과 같은 흐름으로 구현된다.

- **IP2P 실행 및 주의 맵 수집**
  - 각 거리 후보에 대해 InstructPix2Pix 파이프라인을 실행하며,  
    모든 timestep/헤드의 cross-attention을 평균하여 특정 해상도(예: 16×16)의 attention 맵 A_i 를 얻는다.
  - self-attention 프로세서를 후킹하여, ROI 픽셀에서 배경 픽셀로의 attention을 적분함으로써 sa_i 를 계산한다.

- **SAGE 세부 지표 계산 (`_compute_sage_score`)**
  - ROI/배경 대비를 통해 contrast C_i,
  - thresholded attention과 ROI overlap으로부터 F1_i,
  - 배경 상위 90% attention으로부터 L_CA_i 를 계산하고,  
    이들을 `details` 딕셔너리에 저장한다.

- **거리 선택 (`scale_probing`)**
  - 모든 후보의 `details["sa_leakage"]`에 대해 평균 sa_bar 를 계산하고,  
    `sa_leakage ≤ sa_bar` 인 후보들만 남긴다.
  - 남아 있는 후보들에 대해 `precision_f1 * (1 - leakage)`  
    (여기서 `leakage`는 L_CA_i) 를 최종 품질 점수로 사용하고,  
    이 값이 최대인 후보의 거리를 최종 선택한다.

- **시각화 (`_save_attention_grid`)**
  - 각 거리×해상도에 대해 attention heatmap과 ROI 마스크를 함께 표시하고,  
    contrast·F1·SA leakage 등의 지표를 텍스트로 overlay하여,  
    선택된 뷰에서 attention이 ROI에 집중되고 배경으로의 leak이 줄어드는 양상을 직관적으로 확인할 수 있도록 한다.

---

## 5. 특성 및 장점

- **장면/객체 크기에 대한 적응성**
  - ROI가 매우 작은 경우(작은 피규어),  
    근거리에서 self-attention을 통한 전파가 커져 장면 전체가 바뀌는 뷰는 1차 필터에서 제거되고,  
    그 중에서 가장 편집이 잘 걸리는 중간 거리가 선택된다.
  - 얼굴, 가전제품처럼 ROI가 큰 경우에는,  
    적당한 근거리에서 attention 품질이 높게 나오며,  
    Phase 1 필터는 과도하게 leak가 큰 뷰만 제거하고 나머지 중간·원거리 후보 중에서  
    품질 점수로 최적 거리를 찾는다.

- **하이퍼파라미터 의존성 감소**
  - 기존처럼 여러 λ(예: λ_ca, λ_sa)를 수동 튜닝하기보다,  
    self-attention leakage에 대해서는 **per-run 평균**을 기준으로 필터링하기 때문에,  
    다양한 장면·프롬프트 조합에 대해 보다 견고하게 동작한다.

- **해석 가능성**
  - Phase 1: “이 뷰는 self-attention이 커서 편집 신호가 배경까지 과도하게 번진다”  
  - Phase 2: “남은 안전한 뷰들 중에서, ROI에 대한 attention 커버리지는 높고 배경에 대한 leak는 낮은 뷰를 선택한다”
  라는 식으로, 각 단계의 역할이 명확하다.

---

## 6. 한계 및 향후 과제

- 현재 SA 필터는 단순 평균 sa_bar 를 기준으로 한다.  
  장면별로 분포 특성(분산, 왜도 등)이 다르므로,  
  median 또는 특정 퍼센타일 기반 임계값을 사용하는 변형에 대한 ablation이 향후 과제로 남아 있다.

- 제안 방법은 여전히 attention 기반 **proxy metric**에 의존한다.  
  후속 연구로, diffusion 모델의 **noise prediction 차이(denoising direction map)** 를 활용하여  
  실제 픽셀 도메인에서의 편집 변화량을 직접적으로 측정하는 metric과 결합하는 방향을 고려할 수 있다.

이 보고서는 코드 구현(`scale_probing`, `_compute_sage_score`, self-attention leakage 계산부 등)을 참고하여 작성되었으며,  
이를 기반으로 방법론 섹션·수식·시각화(주의 맵 grid + 선택된 뷰 표시)를 갖춘 논문 형식으로 확장할 수 있다.


