# _generate_cameras_by_lens 함수 종합 보고서

## 1. 개요

`_generate_cameras_by_lens`는 3D Gaussian Splatting(3DGS) 기반 diffusion 편집에서 **편집 비용을 최소화하면서 3D 일관성을 유지하는 소수의 편집 뷰**를 자동으로 선택하는 핵심 함수이다. 이 함수는 내부적으로 `_lens_run_generate_by_lens_pipeline`을 호출하며, **ROI 분석 → SAGE-Probing(또는 Two-Phase SAGE) → Fibonacci 샘플링 → 에너지 스코어링 → FPS 다양성 선택**의 5단계 파이프라인을 수행한다.

### 1.1 동기

Diffusion 기반 3DGS 편집에서 편집 비용은 뷰 수에 비례하여 선형 증가한다. 모든 COLMAP 뷰를 편집하면 계산 부담이 크고, 무작위로 소수만 선택하면 ROI에 대한 편집 품질과 3D 일관성이 보장되지 않는다. 따라서 다음 두 조건을 동시에 만족하는 컴팩트한 카메라 집합이 필요하다:

1. **ROI에 국소적·안정적인 편집**: 편집 신호가 관심 영역에 집중되고 배경으로 과도하게 확산되지 않는 뷰
2. **다양한 기하학적 커버리지**: 선택된 뷰들이 서로 다른 방향에서 장면을 커버하여 3D 일관성 최적화에 유리한 기하 정보 제공

---

## 2. 함수 구조 및 호출 흐름

### 2.1 진입점

- **위치**: `threestudio/data/gs_load.py` — `_generate_cameras_by_lens(max_view_num, max_edit_view_num)`
- **호출 시점**: 데이터셋 초기화 시 `lens` 모드가 활성화되어 있을 때, COLMAP 카메라 대신 LENS 파이프라인으로 생성된 카메라로 씬을 대체한다.

### 2.2 주요 단계

1. **초기화**: COLMAP prior 로드(`_lens_load_colmap_prior`), Gaussian 모델 해상(`lens_gaussian_model` 또는 `gaussian_model`), IP2P 파이프라인(공유) 준비
2. **파이프라인 실행**: `_lens_run_generate_by_lens_pipeline` 호출
3. **후처리**: 반환된 `Simple_Camera` 리스트로 씬 카메라 교체, `train_view_indices` 및 `edit_view_indices` 반환

---

## 3. 파이프라인 상세: _lens_run_generate_by_lens_pipeline

### 3.1 Step 0: ROI 마스크 생성 (선택)

- **조건**: `seg_prompt`가 주어지고 `roi_mask`가 없을 때
- **방법**: LangSAM 등 세그멘터로 여러 COLMAP 뷰에서 2D 마스크 추출 → 3D 가우시안에 역투영하여 ROI 가우시안 집합 결정
- **목적**: 이후 단계에서 ROI만 사용하여 거리·방향·가시성 계산

### 3.2 Step 1: ROI 내재 분석 (ROI Intrinsic Analysis)

- **함수**: `_lens_roi_intrinsic_analysis`
- **출력**: ROI 중심 `center`, 주축 `v1, v2, v3`, 고유값, 객체 크기 `object_size`, 정면 방향 `v_front`
- **핵심 연산**:
  - 가중 공분산 행렬로부터 고유값 $\lambda_1 \geq \lambda_2 \geq \lambda_3$ 및 주축 계산
  - 특성 반경 $r_{\text{obj}} = \sqrt{\lambda_1}$ (또는 median 기반 스케일)
  - 전체 장면의 중심 `scene_center`와 ROI 중심 `center` 사이 벡터로부터 전방 방향 `v_front`를 결정:
    $$
    \mathbf{v}_{\text{front}} = \mathrm{normalize}(\text{scene\_center} - \mathbf{c})
    $$
    즉, ROI가 전체 씬 중심을 향하도록 하는 방향을 전방으로 사용하며, COLMAP 카메라의 전방 벡터에는 더 이상 의존하지 않는다.

### 3.3 Step 2: SAGE-Probing / Two-Phase SAGE Selection

이 단계에서 **최적 카메라-객체 거리 $d^*$**를 결정한다.

#### 3.3.1 후보 거리 정의

$$
d_i = m_i \cdot r_{\text{obj}}, \quad m_i \in 1.5, 2.0, 2.5, 3.0, 3.5
$$

각 $d_i$에 대해 ROI 정면 방향에 카메라를 배치하고, 다음 신호를 추출한다:

- **2D ROI 마스크** $M_i$: 3D ROI의 해당 뷰 투영
- **Cross-attention 맵** $A_i$: IP2P UNet의 편집 텍스트 토큰에 대한 cross-attention (timestep·head 평균)
- **Self-attention leakage** $\sigma_i$: ROI → 배경으로의 편집 신호 확산 강도

#### 3.3.2 Two-Phase SAGE Selection (IP2P 사용 시)

**Phase 1: Self-Attention Containment Gate**

- Self-attention leakage $\sigma_i$로 "편집 신호가 배경까지 과도하게 확산되는 뷰"를 제거
- Adaptive threshold: $\tau_{\text{sa}} = \frac{1}{N}\sum_{i=1}^{N} \sigma_i$
- 안전 후보: $\mathcal{I}*{\text{safe}} = i \mid \sigma_i \leq \tau*{\text{sa}}$

**Phase 2: Cross-Attention Quality Ranking**

- 안전 후보에 대해 품질 지표 계산:
  - **F1** $F_i$: Thresholded precision-recall (상위 25% quantile 이진화 후 ROI와의 F1)
  - **Background leakage** $\ell_i$: 배경 픽셀에서 attention의 90th percentile
- 최적 선택: $d^* = d_{i^*}$, $i^* = \arg\max_{i \in \mathcal{I}_{\text{safe}}} F_i \cdot (1 - \ell_i)$

#### 3.3.3 Heuristic 모드 (IP2P 미사용 시)

- `_lens_compute_editability_score`: occupancy 기반 focus, size_penalty 등 휴리스틱 점수
- SAGE 점수: $\text{S}*{\text{total}} = \text{focus} - \lambda*{\text{leak}} \cdot \text{leakage} - \lambda_{\text{ent}} \cdot \text{entropy} - \text{sizepenalty}$

### 3.4 Step 3: Fibonacci Manifold Sampling

- **목적**: 거리 $d^*$에서 ROI 중심을 바라보는 **방향**을 구면(또는 반구) 위에 균등하게 샘플링
- **방법**: Golden angle $\pi(3-\sqrt{5})$ 기반 Fibonacci 구면 샘플
- **필터**: `hemisphere_only`, `cone_half_angle_deg` (COLMAP 평균 방향 기준 cone)
- **결과**: 수십~수백 개의 후보 카메라

### 3.5 Step 4: Energy-based Scoring

- **가시성** $S_{\text{vis}}$: ROI 2D 마스크가 화면에서 차지하는 비율
- **정면/측면** $S_{\text{can}}$: $\max(\cos_{\text{front}}, 0.8 \cdot \cos_{\text{side}})$
- **에너지**: $E = w_{\text{vis}} \cdot S_{\text{vis}} + w_{\text{can}} \cdot S_{\text{can}}$
- 에너지 내림차순 정렬 후 상위 `top_fraction` 비율만 풀에 포함

### 3.6 Step 5: Diversity-aware Selection (FPS)

- **목적**: 에너지 상위 풀에서 **서로 다른 방향**을 커버하도록 FPS(Farthest Point Sampling) 유사 선택
- **선택 규칙**: $\text{score} = \min_{\text{selected}} \angle(\text{cand}, \text{sel}) \times \text{energy}(\text{cand})$ 최대화
- **마무리**: ROI 중심 기준 azimuth 순으로 정렬하여 `final_cameras` 반환

---

## 4. 구현 세부사항

### 4.1 IP2P Attention 수집

- Cross-attention: 편집 텍스트 토큰에 대한 attention을 timestep·head에 걸쳐 평균
- Self-attention leakage: $8\times8$ 등 저해상도 맵 사용, $\sigma_i = \bar{\mathbf{m}}^\top (\bar{S}\mathbf{m}) / \bar{\mathbf{m}}_1$
- 배치 모드(`ip2p_batch_size > 1`) 지원으로 복수 뷰 동시 처리

### 4.2 출력 활용

- `train_view_indices`: 전체 LENS 카메라 인덱스 (방위각 정렬)
- `edit_view_indices`: `train_view_indices`의 앞에서부터 `max_edit_view_num`개 — 실제 편집에 사용되는 키 뷰

### 4.3 주요 설정 파라미터


| 파라미터                                  | 의미                        | 예시                    |
| ------------------------------------- | ------------------------- | --------------------- |
| `lens_distance_multipliers`           | 거리 후보 배수                  | "1.5,2.0,2.5,3.0,3.5" |
| `lens_n_candidates`                   | Fibonacci 후보 개수           | 150                   |
| `lens_hemisphere_only`                | 반구만 사용                    | False                 |
| `lens_cone_half_angle_deg`            | Cone 필터 각도                | 90                    |
| `lens_w_vis`, `lens_w_can`            | 가시성/정면 가중치                | 0.6, 0.4              |
| `lens_top_fraction`                   | 에너지 상위 비율                 | 0.20                  |
| `lens_lambda_leak`, `lens_lambda_ent` | SAGE leakage/entropy      | 1.5, 2.0              |
| `lens_use_ip2p_scoring`               | IP2P 기반 Two-Phase SAGE 사용 | True                  |


---

## 5. 특성 및 장점

- **장면·객체 크기 적응성**: Phase 1 adaptive threshold가 장면별 leakage 분포에 자동 적응
- **하이퍼파라미터 의존성 감소**: per-run 상대적 필터링, 가중치 없는 품질 점수 곱셈
- **해석 가능성**: Phase 1(안전성) vs Phase 2(품질) 역할 분리
- **다양성 보장**: FPS로 기하학적으로 분산된 뷰 집합 구성

---

## 6. 한계 및 향후 과제

- Phase 1 임계값: 산술 평균 기반 → median/percentile 기반 ablation 검토
- Attention proxy metric 의존: noise prediction 차이(denoising direction) 등 픽셀 도메인 metric과의 결합 가능성

