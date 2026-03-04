# _generate_cameras_by_lens 함수 종합 보고서

> Claude 작성 — 2025-03-04 22:41

## 1. 개요

`_generate_cameras_by_lens`는 3D Gaussian Splatting(3DGS) 기반 diffusion 편집에서 **편집 비용을 최소화하면서 3D 일관성을 유지하는 소수의 편집 뷰**를 자동으로 선택하는 핵심 함수이다. 이 함수는 내부적으로 `_lens_run_generate_by_lens_pipeline`을 호출하며, **ROI 분석 → SAGE-Probing → Fibonacci 샘플링 → 에너지 스코어링 → FPS 다양성 선택**의 5단계 파이프라인을 수행한다.

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

### 3.3 Step 2: SAGE-Probing (Unified SAGE Selection)

이 단계에서 **최적 카메라-객체 거리 $d^*$**를 결정한다.

#### 3.3.1 후보 거리 정의

$$
d_i = m_i \cdot r_{\text{obj}}, \quad m_i \in \{1.5, 2.0, 2.5, 3.0, 3.5\}
$$

각 $d_i$에 대해 ROI 정면 방향에 카메라를 배치하고, 다음 신호를 추출한다:

- **2D ROI 마스크** $M_i$: 3D ROI의 해당 뷰 투영
- **Cross-attention 맵** $A_i$: IP2P UNet의 편집 텍스트 토큰에 대한 cross-attention (timestep·head 평균)
- **Self-attention leakage** $\sigma_i$: ROI → 배경으로의 편집 신호 확산 강도

#### 3.3.2 Unified SAGE Score (IP2P 사용 시)

각 후보 거리 $d_i$에 대해 다음 **단일 연속 스코어**로 최적 거리를 선택한다:

$$
\boxed{
S_i = w_{\text{edit}} \cdot F_i \cdot (1 - \ell_i) \;-\; \lambda_{\text{sa}} \cdot \tilde{\sigma}_i
}
$$

$$
d^* = d_{i^*}, \quad i^* = \arg\max_i \; S_i
$$

여기서 **크기 정규화된 SA leakage** $\tilde{\sigma}_i$는 다음과 같이 정의한다:

$$
\tilde{\sigma}_i = \frac{\sigma_i}{o_i^{\,\alpha}}
$$

$o_i$는 뷰 $i$에서 ROI가 화면에서 차지하는 비율(occupancy)이고, $\alpha \in (0, 1]$는 정규화 지수이다.

**SA leakage 정규화의 동기:**

원시 SA leakage $\sigma_i$는 ROI occupancy $o_i$에 근사적으로 비례한다 ($\sigma_i \approx c \cdot o_i$, 실험적으로 $c \approx 0.9\text{–}1.3$). 이는 ROI가 화면에서 많은 토큰을 차지할수록 self-attention의 source 토큰 수가 증가하여, 배경 토큰이 ROI로부터 받는 attention 총량이 기하학적으로 증가하는 것에 기인한다. 따라서 원시 $\sigma_i$는 거리가 멀어질수록 (occupancy가 줄어들수록) 단조 감소하여, 정규화 없이는 항상 가장 먼 거리가 선택되는 편향이 발생한다.

$o_i^\alpha$로 나누면 이 크기 의존성이 제거되어, $\tilde{\sigma}_i$는 **동일한 occupancy 조건에서의 상대적 leakage 강도**만을 반영하게 된다. 지수 $\alpha$는 정규화의 강도를 조절한다: $\alpha = 1$이면 이론적으로 완전한 크기 불변(uniform attention 가정 하), $\alpha < 1$이면 보수적 정규화이다.

**변수 정의:**

| 기호 | 정의 | 산출 방법 |
|------|------|-----------|
| $F_i$ | Thresholded Precision-Recall F1 | CA 맵을 상위 25% quantile로 이진화 후 ROI 마스크와의 F1 |
| $\ell_i$ | CA Background Leakage | 배경 픽셀에서 cross-attention의 90th percentile |
| $\sigma_i$ | SA Propagation Leakage (원시) | ROI → 배경으로의 self-attention 확산 강도 |
| $\tilde{\sigma}_i$ | SA Propagation Leakage (정규화) | $\sigma_i / o_i^\alpha$ |
| $o_i$ | ROI Occupancy | 해당 뷰에서 ROI 마스크가 화면에서 차지하는 비율 |
| $\alpha$ | 정규화 지수 | 기본값 0.7 |
| $w_{\text{edit}}$ | 편집 품질 가중치 | 기본값 10.0 |
| $\lambda_{\text{sa}}$ | SA leakage 페널티 계수 | 기본값 10.0 |

**수식 해석:**

이 수식은 **편집 품질**과 **편집 안전성**의 균형을 단일 연속 함수로 표현한다:

1. **$w_{\text{edit}} \cdot F_i \cdot (1 - \ell_i)$ (편집 품질)**: cross-attention이 ROI 내부에 집중되면서($F_i$ ↑) 배경으로 새지 않는($\ell_i$ ↓) 정도. 가까운 거리에서 $F_i$는 높지만 $\ell_i$도 증가하므로, 이 곱은 중간 거리에서 최대가 된다.
2. **$\lambda_{\text{sa}} \cdot \tilde{\sigma}_i$ (정규화된 SA 전파 페널티)**: self-attention을 통한 편집 신호 전파 강도. 크기 정규화에 의해 거리에 대한 단조 감소 편향이 제거되었다.

$\ell_i$는 품질 항 내부의 $(1 - \ell_i)$에서 한 번만 사용되므로, CA leakage에 대한 이중 페널티가 없다. 두 항의 상호 작용에 의해, 가까운 거리에서는 편집 품질이 높지만 $(1-\ell_i)$로 인해 감쇠되고, 먼 거리에서는 ROI 정렬도($F_i$)가 감소하여, **중간 거리에서 자연스럽게 최적점(sweet spot)이 형성**된다.

#### 3.3.3 Heuristic 모드 (IP2P 미사용 시)

IP2P를 사용하지 않을 때는 attention 맵을 추출할 수 없으므로 다음 휴리스틱 점수를 사용한다:

$$
S_{\text{total}} = \text{focus} - \lambda_{\text{leak}} \cdot \text{leakage} - \lambda_{\text{ent}} \cdot \text{entropy} - \text{size\_penalty}
$$

- **focus**: occupancy 기반 — 최적 비율(0.30)에서 벗어날수록 감소
- **size_penalty**: occupancy가 극단적(< 0.10 또는 > 0.70)일 때 큰 페널티

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
- **선택 규칙**: $\text{score} = \min_{\text{selected}} \angle(\text{cand}, \text{sel}) + \text{energy}_{\text{norm}}(\text{cand})$ 최대화
- **마무리**: ROI 중심 기준 azimuth 순으로 정렬하여 `final_cameras` 반환

---

## 4. 구현 세부사항

### 4.1 IP2P Attention 수집

- **Cross-attention**: 편집 텍스트 토큰에 대한 attention을 timestep·head에 걸쳐 평균. `_LensStoringAttnProcessor`로 `.attn2` 레이어에서 수집
- **Self-attention leakage**: $8\times8$ (spatial=64) 저해상도 SA 맵 사용. `_LensStoringSAOnlySmallProcessor`가 target resolution에서만 explicit attention을 계산하고 나머지는 FlashAttention으로 처리하여 메모리 효율성 확보

$$
\sigma_i = \frac{1}{|\bar{\mathbf{m}}|} \sum_{q \in \text{bg}} \bigl(\bar{S} \cdot \mathbf{m}_{\text{roi}}\bigr)_q
$$

여기서 $\bar{S}$는 평균 SA 행렬, $\mathbf{m}_{\text{roi}}$는 ROI 마스크의 flatten 벡터

- **배치 모드** (`ip2p_batch_size > 1`): `_lens_run_ip2p_unified_batch`로 복수 뷰를 단일 UNet forward pass에서 동시 처리. 배치 차원을 head 차원과 분리하여 per-image attention 맵을 정확히 추출

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
| `lens_use_ip2p_scoring`               | IP2P 기반 Unified SAGE 사용  | True                  |
| `lens_ip2p_batch_size`                | IP2P 배치 크기                | 1                     |


---

## 5. 특성 및 장점

- **장면·객체 크기 적응성**: SA leakage의 occupancy 정규화($\sigma / o^\alpha$)가 객체 크기에 관계없이 일관된 거리 선택을 보장
- **연속적 스코어링**: 이진 게이트 없이 단일 연속 함수로 편집 품질과 안전성을 통합. Fallback 조건이나 불연속점이 없음
- **자연스러운 Sweet-spot 형성**: 편집 품질(가까울수록 유리)과 leakage 페널티(가까울수록 불리)의 길항 작용에 의해, 사전 지식 없이도 적절한 거리가 선택됨
- **해석 가능성**: 각 항이 명확한 물리적 의미를 가짐 — $F_i(1-\ell_i)$는 편집 품질 (CA 집중도 × 배경 비오염), $\tilde{\sigma}_i$는 크기 정규화된 SA 전파 강도
- **다양성 보장**: FPS로 기하학적으로 분산된 뷰 집합 구성

---

## 6. 한계 및 향후 과제

- **정규화 지수 $\alpha$의 선택**: $\alpha = 1$은 uniform attention 가정 하 이론적 최적이나, 실제 attention은 비균일하므로 $\alpha \in [0.5, 0.7]$이 경험적으로 더 효과적. 장면 유형별 최적 $\alpha$ ablation 필요
- **Attention proxy metric 의존**: noise prediction 차이(denoising direction) 등 픽셀 도메인 metric과의 결합 가능성
- **소형 객체의 높은 절대 SA leakage**: 객체가 매우 작은 경우 가장 가까운 거리에서도 occupancy가 낮아 정규화된 SA leakage가 크게 나타날 수 있음. 이 경우 후보 거리 간 SA leakage의 급격한 감소(elbow)를 감지하여 보정하는 후처리가 도움이 됨
