# 3D Gaussian Splatting을 위한 기하학 인식 다시점 일관 편집 파이프라인

## 1. 개요 (Overview)

본 파이프라인은 3D Gaussian Splatting(3DGS)으로 표현된 장면을 텍스트 지시문(InstructPix2Pix 기반)으로 편집할 때, **다시점 기하학적 일관성(multi-view geometric consistency)**을 확보하는 것을 목표로 한다. 기존 방법론들(DGE, VCEdit)이 self-attention 또는 cross-attention 레벨의 일관성만 다루었던 것과 달리, 본 방법은 **매 디노이징 스텝마다 3D 기하학을 경유하여 cross-attention map을 동기화**함으로써, "어디에 무엇을 배치할 것인가"라는 공간적 의사결정을 뷰 간에 명시적으로 통일한다.


## 2. 문제 정의 (Problem Statement)

3DGS 장면의 다시점 이미지를 2D 확산 모델(diffusion model)로 편집할 때, 각 뷰가 독립적으로 디노이즈되면 다음 문제가 발생한다:

- **기하학적 불일치**: 같은 3D 점(예: 코, 눈)이 뷰마다 다른 위치에 생성 → 3DGS 최적화 시 mode collapse
- **스타일 불일치**: 뷰마다 다른 색감, 텍스처 적용
- **편집 누락**: 특정 뷰에서 편집 지시가 반영되지 않음

이 중 **기하학적 불일치**가 가장 치명적이며, 이는 확산 모델의 cross-attention(attn2)이 뷰마다 독립적으로 "텍스트 토큰 → 공간 위치" 매핑을 결정하기 때문에 발생한다.


## 3. 관련 연구 및 차별점

### 3.1 DGE (Direct Gaussian 3D Editing) [Chen et al., ECCV 2024]

DGE는 InstructPix2Pix를 다시점 일관 편집기로 변환하는 training-free 접근법을 제안한다.

**핵심 메커니즘:**
- **Extended Self-Attention**: 배치 내 뷰들의 self-attention key/value를 연결(concatenate)하여, 각 뷰가 다른 뷰의 spatial token을 참조할 수 있도록 함
- **Pivotal Forward Pass**: 대표 뷰(pivotal)를 먼저 처리하여 self-attention 출력을 캐시하고, 나머지 뷰에 feature injection으로 전파
- **Epipolar Constraint**: self-attention 시 에피폴라 기하학을 마스크로 적용하여 기하학적으로 대응하는 토큰끼리만 attend하도록 제한

**한계:**
- Cross-attention(attn2)은 다루지 않음 → "어디에 무엇을 배치할지"는 뷰마다 독립 결정
- Self-attention의 content-based matching만으로는 기하학적 대응을 보장할 수 없음
- Pivotal 선택이 배치 내 랜덤 → 최적 커버리지 미보장
- 한 번 편집한 이미지로 3DGS를 직접 최적화하므로, 편집 품질이 일관성에 전적으로 의존

### 3.2 VCEdit (View-Consistent Editing) [Wang et al., ECCV 2024]

VCEdit은 3DGS 편집을 위한 두 가지 일관성 모듈을 제안한다.

**핵심 메커니즘:**
- **Cross-attention Consistency Module**: 뷰 간 cross-attention map의 불일치를 줄이는 모듈
- **Editing Consistency Module**: 편집 결과의 일관성을 반복적(iterative) 패턴으로 개선

**한계:**
- 2D feature space에서의 일관성만 다룸 → 3D 기하학 정보를 명시적으로 활용하지 않음
- 반복적 최적화 필요 → 느린 수렴 (A100에서 10-25분)
- Cross-attention 일관성이 attention 가중치 수준에 머물며, 3D 역투영(inverse rendering)을 사용하지 않음

### 3.3 TokenFlow [Geyer et al., ICLR 2024]

비디오 편집에서 self-attention token을 프레임 간 전파하여 일관성을 확보하는 training-free 방법론.

**한계:**
- 2D optical flow 기반 → 3D 기하학 미고려
- 비디오(시간 축) 전용 → 다시점(공간 축)에 직접 적용 불가

### 3.4 본 방법의 차별점 요약

| 측면 | DGE | VCEdit | TokenFlow | **본 방법** |
|------|-----|--------|-----------|------------|
| Self-Attention 일관성 | Extended SA + Epipolar | — | Token 전파 | Extended SA + Feature Injection |
| Cross-Attention 일관성 | ✗ 미처리 | 2D 일관성 모듈 | ✗ 미처리 | **3D 역투영 기반 매 스텝 동기화** |
| 3D 기하학 활용 | Epipolar (간접) | ✗ | ✗ | **GS inverse/re-render (직접)** |
| 동적 갱신 | ✗ 1회 고정 | 반복 최적화 | ✗ 1회 고정 | **매 디노이징 스텝 갱신** |
| Pivotal 선택 | 배치 내 랜덤 | — | — | **Canonical-Progressive** |
| Training 필요 | ✗ | ✗ | ✗ | ✗ |


## 4. 파이프라인 구조

### 4.1 전체 흐름

```
Training Step
  └─ edit_multiview()
       └─ 3DGS 렌더링 → 다시점 이미지 (N장)
       └─ guidance(use_multiview=True)
            ├─ VAE 인코딩 → latent z₀
            ├─ 텍스트 인코딩 → text embedding
            └─ edit_latents_multiview()
                 ├─ [Phase 3] Target Denoise Loop
                 │    ├─ 매 timestep t:
                 │    │    ├─ Pivotal Forward (대표 뷰 처리, kf_attn_output 캐시)
                 │    │    ├─ Per-Step Cross-Attn Consistency (3D 경유)
                 │    │    └─ Batch Forward (일관된 cross-attn map으로 편집)
                 │    └─ 최종 편집된 latent
                 └─ VAE 디코딩 → 편집된 이미지
```

### 4.2 핵심 컴포넌트 상세

#### 4.2.1 Per-Step Cross-Attention Consistency (핵심 기여)

기존 방법들이 cross-attention을 무시하거나 2D에서만 처리하는 것과 달리, 본 방법은 **매 디노이징 스텝마다** 다음 과정을 수행한다:

**Step 1. Cross-Attention Map 수집**
- Pivotal forward 시 `CrossAttentionStoreProcessor`를 attn2 모듈에 설치
- Pivotal 뷰들의 cross-attention probability를 수집:
  $A_v^{(t)} \in \mathbb{R}^{HW \times L}$ (뷰 $v$, timestep $t$, 토큰 수 $L$)

**Step 2. 3D Inverse Rendering (2D → 3D)**
- 각 Gaussian $g$에 대해, 보이는 뷰들의 attention weight를 역투영하여 축적:

$$M_{3D}^{(t)}(g, l) = \frac{\sum_{v \in \mathcal{V}_{piv}} w_v(g) \cdot A_v^{(t)}(\pi_v(g), l)}{\sum_{v \in \mathcal{V}_{piv}} w_v(g)}$$

여기서 $\pi_v(g)$는 Gaussian $g$의 뷰 $v$에서의 2D 투영, $w_v(g)$는 가시성 가중치.

**Step 3. 3D → 2D Re-Rendering**
- 3D attention field $M_{3D}^{(t)}$를 각 target 뷰 카메라에서 Gaussian splatting으로 렌더링:

$$\hat{A}_i^{(t)}(p, l) = \text{GS-Render}(M_{3D}^{(t)}(\cdot, l), \text{cam}_i, p)$$

**Step 4. Cross-Attention 대체**
- 원래의 $\text{softmax}(QK^T)V$ 대신, 일관된 map으로 대체:

$$h_{out} = \hat{A}_i^{(t)} \cdot V_{valid}$$

여기서 $V_{valid}$는 유효 텍스트 토큰에 대응하는 value 벡터.

**효과**: 같은 3D Gaussian에 대해 모든 뷰가 동일한 cross-attention weight를 받으므로, "코는 여기에, 눈은 저기에"라는 공간 배치가 기하학적으로 일치하게 된다.

#### 4.2.2 Extended Self-Attention with Feature Injection

DGE에서 가져온 메커니즘을 개선하여 사용:

1. **Pivotal Forward**: 각 배치의 대표 뷰를 먼저 UNet에 통과시키며, 모든 pivotal 뷰 간 extended self-attention 수행. Self-attention 출력(`kf_attn_output`)을 캐시.

2. **Batch Forward**: 각 배치의 뷰들이 UNet을 통과할 때:
   - Self-attention(attn1): 캐시된 `kf_attn_output`에서 가장 가까운 pivotal의 feature를 similarity 기반으로 injection
   - Cross-attention(attn2): Per-step consistent map으로 대체

#### 4.2.3 Adaptive Batching

디노이징 단계에 따라 배치 구성 전략을 동적으로 변경:

- **초기 (t ≥ threshold)**: 이웃한 뷰들끼리 순차 배치 → 인접 뷰 간 구조적 일관성 우선
- **후기 (t < threshold)**: Sliding window 방식으로 매 스텝마다 배치 시작점을 이동:
  
  $\text{offset}(s) = (s \times \text{stride}) \mod N$

  → 매 스텝마다 다른 뷰 조합이 같은 배치에 포함되어, 전체적인 cross-view 정보 전파

#### 4.2.4 Canonical-Progressive Pivotal Selection

각 배치의 대표 뷰(pivotal)를 선택하는 전략:

- **초기 (t ≥ threshold)**: Canonical Score가 가장 높은 뷰 선택

  $\text{score}(i) = \cos(\vec{d}_i, \bar{\vec{d}})$

  여기서 $\vec{d}_i$는 뷰 $i$에서 물체 중심까지의 방향, $\bar{\vec{d}}$는 전체 뷰 방향의 평균.
  → 가장 정면에 가까운 대표적 뷰가 pivotal로 선택되어 초기 구조 결정에 유리.

- **후기 (t < threshold)**: Progressive Coverage — pivotal로 가장 적게 선택된 뷰를 우선 (동률 시 canonical score로 tie-break)

  $\text{pivotal}(b) = \arg\min_{i \in b} \text{count}(i), \quad \text{tie-break by } \text{score}(i)$

  → 모든 뷰가 골고루 pivotal 역할을 수행하여 정보 전파의 균형 확보.


## 5. 구현 상세

### 5.1 Config 옵션

| Config | 기본값 | 설명 |
|--------|--------|------|
| `per_step_cross_attn_consistency` | `false` | Per-step 3D cross-attn consistency 활성화 |
| `per_step_cross_attn_t_start` | `500` | t ≥ 이 값에서만 per-step consistency 적용 |
| `target_batch_strategy` | `fixed` | `fixed` (순차) / `adaptive` (시간 의존) |
| `target_batch_neighbor_threshold` | `500` | 초기/후기 전환 timestep |
| `target_batch_late_mode` | `sliding_window` | 후기 모드: `sliding_window` / `random` |
| `target_batch_sliding_stride` | `-1` | sliding window stride (-1 = batch_size // 2) |
| `target_key_selection_mode` | `random` | `random` / `canonical_progressive` / `fixed` |
| `feature_injection_mode` | `similarity` | `similarity` / `3d_anchor` / `none` |
| `camera_batch_size` | `5` | 배치 당 뷰 수 |

### 5.2 핵심 클래스 및 함수

| 클래스/함수 | 역할 |
|-------------|------|
| `ConsistentCrossAttnProcessor` | attn2 프로세서: 일관된 cross-attn map으로 출력 대체 |
| `CrossAttentionStoreProcessor` | attn2 프로세서: cross-attn probability 수집 |
| `_per_step_build_consistent_maps()` | 수집된 map을 3D inverse render → 2D re-render |
| `_build_target_batches()` | 적응적 배치 구성 |
| `_select_pivotals_for_batches()` | Canonical-progressive pivotal 선택 |
| `make_dge_block()` | Extended SA + feature injection이 적용된 DGE 블록 |


## 6. Phase 1 제거의 근거

기존 DGE 파이프라인은 Phase 1(key denoise loop)에서 별도의 뷰 그룹을 먼저 완전히 디노이즈한 후 cross-attention map을 수집했다. 본 방법에서 이를 제거한 이유:

1. **Static map의 한계**: Phase 1에서 한 번 만든 map은 디노이징 과정에서 변화하는 latent 상태를 반영하지 못함
2. **Pivotal forward 재활용**: Target loop에서 매 스텝 수행하는 pivotal forward에서 직접 cross-attn을 수집하면, 현재 상태를 반영하는 fresh map 생성 가능
3. **계산 비용 절감**: key view 20 step 디노이징 비용 제거 (UNet forward × 20 절약)


## 7. 기대 효과 및 향후 연구

### 기대 효과
- 기하학적 일관성 향상: cross-attn의 3D 동기화로 "코가 여러 군데" 같은 아티팩트 감소
- 편집 품질 향상: 매 스텝 fresh map으로 디노이징 과정과 동기화된 guidance 제공
- 속도 향상: Phase 1 제거로 전체 편집 시간 단축

### 향후 연구 방향
- **Geometry-biased Self-Attention**: Extended SA에 3D correspondence bias를 추가하여 self-attention도 기하학 인식으로 만들기
- **Correlated Noise**: 3D 공간에서 noise field를 생성하여 뷰 간 noise 일관성 확보
- **해상도 확장**: 현재 32×32, 64×64 해상도에서만 일관성 적용 → 더 높은/낮은 해상도까지 확장
