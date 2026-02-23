# LENS 카메라 생성 파이프라인 보고서

본 문서는 `threestudio/data/gs_load.py`의 **`_lens_run_generate_by_lens_pipeline`** 에서 사용하는 카메라 자동 생성 파이프라인을 정리한 기술 보고서입니다.  
**ROI 분석 → SAGE-Probing → Fibonacci 샘플링 → 에너지 스코어링 → 다양성 선택** 5단계로 구성됩니다.

---

## 1. 개요

- **목적**: 3D Gaussian 씬과 (선택) COLMAP 카메라 정보를 바탕으로, 편집에 적합한 **소수의 뷰(카메라)** 를 자동으로 선택한다.
- **입력**: `gaussians`, `cam_centers`, `cam_forwards`, `fovy`, `h`, `w`, (선택) `roi_mask`, `seg_prompt`, `edit_prompt`, IP2P 파이프라인 등.
- **출력**: 선택된 카메라 리스트 `final_cameras` (ROI 중심 기준 azimuth로 정렬).

---

## 2. (선택) Step 0: ROI 마스크 생성

- **함수**: `_lens_compute_roi_mask_from_segmentation`
- **조건**: `seg_prompt`가 주어지고 `roi_mask`가 없을 때만 수행.

**동작**:
- COLMAP 카메라 중 최대 `n_views`개를 골라 각 뷰에서 렌더.
- **LangSAM** 등 세그멘터로 `seg_prompt`에 맞는 2D 마스크 추출.
- 각 가우시안을 여러 뷰에 투영해 마스크에 걸린 비율을 누적하고, `threshold` 이상인 가우시안만 ROI로 사용.
- 결과는 3D 불리언 마스크(어떤 가우시안이 ROI인지)로, 이후 Step 1·2·4에서 ROI만 사용하는 데 쓰인다.

---

## 3. Step 1: ROI 내재 분석 (ROI Intrinsic Analysis)

- **함수**: `_lens_roi_intrinsic_analysis(gaussians, roi_mask, cam_forwards)`

**목적**: ROI 가우시안들의 기하적 특성(중심, 주축, 크기)과 “앞쪽” 방향을 구한다.

**연산**:
1. **ROI 제한**: `roi_mask`가 있으면 해당 가우시안의 `xyz`, `opacity`만 사용.
2. **가중 중심**: opacity를 가중치로 한 가우시안 중심  
   \(\mathbf{c} = \sum w_i \mathbf{x}_i / \sum w_i\).
3. **가중 공분산**: \(\mathbf{C} = \sum w_i (\mathbf{x}_i - \mathbf{c})(\mathbf{x}_i - \mathbf{c})^\top / \sum w_i\).
4. **고유값/고유벡터**: \(\mathbf{C}\)를 대각화해 고유값 \(\lambda_1 \ge \lambda_2 \ge \lambda_3\)와 주축 \(\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3\) 계산.
5. **객체 크기**: `object_size = 2 * sqrt(median(|λ|))` (중앙값 기반 스케일).
6. **앞 방향 `v_front`**: COLMAP 카메라들의 평균 전방 벡터를 \(\mathbf{v}_1\)에 투영한 뒤, “객체를 바라보는” 방향으로 정규화. (수치 불안정 시 \(\mathbf{v}_3\) 사용.)

**보정**:
- COLMAP 중심까지의 거리 중앙값 `colmap_median_dist`를 구하고, `object_size > colmap_median_dist`이면 `object_size`를 `colmap_median_dist`로 상한.

**반환**: `center`, `v1,v2,v3`, `eigenvalues`, `object_size`, `v_front` 등. 이후 단계에서 거리 스케일과 “정면/측면” 점수 계산에 사용.

---

## 4. Step 2: SAGE-Probing (Sharpness-Aware Scale Probing)

- **함수**: `_lens_scale_probing(...)`

**목적**: ROI 정면 방향(`v_front`)에서 **카메라-객체 거리**를 여러 배수로 시험해, “편집 용이성” 점수가 가장 좋은 **최적 거리 \(d^*\)** 를 정한다.

**동작**:
1. **기준 반경**: \(r_{\text{obj}} = \sqrt{\lambda_1}\) (가장 큰 고유값).
2. **후보 거리**: `distance_multipliers`(예: [1.5, 2.0, 2.5, 3.0, 3.5])에 대해 \(d = \text{mult} \times r_{\text{obj}}\).
3. 각 \(d\)에 대해:
   - 카메라 위치: \(\text{eye} = \text{center} + d \cdot \mathbf{v}_{\text{front}}\).
   - 해당 뷰로 렌더 후, ROI가 보이는 영역의 2D 마스크 계산(`_lens_project_roi_mask`).
   - **편집 용이성 점수**:
     - **IP2P 사용 시**: `_lens_run_ip2p_unified`로 attention 맵 추출 후 SAGE 점수 계산.
     - **미사용 시**: `_lens_compute_editability_score`(heuristic: occupancy 기반 focus·size_penalty).
4. **SAGE 점수** (`_lens_compute_sage_score`):
   - **focus**: ROI 마스크 안의 (정규화된) attention 비율.
   - **leakage**: ROI 밖(배경)으로 빠지는 attention 비율.
   - **entropy**: attention 맵의 정규화 엔트로피. `entropy > entropy_thresh`이면 \(-\infty\) (불안정 뷰 제외).
   - **occupancy**: ROI가 화면에서 차지하는 비율; `occupancy_lo`~`occupancy_hi` 밖이면 `size_penalty` 부여.
   - **총점**:  
     \(\text{S}_{\text{total}} = \text{focus} - \lambda_{\text{leak}} \cdot \text{leakage} - \lambda_{\text{ent}} \cdot \text{entropy} - \text{size\_penalty}\).
5. 점수가 최대인 \(d\)를 \(d^*\)로 선택. 모두 \(-\infty\)면 `distance_multipliers` 중앙값을 fallback으로 사용.

이 \(d^*\)가 Step 3에서 후보 카메라를 뿌릴 **반경(거리)** 가 된다.

---

## 5. Step 3: Fibonacci Manifold Sampling

- **함수**: `_lens_fibonacci_camera_candidates(...)`

**목적**: ROI 중심을 바라보는 **방향**을 구면(또는 반구) 위에 균등하게 뿌리고, 그 방향들에 대해 거리 `distance`(= \(d^*\))로 카메라 후보를 만든다.

**동작**:
1. **Fibonacci 구면 샘플** (`_lens_fibonacci_sphere_samples(n_candidates)`):  
   golden angle \(\pi(3-\sqrt{5})\) 기반으로 구면 위에 \(n\)개 방향 생성.
2. **필터** (선택):
   - `hemisphere_only=True`: world_up과 반대 쪽 반구 제거.
   - `colmap_cam_centers`가 있으면: COLMAP 카메라들의 평균 방향(중심에서 바깥으로)과의 각이 `cone_half_angle_deg` 이내인 방향만 유지 (cone filter).
3. 각 방향 \(\mathbf{d}\)에 대해 \(\text{eye} = \text{center} + \text{distance} \cdot \mathbf{d}\), target=center로 **Simple_Camera** 생성.

**결과**: 수십~수백 개의 후보 카메라 리스트. Step 4에서 한 번에 스코어링된다.

---

## 6. Step 4: Energy-based Scoring

- **함수**: `_lens_score_candidates(...)`

**목적**: 각 후보 카메라에 대해 “가시성”과 “정면/측면성”을 반영한 **에너지(스코어)** 를 부여하고, 에너지 순으로 정렬한다.

**연산** (카메라별):
1. 해당 뷰로 렌더.
2. **가시성 \(S_{\text{vis}}\)**: ROI 2D 마스크가 화면에서 차지하는 비율. (`roi_mask` 없으면 1.0.)
3. **정면/측면 점수 \(S_{\text{can}}\)**:
   - 뷰 방향 \(\mathbf{v}_{\text{view}} = \text{normalize}(\text{center} - \text{cam\_center})\).
   - \(\cos_{\text{front}} = |\mathbf{v}_{\text{view}} \cdot \mathbf{v}_{\text{front}}|\),  
     \(\cos_{\text{side}} = |\mathbf{v}_{\text{view}} \cdot \mathbf{v}_2|\).
   - \(S_{\text{can}} = \max(\cos_{\text{front}}, 0.8 \cdot \cos_{\text{side}})\).
4. **에너지**:  
   \(E = w_{\text{vis}} \cdot S_{\text{vis}} + w_{\text{can}} \cdot S_{\text{can}}\).

**출력**: `(candidate_index, energy, S_vis, S_can)` 리스트를 **에너지 내림차순**으로 정렬한 결과. Step 5는 이 리스트의 상위만 풀(pool)로 쓴다.

---

## 7. Step 5: Diversity-aware Selection

- **함수**: `_lens_diversity_selection(cameras, scored, center, n_select, top_fraction)`

**목적**: 에너지 상위 풀에서 **서로 다른 방향**을 커버하도록 **\(n_{\text{select}}\)개**를 골라 최종 카메라 집합을 만든다.

**동작**:
1. **풀 크기**: `n_pool = max(ceil(len(scored) * top_fraction), n_select)`. 상위 `n_pool`개 후보 인덱스와 에너지를 풀에 넣는다.
2. **FPS 유사 선택**:
   - 첫 번째로 **에너지 1위** 카메라를 선택.
   - 이후 반복: 아직 선택되지 않은 풀 내 카메라 중, **이미 선택된 카메라들과의 최소 각도**가 가장 큰 후보를  
     \(\text{score} = \min_{\text{selected}} \angle(\text{cand}, \text{sel}) \times \text{energy}(\text{cand})\)  
     로 곱해 그 값이 최대인 것을 선택.
3. \(n_{\text{select}}\)개가 찰 때까지 반복 (또는 풀 소진).

**마무리**: 선택된 카메라들을 ROI `center` 기준 **azimuth**(\(\operatorname{atan2}(z, x)\)) 순으로 정렬해 `final_cameras`로 반환.

---

## 8. 파이프라인 요약 표

| 단계 | 이름 | 입력 | 출력 | 비고 |
|------|------|------|------|------|
| 0 | ROI 마스크 (선택) | seg_prompt, gaussians, cams | roi_mask (3D) | LangSAM 등으로 ROI 가우시안 지정 |
| 1 | ROI 내재 분석 | gaussians, roi_mask, cam_forwards | center, v1,v2,v3, eigenvalues, object_size, v_front | 공분산·고유값·앞방향 |
| 2 | SAGE-Probing | roi_info, distance_multipliers, ip2p 등 | 최적 거리 \(d^*\) | focus/leakage/entropy로 거리 결정 |
| 3 | Fibonacci 샘플링 | center, \(d^*\), n_candidates, cone 등 | 후보 카메라 리스트 | 구면/반구/cone 필터 |
| 4 | 에너지 스코어링 | 후보 카메라, w_vis, w_can | (index, energy, S_vis, S_can) 정렬 | 가시성 + 정면/측면 |
| 5 | 다양성 선택 | scored, n_select, top_fraction | 최종 카메라 \(n_{\text{select}}\)개 | FPS 유사 + azimuth 정렬 |

---

## 9. 주요 파라미터 (함수 시그니처 기준)

| 파라미터 | 의미 | 예시 |
|----------|------|------|
| `n_candidates` | Fibonacci 후보 개수 | 150 |
| `n_select` | 최종 선택 카메라 수 | 20 |
| `hemisphere_only` | 반구만 사용 여부 | False |
| `cone_half_angle_deg` | COLMAP 평균 방향 기준 cone 각도 | 90 |
| `w_vis`, `w_can` | 가시성 / 정면·측면 가중치 | 0.6, 0.4 |
| `top_fraction` | 에너지 상위 비율로 풀 크기 결정 | 0.20 |
| `lambda_leak`, `lambda_ent` | SAGE leakage·entropy 계수 | 1.5, 2.0 |
| `entropy_thresh` | 이보다 크면 SAGE \(-\infty\) | 0.97 |

---

## 10. 참고 코드 위치

- **파이프라인 진입점**: `threestudio/data/gs_load.py` — `_lens_run_generate_by_lens_pipeline` (약 828–960행).
- **Step 0**: `_lens_compute_roi_mask_from_segmentation` (약 764행~).
- **Step 1**: `_lens_roi_intrinsic_analysis` (약 341행~).
- **Step 2**: `_lens_scale_probing` (약 613행~), SAGE 점수 `_lens_compute_sage_score` (약 408행~).
- **Step 3**: `_lens_fibonacci_camera_candidates` (약 680행~), `_lens_fibonacci_sphere_samples` (약 668행~).
- **Step 4**: `_lens_score_candidates` (약 704행~).
- **Step 5**: `_lens_diversity_selection` (약 729행~).
