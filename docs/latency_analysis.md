# 레이턴시 요약 분석 (왜 오래 걸리는가)

실제 측정값 예시 (Total 117.9s) 기준으로, 오래 걸리는 구간의 **원인**을 코드와 함께 정리한 문서입니다.

---

## 1. edit_multiview (94.8s, 80.5%)

전체 시간의 대부분을 차지합니다. **멀티뷰 일관성을 위해 여러 뷰에 대해 디퓨전 디노이징을 반복**하기 때문입니다.

### 1.1 guidance_batch (47.7s) — 그 중 대부분이 디노이징

| 구간 | 측정값 | 원인 |
|------|--------|------|
| **target_denoise_loop** | **34.8s (29.5%)** | **가장 큰 병목** |
| key_view_denoise_loop | 10.3s (8.7%) | 두 번째로 큼 |
| render_consistent_maps | 1.1s | |
| inverse_render_2d_to_3d | 0.18s | |
| 기타 (setup, processor 교체 등) | ~0.05s | |

---

#### target_denoise_loop (34.8s) — 왜 오래 걸리나

- **역할**: 모든 뷰(키 뷰 + 나머지)를 **디퓨전 스케줄러 타임스텝만큼** 반복 디노이징.
- **코드**: `dge_guidance.py` 648행 근처  
  `for t_step in self.scheduler.timesteps:` (기본 `diffusion_steps=20`회)
  - 매 타임스텝마다:
    1. **Pivotal forward 1회**: `pivotal_idx`로 고른 뷰 1개에 대해 UNet 1회 (CFG 3배치).
    2. **뷰 배치 루프**: `for b in range(0, n_views, camera_batch_size)`  
       각 배치마다 **UNet 1회** + epipolar 제약 구성.
- **비용**:  
  `(타임스텝 수) × (1 + ceil(n_views / camera_batch_size))` 회의 **UNet forward**.  
  예: 20 step, 뷰 25개, batch 4 → 20 × (1 + 7) = **160회 UNet** → 34.8s의 주된 원인.

**요약**: 스텝 수 × (pivotal 1회 + 전체 뷰를 배치로 나눈 횟수)만큼 UNet을 돌리기 때문에 시간이 길어짐.

---

#### key_view_denoise_loop (10.3s) — 왜 오래 걸리나

- **역할**: **키 뷰만** 먼저 디노이징해서 cross-attention을 수집 (multiview 일관성용).
- **코드**: `dge_guidance.py` 528행 근처  
  `for t_step in self.scheduler.timesteps:` (동일하게 ~20회)
  - 매 타임스텝: 키 뷰에 대해 **UNet 1회** (배치 크기 = `n_key * 3` for CFG).
- **비용**: **20회 UNet** (키 뷰 수만큼의 배치). 키 뷰가 3~4개면 배치가 작아서 step당 비용은 target보다 작지만, 20번 반복이 쌓여서 10.3s.

**요약**: 디퓨전 스텝 수(20)만큼 UNet을 돌리기 때문에, 키 뷰만 해도 10초대가 나옴.

---

#### render_consistent_maps (1.1s)

- **역할**: 2D→3D로 만든 attention을 다시 **모든 뷰 × 해상도 × attention 채널**에 대해 3DGS로 렌더링.
- **코드**: `dge_guidance.py` 614행 근처  
  `for view_idx in range(n_views):` → `for res in M_3d_by_res:` → `for c in range(attn_len):` → `gs_render(...)`.
- **비용**: `n_views × num_resolutions × attn_len` 회의 **Gaussian splatting 렌더**.  
  뷰 수·해상도·토큰 수에 비례해 1초 전후로 나옴.

---

### 1.2 edit_multiview 내 나머지 (47s 중 46s는 기타)

- **build_save_list (0.31s)**, **save_image_grid (0.24s)**: 편집 결과를 리스트로 만들고 그리드 이미지 저장 (I/O).
- **render_and_load_originals (0.20s)**: 편집 전 원본 이미지 렌더/로드.
- **collect_cameras_and_sort**, **assign_outputs**, **concat_batch**: 상대적으로 매우 짧음.

**edit_multiview가 94.8s인 이유**:  
- 대략 **47.7s**가 guidance_batch(대부분 디노이징)이고,  
- 나머지 **47s**는 **동일 run 내 다른 작업**(렌더, 마스크 업데이트, 캐시 처리 등)이 같은 `edit_multiview` 블록 안에서 한 번에 실행되거나, **한 번의 edit_multiview 호출이 여러 번 쌓인 누적**으로 해석할 수 있음.  
- 즉, **실제 병목은 “스텝 수 × (키 뷰 디노이징 + 전체 뷰 디노이징)”** 이고, 그 합이 약 45초대이며, 나머지 시간은 주변 처리와 I/O.

---

## 2. update_mask at step 600 (7.0s) / update_mask (5.5s)

- **역할**: 특정 step(예: 600)에서 **target_prompt**로 마스크를 갱신하거나, 초기 **seg_prompt**로 마스크를 만듦.
- **코드**: `DGE.py` 161행 `update_mask()`
  - **Pass 1**: 거리 기준 상위 **30개 뷰**에 대해  
    - `self(cur_batch)` → **3DGS 렌더 1회**  
    - `text_segmentor(image, seg_object)` → **세그멘테이션 1회** (LangSAM 등, 비용 큼)  
  - **Pass 2**: 통과한 뷰들에 대해 `apply_weights` (렌더 + 마스크 백프로젝션), 파일 저장.
- **비용**: **30회 렌더 + 30회 세그멘테이션** + Pass2 렌더/저장.  
  세그멘테이션 모델이 무거워서 5~7초대가 나옴.

**요약**: 뷰 30개 × (렌더 + 세그멘터) 반복이 주된 원인.

---

## 3. render_forward (5.2s, 4.4%)

- **역할**: 매 **training step**마다 배치에 있는 카메라만큼 **3D Gaussian 렌더링**.
- **코드**: `DGE.py` 649행 `forward()`  
  `for id, cam in enumerate(batch["camera"]):`  
  - 카메라당: `render(cam, gaussian, pipe, ...)` 1회 (RGB)  
  - 같은 카메라로 `override_color=mask` 로 **한 번 더 렌더** (semantic/mask용).
- **비용**: **배치 크기 × 2** 회의 3DGS 렌더.  
  배치가 4면 8회, 8이면 16회. 매 step마다 호출되므로 전체 구간 합이 5.2s.

**요약**: step마다 (카메라 수 × 2)번 렌더하기 때문에, 누적 시간이 5초대.

---

## 4. camera_generation (4.5s) — 보통 초기 1회

- **역할**: 학습 시작 시 **Lens**로 편집용 카메라를 생성 (SAGE probing, ROI, Fibonacci 등).
- **코드**: `gs_load.py` 2090행 근처 `camera_generation.lens`  
  내부 `_timeit("camera_generation.lens.<section>")` 로 구간별 측정.

| 구간 | 측정값 | 원인 |
|------|--------|------|
| **sage_probing** | **2.6s** | 여러 거리 배수에 대해 **IP2P 편집 + 렌더 + entropy 계산** 반복. 가장 무거움. |
| **roi_mask_segmentation** | **1.6s** | 여러 뷰에서 **LangSAM 세그멘테이션** + 3D 백프로젝션. |
| energy_scoring | 0.1s | 후보 카메라별 에너지 점수 (렌더 위주). |
| fibonacci / diversity_selection / roi_analysis | 0.03s 이하 | 상대적으로 가벼움. |

**요약**: SAGE probing(IP2P+렌더 반복)과 ROI 세그멘테이션이 대부분의 4.5초를 차지.

---

## 5. 요약 표

| 구간 | 비중 | 주된 원인 (한 줄) |
|------|------|-------------------|
| **edit_multiview** | 80.5% | 스텝 수(20) × (키 뷰 디노이징 + 전체 뷰 배치 디노이징) → UNet 수백 회 |
| **target_denoise_loop** | 29.5% | 20 step × (1 + ceil(n_views/batch)) 회 UNet |
| **key_view_denoise_loop** | 8.7% | 20 step × 1 회 UNet (키 뷰만) |
| **update_mask** | ~5% | 30 뷰 × (렌더 + 세그멘터) |
| **render_forward** | 4.4% | 매 step (배치 크기 × 2) 회 3DGS 렌더 |
| **camera_generation** | 3.8% | SAGE probing(IP2P 반복) + ROI 세그멘테이션 (초기 1회) |

---

## 6. 줄이고 싶을 때 참고 (튜닝 방향)

- **edit_multiview / target_denoise_loop**:  
  - `diffusion_steps` 감소 (예: 20 → 10),  
  - `camera_batch_size` 증가 (메모리 허용 범위에서),  
  - `multiview_num_key_views` 감소.
- **key_view_denoise_loop**:  
  - `diffusion_steps` 감소가 그대로 반영됨.
- **update_mask**:  
  - 마스크용 뷰 수(30) 감소, 또는 세그멘터 경량화/캐시 활용.
- **render_forward**:  
  - 배치 크기 조정, 렌더 해상도 조정.
- **camera_generation**:  
  - SAGE 거리 배수/후보 수 감소, ROI 세그멘트 뷰 수 감소.

---

## 7. edit_all_view vs edit_multiview — 총 시간 차이 (53s vs 91s)

두 경로를 같은 설정으로 돌렸을 때 **Total 53s vs 91s**처럼 차이가 크게 나는 이유를 정리합니다.

### 7.1 guidance_batch는 비슷함

- **edit_all_view**: `edit_all_view.guidance_batch` **34.5s**. 그중 `edit_latents` → `diffusion_loop` **33s**, 그 33s 중 **22s가 compute_epipolar_constrains**, 나머지 pivotal_forward(4.65s) + unet_forward(4.2s) + pivotal_setup(1.9s) 등.
- **edit_multiview**: `edit_multiview.guidance_batch` **34.75s**. epipolar 없음. key_view_denoise(10.2s) + target_denoise(20.8s) + render_consistent_maps(2.5s) + inverse_render(0.37s) 등으로 비슷한 ~35s.

즉 **편집 연산(guidance_batch) 한 번 비용은 둘 다 비슷** (~35s). edit_all_view는 그중 22s를 epipolar에 씀.

### 7.2 총 시간 차이는 "edit_* 블록 전체"에서 남

- **edit_all_view 블록**: 35.6s. guidance_batch 34.5s + render_single 0.34s + load_original 0.16s + 기타.
- **edit_multiview 블록**: **70s**. guidance_batch 34.75s + build_save_list 0.75s + save_image_grid 0.24s + render_and_load_originals 0.23s + …  
  숫자만 보면 **70 − 34.75 ≈ 35s**가 guidance_batch **밖**에서 나옴.

**차이(38s ≈ 91−53)**의 대부분은 **edit_multiview 블록이 edit_all_view 블록보다 34s 정도 더 길기 때문**.

### 7.3 그 35s는 어디서 나오는가

- **edit_multiview**에서는 `render_and_load_originals` 안에서 **뷰마다** `self(cur_batch)`(3DGS 렌더 1회) + 이미지 로드. **뷰 수가 많으면** (예: 25 view × ~1.4s ≈ **35s**) 이 루프만으로 30초대 가능.
- 요약에 `render_and_load_originals: 0.23s`처럼 작게 나오는 경우는 뷰가 매우 적은 run이거나, 다른 실험(다른 뷰 수)과 비교했을 수 있음.
- **edit_all_view**도 `collect_cameras` 안에 "뷰마다 렌더 + 로드" 루프가 있지만, 뷰 수가 적으면 1~2초대로 끝나 블록 전체가 35.6s.

**정리**: guidance_batch는 둘 다 ~35s로 비슷. 총 시간 차이는 edit_multiview 블록이 70s로 35s 더 길어서 발생하며, 그 34s는 주로 **뷰마다 3DGS 렌더 + 로드** 구간(뷰 수에 비례)에서 나감. 같은 뷰 수로 맞추면 "편집 연산" 비용은 비슷하고, edit_multiview가 더 오래 걸리는 이유는 **블록 안의 렌더/로드 루프(뷰 수 비례)** 때문.

이 문서는 `script/verify_edit_multiview.py` 및 레이턴시 로그와 함께 사용하면, “어디가 왜 오래 걸리는지” 빠르게 짚을 수 있습니다.
