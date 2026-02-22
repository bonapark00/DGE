# edit_latents_multiview: Key view vs Target view에서 일어나는 일

`edit_latents_multiview` 안에서 **key view**와 **target view**가 각각 어떻게 쓰이고, 어떤 연산이 이루어지는지 코드 기준으로 정리한 문서입니다.

---

## 1. 전체 흐름 (한 줄)

- **Key view**: 먼저 **키 뷰만** 디노이징하면서 cross-attention을 **수집** → 2D attention을 3D로 올렸다가 다시 **모든 뷰용 2D map**으로 렌더.
- **Target view**: 그 **consistent 2D map**을 cross-attention 대신 넣어서 **전체 뷰(키+나머지)** 를 한꺼번에 디노이징. 매 스텝 끝에 키 뷰 위치만 `key_edited`로 덮어씀.

---

## 2. Key view에서 일어나는 일

Key view는 **고정된 N개 뷰** (예: `key_indices = [0, 5, 10, 15]`, 4개). 이 뷰들만 먼저 처리해서 “어디를 편집할지”에 해당하는 **cross-attention map**을 만듭니다.

### 2.1 사전 준비

- **valid_token_indices**: 프롬프트(`prompt_text`)에서 실제로 쓰는 토큰 인덱스만 뽑음 → `attn_len`개.
- **CrossAttentionStoreProcessor 설치**: UNet의 모든 `attn2`에 `storing_processor`를 달아서, **특정 spatial resolution**(32×32, 64×64 등)에서 cross-attention 출력을 **저장**하도록 함.

### 2.2 key_view_denoise_loop (528~559행 근처)

- **입력**: `latents[key_indices]`만 사용. 노이즈 붙여서 `latents_key`로 둠.
- **루프**: `scheduler.timesteps`(기본 20스텝) 동안
  - **매 스텝**:
    - `register_pivotal(self.unet, True)` → 이번 forward는 “pivot”으로 간주.
    - 키 뷰만 넣어서 **UNet 1회** (배치 = key 4개 × CFG 3 = 12).
    - UNet 내부에서 cross-attention이 돌 때 **CrossAttentionStoreProcessor**가 해당 해상도에서 **attention map을 저장** (prompt 토큰별로 “어디가 활성인지”).
    - CFG로 noise 예측 합친 뒤 `scheduler.step` → `latents_key` 업데이트.
- **결과**:
  - **`key_edited`**: 키 뷰만 끝까지 디노이징된 latents (나중에 전체 latents의 키 위치에 계속 복사됨).
  - **`storing_processor.maps`**: 스텝별·해상도별로 쌓인 **키 뷰 cross-attention map** (2D, H×W×attn_len 형태에 가까움).

### 2.3 build_key_cross_attn_by_res (565~584행)

- `storing_processor.maps`를 해상도별로 묶어서 **키 뷰 평균 cross-attention**을 만듦.
- 스텝·head 차원은 mean으로 줄이고, **positive branch만** 사용 (key 뷰 4개분만 슬라이스).
- 결과: **`key_cross_attn_by_res[res]`** = 해상도 `res`당 (n_key, H*W, attn_len) 형태.

### 2.4 inverse_render_2d_to_3d (611~616행)

- **목적**: 키 뷰 2D attention을 **3D Gaussian 공간**으로 올림.
- **방법**:
  - 해상도별로 키 뷰 카메라에 맞춰 **저해상도 카메라** 만듦 (`_camera_at_res`).
  - 각 키 뷰의 attention map `M_v` (H×W×attn_len)를 **채널(토큰)마다** 2D 이미지로 보고,  
    `gaussian.apply_weights(cam_low, weights[:, c], weights_cnt, img_w)` 로 **3D 가우시안당 가중치**를 누적.
  - `weights / (weights_cnt + eps)` 로 정규화 → **`M_3d_by_res[res]`** = (N_gaussian, attn_len).  
    즉 “각 가우시안이 각 편집 토큰에 얼마나 기여하는지” 3D 필드.

### 2.5 render_consistent_maps (618~634행)

- **목적**: 방금 만든 3D attention을 **모든 뷰**에 대해 다시 2D로 렌더링해서, **뷰마다 같은 3D 기준의 2D map**을 만듦.
- **방법**:
  - **모든 뷰** `view_idx in range(n_views)` 에 대해:
    - 해당 뷰 카메라로 `_camera_at_res(cam, side, side)`.
    - `M_3d_by_res[res]`의 각 채널(attn_len개)을 **가우시안 색상**으로 넣어서 `gs_render(..., override_color=color_c)` 로 렌더.
  - 결과: **`M_con_by_view_res[view_idx][res]`** = (H, W, attn_len).  
    뷰마다 **같은 3D 필드를 그 뷰에서 본 2D map**이라서, “consistent” map.

정리하면, **key view** 단계에서는:

1. **키 뷰만** 디노이징하면서 cross-attention을 **저장**하고  
2. 그 2D map을 **3D로 역투영**한 뒤  
3. **모든 뷰**에 대해 3D를 다시 2D로 렌더해 **consistent map** `M_con_by_view_res`를 만듦.

---

## 3. Target view에서 일어나는 일

“Target view”는 **키 뷰 + 나머지 뷰 전부**를 한꺼번에 디노이징하는 단계를 말합니다.  
이때 **일반 cross-attention 대신**, 위에서 만든 **consistent map**을 넣어서 “어디를 편집할지”를 모든 뷰가 공유하게 합니다.

### 3.1 사전 준비 (target 쪽)

- **ConsistentCrossAttnProcessor 설치**: 모든 `attn2`를 `ConsistentCrossAttnProcessor`로 바꿈.  
  이 processor는 **`_consistent_attn_map_current`** 에 들어 있는 map을 사용해, 기존 encoder hidden states 대신 **consistent map**을 attention value로 씀.
- **noise_and_init_latents**: 전체 latents에 노이즈 붙이고, **키 뷰 위치만** 이미 구해 둔 `key_edited`로 덮어씀.  
  → 이후 디노이징에서는 “키 뷰는 이미 완성본”으로 고정하고, 나머지만 업데이트하는 셈.

### 3.2 target_denoise_loop (662~729행)

- **입력**: `latents` (전체 뷰, 단 key 위치는 `key_edited`), `M_con_by_view_res` (뷰별·해상도별 consistent map).
- **루프**: 같은 `scheduler.timesteps`(20스텝) 동안, **전체 뷰**를 배치 단위로 디노이징.

  **타임스텝마다:**

  1. **Pivotal (현재 코드에선 주석 처리됨)**  
     - 원래는: 배치당 대표 뷰(pivotal_idx)에 대해 UNet 1회 돌려서 `pivot_hidden_states` / `kf_attn_output` 를 채움.  
     - 그 다음 배치 forward에서 DGEBlock이 “가장 가까운 pivot” hidden과의 similarity로 어디를 gather할지 정함.  
     - 지금은 674행 근처 `forward_unet`(pivotal)이 주석이라, 이 단계는 생략된 상태일 수 있음.

  2. **배치 루프** `for b in range(0, n_views, camera_batch_size)`  
     - **batch_prep**  
       - 이번 배치에 **키가 아닌 뷰가 하나라도 있으면** (`is_target_batch`):  
         `M_con_by_view_res[i][res]`를 모아서 **`data[res]`** 에 넣고,  
         모든 `attn2`에 **`_consistent_attn_map_current = data`** 로 넣어줌.  
         → 이 배치에서 UNet이 돌 때 cross-attention은 **텍스트 대신 이 2D map**을 사용.
       - 키만 있는 배치면 consistent map 없이 일반 cross-attention (backup processor).
       - `register_batch_idx`, `register_cams`, `register_epipolar_constrains(self.unet, {})` (epipolar는 비활성).
     - **batch_forward**  
       - `latents[batch_slice]`에 대해 CFG 3배치로 **UNet 1회**.  
       - 내부에서 cross-attention 시 **ConsistentCrossAttnProcessor**가 `_consistent_attn_map_current`를 쓰므로,  
         **target view들은 “키 뷰에서 뽑은 3D→2D consistent map”으로 편집 위치가 맞춰짐**.

  3. **merge_and_step**  
     - 배치별 noise 예측을 concat → CFG 결합 → `scheduler.step`으로 **전체 latents** 한 번 업데이트.  
     - 그 다음 **`latents[key_indices] = key_edited`** 로 키 뷰 위치만 다시 덮어씀.  
       → 키 뷰는 항상 “이미 완성된 key_edited”로 유지.

### 3.3 Target view에서 “일어나는 일” 요약

- **키 뷰 인덱스** (`key_indices`):  
  latents 상에서는 **업데이트하지 않고** 매 스텝 끝에 `key_edited`로 덮어쓰기만 함.  
  즉 **연산은 key_view_denoise_loop에서 이미 끝났고**, target_denoise_loop에서는 “결과만 유지”.
- **나머지 뷰 (진짜 target)**  
  - 매 스텝, 배치마다 **consistent map** `M_con_by_view_res[view_idx][res]`를 `_consistent_attn_map_current`로 넣어서 UNet을 돌림.  
  - 그래서 **“어디를 편집할지”가 키 뷰에서 나온 3D→2D map과 일치**하고, 뷰 간 일관성이 생김.  
  - Epipolar는 쓰지 않음 (빈 dict 등록).

---

## 4. 표로 보는 차이

| 구분 | Key view | Target view |
|------|----------|-------------|
| **어떤 뷰** | `key_indices`만 (예: 4개) | 전체 뷰 (키 + 나머지, 예: 20개) |
| **디노이징 시점** | key_view_denoise_loop (먼저 한 번만) | target_denoise_loop (그 다음, 매 스텝) |
| **Cross-attention** | 일반 텍스트 cross-attention + **저장**(StoreProcessor) | **Consistent map** (3D→2D) 사용 (ConsistentCrossAttnProcessor) |
| **2D→3D→2D** | 키 뷰 attention을 3D로 올리고, 모든 뷰로 2D 재렌더 → consistent map 생성 | 그 map을 **소비**만 함 (수정 안 함) |
| **latents 갱신** | `latents_key`만 업데이트 → `key_edited` | 전체 `latents` 업데이트 후, 키 위치만 `key_edited`로 덮어씀 |
| **Pivotal** | 키 뷰 forward가 곧 pivot 역할 (hidden 저장용) | 원래는 배치당 pivotal 1회로 pivot hidden 채움 (현재는 해당 호출 주석 가능성 있음) |

---

## 5. 한 줄로 정리

- **Key view**: 키 뷰만 디노이징 + cross-attention **수집** → 2D→3D→2D로 **consistent map** 제작.
- **Target view**: 그 **consistent map**을 cross-attention 자리에도 넣어서 **전체 뷰** 디노이징하고, 키 위치만 매 스텝 `key_edited`로 고정.

이렇게 해서 “편집할 영역”이 키 뷰와 모든 target 뷰에서 **같은 3D 기준**으로 맞춰지게 됩니다.
