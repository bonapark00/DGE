# edit_latents_multiview 최적화 제안

불필요한 연산·중복을 줄여 속도를 높일 수 있는 후보를 정리했습니다. 적용 시 레이턴시/품질 측정을 권장합니다.

---

## 레이턴시 요약에서 "edit_multiview 합이 안 맞는" 이유 (수정됨)

**원인**: guidance `__call__` 안에서 `timeit("edit_multiview")`로 edit_latents_multiview만 감쌌는데, DGE에서 전체 `edit_multiview()`를 감싼 블록도 같은 이름 `"edit_multiview"`를 씀.  
레이턴시 로거는 **같은 이름이면 시간을 누적**하므로, 상위 1회(예: 56s) + 내부 1회(예: 55s)가 둘 다 `"edit_multiview"`에 더해져 **총 109s**처럼 나왔고, 트리에는 **직계 자식**만 보여주기 때문에 `edit_multiview.xxx`(56s 분량)만 보이고, 내부 55s는 같은 키에만 더해져서 **별도 항목으로는 안 보임** → 합이 맞지 않음.

**수정**: guidance 쪽 timeit 이름을 `"edit_multiview.guidance_batch.edit_latents_multiview"`로 변경.  
이제 상위 `edit_multiview`에는 DGE의 전체 호출 시간만 들어가고, 55s 구간은 `edit_multiview.guidance_batch` 아래 자식으로 표시되어 **직계 자식 합 = edit_multiview**로 맞음.

---

## 0. 레이턴시 요약에서 “forward_unet 17s / per_step_setup 1.15s”가 나오는 이유

- **forward_unet (17s, ~11%)**  
  key_view_denoise_loop에서 **스텝마다 UNet을 1회씩** 돌립니다 (스텝 수 ≈ 20).  
  한 번이 약 0.85s 수준이면 20 × 0.85 ≈ 17s가 됩니다.  
  key 뷰만 넣어도 UNet + DGE 블록 + CrossAttentionStoreProcessor(저장) 비용이 커서, **구조상 비용이 큰 구간**입니다.  
  줄이려면 스텝 수 감소(diffusion_steps), 또는 key 뷰 수 감소 등이 필요합니다.

- **per_step_setup (1.15s, ~0.7%)**  
  매 스텝마다 다음을 반복합니다:  
  - `t_step < 100`이면 `use_normal_unet()` → **register_normal_attention**(UNet 전체 순회 + 블록마다 closure 생성) + **register_normal_attn_flag**(전체 순회)  
  - 그렇지 않으면 `register_normal_attn_flag(self.unet, False)` (전체 순회)  
  - 그리고 **register_pivotal(self.unet, True)** (전체 순회)  

  즉 **스텝당 2~3번 UNet 전체 순회**가 20번 반복되어 1s대가 나옵니다.  
  **해결**: t가 100을 넘을 때만 normal ↔ extended 전환하도록 바꾸면, 전환은 최대 1~2번으로 줄어들어 per_step_setup이 크게 줄어듭니다.  
  → **적용 완료**: `key_view_denoise_loop`와 `target_denoise_loop` 모두에서 “경계 넘을 때만 전환”하도록 수정해 두었습니다.

---

## 1. pivotal_forward — 제거하면 안 됨 (품질 필수)

### 1.1 target_denoise_loop의 pivotal_forward

**현재**: 매 타임스텝마다 `pivotal_forward`에서 UNet 1회 호출하고, **노이즈 예측값만** 버림 (약 6.3s/스텝).

**중요**: 이 forward의 **출력(noise_pred)은 사용하지 않지만**, UNet 내부 DGE 블록의 **부수 효과가 품질에 필수**입니다.

- **`register_pivotal(self.unet, True)`** 후 forward 시, 각 DGE 블록에서:
  - `self.pivot_hidden_states = norm_hidden_states` (norm 후 hidden states 캐시)
  - `self.kf_attn_output = self.attn_output` (self-attn 출력 캐시)
- **`register_pivotal(False)`** 후 배치 forward 시:
  - `closest_cam_pivot_hidden_states = self.pivot_hidden_states[1][closest_cam]` 로 유사도 계산
  - `self.attn_output = self.kf_attn_output.view(...)[:, closest_cam]` 로 pivot에서 구한 attention을 사용
  - feature_injection 등에서도 `kf_attn_output` 사용

즉, **pivotal_forward를 제거하면 target 배치들이 pivot 캐시 없이 돌아가서 퀄리티가 크게 떨어집니다.**  
**이 호출은 제거하지 말 것.** (참고: `threestudio/utils/dge_utils.py` DGE 블록 내 `pivotal_pass` 분기)

---

## 2. 반복된 `unet.named_modules()` 제거 (효과 중간)

**현재**: `self.unet.named_modules()`를 여러 번 순회함.

- install_store_processor
- restore_attn2_processors
- install_consistent_processor (2회)
- **target_denoise_loop 내부**: 매 배치마다 `_consistent_attn_map_current` 설정 (695–696행)
- **target_denoise_loop 내부**: 매 타임스텝 끝 merge_and_step에서 None 설정 (717–719행)

**제안**: 함수 시작 시 attn2 모듈 리스트를 한 번만 만들고 재사용.

```python
# setup 직후 한 번만
_attn2_mods = [m for n, m in self.unet.named_modules() if n.endswith(".attn2") and hasattr(m, "processor")]

# 사용처 예:
for mod in _attn2_mods:
    mod.processor = storing_processor
# ...
for mod in _attn2_mods:
    setattr(mod, "_consistent_attn_map_current", data if ... else None)
```

- **효과**: UNet이 클수록 이득이 큼. 루프당 전체 모듈 순회 제거.

---

## 3. target 루프: consistent map 배치 데이터 선계산 (효과 중간)

**현재**: 매 타임스텝·매 배치마다 `M_con_by_view_res[i][res].reshape(-1, attn_len)`를 모아 `torch.stack(..., dim=0).to(device)` 수행.

**제안**: target_denoise_loop **진입 전**에 배치별로 한 번만 계산.

```python
# noise_and_init_latents 직후, target_denoise_loop 전에
precomputed_batch_data = []
for b in range(0, n_views, camera_batch_size):
    batch_indices = list(range(b, min(b + camera_batch_size, n_views)))
    is_target_batch = any(i not in key_indices_set for i in batch_indices)
    data = {"attn_len": attn_len}
    if is_target_batch:
        for res in collected_resolutions:
            maps_batch = [M_con_by_view_res[i][res].reshape(-1, attn_len) for i in batch_indices
                         if i < len(M_con_by_view_res) and res in M_con_by_view_res.get(i, {})]
            if maps_batch:
                data[res] = torch.stack(maps_batch, dim=0).to(device)
    precomputed_batch_data.append(data)

# 루프 안에서는
for bi, b in enumerate(range(0, n_views, camera_batch_size)):
    ...
    data = precomputed_batch_data[bi]
    if is_target_batch and data.get("attn_len") and any(k in data for k in collected_resolutions):
        for mod in _attn2_mods:
            setattr(mod, "_consistent_attn_map_current", data)
    ...
```

- **효과**: 타임스텝 × 배치 수만큼 반복되던 stack/to(device) 및 dict 구성 제거.

---

## 4. key_indices를 set으로 (효과 작음)

**현재**: `any(i not in key_indices for i in batch_indices)`에서 `key_indices`가 list라 매번 O(n_key) 검색.

**제안**:

```python
key_indices_set = set(key_indices)
# 이후
is_target_batch = any(i not in key_indices_set for i in batch_indices)
```

- **효과**: 배치·타임스텝마다 반복되므로 작지만 비용 없이 적용 가능.

---

## 5. text/cond 재쪼개기 제거 (효과 작음)

**현재**: 519–521행에서 `text_embeddings`, `image_cond_latents`를 chunk한 뒤, 655–656행 **noise_and_init_latents** 안에서 같은 텐서를 다시 chunk함.

```python
# 655-656: 이미 위에서 chunk한 것과 동일
positive_text_embedding, negative_text_embedding, _ = text_embeddings.chunk(3)
split_image_cond_latents, _, zero_image_cond_latents = image_cond_latents.chunk(3)
```

**제안**: 위에서 만든 `positive_text_embedding`, `negative_text_embedding`, `split_image_cond_latents`, `zero_image_cond_latents`를 그대로 재사용.  
noise_and_init_latents 블록에서는 이 변수들을 덮어쓰지 않고, 이미 존재하면 그대로 사용.

- **효과**: chunk 2회 제거. 메모리/연산 모두 미미하지만 중복 제거.

---

## 6. key_view_denoise_loop 내부 반복 연산 (효과 작음)

**현재**: 매 타임스텝마다 다음을 다시 계산함.

- `pivot_text` = positive/negative 텍스트 embedding concat (key_indices만)
- `pivot_image_cond` = image cond latents concat (key_indices만)

key 뷰 인덱스가 고정이므로 **타임스텝에 따라 변하는 것은 `latents_key`뿐**입니다.

**제안**: pivot_text, pivot_image_cond는 루프 **밖**에서 한 번만 계산.

```python
# 루프 전
pivot_text = torch.cat([
    positive_text_embedding[key_indices], negative_text_embedding[key_indices], negative_text_embedding[key_indices]
], dim=0)
pivot_image_cond = torch.cat([
    split_image_cond_latents[key_indices], split_image_cond_latents[key_indices], zero_image_cond_latents[key_indices]
], dim=0)

# 루프 안에서는
latent_model_input = torch.cat([latents_key] * 3)
latent_model_input = torch.cat([latent_model_input, pivot_image_cond], dim=1)
```

- **효과**: 매 스텝 concat 2회 제거. 작지만 코드 단순화.

---

## 7. use_normal_unet / register_normal_attn_flag (per_step_setup이 오래 걸리는 주된 원인)

**현재**: key_view_denoise_loop에서 매 타임스텝마다 `per_step_setup`으로:
- `if t_step < 100: self.use_normal_unet()` → **register_normal_attention**(전체 UNet 순회 + 블록마다 closure 생성) + **register_normal_attn_flag**(전체 순회)
- `else: register_normal_attn_flag(self.unet, False)` (전체 순회)
- **register_pivotal(self.unet, True)** (전체 순회)

즉 **매 스텝마다 2~3번 전체 UNet 순회**가 반복되고, t<100일 때는 `register_normal_attention`으로 블록 수만큼 새 closure를 만들어 붙입니다. 스텝이 20번이면 이게 20번 반복되어 **per_step_setup이 1s대**로 나올 수 있습니다.

**제안**: 스텝은 시간 순으로 감소하므로, **경계(100)를 넘을 때 한 번만** 전환.

```python
use_normal = True  # 루프 진입 시 (보통 첫 t가 100 이상이면 False로 시작)
for t_step in self.scheduler.timesteps:
    if t_step < 100:
        if not use_normal:
            self.use_normal_unet()
            use_normal = True
    else:
        if use_normal:
            register_normal_attn_flag(self.unet, False)
            use_normal = False
    # register_pivotal / pivot 텐서 구성 등
```

- **효과**: per_step_setup 구간 **대폭 단축** (1s대 → 수십 ms 수준 기대).

---

## 8. render_consistent_maps (효과는 크나 구현 난이도 있음)

**현재**: `n_views × num_resolutions × attn_len` 번의 **gs_render** 호출.

- 뷰마다, 해상도마다, 토큰 채널마다 3DGS 렌더 1회.

**가능한 방향** (API 지원 여부에 따라):

- **채널 배치**: 한 (view, res)에 대해 attn_len개 채널을 한 번에 렌더할 수 있다면, 호출 수를 `n_views × num_resolutions`로 줄일 수 있음.  
  현재 `override_color=color_c`가 채널당 한 벡터이므로, 렌더러가 다중 채널/다중 색상을 한 번에 받는 인터페이스가 있어야 함.
- **뷰 배치**: 같은 (res, c)에 대해 여러 뷰를 한 번에 렌더하는 것은 일반적으로 3DGS API와 맞지 않으므로, 현 구조에서는 어렵다면 제외.

**효과**: gs_render가 병목일 때만 의미 있음. 프로파일링으로 확인 후 검토.

---

## 9. inverse_render_2d_to_3d (효과는 중간·난이도 있음)

**현재**: `for c in range(attn_len): gaussian.apply_weights(cam_low, weights[:, c:c+1], weights_cnt, img_w)`  
→ attn_len번의 apply_weights 호출.

**가능한 방향**: `apply_weights`가 여러 채널을 한 번에 받을 수 있다면, 한 키 뷰당 1회로 줄일 수 있음.  
(CUDA/인터페이스가 채널 배치를 지원하는지 확인 필요.)

---

## 10. 기타

- **storing_processor.maps**: `build_key_cross_attn_by_res` 이후에는 더 이상 쓰이지 않으므로, `storing_processor.reset()` 또는 `storing_processor.maps.clear()`로 메모리 해제 가능.
- **current_H, current_W**: setup에서 설정되지만 edit_latents_multiview 내에서는 사용되지 않는 것 같으면 제거해도 됨 (다른 경로에서만 쓰이는지 확인 후).

---

## 적용 우선순위 제안

1. **1. pivotal_forward** – **제거하지 말 것.** 품질에 필수.
2. **2. attn2 모듈 캐시** – 구현 간단, 루프 비용 감소.
3. **3. precomputed_batch_data** – target 루프 내 반복 연산 제거.
4. **4, 5, 6, 7** – 낮은 리스크의 작은 최적화.
5. **8, 9** – 렌더/apply_weights API 확인 후, 병목일 때만 시도.

변경 후에는 동일 설정으로 레이턴시와 시각적 결과를 비교해 보는 것을 권장합니다.
