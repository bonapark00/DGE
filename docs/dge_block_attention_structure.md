# DGEBlock의 attention 구조 & IP2P 대비 변경점

DGEBlock은 diffusers의 **BasicTransformerBlock**을 상속해 **forward 전체**를 갈아끼우고, **attn1의 forward**만 별도로 바꾼 구조입니다. attn2 레이어 자체는 그대로 두고 **processor만** 바꿔서 씁니다.

---

## 1. IP2P / diffusers 기본 블록 구조

Stable Diffusion / InstructPix2Pix의 한 Transformer 블록(**BasicTransformerBlock**)은 보통 다음 순서입니다.

```
입력 hidden_states (B, seq, dim)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  1. Self-Attention (attn1)                               │
│     norm1(hidden_states) → attn1(norm1_out) → + residual │
├─────────────────────────────────────────────────────────┤
│  2. Cross-Attention (attn2)                              │
│     norm2(hidden_states) → attn2(norm2_out, encoder_*)   │
│     → + residual                                         │
├─────────────────────────────────────────────────────────┤
│  3. Feed-Forward                                         │
│     norm3(hidden_states) → ff(norm3_out) → + residual    │
└─────────────────────────────────────────────────────────┘
    │
    ▼
출력 hidden_states
```

- **attn1**: self-attention. query/key/value 모두 **같은** hidden_states에서 나옴. 공간 내에서만 attention.
- **attn2**: cross-attention. query = 현재 hidden (공간), key/value = **encoder_hidden_states** (텍스트 또는 이미지 조건). 공간 ↔ 조건 간 attention.
- **norm1, norm2, norm3**, **ff**: 블록 그대로 유지.

---

## 2. DGEBlock에서의 구조 (한 블록 안 순서)

DGE는 **같은 블록**을 쓰되, **forward 전체**를 `make_dge_block(block_class)` 로 덮어씁니다. 그래서 "연산 순서"와 "attn1이 하는 일"이 IP2P와 다릅니다.

```
입력 hidden_states  (3*n_frames, seq, dim)  ← CFG 3배치 × 뷰 수
    │
    ▼
  norm1(hidden_states)  →  norm_hidden_states
    │
    ├── [pivotal_pass]  →  pivot_hidden_states = norm_hidden_states 저장
    │
    └── [아닐 때]  →  camera_distance, closest_cam
                      →  closest_cam_pivot_hidden_states = pivot_hidden_states[closest_cam]
                      →  sim = einsum(현재 뷰, pivot 뷰)  (spatial similarity)
                      →  epipolar 마스크 (선택)  →  idx1, idx2 (어느 pivot 픽셀 쓸지)
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  1. Spatio-temporal attention 단계 (attn1 = extended attention)                       │
│                                                                                       │
│  [pivotal_pass]  attn1(norm_hidden_states)                                            │
│                  →  Qt·[K1,...,KT] / √d  (이번에 들어온 n_frames끼리 서로 attend)        │
│                  →  attn_output  →  kf_attn_output 저장  ← key/pivot frame끼리 ST-attn │
│                                                                                       │
│  [아닐 때]  attn1으로 Q·K·V 다시 계산 안 함                                            │
│             →  attn_output = kf_attn_output[closest_cam]  (pivot의 attn1 결과만 참조)  │
│             →  feature_injection:  idx1/idx2로 gather                                 │
│                →  attn_output = pivot 쪽 attn 값을 현재 뷰 위치에 맞게 주입               │
└─────────────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
  hidden_states = attn_output + hidden_states   (residual)
    │
    ▼
  norm2(hidden_states)  →  attn2(norm2_out, encoder_hidden_states)  →  + residual
    │
    ▼
  norm3  →  ff  →  + residual  →  출력
```

요약하면:

- **norm1** 직후에 **pivot용 저장** 또는 **similarity + epipolar + idx1/idx2** 계산이 들어가고,
- **attn1**은 **spatio-temporal attention**을 담당한다.
  - **pivotal_pass일 때**: 이번에 들어온 뷰들(key 또는 pivot)만으로 **attn1을 한 번 실행** → `Qt·[K1,...,KT]`처럼 **그 뷰들끼리 서로 attend**하고, 결과를 `kf_attn_output`으로 저장.
  - **아닐 때**: attn1을 다시 돌리지 않고, 저장된 **kf_attn_output**을 가장 가까운 pivot 기준으로 **idx1/idx2로 gather (feature injection)** 만 함.
- 그 다음 **attn2 → norm3 → ff** 는 IP2P와 같은 순서.

---

## 3. Spatio-temporal attention (key frame끼리 서로 attend)

논문에서 말하는 **"modified spatio-temporal self-attention"**—각 프레임이 **모든 프레임**에 attend하는 부분—은 DGE에서는 **attn1을 대체한 extended attention** 안에 구현되어 있다.

### 3.1 논문 쪽 정의와 대응

- **수식**:
  `STAttn(Q, K, t) = Softmax(Qt · [K1,...,KT] / √d)`
  - `Qt`: 한 viewpoint(프레임) t의 query
  - `[K1,...,KT]`: **모든 T개 viewpoint의 key**를 concat
  - 즉 **한 프레임의 query가 모든 프레임의 key에** attend.
- **출력**:
  `Φt = STAttn(Q, K, t) · [V1,..., VT]`
  → 모든 프레임의 value를 attention 가중으로 합침.

### 3.2 코드에서의 위치

- **파일**: `threestudio/utils/dge_utils.py`
- **함수**: `register_extended_attention` 안의 **`sa_forward`** (323~410행 근처).
- 이 함수가 **attn1.forward**로 붙어 있어서, attn1이 호출될 때마다 이 spatio-temporal attention이 실행된다.

**수식과의 대응**:

- 입력 `x`: `(batch_size, sequence_length, dim)` = `(3 * n_frames, seq, dim)`
  (CFG 3 × 뷰 수).
- `n_frames` = 이번 forward에 들어온 **뷰 개수**.
- `k_text` 등:
  `k[:n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)`
  → **모든 n_frames 뷰의 key를 concat**한 것 `[K1,...,KT]`에 해당.
- `sim_text = torch.bmm(q_text[:, j], k_text[:, j].transpose(-1, -2)) * self.scale`
  → 각 뷰의 query `Qt`가 **모든 뷰의 key**에 attend (softmax 후 value 곱하면 `Φt`).

즉 **"key frame끼리 서로 attend"** 하려면, **이 attn1이 key view만 들어온 상태로 호출되면** 된다.

### 3.3 언제 "key frame끼리만" spatio-temporal attention이 일어나는가

- **edit_latents_multiview – key_view_denoise_loop**
  - `latent_model_input` = **key_indices 뷰만** (예: 4개) × CFG 3.
  - 이걸로 UNet을 돌리므로 **attn1에는 key view만** 들어가고,
  → **key frame끼리만 서로 attend**하는 spatio-temporal attention이 됨.
- **edit_latents – pivotal_forward**
  - `latent_model_input` = **pivotal_idx 뷰만** (그 스텝의 pivot 뷰들).
  - 이때도 attn1에는 "그 스텝의 pivot view들만" 들어가서,
  → 그 pivot view들끼리 서로 attend.

**Non-pivotal pass**에서는 배치에 **key를 포함한 전체 뷰**가 들어가지만, attn1에서는 **실제로 Q·K·V를 다시 계산하지 않고**, pivotal pass에서 저장한 **kf_attn_output**을 가장 가까운 pivot 뷰 기준으로 **gather (feature injection)** 만 한다.
즉 "모든 뷰끼리 서로 attend"하는 full spatio-temporal은 **pivotal (또는 key) view만 넣었을 때의 attn1**에서만 수행된다.

---

## 4. attn1: IP2P vs DGE (바뀐 부분)


| 구분     | IP2P (기본)                       | DGE                                                                                                                        |
| ------ | ------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **역할** | 단순 self-attention (한 배치 내 공간만)  | **뷰 간(spatio-temporal) attention** + pivot 저장/주입                                                                           |
| **구현** | `attn1.forward` 그대로 (q,k,v = x) | `attn1.forward` 를 **교체** (`register_extended_attention` 의 `sa_forward`)                                                    |
| **입력** | (B, seq, dim)                   | (3*n_frames, seq, dim) — CFG 3배치 × 프레임(뷰)                                                                                  |
| **동작** | 전체에 대해 한 번에 attention           | **text / image / uncond 브랜치별로** 분리해서, **n_frames끼리** attention (같은 브랜치 내 뷰 간). 즉 "self"가 "같은 CFG 브랜치 내 여러 뷰"에 대한 attention |


**Extended attention (sa_forward)** 요약:

- `q, k, v` 를 text/image/uncond 3묶음으로 나눔.
- 각 묶음 안에서만 `sim = q @ k^T`, softmax, `attn = sim @ v` 수행.
- 결과적으로 **뷰 간** attention (한 뷰의 픽셀이 다른 뷰의 픽셀을 봄).
→ pivot pass일 때 이 출력이 `kf_attn_output`으로 저장되고, non-pivotal일 때는 이걸 **가장 가까운 pivot 뷰**에서 가져와서 **idx1/idx2로 gather** 해서 씀 (feature injection).

그래서 **attn1은 "블록 구조상 1번째 attention" 자리는 그대로인데, 하는 일은 완전히 DGE용으로 바뀐 블록**이라고 보면 됩니다.

---

## 5. attn2: IP2P vs DGE (바뀐 부분)


| 구분            | IP2P (기본)                                     | DGE                                                                         |
| ------------- | --------------------------------------------- | --------------------------------------------------------------------------- |
| **레이어**       | 그대로 유지 (norm2 → attn2 → residual)             | **동일**                                                                      |
| **연산**        | cross-attention (query=공간, key/value=encoder) | **동일** (processor가 할 일만 다름)                                                 |
| **processor** | 기본 `Attention` forward                        | **교체 가능**: `CrossAttentionStoreProcessor`, `ConsistentCrossAttnProcessor` 등 |


즉, **attn2 모듈 자체는 IP2P와 동일**하고, **어떤 processor를 붙이느냐**만 다릅니다.

- **기본**: diffusers 기본 cross-attention (텍스트/이미지 조건으로 key, value 생성).
- **CrossAttentionStoreProcessor**: forward 하면서 **attention map(공간×토큰)** 을 저장 → key view denoise 시 수집.
- **ConsistentCrossAttnProcessor**: encoder 대신 **미리 만든 consistent map**을 value로 넣음 → target view denoise 시 사용.

정리하면, **구조(순서, norm2–attn2–residual)는 IP2P 그대로**이고, **attn2의 "processor"만 바꾼 것**입니다.

---

## 6. 블록 단위 비교 표


| 단계        | IP2P (BasicTransformerBlock) | DGE (DGEBlock)                                                                               |
| --------- | ---------------------------- | -------------------------------------------------------------------------------------------- |
| norm1     | ✓                            | ✓ (그 후 pivot 저장 또는 similarity/idx 계산 추가)                                                     |
| attn1     | self-attention (공간만)         | **교체**: extended attn (뷰 간 ST-attn) + pivotal 시 저장, 비pivotal 시 pivot attn gather (feature injection) |
| residual  | attn1 + residual             | attn1 단계 출력(또는 injected) + residual                                                          |
| norm2     | ✓                            | ✓                                                                                            |
| attn2     | cross-attention (텍스트/이미지)    | **같은 레이어**, processor만 교체 가능 (store / consistent)                                            |
| residual  | attn2 + residual             | ✓                                                                                            |
| norm3, ff | ✓                            | ✓                                                                                            |


---

## 7. 한 줄 요약

- **구조**: IP2P와 동일하게 **norm1 → attn1 → residual → norm2 → attn2 → residual → norm3 → ff**.
- **Spatio-temporal attention**: 논문의 "each frame attends to every other frame"은 **attn1(extended attention)** 에서 구현됨. **key view만**(또는 pivot view만) 넣었을 때(key_view_denoise_loop / pivotal_forward) 그 뷰들끼리만 서로 attend; non-pivotal pass에서는 full ST-attn 대신 pivot의 attn 결과를 gather해서 씀.
- **바뀐 블록/부분**:
  - **attn1**: forward 전체가 **extended attention + pivot 저장/feature injection**으로 교체됨 (뷰 간 spatio-temporal attention + DGE용 보조 연산).
  - **attn2**: 레이어는 그대로, **processor만** Store/Consistent 등으로 바꿔서 cross-attention 수집·대체 사용.
  - **norm1 직후**: pivot_hidden_states 저장, 또는 similarity/epipolar/idx1,idx2 계산이 추가됨.

즉, "IP2P 구조를 그대로 따르되, **첫 번째 attention(attn1)의 역할만 완전히 DGE용으로 바꾸고**(spatio-temporal + feature injection), **두 번째 attention(attn2)은 processor만 바꾼다**"고 보면 됩니다.
