# `edit_latents_multiview` 현재 구조 기술 보고서

> **기준 커밋**: `camera-selection` 브랜치 현재 상태
> **기준 파일**: `threestudio/models/guidance/dge_guidance.py` (L886–L1290)
> **수식**: `$...$` 인라인, `$$...$$` 블록 LaTeX

---

## 목차

1. [구조 변화 요약 — 구 Phase 1/2/3 → 현재](#1-구조-변화-요약)
2. [기호 정의](#2-기호-정의)
3. [전체 파이프라인 개요](#3-전체-파이프라인-개요)
4. [Setup 단계](#4-setup-단계)
5. [Target Denoise Loop](#5-target-denoise-loop)
   - 5.1 Attention 모드 전환 (extended ↔ normal)
   - 5.2 배치 구성 (Adaptive Batching)
   - 5.3 Pivotal Forward
   - 5.4 Per-step Cross-Attention Consistency (optional)
   - 5.5 Batch Forward
   - 5.6 CFG + DDIM Step
6. [4가지 config 조합과 코드 경로](#6-4가지-config-조합과-코드-경로)
7. [DGEBlock 내부 연산](#7-dgeblock-내부-연산)
8. [VcEdit CCM과의 비교](#8-vcedit-ccm과의-비교)
9. [성능 분석 (실측 latency)](#9-성능-분석-실측-latency)
10. [핵심 수식 정리](#10-핵심-수식-정리)
11. [구현 파일 참조](#11-구현-파일-참조)

---

## 1. 구조 변화 요약

### 기존 구조 (Phase 1 / 2 / 3)

```
Phase 1  키 뷰 DDIM 디노이징 + attn2 CA map 수집
Phase 2  inverse render (2D→3D) + re-render (3D→2D)
Phase 3  모든 뷰 디노이징, ConsistentCrossAttnProcessor 사용
```

### 현재 구조 (Phase 1 제거)

```
Setup    키 인덱스 결정, UNet 모듈 캐시, attn2 processor 설치
Target   모든 뷰를 단일 denoise loop로 처리:
Loop       ┌─ (option A) per_step CA consistency:
           │    pivotal forward → CA map 수집 → inverse render → re-render → batch forward
           └─ (option B) extended attention only:
                pivotal forward (kf_attn_output 캐시용) → batch forward
```

**핵심 변화**: Phase 1의 **키 뷰 독립 DDIM 루프가 제거**됐다. 이전에는 키 뷰를 전체 T 스텝 디노이징한 뒤 그 결과로 `key_edited`를 만들고 타깃 루프에서 고정했지만, 현재는 **모든 뷰(키 포함)가 타깃 루프 한 번에서 함께 디노이징**된다. CA map은 per-step으로 매 timestep에서 pivotal forward 시 수집·투영하는 방식으로 전환됐다.

---

## 2. 기호 정의

| 기호 | 의미 |
|------|------|
| $n$ | 전체 뷰 수 (`n_views`) |
| $B$ | 배치 크기 (`camera_batch_size`) |
| $T$ | 디노이징 타임스텝 수 (기본 20, `diffusion_steps`) |
| $\mathcal{I}_{\text{targ}} = \{0,\ldots,n{-}1\}$ | 타깃 뷰 인덱스 (현재 전체 뷰) |
| $\Pi_b$ | 배치 $b$의 pivot 인덱스 (local) |
| $\pi_b$ | 배치 $b$의 pivot 뷰 (global index `target_indices[Π_b]`) |
| $\mathcal{B}_b$ | 배치 $b$의 뷰 인덱스 집합 (local) |
| $\mathbf{z}_v$ | 뷰 $v$의 latent |
| $\mathbf{c}_v$ | 뷰 $v$의 이미지 조건 latent |
| $R$ | 대상 해상도 집합 (`{32×32, 64×64}`) |
| $L_{\text{tok}}$ | 유효 텍스트 토큰 수 (`attn_len`) |
| $G$ | 3D Gaussian Splatting 모델 |
| $M_{\text{con}}^{(v,r)}$ | 뷰 $v$, 해상도 $r$의 consistent CA map $\in \mathbb{R}^{H_r \times W_r \times L_{\text{tok}}}$ |
| `_need_pivotal` | `_per_step_mode or (not use_normal_attn_target)` |
| `_per_step_mode` | `cfg.per_step_cross_attn_consistency` |

---

## 3. 전체 파이프라인 개요

```
입력: {I_v, I_v^c}_{v=0}^{n-1},  P,  G,  T

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 [Setup]
   키 인덱스 결정 (lens_fps / uniform / uniform_random / manual)
   UNet 모듈 캐시 (_dge_blocks, _attn2_modules)
   if per_step_mode: ConsistentCrossAttnProcessor 설치
   latents ← add_noise(latents, t)
   target_indices = [0, ..., n-1]  (Phase 1 제거 후 전체 뷰)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 [Target Denoise Loop]  t = T, T-Δ, …, 0
   per_timestep_setup:
     - use_normal_attn 전환 (warmup ↔ extended)
     - batches 구성 (adaptive / fixed)
     - pivotal_per_batch 선택 (canonical_progressive / random / fixed)

   if _need_pivotal:   ← per_step_mode OR extended attn 활성
     [Pivotal Forward]
       register_pivotal = True
       if per_step_mode AND t >= t_start:
         CrossAttentionStoreProcessor 임시 설치
       UNet([z_π^(t) ‖ c_π] × 모든 pivot 배치)
         → kf_attn_output 갱신 (for extended attn injection)
         → if per_step_mode: CA map 수집
       register_pivotal = False
       if per_step_mode: inverse render + re-render → M_con^(v,r)

   [Batch Forward]  배치 b마다:
     if per_step_mode: _consistent_attn_map_current ← M_con^(v,r)
     if _need_pivotal: DGEBlock attrs 세팅 (batch_idx, cams, pivot...)
     UNet([z_b^(t) ‖ c_b]) × 3 (CFG)  ← batch_noise_pred
     CFG 결합 → noise_pred_accum

   DDIM step → z_targ^(t-1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
출력: latents (z_v^(0) for all v)
```

---

## 4. Setup 단계

### 4.1 키 인덱스 결정

`key_selection_strategy`에 따라 $K$개의 키 뷰를 선정한다.

| 전략 | 방법 |
|------|------|
| `lens_fps` | GS visibility + canonical 방향 기반 FPS (`select_key_views_by_lens_fps`) |
| `uniform` | `linspace(0, n-1, K)` |
| `uniform_random` | 각 $K$개 구간에서 랜덤 1개 선택 |
| `manual` | 외부 전달값 그대로 사용 |

키 인덱스는 **Phase 3 (=현재 target loop)의 pivotal 후보 풀 및 feature injection 소스**로 사용된다.

### 4.2 UNet 모듈 캐시

반복적인 `named_modules()` 순회를 피하기 위해 루프 시작 전 1회만 수집한다:

```python
_dge_blocks   = [(n, m) for n, m ... if isinstance_str(m, "BasicTransformerBlock")]
_attn2_modules = [(n, m) for n, m ... if n.endswith(".attn2") and hasattr(m, "processor")]
```

### 4.3 Processor 설치

`per_step_mode=True`일 때만 `ConsistentCrossAttnProcessor`를 attn2에 설치한다. 그렇지 않으면 기존 processor 유지.

### 4.4 Latent 초기화

$$\mathbf{z}_v^{(T)} = \sqrt{\bar{\alpha}_T}\,\mathbf{z}_v + \sqrt{1-\bar{\alpha}_T}\,\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

현재는 키 뷰 고정(`key_edited`)이 없으므로 **모든 뷰가 동일하게 noise 추가**된다.

---

## 5. Target Denoise Loop

### 5.1 Attention 모드 전환

`target_use_extended_attention` config에 따라 매 timestep에서 `use_normal_attn_target` 플래그를 관리한다.

```
target_use_extended_attention = True (기본):
  t >= 100  → register_normal_attn_flag(unet, False)  # extended attn ON
  t < 100   → use_normal_unet()                        # warmup: normal

target_use_extended_attention = False:
  항상 use_normal_unet()                               # normal self-attn만
```

`use_normal_attn_target=True`이면 `DGEBlock.forward`가 첫 줄에서 `block_class.forward()`로 bypass되므로, DGE 연산 전체가 스킵된다.

`_need_pivotal = _per_step_mode or (not use_normal_attn_target)`: DGE 경로 또는 per-step CA가 필요한 경우에만 pivotal forward를 실행한다.

### 5.2 배치 구성 (Adaptive Batching)

`target_batch_strategy=adaptive`일 때 `_build_target_batches`가 timestep에 따라 두 가지 모드로 동작한다.

```
t >= target_batch_neighbor_threshold (기본 900):
  early mode: sequential batches  [0..B-1], [B..2B-1], ...
              → 인접 뷰끼리 묶여 feature injection 효과 극대화

t < target_batch_neighbor_threshold:
  late mode (sliding_window): offset = (step_index × stride) % n
              → 매 step마다 배치 시작점이 stride씩 밀림
              → 모든 뷰가 고르게 각 배치 위치를 경험
```

`target_batch_strategy=fixed`이면 항상 sequential batches.

### 5.3 Pivotal Forward

`_need_pivotal=True`일 때 매 timestep에 실행된다.

**Pivotal 선택 전략** (`target_key_selection_mode`):

| 모드 | 방법 |
|------|------|
| `canonical_progressive` | 초기(early): frontal 뷰 선호 (canonical score 높은 쪽); 후기(late): least-used 뷰 선호 |
| `fixed` | 각 배치의 첫 번째 인덱스 |
| `random` | 각 배치에서 랜덤 선택 |

**배치화**: `pivotal_per_batch`는 배치 수만큼의 pivot index 리스트. 이를 한 번에 묶어 **단일 UNet forward**로 처리한다.

$$\text{input} = \bigl[\mathbf{z}_\Pi^{(t)};\,\mathbf{c}_\Pi\bigr] \in \mathbb{R}^{(3 \cdot n_{\text{pivot}}) \times 8 \times H_l \times W_l}$$

`register_pivotal(unet, True)` → UNet forward → `register_pivotal(unet, False)`

이 forward에서:
- **attn1**: Extended ST-Attention 실행 → `kf_attn_output` 갱신
- **attn2**: `ConsistentCrossAttnProcessor` (per_step_mode시) 또는 원본 processor

### 5.4 Per-step Cross-Attention Consistency (optional)

`_per_step_mode=True`이고 `t >= per_step_t_start`일 때만 활성화.

**① CA Map 수집 (pivotal forward 직전)**

```python
_step_store_proc = CrossAttentionStoreProcessor(valid_indices, target_resolutions)
# 모든 attn2의 processor를 임시 교체
for _name, _mod in _attn2_modules:
    _step_saved_procs[_name] = _mod.processor
    _mod.processor = _step_store_proc
```

Pivotal forward가 끝나면 원본 processor 복원.

**② Inverse Render: 2D → 3D** (`inverse_render_2d_to_3d`)

수집된 CA map $M_{\text{key}}^{(r)} \in \mathbb{R}^{n_{\text{pivot}} \times L_r \times L_{\text{tok}}}$을 3D로 역투영.

$$M_{\text{3D}}^{(r)}(g, c) = \frac{\displaystyle\sum_{k \in \Pi} \text{apply\_weights}(G, \text{cam}_k^{(r)}, M_{\text{key},k}^{(r)}[\,:\,,c\,])}{\text{count}(g) + \epsilon}$$

**③ Re-render: 3D → 2D** (`render_3d_to_2d`)

$$M_{\text{con}}^{(v,r)}[\,:\,,c\,] = \text{GS\_Render}\!\left(G,\; \text{cam}_v^{(r)},\; \text{color} = M_{\text{3D}}^{(r)}[\,:\,,c\,]\right)$$

루프: `for v in target_indices: for r in R: for c in range(attn_len): gs_render()`

> **현재 병목**: 이 3중 루프가 전체 시간의 ~60%를 차지한다 (`render_3d_to_2d: 27.6s / 53.5s`).
> `n=20, |R|=2, attn_len≈9`일 때 1 timestep당 360회 GS render × 활성 timestep 수.

### 5.5 Batch Forward

배치 $b$, 뷰 집합 $\mathcal{B}_b$에 대해:

**1) DGEBlock 속성 세팅** (`_need_pivotal=True`일 때)

```python
for _, mod in _dge_blocks:
    setattr(mod, "batch_idx", batch_idx)
    setattr(mod, "cams", [target_cams[j] for j in batch_local])
    setattr(mod, "pivot_this_batch", _pivot_in_batch)
    setattr(mod, "key_cams", key_cams_batch)
    setattr(mod, "epipolar_constrains", {})
```

**2) Consistent CA map 주입** (`per_step_mode=True`일 때)

```python
for _, mod in _attn2_modules:
    setattr(mod, "_consistent_attn_map_current", _attn_map_val)
```

**3) UNet forward** (CFG 3배치)

$$\text{input} = \bigl[\mathbf{z}_b^{(t)};\,\mathbf{c}_b\bigr]^{\times 3} \in \mathbb{R}^{3B \times 8 \times H_l \times W_l}$$

이 forward에서 DGEBlock은 `pivotal_pass=False` 경로를 탄다 → feature injection 실행.

### 5.6 CFG + DDIM Step

$$\hat{\boldsymbol{\epsilon}}_v = \boldsymbol{\epsilon}^{\text{unc}} + s_{\text{txt}}\bigl(\boldsymbol{\epsilon}^{\text{txt}} - \boldsymbol{\epsilon}^{\text{img}}\bigr) + s_{\text{img}}\bigl(\boldsymbol{\epsilon}^{\text{img}} - \boldsymbol{\epsilon}^{\text{unc}}\bigr)$$

$$\mathbf{z}_v^{(t-1)} = \text{DDIM\_step}\!\left(\hat{\boldsymbol{\epsilon}}_v,\; t,\; \mathbf{z}_v^{(t)}\right)$$

누적 accumulator 방식: 뷰가 여러 배치에 걸칠 경우 `noise_pred_accum / noise_pred_count`로 평균.

---

## 6. 4가지 Config 조합과 코드 경로

| # | `ext_attn` | `per_step` | `_need_pivotal` | Pivotal fwd | CA map 3D | DGEBlock attrs | 의미 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **TT** | T | T | **T** | O | O | O | Extended self-attn + 매 timestep 3D CA consistency (full pipeline) |
| **TF** | T | F | **T** | O | X | O | Extended self-attn만 (kf_attn으로 feature injection); CA consistency 없음 |
| **FT** | F | T | **T** | O (CA용) | O | O (불필요하나 무해) | Normal self-attn + cross-attn만 3D 일관성 적용 |
| **FF** | F | F | **F** | X | X | X | 순수 per-view IP2P; DGE 경로 완전 스킵 |

> **주의**: `ext_attn=True`의 warmup 구간(`t < 100`)에서는 일시적으로 `use_normal_attn_target=True`가 되므로 그 timestep에서 `_need_pivotal`은 `_per_step_mode`에만 의존한다.

---

## 7. DGEBlock 내부 연산

### 7.1 Pivotal Pass (attn1)

`pivotal_pass=True`이면 Extended Spatio-Temporal Self-Attention을 수행하고 `kf_attn_output`을 저장·갱신한다.

$$\text{ST-Attn}(Q_t, K_{1:n_\pi}) = \text{softmax}\!\!\left(\frac{Q_t \cdot [K_1,\ldots,K_{n_\pi}]^\top}{\sqrt{d}}\right) \cdot [V_1,\ldots,V_{n_\pi}]$$

### 7.2 Non-Pivotal Pass: Feature Injection (attn1, `feature_injection_mode`별)

Q-K-V attention을 수행하지 않고 `kf_attn_output` 캐시에서 feature를 주입한다.

**similarity 모드** (현재 기본):

```
1. k*(v) = argmin_k ‖o_v - o_k‖₂        (가장 가까운 pivot 뷰)
2. sim[v,p,q] = cosine(H_v[p], H_pivot[q])
3. j*_p = argmax_q sim[v,p,q]            (에피폴라 제약 없음)
4. O_self[v,p] = kf_attn_output[k*(v), j*_p, :]
5. H ← O_self + H
```

**3d_anchor 모드** (`blend` / `gather`): GS 기반 픽셀–가우시안 대응 사용 (별도 문서 참조).

**none 모드**: feature injection 스킵 (`disable_feature_injection=True`).

### 7.3 Consistent Cross-Attention (attn2, per_step_mode)

표준 $\text{softmax}(QK^\top/\sqrt{d})$를 생략하고 3D 투영된 map을 직접 사용:

$$\mathbf{O}_{\text{cross}}^{(v)} = M_{\text{con}}^{(v,r)} \cdot W_v \mathbf{E}_{\text{text}}[\,:\,,\mathcal{T}\,]$$

---

## 8. VcEdit CCM과의 비교

### 8.1 아키텍처 차이

| 항목 | VcEdit (`pipeline_con.py`) | DGE (`edit_latents_multiview`) |
|------|------|------|
| **CA map 소스 뷰** | **모든 뷰** 각각 UNet forward 후 수집 | **pivot 뷰만** (배치당 1~수개) |
| **적용 주기** | `attn_ctrl_steps` 간격 (기본 매 step) | `per_step_t_start` 이상 timestep에서만 |
| **CA map 수집** | 모든 뷰의 CA map → 평균 → 3D | pivot 뷰 CA map → 3D |
| **inverse render** | attn_len 채널 × 뷰별 루프 | attn_len 채널 × 뷰별 루프 (동일 구조) |
| **re-render** | ceil(channels/3)로 묶어서 views × 루프 | 채널별 개별 render (현재 미최적화) |
| **적용 방식** | controller.consist_store 교체 → 다음 fwd에서 사용 | `_consistent_attn_map_current`로 해당 step batch forward에서 즉시 사용 |
| **self-attn** | 변경 없음 (표준 IP2P self-attn) | DGE feature injection (extended ST-attn / cosine 유사도 gather) |
| **배치화** | 뷰별 순차 UNet forward | 다수 pivot을 한 번에 배치화 + 타깃 뷰도 배치 |

### 8.2 UNet Forward 횟수 비교 (1 timestep 기준, n=20, B=5, n_pivot=4)

| 단계 | VcEdit | DGE (TT 조합) | 비고 |
|------|------|------|------|
| CA 수집용 forward | 20회 (뷰별 순차) | 1회 (4 pivot 배치) | DGE **20x 절약** |
| 실제 denoise forward | 20회 (뷰별 순차) | 4회 (배치, B=5) | DGE **5x 절약** |
| **합계** | **40회** | **5회** | **DGE ~8x 절약** |

### 8.3 GS render 횟수 비교 (1 timestep, per_step_t_start 이하 기준)

| 단계 | VcEdit | DGE | 비고 |
|------|------|------|------|
| inverse render (apply_weights) | 20 views × channels | 4 pivot × channels | DGE 5x 절약 |
| re-render (GS render) | ceil(channels/3) × 20 views | channels × 20 views | **DGE 3x 불리** (채널별 개별 render) |

> **현재 re-render 최적화 기회**: `render_3d_to_2d`를 3채널 단위로 묶으면 VcEdit와 동등 수준으로 줄일 수 있다 (~3x 단축).

### 8.4 Quality 관점 차이

| 관점 | VcEdit | DGE |
|------|------|------|
| CA map 정보량 | 모든 뷰 평균 → 풍부하지만 노이즈 | pivot만 → 단순하지만 깨끗한 single-source |
| self-attn 일관성 | 없음 (뷰별 독립) | DGE feature injection으로 cross-view 일관성 |
| 3D anchor | 없음 | 있음 (3d_anchor 모드) |
| 적용 타임스텝 | 모든 step (attn_ctrl_steps 간격) | `per_step_t_start` 이상만 → 초기 구조 형성 단계 집중 |

---

## 9. 성능 분석 (실측 latency)

`tt` 조합 (`ext_attn=True`, `per_step=True`), face 데이터셋, `per_step_t_start=500` 기준:

```
target_denoise_loop:        53.5s (100%)
├─ per_step_consistent_maps: 32.0s ( 60%)  ← 주요 병목
│   ├─ render_3d_to_2d:       27.6s ( 52%)  ← 핵심 병목 (채널별 개별 GS render)
│   └─ inverse_render_2d_to_3d: 4.4s (  8%)
├─ batch_forward:            10.9s ( 20%)
└─ pivotal_forward:           7.3s ( 14%)
```

### 병목 원인 분석 (`render_3d_to_2d`)

```python
for global_idx in target_indices:    # 20 views
    for res in M_3d_by_res:          # 2 resolutions
        for c in range(attn_len):    # ~9 tokens
            gs_render(...)           # 매번 full GS render (~4ms/call)
```

1 active timestep당 $20 \times 2 \times 9 = 360$회 GS render.
`per_step_t_start=500`이면 20 step 중 약 18 step이 활성 → 총 **~6,480회** GS render.

### 최적화 방향

1. **re-render 3채널 배치화**: `attn_len` 채널을 3개씩 묶어 `override_color`에 한 번에 전달 → ~3x 단축 (~27.6s → ~9s)
2. **per_step_t_start 조정**: 값을 높이면 활성 timestep 수 감소 (품질 trade-off)
3. **attn_len 축소**: 편집에 핵심적인 토큰만 선택

---

## 10. 핵심 수식 정리

**(S1) Latent 노이즈 추가:**

$$\mathbf{z}_v^{(T)} = \sqrt{\bar{\alpha}_T}\,\mathbf{z}_v + \sqrt{1-\bar{\alpha}_T}\,\boldsymbol{\epsilon}$$

**(S2) Extended ST Self-Attention (attn1, pivotal):**

$$\text{ST-Attn}(Q_t, K_{1:n_\pi}) = \text{softmax}\!\!\left(\frac{Q_t \cdot [K_1,\ldots,K_{n_\pi}]^\top}{\sqrt{d}}\right) \cdot [V_1,\ldots,V_{n_\pi}]$$

**(S3) Inverse Render (2D → 3D, per-step):**

$$M_{\text{3D}}^{(r)}(g, c) = \frac{\sum_{k \in \Pi_t} \text{apply\_weights}(G,\, \text{cam}_k^{(r)},\, M_k^{(r)}[\,:\,,c\,])}{\text{count}(g) + \epsilon}$$

**(S4) Re-render (3D → 2D, per-step):**

$$M_{\text{con}}^{(v,r)}[\,:\,,c\,] = \text{GS\_Render}(G,\; \text{cam}_v^{(r)},\; M_{\text{3D}}^{(r)}[\,:\,,c\,])$$

**(S5) DGE Feature Injection (attn1, non-pivotal, similarity):**

$$j^*_p = \arg\max_{q}\; \text{cosine}(\tilde{\mathbf{H}}_v^{(p)},\, \tilde{\mathbf{H}}_\pi^{(q)})$$

$$\mathbf{O}_{\text{self}}^{(v)} = \texttt{kf\_attn\_output}[\pi,\; j^*_{0:L},\; :]$$

**(S6) Consistent Cross-Attention (attn2, per_step_mode):**

$$\mathbf{O}_{\text{cross}}^{(v)} = M_{\text{con}}^{(v,r)} \cdot W_v \mathbf{E}_{\text{text}}[\,:\,,\mathcal{T}\,]$$

**(S7) CFG (IP2P):**

$$\hat{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}^{\text{unc}} + s_{\text{txt}}(\boldsymbol{\epsilon}^{\text{txt}} - \boldsymbol{\epsilon}^{\text{img}}) + s_{\text{img}}(\boldsymbol{\epsilon}^{\text{img}} - \boldsymbol{\epsilon}^{\text{unc}})$$

---

## 11. 구현 파일 참조

| 기능 | 파일 | 위치 |
|------|------|------|
| 전체 파이프라인 | `threestudio/models/guidance/dge_guidance.py` | `DGEGuidance.edit_latents_multiview` (L886~) |
| Per-step CA map 구성 | 같은 파일 | `_per_step_build_consistent_maps` (L55~134) |
| 배치 구성 | 같은 파일 | `_build_target_batches` (L137~173) |
| Pivot 선택 | 같은 파일 | `_select_pivotals_for_batches` (L176~) |
| DGEBlock forward | `threestudio/utils/dge_utils.py` | `make_dge_block` → `DGEBlock.forward` |
| Extended ST-Attn | `dge_utils.py` | `register_extended_attention` → `sa_forward` |
| Pivotal 등록 | `dge_utils.py` | `register_pivotal` (L265) |
| Normal attn 플래그 | `dge_utils.py` | `register_normal_attn_flag` (L331) |
| CA map 저장 processor | `dge_guidance.py` | `CrossAttentionStoreProcessor` |
| Consistent CA processor | `dge_guidance.py` | `ConsistentCrossAttnProcessor` |
| 키 뷰 선택 (lens_fps) | `threestudio/data/gs_load.py` | `select_key_views_by_lens_fps` |

---

*작성 기준: `camera-selection` 브랜치, 2026-03-03*
