# 멀티뷰 편집 파이프라인 기술 보고서

> **수식 렌더링 안내**: `$...$`(인라인) 및 `$$...$$`(블록) LaTeX 수식 사용.
> VS Code에서 렌더링하려면 **Markdown All in One** 또는 **Markdown+Math** 확장을 설치하거나, GitHub에서 열어주세요.

---

## 목차

1. [개요 및 동기](#1-개요-및-동기)
2. [기호 정의](#2-기호-정의)
3. [전체 파이프라인 개요](#3-전체-파이프라인-개요)
4. [Phase 1 — 키 뷰 디노이징 및 Cross-Attention 수집](#4-phase-1--키-뷰-디노이징-및-cross-attention-수집)
5. [Phase 2 — 3D 기반 일관 Attention Map 구성](#5-phase-2--3d-기반-일관-attention-map-구성)
6. [Phase 3 — 타깃 뷰 디노이징](#6-phase-3--타깃-뷰-디노이징)
7. [DGEBlock 내부 연산 순서](#7-dgeblock-내부-연산-순서)
8. [Attention 연산 전체 흐름 요약](#8-attention-연산-전체-흐름-요약)
9. [핵심 수식 목록](#9-핵심-수식-목록)
10. [알고리즘 박스](#10-알고리즘-박스)
11. [IP2P 대비 변경점 정리](#11-ip2p-대비-변경점-정리)
12. [구현 파일 참조](#12-구현-파일-참조)

---

## 1. 개요 및 동기

### 1.1 문제 설정

3D Gaussian Splatting(3DGS)으로 표현된 장면을 텍스트 지시어(text instruction)로 편집할 때, 각 카메라 뷰를 독립적으로 InstructPix2Pix(IP2P)에 입력하면 **뷰 간 편집 불일치(multi-view inconsistency)** 가 발생한다. 같은 3D 위치라도 뷰마다 다른 편집 결과가 적용되어 3DGS 최적화 시 충돌이 생긴다.

### 1.2 핵심 아이디어

본 방법은 두 가지 메커니즘으로 뷰 간 일관성을 확보한다.

**① 3D-일관 Cross-Attention (Consistent Cross-Attention)**

소수의 키 뷰(key views)에서 IP2P UNet의 cross-attention map — "텍스트 토큰이 이미지의 어느 공간 위치에 attention을 주는가" — 을 수집한다. 이 2D map을 3DGS를 통해 3D 공간으로 역투영(inverse rendering)한 뒤, 모든 뷰의 카메라로 재렌더링한다. 결과적으로 **모든 뷰가 동일한 3D attention 필드를 2D로 바라보는** 기하적으로 일관된 map을 얻는다. 타깃 뷰 디노이징 시 이 map을 cross-attention 가중치로 직접 대체한다.

**② DGE 기반 Cross-View Self-Attention (Feature Injection)**

타깃 뷰 디노이징 중 UNet의 self-attention(attn1)을 대체하여, pivot 뷰의 feature를 타깃 뷰에 주입한다. **대응 방식**은 설정에 따라 다르다.  
- **similarity**: **코사인 유사도**로 대응 픽셀(idx1)을 찾고 pivot self-attention 출력을 gather하여 residual에 더한다. (에피폴라 제약 미사용.)  
- **3d_anchor**: 3DGS prior로 픽셀–가우시안 대응을 쓰며, **blend** 스타일이면 $t_j$, $F(v,p)$, $\lambda(F-h)$ 혼합; **gather** 스타일이면 similarity와 동일 구조(pivot 맵 그대로 + 3D GS remap idx_3d로 gather + residual, λ 없음).  

설정: `feature_injection_mode` (`"similarity"` | `"3d_anchor"`), 3d_anchor 시 `injection_3d_anchor_style` (`"blend"` | `"gather"`).

---

## 2. 기호 정의

| 기호 | 의미 |
|------|------|
| $n$ | 전체 뷰 수 |
| $K$ | 키 뷰 수 |
| $\mathcal{I}_{\text{key}} \subset \{0,\ldots,n-1\}$ | 키 뷰 인덱스 집합, $\lvert\mathcal{I}_{\text{key}}\rvert = K$ |
| $\mathcal{I}_{\text{targ}}$ | 타깃 뷰 인덱스 집합 |
| $\mathbf{z}_v \in \mathbb{R}^{4 \times H_l \times W_l}$ | 뷰 $v$의 VAE latent |
| $\mathbf{c}_v \in \mathbb{R}^{4 \times H_l \times W_l}$ | 뷰 $v$의 이미지 조건 latent |
| $\tau_\theta(P) \in \mathbb{R}^{77 \times 768}$ | 프롬프트 $P$의 CLIP 텍스트 임베딩 |
| $T$ | DDIM 디노이징 총 타임스텝 수 (기본 20) |
| $r$ | UNet 공간 해상도 인덱스 (예: $32{\times}32$, $64{\times}64$) |
| $L_r = H_r \times W_r$ | 해상도 $r$에서의 spatial token 수 |
| $\mathcal{T}$ | 유효 텍스트 토큰 인덱스 집합; $L_{\text{tok}} = \lvert\mathcal{T}\rvert$ |
| $d$ | attention head당 채널 차원; $h$: head 수 |
| $N_g$ | 3D Gaussian 개수 |
| $G$ | 3D Gaussian Splatting 모델 |
| $\mathbf{o}_v \in \mathbb{R}^3$ | 뷰 $v$의 카메라 중심 (world 좌표) |
| $s_{\text{txt}},\, s_{\text{img}}$ | CFG guidance scale (기본 7.5, 1.5) |

---

## 3. 전체 파이프라인 개요

```
입력: {I_v}     렌더 이미지 (n개)
      {I_v^c}   원본(조건) 이미지 (n개)
      G         3D Gaussian 모델
      P         편집 프롬프트 (텍스트)
      I_key     키 뷰 인덱스 집합 (K개)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 [VAE 인코딩]
   z_v  = VAE_encode(I_v)           ∀v
   c_v  = VAE_encode_mode(I_v^c)    ∀v
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 [Phase 1]  키 뷰 디노이징 + Cross-Attention 저장
   • CrossAttentionStoreProcessor → 모든 attn2에 설치
   • DDIM forward: z_k^(T) ← add_noise(z_k, T)    k ∈ I_key
   • for t = T … 0:
       ε_k = UNet([z_k^(t); c_k], t, τ(P))   [pivotal_pass=True]
       ↳ attn1: Extended ST-Attn (키 뷰끼리 서로 attend)
                → kf_attn_output 저장하지 않음 (타깃 단계에서 미사용; 메모리 절약)
       ↳ attn2: 표준 Cross-Attn + A[:,:,T] 해상도별 저장
       z_k^(t-1) ← DDIM_step(CFG(ε))
   • key_edited ← {z_k^(0)}
   • M_key^r   ← 저장된 attn2 map 집계  [K, L_r, L_tok]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 [Phase 2]  3D-일관 Attention Map 구성
   ① 역투영 (2D→3D):
      M_3D^r[g,c] = apply_weights(G, key_cams, M_key^r[:,c])
                    [N_g, L_tok]  (해상도 r별)
   ② 재렌더링 (3D→2D, 모든 뷰):
      M_con^(v,r)[:,c] = GS_Render(G, cam_v, color=M_3D^r[:,c])
                         [n, H_r, W_r, L_tok]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 [Phase 3]  타깃 뷰 디노이징 (Consistent Cross-Attn 사용)
   • ConsistentCrossAttnProcessor → 모든 attn2에 설치
   • z_v^(T) ← add_noise(z_v, T);   z^(T)[I_key] ← key_edited
   • for t = T … 0:
       ① Pivotal forward [pivotal_pass=True]
          UNet([z_π^(t); c_π]) → kf_attn_output 갱신
       ② 배치 forward [pivotal_pass=False]  (배치 단위, CFG 3×)
          set _consistent_attn_map_current = M_con^(v,r)
          ε_b = UNet([z_b^(t); c_b], t, τ(P))
          ↳ attn1 (DGE): similarity면 유사도 idx1 gather; 3d_anchor면 blend($λ(F−h)$) 또는 gather(3D GS idx_3d gather)
          ↳ attn2 (Consistent): O = M_con^(v,r) · V_valid
       ③ CFG + DDIM_step
          z^(t-1)[I_key] ← key_edited   (매 스텝 키 뷰 고정)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
출력: {I_v^{edit}} = VAE_decode(z_v^(0))   ∀v
```

---

## 4. Phase 1 — 키 뷰 디노이징 및 Cross-Attention 수집

### 4.1 VAE 인코딩

$$\mathbf{z}_v = \mathcal{E}(I_v), \qquad \mathbf{c}_v = \mathcal{E}_{\text{mode}}(I_v^{\text{cond}})$$

IP2P의 3-way CFG를 위해 이미지 조건을 다음과 같이 구성한다.

$$\tilde{\mathbf{c}}_v = \bigl[\underbrace{\mathbf{c}_v}_{\text{text-cond}},\ \underbrace{\mathbf{c}_v}_{\text{img-cond}},\ \underbrace{\mathbf{0}}_{\text{uncond}}\bigr]$$

### 4.2 CrossAttentionStoreProcessor 설치

UNet의 모든 `attn2` 레이어에 `CrossAttentionStoreProcessor`를 설치한다. 이 processor는 표준 cross-attention 연산을 수행하면서, 유효 텍스트 토큰 집합 $\mathcal{T}$에 대한 attention 가중치를 spatial 해상도별로 버퍼에 저장한다.

- 유효 토큰 $\mathcal{T}$: BOS·EOS·패딩 토큰을 제외한 콘텐츠 토큰만 포함 (`_get_valid_token_indices`)
- 저장 대상 해상도: UNet이 실제로 사용하는 모든 해상도 ($32{\times}32$, $64{\times}64$ 등)

### 4.3 DDIM Forward (노이즈 추가)

$$\mathbf{z}_k^{(T)} = \sqrt{\bar{\alpha}_T}\,\mathbf{z}_k + \sqrt{1 - \bar{\alpha}_T}\,\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}), \quad k \in \mathcal{I}_{\text{key}}$$

### 4.4 키 뷰 DDIM 디노이징 루프

각 타임스텝 $t$마다 배치 크기 $3K$로 UNet forward를 수행한다.

**UNet 입력 구성 (채널 방향 concat):**

$$\text{input} = \bigl[\mathbf{z}_k^{(t)} \;\|\; \tilde{\mathbf{c}}_k\bigr] \in \mathbb{R}^{3K \times 8 \times H_l \times W_l}$$

**텍스트 임베딩 (3-way CFG):**

$$\mathbf{E} = \bigl[\tau_\theta^+(P),\ \tau_\theta^-(P),\ \tau_\theta^-(P)\bigr] \in \mathbb{R}^{3K \times 77 \times 768}$$

**UNet forward (`pivotal_pass=True`):**

$$\hat{\boldsymbol{\epsilon}}_k^{(t)} = \text{UNet}(\text{input},\ t,\ \mathbf{E})$$

이때 각 DGEBlock 내부에서는 §7.1(A)의 Extended ST-Attention이 실행된다. **Phase 1 키 뷰 루프에서는** `kf_attn_output`을 **저장하지 않는다** (타깃 단계에서 사용하지 않으므로 메모리 절약). Phase 3 Pivotal forward에서만 저장·갱신한다.

**CFG 결합:**

$$\hat{\boldsymbol{\epsilon}}_k = \boldsymbol{\epsilon}^{\text{unc}} + s_{\text{txt}}\bigl(\boldsymbol{\epsilon}^{\text{txt}} - \boldsymbol{\epsilon}^{\text{img}}\bigr) + s_{\text{img}}\bigl(\boldsymbol{\epsilon}^{\text{img}} - \boldsymbol{\epsilon}^{\text{unc}}\bigr)$$

**DDIM step:**

$$\mathbf{z}_k^{(t-1)} = \sqrt{\bar{\alpha}_{t-1}}\,\hat{\mathbf{z}}_k^{(0)} + \sqrt{1 - \bar{\alpha}_{t-1}}\,\hat{\boldsymbol{\epsilon}}_k$$

루프 종료 후: $\texttt{key\_edited} \leftarrow \{\mathbf{z}_k^{(0)}\}_{k \in \mathcal{I}_{\text{key}}}$

### 4.5 Cross-Attention Map 집계

`CrossAttentionStoreProcessor`가 버퍼에 저장한 attention 가중치를 타임스텝·head 차원으로 평균하여 키 뷰의 cross-attention map을 구성한다.

$$M_{\text{key}}^{(r)} = \frac{1}{T \cdot h} \sum_{t=0}^{T} \sum_{\text{head}} \mathbf{A}^{(t,r)}\bigl[\,:\,,\mathcal{T}\,\bigr] \in \mathbb{R}^{K \times L_r \times L_{\text{tok}}}$$

- $\mathbf{A}^{(t,r)} \in \mathbb{R}^{(h \cdot L_r) \times L_{\text{text}}}$: 타임스텝 $t$, 해상도 $r$의 cross-attention 가중치
- 배치의 positive 브랜치 ($0$번~$K$번 행) 슬라이스만 사용

이 map은 "텍스트 토큰 $c$가 키 뷰의 어느 공간 위치에 attention을 두는가"를 인코딩한다.

---

## 5. Phase 2 — 3D 기반 일관 Attention Map 구성

Phase 1에서 얻은 2D cross-attention map을 3DGS를 매개로 3D로 올렸다가 다시 각 뷰의 2D로 내려 **기하적으로 일관된(geometrically consistent) attention map**을 만든다.

### 5.1 역투영: 2D Cross-Attention → 3D Gaussian (`inverse_render_2d_to_3d`)

**입력:** $M_{\text{key}}^{(r)} \in \mathbb{R}^{K \times L_r \times L_{\text{tok}}}$

**출력:** $M_{\text{3D}}^{(r)} \in \mathbb{R}^{N_g \times L_{\text{tok}}}$

각 토큰 채널 $c$에 대해 키 뷰 카메라의 alpha-compositing 가중치를 이용해 가우시안별로 누적한다.

$$M_{\text{3D}}^{(r)}(g, c) = \frac{\displaystyle\sum_{k \in \mathcal{I}_{\text{key}}} w_k(g,\, c)}{\displaystyle\max\!\left(\sum_k \mathbf{1}[g \text{ 가 cam}_k\text{ 에 기여}],\; 1\right) + \epsilon}$$

$w_k(g,c)$: 키 뷰 $k$에서 가우시안 $g$가 채널 $c$의 attention map 픽셀에 기여하는 rasterizer alpha weight.

구현: `gaussian.apply_weights(cam_low, weights[:, c:c+1], weights_cnt, img_w)` 를 키 뷰·채널 루프로 호출한 뒤 `weights / (weights_cnt + ε)` 로 정규화.

**해석:** $M_{\text{3D}}^{(r)}(g,c)$는 "3D Gaussian $g$가 편집 토큰 $c$와 얼마나 의미적으로 연관되는가"를 나타내는 3D semantic attention 필드다.

### 5.2 재렌더링: 3D → 각 뷰의 2D Consistent Map (`render_consistent_maps`)

**입력:** $M_{\text{3D}}^{(r)} \in \mathbb{R}^{N_g \times L_{\text{tok}}}$

**출력:** $M_{\text{con}}^{(v,r)} \in \mathbb{R}^{H_r \times W_r \times L_{\text{tok}}}$, $\forall v \in \{0,\ldots,n-1\}$

각 뷰 $v$, 해상도 $r$, 토큰 채널 $c$에 대해:

$$M_{\text{con}}^{(v,r)}\bigl[\,:\,,c\,\bigr] = \text{GS\_Render}\!\left(G,\; \text{cam}_v^{(r)},\; \text{color} = M_{\text{3D}}^{(r)}\bigl[\,:\,,c\,\bigr]\right)$$

가우시안의 "색상"을 $M_{\text{3D}}^{(r)}[\,:\,,c\,]$로 설정하고 Gaussian splatting으로 렌더링한다.

**핵심 성질:**

- 모든 뷰가 동일한 $M_{\text{3D}}^{(r)}$를 공유하므로 **뷰 간 기하적 일관성이 보장**된다.
- 서로 다른 뷰 $v_1, v_2$에서 동일한 3D 위치 $g$를 가리키는 픽셀들은 동일한 attention 값을 갖는다.
- 키 뷰가 직접 관찰하지 못하는 뷰에도 3DGS의 기하 구조를 통해 attention이 전파된다.

---

## 6. Phase 3 — 타깃 뷰 디노이징

### 6.1 초기화

**ConsistentCrossAttnProcessor 설치:** 모든 `attn2`에 설치한다. 배치 처리 시 `_consistent_attn_map_current`에 설정된 $M_{\text{con}}^{(v,r)}$를 attention 가중치로 직접 사용한다.

**Latent 초기화:**

$$\mathbf{z}_v^{(T)} = \sqrt{\bar{\alpha}_T}\,\mathbf{z}_v + \sqrt{1-\bar{\alpha}_T}\,\boldsymbol{\epsilon}$$

$$\mathbf{z}^{(T)}\bigl[\mathcal{I}_{\text{key}}\bigr] \leftarrow \texttt{key\_edited}$$

`skip_key_views_in_target_loop=True` 일 때: $\mathcal{I}_{\text{targ}} = \{0,\ldots,n-1\} \setminus \mathcal{I}_{\text{key}}$

### 6.2 타임스텝별 연산

각 타임스텝 $t$에 대해 아래 세 단계를 순서대로 수행한다.

**① Pivotal Forward**

각 배치에서 무작위로 선택된 pivot 뷰 $\pi$에 대해 UNet을 1회 실행한다.

$$\hat{\boldsymbol{\epsilon}}_\pi^{(t)} = \text{UNet}\bigl([\mathbf{z}_\pi^{(t)};\,\mathbf{c}_\pi],\; t,\; \tau_\theta(P)\bigr) \quad \text{[pivotal\_pass=True]}$$

이 forward에서 각 DGEBlock의 `kf_attn_output`이 갱신된다. 이후 배치 forward에서 타깃 뷰들이 이 캐시를 feature injection 소스로 사용한다.

**② Batch Forward**

타깃 뷰를 `camera_batch_size` 단위의 배치로 나눠 처리한다. 배치 $b$, 뷰 집합 $\mathcal{B}_b$에 대해:

Consistent map을 각 `attn2`에 주입한 뒤 UNet forward를 실행한다 (CFG 3배치, `pivotal_pass=False`). DGEBlock 내부에서 §7의 DGE 연산이 수행된다.

**③ CFG 결합 + DDIM Step + 키 뷰 고정**

$$\hat{\boldsymbol{\epsilon}}_{\text{targ}} = \boldsymbol{\epsilon}^{\text{unc}} + s_{\text{txt}}\bigl(\boldsymbol{\epsilon}^{\text{txt}} - \boldsymbol{\epsilon}^{\text{img}}\bigr) + s_{\text{img}}\bigl(\boldsymbol{\epsilon}^{\text{img}} - \boldsymbol{\epsilon}^{\text{unc}}\bigr)$$

$$\mathbf{z}_{\text{targ}}^{(t-1)} = \text{DDIM\_step}\!\left(\hat{\boldsymbol{\epsilon}}_{\text{targ}},\; t,\; \mathbf{z}_{\text{targ}}^{(t)}\right)$$

$$\mathbf{z}^{(t-1)}\bigl[\mathcal{I}_{\text{key}}\bigr] \leftarrow \texttt{key\_edited} \quad \text{(매 스텝 끝에 키 뷰 고정)}$$

키 뷰는 Phase 1에서 이미 완성된 `key_edited`로 고정되어, Phase 3에서 다시 디노이징되지 않는다.

---

## 7. DGEBlock 내부 연산 순서

`make_dge_block`은 UNet의 각 `BasicTransformerBlock`을 **DGEBlock**으로 교체한다. IP2P와 동일한 블록 구조(norm1→attn1→residual→norm2→attn2→residual→norm3→FFN)를 유지하지만, **attn1의 역할을 완전히 대체**하고 **attn2는 processor만 교체**한다.

### DGEBlock 전체 도식

```
입력 hidden_states: (B, L, D),   B = 3 × n_frames

┌──────────────────────────────────────────────────────────────────┐
│ ① norm1 (LayerNorm)                                              │
│    norm_h = LayerNorm(hidden_states)    → (3, n_frames, L, D)    │
│                                                                  │
│    [pivotal_pass=True ]  → pivot_hidden_states = norm_h 저장     │
│    [pivotal_pass=False]  → 카메라 거리 계산                       │
│                            → 코사인 유사도 계산                   │
│                            → idx1 (/ idx2) 탐색 (에피폴라 미사용) │
└──────────────────────────┬───────────────────────────────────────┘
                           │
           ┌───────────────┴───────────────┐
           │ pivotal=True                  │ pivotal=False
           ▼                              ▼
  ┌──────────────────────┐      ┌────────────────────────────────┐
  │ [A] Extended         │      │ [B] DGE Feature Injection      │
  │     ST-Attention     │      │     (Q-K 연산 없음)             │
  │  키 뷰끼리 서로 attend│      │  코사인 유사도로 대응 픽셀 탐색 │
  │  → kf_attn_output    │      │  → kf_attn_output에서 gather    │
  │    저장 (pivotal만)  │      └────────────────────────────────┘
  └──────────────────────┘
           │                              │
           └───────────────┬──────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│ ② residual (attn1)                                               │
│    hidden_states = attn_output + hidden_states                   │
└──────────────────────────┬───────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│ ③ norm2 + Cross-Attention (attn2)                                │
│                                                                  │
│  [Phase 1] CrossAttentionStoreProcessor                          │
│    Q = W_q·H,  K = W_k·E_text,  V = W_v·E_text                  │
│    A = softmax(QK^T / √d)                                        │
│    O = A·V                                                       │
│    저장: A[:, :, T]  (해상도별 버퍼에 누적)                        │
│                                                                  │
│  [Phase 3] ConsistentCrossAttnProcessor                          │
│    O = M_con^(v,r) · V_valid     (Q-K 계산 없음)                 │
│                                                                  │
│  hidden_states = O + hidden_states                               │
└──────────────────────────┬───────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│ ④ norm3 + Feed-Forward                                           │
│    hidden_states = FFN(LayerNorm(hidden_states)) + hidden_states │
└──────────────────────────────────────────────────────────────────┘
```

### 7.1 [A] Pivotal Pass: Extended Spatio-Temporal Self-Attention

`register_extended_attention`에 의해 교체된 `sa_forward`가 실행된다. 이 연산이 "키 뷰(또는 pivot 뷰)들이 서로 attend"하는 **Spatio-Temporal Self-Attention**이다.

배치가 (text-cond, img-cond, uncond) 3개의 CFG 브랜치로 구성되므로, 각 브랜치 내에서만 독립적으로 attention을 수행한다.

브랜치 `txt`를 예로 들면 ($n_k$개 키 뷰):

**Q, K, V 투영:**

$$Q = W_q \mathbf{H}_{[:n_k]}, \quad K = W_k \mathbf{H}_{[:n_k]}, \quad V = W_v \mathbf{H}_{[:n_k]}$$

**Cross-view Key / Value 구성 (모든 뷰 concat):**

$$K^{\text{cat}} = \bigl[K_1 \;\|\; K_2 \;\|\; \cdots \;\|\; K_{n_k}\bigr] \in \mathbb{R}^{n_k \times (n_k L) \times d}$$

$$V^{\text{cat}} = \bigl[V_1 \;\|\; V_2 \;\|\; \cdots \;\|\; V_{n_k}\bigr] \in \mathbb{R}^{n_k \times (n_k L) \times d}$$

**head $j$별 Spatio-Temporal Attention:**

$$\text{sim}^{(j)} = \frac{Q^{(j)} \cdot \bigl(K^{\text{cat},(j)}\bigr)^\top}{\sqrt{d/h}} \in \mathbb{R}^{n_k \times L \times (n_k L)}$$

$$\Phi^{(j)} = \text{softmax}\!\left(\text{sim}^{(j)}\right) \cdot V^{\text{cat},(j)}$$

논문 표기로 요약하면:

$$\text{ST-Attn}(Q_t,\, K_{1:n_k}) = \text{softmax}\!\!\left(\frac{Q_t \cdot [K_1,\ldots,K_{n_k}]^\top}{\sqrt{d}}\right) \cdot [V_1,\ldots,V_{n_k}]$$

각 뷰 $t$의 query $Q_t$가 **모든 키 뷰의 key** $[K_1,\ldots,K_{n_k}]$에 attend하므로 진정한 의미의 cross-view attention이다.

**결과 캐시:**

$$\texttt{kf\_attn\_output} \leftarrow \text{attn\_output} \in \mathbb{R}^{3n_k \times L \times D}$$

이 캐시는 이후 non-pivotal pass에서 feature injection 소스로 사용된다.

### 7.2 [B] Non-Pivotal Pass: DGE Cross-View Feature Injection

`pivotal_pass=False`인 경우 self-attention Q-K-V 연산을 수행하지 않고, 캐시된 `kf_attn_output`과 대응 관계로 feature를 주입한다. **모드**에 따라 경로가 나뉜다.  
- **similarity**: 아래 (B-1)–(B-4)처럼 코사인 유사도로 idx1을 구한 뒤 `kf_attn_output`을 gather.  
- **3d_anchor (blend)**: 3DGS 기반 $t_j$, $F(v,p)$를 구하고 $\mathrm{attn\_output} = \lambda(F - h)$로 residual에 더함 ($(1-\lambda)h + \lambda F$).  
- **3d_anchor (gather)**: similarity와 동일 구조. pivot 맵 그대로 두고, 3D GS로 대응 픽셀 idx_3d(proj(pivot, j*))를 구한 뒤 gather → $h + \mathrm{attn\_output}$ (λ 없음).  
자세한 3d_anchor 수식은 `docs/edit_multiview_3d_anchor_algorithm.md` 및 `docs/edit_multiview_self_attn_output_similarity_vs_3d_anchor.md` 참조.

#### (B-1) 가장 가까운 Pivot 뷰 선택 (similarity 모드)

$$D_{\text{cam}}(v, k) = \|\mathbf{o}_v - \mathbf{o}_k\|_2$$

$$k^*(v) = \arg\min_{k \in \mathcal{I}_{\text{key}}} D_{\text{cam}}(v, k)$$

설정에 따라 1개 또는 2개의 nearest pivot을 선택한다 (`batch_idxs` 길이로 결정).

#### (B-2) 코사인 유사도 계산

타깃 뷰의 `norm_hidden_states` (positive CFG 브랜치, 인덱스 1) 와 pivot의 cached `pivot_hidden_states` 간 코사인 유사도를 계산한다.

$$\text{sim}_{v,p,q} = \frac{\tilde{\mathbf{H}}_v^{(p)} \cdot \tilde{\mathbf{H}}_{k^*(v)}^{(q)}}{\|\tilde{\mathbf{H}}_v^{(p)}\| \cdot \|\tilde{\mathbf{H}}_{k^*(v)}^{(q)}\|}$$

- $p \in \{0,\ldots,L{-}1\}$: 타깃 뷰의 spatial 위치
- $q \in \{0,\ldots,L{-}1\}$: pivot 뷰의 spatial 위치
- Shape: $(n_{\text{frames}},\; 1 \text{ or } 2,\; L,\; L)$

구현: `torch.einsum('bld,bcsd->bcls', norm_h[1], pivot_h)` 후 L2 정규화.

**참고:** 본 파이프라인(edit_latents_multiview)에서는 **에피폴라 제약을 사용하지 않는다.** DGEBlock에 `epipolar_constrains={}`가 설정되어 있어, 유사도만으로 대응 픽셀을 선택한다.

#### (B-3) 최적 대응 픽셀 탐색

$$j^*_p = \arg\max_{q}\;\text{sim}_{v,p,q}$$

코사인 유사도가 최대인 pivot 위치 $j^*_p$를 찾는다. (에피폴라 마스킹 없음.)

#### (B-4) Feature Gather (Injection)

**Pivot 1개 사용 시:**

$$\mathbf{O}_{\text{self}}^{(v)} = \texttt{kf\_attn\_output}\bigl[k^*(v),\; j^*_{0:L},\; :\bigr] \in \mathbb{R}^{L \times D}$$

**Pivot 2개 사용 시 (카메라 거리 기반 가중 혼합):**

$$w_1 = \sigma\!\left(\frac{D_{\text{cam}}(v, k_2^*)}{D_{\text{cam}}(v, k_1^*) + D_{\text{cam}}(v, k_2^*)}\right)$$

$$\mathbf{O}_{\text{self}}^{(v)} = w_1 \cdot \text{gather}\!\left(\texttt{kf\_attn},\; j^*_{1}\right) + (1-w_1) \cdot \text{gather}\!\left(\texttt{kf\_attn},\; j^*_{2}\right)$$

### 7.3 Residual (attn1)

$$\mathbf{H} \leftarrow \mathbf{O}_{\text{self}} + \mathbf{H}$$

### 7.4 Consistent Cross-Attention (attn2, Phase 3)

**표준 Cross-Attention (IP2P 기본, 참고용):**

$$\mathbf{A}^{\text{std}} = \text{softmax}\!\!\left(\frac{W_q\mathbf{H} \cdot (W_k\mathbf{E}_{\text{text}})^\top}{\sqrt{d}}\right), \quad \mathbf{O}^{\text{std}} = \mathbf{A}^{\text{std}} \cdot W_v\mathbf{E}_{\text{text}}$$

**Consistent Cross-Attention (본 방법):**

$$\mathbf{V}_{\text{valid}} = W_v\mathbf{E}_{\text{text}}\bigl[\,:\,,\mathcal{T}\,\bigr] \in \mathbb{R}^{L_{\text{tok}} \times D}$$

$$\mathbf{O}_{\text{cross}}^{(v)} = M_{\text{con}}^{(v,r)} \cdot \mathbf{V}_{\text{valid}} \in \mathbb{R}^{L_r \times D}$$

Q-K attention을 **완전히 생략**하고, 3D 기반으로 사전 계산된 $M_{\text{con}}^{(v,r)} \in \mathbb{R}^{L_r \times L_{\text{tok}}}$를 attention 가중치로 직접 사용한다.

**의미:** 타깃 뷰의 각 spatial position이 "어떤 텍스트 토큰의 feature를 얼마나 가져올지"를 3D 기하로 결정한다. UNet이 현재 denoising 상태에서 Q-K를 새로 계산하는 대신, 키 뷰에서 관찰된 semantic attention 구조를 3D로 일관되게 전파한다.

### 7.5 Residual + Feed-Forward

$$\mathbf{H} \leftarrow \mathbf{O}_{\text{cross}} + \mathbf{H}$$

$$\mathbf{H} \leftarrow \text{FFN}\!\left(\text{LayerNorm}(\mathbf{H})\right) + \mathbf{H}$$

---

## 8. Attention 연산 전체 흐름 요약

| 단계 | 패스 | Attention 레이어 | 연산 | 역할 |
|------|------|-----------------|------|------|
| Phase 1: 키 뷰 루프 | `pivotal_pass=True` | **attn1** (Extended ST-Attn) | 키 뷰 $K$개가 서로 attend; 전체 뷰 concat Key/Value 사용 → `kf_attn_output` **저장하지 않음** (타깃 단계 미사용) | Spatio-temporal feature 계산 |
| Phase 1: 키 뷰 루프 | `pivotal_pass=True` | **attn2** (StoreProcessor) | 표준 $\text{softmax}(QK^\top/\sqrt{d})V$; $\mathbf{A}[\,:\,,\mathcal{T}\,]$를 해상도별 저장 | Text-image 대응 관계 수집 |
| Phase 3: Pivotal Forward | `pivotal_pass=True` | **attn1** (Extended ST-Attn) | Pivot 뷰들끼리 ST-attention → `kf_attn_output` 갱신 | 타깃 루프용 feature 갱신 |
| Phase 3: Pivotal Forward | `pivotal_pass=True` | **attn2** (ConsistentProc) | $\mathbf{O} = M_{\text{con}}^{(v,r)} \cdot \mathbf{V}_{\text{valid}}$ | 일관 cross-attention |
| Phase 3: Batch Forward | `pivotal_pass=False` | **attn1** (DGE Injection) | Q-K 연산 없음; **similarity**: 유사도 idx1 gather. **3d_anchor**: blend면 $t_j$, $F$, $\lambda(F-h)$; gather면 3D GS idx_3d gather (에피폴라 미사용) | Cross-view feature injection |
| Phase 3: Batch Forward | `pivotal_pass=False` | **attn2** (ConsistentProc) | $\mathbf{O} = M_{\text{con}}^{(v,r)} \cdot \mathbf{V}_{\text{valid}}$; Q-K 생략 | 3D-일관 cross-attention |

---

## 9. 핵심 수식 목록

**(S1) Extended Spatio-Temporal Self-Attention (attn1, pivotal):**

$$\text{ST-Attn}(Q_t,\, K_{1:n_k}) = \text{softmax}\!\!\left(\frac{Q_t \cdot [K_1,\ldots,K_{n_k}]^\top}{\sqrt{d}}\right) \cdot [V_1,\ldots,V_{n_k}]$$

**(S2) Cross-Attention 저장 (attn2, Phase 1):**

$$M_{\text{key}}^{(r)} = \underset{t,\, \text{head}}{\operatorname{mean}} \;\text{softmax}\!\!\left(\frac{W_q\mathbf{H}\cdot(W_k\mathbf{E}_{\text{text}})^\top}{\sqrt{d}}\right)\!\bigl[\,:\,,\mathcal{T}\,\bigr]$$

**(S3) 역투영 (2D → 3D):**

$$M_{\text{3D}}^{(r)}(g, c) = \frac{\displaystyle\sum_{k \in \mathcal{I}_{\text{key}}} \text{apply\_weights}\!\left(G,\, \text{cam}_k^{(r)},\, M_{\text{key},k}^{(r)}[\,:\,,c\,]\right)}{\text{count}(g) + \epsilon}$$

**(S4) 재렌더링 (3D → 2D):**

$$M_{\text{con}}^{(v,r)}[\,:\,,c\,] = \text{GS\_Render}\!\left(G,\; \text{cam}_v^{(r)},\; \text{color} = M_{\text{3D}}^{(r)}[\,:\,,c\,]\right)$$

**(S5) DGE Feature Injection — 최적 대응 탐색 (attn1, non-pivotal, 에피폴라 미사용):**

$$j^*_p = \arg\max_{q}\;\text{sim}_{v,p,q}$$

$$\mathbf{O}_{\text{self}}^{(v)} = \texttt{kf\_attn\_output}\bigl[k^*(v),\; j^*_{0:L},\; :\bigr]$$

**(S6) Consistent Cross-Attention (attn2, Phase 3):**

$$\mathbf{O}_{\text{cross}}^{(v)} = M_{\text{con}}^{(v,r)} \cdot W_v\mathbf{E}_{\text{text}}\bigl[\,:\,,\mathcal{T}\,\bigr]$$

**(S7) Classifier-Free Guidance (IP2P):**

$$\hat{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}^{\text{unc}} + s_{\text{txt}}\bigl(\boldsymbol{\epsilon}^{\text{txt}} - \boldsymbol{\epsilon}^{\text{img}}\bigr) + s_{\text{img}}\bigl(\boldsymbol{\epsilon}^{\text{img}} - \boldsymbol{\epsilon}^{\text{unc}}\bigr)$$

---

## 10. 알고리즘 박스

```
Algorithm 1  edit_latents_multiview

Input : {I_v, I_v^c}_{v=0}^{n-1},  P,  G,  I_key,  T,  s_txt,  s_img
Output: {I_v^{edit}}_{v=0}^{n-1}

══════════════════════════════════════════════════════════════════
 VAE Encoding
══════════════════════════════════════════════════════════════════
 z_v ← VAE_encode(I_v)                         ∀v
 c_v ← VAE_encode_mode(I_v^c)                  ∀v

══════════════════════════════════════════════════════════════════
 Phase 1: Key-View Denoising + Cross-Attention Storing
══════════════════════════════════════════════════════════════════
 Install CrossAttentionStoreProcessor on all attn2
 For k ∈ I_key:
     z_k^(T) ← sqrt(ᾱ_T)·z_k + sqrt(1-ᾱ_T)·ε,   ε ~ N(0,I)
 For t = T, T-Δ, …, 0:
     register_pivotal = True
     ε_k ← UNet([z_k^(t) ‖ c_k], t, τ(P))     // 3K batch
       ▷ attn1: ST-Attn(Q_t, [K_1,…,K_K]) → kf_attn_output 저장하지 않음
       ▷ attn2: softmax(QK^T/√d)·V → A[:,:,T] 해상도별 저장
     ε_guided ← CFG(ε^txt, ε^img, ε^unc)
     z_k^(t-1) ← DDIM_step(ε_guided, t, z_k^(t))
 End
 key_edited ← {z_k^(0)}_{k∈I_key}
 M_key^r    ← mean_{t, head}(stored A[:,:,T])  // [K, L_r, L_tok]

══════════════════════════════════════════════════════════════════
 Phase 2: 3D-Consistent Attention Map Construction
══════════════════════════════════════════════════════════════════
 For each resolution r:
   For c = 0, …, L_tok-1:
     M_3D^r[:,c] ← apply_weights(G, key_cams, M_key^r[:,c])
                   / (count + ε)                // [N_g]
   For v = 0, …, n-1:
     For c = 0, …, L_tok-1:
       M_con^(v,r)[:,c] ← GS_Render(G, cam_v^r, M_3D^r[:,c])
                                                // [H_r, W_r]

══════════════════════════════════════════════════════════════════
 Phase 3: Target-View Denoising
══════════════════════════════════════════════════════════════════
 Install ConsistentCrossAttnProcessor on all attn2
 z_v^(T) ← add_noise(z_v, T)                   ∀v
 z^(T)[I_key] ← key_edited
 For t = T, T-Δ, …, 0:

   ── (i) Pivotal Forward ──────────────────────────────────────
   register_pivotal = True
   UNet([z_π^(t) ‖ c_π], t, τ(P))
     ▷ attn1: ST-Attn → kf_attn_output 갱신
     ▷ attn2: M_con^(v,r) · V_valid
   register_pivotal = False

   ── (ii) Batch Forward ───────────────────────────────────────
   For each batch b  (views B_b ⊆ I_targ):
     Set _consistent_attn_map_current ← M_con^(v,r)  for v ∈ B_b
     ε_b ← UNet([z_b^(t) ‖ c_b], t, τ(P))    // 3·|B_b| batch
       [DGEBlock – attn1, non-pivotal]
         1. k*(v) = argmin_k ‖o_v - o_k‖₂
         2. sim[v,p,q] = cosine(H_v[p], H_pivot[q])
         3. j*_p = argmax_q sim[v,p,q]        (에피폴라 미사용)
         4. O_self[v,p] = kf_attn_output[k*(v), j*_p, :]
         5. H ← O_self + H                     (residual)
       [DGEBlock – attn2, ConsistentCrossAttnProcessor]
         6. O_cross = M_con^(v,r) · V_valid
         7. H ← O_cross + H                    (residual)
       [DGEBlock – FFN]
         8. H ← FFN(LN(H)) + H
   End

   ── (iii) CFG + DDIM step ────────────────────────────────────
   ε_guided ← CFG(ε^txt, ε^img, ε^unc)
   z_targ^(t-1) ← DDIM_step(ε_guided, t, z_targ^(t))
   z^(t-1)[I_key] ← key_edited        // 키 뷰 매 스텝 고정
 End

══════════════════════════════════════════════════════════════════
 VAE Decoding
══════════════════════════════════════════════════════════════════
 I_v^{edit} ← VAE_decode(z_v^(0))     ∀v
```

---

## 11. IP2P 대비 변경점 정리

### 11.1 BasicTransformerBlock 구조 비교

| 단계 | IP2P (기본) | DGEBlock (본 방법) |
|------|-------------|-------------------|
| **norm1** | LayerNorm | 동일 + pivot 저장 또는 sim/idx 계산 추가 |
| **attn1** | Self-Attn (배치 내 공간 토큰끼리) | **완전 교체**: pivotal → Extended ST-Attn (뷰 간, kf_attn 저장); non-pivotal → Q-K 없이 **similarity**면 유사도 idx1 gather, **3d_anchor**면 blend($\lambda(F-h)$) 또는 gather(3D GS idx_3d) (Feature Injection, 에피폴라 미사용) |
| **residual** | attn1 + x | 동일 |
| **norm2** | LayerNorm | 동일 |
| **attn2** | Cross-Attn (공간↔텍스트) | 레이어 동일, **processor만 교체**: Phase 1 → StoreProcessor; Phase 3 → ConsistentProcessor |
| **residual** | attn2 + x | 동일 |
| **norm3 + FFN** | 표준 MLP | 동일 |

### 11.2 두 Attention의 역할 변화

```
         IP2P attn1                       DGE attn1
  ┌───────────────────────┐     ┌──────────────────────────────────────┐
  │ Self-Attention         │     │ [pivotal] Extended ST-Attention       │
  │ Q, K, V ← 같은 배치   │ →   │   → 뷰 간 cross-view attend           │
  │ 배치 내 공간 토큰끼리  │     │   → kf_attn_output 저장 (pivotal 시에만) │
  │ 단순 attend            │     │ [non-pivotal] Feature Injection       │
  └───────────────────────┘     │   → Q-K 연산 없음                     │
                                │   → 코사인 유사도로 idx 탐색 (에피폴라 미사용) │
                                │   → kf_attn에서 gather                │
                                └──────────────────────────────────────┘

         IP2P attn2                       DGE attn2
  ┌───────────────────────┐     ┌──────────────────────────────────────┐
  │ Cross-Attention        │     │ [Phase 1] CrossAttentionStoreProc    │
  │ Q ← 공간 hidden        │ →   │   → 표준 CA 동일하게 수행             │
  │ K, V ← E_text          │     │   → A[:,:,T] 해상도별 저장            │
  │                        │     │ [Phase 3] ConsistentCrossAttnProc    │
  │                        │     │   → Q-K 연산 생략                    │
  │                        │     │   → O = M_con · V_valid              │
  └───────────────────────┘     └──────────────────────────────────────┘
```

### 11.3 추가 연산 비용 분석

| 추가 단계 | 비용 원인 | 실측 비중 (예시) |
|-----------|-----------|-----------------|
| Phase 1: 키 뷰 DDIM 루프 | $T \times 1$회 UNet forward (키 뷰 $K$개 배치) | ~18% (`key_view_denoise_loop`) |
| Phase 2: 역투영 | $K \times L_{\text{tok}}$회 alpha-composite 연산 | <1% (`inverse_render_2d_to_3d`) |
| Phase 2: 재렌더링 | $n \times R \times L_{\text{tok}}$회 Gaussian splatting | ~2% (`render_consistent_maps`) |
| Phase 3: Pivotal forward | $T \times \lceil n / B \rceil$회 추가 UNet forward | ~7% (`pivotal_forward`) |
| Phase 3: DGE (유사도·gather) | 카메라 거리, 코사인 유사도, gather | ~3% (`per_timestep_setup` 등) |

Phase 3 배치 forward 자체 (`batch_forward`, ~22%)가 전체에서 가장 큰 비중을 차지하며, 이는 IP2P 단순 편집과 동일하게 필요한 비용이다.

---

## 12. 구현 파일 참조

| 기능 | 파일 | 함수 / 클래스 |
|------|------|--------------|
| 전체 파이프라인 진입점 | `threestudio/models/guidance/dge_guidance.py` | `DGEGuidance.edit_latents_multiview` |
| Cross-attention 저장 (Phase 1 attn2) | `dge_guidance.py` | `CrossAttentionStoreProcessor` |
| Consistent cross-attention (Phase 3 attn2) | `dge_guidance.py` | `ConsistentCrossAttnProcessor` |
| DGEBlock 생성 | `threestudio/utils/dge_utils.py` | `make_dge_block`, `DGEBlock.forward` |
| Extended ST-Attention (attn1 교체) | `dge_utils.py` | `register_extended_attention` → `sa_forward` |
| Normal attention 플래그 설정 | `dge_utils.py` | `register_normal_attn_flag` |
| Pivotal 플래그 등록 | `dge_utils.py` | `register_pivotal` |
| 카메라·배치 정보 등록 | `dge_utils.py` | `register_cams`, `register_batch_idx` |
| 2D → 3D 역투영 | `gaussiansplatting/scene/gaussian_model*.py` | `gaussian.apply_weights` |
| 3D → 2D 재렌더링 | `gaussiansplatting/gaussian_renderer/__init__.py` | `render` (= `gs_render`) |

---

*기준 코드: `dge_guidance.py` L519–824 (`edit_latents_multiview`; target 루프에서 `epipolar_constrains={}` 설정),
`dge_utils.py` (`make_dge_block`, `register_extended_attention`).*
