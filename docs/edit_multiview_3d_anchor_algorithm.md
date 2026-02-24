# 3d_anchor Feature Injection 알고리즘

> **수식 렌더링**: 인라인 수식은 `$...$`, 블록 수식은 `$$...$$` (LaTeX). VS Code에서는 **Markdown All in One** 또는 **Markdown+Math** 확장, GitHub에서는 자동 렌더링.

`edit_latents_multiview`에서 `feature_injection_mode="3d_anchor"`일 때 사용하는, **3DGS prior 기반** feature injection 방식의 알고리즘 설명이다.  
3d_anchor는 **스타일**을 선택할 수 있다: `injection_3d_anchor_style`이 **`"blend"`** (기본)이면 $t_j$, $F(v,p)$, $\lambda(F-h)$ 혼합; **`"gather"`**이면 similarity와 같은 구조(pivot self-attn 그대로 + 3D GS remap gather + residual, λ 없음).  
(기존 `similarity` 방식은 2D–2D 코사인 유사도로 대응 픽셀을 찾고, 에피폴라 미사용.)

---

## 1. 핵심 아이디어

- **대응(γ)을 2D–2D가 아니라 3D anchor로 정의**
  - 각 뷰의 각 토큰(픽셀) $p$에 대해, “이 픽셀을 **어느 3D 가우시안(들)**이 설명하는가”를 **correspondence**로 둔다.
  - 즉, **γ = “다른 뷰의 픽셀”이 아니라 “3D 가우시안(또는 top-K 가우시안)”**이다.

- **TokenFlow 스타일 전파**
  - Pivot(또는 key) 뷰에서 얻은 토큰을 3D anchor를 따라 모든 뷰/위치로 전파한다.
  - **blend** 스타일: 주입을 **residual 친화적**으로 $h_{\mathrm{out}} = (1-\lambda)\, h + \lambda\, F(v,p)$.
  - **gather** 스타일: pivot self-attention 출력을 3D GS로 remap해 gather한 뒤 $h_{\mathrm{out}} = h + \mathrm{attn\_output}$ (완전 대체, λ 없음).

---

## 2. 사전 계산(캐시) — Target phase 시작 전 1회

`build_gaussian_provenance_cache`를 호출해 해상도별로 아래를 만든다.

### 2.1 픽셀 → 가우시안: $g(p)$

- **입력**: 3DGS $G$, 모든 뷰 카메라, 해상도 $(H,W)$, top-K(기본 2).
- **방법**: 래스터라이저가 per-pixel contributing gaussian id를 주지 않으므로, **proximity 기반**으로 근사한다.
  - 모든 가우시안을 해당 뷰에 투영해 2D 픽셀 좌표 $(u_g, v_g)$를 얻는다.
  - 각 픽셀 $p=(u,v)$에 대해, **투영이 그 픽셀에 가장 가까운 K개 가우시안**을 찾고,
  - **가중치**는 역거리(또는 거리 기반)로 정규화: $a_j(p)$.
- **출력** (해상도별):
  - `pix2g_id[v]`: 뷰 $v$의 각 픽셀(linear index)마다 **top-K 가우시안 ID** → shape `[n_views, H*W, K]`.
  - `pix2g_w[v]`: 같은 픽셀의 **가중치** $a$ → shape `[n_views, H*W, K]`.

즉, **$g(p) = \{(j_1, a_1), (j_2, a_2), \ldots\}$** 가 뷰·해상도·픽셀마다 정의된다.

### 2.2 가우시안 → 2D 위치: $\mathrm{proj}(\mathrm{view}, j)$

- **모든 뷰**에 대해: 가우시안 $j$의 중심을 그 뷰에 투영한 픽셀 $(u,v)$.
- **출력**:
  - `g2uv_all[(H,W)]`: `[n_views, N_g, 2]` — 뷰별로 각 가우시안의 (u,v).
  - (키 뷰만 쓰는) `g2uv`, `g_vis`: 키 뷰에서의 투영 + visibility(정면/가림 여부).

이렇게 하면 **다른 뷰 $u$에서 “같은 가우시안 $j$가 투영되는 픽셀”** $\mathrm{proj}(u, j)$를 바로 참조할 수 있다.

---

## 3. 타임스텝·배치별 동작 (Phase 3 target denoise 루프)

매 타임스텝 $t$에서:

1. **Pivotal forward**  
   현재 배치에서 정해진 **pivot 뷰 하나**에 대해 UNet을 한 번 돌려,  
   그 뷰의 **self-attention 출력**을 `kf_attn_output`에 저장한다.  
   shape: `[3, 1, L, D]` (CFG 3, pivot 1, spatial L, channel D).

2. **Batch forward**  
   타깃 뷰들을 배치로 UNet에 넣을 때, **DGEBlock의 attn1 자리**에서 3d_anchor injection을 한다.

---

## 4. DGEBlock 내 3d_anchor injection (해상도별)

해당 블록의 spatial size가 $(H,W)$ (즉 $L=H\times W$)일 때, **캐시에서** 이 해상도에 해당하는 것만 꺼낸다.

**스타일 분기**: `injection_3d_anchor_style`이 **`"gather"`**이면 아래 Step A–C 대신 **gather 경로**(§4.5)를 사용한다. **`"blend"`**이면 기존 Step A–C를 사용한다.

### Step A: Canonical token per gaussian — $t_j$ (blend 경로)

**의미**: 가우시안 $j$에 대응하는 “대표 토큰”을 **pivot 뷰에서** proj(pivot, $j$) 위치의 토큰으로 정의한다.

- **현재 구현 (단일 pivot)**  
  - `g2uv_all[(H,W)][pivot_view_index]` → pivot 뷰에서의 가우시안별 (u,v) → linear index.
  - `t_j = kf_attn_output[1, 0][linear, :]`  
    즉 pivot 뷰의 **positive CFG 브랜치** self-attention 출력에서, proj(pivot, $j$) 위치의 토큰.
  - shape: `t_j` → `[N_g, D]`.

- **다중 키 뷰 확장 시 (수식)**  
  - $t_j = \frac{\sum_{k \in \mathrm{keys}} w_{k,j}\, \phi_k(\mathrm{proj}(k,j))}{\sum_k w_{k,j}}$.  
  - $\phi_k(p)$: 키 뷰 $k$의 해당 블록 출력 토큰, $w_{k,j}$: visibility/신뢰도 등.

### Step B: 타깃 뷰 픽셀별 주입값 — $F(v,p)$ (blend 경로)

**의미**: 타깃 뷰 $v$의 픽셀 $p$에는, 그 픽셀이 “의미하는” 3D 가우시안들의 canonical token을 **가중 평균**으로 넣는다.

- $g(p)$가 위 캐시의 `pix2g_id[v,p,:]`, `pix2g_w[v,p,:]`에 해당.
- 수식:
  $$
  F(v,p) = \frac{\sum_{(j,a)\in g(p)} a \cdot t_j}{\sum_{(j,a)\in g(p)} a}.
  $$
- 구현:
  - `gathered = t_j[pix2g_id]` → `[n_frames, L, K, D]`,
  - `pix2g_w.unsqueeze(-1) * gathered` 후 K 차원으로 합하고,  
    `pix2g_w.sum(dim=-1, keepdim=True)`로 나누어 정규화 → `F_vp` `[n_frames, L, D]`.

### Step C: Residual 친화적 출력 (blend 경로)

- Self-attention 출력을 **완전 대체**하지 않고:
  $$
  h_{\mathrm{out}} = (1-\lambda)\, h + \lambda\, F(v,p),
  $$
  여기서 $h$는 **현재 블록 입력** `hidden_states`(attn1 들어가기 직전).
- 구현상으로는 **residual 연결**이 $h + \mathrm{attn\_output}$이므로,
  $$
  \mathrm{attn\_output} = \lambda\, (F - h)
  $$
  로 두면, $h + \mathrm{attn\_output} = (1-\lambda)h + \lambda F$가 된다.
- $\lambda$ = config `injection_lambda` (기본 0.5).

### Step D: gather 스타일 경로 (3d_anchor_style = "gather")

Similarity와 **동일한 구조**: pivot self-attention 맵을 그대로 쓰고, **대응만 3D GS**로 정한 뒤 gather → residual.

1. **Self-Attention 단계**  
   - Pivot의 `kf_attn_output`을 그대로 `self.attn_output`에 복사(expand)한다. (blend에서는 0 placeholder.)

2. **Feature Injection 단계**  
   - 타깃 위치 $p$마다 **top-1 가우시안** $j^* = \mathrm{argmax}_j a_j(p)$ (캐시의 `pix2g_id[:, :, 0]`).
   - Pivot에서의 대응 픽셀 = $\mathrm{proj}(\mathrm{pivot}, j^*)$ → `g2uv_all[pivot_view_index][j*]` → linear index **idx_3d** `[n_frames, L]`.
   - **Gather**: $\mathrm{attn\_output} = \mathrm{self.attn\_output}.gather(\mathrm{dim}=2,\ \mathrm{index}=\mathrm{idx\_3d})$ (similarity의 idx1 gather와 동일 방식).

3. **Residual**  
   - $\mathrm{hidden\_states} = \mathrm{hidden\_states} + \mathrm{attn\_output}$ (λ blend 없음).

---

## 5. 요약 흐름도

**blend 스타일:**

```
[캐시 구축 1회]
  픽셀 p → g(p) = {(j1,a1), (j2,a2)}   (pix2g_id, pix2g_w)
  가우시안 j → proj(view, j) = (u,v)   (g2uv_all)

[매 타임스텝·매 배치]
  Pivotal forward  →  kf_attn_output (pivot 뷰 1개 토큰맵)

  [각 DGEBlock, non-pivotal pass, blend]
    1. self.attn_output = 0 (placeholder)
    2. t_j = φ_pivot( proj(pivot, j) )     [N_g, D]
    3. F(v,p) = Σ a·t_j / Σa over g(p)    [n_frames, L, D]
    4. attn_output = λ (F - h)
    5. hidden_states = h + attn_output    → (1-λ)h + λF
```

**gather 스타일:**

```
  [각 DGEBlock, non-pivotal pass, gather]
    1. self.attn_output = kf_attn_output (pivot 맵 그대로 expand)
    2. j* = top-1 gaussian per pixel (pix2g_id[:,:,0])
    3. idx_3d = proj(pivot, j*) 의 linear index (g2uv_all)
    4. attn_output = self.attn_output.gather(dim=2, index=idx_3d)
    5. hidden_states = h + attn_output   (λ 없음)
```

---

## 6. 기호·설정 정리

| 기호/설정 | 의미 |
|-----------|------|
| $g(p)$ | 픽셀 $p$를 설명하는 top-K 가우시안 ID와 가중치 $\{(j,a)\}$. |
| $\mathrm{proj}(v, j)$ | 뷰 $v$에서 가우시안 $j$가 투영되는 2D 픽셀 (또는 linear index). |
| $t_j$ | 가우시안 $j$에 대한 canonical token (현재: pivot 뷰 proj(pivot,j) 토큰). |
| $F(v,p)$ | 타깃 뷰 $v$의 픽셀 $p$에 주입할 feature: $g(p)$에 대한 $t_j$의 가중 평균. |
| $\lambda$ | `injection_lambda`: residual 혼합 비율 (blend 스타일만). |
| K | 픽셀당 top-K 가우시안 수 (캐시 빌드 시, 기본 2). |
| `injection_3d_anchor_style` | `"blend"` (기본) 또는 `"gather"` (similarity와 동일 구조, 3D GS remap). |

---

## 7. similarity 방식과의 차이

| 항목 | similarity | 3d_anchor (blend) | 3d_anchor (gather) |
|------|------------|-------------------|---------------------|
| 대응 정의 | 2D–2D: 코사인 유사도 argmax | 3D: $g(p)$ 가중 평균 $F(v,p)$ | 3D: top-1 가우시안 → proj(pivot, j*) |
| 기하 | 유사도만 사용 | 3DGS $t_j$, $F(v,p)$ | 3DGS idx_3d |
| 주입 소스 | pivot `kf_attn_output` 1:1 gather | pivot에서 $t_j$ → $F(v,p)$ | pivot `kf_attn_output` 1:1 gather (idx_3d) |
| 출력 형태 | attn_output = gather, $h + \mathrm{attn}$ | attn_output = $\lambda$(F−h) → (1−$\lambda$)h + $\lambda$F | attn_output = gather, $h + \mathrm{attn}$ (λ 없음) |

- **gather** 스타일은 similarity와 **구조가 동일**하고, 대응 인덱스만 **유사도(idx1)** 대신 **3D GS(idx_3d)**를 쓴다.

이 문서는 현재 코드 기준이며, `dge_utils.py`의 DGEBlock 3d_anchor 분기와 `build_gaussian_provenance_cache`의 `pix2g_id`/`pix2g_w`/`g2uv_all` 반환 구조에 맞춰 작성되었다.
