# Self-Attention 출력 처리: Similarity vs 3d_anchor

타깃 뷰 배치에서 **attn1(self-attention)은 타깃 뷰에 대해 직접 수행되지 않고**, pivot 뷰의 self-attention 출력(`kf_attn_output`)을 이용해 **대응 관계**로 타깃 뷰에 넣어준다.  
두 방식은 이 “넣어줄 값”을 어떻게 만들고, residual에 어떻게 더하느냐에서 다르다.

---

## 공통 구조 (DGEBlock)

- **Pivotal pass**: pivot 뷰만 실제로 attn1을 돌리고, 그 결과를 `kf_attn_output`에 저장한다.  
  → 이게 “pivot 뷰의 **self-attention 출력**” $\phi_{\mathrm{pivot}}(\cdot)$.
- **Non-pivotal (타깃 배치)**: attn1을 타깃 뷰에 대해 돌리지 않고, 위 캐시와 대응 관계로 **attn_output**을 만든 뒤  
  `hidden_states = hidden_states + attn_output` 로 residual만 한다.

즉, “self-attention 출력을 어떻게 다룬다” = **타깃 뷰용으로 쓸 attn_output을 어떻게 정하고, residual에 어떻게 더하느냐**의 차이다.

---

## 1. Similarity 기반

**요점**: “대체”와 “gather”는 서로 다른 단계에서 나온다.  
1단계에서는 **pivot 전체 맵을 그대로 올려놓고**, 2단계에서 **유사도 idx1으로 그 맵을 재배치(gather)** 해서 최종 attn_output을 만든다. 그래서 충돌이 아니다.

### 흐름 개요 (입력·출력·대체·합산)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  [1단계] Self-Attention 단계 — “재료” 준비                                                 │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   INPUT                          동작 (대체)                      OUTPUT                 │
│   ─────                           ──────────                       ──────                 │
│                                                                                         │
│   kf_attn_output                  closest_cam으로 슬라이스           self.attn_output      │
│   (pivot의 attn1 출력)             → pivot “전체 맵”만 복사          (아직 위치별 대응 없음) │
│   [3, n_pivot, L, D]              (위치 p → pivot[p] 그대로)         [3, n_frames, L, D]   │
│                                                                                         │
│   의미: self.attn_output[f, p] = pivot의 위치 p 값 (타깃 위치 p에 “뭘 넣을지”는 아직 미정)   │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  [2단계] Feature Injection — 유사도로 “재배치(gather)”                                    │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   INPUT                              동작 (재배치)                   OUTPUT               │
│   ─────                              ────────────                   ──────               │
│                                                                                         │
│   self.attn_output                   타깃 norm ↔ pivot norm        attn_output          │
│   (1단계에서 준 “재료”)                 → 코사인 유사도 → idx1[p]        (최종 주입값)         │
│   + norm_hidden_states               “타깃 위치 p엔 pivot의 idx1[p]   [3, n_frames, L, D]  │
│   (타깃·pivot)                         위치 값을 넣자” → gather                          │
│                                                                                         │
│   attn_output[f, p] = self.attn_output[f, idx1[f,p], :]  ← pivot의 idx1[p]번째 값을 p에   │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  [3단계] Residual — 뭘 더하는지                                                           │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   INPUT (둘 다 그대로 사용)              연산                    OUTPUT                  │
│   ─────────────────────────             ────                    ──────                  │
│                                                                                         │
│   hidden_states  ──────────────┐                                                       │
│   (블록 입력 h, 타깃 뷰)         │         + (element-wise)        hidden_states (새 값)  │
│                                 ├──────────────→  h_new = h + attn_output               │
│   attn_output   ──────────────┘                                                       │
│   (2단계에서 만든 “넣을 값”)                                                              │
│                                                                                         │
│   타깃 위치 p에서:  h_new(p) = h(p) + φ_pivot(idx1(p))                                   │
│                    ↑        ↑                                                            │
│                    기존 값   “대체용”으로 쓸 값 = pivot self-attention 출력을 유사도로     │
│                             remap한 것 (타깃 attn1은 수행 안 함 → 완전히 pivot 값으로 대체)  │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

**한 줄 요약**:  
- **대체되는 것**: 타깃 뷰의 “self-attention 출력 역할”이 **pivot의 self-attention 출력을 유사도(idx1)로 재배치한 값**으로 **완전히 대체**된다.  
- **더해지는 것**: 그 대체값(`attn_output`)을 **입력 `hidden_states`에 그대로 더해서** `h_new = h + attn_output` 로 만든다.

### 1.1 Self-Attention 단계: “재료” 준비

- 타깃 뷰에 대해 **Q·K·V self-attention은 수행하지 않음**.
- `kf_attn_output`(pivot의 self-attention 출력)에서 **가장 가까운 pivot 뷰(closest_cam)** 것만 슬라이스해서 `self.attn_output`에 **그대로** 둔다.

  ```text
  self.attn_output = kf_attn_output.view(3, n_pivot, L, D)[:, closest_cam, :, :]   # [3, n_frames, L, D]
  ```

- 이때 **의미**는 다음과 같다.
  - `self.attn_output[b, f, p, :]` = (프레임 f에 대응하는) **pivot의 self-attention 출력 at 위치 p**.
  - 즉, pivot의 **전체 맵 [0..L-1]** 을 그대로 한 번 복사해 둔 것뿐이다.
  - 아직 “타깃 위치 p에는 pivot의 **어느** 위치를 넣을지”는 정해진 게 아니다.  
    → **위치별 remapping(대응)은 이 단계에서 하지 않는다.**

### 1.2 Feature Injection 단계: 유사도로 “재배치(gather)”

- 타깃 뷰의 **norm hidden**과 pivot의 **norm hidden**으로 **코사인 유사도**를 구해,  
  타깃 위치 $p$마다 “pivot에서 가장 비슷한 위치” **idx1[p]** 를 만든다.
- **최종 attn_output**은, 방금 올려둔 pivot 맵 `self.attn_output`에서 **idx1으로 gather**한 값이다.

  ```text
  attn_output[b, f, p, :] = self.attn_output[b, f, idx1[b,f,p], :]   # = pivot의 idx1[p] 위치 값
  ```

- 정리하면:
  - 1단계: pivot **전체 맵**을 메모리에 올려둠 → `self.attn_output[f, p] = pivot[p]` (인덱스 p 그대로).
  - 2단계: “타깃 위치 p에는 pivot의 **idx1[p]** 번째 값을 쓰자”고 **gather**로 재배치 →  
    `attn_output[f, p] = pivot[idx1[p]]`.
- 따라서 **“대체”**는 “타깃을 pivot 맵으로 통째로 덮어쓴다”가 아니라,  
  **“pivot 맵 전체를 재료로 올려둔 뒤, 그 재료를 유사도(idx1)에 맞게 고른다(gather)”** 라고 보면 된다.  
  대체와 gather가 충돌하는 게 아니라, **먼저 재료(pivot 맵) 준비 → 그다음 gather로 위치별로 고르는** 두 단계다.

### 1.2.1 idx1이 뭔가?

- **idx1**은 **타깃 뷰의 각 위치(픽셀) $p$에 대해, “pivot 뷰의 L개 위치 중 어느 위치가 가장 비슷한가”를 나타내는 인덱스**다.
- 구하는 방법 (코드):
  - 타깃의 norm hidden `norm_hidden_states[1]` (shape `[n_frames, L, D]`)와  
    pivot의 norm hidden `closest_cam_pivot_hidden_states` (shape `[n_frames, 1, L, D]`) 사이 **코사인 유사도**를 계산 → `sim` shape `[n_frames, L, L]` (타깃 위치 × pivot 위치).
  - 각 타깃 위치 $p$에 대해 `idx1[p] = argmax_q sim(p, q)` (유사도가 최대인 pivot 위치 $q$).
- 따라서:
  - **idx1** shape: `[3, n_frames, L]` (CFG 3브랜치 × 프레임 × L개 위치).
  - **의미**: `idx1[b, f, p]` = “타깃 (프레임 f, 위치 p)에 넣을 pivot self-attention 값은 pivot의 **그 인덱스** 위치 값이다”.

### 1.2.2 Gather 했을 때 비는 픽셀 있나? 순서만 바뀌나?

- **비는 픽셀은 없다.**  
  Gather 연산은 **타깃의 L개 위치 각각**에 대해, `self.attn_output[b, f, idx1[b,f,p], :]` 한 개씩을 **대입**한다.  
  → 타깃 위치가 L개이므로 출력도 L개 위치가 전부 채워진다. **빈 픽셀(빈 위치)은 생기지 않는다.**

- **“전체 순서/위치만 바뀌는” 것은 아니다.**  
  - “순서만 바뀐다” = 같은 L개 값이 재배치되는 것처럼 느껴지는데, 실제로는 **타깃 위치 1:1로 pivot 값 하나씩을 매핑**하는 것이다.
  - 타깃 위치 $p$ → pivot 위치 **idx1[p]** 값 하나를 가져와서 $p$에 쓴다.
  - 따라서:
    - **서로 다른 타깃 위치 $p_1, p_2$가 같은 pivot 위치를 가리킬 수 있다** (idx1[p1]=idx1[p2]) → 같은 pivot 값이 두 타깃 위치에 복사됨.
    - **어떤 pivot 위치는 아무 타깃 위치에서도 쓰이지 않을 수 있다** (어떤 $q$에 대해 idx1[p]=q인 $p$가 없음).
  - 즉, **L개 타깃 위치는 전부 값이 채워지고**, 그 값들은 pivot의 L개 값 중 **중복 허용해서** 고른 것이다. “순서만 바꾼 permutation”이 아니라 **타깃 위치마다 pivot의 한 위치를 (유사도로) 골라 넣는 것**이다.

### 1.3 Residual에서의 처리

- 그대로 표준 residual 한 번:

  ```text
  hidden_states = hidden_states + attn_output
  ```

- 수식으로 쓰면 (타깃 한 위치 $p$):

  $$h_{\mathrm{new}}(p) = h(p) + \phi_{\mathrm{pivot}}(\mathrm{idx1}(p))$$

- **정리**:  
  - Self-attention 출력 **역할**을 하는 것은 **오직 pivot의 self-attention 출력**을 similarity로 remap한 것.  
  - 타깃의 $h(p)$는 “더해지는 입력”으로만 쓰이고, “무엇을 넣을지” 결정에는 관여하지 않는다.

---

## 2. 3d_anchor

3d_anchor는 **두 가지 스타일**을 지원한다. 설정 `injection_3d_anchor_style`으로 선택한다.

- **`"blend"`** (기본): Self-attention 단계는 placeholder(0), feature injection에서 $t_j$, $F(v,p)$, $\lambda(F-h)$로 blend.
- **`"gather"`**: Similarity와 **동일한 구조** — pivot self-attention 맵 그대로 사용 → **3D GS mapping**으로 대응 픽셀(idx_3d) 계산 → gather → residual은 $h + \mathrm{attn\_output}$ (λ 없음).

아래 2.1–2.3은 **blend** 스타일, 2.4는 **gather** 스타일이다.

### 2.1 Self-Attention 단계에서 하는 일 (blend)

- 타깃 뷰에 대해 **attn1 연산 역시 수행하지 않음**.
- **blend** 스타일일 때: `self.attn_output`은 **placeholder**로 0으로 둔다. 실제로 쓰는 값은 feature_injection 단계에서 다시 계산한다.

  ```text
  self.attn_output = torch.zeros_like(norm_hidden_states.view(batch_size, sequence_length, dim))
  ```

### 2.2 Feature Injection 단계에서 하는 일 (blend)

- **Canonical token** $t_j$: 가우시안 $j$에 대해 pivot **self-attention 출력**에서 proj(pivot, $j$) 위치 값을 쓴다.  
  → $t_j = \phi_{\mathrm{pivot}}(\mathrm{proj}(\mathrm{pivot}, j))$ (현재 구현은 단일 pivot).
- **타깃 픽셀별 주입값** $F(v,p)$: 그 픽셀이 “의미하는” 3D 가우시안들 $g(p) = \{(j,a)\}$에 대해 가중 평균.

  $$F(v,p) = \frac{\sum_{(j,a)\in g(p)} a \cdot t_j}{\sum a}$$

- **blend** 스타일에서 **attn_output**은 residual 친화적으로:

  $$\mathrm{attn\_output} = \lambda\, (F - h)$$

  - $h$ = 현재 블록 **입력** `hidden_states` (attn1 들어가기 직전).
  - residual 후 **blend** $h_{\mathrm{new}} = (1-\lambda)h + \lambda F$가 되도록 한다.

### 2.3 Residual에서의 처리 (blend)

- 동일한 residual 식:

  ```text
  hidden_states = hidden_states + attn_output
  ```

- 수식으로:

  $$h_{\mathrm{new}} = h + \lambda(F - h) = (1-\lambda)\, h + \lambda\, F$$

- **정리**: **blend** 스타일에서는 $h$와 3D anchor 기반 $F$를 **비율 $\lambda$로 혼합**한다.

### 2.4 3d_anchor 스타일 "gather" (similarity와 동일 구조)

**gather** 스타일은 **대응 관계만 3D GS**로 정하고, self-attention·residual 처리 흐름은 **similarity와 동일**하다.

- **Self-Attention 단계**  
  - Pivot의 self-attention 출력을 **그대로** 가져와 `self.attn_output`에 둔다 (similarity의 1단계와 동일).  
  - `self.attn_output = kf_attn_output.view(3, n_pivot, L, D)[:, 0].unsqueeze(1).expand(3, n_frames, L, D)` (pivot 1개 기준).

- **Feature Injection 단계**  
  - **유사도 대신 3D GS mapping**으로 “타깃 위치 $p$에 대응하는 pivot 픽셀”을 정한다.  
    - 픽셀 $p$의 top-1 가우시안 $j^* = \mathrm{argmax}_j a_j(p)$ (또는 `pix2g_id[:, :, 0]`).  
    - Pivot에서의 대응 픽셀 = $\mathrm{proj}(\mathrm{pivot}, j^*)$ → `g2uv_all[pivot_view_index][j*]` → linear index **idx_3d[p]**.
  - **Gather** (similarity와 동일):  
    $\mathrm{attn\_output}[f,p,:] = \mathrm{self.attn\_output}[f,\, \mathrm{idx\_3d}[f,p],\, :]$.

- **Residual**  
  - **λ blend 없음**:  
    $\mathrm{hidden\_states} = \mathrm{hidden\_states} + \mathrm{attn\_output}$  
  - 타깃 hidden은 “더해지는 입력”만 담당하고, 넣을 값은 3D GS로 remap한 pivot self-attention 출력뿐이다.

**요약**: 구조는 similarity와 같고, **remap 인덱스만** cosine similarity(idx1) 대신 **3D GS 기반 idx_3d**를 쓴다.

---

## 3. 비교 요약

| 항목 | Similarity | 3d_anchor (blend) | 3d_anchor (gather) |
|------|------------|-------------------|---------------------|
| **타깃 뷰 attn1** | 수행 안 함 | 수행 안 함 | 수행 안 함 |
| **Self-attn 단계** | pivot 맵 그대로 복사 | placeholder 0 | pivot 맵 그대로 복사 |
| **대응 관계** | 코사인 유사도 idx1 | — | 3D GS top-1 → proj(pivot, j*) → idx_3d |
| **attn_output 내용** | pivot 맵을 **idx1로 gather**한 값 | $\lambda (F - h)$ | pivot 맵을 **idx_3d로 gather**한 값 |
| **현재 hidden $h$의 역할** | residual에서만 더해짐 | $\lambda(F-h)$로 blend에 사용 | residual에서만 더해짐 |
| **최종 residual** | $h + \phi_{\mathrm{pivot}}(\mathrm{idx1}(\cdot))$ | $(1-\lambda)h + \lambda F$ | $h + \phi_{\mathrm{pivot}}(\mathrm{idx\_3d}(\cdot))$ |
| **의미** | pivot self-attn으로 **완전 대체** 후 residual | $h$와 $F$를 **λ 혼합** | pivot self-attn을 3D GS로 remap해 **완전 대체** 후 residual |

---

## 4. 직관 정리

- **Similarity**:  
  “타깃 각 위치에, pivot self-attention에서 **가장 비슷한 위치** 값을 그대로 넣고, 입력에 더한다.”  
  → Self-attention 출력을 **완전히** pivot에서 가져온 값으로 대체.

- **3d_anchor (blend)**:  
  “타깃 각 위치에, 3D 가우시안 대응으로 만든 **$F$**를, 현재 입력 **$h$와 섞어서** 넣는다.”  
  → $(1-\lambda)h + \lambda F$ 형태로 residual-friendly 하게 혼합.

- **3d_anchor (gather)**:  
  “타깃 각 위치에, pivot self-attention에서 **3D GS로 대응한 pivot 픽셀** 값을 그대로 넣고, 입력에 더한다.”  
  → Similarity와 같은 “완전 대체 + residual”, 단 대응은 **유사도가 아니라 3D GS mapping**으로 정한다.

이 문서는 `dge_utils.py`의 DGEBlock (attn1 → feature_injection → residual) 흐름을 기준으로 작성되었다.
