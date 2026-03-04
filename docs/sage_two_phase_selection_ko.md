## Attention-Guided Editing View Selection (AGEVS) — 기술 보고서

### 1. 개요

Diffusion 기반 3DGS 편집에서 편집 비용은 뷰의 수에 비례하여 증가한다. 편집 뷰 수를 최소화하면서 3D 일관성을 유지하기 위해, 본 연구에서는 (i) ROI(Region of Interest)에 대한 안정적이고 국소적인 편집을 보장하고, (ii) 기하학적으로 다양한 커버리지를 제공하는 컴팩트한 카메라 집합을 자동으로 선택하는 **Attention-Guided Editing View Selection (AGEVS)** 파이프라인을 제안한다.

AGEVS는 다음 다섯 단계로 구성된다:

1. **ROI Intrinsic Analysis**: ROI 포인트 클라우드의 가중 PCA 분석으로 중심, 주축, 특성 반경 추출
2. **SAGE Scale Probing**: Diffusion attention 기반 2-Phase 선택으로 최적 카메라 거리 결정
3. **Manifold Camera Sampling**: 최적 거리 구면/궤도상에 후보 카메라 균등 배치
4. **Energy-based Scoring**: 가시성과 기하 정렬에 기반한 후보 에너지 평가
5. **Diversity-aware Selection**: 에너지 가중 Farthest-Point Sampling으로 최종 다양 뷰 선택

---

### 2. Step 1: ROI Intrinsic Analysis

3D Gaussian Splatting으로 표현된 장면에서 ROI에 해당하는 가우시안 부분집합을 추출하고, opacity-가중 PCA를 수행하여 ROI의 기하학적 특성을 분석한다.

**가중 중심 및 공분산.** ROI 가우시안의 좌표 $\{\mathbf{x}_j\}_{j=1}^{J}$와 opacity $\{w_j\}$에 대해:

$$
\mathbf{c} = \frac{\sum_j w_j \mathbf{x}_j}{\sum_j w_j}, \qquad
\mathbf{\Sigma} = \frac{\sum_j w_j (\mathbf{x}_j - \mathbf{c})(\mathbf{x}_j - \mathbf{c})^\top}{\sum_j w_j}
$$

$\mathbf{\Sigma}$의 고유값 분해 $\mathbf{\Sigma} = \mathbf{V}\,\mathrm{diag}(\lambda_1, \lambda_2, \lambda_3)\,\mathbf{V}^\top$ ($\lambda_1 \geq \lambda_2 \geq \lambda_3$)로부터 주축 $\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3$을 얻는다. ROI의 특성 반경은 $r_{\text{obj}} = \sqrt{\lambda_1}$로 정의된다.

**전방 방향 벡터 $\mathbf{v}_{\text{front}}$.** COLMAP 카메라 중심들의 평균으로 정의한 장면 중심 $\mathbf{c}_{\text{scene}}$과 ROI 중심 $\mathbf{c}$ 사이의 벡터를 이용하여
$$
\mathbf{v}_{\text{front}} = \mathrm{normalize}(\mathbf{c}_{\text{scene}} - \mathbf{c})
$$
로 정의한다. 즉, ROI가 COLMAP 뷰들의 중심(평균)인 장면 중심을 향하도록 하는 방향을 전방으로 사용하며, COLMAP 카메라의 전방 벡터에는 더 이상 의존하지 않는다.

---

### 3. Step 2: Two-Phase SAGE Scale Probing

$N$개의 후보 거리 $d_i = m_i \cdot r_{\text{obj}}$ ($m_i \in \{1.5, 2.0, 2.5, 3.0, 3.5\}$)에서 뷰를 렌더링하고, InstructPix2Pix (IP2P) 파이프라인을 실행하여 diffusion attention 신호를 수집한 뒤, 2-Phase 선택으로 최적 거리를 결정한다.

#### 3.1 Phase 1: Self-Attention Containment Gate

IP2P denoising 과정에서 UNet self-attention 프로세서를 후킹하여 self-attention leakage $\sigma_i$를 측정한다. 가용한 가장 저해상도의 self-attention 맵(우선순위: $64 \to 256 \to 1024$ tokens)을 선택하여 모든 timestep·layer·head에 걸쳐 온라인 평균한 self-attention 행렬 $\bar{S} \in \mathbb{R}^{L \times L}$를 구성한다.

ROI 마스크를 동일 해상도로 다운샘플링하여 이진화한 벡터 $\mathbf{m} \in \{0,1\}^L$ (임계값 0.3)과 배경 벡터 $\bar{\mathbf{m}} = \mathbf{1} - \mathbf{m}$을 정의하고:

$$
\sigma_i = \frac{\bar{\mathbf{m}}^\top (\bar{S}\,\mathbf{m})}{\|\bar{\mathbf{m}}\|_1}
$$

이는 배경 픽셀들이 ROI 픽셀에 부여하는 평균 attention weight이다.

**Adaptive Filtering.** Per-run 평균 임계값 $\tau_{\text{sa}} = \frac{1}{N}\sum_{i=1}^{N}\sigma_i$를 설정하고, 안전 후보 집합을 $\mathcal{I}_{\text{safe}} = \{i \mid \sigma_i \leq \tau_{\text{sa}}\}$로 정의한다. $\mathcal{I}_{\text{safe}} = \emptyset$이면 전체 후보를 사용한다.

#### 3.2 Phase 2: Cross-Attention Quality Ranking

안전 후보에 대해, cross-attention 맵 $A_i$를 ROI 마스크 해상도로 보간·정규화한 뒤 두 가지 품질 지표를 계산한다.

**Thresholded Precision-Recall F1.** $A_i$의 상위 25% quantile로 이진화한 고활성 영역 $\hat{A}_i$와 ROI 마스크 $M_i$ 사이의 F1:

$$
P_i = \frac{\sum \hat{A}_i \odot M_i}{\sum \hat{A}_i + \epsilon}, \quad
R_i = \frac{\sum \hat{A}_i \odot M_i}{\sum M_i + \epsilon}, \quad
F_i = \frac{2P_i R_i}{P_i + R_i + \epsilon}
$$

**Background Cross-Attention Leakage.** 배경 픽셀에서의 attention 90th percentile: $\ell_i = Q_{0.9}(A_i[\bar{M}_i > 0.5])$

**최종 선택:**

$$
d^* = d_{i^*}, \quad i^* = \underset{i \in \mathcal{I}_{\text{safe}}}{\arg\max}\; F_i \cdot (1 - \ell_i)
$$

---

### 4. Step 3: Manifold Camera Sampling

최적 거리 $d^*$ 구면 위에 $K$개 후보 카메라를 균등 배치한다. 두 가지 모드를 지원한다.

**Circular Orbit 모드.** COLMAP 카메라 분포에 PCA를 적용하여 궤도 평면의 주축 $\mathbf{u}_1, \mathbf{u}_2$와 극축(polar axis, 최소 분산 방향)을 추출한다. $K$개의 등간격 각도 $\theta_k = 2\pi k / K$에서:

$$
\mathbf{d}_k = \cos\theta_k \cdot \mathbf{u}_1 + \sin\theta_k \cdot \mathbf{u}_2 + \alpha\sin\theta_k \cdot \mathbf{p}_{\text{polar}}
$$

여기서 $\alpha = 0.06$은 수직 미소 변조(vertical perturbation) 진폭이다. 카메라 위치는 $\mathbf{e}_k = \mathbf{c} + \mathbf{d}_k \cdot d^*$.

**Fibonacci Sphere + Cone 모드.** Golden angle spiral로 단위구에 $K$개 점을 균등 배치한 뒤, 전방 방향 $\mathbf{v}_{\text{front}}$로부터 반각 $\theta_{\text{cone}}$ 이내의 방향만 남긴다. 반구 필터가 활성화된 경우 world-up 방향 기준 하반구 점도 제거한다.

---

### 5. Step 4: Energy-based Scoring

각 후보 카메라에 대해 가시성과 기하 정렬을 결합한 에너지를 계산한다.

**ROI 가시성** $S_{\text{vis}}$: 3D ROI 마스크를 해당 뷰로 투영한 2D 마스크의 점유율.

**기하 후보도** $S_{\text{can}}$: 뷰 방향 $\hat{\mathbf{v}}$와 ROI 주축 간의 정렬:

$$
S_{\text{can}} = \max\bigl(|\hat{\mathbf{v}} \cdot \mathbf{v}_{\text{front}}|,\; 0.8 \cdot |\hat{\mathbf{v}} \cdot \mathbf{v}_2|\bigr)
$$

전방 뷰를 선호하되 측면 뷰에도 0.8 가중치를 부여한다. 최종 에너지:

$$
E_k = w_{\text{vis}} \cdot S_{\text{vis},k} + w_{\text{can}} \cdot S_{\text{can},k}
$$

기본값: $w_{\text{vis}} = 0.6$, $w_{\text{can}} = 0.4$.

---

### 6. Step 5: Diversity-aware Selection

에너지 상위 풀(pool, 전체의 $\rho$ 비율 이상, 기본 $\rho = 0.20$)에서 **에너지 가중 Farthest-Point Sampling (FPS)**으로 $n_{\text{select}}$개의 다양한 뷰를 선택한다.

1. 에너지가 가장 높은 카메라를 시드(seed)로 선택한다.
2. 매 반복마다, 이미 선택된 뷰들과의 **최소 각거리** $\delta_{\text{ang}}$, **최소 방위각 차이** $\delta_{\phi}$, **최소 앙각 차이** $\delta_y$를 계산한다.

$$
\text{diversity}(k) = \delta_{\text{ang}}(k) + w_\phi \cdot \delta_\phi(k) - w_y \cdot \delta_y(k)
$$

3. 정규화된 에너지 $\tilde{E}_k \in [0, 1]$과 가산 결합하여 최고 점수 카메라를 선택한다:

$$
k^* = \underset{k \in \text{remaining}}{\arg\max}\; \text{diversity}(k) + \tilde{E}_k
$$

4. 최종 선택된 카메라들은 방위각(azimuth) 순으로 정렬되어 일관된 뷰 순서를 보장한다.

---

### 7. 보조 지표 및 시각화

Phase 2의 $F_i$, $\ell_i$ 외에 디버깅용으로 다음 지표가 기록된다:

| 지표 | 정의 | 범위 |
|------|------|------|
| **Contrast** $C_i$ | $(\mu_{\text{roi}} - \mu_{\text{bg}}) / (\mu_{\text{roi}} + \mu_{\text{bg}} + \epsilon)$ | $[-1, 1]$ |
| **Focus** | $\sum A_i \odot M_i\; /\; (\sum A_i + \epsilon)$ | $[0, 1]$ |
| **Occupancy** $o_i$ | $\text{mean}(M_i)$ | $[0, 1]$ |
| **Size Penalty** | $\mathbb{1}[o_i < 0.02 \;\text{or}\; o_i > 0.70]$ | $\{0, 1\}$ |

진단 grid 이미지는 행(렌더링/해상도별 attention heatmap) $\times$ 열(거리 배수)로 구성되며, 각 셀에 SA leakage, Contrast, F1 값이 오버레이된다.

---

### 8. 특성

- **장면 적응성**: Phase 1의 per-run adaptive threshold가 ROI 크기와 장면 복잡도에 자동 적응. Step 3의 궤도 평면 추정이 COLMAP 카메라 분포에 자동 정렬.
- **하이퍼파라미터 의존성 감소**: SA 필터링은 상대적 임계값, Phase 2 품질 점수 $F_i(1-\ell_i)$는 가중치 없는 곱 형태. 에너지 및 다양성 결합도 가산 형태로 직관적.
- **효율성**: SA leakage 계산은 최저 해상도 선택 + 온라인 평균으로 메모리 효율적. IP2P 배치 모드로 다중 뷰 동시 처리 지원.
- **해석 가능성**: 5단계 파이프라인의 각 단계가 독립적 역할을 수행하며, 시각화 grid로 선택 근거를 직관적으로 확인 가능.

---

### 9. 한계 및 향후 과제

- Phase 1 필터링 임계값은 산술 평균에 기반하며, median/퍼센타일 기반 변형에 대한 ablation이 필요하다.
- 제안 방법은 attention 기반 proxy metric에 의존하며, denoising direction map 등 픽셀 도메인 metric과의 결합이 향후 과제이다.
- Energy scoring의 가시성-기하 가중치($w_{\text{vis}}, w_{\text{can}}$)와 다양성 가중치($w_\phi, w_y$)는 현재 수동 설정이며, 장면 유형별 적응적 조정이 가능할 것이다.


---
---

## LaTeX 본문 (ECCV 형식)

아래는 위 보고서의 내용을 ECCV 학회 제출용 LaTeX 본문으로 작성한 것이다.

---

```latex
\subsection{Attention-Guided Editing View Selection}
\label{sec:agevs}

A major bottleneck in diffusion-based 3DGS editing is that editing cost grows linearly with the number of views.
To minimize the number of edited views while preserving 3D consistency, we select a compact set of cameras that (i) yields stable, localized edits on the region of interest (ROI), and (ii) provides diverse geometric coverage.
We achieve this with \textbf{Attention-Guided Editing View Selection (AGEVS)}, which first determines an optimal camera distance for localized editing, and then constructs a diffusion-stable and diverse view set at that distance.

\noindent\textbf{ROI Geometric Analysis.}
Given a pre-trained 3DGS scene and a text-specified ROI segmented by LangSAM~\cite{langsam}, we perform opacity-weighted PCA on the ROI Gaussians to extract the centroid~$\mathbf{c}$, principal axes~$(\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3)$, eigenvalues~$(\lambda_1 \ge \lambda_2 \ge \lambda_3)$, and a characteristic radius~$r_{\mathrm{obj}} = \sqrt{\lambda_1}$.
A front-facing direction~$\mathbf{v}_{\mathrm{front}}$ is defined using only the global scene center $\mathbf{c}_{\mathrm{scene}}$ and the ROI centroid $\mathbf{c}$:
\begin{equation}
  \mathbf{v}_{\mathrm{front}} = \mathrm{normalize}\bigl(\mathbf{c}_{\mathrm{scene}} - \mathbf{c}\bigr),
\end{equation}
so that the ROI is considered to face toward the scene center without relying on COLMAP camera forward vectors.

\noindent\textbf{Two-Phase SAGE Scale Probing.}
We evaluate $N$ candidate distances $d_i = m_i \cdot r_{\mathrm{obj}}$ with multipliers $m_i \in \{1.5, 2.0, 2.5, 3.0, 3.5\}$.
For each candidate, we render the 3DGS view and run InstructPix2Pix~(IP2P) to collect both cross-attention and self-attention maps from the UNet denoising process.
View selection is then performed in two phases.

\textit{Phase~1: Self-Attention Containment Gate.}\;
We hook the UNet's self-attention layers and compute a leakage score~$\sigma_i$ that measures how strongly the editing signal propagates from ROI to background through self-attention pathways.
Let $\bar{S} \in \mathbb{R}^{L \times L}$ be the self-attention matrix averaged across all timesteps, layers, and heads at the coarsest available spatial resolution (\eg, $8{\times}8{=}64$ tokens).
Given the downsampled, binarized ROI vector $\mathbf{m} \in \{0,1\}^L$ and background vector $\bar{\mathbf{m}} = \mathbf{1} - \mathbf{m}$:
%
\begin{equation}
  \sigma_i \;=\; \frac{\bar{\mathbf{m}}^{\!\top}(\bar{S}\,\mathbf{m})}
                      {\lVert\bar{\mathbf{m}}\rVert_1},
  \label{eq:sa-leakage}
\end{equation}
%
which is the mean attention weight that background tokens assign to ROI tokens.
We set a per-run adaptive threshold $\tau_{\mathrm{sa}} = \frac{1}{N}\sum_i \sigma_i$ and retain only the safe set $\mathcal{I}_{\mathrm{safe}} = \{i \mid \sigma_i \le \tau_{\mathrm{sa}}\}$; if empty, all candidates are kept.
This relative filtering automatically absorbs scene-specific scale variations without requiring manual threshold tuning.

\textit{Phase~2: Cross-Attention Quality Ranking.}\;
For each surviving candidate, we interpolate the timestep- and head-averaged cross-attention map $A_i$ to the ROI mask resolution and compute two quality metrics.
The \emph{thresholded F1}~score $F_i$ measures overlap between the top-25\% attention activation region $\hat{A}_i = \mathbb{1}[A_i \ge Q_{0.75}(A_i)]$ and the ROI mask~$M_i$:
%
\begin{equation}
  P_i = \frac{\sum \hat{A}_i \odot M_i}{\sum \hat{A}_i + \epsilon},
  \quad
  R_i = \frac{\sum \hat{A}_i \odot M_i}{\sum M_i + \epsilon},
  \quad
  F_i = \frac{2P_iR_i}{P_i + R_i + \epsilon}.
  \label{eq:f1}
\end{equation}
%
The \emph{background cross-attention leakage}~$\ell_i = Q_{0.9}\!\bigl(A_i[\bar{M}_i{>}0.5]\bigr)$ captures the 90th-percentile attention intensity on background pixels, penalizing views where strong attention spills beyond the ROI.
The optimal editing distance is then selected by:
%
\begin{equation}
  d^{*} = d_{i^*}, \qquad
  i^{*} = \underset{i \,\in\, \mathcal{I}_{\mathrm{safe}}}{\arg\max}\;
          F_i \cdot (1 - \ell_i).
  \label{eq:view-selection}
\end{equation}
%
The product $F_i(1{-}\ell_i)$ jointly rewards high ROI attention coverage and penalizes background leakage.
Since Phase~1 has already filtered views with excessive self-attention spread, cross-attention quality alone provides a reliable ranking among the remaining candidates.

\noindent\textbf{Diverse View Set Construction.}
At the optimal distance~$d^*$, we generate $K$ candidate cameras uniformly distributed on a circular orbit whose plane is estimated via PCA on the COLMAP camera positions.
Each candidate is scored by a linear combination of ROI visibility~$S_{\mathrm{vis}}$ (fraction of the ROI projection visible) and geometric alignment~$S_{\mathrm{can}} = \max(|\hat{\mathbf{v}}{\cdot}\mathbf{v}_{\mathrm{front}}|,\; 0.8\,|\hat{\mathbf{v}}{\cdot}\mathbf{v}_2|)$, yielding $E_k = w_{\mathrm{vis}} S_{\mathrm{vis},k} + w_{\mathrm{can}} S_{\mathrm{can},k}$.
From the top-$\rho$ fraction of candidates by energy, we perform \emph{energy-weighted farthest-point sampling}~(FPS): starting from the highest-energy view, we iteratively select the candidate maximizing the sum of its angular diversity from already-selected views and its normalized energy.
This produces a compact, geometrically diverse camera set that balances ROI editability and multi-view coverage.
```
