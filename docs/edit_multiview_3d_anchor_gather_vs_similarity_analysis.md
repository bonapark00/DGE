# 3d_anchor-gather가 similarity보다 consistency·선명도가 떨어지는 원인 분석

실험에서 **3d_anchor-gather**가 **similarity** 대비 consistency가 낮고 블러리하게 나오는 경우에 대한 가능 원인을 정리한다.

---

## 1. Many-to-one 과도한 수렴 (가장 유력)

**현상**: 3D GS에서는 “한 가우시안”이 2D에서 **많은 픽셀**에 대해 “투영이 가장 가까운 1등”이 될 수 있다.

- **pix2g_id**는 픽셀마다 “그 뷰에서 **투영 위치(2D)가 가장 가까운** K개 가우시안”으로 정렬된다 (거리 기반 top-K, `d2.topk(..., largest=False)`).
- **gather**에서는 **top-1만** 쓰므로, 타깃 픽셀 `p` → `j* = pix2g_id[:,:,0]` → pivot 픽셀 = `proj(pivot, j*)` **한 곳**으로만 매핑된다.
- 따라서 **서로 다른 많은 타깃 픽셀**이 **같은 pivot 픽셀**로 모일 수 있다 (같은 `j*` → 같은 `proj(pivot, j*)`).
- 결과: 여러 타깃 위치가 **동일한** pivot self-attention 값을 복사받음 → 공간적 변이가 줄어들어 **블러/단조로움**처럼 보이고, 뷰마다 “같은 3D 점”이 다른 pivot 픽셀과 연결되면 **일관성**도 떨어진다.

**similarity와의 차이**  
- similarity는 **타깃 픽셀마다** “지금 hidden과 가장 비슷한 **pivot 위치**”를 **feature 기준**으로 따로 고른다.  
- 같은 3D 표면이라도 픽셀별로 다른 pivot 위치를 선택할 수 있어, 공간적 변이가 더 잘 유지된다.

**정리**: 3d_anchor-gather는 “기하적으로 같은 3D 점 → pivot에서 한 픽셀”로 많이 수렴하기 때문에, **many-to-one이 과도**해져 블러/일관성 저하로 이어진다.

---

## 2. Pivot 뷰에서 visibility 미검사

**현상**: pivot 뷰에서 가우시안 `j*`가 **가려져 있거나 뒤에 있는지** 전혀 검사하지 않고, `proj(pivot, j*)`에서 무조건 gather한다.

- **g_vis**는 캐시에서 **key view**에 대해서만 계산된다 (`build_gaussian_provenance_cache`의 `g_vis_cache`는 `key_cam_indices` 기준).
- **pivot**은 target view 중 하나로 **랜덤** 선택되므로, key가 아닌 경우 pivot view에 대한 **g_vis가 없다**.
- 따라서 gather 시 `g2uv_all[pivot_view_index][j*]`만 사용하고, “pivot 뷰에서 j*가 실제로 보이는지”는 사용하지 않는다.
- **결과**:  
  - pivot 뷰에서 **가려진** 가우시안의 2D 위치에서 feature를 가져오면, 그 픽셀에는 **다른 표면**이 그려져 있을 수 있음.  
  - 그 값을 타깃에 복사하면 **기하와 맞지 않는 feature**가 들어가 **일관성·선명도**가 깨진다.

**가능한 대응**  
- pivot view에 대해서도 visibility를 계산해 두고, `j*`가 pivot에서 보이지 않으면 gather를 스킵하거나 fallback(예: 해당 픽셀은 타깃 hidden 그대로 또는 blend로 넘김)하는 방식이 필요하다.

---

## 3. Top-1만 사용하는 경직성 (K=2인데 1개만 사용)

**현상**: 캐시는 **K=2**로 만들어지지만, gather 경로에서는 **top-1 가우시안만** 사용한다.

- 픽셀 경계·반투명·겹침 구간에서는 1등과 2등 가우시안의 기여가 비슷할 수 있다.
- 이때 top-1만 쓰면:
  - 작은 변화(노이즈, 타임스텝)에 따라 `j*`가 바뀌어 **flickering** 또는 불안정한 대응이 생기거나,
  - “실제로는 두 가우시안이 섞인 픽셀”을 한 가우시안에만 강하게 묶어 **잘못된 대응**이 될 수 있다.
- **blend** 경로는 `pix2g_w`로 **가중 평균**을 하므로 이런 구간에서 더 안정적이다.

**정리**: top-1만 쓰는 것이 **블러/일관성 저하의 직접 원인**이라기보다는, many-to-one + visibility와 겹칠 때 **불안정성·잘못된 대응**을 키울 수 있다.

---

## 4. 기하 대응 vs 의미 대응

**현상**: 3d_anchor는 “**같은 3D 점**”으로 대응을 정하고, similarity는 “**지금 feature가 가장 비슷한 2D 위치**”로 대응을 정한다.

- 3D GS 대응: “타깃 픽셀 p → 3D 가우시안 j* → pivot에서 proj(pivot, j*)”  
  → **기하적으로는** 같은 3D 점을 가리키지만, pivot의 그 2D 위치에서의 self-attention이 타깃 픽셀의 **시각/의미**와 항상 맞는 것은 아니다 (시점 차이, 가림, 텍스처 왜곡 등).
- similarity: “타깃 픽셀의 norm hidden과 가장 유사한 pivot 픽셀”  
  → **의미·외관** 기준으로 맞춤이라, 편집/텍스처 관점에서는 더 “맞는” 값을 가져올 수 있다.

**정리**: 3d_anchor-gather가 “기하적으로는 맞지만 의미적으로는 어긋난” pivot 값을 많이 가져오면, **일관성·선명도**가 similarity보다 나쁠 수 있다.

---

## 5. 요약 및 개선 방향

| 원인 | 설명 | 개선 방향 (참고) |
|------|------|------------------|
| **Many-to-one 과다** | 같은 j* → 같은 pivot 픽셀로 많은 타깃 픽셀이 수렴 → 블러/단조로움 | top-K 가중 gather(soft gather), 또는 3D 대응이 과도하게 겹치는 구간만 blend/fallback |
| **Pivot visibility 미사용** | pivot 뷰에서 j*가 가려져 있어도 proj(pivot,j*)에서 gather → 잘못된 feature | pivot view용 g_vis 계산 후, 비가시 픽셀은 fallback(타깃 hidden 유지 또는 blend) |
| **Top-1 경직** | K=2인데 1개만 사용 → 경계/겹침에서 불안정·잘못 대응 | top-K 가중 평균으로 soft gather (blend와 비슷한 아이디어) |
| **기하 vs 의미** | 3D 기하는 맞지만 pivot 2D feature가 타깃과 의미적으로 안 맞을 수 있음 | 의미 보정(예: similarity와의 혼합), 또는 pivot 선택/가중치 재검토 |

**실제 구현 우선순위 제안**  
1. **Pivot visibility**: pivot view에 대해 g_vis(또는 동등한 visibility)를 구해, 비가시인 경우 해당 픽셀만 fallback 처리.  
2. **Soft gather**: top-1 대신 `pix2g_w`로 top-K 가중 평균 gather (blend처럼 한 픽셀당 하나의 스칼라 값이 아니라, pivot 여러 위치에서 가중 평균한 feature를 쓰는 형태)로 many-to-one 완화.

이 문서는 `dge_utils.py`의 3d_anchor-gather 경로 및 `build_gaussian_provenance_cache` 동작을 기준으로 작성되었다.
