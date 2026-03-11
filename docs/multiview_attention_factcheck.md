# 논문 초안 vs 코드 사실관계 검토

> 기준: `edit_latents_multiview_current.md`, `dge_guidance.py`, `dge_utils.py`

## 수정한 사실관계

| 원문 | 문제 | 수정 |
|------|------|------|
| "At each diffusion timestep t" | CA consistency는 `t >= per_step_t_start`(기본 500)일 때만 적용됨 | "At selected timesteps"로 변경 |
| "With cached" | 문장 미완성 | 삭제 |
| "optionally blending two nearest references" | multiview target loop 기본 경로에서는 epipolar 비활성, 단일 reference만 사용 | "optionally blending" 제거 |

## 코드와 일치하는 부분

- **Self-attention**: camera space에서 가장 가까운 reference 선택 → cosine similarity → argmax로 매칭 → residual injection ✓
- **Cross-attention**: reference view CA map → inverse splatting (2D→3D) → re-render (3D→2D) → target view에 적용 ✓
- **Adaptive reference**: early=canonical( frontal), late=least-used ✓
- **Reference view self-attention**: pivot view는 extended ST-attention 수행 후 `kf_attn_output` 캐시 ✓

## 의도적으로 생략한 구현 세부사항

- `per_step_t_start`, `target_batch_neighbor_threshold` 등 구체적 수치
- `ConsistentCrossAttnProcessor`, `_consistent_attn_map_current` 등 변수명
- warmup 구간 (t < 100에서 normal attention) — 논문에서 굳이 언급하지 않아도 됨
- 배치 전략 (adaptive vs fixed, sliding_window) — Adaptive Reference Selection으로 요약
