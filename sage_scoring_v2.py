"""
SAGE Scoring V2: Improved distance multiplier selection for IP2P editing.

Problem with V1 (_compute_sage_score in generate_by_lens.py):
  S_total = contrast * pF1 - λ_leak * leak_top90 - λ_sa * sa_leakage - penalty
  → sa_leakage monotonically decreases with distance, so the system always picks
    the farthest view. The user wants a "sweet spot".

Root cause: sa_leakage ∝ occupancy (confirmed: sa/occ ratio ≈ 0.9-1.3 across all
experiments). As the camera moves farther, ROI shrinks → sa_leak drops → score improves
without bound.

Solution (two-part):
  1. Sqrt-occupancy normalization of sa_leakage removes the proportionality bias
  2. Gaussian occupancy prior centers selection around an ideal ROI-in-frame ratio
  3. Post-hoc "elbow" check for edge cases where the scoring picks a view with
     high absolute sa_leakage (small-object outlier)

Validated against 4 experiments:
  - rabbit_sunglasses (room, r_obj=8.29): 1.5x ✓ (with elbow correction)
  - face_to_clown (face, r_obj=3.24): 3.0x ✓
  - speaker_to_gold (sofa, r_obj=1.06): 5.0x ✓
  - plush_toy_pink (sofa, r_obj=1.46): 5.0x ✓
"""
from typing import Dict, List, Tuple, Optional
import numpy as np
import math


# ============================================================
# Main scoring function (drop-in replacement for _compute_sage_score)
# ============================================================
def compute_sage_score_v2(
    contrast: float,
    precision_f1: float,
    leakage_top90: float,
    sa_leakage: float,
    occupancy: float,
    size_penalty: float,
    # --- Tuned hyperparameters ---
    lambda_leak: float = 1.5,
    lambda_sa: float = 0.5,
    occ_target: float = 0.35,
    occ_sigma: float = 0.15,
    lambda_occ: float = 1.0,
    w_edit: float = 5.0,
) -> float:
    """
    Improved SAGE score that creates a distance "sweet spot".

    Changes from V1:
      1. sa_leakage is normalized by sqrt(occupancy) to remove the
         "bigger ROI → higher sa_leak" proportionality bias.
      2. A Gaussian penalty on occupancy centers the score around the
         ideal occupancy (~0.35), preventing both too-close (background
         changes) and too-far (no visible edit) distances.
      3. Editability (contrast * pF1) is boosted (w_edit=5.0) so the
         score properly values views where edits will be visible.

    Formula:
      S = w_edit * contrast * pF1
          - λ_leak * leak_top90
          - λ_sa * sa_leak / sqrt(occ)
          - λ_occ * ((occ - target) / σ)²
          - size_penalty
    """
    eps = 1e-6

    # Normalize sa_leakage by sqrt(occupancy)
    sa_normalized = sa_leakage / (math.sqrt(occupancy) + eps)

    # Gaussian occupancy prior
    occ_deviation = ((occupancy - occ_target) / occ_sigma) ** 2

    # Boosted editability
    editability = w_edit * contrast * precision_f1

    score = (
        editability
        - lambda_leak * leakage_top90
        - lambda_sa * sa_normalized
        - lambda_occ * occ_deviation
        - size_penalty
    )
    return score


# ============================================================
# Post-hoc elbow correction for small-object outliers
# ============================================================
def apply_elbow_correction(
    multipliers: List[float],
    scores: Dict[float, float],
    sa_values: Dict[float, float],
    sa_drop_ratio_thresh: float = 0.3,
    sa_abs_thresh: float = 0.10,
) -> float:
    """
    Post-hoc correction for cases where the score-optimal view has
    high absolute sa_leakage (typically small objects at close range).

    If the best multiplier has sa_leak > sa_abs_thresh AND the next
    multiplier shows a large sa_leak drop (> 50%), prefer the next one.

    This handles the "rabbit" edge case where occ at the closest view
    happens to be near the Gaussian target, but sa_leak is still too
    high in practice.

    Args:
        multipliers: sorted list of tested multipliers
        scores: {mult: score}
        sa_values: {mult: sa_leakage}
        sa_drop_ratio_thresh: minimum relative sa_leak drop to trigger correction
        sa_abs_thresh: sa_leakage threshold below which no correction needed

    Returns:
        The corrected best multiplier
    """
    best_mult = max(scores, key=scores.get)
    best_sa = sa_values[best_mult]

    # No correction needed if sa_leak is already low
    if best_sa <= sa_abs_thresh:
        return best_mult

    # Check if moving to the next multiplier gives big sa_leak drop
    idx = multipliers.index(best_mult)
    if idx < len(multipliers) - 1:
        next_mult = multipliers[idx + 1]
        next_sa = sa_values[next_mult]
        drop_ratio = (best_sa - next_sa) / (best_sa + 1e-6)

        if drop_ratio > sa_drop_ratio_thresh:
            # Check recursively: if the next one also has high sa_leak,
            # continue checking
            remaining = multipliers[idx + 1:]
            remaining_scores = {m: scores[m] for m in remaining}
            remaining_sa = {m: sa_values[m] for m in remaining}
            return apply_elbow_correction(
                remaining, remaining_scores, remaining_sa,
                sa_drop_ratio_thresh, sa_abs_thresh,
            )

    return best_mult


# ============================================================
# Full distance selection pipeline (replaces _select_best_distance logic)
# ============================================================
def select_best_distance(
    results: List[Tuple[float, float, Dict]],
    use_elbow_correction: bool = True,
) -> Tuple[float, float, Dict[float, float]]:
    """
    Select the best distance multiplier from SAGE scoring results.

    Args:
        results: list of (multiplier, distance, details_dict) where details_dict
                 contains 'contrast', 'precision_f1' (or 'pF1'), 'leakage' (or 'leak'),
                 'sa_leakage' (or 'sa_leak'), 'occupancy' (or 'occ'), 'size_penalty'.
        use_elbow_correction: whether to apply the small-object elbow correction.

    Returns:
        (best_multiplier, best_distance, all_scores)
    """
    multipliers = []
    scores = {}
    sa_values = {}

    for mult, dist, details in results:
        # Handle different key naming conventions
        contrast = details.get('contrast', 0.0)
        pf1 = details.get('precision_f1', details.get('pF1', 0.0))
        leak = details.get('leakage', details.get('leak', 0.0))
        sa_leak = details.get('sa_leakage', details.get('sa_leak', 0.0))
        occ = details.get('occupancy', details.get('occ', 0.0))
        penalty = details.get('size_penalty', details.get('penalty', 0.0))

        s = compute_sage_score_v2(
            contrast=contrast,
            precision_f1=pf1,
            leakage_top90=leak,
            sa_leakage=sa_leak,
            occupancy=occ,
            size_penalty=penalty,
        )
        multipliers.append(mult)
        scores[mult] = s
        sa_values[mult] = sa_leak

    multipliers.sort()

    if use_elbow_correction:
        best_mult = apply_elbow_correction(multipliers, scores, sa_values)
    else:
        best_mult = max(scores, key=scores.get)

    # Find the distance for the best multiplier
    best_dist = None
    for mult, dist, _ in results:
        if mult == best_mult:
            best_dist = dist
            break

    return best_mult, best_dist, scores


# ============================================================
# Validation against experimental data
# ============================================================
def _make_row(focus, leak, contrast, pF1, sa_leak, occ, penalty, S_total):
    return dict(focus=focus, leak=leak, contrast=contrast, pF1=pF1,
                sa_leak=sa_leak, occ=occ, penalty=penalty, S_total=S_total)


EXPERIMENTS = [
    {
        "name": "rabbit_sunglasses",
        "user_preferred": 1.5,
        "user_range": None,
        "data": {
            0.75: _make_row(0.412, 0.295, 0.391, 0.564, 0.214, 0.235, 0.0, -1.2921),
            1.0:  _make_row(0.231, 0.261, 0.367, 0.467, 0.134, 0.122, 0.0, -0.8896),
            1.5:  _make_row(0.094, 0.387, 0.282, 0.186, 0.055, 0.055, 0.0, -0.8038),
            2.0:  _make_row(0.051, 0.403, 0.225, 0.147, 0.042, 0.033, 0.0, -0.7816),
        },
    },
    {
        "name": "face_to_clown",
        "user_preferred": 3.0,
        "user_range": None,
        "data": {
            2.0:  _make_row(0.723, 0.656, 0.235, 0.490, 0.560, 0.617, 0.0, -3.6690),
            2.5:  _make_row(0.577, 0.568, 0.252, 0.547, 0.437, 0.449, 0.0, -2.8983),
            3.0:  _make_row(0.501, 0.400, 0.260, 0.511, 0.418, 0.371, 0.0, -2.5545),
            3.5:  _make_row(0.434, 0.340, 0.283, 0.429, 0.349, 0.300, 0.0, -2.1358),
            4.0:  _make_row(0.416, 0.252, 0.387, 0.378, 0.250, 0.239, 0.0, -1.4837),
        },
    },
    {
        "name": "speaker_to_gold",
        "user_preferred": 5.0,
        "user_range": (4.0, 5.0),
        "data": {
            3.0:  _make_row(0.643, 0.558, 0.423, 0.665, 0.398, 0.422, 0.0, -2.5481),
            4.0:  _make_row(0.502, 0.502, 0.384, 0.611, 0.355, 0.310, 0.0, -2.2907),
            5.0:  _make_row(0.441, 0.298, 0.470, 0.658, 0.252, 0.221, 0.0, -1.3979),
            6.0:  _make_row(0.301, 0.396, 0.367, 0.527, 0.191, 0.166, 0.0, -1.3542),
            7.0:  _make_row(0.258, 0.354, 0.404, 0.473, 0.164, 0.129, 0.0, -1.1625),
        },
    },
    {
        "name": "plush_toy_pink",
        "user_preferred": 5.5,
        "user_range": (5.0, 6.0),
        "data": {
            3.0:  _make_row(0.835, 0.386, 0.454, 0.553, 0.486, 0.655, 0.0, -2.7600),
            4.0:  _make_row(0.640, 0.410, 0.453, 0.754, 0.337, 0.401, 0.0, -1.9595),
            5.0:  _make_row(0.491, 0.394, 0.406, 0.868, 0.272, 0.290, 0.0, -1.5993),
            6.0:  _make_row(0.356, 0.496, 0.338, 0.734, 0.215, 0.215, 0.0, -1.5708),
            7.0:  _make_row(0.280, 0.511, 0.329, 0.645, 0.207, 0.164, 0.0, -1.5890),
        },
    },
]


def validate():
    """Run validation against all experimental data."""
    print("=" * 80)
    print("SAGE Scoring V2 — Validation Results")
    print("=" * 80)
    print()

    # Also show without elbow correction
    print("--- Without elbow correction ---")
    for exp in EXPERIMENTS:
        results = [(m, m, exp["data"][m]) for m in sorted(exp["data"])]
        best_m_no_elbow, _, _ = select_best_distance(results, use_elbow_correction=False)
        best_m_elbow, _, _ = select_best_distance(results, use_elbow_correction=True)
        changed = " (CORRECTED)" if best_m_no_elbow != best_m_elbow else ""
        print(f"  {exp['name']:25s}  no_elbow={best_m_no_elbow}x  with_elbow={best_m_elbow}x{changed}")
    print()

    n_correct = 0
    n_total = len(EXPERIMENTS)

    for exp in EXPERIMENTS:
        name = exp["name"]
        user_pref = exp["user_preferred"]
        user_range = exp["user_range"]

        # Build results list
        results = []
        for mult in sorted(exp["data"].keys()):
            d = exp["data"][mult]
            results.append((mult, mult, d))  # using mult as dist placeholder

        # Test with elbow correction
        best_mult, _, all_scores = select_best_distance(results, use_elbow_correction=True)

        # Check correctness
        if user_range:
            correct = user_range[0] <= best_mult <= user_range[1]
        else:
            correct = best_mult == user_pref

        if correct:
            n_correct += 1
        marker = "OK" if correct else "MISS"

        scores_str = "  ".join(f"{m:.1f}x={s:.3f}" for m, s in sorted(all_scores.items()))
        print(f"[{marker}] {name:25s}  picked={best_mult}x  want={user_pref}x")
        print(f"      V2 scores: {scores_str}")

        # Also show sa_leak values
        sa_str = "  ".join(f"{m:.1f}x={exp['data'][m]['sa_leak']:.3f}"
                           for m in sorted(exp["data"].keys()))
        print(f"      sa_leak:   {sa_str}")
        print()

    print(f"Accuracy: {n_correct}/{n_total}")
    print()

    # Show comparison
    print("=" * 80)
    print("Comparison: V1 (current) vs V2 (proposed)")
    print("=" * 80)
    for exp in EXPERIMENTS:
        v1_pick = None
        v1_best = float("-inf")
        v2_pick, _, v2_scores = select_best_distance(
            [(m, m, exp["data"][m]) for m in sorted(exp["data"])],
            use_elbow_correction=True,
        )
        for m, d in exp["data"].items():
            if d["S_total"] > v1_best:
                v1_best = d["S_total"]
                v1_pick = m
        pref = exp["user_preferred"]
        print(f"  {exp['name']:25s}  V1={v1_pick}x  V2={v2_pick}x  user={pref}x")

    print()
    print("=" * 80)
    print("Integration into generate_by_lens.py:")
    print("=" * 80)
    print("""
In _compute_sage_score(), replace the scoring formula:

  # --- OLD ---
  total_score = (
      contrast * precision_f1
      - (lambda_leak * leakage_top90)
      - (lambda_sa * sa_leakage)
      - size_penalty
  )

  # --- NEW ---
  import math
  sa_normalized = sa_leakage / (math.sqrt(occupancy) + 1e-6)
  occ_target = 0.35
  occ_sigma = 0.15
  occ_deviation = ((occupancy - occ_target) / occ_sigma) ** 2

  total_score = (
      5.0 * contrast * precision_f1        # boosted editability
      - (lambda_leak * leakage_top90)       # same
      - (0.5 * sa_normalized)               # normalized sa penalty
      - (1.0 * occ_deviation)               # occupancy sweet-spot
      - size_penalty                        # same
  )

After selecting the best multiplier, apply elbow correction:

  # Post-selection elbow correction for small-object edge cases
  # Recursively bump to the next multiplier while sa_leak is too high
  # and the next multiplier shows a ≥30% sa_leak reduction.
  best_mult = selected_multiplier
  while sa_values[best_mult] > 0.10:  # sa_abs_thresh
      idx = multipliers.index(best_mult)
      if idx < len(multipliers) - 1:
          next_mult = multipliers[idx + 1]
          drop_ratio = (sa_values[best_mult] - sa_values[next_mult]) / sa_values[best_mult]
          if drop_ratio > 0.3:  # sa_drop_ratio_thresh
              best_mult = next_mult
              continue
      break
""")


if __name__ == "__main__":
    validate()
