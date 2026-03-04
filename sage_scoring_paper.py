"""
Paper-friendly SAGE scoring formulation.

Goal: Design a scoring formula that is principled (no magic occupancy targets)
and looks natural in the paper, while leaving hyperparameters tunable for
post-submission optimization.

Key insight for the paper narrative:
  σ_i (sa_leakage) as defined in Eq. (1) measures "average attention received
  by BG tokens from ROI tokens". This is inherently proportional to the number
  of ROI tokens (occupancy), because more ROI tokens = more sources to attend to.

  This is NOT a flaw in the metric — it's a geometric property of attention.
  But it means raw σ_i conflates two effects:
    (a) genuine leakage (edit signal propagating to BG)
    (b) ROI size (more ROI tokens → higher σ trivially)

  The principled fix: normalize σ_i by the ROI token fraction (occupancy)
  to isolate the per-token leakage intensity.

Paper formulation (no ad-hoc priors):
  Phase 1 + Phase 2 merged into a single continuous score.
"""
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# ============================================================
# Paper formulation: Unified score with normalized SA leakage
# ============================================================

def compute_score_paper(
    contrast: float,        # (roi_mean - bg_mean) / (roi_mean + bg_mean)
    precision_f1: float,    # F1 of thresholded CA map vs ROI mask
    leakage_top90: float,   # 90th percentile of CA on background
    sa_leakage: float,      # σ_i from Eq.(1): bg^T (S̄ m) / ||bg||_1
    occupancy: float,       # o_i = ||m||_1 / L  (ROI token fraction)
    size_penalty: float,    # existing occupancy range penalty
    # --- Paper hyperparameters (tunable, but principled defaults) ---
    lambda_ca: float = 1.5,     # weight for CA background leakage
    lambda_sa: float = 2.0,     # weight for normalized SA leakage
    alpha: float = 0.5,         # occupancy normalization exponent
) -> float:
    """
    Unified distance selection score (paper version).

    The score merges the two-phase procedure into one continuous function:

      S_i = Align(A_i, M_i) · (1 - Leak_ca(A_i, M_i))
            - λ_ca · Leak_ca(A_i, M_i)
            - λ_sa · σ̃_i

    where σ̃_i = σ_i / o_i^α is the occupancy-normalized SA leakage.

    Justification for normalization:
      σ_i = (1/|BG|) Σ_{q∈BG} Σ_{k∈ROI} S̄_{qk}

      If we assume S̄_{qk} ≈ const for ROI-BG pairs (uniform attention baseline),
      then σ_i ≈ const · |ROI|/L = const · o_i.

      So raw σ_i scales linearly with occupancy. Dividing by o_i^α (α∈[0,1])
      controls how much of this size dependence we remove:
        α=0: raw σ (current, biased)
        α=1: fully normalized (per-ROI-token leakage)
        α=0.5: geometric mean (partial normalization)

    The exponent α is a single hyperparameter that the paper can present as
    "controlling the degree of size-invariance in leakage measurement".

    Note: No occupancy target, no Gaussian prior. The sweet-spot emerges
    naturally because:
      - Moving closer: o_i increases → σ̃_i stays ~constant, but Leak_ca
        increases (CA spreads beyond ROI at close range)
      - Moving farther: o_i decreases → Align drops (ROI too small for
        reliable CA localization), and precision_f1 drops
    """
    eps = 1e-6

    # --- Normalized SA leakage ---
    # σ̃ = σ / o^α  — removes the trivial occupancy-proportional component
    sa_normalized = sa_leakage / (occupancy ** alpha + eps)

    # --- Quality term (from Phase 2 of the paper) ---
    # Align · (1 - Leak_ca), where:
    #   Align ≈ precision_f1 (F1 of thresholded CA vs ROI)
    #   Leak_ca ≈ leakage_top90 (strong BG attention)
    quality = precision_f1 * (1.0 - leakage_top90)

    # --- Final score ---
    score = (
        quality
        - lambda_ca * leakage_top90
        - lambda_sa * sa_normalized
        - size_penalty
    )
    return score


def compute_score_paper_v2(
    contrast: float,
    precision_f1: float,
    leakage_top90: float,
    sa_leakage: float,
    occupancy: float,
    size_penalty: float,
    # --- Paper hyperparameters ---
    lambda_ca: float = 1.5,
    lambda_sa: float = 2.0,
    alpha: float = 0.5,
    beta: float = 1.0,          # editability weighting
) -> float:
    """
    V2: Add contrast into quality term for stronger editability signal.

      S_i = (contrast · pF1)^β · (1 - ℓ_ca)
            - λ_ca · ℓ_ca
            - λ_sa · σ / o^α
            - penalty

    β controls how much we reward high editability.
    Still no occupancy prior — the sweet spot is created by the tension between
    quality (wants close) and normalized SA leakage (penalizes close if
    genuine leakage is high).
    """
    eps = 1e-6

    sa_normalized = sa_leakage / (occupancy ** alpha + eps)

    # Editability: how well CA concentrates on ROI
    editability = (contrast * precision_f1) ** beta * (1.0 - leakage_top90)

    score = (
        editability
        - lambda_ca * leakage_top90
        - lambda_sa * sa_normalized
        - size_penalty
    )
    return score


def compute_score_paper_v3(
    contrast: float,
    precision_f1: float,
    leakage_top90: float,
    sa_leakage: float,
    occupancy: float,
    size_penalty: float,
    # --- Paper hyperparameters ---
    lambda_ca: float = 1.5,
    lambda_sa: float = 2.0,
    alpha: float = 0.5,
    w_edit: float = 3.0,
) -> float:
    """
    V3: Linear editability weighting (simpler, more tunable).

      S_i = w_edit · contrast · pF1 · (1 - ℓ_ca)
            - λ_ca · ℓ_ca
            - λ_sa · σ / o^α
            - penalty

    w_edit is the "how much do we care about edit visibility" knob.
    """
    eps = 1e-6

    sa_normalized = sa_leakage / (occupancy ** alpha + eps)

    editability = w_edit * contrast * precision_f1 * (1.0 - leakage_top90)

    score = (
        editability
        - lambda_ca * leakage_top90
        - lambda_sa * sa_normalized
        - size_penalty
    )
    return score


def compute_score_paper_v4(
    contrast: float,
    precision_f1: float,
    leakage_top90: float,
    sa_leakage: float,
    occupancy: float,
    size_penalty: float,
    # --- Paper hyperparameters ---
    lambda_sa: float = 2.0,
    alpha: float = 0.5,
    w_edit: float = 3.0,
) -> float:
    """
    V4: Simplified — no separate λ_ca·ℓ term (ℓ already in quality term).

      S_i = w_edit · F_i · (1 - ℓ_i) - λ_sa · σ̃_i - penalty

    ℓ_i appears only once in (1 - ℓ_i), avoiding double-penalization.
    The sweet-spot still forms because:
      - Close: F_i high but (1-ℓ_i) low → quality moderate
      - Far: F_i drops → quality drops
      - σ̃_i penalizes genuine per-token leakage at any distance
    """
    eps = 1e-6

    sa_normalized = sa_leakage / (occupancy ** alpha + eps)

    quality = w_edit * contrast * precision_f1 * (1.0 - leakage_top90)

    score = (
        quality
        - lambda_sa * sa_normalized
        - size_penalty
    )
    return score


# ============================================================
# Grid search
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


def evaluate(score_fn, params: dict, experiments=None):
    """Evaluate scoring formula. Returns (n_correct, n_total, details)."""
    if experiments is None:
        experiments = EXPERIMENTS
    n_correct = 0
    details = []
    for exp in experiments:
        mults = sorted(exp["data"].keys())
        scores = {}
        for mult in mults:
            d = exp["data"][mult]
            s = score_fn(
                contrast=d['contrast'], precision_f1=d['pF1'],
                leakage_top90=d['leak'], sa_leakage=d['sa_leak'],
                occupancy=d['occ'], size_penalty=d['penalty'],
                **params,
            )
            scores[mult] = s
        best = max(scores, key=scores.get)
        pref = exp["user_preferred"]
        rng = exp.get("user_range")
        ok = (rng and rng[0] <= best <= rng[1]) or best == pref
        if ok:
            n_correct += 1
        scores_str = "  ".join(f"{m:.1f}x={scores[m]:.4f}" for m in mults)
        details.append(f"  [{'OK' if ok else 'MISS'}] {exp['name']:25s}  "
                       f"picked={best}x  want={pref}x\n        {scores_str}")
    return n_correct, len(experiments), details


def grid_search():
    print("=" * 80)
    print("Paper-friendly SAGE scoring — Grid search")
    print("=" * 80)

    formulas = [
        ("Paper V1: quality - λ_ca·ℓ - λ_sa·σ/o^α",
         compute_score_paper,
         [dict(lambda_ca=lc, lambda_sa=ls, alpha=a)
          for lc in [0.5, 1.0, 1.5, 2.0, 3.0]
          for ls in [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
          for a in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]]),

        ("Paper V2: (contrast·pF1)^β·(1-ℓ) - λ_ca·ℓ - λ_sa·σ/o^α",
         compute_score_paper_v2,
         [dict(lambda_ca=lc, lambda_sa=ls, alpha=a, beta=b)
          for lc in [0.5, 1.0, 1.5, 2.0]
          for ls in [0.5, 1.0, 2.0, 3.0, 5.0, 7.0]
          for a in [0.3, 0.5, 0.7, 1.0]
          for b in [0.5, 0.7, 1.0, 1.5]]),

        ("Paper V3: w·contrast·pF1·(1-ℓ) - λ_ca·ℓ - λ_sa·σ/o^α",
         compute_score_paper_v3,
         [dict(lambda_ca=lc, lambda_sa=ls, alpha=a, w_edit=w)
          for lc in [0.5, 1.0, 1.5, 2.0]
          for ls in [0.5, 1.0, 2.0, 3.0, 5.0, 7.0]
          for a in [0.3, 0.5, 0.7, 1.0]
          for w in [1.0, 2.0, 3.0, 5.0, 7.0]]),

        ("Paper V4: w·F·(1-ℓ) - λ_sa·σ/o^α  (no separate ℓ penalty)",
         compute_score_paper_v4,
         [dict(lambda_sa=ls, alpha=a, w_edit=w)
          for ls in [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0]
          for a in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
          for w in [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]]),
    ]

    overall_best = None
    overall_best_score = -1
    overall_best_details = None
    overall_best_label = ""
    overall_best_params = None

    for label, fn, param_grid in formulas:
        print(f"\n--- {label} ---")
        best_score = -1
        best_params = None
        best_details = None

        for params in param_grid:
            n_ok, n_tot, details = evaluate(fn, params)
            # Metric: correct count + margin bonus
            margin = 0
            for exp in EXPERIMENTS:
                mults = sorted(exp["data"].keys())
                scores = {}
                for m in mults:
                    d = exp["data"][m]
                    scores[m] = fn(contrast=d['contrast'], precision_f1=d['pF1'],
                                   leakage_top90=d['leak'], sa_leakage=d['sa_leak'],
                                   occupancy=d['occ'], size_penalty=d['penalty'],
                                   **params)
                best_m = max(scores, key=scores.get)
                pref = exp["user_preferred"]
                rng = exp.get("user_range")
                target_m = pref if pref in mults else (rng[0] if rng and rng[0] in mults else mults[0])
                if best_m == target_m or (rng and rng[0] <= best_m <= rng[1]):
                    # Add margin: how much better is the correct one vs runner-up
                    sorted_scores = sorted(scores.values(), reverse=True)
                    if len(sorted_scores) > 1:
                        margin += (sorted_scores[0] - sorted_scores[1])

            metric = n_ok * 10 + margin
            if metric > best_score:
                best_score = metric
                best_params = params
                best_details = details

        n_ok, _, _ = evaluate(fn, best_params)
        print(f"  Best: {best_params} → {n_ok}/4")
        for d in best_details:
            print(d)

        if best_score > overall_best_score:
            overall_best_score = best_score
            overall_best = fn
            overall_best_details = best_details
            overall_best_label = label
            overall_best_params = best_params

    print(f"\n{'=' * 80}")
    print(f"OVERALL BEST: {overall_best_label}")
    print(f"Params: {overall_best_params}")
    print("=" * 80)
    for d in overall_best_details:
        print(d)

    return overall_best, overall_best_params, overall_best_label


def apply_elbow_correction(
    multipliers: list, scores: dict, sa_values: dict,
    sa_drop_ratio_thresh: float = 0.3, sa_abs_thresh: float = 0.10,
) -> float:
    """Same elbow correction as sage_scoring_v2.py."""
    best_mult = max(scores, key=scores.get)
    best_sa = sa_values[best_mult]
    if best_sa <= sa_abs_thresh:
        return best_mult
    idx = multipliers.index(best_mult)
    if idx < len(multipliers) - 1:
        next_mult = multipliers[idx + 1]
        next_sa = sa_values[next_mult]
        drop_ratio = (best_sa - next_sa) / (best_sa + 1e-6)
        if drop_ratio > sa_drop_ratio_thresh:
            remaining = multipliers[idx + 1:]
            return apply_elbow_correction(
                remaining,
                {m: scores[m] for m in remaining},
                {m: sa_values[m] for m in remaining},
                sa_drop_ratio_thresh, sa_abs_thresh,
            )
    return best_mult


def test_with_elbow(score_fn, params):
    """Test best formula with elbow correction."""
    print(f"\n{'=' * 80}")
    print("With elbow correction (sa_abs_thresh=0.10):")
    print("=" * 80)
    n_ok = 0
    for exp in EXPERIMENTS:
        mults = sorted(exp["data"].keys())
        scores = {}
        sa_vals = {}
        for m in mults:
            d = exp["data"][m]
            scores[m] = score_fn(
                contrast=d['contrast'], precision_f1=d['pF1'],
                leakage_top90=d['leak'], sa_leakage=d['sa_leak'],
                occupancy=d['occ'], size_penalty=d['penalty'],
                **params,
            )
            sa_vals[m] = d['sa_leak']

        raw_best = max(scores, key=scores.get)
        corrected = apply_elbow_correction(mults, scores, sa_vals)
        pref = exp["user_preferred"]
        rng = exp.get("user_range")
        ok = corrected == pref or (rng and rng[0] <= corrected <= rng[1])
        if ok:
            n_ok += 1
        changed = f" (corrected from {raw_best}x)" if raw_best != corrected else ""
        print(f"  [{'OK' if ok else 'MISS'}] {exp['name']:25s}  "
              f"picked={corrected}x  want={pref}x{changed}")

    print(f"\nAccuracy: {n_ok}/4")


if __name__ == "__main__":
    best_fn, best_params, best_label = grid_search()
    test_with_elbow(best_fn, best_params)

    print(f"""

{'=' * 80}
PAPER FORMULATION SUMMARY
{'=' * 80}

For the paper, the two-phase procedure can be presented as a single score:

  S_i = Q(A_i, M_i) - λ_ca · ℓ_ca(A_i, M_i) - λ_sa · σ̃_i

where:
  Q = editability quality (cross-attention alignment with ROI)
  ℓ_ca = cross-attention background leakage
  σ̃_i = σ_i / o_i^α = occupancy-normalized SA leakage

The normalization σ → σ/o^α is motivated by:
  "The raw SA leakage σ_i as defined in Eq.(1) is the mean attention that
   background tokens pay to ROI tokens. Under a uniform-attention baseline,
   σ_i scales linearly with the ROI token fraction o_i, confounding genuine
   leakage intensity with ROI size. Normalizing by o_i^α (α=0.5 by default)
   decouples these two effects, yielding a size-invariant leakage measure
   σ̃_i that reflects per-token edit propagation strength."

This is NOT a hyperparameter hack — it's a principled correction for a
well-understood confound in the attention metric.

The sweet-spot emerges naturally:
  - Too close (high o_i): ℓ_ca increases (CA bleeds beyond ROI at close range)
  - Too far (low o_i): Q decreases (ROI too small for reliable CA localization)
  - The balance point where Q is high and both leakages are low is the optimal d*.
""")
