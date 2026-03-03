"""
Analysis of SA-leakage scoring and distance multiplier selection.

Goal: Find a better scoring formula so the system automatically picks the
distance multiplier the user prefers (the "sweet spot" where edits are
localised but still visible).

Key insight from experimental data:
  - Current S_total monotonically improves as distance increases (sa_leak drops),
    so the system always picks the farthest multiplier.
  - The user's preferred multiplier corresponds to a specific occupancy range
    (the ROI covers ~20-35% of the image). Too close → background changes;
    too far → no visible edit.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


# ============================================================
# 1.  Raw experimental data extracted from terminal logs
# ============================================================
@dataclass
class Experiment:
    name: str
    scene: str
    seg_prompt: str
    r_obj: float                          # sqrt(lambda_max) of ROI
    # Per-multiplier data: {mult: (focus, leak, contrast, pF1, sa_leak, occ, penalty, S_total)}
    data: Dict[float, Dict[str, float]]
    system_pick: float                    # multiplier chosen by current scoring
    user_preferred: float                 # multiplier the user wants
    user_preferred_range: Optional[Tuple[float, float]] = None  # e.g., (4.0, 5.0)


def _make_row(focus, leak, contrast, pF1, sa_leak, occ, penalty, S_total):
    return dict(focus=focus, leak=leak, contrast=contrast, pF1=pF1,
                sa_leak=sa_leak, occ=occ, penalty=penalty, S_total=S_total)


experiments = [
    # Case 1: Rabbit sunglasses  (room scene, small ROI object)
    Experiment(
        name="rabbit_sunglasses",
        scene="room",
        seg_prompt="face of the grey rabbit figure",
        r_obj=8.2934,
        data={
            0.75: _make_row(0.412, 0.295, 0.391, 0.564, 0.214, 0.235, 0.0, -1.2921),
            1.0:  _make_row(0.231, 0.261, 0.367, 0.467, 0.134, 0.122, 0.0, -0.8896),
            1.5:  _make_row(0.094, 0.387, 0.282, 0.186, 0.055, 0.055, 0.0, -0.8038),
            2.0:  _make_row(0.051, 0.403, 0.225, 0.147, 0.042, 0.033, 0.0, -0.7816),
        },
        system_pick=2.0,
        user_preferred=1.5,
    ),
    # Case 2: Face → Clown  (face scene, large ROI)
    Experiment(
        name="face_to_clown",
        scene="face",
        seg_prompt="a face",
        r_obj=3.2378,   # d/mult: 6.4756/2.0 = 3.2378
        data={
            2.0:  _make_row(0.723, 0.656, 0.235, 0.490, 0.560, 0.617, 0.0, -3.6690),
            2.5:  _make_row(0.577, 0.568, 0.252, 0.547, 0.437, 0.449, 0.0, -2.8983),
            3.0:  _make_row(0.501, 0.400, 0.260, 0.511, 0.418, 0.371, 0.0, -2.5545),
            3.5:  _make_row(0.434, 0.340, 0.283, 0.429, 0.349, 0.300, 0.0, -2.1358),
            4.0:  _make_row(0.416, 0.252, 0.387, 0.378, 0.250, 0.239, 0.0, -1.4837),
        },
        system_pick=4.0,
        user_preferred=3.0,
    ),
    # Case 3: Speaker → Gold  (blue_sofa scene)
    Experiment(
        name="speaker_to_gold",
        scene="blue_sofa",
        seg_prompt="speaker",
        r_obj=1.0637,   # d/mult: 3.1909/3.0 ≈ 1.0636
        data={
            3.0:  _make_row(0.643, 0.558, 0.423, 0.665, 0.398, 0.422, 0.0, -2.5481),
            4.0:  _make_row(0.502, 0.502, 0.384, 0.611, 0.355, 0.310, 0.0, -2.2907),
            5.0:  _make_row(0.441, 0.298, 0.470, 0.658, 0.252, 0.221, 0.0, -1.3979),
            6.0:  _make_row(0.301, 0.396, 0.367, 0.527, 0.191, 0.166, 0.0, -1.3542),
            7.0:  _make_row(0.258, 0.354, 0.404, 0.473, 0.164, 0.129, 0.0, -1.1625),
        },
        system_pick=7.0,
        user_preferred=5.0,
        user_preferred_range=(4.0, 5.0),
    ),
    # Case 4: Plush toy → Pink  (blue_sofa scene)
    Experiment(
        name="plush_toy_pink",
        scene="blue_sofa",
        seg_prompt="plush toy",
        r_obj=1.4629,   # d/mult: 4.3888/3.0 ≈ 1.4629
        data={
            3.0:  _make_row(0.835, 0.386, 0.454, 0.553, 0.486, 0.655, 0.0, -2.7600),
            4.0:  _make_row(0.640, 0.410, 0.453, 0.754, 0.337, 0.401, 0.0, -1.9595),
            5.0:  _make_row(0.491, 0.394, 0.406, 0.868, 0.272, 0.290, 0.0, -1.5993),
            6.0:  _make_row(0.356, 0.496, 0.338, 0.734, 0.215, 0.215, 0.0, -1.5708),
            7.0:  _make_row(0.280, 0.511, 0.329, 0.645, 0.207, 0.164, 0.0, -1.5890),
        },
        system_pick=6.0,
        user_preferred=5.5,  # user says 5.0-6.0 both OK
        user_preferred_range=(5.0, 6.0),
    ),
]


# ============================================================
# 2.  Diagnostic analysis
# ============================================================
def print_analysis():
    print("=" * 80)
    print("ANALYSIS: Relationship between occupancy, sa_leakage, and user preference")
    print("=" * 80)

    for exp in experiments:
        print(f"\n--- {exp.name} (r_obj={exp.r_obj:.4f}) ---")
        print(f"  System picks: {exp.system_pick}x | User wants: {exp.user_preferred}x")
        print(f"  {'mult':>5s}  {'occ':>6s}  {'sa_leak':>7s}  {'contrast':>8s}  {'pF1':>6s}  {'focus':>6s}  {'S_total':>8s}")
        for mult in sorted(exp.data):
            d = exp.data[mult]
            marker = " ← USER" if mult == exp.user_preferred else ""
            if exp.user_preferred_range and exp.user_preferred_range[0] <= mult <= exp.user_preferred_range[1]:
                marker = " ← USER"
            print(f"  {mult:5.1f}  {d['occ']:6.3f}  {d['sa_leak']:7.3f}  {d['contrast']:8.3f}  {d['pF1']:6.3f}  {d['focus']:6.3f}  {d['S_total']:8.4f}{marker}")

    # User-preferred occupancy range
    print("\n" + "=" * 80)
    print("User-preferred occupancy values:")
    for exp in experiments:
        pref = exp.user_preferred
        if pref in exp.data:
            occ = exp.data[pref]['occ']
            sa = exp.data[pref]['sa_leak']
        else:
            # interpolate for range
            lo, hi = exp.user_preferred_range or (pref, pref)
            occ = np.mean([exp.data[m]['occ'] for m in [lo, hi] if m in exp.data])
            sa = np.mean([exp.data[m]['sa_leak'] for m in [lo, hi] if m in exp.data])
        print(f"  {exp.name:25s}  mult={pref:.1f}x  occ={occ:.3f}  sa_leak={sa:.3f}")


# ============================================================
# 3.  Proposed new scoring: occupancy-aware with sweet-spot bonus
# ============================================================
def compute_new_score(
    contrast: float,
    precision_f1: float,
    leakage_top90: float,
    sa_leakage: float,
    occupancy: float,
    size_penalty: float,
    # --- New parameters ---
    occ_target: float = 0.25,
    occ_sigma: float = 0.15,
    lambda_leak: float = 1.5,
    lambda_sa: float = 5.0,
    lambda_occ: float = 3.0,     # weight for occupancy sweet-spot term
    min_occ: float = 0.03,       # below this, always penalize (too far)
) -> float:
    """
    New score = contrast * pF1
                - lambda_leak * leak_top90
                - lambda_sa * sa_leakage
                - lambda_occ * occ_deviation   # Gaussian penalty for being away from target occ
                - size_penalty

    The key addition: a Gaussian penalty centred on the target occupancy.
    This prevents the score from monotonically improving as distance increases,
    creating a "sweet spot" where the ROI occupies the ideal fraction of the frame.
    """
    # Occupancy deviation: Gaussian penalty
    occ_deviation = ((occupancy - occ_target) / occ_sigma) ** 2

    # Additional penalty for very low occupancy (object too far to edit)
    low_occ_penalty = 0.0
    if occupancy < min_occ:
        low_occ_penalty = 2.0

    score = (
        contrast * precision_f1
        - lambda_leak * leakage_top90
        - lambda_sa * sa_leakage
        - lambda_occ * occ_deviation
        - size_penalty
        - low_occ_penalty
    )
    return score


def compute_score_v2(
    contrast: float,
    precision_f1: float,
    leakage_top90: float,
    sa_leakage: float,
    occupancy: float,
    size_penalty: float,
    # --- Parameters ---
    occ_target: float = 0.25,
    occ_sigma: float = 0.18,
    lambda_leak: float = 1.5,
    lambda_sa: float = 3.0,        # reduced from 5.0
    lambda_occ: float = 2.5,
    w_editability: float = 2.0,    # bonus weight for editability (contrast * pF1)
) -> float:
    """
    V2: Normalize sa_leakage by occupancy to remove the "bigger ROI → higher sa_leak" bias.

    sa_leakage_normalized = sa_leakage / (occupancy + eps)

    This makes the penalty scale-invariant: a 30% ROI with sa_leak=0.3 is penalized
    the same as a 5% ROI with sa_leak=0.05.

    Then use a Gaussian occupancy prior to prefer a sweet-spot occupancy.
    """
    eps = 1e-6

    # Normalize sa_leakage by occupancy
    sa_norm = sa_leakage / (occupancy + eps)

    # Occupancy sweet-spot (Gaussian)
    occ_penalty = ((occupancy - occ_target) / occ_sigma) ** 2

    # Editability: boosted contrast*pF1
    editability = w_editability * contrast * precision_f1

    score = (
        editability
        - lambda_leak * leakage_top90
        - lambda_sa * sa_norm
        - lambda_occ * occ_penalty
        - size_penalty
    )
    return score


def compute_score_v3(
    contrast: float,
    precision_f1: float,
    leakage_top90: float,
    sa_leakage: float,
    occupancy: float,
    size_penalty: float,
    # --- Parameters ---
    occ_target: float = 0.28,
    occ_sigma: float = 0.15,
    lambda_leak: float = 1.0,
    lambda_sa: float = 2.0,
    lambda_occ: float = 3.0,
    w_edit: float = 3.0,
) -> float:
    """
    V3: Log-occupancy based normalization of sa_leakage.

    Key ideas:
      1. sa_leak ∝ occupancy approximately, so normalize by occ to get "per-pixel leakage"
      2. Use log(occ) to make the sweet-spot more symmetric in log-space
         (since multiplier effects are multiplicative on occ)
      3. Boost editability reward (contrast * pF1) since the user wants
         edits to actually show up
    """
    eps = 1e-6

    # Per-pixel SA leakage: how much each BG pixel attends to ROI, normalized
    sa_per_pixel = sa_leakage / (occupancy + eps)

    # Log-occupancy sweet spot
    log_occ = np.log(occupancy + eps)
    log_target = np.log(occ_target)
    log_dev = ((log_occ - log_target) / 0.8) ** 2  # sigma in log-space

    editability = w_edit * contrast * precision_f1

    score = (
        editability
        - lambda_leak * leakage_top90
        - lambda_sa * sa_per_pixel
        - lambda_occ * log_dev
        - size_penalty
    )
    return score


# ============================================================
# 4.  Grid search for best hyperparameters
# ============================================================
def evaluate_formula(score_fn, params: dict) -> Tuple[int, int, float, List[str]]:
    """
    Evaluate a scoring formula across all experiments.

    Returns:
        (n_correct, n_total, avg_rank_error, details)
    """
    n_correct = 0
    n_total = len(experiments)
    rank_errors = []
    details = []

    for exp in experiments:
        mults = sorted(exp.data.keys())
        scores = {}
        for mult in mults:
            d = exp.data[mult]
            s = score_fn(
                contrast=d['contrast'],
                precision_f1=d['pF1'],
                leakage_top90=d['leak'],
                sa_leakage=d['sa_leak'],
                occupancy=d['occ'],
                size_penalty=d['penalty'],
                **params,
            )
            scores[mult] = s

        # Find best multiplier
        best_mult = max(scores, key=scores.get)

        # Check correctness
        if exp.user_preferred_range:
            correct = exp.user_preferred_range[0] <= best_mult <= exp.user_preferred_range[1]
        else:
            correct = best_mult == exp.user_preferred
        if correct:
            n_correct += 1

        # Rank error: how far is the best from the user's preference?
        rank_error = abs(mults.index(best_mult) - mults.index(exp.user_preferred if exp.user_preferred in mults else
                         (exp.user_preferred_range[0] if exp.user_preferred_range and exp.user_preferred_range[0] in mults else mults[0])))
        rank_errors.append(rank_error)

        scores_str = ", ".join(f"{m}x={scores[m]:.4f}" for m in mults)
        marker = "OK" if correct else "MISS"
        details.append(
            f"  [{marker}] {exp.name:25s}  picked={best_mult}x  want={exp.user_preferred}x  ({scores_str})"
        )

    return n_correct, n_total, np.mean(rank_errors), details


def grid_search():
    """
    Grid search over hyperparameters for each scoring formula.
    """
    print("\n" + "=" * 80)
    print("GRID SEARCH: Finding best hyperparameters")
    print("=" * 80)

    best_overall = None
    best_overall_score = -1
    best_overall_details = None
    best_overall_label = ""

    # --- V1: Gaussian occupancy penalty ---
    print("\n--- V1: Gaussian occupancy sweet-spot ---")
    for occ_target in [0.15, 0.20, 0.25, 0.30, 0.35]:
        for occ_sigma in [0.10, 0.15, 0.20, 0.25]:
            for lambda_sa in [2.0, 3.0, 5.0, 7.0]:
                for lambda_occ in [1.0, 2.0, 3.0, 5.0]:
                    params = dict(occ_target=occ_target, occ_sigma=occ_sigma,
                                  lambda_sa=lambda_sa, lambda_occ=lambda_occ)
                    n_ok, n_tot, avg_rank, details = evaluate_formula(compute_new_score, params)
                    metric = n_ok + (1 - avg_rank / 5)
                    if metric > best_overall_score:
                        best_overall_score = metric
                        best_overall = params.copy()
                        best_overall_details = details
                        best_overall_label = "V1"

    print(f"  Best V1: {best_overall} → {best_overall_score:.2f}")
    for d in best_overall_details:
        print(d)

    # --- V2: SA normalized by occupancy + Gaussian ---
    print("\n--- V2: SA normalized by occupancy + Gaussian ---")
    best_v2 = None
    best_v2_score = -1
    best_v2_details = None
    for occ_target in [0.15, 0.20, 0.25, 0.30, 0.35]:
        for occ_sigma in [0.10, 0.15, 0.18, 0.25]:
            for lambda_sa in [1.0, 2.0, 3.0, 5.0]:
                for lambda_occ in [1.0, 2.0, 3.0, 5.0]:
                    for w_edit in [1.0, 2.0, 3.0]:
                        params = dict(occ_target=occ_target, occ_sigma=occ_sigma,
                                      lambda_sa=lambda_sa, lambda_occ=lambda_occ,
                                      w_editability=w_edit)
                        n_ok, n_tot, avg_rank, details = evaluate_formula(compute_score_v2, params)
                        metric = n_ok + (1 - avg_rank / 5)
                        if metric > best_v2_score:
                            best_v2_score = metric
                            best_v2 = params.copy()
                            best_v2_details = details

    print(f"  Best V2: {best_v2} → {best_v2_score:.2f}")
    for d in best_v2_details:
        print(d)
    if best_v2_score > best_overall_score:
        best_overall_score = best_v2_score
        best_overall = best_v2
        best_overall_details = best_v2_details
        best_overall_label = "V2"

    # --- V3: Log-occupancy + per-pixel SA ---
    print("\n--- V3: Log-occupancy + per-pixel SA ---")
    best_v3 = None
    best_v3_score = -1
    best_v3_details = None
    for occ_target in [0.15, 0.20, 0.25, 0.30, 0.35]:
        for lambda_sa in [0.5, 1.0, 2.0, 3.0]:
            for lambda_occ in [1.0, 2.0, 3.0, 5.0]:
                for w_edit in [1.0, 2.0, 3.0, 5.0]:
                    params = dict(occ_target=occ_target, lambda_sa=lambda_sa,
                                  lambda_occ=lambda_occ, w_edit=w_edit)
                    n_ok, n_tot, avg_rank, details = evaluate_formula(compute_score_v3, params)
                    metric = n_ok + (1 - avg_rank / 5)
                    if metric > best_v3_score:
                        best_v3_score = metric
                        best_v3 = params.copy()
                        best_v3_details = details

    print(f"  Best V3: {best_v3} → {best_v3_score:.2f}")
    for d in best_v3_details:
        print(d)
    if best_v3_score > best_overall_score:
        best_overall_score = best_v3_score
        best_overall = best_v3
        best_overall_details = best_v3_details
        best_overall_label = "V3"

    # --- V4: Asymmetric occupancy + sa normalization ---
    print("\n--- V4: Asymmetric occupancy + sa normalization ---")
    best_v4 = None
    best_v4_score = -1
    best_v4_details = None
    for occ_lo in [0.05, 0.10, 0.15, 0.20]:
        for occ_hi in [0.30, 0.35, 0.40, 0.50]:
            for lambda_sa in [1.0, 2.0, 3.0, 5.0]:
                for ltc in [2.0, 5.0, 8.0, 10.0]:
                    for ltf in [1.0, 3.0, 5.0, 8.0]:
                        for w_edit in [1.0, 2.0, 3.0]:
                            params = dict(occ_ideal_lo=occ_lo, occ_ideal_hi=occ_hi,
                                          lambda_sa=lambda_sa, lambda_too_close=ltc,
                                          lambda_too_far=ltf, w_edit=w_edit)
                            n_ok, n_tot, avg_rank, details = evaluate_formula(compute_score_v4, params)
                            metric = n_ok + (1 - avg_rank / 5)
                            if metric > best_v4_score:
                                best_v4_score = metric
                                best_v4 = params.copy()
                                best_v4_details = details

    print(f"  Best V4: {best_v4} → {best_v4_score:.2f}")
    for d in best_v4_details:
        print(d)
    if best_v4_score > best_overall_score:
        best_overall_score = best_v4_score
        best_overall = best_v4
        best_overall_details = best_v4_details
        best_overall_label = "V4"

    # --- V5: Sqrt-occupancy normalization + Gaussian ---
    print("\n--- V5: Sqrt-occupancy normalization + Gaussian ---")
    best_v5 = None
    best_v5_score = -1
    best_v5_details = None
    for occ_target in [0.15, 0.20, 0.25, 0.30, 0.35]:
        for occ_sigma in [0.10, 0.15, 0.20, 0.30]:
            for lambda_sa in [0.5, 1.0, 2.0, 3.0]:
                for lambda_occ in [1.0, 2.0, 3.0, 5.0]:
                    for w_edit in [1.0, 2.0, 3.0, 5.0]:
                        for sa_mode in ["sqrt", "linear"]:
                            params = dict(occ_target=occ_target, occ_sigma=occ_sigma,
                                          lambda_sa=lambda_sa, lambda_occ=lambda_occ,
                                          w_edit=w_edit, sa_norm_mode=sa_mode)
                            n_ok, n_tot, avg_rank, details = evaluate_formula(compute_score_v5, params)
                            metric = n_ok + (1 - avg_rank / 5)
                            if metric > best_v5_score:
                                best_v5_score = metric
                                best_v5 = params.copy()
                                best_v5_details = details

    print(f"  Best V5: {best_v5} → {best_v5_score:.2f}")
    for d in best_v5_details:
        print(d)
    if best_v5_score > best_overall_score:
        best_overall_score = best_v5_score
        best_overall = best_v5
        best_overall_details = best_v5_details
        best_overall_label = "V5"

    print(f"\n{'=' * 80}")
    print(f"OVERALL BEST: {best_overall_label} with params={best_overall}")
    print(f"Score: {best_overall_score:.2f}")
    for d in best_overall_details:
        print(d)
    print("=" * 80)

    return best_overall_label, best_overall


# ============================================================
# 5.  Additional analysis: sa_leak / occupancy relationship
# ============================================================
def analyze_sa_occ_ratio():
    """Check if sa_leakage is roughly proportional to occupancy."""
    print("\n" + "=" * 80)
    print("sa_leakage / occupancy ratio analysis")
    print("=" * 80)
    for exp in experiments:
        print(f"\n  {exp.name}:")
        for mult in sorted(exp.data):
            d = exp.data[mult]
            ratio = d['sa_leak'] / (d['occ'] + 1e-6)
            print(f"    mult={mult:5.1f}x  occ={d['occ']:.3f}  sa_leak={d['sa_leak']:.3f}  "
                  f"sa/occ={ratio:.3f}  contrast={d['contrast']:.3f}  pF1={d['pF1']:.3f}")


# ============================================================
# 6.  Exclude rabbit (outlier) and re-search
# ============================================================
def grid_search_exclude_rabbit():
    """Same grid search but excluding the rabbit case as an outlier."""
    global experiments
    original = experiments
    experiments = [e for e in original if e.name != "rabbit_sunglasses"]

    print("\n" + "=" * 80)
    print("GRID SEARCH (excluding rabbit outlier)")
    print("=" * 80)

    best_label, best_params = grid_search()

    # Now test on rabbit too
    experiments = original
    print("\n--- Testing best params on rabbit outlier ---")
    rabbit = [e for e in experiments if e.name == "rabbit_sunglasses"][0]
    mults = sorted(rabbit.data.keys())

    fn_map = {"V1": compute_new_score, "V2": compute_score_v2,
              "V3": compute_score_v3, "V4": compute_score_v4, "V5": compute_score_v5}
    fn = fn_map[best_label]

    for mult in mults:
        d = rabbit.data[mult]
        s = fn(
            contrast=d['contrast'], precision_f1=d['pF1'],
            leakage_top90=d['leak'], sa_leakage=d['sa_leak'],
            occupancy=d['occ'], size_penalty=d['penalty'],
            **best_params,
        )
        marker = " ← USER" if mult == rabbit.user_preferred else ""
        print(f"  mult={mult:5.1f}x  occ={d['occ']:.3f}  score={s:.4f}{marker}")

    experiments = original
    return best_label, best_params


# ============================================================
# 7. Fine-grained grid search on the best formula
# ============================================================
def compute_score_v4(
    contrast: float,
    precision_f1: float,
    leakage_top90: float,
    sa_leakage: float,
    occupancy: float,
    size_penalty: float,
    # --- Parameters ---
    lambda_leak: float = 1.5,
    lambda_sa: float = 3.0,
    # Occupancy sweet-spot as soft preference (not hard Gaussian)
    occ_ideal_lo: float = 0.15,     # below this, start penalizing (too far)
    occ_ideal_hi: float = 0.40,     # above this, start penalizing (too close)
    lambda_too_close: float = 5.0,  # penalty rate for being too close
    lambda_too_far: float = 3.0,    # penalty rate for being too far
    w_edit: float = 1.0,
) -> float:
    """
    V4: Asymmetric occupancy penalty with sa_leak normalization.

    Instead of a Gaussian, use a piecewise linear penalty:
      - occ in [occ_ideal_lo, occ_ideal_hi]: no penalty
      - occ > occ_ideal_hi: penalty grows (too close, background changes)
      - occ < occ_ideal_lo: penalty grows (too far, no visible edit)

    Also normalize sa_leakage by occupancy to remove the proportionality bias.
    """
    eps = 1e-6

    # Normalize sa_leakage by occupancy
    sa_norm = sa_leakage / (occupancy + eps)

    # Asymmetric occupancy penalty
    occ_penalty = 0.0
    if occupancy > occ_ideal_hi:
        occ_penalty = lambda_too_close * (occupancy - occ_ideal_hi)
    elif occupancy < occ_ideal_lo:
        occ_penalty = lambda_too_far * (occ_ideal_lo - occupancy)

    editability = w_edit * contrast * precision_f1

    score = (
        editability
        - lambda_leak * leakage_top90
        - lambda_sa * sa_norm
        - occ_penalty
        - size_penalty
    )
    return score


def compute_score_v5(
    contrast: float,
    precision_f1: float,
    leakage_top90: float,
    sa_leakage: float,
    occupancy: float,
    size_penalty: float,
    # --- Parameters ---
    lambda_leak: float = 1.5,
    lambda_sa: float = 2.0,
    occ_target: float = 0.30,
    occ_sigma: float = 0.20,
    lambda_occ: float = 2.0,
    w_edit: float = 2.0,
    sa_norm_mode: str = "sqrt",  # "raw", "linear", "sqrt"
) -> float:
    """
    V5: Sqrt-occupancy normalization of sa_leakage + Gaussian occ prior.

    sa_leak is roughly proportional to occ (sa/occ ratio ≈ 0.9-1.3).
    Using sqrt normalization gives a middle ground:
      - For large occ (face, 0.6): sqrt(0.6)=0.77, sa/sqrt(occ) ≈ 0.72
      - For small occ (rabbit, 0.05): sqrt(0.05)=0.22, sa/sqrt(occ) ≈ 0.25
    This dampens the proportionality without fully removing it (which would
    over-penalize small objects).
    """
    eps = 1e-6

    if sa_norm_mode == "raw":
        sa_term = sa_leakage
    elif sa_norm_mode == "linear":
        sa_term = sa_leakage / (occupancy + eps)
    else:  # sqrt
        sa_term = sa_leakage / (np.sqrt(occupancy) + eps)

    # Gaussian occupancy prior
    occ_penalty = lambda_occ * ((occupancy - occ_target) / occ_sigma) ** 2

    editability = w_edit * contrast * precision_f1

    score = (
        editability
        - lambda_leak * leakage_top90
        - lambda_sa * sa_term
        - occ_penalty
        - size_penalty
    )
    return score


def fine_grid_search(best_label: str, best_params: dict):
    """Fine-grained search around the best params found."""
    print("\n" + "=" * 80)
    print(f"FINE GRID SEARCH around {best_label} params={best_params}")
    print("=" * 80)

    if best_label == "V1":
        fn = compute_new_score
    elif best_label == "V2":
        fn = compute_score_v2
    else:
        fn = compute_score_v3

    # Generate fine grid around each parameter
    def neighborhood(val, delta=0.3, n=5):
        return np.linspace(max(0.01, val - delta * val), val + delta * val, n)

    best_score = -1
    best_p = None
    best_details = None

    if best_label == "V4":
        fn = compute_score_v4
        for olo in neighborhood(best_params['occ_ideal_lo'], 0.4, 7):
            for ohi in neighborhood(best_params['occ_ideal_hi'], 0.3, 5):
                for ls in neighborhood(best_params['lambda_sa'], 0.3, 5):
                    for ltc in neighborhood(best_params['lambda_too_close'], 0.3, 5):
                        for ltf in neighborhood(best_params['lambda_too_far'], 0.3, 5):
                            for we in neighborhood(best_params['w_edit'], 0.3, 5):
                                p = dict(occ_ideal_lo=olo, occ_ideal_hi=ohi, lambda_sa=ls,
                                         lambda_too_close=ltc, lambda_too_far=ltf, w_edit=we)
                                n_ok, _, avg_rank, details = evaluate_formula(fn, p)
                                metric = n_ok + (1 - avg_rank / 5)
                                if metric > best_score:
                                    best_score = metric
                                    best_p = p.copy()
                                    best_details = details

    elif best_label == "V5":
        fn = compute_score_v5
        for ot in neighborhood(best_params['occ_target'], 0.3, 7):
            for os_ in neighborhood(best_params['occ_sigma'], 0.3, 5):
                for ls in neighborhood(best_params['lambda_sa'], 0.3, 5):
                    for lo in neighborhood(best_params['lambda_occ'], 0.3, 5):
                        for we in neighborhood(best_params['w_edit'], 0.3, 5):
                            p = dict(occ_target=ot, occ_sigma=os_, lambda_sa=ls,
                                     lambda_occ=lo, w_edit=we,
                                     sa_norm_mode=best_params.get('sa_norm_mode', 'sqrt'))
                            n_ok, _, avg_rank, details = evaluate_formula(fn, p)
                            metric = n_ok + (1 - avg_rank / 5)
                            if metric > best_score:
                                best_score = metric
                                best_p = p.copy()
                                best_details = details

    elif best_label == "V2":
        for ot in neighborhood(best_params['occ_target'], 0.3, 7):
            for os_ in neighborhood(best_params['occ_sigma'], 0.3, 5):
                for ls in neighborhood(best_params['lambda_sa'], 0.3, 5):
                    for lo in neighborhood(best_params['lambda_occ'], 0.3, 5):
                        for we in neighborhood(best_params['w_editability'], 0.3, 5):
                            p = dict(occ_target=ot, occ_sigma=os_, lambda_sa=ls,
                                     lambda_occ=lo, w_editability=we)
                            n_ok, _, avg_rank, details = evaluate_formula(fn, p)
                            metric = n_ok + (1 - avg_rank / 5)
                            if metric > best_score:
                                best_score = metric
                                best_p = p.copy()
                                best_details = details

    elif best_label == "V3":
        for ot in neighborhood(best_params['occ_target'], 0.3, 7):
            for ls in neighborhood(best_params['lambda_sa'], 0.3, 5):
                for lo in neighborhood(best_params['lambda_occ'], 0.3, 5):
                    for we in neighborhood(best_params['w_edit'], 0.3, 5):
                        p = dict(occ_target=ot, lambda_sa=ls,
                                 lambda_occ=lo, w_edit=we)
                        n_ok, _, avg_rank, details = evaluate_formula(fn, p)
                        metric = n_ok + (1 - avg_rank / 5)
                        if metric > best_score:
                            best_score = metric
                            best_p = p.copy()
                            best_details = details

    elif best_label == "V1":
        for ot in neighborhood(best_params['occ_target'], 0.3, 7):
            for os_ in neighborhood(best_params['occ_sigma'], 0.3, 5):
                for ls in neighborhood(best_params['lambda_sa'], 0.3, 5):
                    for lo in neighborhood(best_params['lambda_occ'], 0.3, 5):
                        p = dict(occ_target=ot, occ_sigma=os_, lambda_sa=ls,
                                 lambda_occ=lo)
                        n_ok, _, avg_rank, details = evaluate_formula(fn, p)
                        metric = n_ok + (1 - avg_rank / 5)
                        if metric > best_score:
                            best_score = metric
                            best_p = p.copy()
                            best_details = details

    print(f"  Fine-tuned: {best_p}")
    print(f"  Score: {best_score:.3f}")
    for d in best_details:
        print(d)
    return best_label, best_p


# ============================================================
#  Main
# ============================================================
if __name__ == "__main__":
    print_analysis()
    analyze_sa_occ_ratio()

    print("\n\n" + "#" * 80)
    print("# FULL GRID SEARCH (all 4 experiments)")
    print("#" * 80)
    best_label, best_params = grid_search()

    print("\n\n" + "#" * 80)
    print("# FINE GRID SEARCH")
    print("#" * 80)
    best_label, best_params = fine_grid_search(best_label, best_params)

    print("\n\n" + "#" * 80)
    print("# SEARCH EXCLUDING RABBIT OUTLIER")
    print("#" * 80)
    grid_search_exclude_rabbit()

    # ============================================================
    # V5 + sa_leak threshold constraint (handles rabbit outlier)
    # ============================================================
    print("\n\n" + "#" * 80)
    print("# V5 + sa_leak threshold constraint")
    print("#" * 80)

    # Best V5 params from search
    v5_params = {'occ_target': 0.35, 'occ_sigma': 0.15, 'lambda_sa': 0.5,
                 'lambda_occ': 1.0, 'w_edit': 5.0, 'sa_norm_mode': 'sqrt'}

    for sa_thresh in [0.05, 0.08, 0.10, 0.12, 0.15]:
        n_ok = 0
        details = []
        for exp in experiments:
            mults = sorted(exp.data.keys())
            scores = {}
            for mult in mults:
                d = exp.data[mult]
                s = compute_score_v5(
                    contrast=d['contrast'], precision_f1=d['pF1'],
                    leakage_top90=d['leak'], sa_leakage=d['sa_leak'],
                    occupancy=d['occ'], size_penalty=d['penalty'],
                    **v5_params,
                )
                # Hard constraint: if sa_leak > threshold, heavily penalize
                if d['sa_leak'] > sa_thresh:
                    s -= 10.0 * (d['sa_leak'] - sa_thresh)
                scores[mult] = s

            best_mult = max(scores, key=scores.get)
            if exp.user_preferred_range:
                correct = exp.user_preferred_range[0] <= best_mult <= exp.user_preferred_range[1]
            else:
                correct = best_mult == exp.user_preferred
            if correct:
                n_ok += 1
            marker = "OK" if correct else "MISS"
            scores_str = ", ".join(f"{m}x={scores[m]:.3f}" for m in mults)
            details.append(f"    [{marker}] {exp.name:25s}  picked={best_mult}x  want={exp.user_preferred}x  ({scores_str})")

        print(f"  sa_thresh={sa_thresh:.2f} → {n_ok}/4 correct")
        for d in details:
            print(d)

    # ============================================================
    # V6: V5 + soft sa_leak penalty with diminishing-returns shape
    # ============================================================
    print("\n\n" + "#" * 80)
    print("# V6: Combined approach - editability sweet spot + sa_leak knee")
    print("#" * 80)

    def compute_score_v6(contrast, precision_f1, leakage_top90, sa_leakage,
                         occupancy, size_penalty,
                         occ_target=0.30, occ_sigma=0.15,
                         lambda_leak=1.5, lambda_sa=0.5, lambda_occ=1.0,
                         w_edit=5.0, sa_knee=0.10, lambda_sa_excess=8.0):
        """
        V6: V5 with an additional steep penalty when sa_leak exceeds a knee threshold.
        Below the knee, sa_leak penalty is gentle. Above, it's steep.
        This creates a "cliff" effect that prevents selecting views with too much leakage.
        """
        eps = 1e-6
        sa_term = sa_leakage / (np.sqrt(occupancy) + eps)
        occ_penalty = lambda_occ * ((occupancy - occ_target) / occ_sigma) ** 2
        editability = w_edit * contrast * precision_f1

        # Soft knee: extra penalty for sa_leak above threshold
        sa_excess_penalty = 0.0
        if sa_leakage > sa_knee:
            sa_excess_penalty = lambda_sa_excess * (sa_leakage - sa_knee) ** 2

        score = (
            editability
            - lambda_leak * leakage_top90
            - lambda_sa * sa_term
            - occ_penalty
            - sa_excess_penalty
            - size_penalty
        )
        return score

    best_v6 = None
    best_v6_score = -1
    best_v6_details = None
    for occ_target in [0.25, 0.30, 0.35, 0.40]:
        for occ_sigma in [0.12, 0.15, 0.20, 0.25]:
            for lambda_sa in [0.3, 0.5, 1.0]:
                for lambda_occ in [0.5, 1.0, 2.0]:
                    for w_edit in [3.0, 5.0, 7.0]:
                        for sa_knee in [0.06, 0.08, 0.10, 0.12, 0.15]:
                            for lse in [3.0, 5.0, 8.0, 12.0]:
                                params = dict(occ_target=occ_target, occ_sigma=occ_sigma,
                                              lambda_sa=lambda_sa, lambda_occ=lambda_occ,
                                              w_edit=w_edit, sa_knee=sa_knee,
                                              lambda_sa_excess=lse)
                                n_ok, n_tot, avg_rank, details = evaluate_formula(compute_score_v6, params)
                                metric = n_ok + (1 - avg_rank / 5)
                                if metric > best_v6_score:
                                    best_v6_score = metric
                                    best_v6 = params.copy()
                                    best_v6_details = details

    print(f"  Best V6: {best_v6} → {best_v6_score:.2f}")
    for d in best_v6_details:
        print(d)

    # ============================================================
    # Elbow-detection approach
    # ============================================================
    print("\n\n" + "#" * 80)
    print("# ELBOW DETECTION: Find the 'knee' of the sa_leak curve")
    print("#" * 80)

    for exp in experiments:
        mults = sorted(exp.data.keys())
        sa_values = [exp.data[m]['sa_leak'] for m in mults]
        occ_values = [exp.data[m]['occ'] for m in mults]

        # Method: find the multiplier where sa_leak levels off
        # Use "second derivative" or ratio-of-ratios
        if len(mults) >= 3:
            # Compute rate of change of sa_leak
            rates = []
            for i in range(1, len(mults)):
                delta_mult = mults[i] - mults[i-1]
                delta_sa = sa_values[i-1] - sa_values[i]  # positive = improvement
                rate = delta_sa / delta_mult  # improvement per unit multiplier
                rates.append(rate)

            # The "elbow" is where the rate drops significantly
            # Find the last multiplier with a "big" rate before it drops
            best_elbow = mults[0]
            for i in range(len(rates)):
                if i < len(rates) - 1:
                    ratio = rates[i+1] / (rates[i] + 1e-6)
                    if ratio < 0.6:  # rate dropped by 40%+
                        best_elbow = mults[i+1]  # the last "good" step
                        break
                else:
                    best_elbow = mults[i+1]

            marker = ""
            if best_elbow == exp.user_preferred:
                marker = " MATCH!"
            elif exp.user_preferred_range and exp.user_preferred_range[0] <= best_elbow <= exp.user_preferred_range[1]:
                marker = " MATCH!"
            print(f"  {exp.name:25s}  elbow={best_elbow}x  user={exp.user_preferred}x{marker}")
            print(f"    rates: {['%.3f' % r for r in rates]}")

    # Marginal analysis
    print("\n\n" + "#" * 80)
    print("# MARGINAL ANALYSIS: sa_leak drop rate")
    print("#" * 80)
    for exp in experiments:
        print(f"\n  {exp.name}:")
        mults = sorted(exp.data.keys())
        for i in range(1, len(mults)):
            prev_m, curr_m = mults[i-1], mults[i]
            prev_d, curr_d = exp.data[prev_m], exp.data[curr_m]
            sa_drop = prev_d['sa_leak'] - curr_d['sa_leak']
            sa_drop_pct = sa_drop / (prev_d['sa_leak'] + 1e-6) * 100
            edit_drop = (prev_d['contrast'] * prev_d['pF1']) - (curr_d['contrast'] * curr_d['pF1'])
            marker = " ← USER" if curr_m == exp.user_preferred else ""
            if exp.user_preferred_range and exp.user_preferred_range[0] <= curr_m <= exp.user_preferred_range[1]:
                marker = " ← USER"
            print(f"    {prev_m:.1f}x → {curr_m:.1f}x:  "
                  f"Δsa_leak={sa_drop:+.3f} ({sa_drop_pct:+.0f}%)  "
                  f"Δeditability={edit_drop:+.3f}  "
                  f"sa_leak={curr_d['sa_leak']:.3f}  occ={curr_d['occ']:.3f}{marker}")

    # Final summary
    print("\n\n" + "=" * 80)
    print("SUMMARY OF PROPOSED CHANGES TO generate_by_lens.py")
    print("=" * 80)
    print(f"""
The current scoring formula:
  S_total = contrast * pF1 - λ_leak * leak_top90 - λ_sa * sa_leakage - penalty

Problem: sa_leakage monotonically decreases with distance → system always
picks the farthest view. The user wants a sweet spot.

Proposed fix (best formula: {best_label}):
  Parameters: {best_params}

Key changes:
  1. ADD occupancy sweet-spot penalty (Gaussian around target occupancy)
  2. Optionally NORMALIZE sa_leakage by occupancy (sa_leak/occ) to
     remove the "bigger ROI = higher sa_leak" proportionality bias
  3. Boost editability reward (contrast * pF1) to balance against penalties

For the rabbit outlier (very small ROI):
  - The occupancy Gaussian naturally handles it if the target is set correctly
  - A fallback: if no multiplier yields occ > min_occ_threshold, clamp to
    the closest multiplier that does
""")
