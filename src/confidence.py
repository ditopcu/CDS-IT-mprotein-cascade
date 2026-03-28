"""
confidence.py — Confidence scoring, conformal prediction, and compound probabilities.

Cascade compound confidence score, split conformal prediction with
finite-sample coverage guarantees, and 9-class probability matrix construction.
"""

import numpy as np
import pandas as pd
from .constants import CLASS9_NAMES, L2_CLASSES
from .calibration import compute_ece, compute_mce, compute_classwise_ece, scaled_brier_score


def compute_cascade_confidence(l1_proba, l2_proba, l3_proba,
                               l1_threshold, pos_idx):
    """Compute compound cascade confidence for all samples.

    Confidence formula:
      - Negative: conf = |p1 - τ| / max(τ, 1-τ)
      - Positive: conf = conf_L1 × max(P_L2) × 2|p3 - 0.5|

    Returns: DataFrame with l1_conf, l2_conf, l3_conf, cascade_conf
    """
    n = len(l1_proba)
    pos_set = set(pos_idx) if pos_idx is not None else set()
    pos_idx_list = list(pos_idx) if pos_idx is not None else []

    rows = []
    for i in range(n):
        is_pos = l1_proba[i] >= l1_threshold
        l1_conf = abs(l1_proba[i] - l1_threshold) / max(l1_threshold, 1 - l1_threshold)
        l1_conf = min(l1_conf, 1.0)

        if is_pos and i in pos_set:
            oi = pos_idx_list.index(i)
            l2_conf = float(np.max(l2_proba[oi]))
            l3_conf = abs(l3_proba[oi] - 0.5) * 2
            cascade_conf = l1_conf * l2_conf * max(l3_conf, 0.01)
        elif is_pos:
            l2_conf = 0.5
            l3_conf = 0.5
            cascade_conf = l1_conf * 0.25
        else:
            l2_conf = 1.0
            l3_conf = 1.0
            cascade_conf = l1_conf

        rows.append({
            'l1_proba': float(l1_proba[i]),
            'l1_positive': bool(is_pos),
            'l1_conf': float(l1_conf),
            'l2_conf': float(l2_conf),
            'l3_conf': float(l3_conf),
            'cascade_conf': float(cascade_conf),
        })
    return pd.DataFrame(rows)


def build_cascade_proba_9class(l1_proba, l2_proba, l3_proba,
                                l1_threshold, pos_idx):
    """Build (N, 9) compound probability matrix from cascade components.

    P(class) = P(pos) × P(heavy|pos) × P(light|heavy, pos)

    Returns: ndarray of shape (N, 9) — normalized probability vectors.
    """
    n = len(l1_proba)
    proba_9 = np.zeros((n, 9), dtype=np.float64)
    pos_set = set(pos_idx) if pos_idx is not None else set()
    pos_idx_list = list(pos_idx) if pos_idx is not None else []
    neg_idx = CLASS9_NAMES.index('NEGATIVE')

    heavy_order = L2_CLASSES
    light_order = ['KAPPA', 'LAMBDA']
    class9_to_idx = {c: i for i, c in enumerate(CLASS9_NAMES)}

    for i in range(n):
        p_pos = float(l1_proba[i])
        p_neg = 1.0 - p_pos

        if p_pos >= l1_threshold and i in pos_set:
            oi = pos_idx_list.index(i)
            l2p = l2_proba[oi]
            l3p_lambda = float(l3_proba[oi])
            l3p_kappa = 1.0 - l3p_lambda

            for hi, h in enumerate(heavy_order):
                for li, l in enumerate(light_order):
                    cls_name = f'{h}_{l}'
                    if cls_name in class9_to_idx:
                        l3_val = l3p_kappa if l == 'KAPPA' else l3p_lambda
                        proba_9[i, class9_to_idx[cls_name]] = p_pos * l2p[hi] * l3_val

            proba_9[i, neg_idx] = p_neg
        else:
            proba_9[i, neg_idx] = p_neg
            remaining = p_pos / 8 if p_pos > 0 else 0
            for j in range(9):
                if j != neg_idx:
                    proba_9[i, j] = remaining

        # Normalize
        row_sum = proba_9[i].sum()
        if row_sum > 0:
            proba_9[i] /= row_sum

    return proba_9


def conformal_prediction(y_true_9class, cascade_proba_9class,
                         alpha=0.05, cal_fraction=0.3, seed=42):
    """Split conformal prediction for 9-class cascade.

    Generates prediction sets with finite-sample coverage guarantee:
    P(Y ∈ C(X)) ≥ 1 − α, requiring only exchangeability.

    Args:
        y_true_9class:        (N,) string class labels
        cascade_proba_9class: (N, 9) compound probability matrix
        alpha:                significance level (default 0.05 → 95% coverage)
        cal_fraction:         fraction used for calibration (default 0.3)

    Returns:
        dict with coverage, mean/median set size, prediction sets, indices
    """
    rng = np.random.RandomState(seed)
    n = len(y_true_9class)
    class_to_idx = {c: i for i, c in enumerate(CLASS9_NAMES)}

    idx_all = np.arange(n)
    rng.shuffle(idx_all)
    n_cal = int(n * cal_fraction)
    cal_idx = idx_all[:n_cal]
    test_idx = idx_all[n_cal:]

    # Calibration: non-conformity scores = 1 - p(true class)
    cal_scores = []
    for i in cal_idx:
        true_cls = y_true_9class[i]
        if true_cls in class_to_idx:
            cal_scores.append(1 - cascade_proba_9class[i, class_to_idx[true_cls]])
        else:
            cal_scores.append(1.0)
    cal_scores = np.array(cal_scores)

    q_hat = np.quantile(cal_scores, 1 - alpha, interpolation='higher')

    # Build prediction sets on test split
    pred_sets = []
    sizes = []
    covered = 0
    for i in test_idx:
        pset = [CLASS9_NAMES[j] for j in range(9)
                if cascade_proba_9class[i, j] >= 1 - q_hat]
        if not pset:
            pset = [CLASS9_NAMES[np.argmax(cascade_proba_9class[i])]]
        pred_sets.append(pset)
        sizes.append(len(pset))
        if y_true_9class[i] in pset:
            covered += 1

    coverage = covered / len(test_idx) if len(test_idx) > 0 else 0

    return {
        'alpha': alpha,
        'q_hat': float(q_hat),
        'coverage': float(coverage),
        'mean_set_size': float(np.mean(sizes)),
        'median_set_size': float(np.median(sizes)),
        'pred_sets': pred_sets,
        'test_idx': test_idx,
        'cal_idx': cal_idx,
    }


def validate_compound_calibration(cascade_proba_9, y_true_9,
                                   class_names=None,
                                   n_bins=10, strategy='quantile'):
    """Validate compound probability calibration across all 9 classes.

    Returns dict with overall ECE, per-class ECE, and per-class reliability data.
    """
    if class_names is None:
        class_names = CLASS9_NAMES

    y_arr = np.asarray(y_true_9)
    if y_arr.dtype.kind in ('U', 'S', 'O'):
        class_to_idx = {c: i for i, c in enumerate(class_names)}
        y_enc = np.array([class_to_idx.get(str(c), -1) for c in y_arr])
    else:
        y_enc = y_arr.astype(int)

    ece_overall, bins_overall = compute_ece(y_enc, cascade_proba_9, n_bins, strategy)
    mce_overall = compute_mce(y_enc, cascade_proba_9, n_bins, strategy)
    cw_ece, class_eces = compute_classwise_ece(y_enc, cascade_proba_9, n_bins)
    bss_overall, _ = scaled_brier_score(y_enc, cascade_proba_9)

    per_class_reliability = {}
    for c_idx, c_name in enumerate(class_names):
        bt = (y_enc == c_idx).astype(int)
        bp = cascade_proba_9[:, c_idx]
        n_c = bt.sum()
        effective_bins = max(3, min(n_bins, n_c // 5))

        if n_c < 10:
            per_class_reliability[c_name] = {
                'n': int(n_c), 'ece': float(class_eces.get(c_idx, 0)),
                'bins': [], 'note': 'too few samples for reliable binning',
            }
            continue

        ece_c, bins_c = compute_ece(bt, bp, effective_bins, strategy)
        per_class_reliability[c_name] = {
            'n': int(n_c), 'ece': float(ece_c), 'bins': bins_c,
        }

    return {
        'ece_overall': float(ece_overall),
        'mce_overall': float(mce_overall),
        'classwise_ece': float(cw_ece),
        'class_eces': {class_names[k]: v for k, v in class_eces.items()},
        'per_class_reliability': per_class_reliability,
        'scaled_brier': float(bss_overall),
        'bins_overall': bins_overall,
    }


def validate_level_independence(l2_errors, l3_errors):
    """Test L2↔L3 error independence via phi coefficient and chi-squared.

    If errors are correlated, the compound probability product assumption
    P(class) = P(heavy) × P(light) may be biased.

    Returns dict with phi coefficient, chi² p-value, joint error rates.
    """
    from scipy.stats import chi2_contingency

    l2e = np.asarray(l2_errors).astype(int)
    l3e = np.asarray(l3_errors).astype(int)
    n = len(l2e)

    both_correct   = ((l2e == 0) & (l3e == 0)).sum()
    l2_wrong_only  = ((l2e == 1) & (l3e == 0)).sum()
    l3_wrong_only  = ((l2e == 0) & (l3e == 1)).sum()
    both_wrong     = ((l2e == 1) & (l3e == 1)).sum()

    table = np.array([[both_correct, l3_wrong_only],
                      [l2_wrong_only, both_wrong]])

    p_l2 = l2e.mean()
    p_l3 = l3e.mean()
    p_both = (l2e & l3e).mean()
    expected_both = p_l2 * p_l3
    denom = np.sqrt(p_l2 * (1 - p_l2) * p_l3 * (1 - p_l3))
    phi = (p_both - expected_both) / denom if denom > 0 else 0.0

    try:
        chi2, p_value, _, _ = chi2_contingency(table)
    except Exception:
        chi2, p_value = 0, 1

    ratio = p_both / expected_both if expected_both > 0 else float('inf')
    if 0.8 <= ratio <= 1.2:
        interp = 'Independent (ratio ≈ 1)'
    elif ratio > 1.2:
        interp = 'Positively correlated (errors co-occur)'
    else:
        interp = 'Negatively correlated'

    return {
        'contingency_table': table.tolist(),
        'phi_coefficient': float(phi),
        'chi2': float(chi2),
        'p_value': float(p_value),
        'joint_error_rate_observed': float(p_both),
        'joint_error_rate_expected': float(expected_both),
        'ratio': float(ratio),
        'interpretation': interp,
        'n': int(n),
    }
