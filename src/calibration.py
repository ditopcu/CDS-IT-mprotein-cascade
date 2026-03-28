"""
calibration.py — Probability calibration assessment and correction.

Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
Brier Skill Score (BSS), isotonic regression calibration, and
comprehensive calibration reporting.
"""

import numpy as np
from sklearn.metrics import brier_score_loss
from sklearn.isotonic import IsotonicRegression


def compute_ece(y_true, y_prob, n_bins=10, strategy='uniform'):
    """Expected Calibration Error.

    For binary: y_prob is P(positive), shape (n,).
    For multiclass: y_prob is (n, K), y_true is int-encoded.

    Returns: (ece_float, list_of_bin_dicts)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if y_prob.ndim == 2:
        confidences = np.max(y_prob, axis=1)
        accuracies = (np.argmax(y_prob, axis=1) == y_true).astype(float)
    else:
        confidences = y_prob
        accuracies = y_true.astype(float)

    if strategy == 'quantile':
        bin_edges = np.unique(np.quantile(confidences, np.linspace(0, 1, n_bins + 1)))
        if len(bin_edges) < 3:
            bin_edges = np.linspace(0, 1, n_bins + 1)
    else:
        bin_edges = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    bin_details = []
    n_total = len(y_true)

    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i > 0:
            mask = (confidences > lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences <= hi)
        n_bin = mask.sum()

        if n_bin == 0:
            bin_details.append({'bin': i, 'lo': lo, 'hi': hi,
                                'n': 0, 'avg_conf': 0, 'avg_acc': 0, 'gap': 0})
            continue

        avg_conf = confidences[mask].mean()
        avg_acc = accuracies[mask].mean()
        gap = abs(avg_acc - avg_conf)
        ece += (n_bin / n_total) * gap

        bin_details.append({
            'bin': i, 'lo': float(lo), 'hi': float(hi), 'n': int(n_bin),
            'avg_conf': float(avg_conf), 'avg_acc': float(avg_acc),
            'gap': float(gap),
        })

    return float(ece), bin_details


def compute_mce(y_true, y_prob, n_bins=10, strategy='uniform'):
    """Maximum Calibration Error — worst-case bin gap."""
    _, bin_details = compute_ece(y_true, y_prob, n_bins, strategy)
    gaps = [b['gap'] for b in bin_details if b['n'] > 0]
    return float(max(gaps)) if gaps else 0.0


def compute_classwise_ece(y_true, y_prob, n_bins=10):
    """Classwise ECE: average ECE across one-vs-rest decomposition.

    Returns: (overall_classwise_ece, dict_of_per_class_ece)
    """
    n_classes = y_prob.shape[1]
    class_eces = {}
    for c in range(n_classes):
        bt = (np.asarray(y_true) == c).astype(int)
        ece_c, _ = compute_ece(bt, y_prob[:, c], n_bins)
        class_eces[c] = ece_c
    return float(np.mean(list(class_eces.values()))), class_eces


def scaled_brier_score(y_true, y_prob):
    """Brier Skill Score: BSS = 1 - Brier / Brier_ref.

    BSS = 1 → perfect, BSS = 0 → no skill, BSS < 0 → worse than prevalence.

    Returns: (bss_float, detail_dict)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if y_prob.ndim == 2:
        bss_list = []
        for c in range(y_prob.shape[1]):
            bt = (y_true == c).astype(int)
            brier = brier_score_loss(bt, y_prob[:, c])
            prev = bt.mean()
            brier_ref = prev * (1 - prev)
            bss_list.append(1 - brier / brier_ref if brier_ref > 0 else 0.0)
        return float(np.mean(bss_list)), {c: bss_list[c] for c in range(len(bss_list))}
    else:
        brier = brier_score_loss(y_true, y_prob)
        prev = y_true.mean()
        brier_ref = prev * (1 - prev)
        bss = 1 - brier / brier_ref if brier_ref > 0 else 0.0
        return float(bss), {'brier': float(brier), 'brier_ref': float(brier_ref)}


def calibrate_oof_isotonic(proba, y_true, fold_indices, n_folds=5):
    """Isotonic calibration using out-of-fold strategy (no leakage).

    Each fold is calibrated using a model fitted on the remaining folds.
    """
    calibrated = np.zeros_like(proba, dtype=float)
    for fold_id in range(n_folds):
        train_mask = fold_indices != fold_id
        test_mask = fold_indices == fold_id
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(proba[train_mask], y_true[train_mask])
        calibrated[test_mask] = iso.predict(proba[test_mask])
    return calibrated


def calibration_report(y_true, y_prob, level='L1', n_bins=10):
    """Comprehensive calibration report for one cascade level.

    Returns dict with ECE, MCE, classwise ECE, Brier, BSS, and bin details.
    """
    is_multi = y_prob.ndim == 2

    if is_multi:
        ece, bins = compute_ece(y_true, y_prob, n_bins)
        mce = compute_mce(y_true, y_prob, n_bins)
        cw_ece, class_eces = compute_classwise_ece(y_true, y_prob, n_bins)
        bss, bss_detail = scaled_brier_score(y_true, y_prob)
        brier = np.mean([
            brier_score_loss((np.asarray(y_true) == c).astype(int), y_prob[:, c])
            for c in range(y_prob.shape[1])
        ])
    else:
        ece, bins = compute_ece(y_true, y_prob, n_bins)
        mce = compute_mce(y_true, y_prob, n_bins)
        cw_ece = ece
        class_eces = {0: ece}
        brier = brier_score_loss(y_true, y_prob)
        bss, bss_detail = scaled_brier_score(y_true, y_prob)

    return {
        'level': level,
        'ece': float(ece),
        'mce': float(mce),
        'classwise_ece': float(cw_ece),
        'class_eces': class_eces,
        'scaled_brier': float(bss),
        'brier': float(brier),
        'bss_detail': bss_detail,
        'bin_details': bins,
        'n_samples': len(y_true),
        'is_multiclass': is_multi,
    }
