"""
evaluation.py — Classification evaluation and cross-validation utilities.

Binary and multi-class evaluation metrics, generic CV runner,
cascade-level metrics, and bootstrap confidence intervals.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve,
    confusion_matrix, precision_recall_fscore_support,
    balanced_accuracy_score, cohen_kappa_score,
)

from .constants import CLASS9_NAMES


# ═══════════════════════════════════════════════════════════════
# Per-Fold Evaluation
# ═══════════════════════════════════════════════════════════════

def eval_binary(y_true, y_pred, y_prob=None):
    """Evaluate binary classification. Returns dict of metrics."""
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    m = {
        'accuracy':    accuracy_score(y_true, y_pred),
        'sensitivity': tp / (tp + fn) if tp + fn > 0 else 0,
        'specificity': tn / (tn + fp) if tn + fp > 0 else 0,
        'f1_macro':    f1_score(y_true, y_pred, average='macro'),
    }
    if y_prob is not None:
        m['auc_roc'] = roc_auc_score(y_true, y_prob)
    return m


def eval_multiclass(y_true, y_pred, y_prob=None, n_classes=4):
    """Evaluate multi-class classification. Returns dict of metrics."""
    m = {
        'macro_f1':     f1_score(y_true, y_pred, average='macro'),
        'weighted_f1':  f1_score(y_true, y_pred, average='weighted'),
        'balanced_acc': balanced_accuracy_score(y_true, y_pred),
        'kappa':        cohen_kappa_score(y_true, y_pred),
    }
    if y_prob is not None:
        try:
            m['auc_ovr'] = roc_auc_score(y_true, y_prob,
                                          multi_class='ovr', average='macro')
        except Exception:
            m['auc_ovr'] = 0.0
    for c in range(n_classes):
        mask = y_true == c
        m[f'recall_{c}'] = (y_pred[mask] == c).mean() if mask.sum() > 0 else 0.0
    return m


# ═══════════════════════════════════════════════════════════════
# Cross-Validation Runner
# ═══════════════════════════════════════════════════════════════

def run_cv(train_fn, X, fold_indices, y, metrics_list, eval_fn,
           label='Model', is_binary=True):
    """Generic k-fold cross-validation runner.

    Args:
        train_fn:     callable(X_train, y_train, X_val, y_val, fold_num) → model
        X:            feature matrix
        fold_indices: array of fold assignments (0..k-1)
        y:            target array
        metrics_list: list of metric names to track
        eval_fn:      callable(y_true, y_pred, y_prob) → dict
        label:        string label for logging
        is_binary:    if True, binary; if False, multi-class

    Returns:
        oof_proba, oof_pred, cv_metrics (dict of lists), models (list)
    """
    n_folds = len(np.unique(fold_indices))
    N = len(y)

    if is_binary:
        oof_proba = np.zeros(N, dtype=np.float32)
        oof_pred = np.zeros(N, dtype=np.int8)
    else:
        n_cls = len(np.unique(y))
        oof_proba = np.zeros((N, n_cls), dtype=np.float32)
        oof_pred = np.zeros(N, dtype=np.int8)

    cv_metrics = {m: [] for m in metrics_list}
    models = []

    for fn in range(n_folds):
        te_mask = fold_indices == fn
        tri, tei = np.where(~te_mask)[0], np.where(te_mask)[0]
        model = train_fn(X[tri], y[tri], X[tei], y[tei], fn)
        models.append(model)

        if is_binary:
            prob = model.predict_proba(X[tei])[:, 1]
            pred = (prob >= 0.5).astype(np.int8)
            oof_proba[tei] = prob
            oof_pred[tei] = pred
            met = eval_fn(y[tei], pred, prob)
        else:
            prob = model.predict_proba(X[tei])
            pred = np.argmax(prob, axis=1)
            oof_proba[tei] = prob
            oof_pred[tei] = pred
            met = eval_fn(y[tei], pred, prob)

        for k in metrics_list:
            if k in met:
                cv_metrics[k].append(met[k])

    return oof_proba, oof_pred, cv_metrics, models


def print_cv_summary(cv_metrics, metrics=None):
    """Print CV summary as mean ± std for each metric."""
    if metrics is None:
        metrics = list(cv_metrics.keys())
    for m in metrics:
        if m in cv_metrics and len(cv_metrics[m]) > 0:
            vals = np.array(cv_metrics[m])
            print(f'  {m:<20s}: {vals.mean():.4f} ± {vals.std():.4f}')


# ═══════════════════════════════════════════════════════════════
# Threshold Analysis
# ═══════════════════════════════════════════════════════════════

def find_optimal_thresholds(y_true, y_prob, name='Model'):
    """Find optimal thresholds: Youden, sensitivity ≥95%, sensitivity ≥98%.

    Returns dict with threshold, sensitivity, specificity for each method.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    spec = 1 - fpr
    results = {}

    # Youden's J statistic
    j = tpr + spec - 1
    idx = np.argmax(j)
    results['youden'] = {
        'threshold':   float(thresholds[idx]),
        'sensitivity': float(tpr[idx]),
        'specificity': float(spec[idx]),
    }

    # Sensitivity targets
    for target, label in [(0.95, 'sens_95'), (0.98, 'sens_98')]:
        mask = tpr >= target
        if mask.any():
            idx2 = np.where(mask)[0][np.argmax(spec[mask])]
            results[label] = {
                'threshold':   float(thresholds[idx2]),
                'sensitivity': float(tpr[idx2]),
                'specificity': float(spec[idx2]),
            }
    return results


# ═══════════════════════════════════════════════════════════════
# Cascade-Level (9-Class) Metrics
# ═══════════════════════════════════════════════════════════════

def compute_cascade_metrics(y_true, y_pred, class_names=None):
    """Compute comprehensive 9-class cascade metrics.

    Returns dict with accuracy, macro/weighted F1, Cohen's κ,
    confusion matrix, and per-class precision/recall/F1.
    """
    if class_names is None:
        class_names = CLASS9_NAMES

    metrics = {
        'accuracy':    accuracy_score(y_true, y_pred),
        'macro_f1':    f1_score(y_true, y_pred, labels=class_names,
                                average='macro', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, labels=class_names,
                                average='weighted', zero_division=0),
        'kappa':       cohen_kappa_score(y_true, y_pred),
        'confusion':   confusion_matrix(y_true, y_pred, labels=class_names),
    }

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=class_names, zero_division=0)
    metrics['per_class'] = {
        c: {'precision': float(p[i]), 'recall': float(r[i]),
            'f1': float(f[i]), 'support': int(s[i])}
        for i, c in enumerate(class_names)
    }
    return metrics


# ═══════════════════════════════════════════════════════════════
# Bootstrap Confidence Intervals
# ═══════════════════════════════════════════════════════════════

def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=2000,
                 alpha=0.05, seed=42):
    """Compute bootstrap confidence interval for a metric function.

    Args:
        y_true, y_pred: arrays
        metric_fn: callable(y_true, y_pred) → float
        n_boot: number of bootstrap iterations
        alpha: significance level (default 0.05 → 95% CI)

    Returns:
        (mean, lower, upper) tuple
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    scores = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        try:
            scores.append(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            continue

    scores = np.array(scores)
    lo = np.percentile(scores, 100 * alpha / 2)
    hi = np.percentile(scores, 100 * (1 - alpha / 2))
    return float(np.mean(scores)), float(lo), float(hi)


# ═══════════════════════════════════════════════════════════════
# Error Attribution
# ═══════════════════════════════════════════════════════════════

def attribute_errors(y_true_9, cascade_pred, l1_proba, l1_threshold,
                     l2_pred_full, l3_pred_full, y_binary_enc,
                     y_heavy, y_light, pos_idx):
    """Attribute each cascade error to its source level (L1/L2/L3).

    Returns DataFrame with one row per misclassified sample, including
    error type (L1_FN, L1_FP, L2_error, L3_error) and L1 probability.
    """
    import pandas as pd

    pos_set = set(pos_idx)
    pos_idx_list = list(pos_idx)
    rows = []

    for i in range(len(y_true_9)):
        if cascade_pred[i] == y_true_9[i]:
            continue

        is_true_pos = y_binary_enc[i] == 1
        l1_pred_pos = l1_proba[i] >= l1_threshold

        if is_true_pos and not l1_pred_pos:
            error_type = 'L1_FN'
        elif not is_true_pos and l1_pred_pos:
            error_type = 'L1_FP'
        elif is_true_pos and l1_pred_pos and i in pos_set:
            oi = pos_idx_list.index(i)
            if y_heavy[oi] != l2_pred_full[oi]:
                error_type = 'L2_error'
            else:
                error_type = 'L3_error'
        else:
            error_type = 'unknown'

        rows.append({
            'sample_idx': i,
            'true_class': y_true_9[i],
            'pred_class': cascade_pred[i],
            'error_type': error_type,
            'l1_proba': float(l1_proba[i]),
        })

    return pd.DataFrame(rows)
