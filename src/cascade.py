"""
cascade.py — Cascade assembly and external inference.

Combines L1 (binary detection), L2 (heavy chain), and L3 (light chain)
predictions into 9-class isotype assignments for both OOF and external data.
"""

import numpy as np
from .constants import L2_CLASSES


def assemble_cascade_oof(l1_proba, l2_pred, l3_pred, l1_threshold,
                         pos_idx, y_binary_enc,
                         l2_models=None, l3_models=None, X_peak_full=None):
    """Assemble 9-class predictions from L1→L2→L3 OOF predictions.

    For samples predicted positive by L1:
      - True positives: use OOF L2/L3 predictions
      - False positives: run through L2/L3 models (ensemble averaging)

    Args:
        l1_proba:       (N,) L1 predicted probabilities
        l2_pred:        (N_pos,) L2 predicted class indices
        l3_pred:        (N_pos,) L3 predicted class indices (0=κ, 1=λ)
        l1_threshold:   float, L1 decision threshold
        pos_idx:        indices of truly positive samples
        y_binary_enc:   (N,) binary labels
        l2_models:      list of fitted L2 models (for FP handling)
        l3_models:      list of fitted L3 models (for FP handling)
        X_peak_full:    (N, F) full feature matrix (for FP handling)

    Returns:
        pred_9class:  (N,) string array of 9-class predictions
        fp_indices:   list of false-positive sample indices
        info:         dict with summary statistics
    """
    n = len(l1_proba)
    l1_positive = l1_proba >= l1_threshold
    pred_9class = np.full(n, 'NEGATIVE', dtype='U20')
    fp_indices = []

    pos_set = set(pos_idx)
    heavy_names = L2_CLASSES
    light_names = ['KAPPA', 'LAMBDA']

    for i in range(n):
        if l1_positive[i]:
            if i in pos_set:
                oi = list(pos_idx).index(i)
                if oi < len(l2_pred):
                    h = heavy_names[l2_pred[oi]]
                    l = light_names[l3_pred[oi]]
                    pred_9class[i] = f'{h}_{l}'
                else:
                    pred_9class[i] = 'NEGATIVE'
            else:
                # False positive — infer with L2/L3 ensemble
                fp_indices.append(i)
                if l2_models is not None and l3_models is not None \
                        and X_peak_full is not None:
                    x = X_peak_full[i:i + 1]
                    l2p = np.mean([m.predict_proba(x)[0] for m in l2_models],
                                  axis=0)
                    l3p = np.mean([m.predict_proba(x)[:, 1] for m in l3_models])
                    h = heavy_names[np.argmax(l2p)]
                    l = light_names[int(l3p >= 0.5)]
                    pred_9class[i] = f'{h}_{l}'

    info = {
        'l1_positive_count': int(l1_positive.sum()),
        'fp_count': len(fp_indices),
        'threshold': float(l1_threshold),
    }
    return pred_9class, fp_indices, info


def run_external_cascade(l1_models, l2_models, l3_models,
                         X_ext_peak, l1_threshold=0.5):
    """Run full L1→L2→L3 cascade on external validation data.

    All models use 5-fold ensemble averaging for inference.

    Returns:
        pred_9class:   (N,) string array of predictions
        l1_proba:      (N,) L1 probabilities
        l2_proba_ext:  (N, 4) L2 probabilities
        l3_proba_ext:  (N,) L3 probabilities (P(lambda))
    """
    n = X_ext_peak.shape[0]
    heavy_names = L2_CLASSES
    light_names = ['KAPPA', 'LAMBDA']

    # L1: ensemble binary prediction
    l1_proba = _ensemble_predict(l1_models, X_ext_peak, task='binary')
    l1_positive = l1_proba >= l1_threshold

    pred_9class = np.full(n, 'NEGATIVE', dtype='U20')
    l2_proba_ext = np.zeros((n, 4), dtype=np.float32)
    l3_proba_ext = np.zeros(n, dtype=np.float32)

    pos_idx = np.where(l1_positive)[0]
    if len(pos_idx) > 0:
        X_pos = X_ext_peak[pos_idx]
        l2p = _ensemble_predict(l2_models, X_pos, task='multi')
        l3p = _ensemble_predict(l3_models, X_pos, task='binary')

        l2_proba_ext[pos_idx] = l2p
        l3_proba_ext[pos_idx] = l3p

        for j, idx in enumerate(pos_idx):
            h = heavy_names[np.argmax(l2p[j])]
            l = light_names[int(l3p[j] >= 0.5)]
            pred_9class[idx] = f'{h}_{l}'

    return pred_9class, l1_proba, l2_proba_ext, l3_proba_ext


def _ensemble_predict(models, X, task='binary'):
    """Run XGBoost ensemble inference (5-fold averaging)."""
    if task == 'binary':
        probas = [m.predict_proba(X)[:, 1] for m in models]
    else:
        probas = [m.predict_proba(X) for m in models]
    return np.mean(probas, axis=0)


def get_fold_models(obj):
    """Extract fold models from dict or list."""
    if isinstance(obj, dict):
        return [obj[k] for k in sorted(obj.keys())]
    return list(obj)
