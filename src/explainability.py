"""
explainability.py — TreeSHAP-based explainability utilities.

Region × channel SHAP aggregation for global explanations,
and helper functions for per-sample waterfall plots.
"""

import numpy as np
import pandas as pd
from .constants import REGIONS, CHANNELS


def aggregate_shap_by_region(shap_values, feature_names,
                              regions=None, channels=None):
    """Aggregate SHAP values into a region × channel heatmap.

    Sums mean |SHAP| for all features belonging to each
    (channel, region) combination.

    Args:
        shap_values:   (n_samples, n_features) SHAP value matrix
        feature_names: list of feature name strings
        regions:       list of region names (default: 5 main regions)
        channels:      list of channel names (default: CHANNELS)

    Returns:
        DataFrame of shape (n_channels, n_regions) with mean |SHAP| sums
    """
    if regions is None:
        regions = ['beta1', 'beta2', 'transition', 'gamma', 'mprotein']
    if channels is None:
        channels = CHANNELS

    matrix = np.zeros((len(channels), len(regions)))
    for ci, ch in enumerate(channels):
        for ri, rgn in enumerate(regions):
            total = 0.0
            for fi, fn in enumerate(feature_names):
                if ch in fn and rgn in fn:
                    total += np.mean(np.abs(shap_values[:, fi]))
            matrix[ci, ri] = total

    return pd.DataFrame(matrix, index=channels, columns=regions)


def get_top_features(shap_values_single, feature_names, top_n=12):
    """Get top-N features by absolute SHAP value for a single sample.

    Args:
        shap_values_single: (n_features,) SHAP values for one sample
        feature_names:      list of feature name strings
        top_n:              number of top features to return

    Returns:
        list of (feature_name, shap_value) tuples, sorted by |SHAP| descending
    """
    order = np.argsort(np.abs(shap_values_single))[::-1][:top_n]
    return [(feature_names[i], float(shap_values_single[i])) for i in order]
