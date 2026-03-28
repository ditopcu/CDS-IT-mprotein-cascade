"""
constants.py — Shared constants for the Cascade M-Protein CDS pipeline.

Electrophoretic region definitions, channel names, class labels,
color palettes, and default configuration presets.
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════
# Electrophoretic Regions (timepoint ranges within 300-point signals)
# ═══════════════════════════════════════════════════════════════
REGIONS = {
    'beta1':      (133, 171),   # transferrin region
    'beta2':      (171, 194),   # complement region
    'transition': (194, 211),   # β₂–γ transition zone
    'gamma':      (211, 263),   # immunoglobulin region
    'beta_full':  (133, 194),   # combined β region
    'mprotein':   (133, 263),   # M-protein search zone
}

# ═══════════════════════════════════════════════════════════════
# Channel Names (6-channel CZE-IT)
# ═══════════════════════════════════════════════════════════════
CHANNELS = [
    'raw_ELP',      # ELP/SPE reference channel
    'dif_IgG',      # anti-IgG immunotyping
    'dif_IgA',      # anti-IgA immunotyping
    'dif_IgM',      # anti-IgM immunotyping
    'dif_Kappa',    # anti-κ immunotyping
    'dif_Lambda',   # anti-λ immunotyping
]

N_CHANNELS = 6
N_TIMEPOINTS = 300

# ═══════════════════════════════════════════════════════════════
# Classification Labels
# ═══════════════════════════════════════════════════════════════
CLASS9_NAMES = [
    'FREE_KAPPA', 'FREE_LAMBDA',
    'IGA_KAPPA',  'IGA_LAMBDA',
    'IGG_KAPPA',  'IGG_LAMBDA',
    'IGM_KAPPA',  'IGM_LAMBDA',
    'NEGATIVE',
]

L2_CLASSES = ['IGG', 'IGA', 'IGM', 'FREE']  # Heavy chain (L2 order)
L3_CLASSES = ['KAPPA', 'LAMBDA']             # Light chain

HEAVY_MAP = {'IGG': 0, 'IGA': 1, 'IGM': 2, 'FREE': 3}
LIGHT_MAP = {'KAPPA': 0, 'LAMBDA': 1}

# ═══════════════════════════════════════════════════════════════
# Metric Lists (for CV tracking)
# ═══════════════════════════════════════════════════════════════
L1_METRICS = ['accuracy', 'sensitivity', 'specificity', 'f1_macro', 'auc_roc']
L2_METRICS = ['macro_f1', 'weighted_f1', 'balanced_acc', 'auc_ovr', 'kappa']

# ═══════════════════════════════════════════════════════════════
# Visualization Palettes
# ═══════════════════════════════════════════════════════════════
CH_COLORS = {
    'raw_ELP':    '#2C2C2C',
    'dif_IgG':    '#D32F2F',
    'dif_IgA':    '#7B1FA2',
    'dif_IgM':    '#1565C0',
    'dif_Kappa':  '#E65100',
    'dif_Lambda': '#2E7D32',
}

REGION_COLORS = {
    'beta1':      ('#AED6F1', 'β₁'),
    'beta2':      ('#A9DFBF', 'β₂'),
    'transition': ('#FAD7A0', 'trans.'),
    'gamma':      ('#F1948A', 'γ'),
}

# ═══════════════════════════════════════════════════════════════
# Default Configuration Presets
# ═══════════════════════════════════════════════════════════════
SEED = 42
N_FOLDS = 5

CONFIG_PRESETS = {
    'fast': {
        'xgboost': {
            'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.1,
            'subsample': 0.8, 'colsample_bytree': 0.6,
            'reg_alpha': 0.1, 'reg_lambda': 1.0,
            'early_stopping_rounds': 20,
        },
        'cnn': {
            'epochs': 50, 'batch_size': 64, 'patience': 10, 'lr': 1e-3,
            'dropout_1': 0.3, 'dropout_2': 0.2,
            'filters': [16, 32, 64], 'kernel_sizes': [7, 5, 3],
        },
        'rocket': {'num_kernels': 5000},
        'tabnet': {
            'n_d': 8, 'n_a': 8, 'n_steps': 3, 'gamma': 1.5,
            'lambda_sparse': 1e-3, 'max_epochs': 50, 'patience': 10,
            'batch_size': 256, 'virtual_batch_size': 128,
            'lr': 2e-2, 'mask_type': 'entmax',
        },
        'optuna': {'n_trials_l1': 30, 'n_trials_l2': 20, 'n_trials_l3': 20},
    },
    'best': {
        'xgboost': {
            'n_estimators': 1000, 'max_depth': 6, 'learning_rate': 0.02,
            'subsample': 0.8, 'colsample_bytree': 0.5,
            'reg_alpha': 0.1, 'reg_lambda': 1.0,
            'early_stopping_rounds': 80,
        },
        'cnn': {
            'epochs': 300, 'batch_size': 32, 'patience': 25, 'lr': 5e-4,
            'dropout_1': 0.4, 'dropout_2': 0.3,
            'filters': [32, 64, 128], 'kernel_sizes': [7, 5, 3],
        },
        'rocket': {'num_kernels': 20000},
        'tabnet': {
            'n_d': 24, 'n_a': 24, 'n_steps': 5, 'gamma': 1.5,
            'lambda_sparse': 1e-3, 'max_epochs': 300, 'patience': 30,
            'batch_size': 128, 'virtual_batch_size': 64,
            'lr': 2e-2, 'mask_type': 'entmax',
        },
        'optuna': {'n_trials_l1': 100, 'n_trials_l2': 80, 'n_trials_l3': 80},
    },
}
