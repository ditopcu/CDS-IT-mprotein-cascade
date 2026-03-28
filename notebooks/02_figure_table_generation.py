# -*- coding: utf-8 -*-
"""Cascade M-Protein CDS — Figure & Table Generation

# Cascade M-Protein CDS — Publication Figure & Table Generation

Generates all publication-quality figures and tables from pipeline outputs.

**Input:** `.pkl` files produced by `01_cascade_training_pipeline.ipynb`
**Output:** Individual CSV files → combined Excel; PNG (300 dpi) + TIFF (600 dpi) figures

## Prerequisites

The following outputs are expected from `01_cascade_training_pipeline.ipynb`:

- ✅ `L1_oof_predictions.pkl` — L1 OOF probabilities & CV metrics (7 models)
- ✅ `L2_oof_predictions.pkl` — L2 OOF probabilities & CV metrics (4 models)
- ✅ `L3_oof_predictions.pkl` — L3 OOF probabilities & CV metrics (4 models)
- ✅ `L4_cascade_results.pkl` — 9-class cascade predictions & bootstrap CIs
- ✅ `L4_baseline_comparison.pkl` — Flat LR & XGBoost baselines
- ✅ `L4_ext_validation_results.pkl` — External validation predictions
- ✅ `L4_calibration_validation.pkl` — Calibration, conformal, confidence results
- ✅ `optuna_best_params.pkl` — Optuna hyperparameters per level
- ✅ Model `.pkl` files (L1/L2/L3 XGBoost-Peak-Optuna)
- ✅ `fold_assignments.pkl`, `dataset.pkl`, `peak_features.pkl`
- ⬜ `learning_curves.pkl` — *TODO: add to Notebook 1*
- ⬜ Tuning-free cascade results — *TODO: add to Notebook 1*
"""

# ══════════════════════════════════════════════════════════════
# 0. SETUP
# ══════════════════════════════════════════════════════════════

# %%capture
# !pip install shap

import numpy as np
import pandas as pd
import pickle, json, os, sys, time, warnings, gc
from pathlib import Path
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score, accuracy_score,
    classification_report, cohen_kappa_score, brier_score_loss,
    balanced_accuracy_score, precision_recall_fscore_support,
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency
import shap

warnings.filterwarnings('ignore')

"""### 0.1 Paths"""

from google.colab import drive
drive.mount('/content/drive')
BASE = Path('/content/drive/MyDrive/001_IT_ML_github')

sys.path.insert(0, str(BASE))

from src.constants import (
    REGIONS, CHANNELS, CLASS9_NAMES, L2_CLASSES, L3_CLASSES,
    L1_METRICS, L2_METRICS, SEED, N_FOLDS, CH_COLORS, REGION_COLORS,
)
from src.features import extract_all_features
from src.evaluation import (
    eval_binary, eval_multiclass, compute_cascade_metrics, bootstrap_ci,
    attribute_errors,
)
from src.cascade import assemble_cascade_oof, run_external_cascade, get_fold_models
from src.calibration import (
    compute_ece, compute_mce, scaled_brier_score,
    calibrate_oof_isotonic, calibration_report, compute_classwise_ece,
)
from src.confidence import (
    compute_cascade_confidence, build_cascade_proba_9class,
    conformal_prediction, validate_compound_calibration,
    validate_level_independence,
)
from src.explainability import aggregate_shap_by_region
from src.utils import Timer, save_pickle, load_pickle

PROCESSED = BASE / 'data' / 'processed'
MODELS    = BASE / 'models'
RESULTS   = BASE / 'results'
CSV_DIR   = RESULTS / 'csv'
FIG_DIR   = RESULTS / 'publication_figures'
TIFF_DIR  = FIG_DIR / 'TIFF_600DPI'
PNG_DIR   = FIG_DIR / 'PNG_300DPI'

for d in [CSV_DIR, TIFF_DIR, PNG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

"""### 0.2 Design System"""

sns.set_style('whitegrid')
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
})

# Color palette
C_INTERNAL = '#2E75B6'
C_EXTERNAL = '#ED7D31'
C_HIGH     = '#70AD47'
C_MEDIUM   = '#FFC000'
C_LOW      = '#FF0000'
C_POS_SHAP = '#E53935'
C_NEG_SHAP = '#1565C0'

# Display order for 9 classes
CLASS_ORDER = ['Negative', 'IgG-κ', 'IgG-λ', 'IgA-κ', 'IgA-λ',
               'IgM-κ', 'IgM-λ', 'Free-κ', 'Free-λ']
CLASS_ORDER_INTERNAL = ['NEGATIVE', 'IGG_KAPPA', 'IGG_LAMBDA', 'IGA_KAPPA', 'IGA_LAMBDA',
                        'IGM_KAPPA', 'IGM_LAMBDA', 'FREE_KAPPA', 'FREE_LAMBDA']

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def save_fig(fig, filename):
    """Save figure as TIFF (600 dpi, LZW) and PNG (300 dpi)."""
    base = filename.replace('.png', '').replace('.tiff', '').replace('.tif', '')
    tiff_path = TIFF_DIR / f'{base}.tif'
    png_path  = PNG_DIR / f'{base}.png'
    try:
        fig.savefig(str(tiff_path), dpi=600, bbox_inches='tight', format='tif',
                    facecolor='white')
        with Image.open(tiff_path) as img:
            img.save(tiff_path, compression='tiff_lzw')
        fig.savefig(str(png_path), dpi=300, bbox_inches='tight', format='png',
                    facecolor='white')
        print(f'  Saved: {base} (TIFF + PNG)')
    except Exception as e:
        print(f'  Error saving {base}: {e}')

"""### 0.3 Load All Data"""

print('Loading pipeline outputs...')

D  = load_pickle(PROCESSED / 'dataset.pkl')
PF = load_pickle(PROCESSED / 'peak_features.pkl')
FA = load_pickle(RESULTS / 'fold_assignments.pkl')
L1 = load_pickle(RESULTS / 'L1_oof_predictions.pkl')
L2 = load_pickle(RESULTS / 'L2_oof_predictions.pkl')
L3 = load_pickle(RESULTS / 'L3_oof_predictions.pkl')
L4 = load_pickle(RESULTS / 'L4_cascade_results.pkl')
EXT = load_pickle(RESULTS / 'L4_ext_validation_results.pkl')

cal_path = RESULTS / 'L4_calibration_validation.pkl'
CAL = load_pickle(cal_path) if cal_path.exists() else None

baseline_path = RESULTS / 'L4_baseline_comparison.pkl'
FLAT = load_pickle(baseline_path) if baseline_path.exists() else None

optuna_path = RESULTS / 'optuna_best_params.pkl'
OPTUNA = load_pickle(optuna_path) if optuna_path.exists() else None

# Core arrays
X_3d         = D['X_3d']
X_peak_all   = PF['X_peak']
feat_names   = PF.get('columns', PF.get('feature_names', [f'f{i}' for i in range(399)]))
fold_indices = FA['fold_indices']

y_binary     = L1['y_true']
y_class9     = np.array(D['y_class9'])
N            = len(y_binary)

pos_mask     = (y_binary == 1)
pos_idx      = np.where(pos_mask)[0]
N_pos        = len(pos_idx)

# L1
L1_MODEL_KEY = 'XGBoost-Peak-Optuna'
if L1_MODEL_KEY not in L1['oof_proba']:
    L1_MODEL_KEY = 'xgb_peak_optuna'
l1_proba = L1['oof_proba'][L1_MODEL_KEY]

if isinstance(L4.get('best_threshold'), (int, float)):
    L1_THRESHOLD = L4['best_threshold']
elif 'threshold_analysis' in L1:
    for k in [L1_MODEL_KEY, 'xgb_peak_optuna', 'XGBoost-Peak-Optuna']:
        if k in L1['threshold_analysis']:
            L1_THRESHOLD = L1['threshold_analysis'][k]['youden']['threshold']
            break
else:
    L1_THRESHOLD = 0.4722

# L2 / L3
l2_key = 'XGBoost-Peak-Optuna'
l2_proba = L2['oof_prob'][l2_key]
l2_pred  = L2['oof_pred'][l2_key]
l2_y     = L2['y_true']

l3_key = 'XGBoost-Peak-Optuna'
if 'oof_proba' in L3 and l3_key in L3['oof_proba']:
    l3_proba = L3['oof_proba'][l3_key]
elif 'oof_prob' in L3 and l3_key in L3['oof_prob']:
    l3_full = L3['oof_prob'][l3_key]
    l3_proba = l3_full[:, 1] if l3_full.ndim == 2 else l3_full
l3_pred = L3['oof_pred'][l3_key]
l3_y    = L3['y_true']

fold_idx_pos = fold_indices[pos_idx]

# Cascade
best_pred = L4['best_pred']
ci_results = L4.get('ci_results', {})

# External
ext_pred     = EXT['ext_pred']
ext_l1_proba = EXT['ext_l1_proba']
ext_l2_proba = EXT.get('ext_l2_proba', None)
ext_l3_proba = EXT.get('ext_l3_proba', None)
y_ext_true   = np.array(D['y_ext_class9'])
X_ext_3d     = D['X_ext_3d']

# External peak features
ext_peak_path = PROCESSED / 'ext_peak_features.pkl'
if ext_peak_path.exists():
    X_ext_peak = load_pickle(ext_peak_path)['X_peak']
else:
    print('Extracting external peak features...')
    df_ext = extract_all_features(X_ext_3d, CHANNELS)
    df_ext = df_ext.replace([np.inf, -np.inf], 0).fillna(0)
    X_ext_peak = df_ext.values.astype(np.float32)
    save_pickle({'X_peak': X_ext_peak, 'columns': list(df_ext.columns)}, ext_peak_path)

# Load models
l1_models = get_fold_models(load_pickle(MODELS / 'L1_xgb_peak_optuna_models.pkl'))
l2_models = get_fold_models(load_pickle(MODELS / 'L2_xgb_peak_optuna_models.pkl'))
l3_models = get_fold_models(load_pickle(MODELS / 'L3_xgb_peak_optuna_models.pkl'))

# Demographics
demog_path = PROCESSED / 'demographics.xlsx'
if demog_path.exists():
    demog_all = pd.read_excel(demog_path)
    demog_train = demog_all[demog_all['cohort'] == 'Development'].reset_index(drop=True)
    demog_ext = demog_all[demog_all['cohort'] == 'External'].reset_index(drop=True)
    has_demog = True
    print(f'  Demographics: train={len(demog_train)}, ext={len(demog_ext)}')
else:
    has_demog = False
    demog_train = None
    demog_ext = None
    print('  ⚠ demographics.xlsx not found — Table 1 & S12 will use placeholders')

# Calibration
l1_cal = calibrate_oof_isotonic(l1_proba, y_binary, fold_indices)
l3_cal = calibrate_oof_isotonic(l3_proba, l3_y, fold_idx_pos)

l1_ece_raw, _ = compute_ece(y_binary, l1_proba)
l1_ece_cal, _ = compute_ece(y_binary, l1_cal)
l2_ece, _     = compute_ece(l2_y, l2_proba)
l3_ece_raw, _ = compute_ece(l3_y, l3_proba)
l3_ece_cal, _ = compute_ece(l3_y, l3_cal)

l1_bss, _ = scaled_brier_score(y_binary, l1_proba)
l2_bss, _ = scaled_brier_score(l2_y, l2_proba)
l3_bss, _ = scaled_brier_score(l3_y, l3_proba)

# Compound probabilities
cascade_proba_9 = build_cascade_proba_9class(
    l1_proba, l2_proba, l3_proba, L1_THRESHOLD, pos_idx)

# Conformal
conformal_results = {}
for alpha in [0.01, 0.05, 0.10, 0.15, 0.20]:
    conformal_results[alpha] = conformal_prediction(
        y_class9, cascade_proba_9, alpha=alpha)

# Compound calibration
compound_report = validate_compound_calibration(
    cascade_proba_9, y_class9, CLASS9_NAMES, n_bins=10, strategy='quantile')

# Independence test
l2_errors = (l2_pred != l2_y)
l3_errors = (l3_pred != l3_y)
indep_result = validate_level_independence(l2_errors, l3_errors)

# Confidence zones
CONF_HIGH, CONF_LOW = 0.7, 0.3
conf_df = compute_cascade_confidence(l1_proba, l2_proba, l3_proba, L1_THRESHOLD, pos_idx)
conf_df['true_class'] = y_class9
conf_df['pred_class'] = best_pred
conf_df['correct'] = (conf_df['true_class'] == conf_df['pred_class']).astype(int)
conf_df['zone'] = conf_df['cascade_conf'].apply(
    lambda c: 'HIGH' if c >= CONF_HIGH else ('MEDIUM' if c >= CONF_LOW else 'LOW'))

# Error attribution
error_df = attribute_errors(
    y_true_9=y_class9, cascade_pred=best_pred,
    l1_proba=l1_proba, l1_threshold=L1_THRESHOLD,
    l2_pred_full=l2_pred, l3_pred_full=l3_pred,
    y_binary_enc=y_binary, y_heavy=l2_y, y_light=l3_y, pos_idx=pos_idx)

print(f'\n  N={N}, N_pos={N_pos}, L1_threshold={L1_THRESHOLD:.4f}')
print(f'  External: {len(y_ext_true)} samples')
print(f'  Cascade accuracy: {accuracy_score(y_class9, best_pred):.4f}')
print('Data loading complete ✓')

# ══════════════════════════════════════════════════════════════
# 1. SHAP COMPUTATION — ALL SAMPLES, ALL LEVELS, BOTH COHORTS
# ══════════════════════════════════════════════════════════════

"""## 1. Load SHAP Values (pre-computed by Notebook 1 §18)
All samples × all levels × both cohorts. If cache not found,
run §18 in `01_cascade_training_pipeline.ipynb` first.
"""


shap_cache_path = RESULTS / 'shap_all_levels_all_samples.pkl'

if shap_cache_path.exists():
    print('Loading SHAP cache from Notebook 1...')
    SHAP_ALL = load_pickle(shap_cache_path)
    shap_l1_int = SHAP_ALL['l1_internal']
    shap_l1_ext = SHAP_ALL['l1_external']
    shap_l2_int = SHAP_ALL['l2_internal']
    shap_l2_ext = SHAP_ALL['l2_external']
    shap_l3_int = SHAP_ALL['l3_internal']
    shap_l3_ext = SHAP_ALL['l3_external']
    shap_l1_base_int = SHAP_ALL.get('l1_base_internal', None)
    shap_l1_base_ext = SHAP_ALL.get('l1_base_external', None)
    print(f'  L1 int: {shap_l1_int.shape}, ext: {shap_l1_ext.shape}')
    print(f'  L2 int: {shap_l2_int.shape}, ext: {shap_l2_ext.shape}')
    print(f'  L3 int: {shap_l3_int.shape}, ext: {shap_l3_ext.shape}')
else:
    print('⚠ SHAP cache not found!')
    print('  Run §18 in 01_cascade_training_pipeline.ipynb first.')
    print('  SHAP-dependent figures will be skipped.')
    shap_l1_int = shap_l1_ext = None
    shap_l2_int = shap_l2_ext = None
    shap_l3_int = shap_l3_ext = None
    shap_l1_base_int = shap_l1_base_ext = None

print(f'\nSHAP ready.' if shap_l1_int is not None else '\nSHAP not available.')

# ══════════════════════════════════════════════════════════════
# 2. TABLES — Individual CSVs
# ══════════════════════════════════════════════════════════════

"""## 2. Tables — CSV Generation
All metrics rounded to 3 decimal places per specification.
"""


def fmt3(v):
    """Format float to 3 decimal places."""
    if isinstance(v, (int, np.integer)): return str(v)
    if isinstance(v, (float, np.floating)): return f'{v:.3f}'
    return str(v)

def fmt_pct(v, dec=1):
    """Format as percentage."""
    return f'{v*100:.{dec}f}%' if isinstance(v, (float, np.floating)) else str(v)

def fmt_ci(vals, n_boot=1000):
    """Compute and format bootstrap CI."""
    lo = np.percentile(vals, 2.5)
    hi = np.percentile(vals, 97.5)
    return f'({lo:.3f}–{hi:.3f})'

"""### Table 1 — Demographic and Clinical Characteristics"""

if has_demog:
    t1_rows = []
    t1_rows.append({'Characteristic': 'N', 'Development (n=2,219)': str(N),
                     'External (n=498)': str(len(y_ext_true))})
    t1_rows.append({'Characteristic': 'Age, years (mean ± SD)',
                     'Development (n=2,219)': f'{demog_train["age"].mean():.1f} ± {demog_train["age"].std():.1f}',
                     'External (n=498)': f'{demog_ext["age"].mean():.1f} ± {demog_ext["age"].std():.1f}' if demog_ext is not None else '—'})
    n_f_dev = (demog_train['sex'] == 'Female').sum()
    n_f_ext = (demog_ext['sex'] == 'Female').sum() if demog_ext is not None else 0
    t1_rows.append({'Characteristic': 'Female, n (%)',
                     'Development (n=2,219)': f'{n_f_dev} ({100*n_f_dev/N:.1f}%)',
                     'External (n=498)': f'{n_f_ext} ({100*n_f_ext/len(y_ext_true):.1f}%)' if demog_ext is not None else '—'})
    t1_rows.append({'Characteristic': 'Class Distribution', 'Development (n=2,219)': '', 'External (n=498)': ''})

    for cls_int, cls_disp in zip(CLASS_ORDER_INTERNAL, CLASS_ORDER):
        n_dev = (y_class9 == cls_int).sum()
        n_ext = (y_ext_true == cls_int).sum()
        t1_rows.append({
            'Characteristic': f'  {cls_disp}',
            'Development (n=2,219)': f'{n_dev} ({100*n_dev/N:.1f}%)',
            'External (n=498)': f'{n_ext} ({100*n_ext/len(y_ext_true):.1f}%)',
        })
    table_1 = pd.DataFrame(t1_rows)
else:
    # Class distribution only
    t1_rows = [{'Characteristic': 'N', 'Development (n=2,219)': str(N),
                'External (n=498)': str(len(y_ext_true))}]
    t1_rows.append({'Characteristic': 'Class Distribution', 'Development (n=2,219)': '', 'External (n=498)': ''})
    for cls_int, cls_disp in zip(CLASS_ORDER_INTERNAL, CLASS_ORDER):
        n_dev = (y_class9 == cls_int).sum()
        n_ext = (y_ext_true == cls_int).sum()
        t1_rows.append({
            'Characteristic': f'  {cls_disp}',
            'Development (n=2,219)': f'{n_dev} ({100*n_dev/N:.1f}%)',
            'External (n=498)': f'{n_ext} ({100*n_ext/len(y_ext_true):.1f}%)',
        })
    table_1 = pd.DataFrame(t1_rows)

table_1.to_csv(CSV_DIR / 'Table_1_Demographics.csv', index=False)
print('Table 1: Demographics ✓')

"""### Table 2 — Classification Performance of the 3-Level Cascade"""

# Compute per-level metrics
ext_metrics = compute_cascade_metrics(y_ext_true, ext_pred)

# L1 metrics
l1_sens = ((l1_proba >= L1_THRESHOLD) & (y_binary == 1)).sum() / (y_binary == 1).sum()
l1_spec = ((l1_proba < L1_THRESHOLD) & (y_binary == 0)).sum() / (y_binary == 0).sum()
l1_auc  = roc_auc_score(y_binary, l1_proba)
l1_f1   = f1_score(y_binary, (l1_proba >= L1_THRESHOLD).astype(int), average='macro')

# External L1
ext_y_bin = np.array([0 if c == 'NEGATIVE' else 1 for c in y_ext_true])
ext_l1_sens = ((ext_l1_proba >= L1_THRESHOLD) & (ext_y_bin == 1)).sum() / max((ext_y_bin == 1).sum(), 1)
ext_l1_spec = ((ext_l1_proba < L1_THRESHOLD) & (ext_y_bin == 0)).sum() / max((ext_y_bin == 0).sum(), 1)
ext_l1_auc  = roc_auc_score(ext_y_bin, ext_l1_proba)

# L2 macro F1
l2_macro_f1 = f1_score(l2_y, l2_pred, average='macro')

# L3
l3_auc = roc_auc_score(l3_y, l3_proba)
l3_acc = accuracy_score(l3_y, (l3_proba >= 0.5).astype(int))

# 9-class overall
oof_acc = accuracy_score(y_class9, best_pred)
oof_f1m = f1_score(y_class9, best_pred, labels=CLASS9_NAMES, average='macro', zero_division=0)
oof_f1w = f1_score(y_class9, best_pred, labels=CLASS9_NAMES, average='weighted', zero_division=0)
oof_kappa = cohen_kappa_score(y_class9, best_pred)

t2_rows = [
    {'Level': 'L1 (Binary Detection)', 'Metric': 'AUC-ROC', 'Internal (OOF)': fmt3(l1_auc), 'External': fmt3(ext_l1_auc)},
    {'Level': '', 'Metric': 'Sensitivity', 'Internal (OOF)': fmt3(l1_sens), 'External': fmt3(ext_l1_sens)},
    {'Level': '', 'Metric': 'Specificity', 'Internal (OOF)': fmt3(l1_spec), 'External': fmt3(ext_l1_spec)},
    {'Level': '', 'Metric': 'F1 (macro)', 'Internal (OOF)': fmt3(l1_f1), 'External': ''},
    {'Level': '', 'Metric': 'Threshold', 'Internal (OOF)': fmt3(L1_THRESHOLD), 'External': f'{L1_THRESHOLD:.3f} (frozen)'},
    {'Level': 'L2 (Heavy Chain)', 'Metric': 'Macro F1', 'Internal (OOF)': fmt3(l2_macro_f1), 'External': ''},
    {'Level': 'L3 (Light Chain)', 'Metric': 'AUC-ROC', 'Internal (OOF)': fmt3(l3_auc), 'External': ''},
    {'Level': '', 'Metric': 'Accuracy', 'Internal (OOF)': fmt3(l3_acc), 'External': ''},
    {'Level': '9-Class Overall', 'Metric': 'Accuracy', 'Internal (OOF)': fmt3(oof_acc),
     'External': fmt3(ext_metrics['accuracy'])},
    {'Level': '', 'Metric': 'Macro F1', 'Internal (OOF)': fmt3(oof_f1m),
     'External': fmt3(ext_metrics['macro_f1'])},
    {'Level': '', 'Metric': 'Weighted F1', 'Internal (OOF)': fmt3(oof_f1w),
     'External': fmt3(ext_metrics['weighted_f1'])},
    {'Level': '', 'Metric': "Cohen's κ", 'Internal (OOF)': fmt3(oof_kappa),
     'External': fmt3(ext_metrics['kappa'])},
]

# Add CIs if available
if ci_results:
    for row in t2_rows:
        if row['Level'] == '9-Class Overall':
            key_map = {'Accuracy': 'Accuracy', 'Macro F1': 'Macro F1',
                       'Weighted F1': 'Weighted F1', "Cohen's κ": 'Kappa'}
            ci_key = key_map.get(row['Metric'])
            if ci_key and ci_key in ci_results:
                _, lo, hi = ci_results[ci_key]
                row['95% CI'] = f'({lo:.3f}–{hi:.3f})'

table_2 = pd.DataFrame(t2_rows)
table_2.to_csv(CSV_DIR / 'Table_2_Cascade_Performance.csv', index=False)
print('Table 2: Cascade Performance ✓')

"""### Table S2 — Cascade vs Flat Model Comparison"""

if FLAT is not None:
    flat_metrics = FLAT.get('metrics', {})
    s2_rows = []
    for model_name, mdict in [('Flat Logistic Regression', flat_metrics.get('LR', {})),
                                ('Flat XGBoost', flat_metrics.get('XGB', {}))]:
        s2_rows.append({
            'Model': model_name,
            'Accuracy': fmt3(mdict.get('accuracy', 0)),
            'Macro F1': fmt3(mdict.get('macro_f1', 0)),
            'Weighted F1': fmt3(mdict.get('weighted_f1', 0)),
            "Cohen's κ": fmt3(mdict.get('kappa', 0)),
        })
    s2_rows.append({
        'Model': 'Cascade XGBoost (selected)',
        'Accuracy': fmt3(oof_acc),
        'Macro F1': fmt3(oof_f1m),
        'Weighted F1': fmt3(oof_f1w),
        "Cohen's κ": fmt3(oof_kappa),
    })
    table_s2 = pd.DataFrame(s2_rows)
else:
    table_s2 = pd.DataFrame([{'Model': '⚠ Run Notebook 1 flat baselines first'}])
table_s2.to_csv(CSV_DIR / 'Table_S2_Cascade_vs_Flat.csv', index=False)
print('Table S2: Cascade vs Flat ✓')

"""### Table S3 — L1 Algorithm Comparison (7 Models)"""

s3_rows = []
l1_cv = L1.get('cv_metrics', {})
for model_name in ['XGBoost-Raw', 'XGBoost-Peak', '1D-CNN', 'MiniRocket',
                    'MiniRocket+Peak', 'TabNet', 'XGBoost-Peak-Optuna']:
    if model_name not in l1_cv:
        continue
    cv = l1_cv[model_name]
    row = {'Algorithm': model_name}
    for m in ['accuracy', 'sensitivity', 'specificity', 'f1_macro', 'auc_roc']:
        vals = cv.get(m, [])
        if vals:
            row[m.replace('_', ' ').title()] = f'{np.mean(vals):.3f} ± {np.std(vals):.3f}'
        else:
            row[m.replace('_', ' ').title()] = '—'
    s3_rows.append(row)
table_s3 = pd.DataFrame(s3_rows)
table_s3.to_csv(CSV_DIR / 'Table_S3_L1_Algorithm_Comparison.csv', index=False)
print(f'Table S3: L1 Algorithm Comparison ({len(s3_rows)} models) ✓')

"""### Table S4 — L2 Heavy Chain Comparison"""

s4_rows = []
l2_cv = L2.get('cv_metrics', {})
for model_name in ['XGBoost-Peak', '1D-CNN', 'MiniRocket', 'XGBoost-Peak-Optuna']:
    if model_name not in l2_cv:
        continue
    cv = l2_cv[model_name]
    row = {'Algorithm': model_name}
    for m in ['balanced_acc', 'macro_f1', 'weighted_f1', 'auc_ovr', 'kappa']:
        vals = cv.get(m, [])
        if vals:
            row[m.replace('_', ' ').title()] = f'{np.mean(vals):.3f} ± {np.std(vals):.3f}'
        else:
            row[m.replace('_', ' ').title()] = '—'
    s4_rows.append(row)
table_s4 = pd.DataFrame(s4_rows)
table_s4.to_csv(CSV_DIR / 'Table_S4_L2_Algorithm_Comparison.csv', index=False)
print(f'Table S4: L2 Algorithm Comparison ({len(s4_rows)} models) ✓')

"""### Table S5 — L3 Light Chain Comparison"""

s5_rows = []
l3_cv = L3.get('cv_metrics', {})
for model_name in ['XGBoost-Peak', '1D-CNN', 'MiniRocket', 'XGBoost-Peak-Optuna']:
    if model_name not in l3_cv:
        continue
    cv = l3_cv[model_name]
    row = {'Algorithm': model_name}
    for m in ['accuracy', 'sensitivity', 'specificity', 'auc_roc']:
        vals = cv.get(m, [])
        if vals:
            row[m.replace('_', ' ').title()] = f'{np.mean(vals):.3f} ± {np.std(vals):.3f}'
        else:
            row[m.replace('_', ' ').title()] = '—'
    s5_rows.append(row)
table_s5 = pd.DataFrame(s5_rows)
table_s5.to_csv(CSV_DIR / 'Table_S5_L3_Algorithm_Comparison.csv', index=False)
print(f'Table S5: L3 Algorithm Comparison ({len(s5_rows)} models) ✓')

"""### Table S6 — Optimized Hyperparameters"""

param_space = {
    'n_estimators':     ('200–1500', 'int'),
    'max_depth':        ('3–12', 'int'),
    'learning_rate':    ('0.005–0.3', 'float (log)'),
    'subsample':        ('0.6–1.0', 'float'),
    'colsample_bytree': ('0.3–0.8', 'float'),
    'reg_alpha':        ('0.001–10', 'float (log)'),
    'reg_lambda':       ('0.001–10', 'float (log)'),
    'min_child_weight': ('1–10', 'int'),
}

s6_rows = []
for param, (rng, ptype) in param_space.items():
    row = {'Parameter': param, 'Search Range': rng, 'Type': ptype}
    if OPTUNA:
        for level in ['L1', 'L2', 'L3']:
            bp = OPTUNA.get(level, {})
            if param in bp:
                v = bp[param]
                row[f'{level} Selected'] = f'{v:.4f}' if isinstance(v, float) else str(v)
            else:
                row[f'{level} Selected'] = '—'
    s6_rows.append(row)
table_s6 = pd.DataFrame(s6_rows)
table_s6.to_csv(CSV_DIR / 'Table_S6_Hyperparameters.csv', index=False)
print('Table S6: Hyperparameters ✓')

"""### Table S7 — Conformal Prediction at Multiple Significance Levels"""

s7_rows = []
for alpha in sorted(conformal_results.keys()):
    cr = conformal_results[alpha]
    s7_rows.append({
        'α (Significance)': f'{alpha:.2f}',
        'Empirical Coverage': fmt3(cr['coverage']),
        'Mean Set Size': f'{cr["mean_set_size"]:.2f}',
        'Median Set Size': f'{cr["median_set_size"]:.0f}',
        'Target Coverage': fmt3(1 - alpha),
        'Achieved': '✓' if cr['coverage'] >= (1 - alpha) else '✗',
    })
table_s7 = pd.DataFrame(s7_rows)
table_s7.to_csv(CSV_DIR / 'Table_S7_Conformal_Prediction.csv', index=False)
print('Table S7: Conformal Prediction ✓')

"""### Table S8 — Confidence-Based Clinical Triaging"""

zone_stats_int = conf_df.groupby('zone').agg(
    N=('correct', 'count'), Accuracy=('correct', 'mean'),
    Errors=('correct', lambda x: (x == 0).sum()),
    Mean_Confidence=('cascade_conf', 'mean')
).reindex(['HIGH', 'MEDIUM', 'LOW'])

# External zones
ext_conf = compute_cascade_confidence(
    ext_l1_proba, ext_l2_proba if ext_l2_proba is not None else np.zeros((len(y_ext_true), 4)),
    ext_l3_proba if ext_l3_proba is not None else np.zeros(len(y_ext_true)),
    L1_THRESHOLD, np.where(ext_l1_proba >= L1_THRESHOLD)[0])
ext_conf['correct'] = (y_ext_true == ext_pred).astype(int)
ext_conf['zone'] = ext_conf['cascade_conf'].apply(
    lambda c: 'HIGH' if c >= CONF_HIGH else ('MEDIUM' if c >= CONF_LOW else 'LOW'))

zone_stats_ext = ext_conf.groupby('zone').agg(
    N=('correct', 'count'), Accuracy=('correct', 'mean'),
    Errors=('correct', lambda x: (x == 0).sum()),
    Mean_Confidence=('cascade_conf', 'mean')
).reindex(['HIGH', 'MEDIUM', 'LOW'])

s8_rows = []
actions = {'HIGH': 'Auto-verify and report', 'MEDIUM': 'Technician verification', 'LOW': 'Expert referral'}
for zone in ['HIGH', 'MEDIUM', 'LOW']:
    if zone in zone_stats_int.index:
        r = zone_stats_int.loc[zone]
        s8_rows.append({
            'Cohort': 'Internal (OOF)', 'Zone': zone,
            'N': int(r['N']), 'Proportion': fmt_pct(r['N'] / N),
            'Accuracy': fmt3(r['Accuracy']), 'Errors': int(r['Errors']),
            'Mean Confidence': fmt3(r['Mean_Confidence']),
            'Recommended Action': actions[zone],
        })
    if zone in zone_stats_ext.index:
        r = zone_stats_ext.loc[zone]
        s8_rows.append({
            'Cohort': 'External', 'Zone': zone,
            'N': int(r['N']), 'Proportion': fmt_pct(r['N'] / len(y_ext_true)),
            'Accuracy': fmt3(r['Accuracy']), 'Errors': int(r['Errors']),
            'Mean Confidence': fmt3(r['Mean_Confidence']),
            'Recommended Action': actions[zone],
        })
table_s8 = pd.DataFrame(s8_rows)
table_s8.to_csv(CSV_DIR / 'Table_S8_Confidence_Triaging.csv', index=False)
print('Table S8: Confidence Triaging ✓')

"""### Table S9 — Workload Impact Simulation"""

zone_rates = zone_stats_int['N'] / N
auto_rate   = zone_rates.get('HIGH', 0)
review_rate = zone_rates.get('MEDIUM', 0)
expert_rate = zone_rates.get('LOW', 0)

T_AUTO, T_TECH, T_EXPERT, T_MANUAL = 0.5, 2.0, 5.0, 3.0

s9_rows = []
for vol in [50, 100, 200]:
    n_auto   = round(vol * auto_rate)
    n_review = round(vol * review_rate)
    n_expert = vol - n_auto - n_review  # ensure sum = vol
    manual_min = vol * T_MANUAL
    ai_min = n_auto * T_AUTO + n_review * T_TECH + n_expert * T_EXPERT
    saved = manual_min - ai_min
    s9_rows.append({
        'Daily Tests': vol, 'Auto-Report (n)': n_auto,
        'Technician Review (n)': n_review, 'Expert Referral (n)': n_expert,
        'Manual Time (min)': f'{manual_min:.0f}',
        'AI-Assisted Time (min)': f'{ai_min:.1f}',
        'Time Saved (min)': f'{saved:.1f}',
        'Reduction (%)': fmt_pct(saved / manual_min),
    })
table_s9 = pd.DataFrame(s9_rows)
table_s9.to_csv(CSV_DIR / 'Table_S9_Workload_Simulation.csv', index=False)
print('Table S9: Workload Simulation ✓')

"""### Table S11 — Feature Importance by Cascade Level (SHAP + Gain)"""

def top_features_table(shap_int, shap_ext, models, feat_names, level, top_n=20):
    """Build top-N feature importance table with SHAP (int+ext) + Gain."""
    mean_shap_int = np.mean(np.abs(shap_int), axis=0)
    mean_shap_ext = np.mean(np.abs(shap_ext), axis=0)

    # Average gain across folds
    gains = np.zeros(len(feat_names))
    for m in models:
        imp = m.feature_importances_
        if len(imp) == len(feat_names):
            gains += imp
    gains /= len(models)

    rank_shap_int = np.argsort(-mean_shap_int)
    rank_shap_ext = np.argsort(-mean_shap_ext)
    rank_gain     = np.argsort(-gains)

    top_idx = rank_shap_int[:top_n]
    rows = []
    for rank, fi in enumerate(top_idx):
        rows.append({
            'Rank': rank + 1,
            'Feature': feat_names[fi],
            f'{level} SHAP (Internal)': f'{mean_shap_int[fi]:.4f}',
            f'{level} SHAP (External)': f'{mean_shap_ext[fi]:.4f}',
            f'{level} SHAP Rank (Ext)': int(np.where(rank_shap_ext == fi)[0][0]) + 1 if fi in rank_shap_ext else '—',
            f'{level} Gain': f'{gains[fi]:.4f}',
            f'{level} Gain Rank': int(np.where(rank_gain == fi)[0][0]) + 1 if fi in rank_gain else '—',
        })
    return rows

s11_l1 = top_features_table(shap_l1_int, shap_l1_ext, l1_models, feat_names, 'L1')
s11_l2 = top_features_table(shap_l2_int, shap_l2_ext, l2_models, feat_names, 'L2')
s11_l3 = top_features_table(shap_l3_int, shap_l3_ext, l3_models, feat_names, 'L3')

table_s11 = pd.DataFrame(s11_l1)
table_s11.to_csv(CSV_DIR / 'Table_S11_Feature_Importance_L1.csv', index=False)
pd.DataFrame(s11_l2).to_csv(CSV_DIR / 'Table_S11_Feature_Importance_L2.csv', index=False)
pd.DataFrame(s11_l3).to_csv(CSV_DIR / 'Table_S11_Feature_Importance_L3.csv', index=False)
print('Table S11: Feature Importance (L1/L2/L3) ✓')

"""### Table S12 — Fairness and Subgroup Performance"""

if has_demog and demog_train is not None:
    # Add prediction-dependent columns for fairness analysis
    dt_fair = demog_train.copy()
    dt_fair['correct'] = (y_class9 == best_pred).astype(int)
    dt_fair['age_group'] = pd.cut(dt_fair['age'], bins=[0, 40, 60, 75, 200],
                                   labels=['<40', '40-59', '60-74', '75+'])

    de_fair = None
    if demog_ext is not None:
        de_fair = demog_ext.copy()
        de_fair['correct'] = (y_ext_true == ext_pred).astype(int)
        de_fair['age_group'] = pd.cut(de_fair['age'], bins=[0, 40, 60, 75, 200],
                                       labels=['<40', '40-59', '60-74', '75+'])

    s12_rows = []
    for cohort_name, df in [('Internal', dt_fair), ('External', de_fair)]:
        if df is None:
            continue
        for subgroup, col in [('Sex', 'sex'), ('Age', 'age_group')]:
            for cat in sorted(df[col].dropna().unique(), key=str):
                mask = df[col] == cat
                n_sub = mask.sum()
                acc = df.loc[mask, 'correct'].mean()
                vals = df.loc[mask, 'correct'].values
                rng = np.random.RandomState(SEED)
                boot_accs = [rng.choice(vals, n_sub, replace=True).mean() for _ in range(1000)]
                lo, hi = np.percentile(boot_accs, [2.5, 97.5])
                s12_rows.append({
                    'Cohort': cohort_name, 'Subgroup': subgroup, 'Category': str(cat),
                    'N': n_sub, 'Accuracy': fmt3(acc), '95% CI': f'({lo:.3f}–{hi:.3f})',
                })
    table_s12 = pd.DataFrame(s12_rows)
else:
    table_s12 = pd.DataFrame([{'Note': 'Demographics not available'}])
table_s12.to_csv(CSV_DIR / 'Table_S12_Fairness_Subgroups.csv', index=False)
print('Table S12: Fairness ✓')

"""### Table S13 — Per-Class Classification Metrics (9-Class)"""

report_oof = classification_report(y_class9, best_pred, labels=CLASS9_NAMES,
                                    output_dict=True, zero_division=0)
report_ext = classification_report(y_ext_true, ext_pred, labels=CLASS9_NAMES,
                                    output_dict=True, zero_division=0)
s13_rows = []
for cls_int, cls_disp in zip(CLASS9_NAMES, ['Free-κ', 'Free-λ', 'IgA-κ', 'IgA-λ',
                                              'IgG-κ', 'IgG-λ', 'IgM-κ', 'IgM-λ', 'Negative']):
    row = {'Class': cls_disp, 'N (Internal)': int(report_oof[cls_int]['support']),
           'Precision (Internal)': fmt3(report_oof[cls_int]['precision']),
           'Recall (Internal)': fmt3(report_oof[cls_int]['recall']),
           'F1 (Internal)': fmt3(report_oof[cls_int]['f1-score'])}
    if cls_int in report_ext:
        row['N (External)'] = int(report_ext[cls_int]['support'])
        row['Precision (External)'] = fmt3(report_ext[cls_int]['precision'])
        row['Recall (External)'] = fmt3(report_ext[cls_int]['recall'])
        row['F1 (External)'] = fmt3(report_ext[cls_int]['f1-score'])
    else:
        row['N (External)'] = 0
        for m in ['Precision (External)', 'Recall (External)', 'F1 (External)']:
            row[m] = '—'
    s13_rows.append(row)
table_s13 = pd.DataFrame(s13_rows)
table_s13.to_csv(CSV_DIR / 'Table_S13_Per_Class_9Class.csv', index=False)
print('Table S13: Per-Class Metrics ✓')

"""### Table S14 — L1 Threshold Sensitivity Analysis"""

thresholds = np.arange(0.05, 1.0, 0.05)
s14_rows = []
for thr in thresholds:
    pred_bin = (l1_proba >= thr).astype(int)
    tp = ((pred_bin == 1) & (y_binary == 1)).sum()
    fn = ((pred_bin == 0) & (y_binary == 1)).sum()
    fp = ((pred_bin == 1) & (y_binary == 0)).sum()
    tn = ((pred_bin == 0) & (y_binary == 0)).sum()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    youden = sens + spec - 1
    s14_rows.append({
        'Threshold': f'{thr:.2f}', 'Sensitivity': fmt3(sens), 'Specificity': fmt3(spec),
        'Accuracy': fmt3(acc), 'F1': fmt3(f1), 'PPV': fmt3(ppv), 'NPV': fmt3(npv),
        'Youden J': fmt3(youden),
    })
table_s14 = pd.DataFrame(s14_rows)
table_s14.to_csv(CSV_DIR / 'Table_S14_L1_Threshold_Sensitivity.csv', index=False)
print('Table S14: Threshold Sensitivity ✓')

# ══════════════════════════════════════════════════════════════
# 3. COMBINED EXCEL
# ══════════════════════════════════════════════════════════════

"""## 3. Combined Excel — All Tables in One Workbook"""

excel_path = RESULTS / 'All_Tables_v14.xlsx'

sheet_map = OrderedDict([
    ('Table 1',   ('Table_1_Demographics.csv', 'Demographic and Clinical Characteristics')),
    ('Table 2',   ('Table_2_Cascade_Performance.csv', 'Classification Performance of the 3-Level Cascade System')),
    ('Table S2',  ('Table_S2_Cascade_vs_Flat.csv', 'Cascade vs Flat Model Comparison (Out-of-Fold)')),
    ('Table S3',  ('Table_S3_L1_Algorithm_Comparison.csv', 'L1 Binary Detection — Algorithm Comparison (7 Models)')),
    ('Table S4',  ('Table_S4_L2_Algorithm_Comparison.csv', 'L2 Heavy Chain — Algorithm Comparison')),
    ('Table S5',  ('Table_S5_L3_Algorithm_Comparison.csv', 'L3 Light Chain — Algorithm Comparison')),
    ('Table S6',  ('Table_S6_Hyperparameters.csv', 'Optimized Hyperparameters by Cascade Level')),
    ('Table S7',  ('Table_S7_Conformal_Prediction.csv', 'Conformal Prediction at Multiple Significance Levels')),
    ('Table S8',  ('Table_S8_Confidence_Triaging.csv', 'Confidence-Based Clinical Triaging Performance')),
    ('Table S9',  ('Table_S9_Workload_Simulation.csv', 'Workload Impact Simulation')),
    ('Table S11', ('Table_S11_Feature_Importance_L1.csv', 'Feature Importance — L1 (SHAP + Gain, Top 20)')),
    ('Table S11 L2', ('Table_S11_Feature_Importance_L2.csv', 'Feature Importance — L2 (SHAP + Gain, Top 20)')),
    ('Table S11 L3', ('Table_S11_Feature_Importance_L3.csv', 'Feature Importance — L3 (SHAP + Gain, Top 20)')),
    ('Table S12', ('Table_S12_Fairness_Subgroups.csv', 'Fairness and Subgroup Performance Metrics')),
    ('Table S13', ('Table_S13_Per_Class_9Class.csv', 'Per-Class Classification Metrics (9-Class)')),
    ('Table S14', ('Table_S14_L1_Threshold_Sensitivity.csv', 'L1 Threshold Sensitivity Analysis')),
])

with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    # Index sheet
    idx_rows = [{'Sheet': k, 'Title': v[1]} for k, v in sheet_map.items()]
    pd.DataFrame(idx_rows).to_excel(writer, sheet_name='Index', index=False)

    # Data sheets
    for sheet_name, (csv_name, title) in sheet_map.items():
        csv_path = CSV_DIR / csv_name
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # Write title in row 0, data from row 2
            title_df = pd.DataFrame([[title] + [''] * (len(df.columns) - 1)], columns=df.columns)
            combined = pd.concat([title_df, df], ignore_index=True)
            combined.to_excel(writer, sheet_name=sheet_name[:31], index=False, header=True)
        else:
            pd.DataFrame([{'Note': f'{csv_name} not found'}]).to_excel(
                writer, sheet_name=sheet_name[:31], index=False)

print(f'\nAll_Tables_v14.xlsx saved with {len(sheet_map)} sheets ✓')

# ══════════════════════════════════════════════════════════════
# 4. FIGURES
# ══════════════════════════════════════════════════════════════

"""## 4. Publication Figures
No figure numbers or titles on figures — only panel labels (A), (B), etc.
File names encode figure identity.
"""


"""### Figure 3 — Nine-Class Confusion Matrices (A: OOF, B: External)"""

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

for ax, (y_t, y_p, label, n_samp) in zip(axes, [
    (y_class9, best_pred, 'A', N),
    (y_ext_true, ext_pred, 'B', len(y_ext_true))
]):
    cm = confusion_matrix(y_t, y_p, labels=CLASS_ORDER_INTERNAL)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', ax=ax,
                xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER,
                vmin=0, vmax=1, cbar_kws={'shrink': 0.8},
                annot_kws={'size': 8})
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True', fontsize=10)
    ax.tick_params(labelsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    acc = accuracy_score(y_t, y_p)
    f1m = f1_score(y_t, y_p, labels=CLASS9_NAMES, average='macro', zero_division=0)
    kap = cohen_kappa_score(y_t, y_p)
    ax.text(0.02, -0.12, f'({label})  n = {n_samp}  |  Acc = {acc:.3f}  |  '
            f'Macro F1 = {f1m:.3f}  |  κ = {kap:.3f}',
            transform=ax.transAxes, fontsize=9, fontweight='bold')

plt.tight_layout()
save_fig(fig, 'Fig3_Confusion_Matrices')
plt.show()

"""### Figure S1 — Cascade vs Flat Per-Class F1 Comparison"""

if FLAT is not None:
    fig_s1, ax = plt.subplots(figsize=(14, 6))

    lr_pred = FLAT['flat_lr']['pred']
    xgb_flat_pred = FLAT['flat_xgb']['pred']

    f1_lr = f1_score(y_class9, lr_pred, labels=CLASS_ORDER_INTERNAL, average=None, zero_division=0)
    f1_xgb = f1_score(y_class9, xgb_flat_pred, labels=CLASS_ORDER_INTERNAL, average=None, zero_division=0)
    f1_cas = f1_score(y_class9, best_pred, labels=CLASS_ORDER_INTERNAL, average=None, zero_division=0)

    x = np.arange(len(CLASS_ORDER))
    w = 0.25
    ax.bar(x - w, f1_lr,  w, label='Flat LR', color='#BDBDBD', edgecolor='white')
    ax.bar(x,     f1_xgb, w, label='Flat XGBoost', color='#90CAF9', edgecolor='white')
    ax.bar(x + w, f1_cas, w, label='Cascade', color=C_INTERNAL, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_ORDER, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('F1 Score', fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_fig(fig_s1, 'FigS1_Cascade_vs_Flat_F1')
    plt.show()

"""### Figure S2 — Learning Curves and Sample Efficiency"""

lc_path = RESULTS / 'L4_learning_curves.pkl'
if lc_path.exists():
    lc = load_pickle(lc_path)
    fig_s2, axes = plt.subplots(1, 3, figsize=(16, 5))

    panels = [
        ('L1', 'AUC-ROC', C_INTERNAL),
        ('L2', 'Macro F1', '#7B1FA2'),
        ('L3', 'AUC-ROC', '#E65100'),
    ]
    labels = ['(A)', '(B)', '(C)']

    for ax, (level, metric_label, color), panel_lbl in zip(axes, panels, labels):
        d = lc[level]
        sizes = d['train_sizes']
        train_mean = d['train_scores'].mean(axis=1)
        train_std  = d['train_scores'].std(axis=1)
        val_mean   = d['val_scores'].mean(axis=1)
        val_std    = d['val_scores'].std(axis=1)

        ax.fill_between(sizes, train_mean - train_std, train_mean + train_std,
                         alpha=0.1, color='#757575')
        ax.plot(sizes, train_mean, 'o-', color='#757575', lw=1.5, ms=4, label='Training')
        ax.fill_between(sizes, val_mean - val_std, val_mean + val_std,
                         alpha=0.15, color=color)
        ax.plot(sizes, val_mean, 's-', color=color, lw=2, ms=5, label='Validation')

        gap = train_mean[-1] - val_mean[-1]
        ax.set_xlabel('Training Size', fontsize=10)
        ax.set_ylabel(metric_label, fontsize=10)
        ax.legend(fontsize=8, loc='lower right')
        ax.text(0.02, 0.98, f'{panel_lbl} {level}\nΔ = {gap:.3f}',
                transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')

    plt.tight_layout()
    save_fig(fig_s2, 'FigS2_Learning_Curves')
    plt.show()
else:
    print('⚠ L4_learning_curves.pkl not found — run Notebook 1 §15 first')

"""### Figure S3 — Per-Level and Compound Calibration Analysis (6-panel)"""

fig_s3, axes = plt.subplots(2, 3, figsize=(16, 10))

# Row 1: Pre/Post calibration per level
for col, (y_t, prob_raw, prob_cal, ece_r, ece_c, bss_val, title, color) in enumerate([
    (y_binary, l1_proba, l1_cal, l1_ece_raw, l1_ece_cal, l1_bss, 'L1 (Binary)', C_INTERNAL),
    (l2_y, l2_proba, l2_proba, l2_ece, l2_ece, l2_bss, 'L2 (Heavy Chain)', '#7B1FA2'),
    (l3_y, l3_proba, l3_cal, l3_ece_raw, l3_ece_cal, l3_bss, 'L3 (Light Chain)', '#E65100'),
]):
    ax = axes[0, col]
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, lw=0.8)

    if prob_raw.ndim == 2:
        conf = np.max(prob_raw, axis=1)
        correct = (np.argmax(prob_raw, axis=1) == y_t).astype(float)
        frac, mean_p = calibration_curve(correct, conf, n_bins=10)
        ax.plot(mean_p, frac, 's-', color=color, ms=4, lw=1.5,
                label=f'ECE = {ece_r:.3f}')
    else:
        frac_r, mean_r = calibration_curve(y_t, prob_raw, n_bins=10)
        frac_c, mean_c = calibration_curve(y_t, prob_cal, n_bins=10)
        ax.plot(mean_r, frac_r, 'o-', color='#E53935', ms=4, lw=1.2, alpha=0.7,
                label=f'Pre (ECE = {ece_r:.3f})')
        ax.plot(mean_c, frac_c, 's-', color=color, ms=4, lw=1.5,
                label=f'Post (ECE = {ece_c:.3f})')

    ax.text(0.05, 0.88, f'BSS = {bss_val:.3f}',
            transform=ax.transAxes, fontsize=7, fontfamily='monospace', va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.legend(fontsize=7, loc='lower right')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect('equal')
    if col == 0: ax.set_ylabel('Observed frequency')

# Row 2: Compound reliability per class group
c9_map = {c: i for i, c in enumerate(CLASS9_NAMES)}
y_9enc = np.array([c9_map[c] for c in y_class9])

class_groups = [
    ('Common', ['NEGATIVE', 'IGG_KAPPA', 'IGG_LAMBDA'], ['#2C2C2C', '#D32F2F', '#E57373']),
    ('IgA + Free', ['IGA_KAPPA', 'IGA_LAMBDA', 'FREE_KAPPA', 'FREE_LAMBDA'],
     ['#7B1FA2', '#CE93D8', '#E65100', '#FFB74D']),
    ('IgM (rare)', ['IGM_KAPPA', 'IGM_LAMBDA'], ['#1565C0', '#64B5F6']),
]

for gi, (grp_name, classes, colors) in enumerate(class_groups):
    ax = axes[1, gi]
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, lw=0.8)
    for cls, col in zip(classes, colors):
        c_idx = CLASS9_NAMES.index(cls)
        bt = (y_9enc == c_idx).astype(int)
        bp = cascade_proba_9[:, c_idx]
        n_c = bt.sum()
        if n_c < 10:
            continue
        eff_bins = max(3, min(10, n_c // 5))
        try:
            frac, mean_p = calibration_curve(bt, bp, n_bins=eff_bins, strategy='quantile')
            ece_c = compound_report['class_eces'].get(cls, 0)
            disp = cls.replace('_', '-').replace('FREE', 'Free').replace('IGG', 'IgG').replace('IGA', 'IgA').replace('IGM', 'IgM').replace('KAPPA', 'κ').replace('LAMBDA', 'λ').replace('NEGATIVE', 'Neg')
            ax.plot(mean_p, frac, 'o-', color=col, ms=3, lw=1,
                    label=f'{disp} (n={n_c})')
        except Exception:
            pass
    ax.set_title(f'Compound: {grp_name}', fontweight='bold', fontsize=10)
    ax.legend(fontsize=6, loc='lower right')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect('equal')
    ax.set_xlabel('Mean predicted probability')
    if gi == 0: ax.set_ylabel('Observed frequency')

plt.tight_layout()
save_fig(fig_s3, 'FigS3_Calibration_6Panel')
plt.show()

"""### Figure S4 — Conformal Prediction Coverage and Calibration (2-panel)"""

fig_s4, axes4 = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: Conformal trade-off
ax = axes4[0]
alphas = sorted(conformal_results.keys())
coverages = [conformal_results[a]['coverage'] for a in alphas]
set_sizes = [conformal_results[a]['mean_set_size'] for a in alphas]
ax.plot(alphas, coverages, 'o-', color=C_INTERNAL, lw=2, ms=6)
ax.axhline(0.95, color='gray', ls='--', alpha=0.5, lw=0.8)
ax.set_xlabel('α (significance level)', fontsize=10)
ax.set_ylabel('Coverage', color=C_INTERNAL, fontsize=10)
ax.tick_params(axis='y', labelcolor=C_INTERNAL)
ax.text(0.02, 0.02, '(A)', transform=ax.transAxes, fontsize=12, fontweight='bold')
ax2 = ax.twinx()
ax2.plot(alphas, set_sizes, 's-', color=C_EXTERNAL, lw=2, ms=6)
ax2.set_ylabel('Mean prediction set size', color=C_EXTERNAL, fontsize=10)
ax2.tick_params(axis='y', labelcolor=C_EXTERNAL)

# Panel B: L1 calibration
ax = axes4[1]
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, lw=0.8)
frac, mean_p = calibration_curve(y_binary, l1_cal, n_bins=10)
ax.plot(mean_p, frac, 's-', color=C_INTERNAL, ms=5, lw=1.5,
        label=f'L1 (ECE = {l1_ece_cal:.3f})')
ax.set_xlabel('Mean predicted probability', fontsize=10)
ax.set_ylabel('Observed frequency', fontsize=10)
ax.legend(fontsize=8, loc='lower right')
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.text(0.02, 0.02, '(B)', transform=ax.transAxes, fontsize=12, fontweight='bold')
plt.tight_layout()
save_fig(fig_s4, 'FigS4_Conformal_Calibration')
plt.show()

"""### Figure S5 — Confidence Zone Distribution and Triaging Accuracy"""

fig_s5, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: Zone distribution by predicted subtype
zone_by_class = conf_df.groupby(['pred_class', 'zone']).size().unstack(fill_value=0)
zone_by_class = zone_by_class.reindex(columns=['HIGH', 'MEDIUM', 'LOW'], fill_value=0)
zone_by_class = zone_by_class.reindex(CLASS_ORDER_INTERNAL)
zone_pct = zone_by_class.div(zone_by_class.sum(axis=1), axis=0)
zone_pct.index = CLASS_ORDER
zone_pct.plot(kind='barh', stacked=True, ax=ax1,
              color=[C_HIGH, C_MEDIUM, C_LOW], edgecolor='white', linewidth=0.5)
ax1.set_xlabel('Proportion', fontsize=10)
ax1.legend(fontsize=8, loc='lower right')
ax1.text(0.02, 0.98, '(A)', transform=ax1.transAxes, fontsize=12,
         fontweight='bold', va='top')

# Panel B: Accuracy by zone
zones = ['HIGH', 'MEDIUM', 'LOW']
accs = [conf_df[conf_df['zone'] == z]['correct'].mean() for z in zones]
ns = [conf_df[conf_df['zone'] == z].shape[0] for z in zones]
bars = ax2.bar(zones, accs, color=[C_HIGH, C_MEDIUM, C_LOW], edgecolor='white')
for bar, acc, n in zip(bars, accs, ns):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{acc:.3f}\nn={n}', ha='center', va='bottom', fontsize=9)
ax2.set_ylabel('Accuracy', fontsize=10)
ax2.set_ylim(0, 1.1)
ax2.text(0.02, 0.98, '(B)', transform=ax2.transAxes, fontsize=12,
         fontweight='bold', va='top')
plt.tight_layout()
save_fig(fig_s5, 'FigS5_Confidence_Zones')
plt.show()

"""### Figure S6 — Global and Local SHAP Explainability (A: Heatmap, B: Waterfall)"""

fig_s6, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1, 1.2]})

# Panel A: Region × Channel heatmap (internal L1)
region_shap = aggregate_shap_by_region(shap_l1_int, feat_names)
ax = axes[0]
display_idx = [c.replace('dif_', 'Δ').replace('raw_', '') for c in CHANNELS]
display_col = ['β₁', 'β₂', 'trans.', 'γ', 'M-prot']
sns.heatmap(region_shap.values, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
            xticklabels=display_col, yticklabels=display_idx,
            cbar_kws={'shrink': 0.8, 'label': 'Mean |SHAP|'})
ax.text(0.02, 0.98, '(A)', transform=ax.transAxes, fontsize=12,
        fontweight='bold', va='top', color='white')

# Panel B: IgA-λ case study waterfall (L1)
# Find a correctly classified IgA-λ sample
iga_lam_idx = np.where((y_class9 == 'IGA_LAMBDA') & (best_pred == 'IGA_LAMBDA'))[0]
if len(iga_lam_idx) > 0:
    sample_idx = iga_lam_idx[0]
    sv = shap_l1_int[sample_idx]
    top_n = 12
    order = np.argsort(np.abs(sv))[::-1][:top_n]
    sv_top = sv[order]
    fn_top = [feat_names[i] for i in order]

    ax = axes[1]
    colors_bar = [C_POS_SHAP if v > 0 else C_NEG_SHAP for v in sv_top]
    y_pos = np.arange(top_n)
    ax.barh(y_pos, sv_top[::-1], color=colors_bar[::-1], height=0.6, edgecolor='white')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(fn_top[::-1], fontsize=7)
    ax.axvline(0, color='gray', lw=0.8, ls='--', alpha=0.5)
    ax.set_xlabel('SHAP value', fontsize=10)
    ax.text(0.02, 0.98, '(B) IgA-λ sample — L1 top features',
            transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')
else:
    axes[1].text(0.5, 0.5, 'No IgA-λ sample available', transform=axes[1].transAxes,
                 ha='center', fontsize=12)

plt.tight_layout()
save_fig(fig_s6, 'FigS6_SHAP_Explainability')
plt.show()

"""### Figure S7 — Global Feature Importance (Beeswarm — Internal + External)"""

fig_s7, axes = plt.subplots(1, 3, figsize=(20, 8))

for col, (shap_int, shap_ext, models, level) in enumerate([
    (shap_l1_int, shap_l1_ext, l1_models, 'L1'),
    (shap_l2_int, shap_l2_ext, l2_models, 'L2'),
    (shap_l3_int, shap_l3_ext, l3_models, 'L3'),
]):
    ax = axes[col]
    mean_int = np.mean(np.abs(shap_int), axis=0)
    mean_ext = np.mean(np.abs(shap_ext), axis=0)
    top_idx = np.argsort(mean_int)[::-1][:20]

    y_pos = np.arange(20)
    w = 0.35
    ax.barh(y_pos + w/2, mean_int[top_idx][::-1], w, label='Internal', color=C_INTERNAL, alpha=0.85)
    ax.barh(y_pos - w/2, mean_ext[top_idx][::-1], w, label='External', color=C_EXTERNAL, alpha=0.85)
    ax.set_yticks(y_pos)
    short_names = [feat_names[i][:30] for i in top_idx[::-1]]
    ax.set_yticklabels(short_names, fontsize=6)
    ax.set_xlabel('Mean |SHAP|', fontsize=9)
    ax.set_title(f'{level}', fontweight='bold', fontsize=11)
    ax.legend(fontsize=7, loc='lower right')

plt.tight_layout()
save_fig(fig_s7, 'FigS7_Beeswarm_SHAP')
plt.show()

"""### Figure S10 — Fairness Across Demographic Subgroups"""

if has_demog and demog_train is not None:
    # Prepare fairness DataFrames (same as Table S12)
    dt_plot = demog_train.copy()
    dt_plot['correct'] = (y_class9 == best_pred).astype(int)
    dt_plot['age_group'] = pd.cut(dt_plot['age'], bins=[0, 40, 60, 75, 200],
                                   labels=['<40', '40-59', '60-74', '75+'])
    de_plot = None
    if demog_ext is not None:
        de_plot = demog_ext.copy()
        de_plot['correct'] = (y_ext_true == ext_pred).astype(int)
        de_plot['age_group'] = pd.cut(de_plot['age'], bins=[0, 40, 60, 75, 200],
                                       labels=['<40', '40-59', '60-74', '75+'])

    fig_s10, axes = plt.subplots(1, 2, figsize=(14, 5))
    panel_labels = ['(A) Internal — Sex & Age', '(B) External — Sex & Age']

    for ax_idx, (df, label) in enumerate([(dt_plot, 'Internal'), (de_plot, 'External')]):
        ax = axes[ax_idx]
        if df is None:
            ax.text(0.5, 0.5, f'{label}: No data', transform=ax.transAxes, ha='center')
            continue

        categories = []
        accs = []
        cis_lo = []
        cis_hi = []
        colors = []

        for subgroup, col in [('Sex', 'sex'), ('Age', 'age_group')]:
            for cat in sorted(df[col].dropna().unique()):
                mask = df[col] == cat
                vals = df.loc[mask, 'correct'].values
                n_sub = len(vals)
                acc = vals.mean()
                rng = np.random.RandomState(SEED)
                boot = [rng.choice(vals, n_sub, replace=True).mean() for _ in range(1000)]
                lo, hi = np.percentile(boot, [2.5, 97.5])
                categories.append(f'{subgroup}: {cat}')
                accs.append(acc)
                cis_lo.append(acc - lo)
                cis_hi.append(hi - acc)
                colors.append(C_INTERNAL if subgroup == 'Sex' else C_EXTERNAL)

        y_pos = np.arange(len(categories))
        ax.barh(y_pos, accs, xerr=[cis_lo, cis_hi], color=colors, alpha=0.8,
                capsize=3, edgecolor='white')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(categories, fontsize=8)
        ax.set_xlabel('Accuracy', fontsize=10)
        ax.set_xlim(0.6, 1.0)
        ax.text(0.02, 0.98, panel_labels[ax_idx], transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top')
    plt.tight_layout()
    save_fig(fig_s10, 'FigS10_Fairness_Subgroups')
    plt.show()

"""### Figure S11 — Per-Level Confusion Matrices (6-panel)"""

fig_s11, axes = plt.subplots(2, 3, figsize=(18, 12))

# L1 OOF
l1_pred_bin = (l1_proba >= L1_THRESHOLD).astype(int)
cm_l1 = confusion_matrix(y_binary, l1_pred_bin)
sns.heatmap(cm_l1, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
axes[0, 0].text(0.02, 0.98, '(A) L1 Internal', transform=axes[0, 0].transAxes,
                fontsize=10, fontweight='bold', va='top')

# L2 OOF
cm_l2 = confusion_matrix(l2_y, l2_pred)
sns.heatmap(cm_l2, annot=True, fmt='d', cmap='Purples', ax=axes[0, 1],
            xticklabels=L2_CLASSES, yticklabels=L2_CLASSES)
axes[0, 1].text(0.02, 0.98, '(B) L2 Internal', transform=axes[0, 1].transAxes,
                fontsize=10, fontweight='bold', va='top')

# L3 OOF
l3_pred_bin = (l3_proba >= 0.5).astype(int)
cm_l3 = confusion_matrix(l3_y, l3_pred_bin)
sns.heatmap(cm_l3, annot=True, fmt='d', cmap='Oranges', ax=axes[0, 2],
            xticklabels=L3_CLASSES, yticklabels=L3_CLASSES)
axes[0, 2].text(0.02, 0.98, '(C) L3 Internal', transform=axes[0, 2].transAxes,
                fontsize=10, fontweight='bold', va='top')

# External L1/L2/L3
ext_l1_pred = (ext_l1_proba >= L1_THRESHOLD).astype(int)
cm_ext_l1 = confusion_matrix(ext_y_bin, ext_l1_pred)
sns.heatmap(cm_ext_l1, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
            xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
axes[1, 0].text(0.02, 0.98, '(D) L1 External', transform=axes[1, 0].transAxes,
                fontsize=10, fontweight='bold', va='top')

# L2/L3 external require per-level true labels — derive from 9-class
ext_pos_mask_true = (ext_y_bin == 1)
if ext_pos_mask_true.sum() > 0:
    ext_heavy_true = []
    ext_light_true = []
    for cls in y_ext_true[ext_pos_mask_true]:
        parts = cls.split('_')
        ext_heavy_true.append(parts[0])
        ext_light_true.append(parts[1])
    le_h = LabelEncoder(); le_h.classes_ = np.array(L2_CLASSES)
    le_l = LabelEncoder(); le_l.classes_ = np.array(L3_CLASSES)

    ext_pred_pos = ext_pred[ext_pos_mask_true]
    ext_heavy_pred = []
    ext_light_pred = []
    for p in ext_pred_pos:
        if p == 'NEGATIVE':
            ext_heavy_pred.append('IGG')
            ext_light_pred.append('KAPPA')
        else:
            parts = p.split('_')
            ext_heavy_pred.append(parts[0])
            ext_light_pred.append(parts[1])

    cm_ext_l2 = confusion_matrix(ext_heavy_true, ext_heavy_pred, labels=L2_CLASSES)
    sns.heatmap(cm_ext_l2, annot=True, fmt='d', cmap='Purples', ax=axes[1, 1],
                xticklabels=L2_CLASSES, yticklabels=L2_CLASSES)
    axes[1, 1].text(0.02, 0.98, '(E) L2 External', transform=axes[1, 1].transAxes,
                    fontsize=10, fontweight='bold', va='top')

    cm_ext_l3 = confusion_matrix(ext_light_true, ext_light_pred, labels=L3_CLASSES)
    sns.heatmap(cm_ext_l3, annot=True, fmt='d', cmap='Oranges', ax=axes[1, 2],
                xticklabels=L3_CLASSES, yticklabels=L3_CLASSES)
    axes[1, 2].text(0.02, 0.98, '(F) L3 External', transform=axes[1, 2].transAxes,
                    fontsize=10, fontweight='bold', va='top')

plt.tight_layout()
save_fig(fig_s11, 'FigS11_PerLevel_Confusion_Matrices')
plt.show()

"""### Figure S12 — ROC and PR Curves"""

fig_s12, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: ROC curves
ax = axes[0]
fpr1, tpr1, _ = roc_curve(y_binary, l1_proba)
ax.plot(fpr1, tpr1, color=C_INTERNAL, lw=2,
        label=f'L1 OOF (AUC = {roc_auc_score(y_binary, l1_proba):.3f})')
fpr1e, tpr1e, _ = roc_curve(ext_y_bin, ext_l1_proba)
ax.plot(fpr1e, tpr1e, '--', color=C_EXTERNAL, lw=2,
        label=f'L1 Ext (AUC = {roc_auc_score(ext_y_bin, ext_l1_proba):.3f})')
fpr3, tpr3, _ = roc_curve(l3_y, l3_proba)
ax.plot(fpr3, tpr3, color='#E65100', lw=2,
        label=f'L3 OOF (AUC = {roc_auc_score(l3_y, l3_proba):.3f})')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, lw=0.8)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.legend(fontsize=8, loc='lower right')
ax.text(0.02, 0.98, '(A)', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

# Panel B: PR curves
ax = axes[1]
prec1, rec1, _ = precision_recall_curve(y_binary, l1_proba)
ax.plot(rec1, prec1, color=C_INTERNAL, lw=2,
        label=f'L1 OOF (AP = {average_precision_score(y_binary, l1_proba):.3f})')
prec1e, rec1e, _ = precision_recall_curve(ext_y_bin, ext_l1_proba)
ax.plot(rec1e, prec1e, '--', color=C_EXTERNAL, lw=2,
        label=f'L1 Ext (AP = {average_precision_score(ext_y_bin, ext_l1_proba):.3f})')
prec3, rec3, _ = precision_recall_curve(l3_y, l3_proba)
ax.plot(rec3, prec3, color='#E65100', lw=2,
        label=f'L3 OOF (AP = {average_precision_score(l3_y, l3_proba):.3f})')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.legend(fontsize=8, loc='lower left')
ax.text(0.02, 0.02, '(B)', transform=ax.transAxes, fontsize=12, fontweight='bold')
plt.tight_layout()
save_fig(fig_s12, 'FigS12_ROC_PR_Curves')
plt.show()

"""### Figure S13 — Error Attribution by Subtype"""

if len(error_df) > 0:
    fig_s13, ax = plt.subplots(figsize=(12, 6))

    error_counts = error_df.groupby(['true_class', 'error_type']).size().unstack(fill_value=0)
    error_counts = error_counts.reindex(CLASS_ORDER_INTERNAL, fill_value=0)
    error_counts.index = CLASS_ORDER

    error_colors = {'L1_FN': '#E53935', 'L1_FP': '#FFB74D', 'L2_error': '#BDBDBD', 'L3_error': '#81C784'}
    error_counts.plot(kind='bar', stacked=True, ax=ax,
                      color=[error_colors.get(c, '#999') for c in error_counts.columns],
                      edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Number of Errors', fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=8, title='Error Source')
    plt.tight_layout()
    save_fig(fig_s13, 'FigS13_Error_Attribution')
    plt.show()

"""### Figure S14 — L1 Threshold Sensitivity Plot"""

fig_s14, ax1 = plt.subplots(figsize=(10, 5))

thrs = table_s14['Threshold'].astype(float).values
sens_vals = table_s14['Sensitivity'].astype(float).values
spec_vals = table_s14['Specificity'].astype(float).values
acc_vals  = table_s14['Accuracy'].astype(float).values
f1_vals   = table_s14['F1'].astype(float).values

ax1.plot(thrs, sens_vals, 'o-', color=C_POS_SHAP, lw=2, ms=4, label='Sensitivity')
ax1.plot(thrs, spec_vals, 's-', color='#757575', lw=2, ms=4, label='Specificity')
ax1.set_xlabel('L1 Decision Threshold', fontsize=10)
ax1.set_ylabel('Sensitivity / Specificity', fontsize=10)
ax1.legend(loc='center left', fontsize=8)

ax2t = ax1.twinx()
ax2t.plot(thrs, acc_vals, '^-', color=C_INTERNAL, lw=1.5, ms=3, alpha=0.7, label='Accuracy')
ax2t.plot(thrs, f1_vals, 'v-', color='#388E3C', lw=1.5, ms=3, alpha=0.7, label='F1')
ax2t.set_ylabel('Accuracy / F1', fontsize=10)
ax2t.legend(loc='center right', fontsize=8)

# Mark operating points
ax1.axvline(0.10, color='#9C27B0', ls=':', alpha=0.6, lw=1.5)
ax1.axvline(L1_THRESHOLD, color=C_INTERNAL, ls='--', alpha=0.8, lw=1.5)
ax1.axvline(0.50, color='#9C27B0', ls=':', alpha=0.6, lw=1.5)

ax1.text(0.11, 0.95, 'Screen', fontsize=7, color='#9C27B0', rotation=90,
         transform=ax1.get_xaxis_transform(), va='top')
ax1.text(L1_THRESHOLD + 0.01, 0.95, 'Youden', fontsize=7, color=C_INTERNAL, rotation=90,
         transform=ax1.get_xaxis_transform(), va='top')

plt.tight_layout()
save_fig(fig_s14, 'FigS14_Threshold_Sensitivity')
plt.show()

# ══════════════════════════════════════════════════════════════
# 5. SUMMARY
# ══════════════════════════════════════════════════════════════

"""## 5. Summary"""

print('=' * 70)
print('FIGURE & TABLE GENERATION COMPLETE')
print('=' * 70)
print(f'\nCSV files: {len(list(CSV_DIR.glob("*.csv")))} tables in {CSV_DIR}')
print(f'Excel:     {excel_path}')
print(f'PNG files: {len(list(PNG_DIR.glob("*.png")))} figures')
print(f'TIFF files: {len(list(TIFF_DIR.glob("*.tif")))} figures')
print(f'\nFigures not generated (require separate input):')
print(f'  - Fig 1: Cascade architecture (HTML)')
print(f'  - Fig 2: CDS pipeline (HTML)')
print(f'  - Fig 4: CDS report (Streamlit screenshot)')
print(f'  - Fig S2: Learning curves (TODO: add to Notebook 1)')
print(f'  - Fig S8: 9-class representative SHAP waterfalls')
print(f'  - Fig S9: 6-channel SHAP overlay')
