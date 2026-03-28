"""
data_loader.py — Data Ingestion & Validation Module
=====================================================
Converts user-supplied Excel files into dataset.pkl compatible with
the cascade M-protein classification pipeline.

Input files (per cohort):
    {Cohort}_signals.xlsx      — Long-format CZE-IT signal data
    {Cohort}_labels.xlsx       — 9-class isotype labels
    {Cohort}_demographics.xlsx — Age & sex (optional)

Where {Cohort} is "Internal" (development) or "External" (validation).

Usage:
    from src.data_loader import build_dataset, validate_dataset

    build_dataset(
        dev_signals='data/raw/Internal_signals.xlsx',
        dev_labels='data/raw/Internal_labels.xlsx',
        ext_signals='data/raw/External_signals.xlsx',   # optional
        ext_labels='data/raw/External_labels.xlsx',      # optional
        dev_demographics='data/raw/Internal_demographics.xlsx',  # optional
        ext_demographics='data/raw/External_demographics.xlsx',  # optional
        output_path='data/processed/dataset.pkl',
    )

    validate_dataset('data/processed/dataset.pkl')
"""

import warnings
import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS — must match pipeline expectations exactly
# ═══════════════════════════════════════════════════════════════════════

N_TIMEPOINTS = 300
N_CHANNELS = 6

# Channel order (index position = channel index in 3D array)
CHANNELS = ['raw_ELP', 'dif_IgG', 'dif_IgA', 'dif_IgM', 'dif_Kappa', 'dif_Lambda']
CH_IDX = {ch: i for i, ch in enumerate(CHANNELS)}

# Curve names as they appear in the signal Excel (Sebia export)
# These map to the 5 immunotyping channels that need diff calculation
DIFF_CURVE_MAP = {
    'IgG':    'dif_IgG',
    'IgA':    'dif_IgA',
    'IgM':    'dif_IgM',
    'Kappa':  'dif_Kappa',
    'Lambda': 'dif_Lambda',
}

# Accepted curve names in signal Excel (Reference is dropped)
EXPECTED_CURVES = {'ELP', 'IgG', 'IgA', 'IgM', 'Kappa', 'Lambda', 'Reference'}

# Also accept Sebia's own spelling "Lamda" → treat as "Lambda"
CURVE_NAME_FIXES = {'Lamda': 'Lambda'}

# Valid 9-class labels
VALID_LABELS = frozenset({
    'FREE_KAPPA', 'FREE_LAMBDA',
    'IGA_KAPPA', 'IGA_LAMBDA',
    'IGG_KAPPA', 'IGG_LAMBDA',
    'IGM_KAPPA', 'IGM_LAMBDA',
    'NEGATIVE',
})

# Encoding maps — alphabetical order for class9
CLASS9_MAP = {label: i for i, label in enumerate(sorted(VALID_LABELS))}
BINARY_MAP = {'NEGATIVE': 0, 'POSITIVE': 1}
HEAVY_MAP = {'IGG': 0, 'IGA': 1, 'IGM': 2, 'FREE': 3}
LIGHT_MAP = {'KAPPA': 0, 'LAMBDA': 1}

# Required columns in signal Excel
SIGNAL_REQUIRED_COLS = {'sample_id', 'curve_name', 'x', 'y'}

# Required columns in label Excel
LABEL_REQUIRED_COLS = {'sample_id', 'final_comment'}

# Required columns in demographics Excel
DEMOG_REQUIRED_COLS = {'sample_id', 'age', 'sex'}


# ═══════════════════════════════════════════════════════════════════════
# SIGNAL PROCESSING
# ═══════════════════════════════════════════════════════════════════════

def _read_signals(path: str) -> pd.DataFrame:
    """Read and validate signal Excel (long format)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Signal file not found: {path}")

    print(f"  Reading signals from {path.name}...")
    df = pd.read_excel(path, engine='openpyxl')
    df.columns = df.columns.str.strip()

    # Check required columns
    missing = SIGNAL_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Signal file missing columns: {missing}")

    # Fix known Sebia spelling variations
    df['curve_name'] = df['curve_name'].replace(CURVE_NAME_FIXES)

    # Validate curve names
    found_curves = set(df['curve_name'].unique())
    unexpected = found_curves - EXPECTED_CURVES
    if unexpected:
        raise ValueError(f"Unexpected curve names: {unexpected}. Expected: {EXPECTED_CURVES}")

    # Drop Reference curve
    n_before = len(df)
    df = df[df['curve_name'] != 'Reference'].copy()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"    Dropped {n_dropped} Reference rows")

    return df


def _compute_diff_channels(df: pd.DataFrame) -> np.ndarray:
    """
    Compute 6-channel signals from long-format data.

    Diff calculation (matches R function get_calculate_ML_data_v3):
        dif_y = ELP_y - channel_y  (for IgG, IgA, IgM, Kappa, Lambda)
        raw_ELP = ELP_y            (kept as-is)

    No clipping applied.

    Returns:
        X_3d: ndarray of shape (N, 6, 300), dtype float32
    """
    sample_ids = sorted(df['sample_id'].unique())
    n_samples = len(sample_ids)
    X_3d = np.zeros((n_samples, N_CHANNELS, N_TIMEPOINTS), dtype=np.float32)

    for i, sid in enumerate(sample_ids):
        sample_df = df[df['sample_id'] == sid]

        # Extract ELP baseline
        elp = sample_df[sample_df['curve_name'] == 'ELP'].sort_values('x')
        if len(elp) != N_TIMEPOINTS:
            raise ValueError(
                f"Sample {sid}: ELP has {len(elp)} timepoints, expected {N_TIMEPOINTS}"
            )
        elp_y = elp['y'].values.astype(np.float32)

        # Channel 0: raw_ELP
        X_3d[i, 0, :] = elp_y

        # Channels 1–5: diff curves
        for curve_name, channel_name in DIFF_CURVE_MAP.items():
            ch_idx = CH_IDX[channel_name]
            curve = sample_df[sample_df['curve_name'] == curve_name].sort_values('x')
            if len(curve) != N_TIMEPOINTS:
                raise ValueError(
                    f"Sample {sid}: {curve_name} has {len(curve)} timepoints, "
                    f"expected {N_TIMEPOINTS}"
                )
            curve_y = curve['y'].values.astype(np.float32)
            # Diff = ELP - channel (no clipping, matches R code)
            X_3d[i, ch_idx, :] = elp_y - curve_y

    return np.array(sample_ids), X_3d


# ═══════════════════════════════════════════════════════════════════════
# LABEL PROCESSING
# ═══════════════════════════════════════════════════════════════════════

def _read_labels(path: str) -> pd.DataFrame:
    """Read and validate label Excel."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")

    print(f"  Reading labels from {path.name}...")
    df = pd.read_excel(path, engine='openpyxl')
    df.columns = df.columns.str.strip()

    missing = LABEL_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Label file missing columns: {missing}")

    # Strip whitespace from labels
    df['final_comment'] = df['final_comment'].str.strip().str.upper()

    # Validate labels
    invalid = set(df['final_comment'].unique()) - VALID_LABELS
    if invalid:
        raise ValueError(
            f"Invalid labels found: {invalid}\n"
            f"Allowed values: {sorted(VALID_LABELS)}"
        )

    # Check duplicates
    dups = df[df['sample_id'].duplicated(keep=False)]
    if len(dups) > 0:
        dup_ids = dups['sample_id'].unique()
        raise ValueError(f"Duplicate sample_ids in labels: {dup_ids[:10]}")

    return df


def _decompose_labels(
    class9: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Decompose 9-class string labels into binary, heavy, light components
    with both string and integer-encoded versions.
    """
    n = len(class9)
    y_binary = np.empty(n, dtype=object)
    y_heavy = np.empty(n, dtype=object)
    y_light = np.empty(n, dtype=object)
    y_binary_enc = np.zeros(n, dtype=np.int8)
    y_heavy_enc = np.full(n, -1, dtype=np.int8)
    y_light_enc = np.full(n, -1, dtype=np.int8)
    y_class9_enc = np.zeros(n, dtype=np.int8)

    for i, label in enumerate(class9):
        y_class9_enc[i] = CLASS9_MAP[label]

        if label == 'NEGATIVE':
            y_binary[i] = 'NEGATIVE'
            y_heavy[i] = np.nan
            y_light[i] = np.nan
            y_binary_enc[i] = 0
        else:
            y_binary[i] = 'POSITIVE'
            y_binary_enc[i] = 1
            parts = label.split('_')
            heavy, light = parts[0], parts[1]
            y_heavy[i] = heavy
            y_light[i] = light
            y_heavy_enc[i] = HEAVY_MAP[heavy]
            y_light_enc[i] = LIGHT_MAP[light]

    pos_mask = y_binary_enc.astype(bool)

    return {
        'y_binary': y_binary,
        'y_heavy': y_heavy,
        'y_light': y_light,
        'y_binary_enc': y_binary_enc,
        'y_heavy_enc': y_heavy_enc,
        'y_light_enc': y_light_enc,
        'y_class9_enc': y_class9_enc,
        'pos_mask': pos_mask,
    }


# ═══════════════════════════════════════════════════════════════════════
# DEMOGRAPHICS
# ═══════════════════════════════════════════════════════════════════════

def _read_demographics(dev_path: Optional[str], ext_path: Optional[str]) -> Optional[pd.DataFrame]:
    """Read and combine demographics files. Returns None if neither provided."""
    frames = []

    for path_str, cohort_label in [(dev_path, 'Development'), (ext_path, 'External')]:
        if path_str is None:
            continue
        path = Path(path_str)
        if not path.exists():
            warnings.warn(f"Demographics file not found, skipping: {path}")
            continue

        print(f"  Reading demographics from {path.name}...")
        df = pd.read_excel(path, engine='openpyxl')
        df.columns = df.columns.str.strip()

        missing = DEMOG_REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"Demographics file {path.name} missing columns: {missing}")

        df['cohort'] = cohort_label
        frames.append(df[['cohort', 'sample_id', 'sex', 'age']])

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)

    # Validate sex values
    valid_sex = {'Male', 'Female'}
    found_sex = set(combined['sex'].dropna().unique())
    invalid_sex = found_sex - valid_sex
    if invalid_sex:
        raise ValueError(f"Invalid sex values: {invalid_sex}. Expected: {valid_sex}")

    return combined


# ═══════════════════════════════════════════════════════════════════════
# FEATURE COLUMN NAMES
# ═══════════════════════════════════════════════════════════════════════

def _build_feature_cols() -> list:
    """
    Build feature column names matching pipeline convention.
    Format: x{timepoint}_{channel_name}
    Order: all timepoints for ch0, then ch1, etc.
    """
    cols = []
    for ch in CHANNELS:
        for t in range(N_TIMEPOINTS):
            cols.append(f"x{t}_{ch}")
    return cols


# ═══════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════

def _validate_cohort(
    X_3d: np.ndarray,
    sample_ids: np.ndarray,
    class9: np.ndarray,
    cohort_name: str,
) -> None:
    """Run validation checks on a single cohort."""
    n = len(sample_ids)
    print(f"  Validating {cohort_name} cohort ({n} samples)...")

    # Shape
    assert X_3d.shape == (n, N_CHANNELS, N_TIMEPOINTS), \
        f"X_3d shape mismatch: {X_3d.shape} vs expected ({n}, {N_CHANNELS}, {N_TIMEPOINTS})"

    # No NaN in signals
    nan_count = np.isnan(X_3d).sum()
    if nan_count > 0:
        raise ValueError(f"{cohort_name}: {nan_count} NaN values in signal data")

    # No duplicate sample IDs
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"{cohort_name}: Duplicate sample IDs detected")

    # All labels valid
    invalid = set(class9) - VALID_LABELS
    if invalid:
        raise ValueError(f"{cohort_name}: Invalid labels: {invalid}")

    # Channel range check (no all-zero channels)
    for ch in range(N_CHANNELS):
        ch_max = np.max(np.abs(X_3d[:, ch, :]))
        if ch_max == 0:
            raise ValueError(f"{cohort_name}: Channel {CHANNELS[ch]} is all zeros")

    # Per-class sample count warnings
    for cls in sorted(VALID_LABELS):
        count = (class9 == cls).sum()
        if count == 0:
            warnings.warn(f"{cohort_name}: Class '{cls}' has 0 samples")
        elif count < 10:
            warnings.warn(f"{cohort_name}: Class '{cls}' has only {count} samples (≥10 recommended)")

    print(f"    ✓ All checks passed")


# ═══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def _process_cohort(
    signals_path: str,
    labels_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process a single cohort: read signals + labels, compute diffs, align.

    Returns:
        sample_ids, X_3d, class9 (all aligned by sample_id)
    """
    # Read signals → compute diff channels
    sig_df = _read_signals(signals_path)
    sample_ids_sig, X_3d = _compute_diff_channels(sig_df)

    # Read labels
    lab_df = _read_labels(labels_path)

    # Align: keep only samples present in BOTH signal and label files
    sig_set = set(sample_ids_sig)
    lab_set = set(lab_df['sample_id'].values)

    common = sorted(sig_set & lab_set)
    only_signals = sig_set - lab_set
    only_labels = lab_set - sig_set

    if only_signals:
        warnings.warn(
            f"{len(only_signals)} samples in signals but not in labels (dropped): "
            f"{sorted(only_signals)[:5]}..."
        )
    if only_labels:
        warnings.warn(
            f"{len(only_labels)} samples in labels but not in signals (dropped): "
            f"{sorted(only_labels)[:5]}..."
        )

    if len(common) == 0:
        raise ValueError("No overlapping sample_ids between signals and labels!")

    # Reindex to common set
    sig_id_to_idx = {sid: i for i, sid in enumerate(sample_ids_sig)}
    common_sig_indices = [sig_id_to_idx[sid] for sid in common]
    X_3d = X_3d[common_sig_indices]

    lab_df = lab_df.set_index('sample_id').loc[common]
    class9 = lab_df['final_comment'].values.astype(object)

    sample_ids = np.array(common, dtype=np.int64)

    print(f"    Aligned: {len(common)} samples")
    return sample_ids, X_3d, class9


def build_dataset(
    dev_signals: str,
    dev_labels: str,
    ext_signals: Optional[str] = None,
    ext_labels: Optional[str] = None,
    dev_demographics: Optional[str] = None,
    ext_demographics: Optional[str] = None,
    output_path: str = 'data/processed/dataset.pkl',
    demographics_path: str = 'data/processed/demographics.xlsx',
) -> Dict[str, Any]:
    """
    Build dataset.pkl from user-supplied Excel files.

    Parameters
    ----------
    dev_signals : str
        Path to Internal_signals.xlsx (required)
    dev_labels : str
        Path to Internal_labels.xlsx (required)
    ext_signals : str, optional
        Path to External_signals.xlsx
    ext_labels : str, optional
        Path to External_labels.xlsx
    dev_demographics : str, optional
        Path to Internal_demographics.xlsx
    ext_demographics : str, optional
        Path to External_demographics.xlsx
    output_path : str
        Where to save dataset.pkl
    demographics_path : str
        Where to save demographics.xlsx

    Returns
    -------
    dict : The constructed dataset dictionary
    """
    print("=" * 60)
    print("Building dataset.pkl")
    print("=" * 60)

    # ── Development cohort (required) ──
    print("\n[1/4] Processing development cohort...")
    sample_ids, X_3d, class9 = _process_cohort(dev_signals, dev_labels)
    X = X_3d.reshape(len(sample_ids), -1)  # (N, 1800)
    labels = _decompose_labels(class9)
    _validate_cohort(X_3d, sample_ids, class9, "Development")

    # ── External cohort (optional) ──
    has_external = ext_signals is not None and ext_labels is not None
    if has_external:
        print("\n[2/4] Processing external cohort...")
        ext_sample_ids, X_ext_3d, ext_class9 = _process_cohort(ext_signals, ext_labels)
        X_ext = X_ext_3d.reshape(len(ext_sample_ids), -1)
        ext_labels_decomposed = _decompose_labels(ext_class9)
        _validate_cohort(X_ext_3d, ext_sample_ids, ext_class9, "External")
    else:
        print("\n[2/4] No external cohort provided, skipping...")
        ext_sample_ids = np.array([], dtype=np.int64)
        X_ext_3d = np.zeros((0, N_CHANNELS, N_TIMEPOINTS), dtype=np.float32)
        X_ext = np.zeros((0, N_CHANNELS * N_TIMEPOINTS), dtype=np.float32)
        ext_class9 = np.array([], dtype=object)
        ext_labels_decomposed = _decompose_labels(ext_class9)

    # ── Demographics (optional) ──
    print("\n[3/4] Processing demographics...")
    demog = _read_demographics(dev_demographics, ext_demographics)
    if demog is not None:
        out_dir = Path(demographics_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        demog.to_excel(demographics_path, index=False, sheet_name='Demographics')
        print(f"    Saved demographics to {demographics_path}")
    else:
        warnings.warn(
            "No demographics files provided. Fairness analysis will not be available. "
            "Demographics can be added later as a separate Excel file."
        )

    # ── Assemble dataset dict ──
    print("\n[4/4] Assembling dataset.pkl...")
    feature_cols = _build_feature_cols()

    D = {
        # Development cohort
        'X': X,
        'X_3d': X_3d,
        'sample_ids': sample_ids,
        'feature_cols': feature_cols,
        'y_class9': class9,
        'y_binary': labels['y_binary'],
        'y_heavy': labels['y_heavy'],
        'y_light': labels['y_light'],
        'y_binary_enc': labels['y_binary_enc'],
        'y_heavy_enc': labels['y_heavy_enc'],
        'y_light_enc': labels['y_light_enc'],
        'y_class9_enc': labels['y_class9_enc'],
        'pos_mask': labels['pos_mask'],

        # External cohort
        'X_ext': X_ext,
        'X_ext_3d': X_ext_3d,
        'ext_sample_ids': ext_sample_ids,
        'y_ext_class9': ext_class9,
        'y_ext_binary': ext_labels_decomposed['y_binary'],
        'y_ext_heavy': ext_labels_decomposed['y_heavy'],
        'y_ext_light': ext_labels_decomposed['y_light'],
        'y_ext_binary_enc': ext_labels_decomposed['y_binary_enc'],
        'y_ext_heavy_enc': ext_labels_decomposed['y_heavy_enc'],
        'y_ext_light_enc': ext_labels_decomposed['y_light_enc'],
        'y_ext_class9_enc': ext_labels_decomposed['y_class9_enc'],
        'ext_pos_mask': ext_labels_decomposed['pos_mask'],

        # Encoding maps
        'binary_map': BINARY_MAP,
        'heavy_map': HEAVY_MAP,
        'light_map': LIGHT_MAP,
        'class9_map': CLASS9_MAP,

        # Signal metadata
        'channels': CHANNELS,
        'ch_idx': CH_IDX,
        'n_timepoints': N_TIMEPOINTS,
        'n_channels': N_CHANNELS,
    }

    # Save
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(D, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n{'=' * 60}")
    print(f"dataset.pkl saved to {output_path}")
    print(f"  Development: {len(sample_ids)} samples, {(class9 != 'NEGATIVE').sum()} positive")
    if has_external:
        print(f"  External:    {len(ext_sample_ids)} samples, {(ext_class9 != 'NEGATIVE').sum()} positive")
    print(f"  Channels:    {N_CHANNELS} × {N_TIMEPOINTS} = {N_CHANNELS * N_TIMEPOINTS} features")
    print(f"{'=' * 60}")

    return D


def validate_dataset(path: str) -> bool:
    """
    Validate an existing dataset.pkl file.

    Returns True if all checks pass.
    """
    print(f"Validating {path}...")
    with open(path, 'rb') as f:
        D = pickle.load(f)

    # Check all required keys exist
    required_keys = {
        'X', 'X_3d', 'sample_ids', 'feature_cols',
        'y_class9', 'y_binary', 'y_heavy', 'y_light',
        'y_binary_enc', 'y_heavy_enc', 'y_light_enc', 'y_class9_enc',
        'pos_mask',
        'X_ext', 'X_ext_3d', 'ext_sample_ids',
        'y_ext_class9', 'y_ext_binary', 'y_ext_heavy', 'y_ext_light',
        'y_ext_binary_enc', 'y_ext_heavy_enc', 'y_ext_light_enc', 'y_ext_class9_enc',
        'ext_pos_mask',
        'binary_map', 'heavy_map', 'light_map', 'class9_map',
        'channels', 'ch_idx', 'n_timepoints', 'n_channels',
    }
    missing = required_keys - set(D.keys())
    if missing:
        raise ValueError(f"Missing keys: {missing}")

    # Validate development cohort
    _validate_cohort(D['X_3d'], D['sample_ids'], D['y_class9'], "Development")

    # Validate external cohort (if non-empty)
    if len(D['ext_sample_ids']) > 0:
        _validate_cohort(D['X_ext_3d'], D['ext_sample_ids'], D['y_ext_class9'], "External")

    # Validate encoding maps
    assert D['class9_map'] == CLASS9_MAP, "class9_map mismatch"
    assert D['binary_map'] == BINARY_MAP, "binary_map mismatch"
    assert D['heavy_map'] == HEAVY_MAP, "heavy_map mismatch"
    assert D['light_map'] == LIGHT_MAP, "light_map mismatch"

    # Validate channel metadata
    assert D['channels'] == CHANNELS, f"Channel order mismatch: {D['channels']}"
    assert D['ch_idx'] == CH_IDX, f"Channel index mismatch: {D['ch_idx']}"
    assert D['n_timepoints'] == N_TIMEPOINTS
    assert D['n_channels'] == N_CHANNELS

    # Validate feature_cols length
    assert len(D['feature_cols']) == N_CHANNELS * N_TIMEPOINTS

    print("✓ All validation checks passed")
    return True


# ═══════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Build dataset.pkl from Excel files for the cascade M-protein pipeline.'
    )
    parser.add_argument('--dev-signals', required=True, help='Internal_signals.xlsx')
    parser.add_argument('--dev-labels', required=True, help='Internal_labels.xlsx')
    parser.add_argument('--ext-signals', default=None, help='External_signals.xlsx (optional)')
    parser.add_argument('--ext-labels', default=None, help='External_labels.xlsx (optional)')
    parser.add_argument('--dev-demographics', default=None, help='Internal_demographics.xlsx (optional)')
    parser.add_argument('--ext-demographics', default=None, help='External_demographics.xlsx (optional)')
    parser.add_argument('--output', default='data/processed/dataset.pkl', help='Output path')
    parser.add_argument('--validate-only', default=None, help='Validate existing dataset.pkl')

    args = parser.parse_args()

    if args.validate_only:
        validate_dataset(args.validate_only)
    else:
        build_dataset(
            dev_signals=args.dev_signals,
            dev_labels=args.dev_labels,
            ext_signals=args.ext_signals,
            ext_labels=args.ext_labels,
            dev_demographics=args.dev_demographics,
            ext_demographics=args.ext_demographics,
            output_path=args.output,
        )
