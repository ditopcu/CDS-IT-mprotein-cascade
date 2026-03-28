# Data Preparation Guide

## Overview

This guide explains how to prepare your Sebia Capillarys 2 CZE-IT (Capillary Zone Electrophoresis with Immunotyping) data for use with the cascade M-protein classification pipeline.

You will need to prepare **Excel files** from your Sebia Phoresis software exports. The `data_loader.py` module will then convert these into the `dataset.pkl` format required by the training pipeline.

---

## Required Files

| # | File | Required? | Description |
|---|------|-----------|-------------|
| 1 | `Internal_signals.xlsx` | **Yes** | Development cohort signal data |
| 2 | `Internal_labels.xlsx` | **Yes** | Development cohort isotype labels |
| 3 | `External_signals.xlsx` | No | External validation cohort signal data |
| 4 | `External_labels.xlsx` | No | External validation cohort isotype labels |
| 5 | `Internal_demographics.xlsx` | No | Development cohort age & sex |
| 6 | `External_demographics.xlsx` | No | External validation cohort age & sex |

> **Note:** Template files for each are provided in `data/templates/`. Delete the example rows and replace with your data.

Place all files in the `data/raw/` directory.

---

## File Formats

### 1. Signal Files (`*_signals.xlsx`)

Long-format table with one row per sample × curve × timepoint.

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | text/numeric | Unique patient or sample identifier |
| `curve_name` | text | Curve name: `ELP`, `IgG`, `IgA`, `IgM`, `Kappa`, `Lambda` |
| `x` | integer | Timepoint index (0–299) |
| `y` | numeric | Raw densitometric intensity value |

Optional columns (`uid`, `data_analisi`, `seq_no`, `seq_count`) are allowed and will be ignored by the loader.

**Requirements per sample:**
- Exactly **6 curves**: ELP, IgG, IgA, IgM, Kappa, Lambda
- Exactly **300 timepoints** per curve (x = 0, 1, 2, ..., 299)
- Total rows per sample: 6 × 300 = **1,800 rows**
- A 7th curve named `Reference` is permitted and will be automatically dropped
- Sebia's spelling `Lamda` is accepted and auto-corrected to `Lambda`

**Important:** Provide **raw signal values** directly from Sebia Phoresis export. Do NOT pre-compute difference curves — the data loader handles this automatically using the formula: `dif_y = ELP_y − channel_y`

**Example rows:**

| sample_id | curve_name | x | y |
|-----------|-----------|---|---|
| 6505578 | ELP | 0 | 22 |
| 6505578 | ELP | 1 | 22 |
| 6505578 | ELP | 2 | 23 |
| ... | ... | ... | ... |
| 6505578 | ELP | 299 | 4 |
| 6505578 | IgG | 0 | 279 |
| 6505578 | IgG | 1 | 281 |
| ... | ... | ... | ... |

---

### 2. Label Files (`*_labels.xlsx`)

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | text/numeric | Must match signal file sample IDs |
| `final_comment` | text | 9-class isotype classification |

**Valid label values (exactly one per sample):**

| Label | Description |
|-------|-------------|
| `IGG_KAPPA` | IgG heavy chain, kappa light chain |
| `IGG_LAMBDA` | IgG heavy chain, lambda light chain |
| `IGA_KAPPA` | IgA heavy chain, kappa light chain |
| `IGA_LAMBDA` | IgA heavy chain, lambda light chain |
| `IGM_KAPPA` | IgM heavy chain, kappa light chain |
| `IGM_LAMBDA` | IgM heavy chain, lambda light chain |
| `FREE_KAPPA` | Free light chain, kappa |
| `FREE_LAMBDA` | Free light chain, lambda |
| `NEGATIVE` | No monoclonal protein detected |

No other values are accepted. Labels are case-insensitive (auto-converted to uppercase).

---

### 3. Demographics Files (`*_demographics.xlsx`) — Optional

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | text/numeric | Must match signal file sample IDs |
| `age` | numeric | Patient age in years |
| `sex` | text | `Male` or `Female` (case-sensitive) |

Demographics are **not used** for model training. They enable fairness and subgroup analysis only. If not provided, the pipeline will issue a warning but proceed normally.

---

## Running the Data Loader

### Option A: Python API

```python
from src.data_loader import build_dataset, validate_dataset

# Minimum (development cohort only)
build_dataset(
    dev_signals='data/raw/Internal_signals.xlsx',
    dev_labels='data/raw/Internal_labels.xlsx',
)

# Full (both cohorts + demographics)
build_dataset(
    dev_signals='data/raw/Internal_signals.xlsx',
    dev_labels='data/raw/Internal_labels.xlsx',
    ext_signals='data/raw/External_signals.xlsx',
    ext_labels='data/raw/External_labels.xlsx',
    dev_demographics='data/raw/Internal_demographics.xlsx',
    ext_demographics='data/raw/External_demographics.xlsx',
    output_path='data/processed/dataset.pkl',
    demographics_path='data/processed/demographics.xlsx',
)

# Validate an existing dataset
validate_dataset('data/processed/dataset.pkl')
```

### Option B: Command Line

```bash
# Minimum
python src/data_loader.py \
    --dev-signals data/raw/Internal_signals.xlsx \
    --dev-labels data/raw/Internal_labels.xlsx

# Full
python src/data_loader.py \
    --dev-signals data/raw/Internal_signals.xlsx \
    --dev-labels data/raw/Internal_labels.xlsx \
    --ext-signals data/raw/External_signals.xlsx \
    --ext-labels data/raw/External_labels.xlsx \
    --dev-demographics data/raw/Internal_demographics.xlsx \
    --ext-demographics data/raw/External_demographics.xlsx \
    --output data/processed/dataset.pkl

# Validate only
python src/data_loader.py --validate-only data/processed/dataset.pkl
```

---

## What the Loader Does

1. **Reads** long-format signal Excel and label Excel
2. **Drops** the Reference curve (if present)
3. **Fixes** `Lamda` → `Lambda` spelling
4. **Computes** difference curves: `dif_y = ELP_y − channel_y` for IgG, IgA, IgM, Kappa, Lambda
5. **Keeps** raw ELP as-is
6. **Assembles** 6-channel × 300-timepoint 3D signal arrays
7. **Aligns** signals and labels by `sample_id` (warns about mismatches)
8. **Decomposes** 9-class labels into binary, heavy chain, and light chain components
9. **Validates** shapes, NaN, duplicates, channel integrity, and class counts
10. **Saves** `dataset.pkl` (and optionally `demographics.xlsx`)

---

## Output: `dataset.pkl`

The generated file is a Python dictionary (pickle format) containing:

| Key | Shape | Description |
|-----|-------|-------------|
| `X_3d` | (N, 6, 300) | 6-channel signal arrays |
| `X` | (N, 1800) | Flattened signals |
| `sample_ids` | (N,) | Sample identifiers |
| `y_class9` | (N,) | 9-class string labels |
| `y_binary` | (N,) | POSITIVE / NEGATIVE |
| `y_heavy` | (N,) | IGG / IGA / IGM / FREE / NaN |
| `y_light` | (N,) | KAPPA / LAMBDA / NaN |
| `channels` | list | Channel order (6 names) |
| ... | ... | Plus integer-encoded labels, masks, and encoding maps |

Channel order: `raw_ELP → dif_IgG → dif_IgA → dif_IgM → dif_Kappa → dif_Lambda`

---

## Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `Signal file missing columns` | Required column not found | Ensure `sample_id`, `curve_name`, `x`, `y` columns exist |
| `Unexpected curve names` | Unrecognized curve name | Check for typos; valid names: ELP, IgG, IgA, IgM, Kappa, Lambda |
| `ELP has N timepoints, expected 300` | Incomplete curve data | Each curve must have exactly 300 data points (x: 0–299) |
| `Invalid labels found` | Label not in valid set | Use only the 9 labels listed above |
| `Duplicate sample_ids` | Same ID appears twice in labels | Each sample must have exactly one label |
| `No overlapping sample_ids` | Signal and label IDs don't match | Verify sample_id values match between files |
| `Channel X is all zeros` | Missing or empty channel data | Check that all 6 curves have non-zero values |
| `Class X has only N samples` | Rare class warning | Not an error; ≥10 samples per class recommended |

---

## Directory Structure

```
data/
├── raw/                              ← Your input files go here
│   ├── Internal_signals.xlsx
│   ├── Internal_labels.xlsx
│   ├── Internal_demographics.xlsx    (optional)
│   ├── External_signals.xlsx         (optional)
│   ├── External_labels.xlsx          (optional)
│   └── External_demographics.xlsx    (optional)
├── processed/                        ← Generated by data_loader
│   ├── dataset.pkl
│   └── demographics.xlsx
└── templates/                        ← Template files with examples
    ├── Internal_signals_template.xlsx
    ├── Internal_labels_template.xlsx
    ├── Internal_demographics_template.xlsx
    ├── External_signals_template.xlsx
    ├── External_labels_template.xlsx
    └── External_demographics_template.xlsx
```
