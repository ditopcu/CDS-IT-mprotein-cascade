# Cascade M-Protein CDS

**An Integrated Clinical Decision Support Framework with Cascade Machine Learning for Automated Monoclonal Protein Classification from Capillary Electrophoresis Immunotyping Signals**

---

## Overview

A 3-level hierarchical cascade classifier for 9-class M-protein isotype classification from 6-channel capillary zone electrophoresis with immunotyping (CZE-IT) signals, integrated within a clinical decision support (CDS) framework.

**Cascade Architecture:**
- **Level 1:** Binary M-protein detection (Negative vs Positive)
- **Level 2:** Heavy chain classification (IgG / IgA / IgM / Free)
- **Level 3:** Light chain typing (κ vs λ)

**CDS Framework:**
- Probability calibration (ECE assessment)
- Conformal prediction (95% coverage guarantee)
- Confidence-based triaging (HIGH / MEDIUM / LOW zones)
- Automated reflex test recommendations
- Per-sample TreeSHAP explainability

## Key Results

| Metric | Internal (OOF) | External |
|--------|:--------------:|:--------:|
| 9-class Accuracy | 0.871 | 0.872 |
| Macro F1 | 0.750 | 0.652 |
| Cohen's κ | 0.792 | 0.796 |
| HIGH-zone Accuracy | 96.2% | 97.2% |
| HIGH-zone Proportion | 62.2% | 35.9% |

## Repository Structure

```
cascade-mprotein-cds/
├── notebooks/
│   ├── 01_cascade_training_pipeline.ipynb   # Model training & evaluation
│   └── 02_figure_table_generation.ipynb     # Publication figures & tables
├── src/
│   ├── constants.py      # Regions, channels, class labels, config presets
│   ├── features.py       # Peak-based feature extraction (399 features)
│   ├── evaluation.py     # Metrics, CV runner, bootstrap CI, error attribution
│   ├── cascade.py        # Cascade assembly, external inference
│   ├── calibration.py    # ECE, MCE, BSS, isotonic calibration
│   ├── confidence.py     # Conformal prediction, confidence scores
│   ├── cds.py            # CDS pipeline, reflex test recommendations
│   ├── explainability.py # SHAP region aggregation
│   ├── data_loader.py    # Excel/TSV → dataset.pkl converter
│   └── utils.py          # Timer, pickle helpers
├── data/
│   └── templates/        # Example Excel templates for data input
├── docs/
│   └── DATA_PREPARATION_GUIDE.md
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Prepare Your Data

📋 **[Data Preparation Guide](docs/DATA_PREPARATION_GUIDE.md)** — step-by-step instructions for formatting your Sebia Capillarys 2 CZE-IT data.

Template Excel files are provided in `data/templates/`. Copy them to `data/raw/`, replace the example rows with your data, and remove the `_template` suffix from filenames.

### 2. Run the Pipeline

#### Google Colab (Recommended)

1. Upload the repository to Google Drive
2. Open `01_cascade_training_pipeline.ipynb` in Colab
3. The notebook will automatically detect and convert your Excel files in `data/raw/` to `dataset.pkl` (or skip if `dataset.pkl` already exists)
4. Run all cells sequentially

#### Local Environment

```bash
git clone https://github.com/ditopcu/CDS-IT-mprotein-cascade.git
cd CDS-IT-mprotein-cascade
pip install -r requirements.txt

# Build dataset from Excel files
python src/data_loader.py \
    --dev-signals data/raw/Internal_signals.xlsx \
    --dev-labels data/raw/Internal_labels.xlsx

# Run the pipeline
jupyter notebook notebooks/01_cascade_training_pipeline.ipynb
```

## Data Availability

De-identified signal data sufficient to reproduce the main analyses are available from the corresponding author upon reasonable request, subject to institutional data sharing agreements.

To use your own Sebia Capillarys 2 data, see the **[Data Preparation Guide](docs/DATA_PREPARATION_GUIDE.md)** and the Excel templates in `data/templates/`.

## Citation

*[Citation information will be added upon publication]*

## License

This project is licensed under the MIT License.

## Acknowledgments

- Başkent University Ankara Hospital (development cohort)
- Ankara University İbn-i Sina Hospital (external validation cohort)
- AI tools (Claude, Anthropic) assisted with manuscript drafting and code review
