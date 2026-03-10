# RQ1 Fairness Analysis — Final Deliverables
## Texas-100X LOS Prediction Fairness Study

---

## Notebooks

### LOS_Prediction_Standard.ipynb (51 code cells)
Compact analysis: data loading, 6 models, fairness metrics, fair model, AFCE pipeline.
- **Status**: Fully executed, 0 errors
- **Run time**: ~40 min on RTX 5070 GPU
- **Key result**: AFCE α=0.1 → Acc=86.62%, RACE DI=0.999, SEX DI=1.000

### LOS_Prediction_Detailed.ipynb (51 code cells)
Full analysis with detailed markdown commentary between cells.
Same 51 code cells as Standard plus extensive section headers and explanations.
- **Status**: Fully executed, 0 errors
- **Run time**: ~41 min on RTX 5070 GPU

### Fairness_Analysis_Complete.ipynb (50 code cells)
Original complete analysis notebook.
- **Run time**: ~35-40 min on RTX 5070 GPU

---

## Folder Structure

```
final_notebooks/
├── LOS_Prediction_Standard.ipynb    # 51 code cells, executed
├── LOS_Prediction_Detailed.ipynb    # 51 code cells, executed
├── Fairness_Analysis_Complete.ipynb  # 50 code cells
├── data/
│   └── texas_100x.csv               # 925,128 records, 12 columns
├── scripts/
│   ├── run_analysis.py              # Reproducible CPU analysis pipeline
│   └── analysis_utils.py            # Helper functions (metrics, fairness, bootstrap)
├── results/
│   ├── extracted_metrics.json       # All metrics from executed notebooks
│   ├── analysis_report.json         # Full structured results from run_analysis.py
│   └── analysis_summary.txt         # Human-readable summary
├── report/
│   ├── RQ1_Analysis_Section.md      # Paper Analysis section (4.1–4.12)
│   └── weekly_meeting_script.md     # Meeting agenda template
├── figures/                         # Generated plots (from notebook execution)
├── models/                          # Saved model artifacts
├── tables/                          # Saved tables
└── processed_data/                  # Intermediate processed data
```

---

## Quick Start

```bash
# 1. Activate environment
cd final_notebooks
..\.venv\Scripts\activate

# 2. Run the CPU analysis script
python scripts/run_analysis.py

# 3. View results
cat results/analysis_summary.txt
```

---

## Key Results Summary

| Configuration      | Accuracy | F1     | RACE DI | SEX DI | ETH DI | Fair? |
|--------------------|----------|--------|---------|--------|--------|-------|
| LightGBM (best)    | 87.87%   | 86.39% | 0.642   | 0.762  | 0.829  | 1/4   |
| Fair Model (λ=5.0) | 87.38%   | 85.41% | 0.751   | 0.756  | 0.844  | 1/4   |
| **AFCE (α=0.1)**   | **86.62%** | **84.97%** | **0.999** | **1.000** | **0.916** | **3/4** |

Accuracy cost of full fairness: **1.25 percentage points** (87.87% → 86.62%)

---

## Data

All notebooks expect `./data/texas_100x.csv` relative to their location.
The dataset contains 925,128 hospital discharge records from 441 Texas hospitals.
