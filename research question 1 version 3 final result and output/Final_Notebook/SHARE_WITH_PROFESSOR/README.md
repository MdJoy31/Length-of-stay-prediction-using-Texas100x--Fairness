# How Reliable Are Fairness Metrics in Clinical AI?
## A Multi-Site Evaluation Across 441 Hospitals

**Authors:** [Author Name], Caslon Chua, Viet Vo
**Affiliation:** Swinburne University of Technology
**Target Journal:** npj Digital Medicine (Springer Nature)

---

## Folder Structure

```
SHARE_WITH_PROFESSOR/
├── RQ1_LOS_Fairness_Analysis.ipynb   ← Main notebook (75 code cells, fully executed)
├── README.md                          ← This file
├── figures/                           ← 102 publication-ready PNG figures
│   ├── FIG01-FIG12                   ← Manuscript figures
│   └── Supporting figures            ← All intermediate visualizations (upgraded v4)
├── tables/                            ← 30 CSV tables
│   ├── Descriptive statistics
│   ├── Model performance
│   ├── Fairness metrics (7 metrics × 4 attributes × 12 models)
│   ├── Stability protocols (P1, P2, P3)
│   ├── Cross-site portability
│   └── Combined reliability scores
└── scripts/                           ← Build & execution scripts
    ├── build_notebook_v4.py          ← Generates the notebook (4046 lines)
    └── run_notebook.py               ← Executes the notebook
```

## Key Results

| Metric | Description | Output |
|--------|-------------|--------|
| **12 ML Models** | LR, RF, HistGB, XGBoost, LightGBM, AdaBoost, GradientBoosting, DT, CatBoost, DNN, Stacking, Blend | Table 03 |
| **7 Fairness Metrics** | DI, SPD, EOPP, EOD, TI, PP, CAL | Table 06 |
| **4 Protected Attributes** | Race (5), Sex (2), Ethnicity (2), Age Group (4) | Table 06 |
| **3 Stability Protocols** | P1: K=30 resampling + VFR, P2: Sample-size sensitivity, P3: Cross-hospital K=20 | Tables 09-15 |
| **30 Random Subsets** | Subset/subgroup fairness variation | Table 16 |
| **Intersectional Analysis** | Race×Sex, Race×Age, Ethnicity×Age | Table 17 |
| **AFCE** | Fairness-through-awareness comparison | Table 18 |
| **Fairness Intervention** | Multi-lambda reweighing + per-group threshold | Table 18-19 |

## Notebook Sections (17 Sections, 75 Code Cells)

1. **Environment Setup** — imports, GPU detection, output dirs
2. **Data Loading & QC** — Texas PUDF, 925,128 records
3. **EDA** — demographics, LOS distributions, hospital patterns
4. **Feature Engineering** — target encoding, feature matrix
5. **Model Training** — 12 classifiers (GPU-accelerated)
6. **Performance Evaluation** — ROC, PR, calibration, confusion matrices
7. **Comprehensive Fairness Analysis** — 7 metrics × 4 attributes × 12 models
8. **Fairness Deep-Dive** — radar charts, calibration by group, intersectional, cross-hospital
9. **3-Protocol Stability** — resampling (K=30), sample-size sensitivity, seed perturbation, threshold sensitivity, GroupKFold
10. **Metric Disagreement Matrix** — 7×7 pairwise disagreement analysis
11. **Cross-Site Fairness Portability** — K=20 hospital clusters, violin plots, Fleiss' kappa
12. **20-30 Subset/Subgroup Analysis** — random subsets + intersectional
13. **AFCE** — fairness-through-awareness models
14. **Fairness Intervention** — reweighing + per-group thresholds
15. **Reliability Dashboard** — combined scores (Table 9 equivalent)
16. **Publication-Ready Figures** — FIG01-FIG12 + combined panels
17. **Summary Dashboard** — final results + JSON export

## Execution

- **Runtime:** ~70 minutes on NVIDIA RTX 5070 Laptop (102 figures, 30 tables)
- **Python:** 3.11.9 with venv
- **Dataset:** Texas PUDF (not included for data governance, ~500 MB)

## Manuscript Figure Mapping

| Figure | Description | File |
|--------|-------------|------|
| FIG01 | Study Pipeline | 39_FIG01_study_pipeline.png |
| FIG02 | Demographics | 40_FIG02_demographics.png |
| FIG03 | LOS Distribution | 41_FIG03_los_distribution.png |
| FIG04 | Reliability Framework | 42_FIG04_reliability_framework.png |
| FIG05 | Fairness Heatmap | 43_FIG05_fairness_heatmap_pub.png |
| FIG06 | Metric Disagreement | 30_metric_disagreement_matrix.png |
| FIG07 | Bootstrap Distributions | 21_bootstrap_ci.png |
| FIG08 | CV Curves | 26_protocol2_cv_curves.png |
| FIG09 | Hospital Violins | 31_cross_site_violin_plots.png |
| FIG10 | Reliability Dashboard | 38_reliability_dashboard.png |
| FIG11 | Failure Modes | 44_FIG11_failure_modes_taxonomy.png |
| FIG12 | Portability Map | 45_FIG12_portability_mechanism.png |
