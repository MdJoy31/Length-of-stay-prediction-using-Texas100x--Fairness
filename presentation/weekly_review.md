# Weekly Research Review — Fairness in LOS Prediction
## Md Rakibul Islam Joy | Masters by Research | February 2026

---

## Slide 1: Research Overview

**Research Question:** How reliable are fairness metrics in ML-based hospital length-of-stay prediction?

**Dataset:** Texas-100X Hospital Discharge Data
- 925,128 patient records from Texas hospitals
- 12 features including demographics, diagnosis, procedure codes
- Binary target: Length of Stay > 3 days (Extended Stay)

**Reference Paper:** Tarek et al. (2025) — "Reliability of Fairness Metrics under Synthetic Augmentation" (CHASE '25)

---

## Slide 2: Methodology

### Data Pipeline
1. **Feature Engineering** — Target encoding, hospital-level features, interaction features (35+ features)
2. **Model Training** — 6 ML models + Stacking Ensemble
   - Logistic Regression, Random Forest, Gradient Boosting
   - XGBoost (GPU), LightGBM (GPU), PyTorch DNN (GPU)
   - Stacking Ensemble (LGB + XGB + GB → meta-LR)
3. **Hyperparameter Optimization** — 50-agent parallel search

### Fairness Analysis
- 6 metrics: DI, WTPR, SPD, EOD, PPV Ratio, Equalized Odds
- 4 protected attributes: Race, Ethnicity, Sex, Age Group
- Subset stability testing: 1K → 925K samples

---

## Slide 3: Model Performance Results

| Model | Accuracy | F1-Score | AUC-ROC |
|-------|----------|----------|---------|
| Logistic Regression | 0.802 | 0.784 | 0.884 |
| Random Forest | 0.862 | 0.848 | 0.941 |
| Gradient Boosting | 0.874 | 0.858 | 0.950 |
| XGBoost (GPU) | 0.878 | 0.863 | 0.953 |
| **LightGBM (GPU)** | **0.879** | **0.864** | **0.954** |
| PyTorch DNN | 0.857 | 0.843 | 0.938 |
| Stacking Ensemble | ~0.880 | ~0.865 | ~0.954 |

**Key Finding:** LightGBM with hospital-level features achieves best performance.
Hospital features provided +3.3% F1 improvement.

![Model Performance](figures/02_roc_curves.png)

---

## Slide 4: Fairness Analysis Results

### Disparate Impact (DI) — Best Model (LightGBM)

| Protected Attribute | DI | Status | WTPR |
|--------------------|----|--------|------|
| RACE | 0.64 | Below 0.8 threshold | 0.83 |
| ETHNICITY | 0.83 | FAIR (≥0.8) | 0.84 |
| SEX | 0.76 | Below 0.8 threshold | 0.84 |
| AGE_GROUP | 0.25 | Below 0.8 threshold | 0.73 |

**Key Observations:**
- ETHNICITY achieves fairness threshold (DI ≥ 0.8)
- RACE, SEX are close but below threshold
- AGE_GROUP shows largest disparity (young adults vs elderly)

![Fairness Heatmap](figures/03_fairness_heatmap.png)

---

## Slide 5: Subset Stability Analysis

**Test:** How do fairness metrics change across 9 data volumes?
- Sizes tested: 1K, 2K, 5K, 10K, 25K, 50K, 100K, 200K, Full (185K)
- 10 random repetitions per size

### Key Findings:
- **DI** is highly unstable below 5K samples (CV > 10%)
- **WTPR** stabilizes earlier (by 5K samples)
- **EOD** is the most volatile metric (CV up to 50%)
- **Minimum recommended sample size:** 10K for reliable fairness assessment

| Size | DI (RACE) | WTPR (RACE) | F1 |
|------|-----------|-------------|-----|
| 1K | 0.39 ± 0.25 | 0.72 ± 0.15 | 0.86 ± 0.02 |
| 5K | 0.63 ± 0.08 | 0.81 ± 0.04 | 0.86 ± 0.01 |
| 10K | 0.64 ± 0.05 | 0.83 ± 0.03 | 0.86 ± 0.00 |
| 50K | 0.64 ± 0.02 | 0.83 ± 0.01 | 0.86 ± 0.00 |
| Full | 0.64 | 0.84 | 0.86 |

![Subset Stability](figures/04_subset_fairness.png)

---

## Slide 6: Fairness Intervention & Paper Comparison

### Lambda-Reweighing Results
| Model | F1 | DI (Race) | WTPR |
|-------|----|-----------|------|
| Standard LightGBM | 0.864 | 0.64 | 0.84 |
| Fair Model (λ=5.0) | 0.854 | 0.68 | 0.81 |
| **Improvement** | -1.2% | **+6.3%** | -3.6% |

### vs Tarek et al. (2025)
| Metric | Paper Best | Ours (Standard) | Improvement |
|--------|-----------|-----------------|-------------|
| F1-Score | 0.550 | **0.864** | **+31.4pp** |
| DI | 1.110 | 0.828 | — |
| WTPR | 0.830 | **0.839** | **+0.9pp** |

**Our F1 is 57% higher** than the paper's best result.

![Paper Comparison](figures/05_paper_comparison.png)

---

## Slide 7: Summary & Next Steps

### Key Contributions
1. **+31% F1 improvement** over reference paper (0.864 vs 0.550)
2. **Hospital features** as key accuracy driver (+3.3% F1)
3. **Fairness metric stability analysis** across 9 data volumes
4. **Minimum 10K samples** required for reliable fairness assessment
5. **Lambda-reweighing** improves DI by ~6% with only ~1% F1 cost

### Limitations
- Protected attributes excluded from features (fairness constraint)
- Accuracy ceiling ~88% due to limited feature set (12 original columns)
- AGE_GROUP fairness remains challenging (inherent clinical differences)

### Next Steps
- Cross-dataset validation (MIMIC-III comparison)
- Causal fairness analysis
- Multi-class LOS prediction (instead of binary)
- Clinical deployment considerations
