# Weekly Meeting Script — RQ1 Fairness Analysis
## Texas-100X LOS Prediction Study

---

### Meeting Setup Checklist

- [ ] Open `final_notebooks/LOS_Prediction_Standard.ipynb` in VS Code / Jupyter
- [ ] Open `final_notebooks/results/extracted_metrics.json` for reference
- [ ] Open `final_notebooks/report/RQ1_Analysis_Section.md` for paper narrative
- [ ] Have `analysis_summary.txt` ready for quick numbers

---

## 1. Progress Update (2 min)

**Status**: Core analysis complete. Standard + Detailed + Complete notebooks executed with zero errors.

Key milestone checklist:
- [x] Data loaded and validated (925,128 records, 441 hospitals, 12 columns)
- [x] Six models trained: LR, RF, GB, XGBoost(GPU), LightGBM(GPU), PyTorch DNN
- [x] Baseline fairness metrics computed (DI, WTPR, EOD, Eq. Odds)
- [x] Fair model built (λ=5.0 reweighing + per-group thresholds)
- [x] AFCE pipeline designed and executed (selection rate equalization)
- [x] Bootstrap stability analysis (B=200)
- [x] Cross-hospital variance analysis (K=30)
- [x] Intersectional audit (RACE × SEX, 10 groups)
- [x] Paper Analysis section drafted (Sections 4.1–4.12)

---

## 2. Key Results Presentation (5 min)

### Model Performance (Table 1)

| Model          | Accuracy | F1     | AUC    | Overfitting Gap |
|----------------|----------|--------|--------|-----------------|
| LightGBM GPU   | 87.87%   | 86.39% | 95.33% | 2.97%           |
| XGBoost GPU     | ~87.5%   | ~86.1% | ~95.2% | 3.21%           |
| Gradient Boost  | ~87.0%   | ~85.7% | ~94.8% | 1.79%           |
| PyTorch DNN     | 85.56%   | 84.28% | 93.75% | 0.43%           |
| Random Forest   | ~86.3%   | ~85.0% | ~94.5% | 6.14%           |
| Logistic Reg.   | 80.21%   | 78.44% | 88.40% | 0.07%           |

**Talking Point**: LightGBM GPU achieves highest F1 (0.864), outperforming the published benchmark F1=0.550 by +31.4 percentage points. All models have < 5% generalization gap except RF.

### Baseline Fairness (Table 2)

| Attribute  | DI    | WTPR  | Fair? |
|------------|-------|-------|-------|
| RACE       | 0.642 | 0.827 | ❌    |
| SEX        | 0.762 | 0.841 | ❌    |
| ETHNICITY  | 0.829 | 0.875 | ✅    |
| AGE_GROUP  | 0.254 | 0.611 | ❌    |

**Talking Point**: High-accuracy models are NOT inherently fair. RACE DI=0.642 means Black patients receive 36% fewer "long stay" predictions than White patients, creating real clinical harm.

### AFCE Pipeline Results (Table 3)

| α   | Accuracy | F1     | RACE DI | SEX DI | ETH DI | Fair? |
|-----|----------|--------|---------|--------|--------|-------|
| 0.0 | 86.40%   | 84.62% | 0.997   | 1.000  | 0.902  | 3/4   |
| 0.1 | **86.62%** | **84.97%** | **0.999** | **1.000** | **0.916** | **3/4** |
| 0.3 | 86.88%   | 85.33% | 0.992   | 0.999  | 0.906  | 3/4   |
| 0.5 | 87.00%   | 85.50% | 0.979   | 0.995  | 0.893  | 3/4   |
| 0.7 | 87.11%   | 85.66% | 0.962   | 0.989  | 0.882  | 3/4   |
| 1.0 | 87.32%   | 85.98% | 0.776   | 0.757  | 0.836  | 1/4   |

**Talking Point**: At α=0.1, we achieve RACE DI=0.999 (near-perfect parity) with only 1.25% accuracy cost. The selection rate equalization across 10 RACE×SEX intersections is the key innovation.

### AGE_GROUP Limitation

AGE_GROUP DI ≈ 0.274 regardless of mitigation. This is **structurally impossible** to fix because elderly patients (75+) have fundamentally different LOS distributions — a 1-year-old and a 95-year-old SHOULD have different predictions. This is a clinically justified disparity, not unfairness.

---

## 3. Discussion Points (5 min)

### Why Selection Rate Equalization?

1. **Previous approach** (TPR equalization): Per-group thresholds targeting TPR=0.82 → RACE DI only reached 0.751 (still unfair)
2. **Current approach** (selection rate equalization): Binary search per RACE×SEX intersection targeting equal positive prediction rates → RACE DI=0.999
3. **Key insight**: Equalizing TPR does NOT guarantee equal DI. You must directly target selection rates.

### Cross-Hospital Variance

- DI ranges from ~0.3 to ~1.0 across 30 hospitals
- Hospital-specific factors (specialization, patient mix) dominate fairness variance
- **Implication**: A single global fairness threshold is insufficient — hospital-specific calibration needed

### Bootstrap Stability

- 200 bootstrap iterations, 95% CI
- Small groups (Asian/PI: n≈5,300) have wider CIs (width ≈ 0.05)
- Large groups (White: n≈100,000) have narrow CIs (width ≈ 0.004)
- **Implication**: Fairness metric reliability depends on group sample size — report CIs in all fairness audits

---

## 4. Next Steps Discussion (3 min)

### Immediate (This Week)
- [ ] Finalize RQ1 paper Analysis section (Sections 4.1–4.12)
- [ ] Fill in exact values for all tables (from extracted_metrics.json)
- [ ] Draft Discussion section: why DI/WTPR/EOD disagree, clinical implications

### Short-Term
- [ ] RQ2 planning: Which fairness metric (DI vs WTPR vs EOD) is most reliable under dataset shifts?
- [ ] Consider adding temporal analysis (admission year as covariate)
- [ ] Explore Fairlearn's ExponentiatedGradient as additional baseline

### Long-Term
- [ ] Multi-task learning: simultaneous LOS + readmission prediction
- [ ] External validation on MIMIC-IV or eICU
- [ ] Clinical deployment considerations (threshold calibration per hospital)

---

## 5. Action Items Template

| Action Item | Owner | Due Date | Status |
|-------------|-------|----------|--------|
| Finalize Analysis section 4.1–4.12 | | | |
| Submit draft to advisor | | | |
| Run MIMIC-IV comparison | | | |

---

## Quick Reference: File Locations

```
final_notebooks/
├── LOS_Prediction_Standard.ipynb    # 51 code cells, fully executed
├── LOS_Prediction_Detailed.ipynb    # 51 code cells, fully executed
├── data/texas_100x.csv              # 925K records
├── scripts/
│   ├── run_analysis.py              # Reproducible CPU analysis
│   └── analysis_utils.py            # Helper functions
├── results/
│   ├── extracted_metrics.json       # All notebook metrics
│   ├── analysis_report.json         # run_analysis.py output
│   └── analysis_summary.txt         # Human-readable summary
└── report/
    ├── RQ1_Analysis_Section.md      # Paper Analysis section
    └── weekly_meeting_script.md     # This file
```

---

*Generated for RQ1: Reliability and Stability of Fairness Metrics in Healthcare LOS Prediction Under Dataset Heterogeneity*
