# CIKM 2026 — Full Audit Report

Generated: 2026-04-21 00:15:56
Runtime: 472s

## Summary

- **Dataset**: 925,128 records × 7 features × 441 hospitals
- **Train/Test**: 740,102 / 185,026
- **Best Standard Model**: LightGBM  Acc=0.8492  AUC=0.9308
- **Fair Model**: λ-reweighed LightGBM + threshold optimization
- **Fair Model Global**: Acc=0.8006  All DI≥0.80: True

## Deliverable 0: Consistency Audit

- Consistency issues found: 0
- **Action**: Update generate_all_figures_tables.py thresholds to match notebook (EOPP=0.10, EOD=0.10, CAL=0.05)

## Deliverable 1: Fairness Reconciliation

- Standard model fair verdicts: 15/28
- Fragile verdicts (margin < 0.02): 1/28

## Deliverable 2: Lambda Selection

- Best λ: 0 → 15/28 fair  Acc=0.8492
- λ=0 (Standard): 15/28 fair  Acc=0.8492

## Deliverable 3: Standard vs Fair

- Standard: 15/28 fair
- Fair: 16/28 fair
- Accuracy cost: 4.9 pp

## Deliverable 4: Cross-Site Transferability (from Section 1)

- K=20 GroupKFold, Standard + Fair models
- Standard model: mean 9.2/28 fair across folds
- Fair model: mean 12.8/28 fair across folds
  - RACE: mean 2.2/7 fair (≥4/7 in 5% of folds)
  - SEX: mean 4.2/7 fair (≥4/7 in 70% of folds)
  - ETHNICITY: mean 4.5/7 fair (≥4/7 in 90% of folds)
  - AGE_GROUP: mean 1.9/7 fair (≥4/7 in 0% of folds)
  - Fair DI_RACE: mean=0.829, min=0.561
  - Fair DI_SEX: mean=0.947, min=0.831
  - Fair DI_ETHNICITY: mean=0.928, min=0.841
  - Fair DI_AGE_GROUP: mean=0.818, min=0.362

## Deliverable 5: Three-Panel Figure

- Saved: output/audit/figures/D5_three_panel_summary.png

## Deliverable 6: Demographic Audit

- Issues found: 3
  - RACE=1 (Native American): 96.8% are ETHNICITY=1 — potential double-coding
  - RACE=2 (Asian/Pacific Islander): 99.4% are ETHNICITY=1 — potential double-coding
  - RACE=3 (Black): 83.1% are ETHNICITY=1 — potential double-coding

## Files Generated

| File | Description |
|------|-------------|
| output/tables/Table6_CrossSite_StdFair_PerFold.csv | Per-fold cross-site results (Std+Fair) |
| output/tables/Table6_CrossSite_PerFold_Detail.csv | Detail table for all 20 folds |
| output/audit/Table6_CrossSite_Summary.csv | Summary statistics |
| output/audit/Table6b_Fleiss_Kappa_StdFair.csv | Fleiss' κ for both models |
| output/audit/D0_consistency_audit.csv + .md | Threshold consistency check |
| output/audit/D1_fairness_reconciliation.csv | 28-row reconciliation |
| output/audit/D2_lambda_selection.csv | Lambda sweep results |
| output/audit/D3_standard_vs_fair.csv | Head-to-head comparison |
| output/audit/D6_demographic_audit.md | RACE×ETH cross-tab |
| output/audit/figures/*.png | Audit figures |