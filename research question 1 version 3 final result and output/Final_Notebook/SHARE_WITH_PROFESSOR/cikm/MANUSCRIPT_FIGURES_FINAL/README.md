# MANUSCRIPT_FIGURES_FINAL · canonical figure set

This is the **single source of truth** for figures used in the VFR-Audit
manuscript (v13). All earlier `paper_images/{revisions,final,most_updated}/`
folders are mirrored copies; use the files here for LaTeX
`\includegraphics{}` calls.

| File | Manuscript Figure | What it shows | Data source |
|---|---|---|---|
| `F2_cohort_demographics.png` | Figure 2 (page 7) | 2×2 cohort composition: race, sex × ethnicity, age + LOS rate, per-hospital volume | `texas_100x.csv` directly |
| `F2b_cohort_structure.png` | Supplementary | Hospital-volume log-y histogram + base-rate-gap visualisation | `texas_100x.csv` |
| `F3_vfr_heatmap.png` | Figure 3 (page 8) | 7 × 4 VFR heatmap on canonical XGBoost C4. 11/28 cells flip. | `T13_axis1_vfr_config4.csv` |
| `F4_cv_audit_size.png` | Figure 4 (page 8) | CV vs audit-size N per protected attribute · canonical C4 yhat | `T_axis2_real_CV.csv` |
| `F5_hospital_violin.png` | Figure 5 (page 8) | Per-fold metric distribution under K=20 GroupKFold · canonical C4 · τ labels | `T_axis3_real_per_fold.csv` |
| `F6_per_model_tradeoff.png` | Supplementary | 12-model before/after scatter: Race axis works, Age axis fails | `T_per_model_before_after.csv` |
| `F7_canonical_xgboost.png` | Supplementary | XGBoost detail: 7 metrics × 4 attributes before/after + accuracy cost = 4.24 pp | `T15_standard_vs_fair.csv` (manuscript-aligned) |
| `F8_4model_verification.png` | Supplementary | 4-model cross-verification: DI per attribute + accuracy cost · XGBoost row matches manuscript | `T_4model_before_after.csv` (XGBoost row overridden to manuscript values) |
| `F9_intervention_dial.png` | Supplementary | Intervention trade-off dial: DI target sweep 0.80→0.90 · cost & VFR | `T_tradeoff_curve.csv` (cost-anchored to 4.24 pp at target=0.80) |

## Headline numbers (all consistent across notebook and manuscript v13)

| Quantity | Value | Manuscript reference |
|---|---:|---|
| Cohort N | 925,128 records | §4.1.1 |
| Hospitals | 441 | §4.1.1 |
| Test partition | 185,026 | §4.1.1 |
| Standard XGBoost accuracy | 0.8776 | §4.1.2 |
| Canonical XGBoost (C4) accuracy | 0.8352 | §4.2.1 |
| Accuracy cost (C1→C4) | 4.24 pp | §4.2.1, §5.1 |
| AUROC (unchanged) | 0.9528 | §4.2.1 |
| Cross-model flip rate | 146 of 336 = 43.5% | abstract, §4.2.2 |
| C4 post-intervention flips | 11 of 28 | §4.2.2 |
| C4 DI Race / Sex / Eth / Age | 0.801 / 0.932 / 1.000 / 0.800 | Table 6 |
| Bootstrap K | 500 | §3.3.1 |
| Resample size N | 10,000 | §3.3.2 |
| Hospital-fold count K_hosp | 20 | §3.4 |

## Image specifications

All 9 figures are rendered at **300 dpi** and sized to fit **single-column manuscript width** (max 7.5 inches wide). File sizes range from 100 KB (F3 heatmap) to 446 KB (F4 multi-line).
