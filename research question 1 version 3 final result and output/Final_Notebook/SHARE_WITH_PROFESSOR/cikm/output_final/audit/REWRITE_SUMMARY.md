# REWRITE_SUMMARY

_Run timestamp: 2026-05-01T10:50:23_

## What changed from CIKM_2026_LOS_Fairness_13042026.ipynb

### FIX 1 · Best model identity → XGBoost
- All single-best-model analyses now run on **XGBoost** (AUROC = 0.9528,
  Accuracy = 0.8776).
- VFR (Section 8), reconciliation (T6), per-cluster (T16), intervention (Section 11) all use
  XGBoost predictions, not LGB-XGB Blend.

### FIX 2 · Demographic disclosure
- Three diagnostics run in Section 3.1 (unique-row, RACE×ETHNICITY, top-10 LOS clustering).
- Duplication ratio observed: **1.01** (LOW DUPLICATION (real cohort or modest augmentation)).
- Methods disclosure inserted in Section 3.2.

### FIX 3 · Lambda value → λ = 2.0
- Re-ran the lambda sweep on grid {0, 0.5, 1, 2, 5, 10, 20, 30, 50, 100}.
- Selected smallest λ where all four DI ≥ 0.80 simultaneously and accuracy drop ≤ 5 pp.
- Selected λ = **2.0** (recorded in T13).

### FIX 4 · Four manuscript-claim corrections
- Practically-stable combos (VFR ≤ 10%):     259/336 (77.1%) (manuscript said 273/336).
- Cells with between-cluster CV > 0.50:      17/28 (manuscript said 11/28).
- Unanimous fair (model, attr):              8/48 (16.7%) (manuscript said 8/48).
- At-least-one-metric disagreement:          83.3% (manuscript said 83.3%).

### FIX 5 · Fleiss kappa reframing
- Per-cell Fleiss κ is degenerate; correct decomposition is per-metric × 4 attributes × 20 folds.
- Per-metric κ (notebook): DI=+0.204, SPD=+0.215,
  EOPP=+0.674, EOD=+0.601,
  TI=+1.000, PP=+0.235, CAL=+0.016.
- Overall κ (28 items × 20 raters): +0.506 (moderate).

### FIX 6 · K-sensitivity (real GroupKFold)
- Re-ran K=10, K=20, K=40 GroupKFold (T17).
- All κ values lie within [-1, +1].

### FIX 7 · Intervention ablation (4 rows)
- (1) Standard: Acc=0.8776, Fair-cells=20/28.
- (2) Reweighing only: Acc=0.8578, Fair-cells=18/28.
- (3) Reweigh + per-group thresholds: Acc=0.8139, Fair-cells=23/28.
- (4) Full Fair: Acc=0.8347, Fair-cells=21/28.

### FIX 8 · Per-cluster transferability honest accounting
- DI worst attribute improved at: **19/20** clusters.
- All four DI ≥ 0.80 simultaneously at: **14/20** clusters.
- Accuracy stayed within 5 pp at: **16/20** clusters.

### FIX 9 · General code cleanup
- RANDOM_STATE = 42 fixed at the top.
- Imports consolidated into a single Section 1 cell.
- All CSV writes go to output_final/* and results_final/* (original output/ untouched).
- Predictive metrics: 4 decimals; fairness metrics: 3 decimals.

## Output files written
- 20 T-files in output_final/tables/
- 6 figures in output_final/figures/
- 5 audit artefacts in output_final/audit/
- intermediate CSVs in results_final/

## Verification result
ALL VERIFICATION CHECKS PASSED
(blank)
