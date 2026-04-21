# D1 Fairness Reconciliation Table

## D1a — Current notebook best model (XGBoost)

28-row table from `output/tables/cikm_vfr_all_metrics.csv` (K=30 bootstrap of the XGBoost test set).

- Verdicts recomputed under **corrected thresholds** (EOPP=EOD=0.10, CAL=0.05).
- "Margin (sigma)" = distance from bootstrap mean to threshold in sigmas.
  Positive = safely passing; negative = safely failing; |margin| < 1 = unstable.
- "VFR at corrected thresholds" is a **Gaussian estimate**; exact recomputation
  requires the per-resample raw metric values (not in the CSV). For precise
  values the VFR cell (Cell 23/24) must be re-run after the threshold fix.

Total XGBoost verdicts at corrected thresholds: **9/28**.

See `results/fairness_reconciliation_XGBoost.csv`.

## D1b — main.tex best model (LGB-XGB Blend)

28-row table using main.tex Table 7 point values, Table 11 pass counts, and
Table 10 VFR values. The bootstrap sigma is **inferred** from the empirical
pass rate (x/30) via the Gaussian inversion sigma = (tau - mu) / Phi^-1(p).
For cells where the pass rate is 0/30 or 30/30, sigma cannot be inferred
exactly — these cells are marked "Very stable".

Total LGB-XGB Blend verdicts at corrected thresholds: **18/28**.

See `results/fairness_reconciliation_LGB_XGB_Blend.csv`.

## Caveat

D1b is complete for the main.tex-reported numbers under the assumption that
they are internally consistent (verified in D0). The notebook itself does not
currently train a LGB-XGB Blend model — that is a Stage 2 change. Until the
notebook is re-run with the corrected model set, D1a (XGBoost) is the only
reconciliation that reflects the **actual notebook state**; D1b reflects the
**manuscript's reported state**.
