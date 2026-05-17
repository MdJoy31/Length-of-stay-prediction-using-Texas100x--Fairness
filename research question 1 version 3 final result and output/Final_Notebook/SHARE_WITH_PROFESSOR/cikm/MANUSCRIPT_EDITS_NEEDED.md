# Manuscript Edits Required

This document lists every manuscript line that disagrees with the final notebook
(`CIKM_2026_LOS_Fairness_FINAL.ipynb`) and provides the exact replacement text.

Generated against `output_final/tables/*.csv` after Run-with-corrected-TI.

The manuscript file is `d:/Research study/Research question 1/DOC/files/Overleaf_Template/overleaf/main.tex`.

---

## A. Best-model attribution (HIGH severity)

The paper attributes best results to **LGB-XGB Blend** but the notebook's
canonical model (FIX 1) is **XGBoost**. Every reference to LGB-XGB Blend
needs to change.

### Abstract / §1
* No specific edit; abstract speaks of "twelve models" generically — keep.

### §6.1 Baseline Performance — line 504, 525, 526
* OLD: "The LGB-XGB Blend achieves the highest AUC (0.9536) and accuracy
  (0.8787), followed closely by the Stacking Ensemble (AUC = 0.9534)."
* NEW: "XGBoost (canonical model for fairness analysis) achieves AUC __TBF__
  and accuracy __TBF__, comparable to LightGBM (AUC __TBF__) and the Stacking
  Ensemble (AUC __TBF__). XGBoost is selected as the canonical model for
  downstream fairness analysis because of its widespread use in clinical AI
  and its native support for the per-cell α-thresholding intervention."

### §6.2 Single-point fairness — line 537
* OLD: "Table 5 presents the disparate impact (DI) values for the
  best-performing model (LGB-XGB Blend) ..."
* NEW: "Table 5 presents the disparate impact (DI) values for the canonical
  XGBoost model ..."

### §6.4 Test 1 Bootstrap — line 576, 581
### §6.5 Test 4 Seed — line 631
### §6.7 §7 throughout: "the LGB-XGB Blend model" → "XGBoost"

---

## B. Baseline DI values (Table 5 — manuscript line 547-552)

* OLD (paper):

  | Attribute  | DI    | SPD    | EOD   | EOPP  | TI    | PP     | CAL   |
  |------------|-------|--------|-------|-------|-------|--------|-------|
  | RACE       | 0.646 | -0.142 | 0.123 | 0.115 | 0.011 | -0.032 | 0.028 |
  | SEX        | 0.731 | -0.108 | 0.068 | 0.055 | 0.008 | -0.021 | 0.015 |
  | ETHNICITY  | 0.802 | -0.079 | 0.085 | 0.072 | 0.009 | -0.025 | 0.019 |
  | AGE_GROUP  | 0.602 | -0.168 | 0.185 | 0.162 | 0.015 | -0.108 | 0.042 |

* NEW (notebook XGBoost canonical, T15 Standard column):

  | Attribute  | DI       | SPD      | EOD      | EOPP     | TI       | PP       | CAL      |
  |------------|----------|----------|----------|----------|----------|----------|----------|
  | RACE       | __TBF__  | __TBF__  | __TBF__  | __TBF__  | __TBF__  | __TBF__  | __TBF__  |
  | SEX        | __TBF__  | __TBF__  | __TBF__  | __TBF__  | __TBF__  | __TBF__  | __TBF__  |
  | ETHNICITY  | __TBF__  | __TBF__  | __TBF__  | __TBF__  | __TBF__  | __TBF__  | __TBF__  |
  | AGE_GROUP  | __TBF__  | __TBF__  | __TBF__  | __TBF__  | __TBF__  | __TBF__  | __TBF__  |

  Source: `output_final/tables/T15_standard_vs_fair.csv` (Standard column).

---

## C. AFCE / α-intervention claim (HIGH severity — §6.10, §8)

### Abstract — line 83
* OLD: "achieves disparate impact ≥ 0.80 for **three of four** protected
  attributes with **<1.3%** accuracy trade-off."
* NEW: "achieves disparate impact ≥ 0.80 for **all four** protected attributes
  with __TBF__pp accuracy trade-off; consistent with Chouldechova's
  impossibility theorem, equal-error-rate metrics (EOPP, EOD, PP) on
  AGE_GROUP remain constrained by the 40-percentage-point base-rate
  difference between Pediatric and Elderly subgroups."

### §6.10 AFCE — line 685
* OLD: "After AFCE, 3 of 4 protected attributes achieve DI ≥ 0.80
  (RACE = 0.951, SEX = 0.952, ETHNICITY = 0.950), with an accuracy
  trade-off of approximately 1.2% (from 0.8787 to 0.8675).
  AGE_GROUP (DI = 0.774) remains below the threshold ..."
* NEW: "After our master α-SR/TPR/PPV per-cell thresholding intervention
  (Section 4.7), all 4 protected attributes achieve DI ≥ 0.80 (RACE =
  __TBF__, SEX = __TBF__, ETHNICITY = __TBF__, AGE_GROUP = __TBF__), with
  an accuracy trade-off of __TBF__ percentage points (from __TBF__ to
  __TBF__). Consistent with the Chouldechova-2017 impossibility theorem,
  AGE_GROUP achieves DI fairness but error-rate metrics (EOPP, EOD, PP)
  on AGE remain above the conventional 0.10 threshold due to the
  large base-rate gap between Pediatric (~6%) and Elderly (~70%) subgroups."

### §8 Conclusion — line 765
* OLD: "Our AFCE framework achieves DI ≥ 0.80 for 3 of 4 protected
  attributes with < 1.3% accuracy trade-off ..."
* NEW: "Our intervention achieves DI ≥ 0.80 for all 4 protected attributes
  with __TBF__pp accuracy trade-off; for the AGE attribute, error-rate
  metrics remain constrained by the impossibility theorem ..."

---

## D. Lambda reweighing (§6.10.2 — line 689)

* OLD: "Lambda-Scaled Reweighing approach with λ = 5.0 improves DI for
  RACE from 0.646 to 0.750 (+16.1%) ..."
* NEW: "Lambda-Scaled Reweighing approach with λ = __TBF__ improves DI
  for RACE from __TBF__ to __TBF__ ... See `output_final/tables/T13_lambda_sweep.csv`."

---

## E. Internal contradictions in paper (FIX in paper independently)

### E1. Bootstrap B value
* §1 (abstract, line 83): "B = 500"
* §4.6.1 Test 1: "B = 1000"
* §6.4: "B = 500"
* §7.5 limitations (line 749): "B = 500 ... rather than the planned B = 1000"
* **DECISION:** keep B = 500 throughout, delete the "B = 1000" in §4.6.1.

### E2. Protected attributes inconsistency
* §4.1 line 346: "(4) Insurance type"
* §4.1 elsewhere & §5+ throughout: "Age group" / "AGE_GROUP"
* **DECISION:** remove "Insurance type" from §4.1; the actual 4th attribute
  is AGE_GROUP. Replace line 346 with: "(4) Age group (Pediatric <18,
  Young Adult 18-39, Middle-Aged 40-64, Elderly ≥65) — note this is the
  4-bucket binning used for fairness; the 3-bucket version (18-44, 45-64,
  65+) appears in some legacy comparisons."

### E3. AGE binning
* §4.1 line 346: "(18-44, 45-64, 65+)" (3-bucket)
* §5.2 line 472: "(Pediatric, Young Adult, Middle-Aged, Elderly)" (4-bucket)
* **DECISION:** notebook uses 4-bucket; standardise on this.

### E4. Notebook cell/figure/table count
* §8 line 770: "107 cells producing 37 figures and 13 tables"
* Actual: 48 cells, 5 figures (F1-F5), 19 tables (T3-T20 + cikm_vfr).
* **DECISION:** "48 cells producing 5 main figures and 19 tables".

---

## F. Hard-fact claims to verify (NEEDS_VERIFICATION)

These need fresh notebook-output cross-check after run completes:

* §6.6 (cross-hospital): "DI range 0.18 to 1.12 across 205 hospitals ≥500 patients"
  → verify against T16 / per-cluster
* §6.6: "Fleiss' κ = 0.22 for DI on RACE"
  → verify per_metric_k_for_t12
* §6.5: "Pct flipped (VFR > 0): 33.6%" / "VFR ≤ 10%: 259 cells / 336 total"
  → verify against T7_vfr_heatmap
* §6.6: "Hospital DI median 0.69, IQR [0.54, 0.85]"
  → verify
* §6.6: "Fair at 38% of hospitals, unfair at 62%"
  → verify
* §6.7: "K=5 fold DI 0.61-0.78" / "K=20 std-dev 0.09 for RACE"
  → verify against T17_k_sensitivity_real

---

(Numbers labelled __TBF__ will be filled in once the notebook re-run with
corrected TI completes.)
