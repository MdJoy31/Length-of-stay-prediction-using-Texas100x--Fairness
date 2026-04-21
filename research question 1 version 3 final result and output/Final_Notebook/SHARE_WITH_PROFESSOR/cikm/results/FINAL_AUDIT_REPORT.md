# CIKM 2026 — Final Audit Report
**Notebook:** `CIKM_2026_LOS_Fairness.ipynb`
**Manuscript:** `main.tex` (provided inline)
**Data:** `texas_100x.csv` (N=925,128; 441 hospitals)
**Date:** 2026-04-20

---

## Executive summary

Six classes of discrepancy exist between the manuscript and the notebook. Code-level fixes are complete. Execution of the full retrained pipeline was attempted and got as far as Cell 19 (Cell 9 in the paper's numbering — the fairness heatmap) before the kernel died with an out-of-memory error. The partial run *confirms* that the Stage 1 threshold fix behaves correctly: under corrected thresholds the fairness heatmap reports `AGE_GROUP: 0–2/7 fair across models (mean 1.4)` — matching main.tex Table 7 (2/7 for Age Group) rather than the pre-patch 5/7 figure.

| # | Finding | Status |
|---|---|---|
| 1 | `FairnessCalculator.THRESHOLDS` used EOPP/EOD=0.20, CAL=0.10 (non-standard). Inflated "fair" count by 3/28. | **FIXED** (Cell 4) — verified in partial notebook run |
| 2 | Notebook trained **Bagging + Extra Trees**; manuscript describes **PyTorch DNN + LGB-XGB Blend** | **FIXED** (Cell 13) — with sklearn MLP fallback for systems without torch |
| 3 | RACE labels in main.tex Table 1 produce mathematically impossible overlap (54% of all patients would be Black AND Hispanic) | **DOCUMENTED** — needs author decision (see D6) |
| 4 | Random seeds not pinned for torch/xgb beyond `np.random.seed` | **FIXED** (Cell 3) |
| 5 | `{lam:.0f}` print format rounds λ=0.5→"0" | **FIXED** (Cell 34) |
| 6 | **Model hyperparameters don't match main.tex Table 3**. Notebook: XGB(n_est=500, depth=8); main.tex: XGB(n_est=1000, depth=10). Similar for LightGBM. This is why the notebook's actual AUC (0.9316) cannot reach main.tex's claimed 0.953. | **NOT FIXED** — changing the hyperparameters is a separate decision |

---

## What the partial notebook run confirmed

The notebook ran for 6 min 41 s (400 s) and completed 7 of 12 model trainings before the kernel died:

| Model | Accuracy | AUROC | F1 | Time |
|---|---|---|---|---|
| Logistic Regression | 0.7593 | 0.8318 | 0.7143 | 0.4 s |
| Decision Tree | 0.8359 | 0.9048 | 0.8138 | 2.4 s |
| Random Forest | 0.8452 | 0.9268 | 0.8246 | 25.3 s |
| Gradient Boosting | 0.8453 | 0.9273 | 0.8246 | **273.7 s** |
| AdaBoost | 0.8044 | 0.8905 | 0.7788 | 63.5 s |
| XGBoost | **0.8501** | **0.9316** | **0.8301** | 4.4 s |
| LightGBM | 0.8492 | 0.9308 | 0.8288 | 4.1 s |
| CatBoost | — kernel died during CatBoost GPU training — | | | |

**The actual notebook best model is XGBoost at AUROC 0.9316 — not the LGB-XGB Blend at 0.953 claimed in main.tex.** A 0.6*LightGBM + 0.4*XGBoost blend of these probabilities would land at ~0.932, nowhere near 0.953. The AUC=0.953 claim in main.tex cannot be reproduced with the notebook's current hyperparameter configuration.

---

## Deliverable 0 — Consistency audit

**Files:** `results/consistency_audit.csv`, `results/consistency_audit.md`

Main.tex's Tables 7 (fairness-best), 10 (VFR), and 11 (subset fluctuation) are **internally consistent with each other** under the standard thresholds stated in Table 7's caption. All VFR values in Table 10 map exactly onto the pass-counts in Table 11 (e.g., Race×EOPP: 16/30 pass → min(16,14)/30 = 46.7% VFR). No hard contradictions.

But main.tex **does not match the notebook**: the notebook's thresholds were EOPP=0.20, EOD=0.20, CAL=0.10 until this patch. Under those relaxed thresholds, the exact same point values in Table 7 yield:

| Attribute | main.tex (standard) | Notebook pre-patch |
|---|---|---|
| Race | 4/7 | 4/7 |
| Sex | 5/7 | 5/7 |
| Ethnicity | 7/7 | 7/7 |
| **Age Group** | **2/7** | **5/7** |
| **Total** | **18/28** | **21/28** |

Stage 1 patch applied the corrected thresholds. Verified by the post-patch Cell 19 output: `AGE_GROUP: 0–2/7 fair across models (mean 1.4)`.

---

## Deliverable 1 — Fairness reconciliation table

**Files:** `results/fairness_reconciliation_XGBoost.csv`, `results/fairness_reconciliation_LGB_XGB_Blend.csv`, `results/fairness_reconciliation.md`

Under the corrected thresholds:

- **XGBoost** (notebook actual): **9/28** verdicts pass. Race CAL=0.188 at 2.0σ, Race DI=0.663 at 1.2σ, Sex DI=0.755 at 2.5σ all fail safely. Race EOPP/EOD (0.119/0.125) fail marginally.
- **LGB-XGB Blend** (main.tex reported): **18/28** verdicts pass. Matches manuscript's 4/7+5/7+7/7+2/7.

The 9 vs 18 difference is driven by the model choice. Since main.tex's numbers cannot be produced by the current notebook without hyperparameter changes and the Blend model being trained, the 18/28 headline depends entirely on an un-reproduced model.

---

## Deliverable 2 — λ selection sweep

**Script:** `_scripts/lambda_sweep.py` (standalone, not executed — needs free memory)

Runs LGB-XGB Blend at λ ∈ {0, 0.5, 1, 3, 5, 10, 15, 30, 50, 100}. Selects smallest λ with all-DI-pass and accuracy drop < 5 pp.

**Prior evidence (from my earlier audit of the pre-patch CSV `output/tables/Table6_Intervention.csv`):** the selected configuration was `Standard (λ=0) + α_sr=0.4, α_tpr=0.9, α_ppv=0.3` — i.e., pure threshold optimization with no reweighing. Manuscript's claim "λ=30 is the selected reweighing intensity" is not supported by the previous run and must be re-checked after retraining.

---

## Deliverable 3 — Standard vs Fair head-to-head

Cell 35 patch saves `results/intervention_standard_vs_fair.csv` when the notebook runs. Not produced in this session because the kernel died before Cell 35.

---

## Deliverable 4 — Per-hospital-cluster transferability

**Script:** `_scripts/per_cluster_intervention.py` (standalone, not executed)

Runs K=20 GroupKFold: train Standard + Fair on 19 clusters, evaluate on held-out cluster with 100-bootstrap CIs. Expected runtime 30–60 min on GPU.

---

## Deliverable 5 — Three-panel figure

**Script:** `_scripts/three_panel_figure.py` (standalone)

Runs once D2 and D4 CSVs exist. 300 dpi, colour-blind-safe, three panels: Pareto frontier, DI before/after, per-cluster trajectories.

---

## Deliverable 6 — Demographic audit

**Files:** `results/demographic_audit.csv`, `results/demographic_audit.md`, `results/race_ethnicity_crosstab_counts.csv`

- **Sex and Ethnicity** labels in main.tex are correct. LOS rates match to within 0.1%.
- **Race labels produce impossible overlap:** 99.4% of "Asian/PI" patients and 83.1% of "Black" patients are simultaneously coded ETHNICITY=1 (Hispanic). Under the main.tex labels, 54% of all patients would be both Black AND Hispanic.
- Most likely explanation: `texas_100x` is a synthetic/augmented dataset (the `_100x` naming suggests 100× oversampling) and the race labels were permuted in augmentation. Second-most-likely: THCIC double-coding (well-documented field quality issue).
- The LOS rates per "race" match main.tex Table 1 exactly (White 40.4%, Black 45.3%, Asian/PI 52.3%, NA 41.0%, Other 33.4%) — so the manuscript's numbers are transcribed correctly from the data; only the race labels themselves are wrong or ambiguous.
- Five age groups in notebook are collapsed to four (Middle-Aged = 40–54 + 55–64) in main.tex Table 1 — undisclosed.

---

## Run instructions (after freeing memory)

```bash
cd "fairness_project_v1/research question 1 version 3 final result and output/Final_Notebook/SHARE_WITH_PROFESSOR/cikm"

# Full notebook re-run — needs ~8-16 GB free RAM, ~30-60 min on GPU
python _scripts/run_notebook.py

# Standalone analyses (each ~30-60 min)
python _scripts/lambda_sweep.py
python _scripts/per_cluster_intervention.py

# Figure (seconds)
python _scripts/three_panel_figure.py

# Re-run audits on the new outputs
python _scripts/consistency_audit.py
python _scripts/reconciliation_table.py
```

---

## Open decisions for the author

1. **RACE labels (D6).** Choose one:
   - Obtain the THCIC PUDF FY 2019–2023 data dictionary and re-label.
   - Treat RACE as anonymous categories (Group A–E) and disclose the data is synthetic/augmented.
   - Keep labels but add a Methods paragraph explaining the RACE × ETHNICITY overlap pattern (THCIC double-coding).

2. **Model hyperparameters.** main.tex Table 3 lists n_est=1000, depth=10 for XGBoost and n_est=1500, num_leaves=255 for LightGBM. Notebook uses n_est=500/depth=8 and n_est=500/num_leaves=63. Either:
   - Update the notebook hyperparameters to match main.tex (required to reproduce AUC=0.953).
   - Or update main.tex Table 3 to match notebook and revise the performance claims downward.

3. **λ=30 claim.** If the post-patch λ sweep confirms the pre-patch finding that λ=0 is selected, the paper's methodological framing ("intersectional λ-reweighing intervention") must change to something like "per-group threshold optimization with optional reweighing".

4. **Threshold disclosure.** main.tex's Table 7 caption states the standard thresholds; the notebook is now aligned with them. No change needed if the author always intended standard thresholds.

---

## Final verdict

**Data-ready to submit?** **NO — Needs-Rerun plus one label decision.**

Code fixes are complete and applied. The partial notebook run confirms the threshold patch works (fairness verdicts at corrected thresholds already match main.tex for Age Group). The three outstanding issues are:
- Memory pressure on this machine (3 GB free of 33 GB total) prevents completing the full retrain.
- The AUC=0.953 claim is **not reproducible at the current notebook hyperparameters**; retraining at main.tex Table 3's hyperparameters is required.
- The RACE label interpretation needs a one-line author decision before this goes to reviewers.

Everything else is in `results/` and can be regenerated deterministically once the notebook re-runs with the patches in place.
