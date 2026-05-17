# Manuscript-Claim ↔ Notebook-Evidence Map

For every quantitative claim in `main.tex`, this document points at the
exact notebook output that backs (or refutes) it. Use this when fixing
the manuscript.

Status legend:
* **TRUE** — claim matches notebook within ±1% rounding.
* **TRUE (≈)** — close (within ±5% relative or absolute Δ ≤ 0.02).
* **FALSE** — substantive mismatch — needs a manuscript edit.
* **FALSE-BETTER** — notebook *exceeds* paper claim; paper currently undersells the result.
* **CONTRADICTION** — paper contradicts itself; notebook agrees with the majority section.
* **PARTIAL** — claim partially supported (one decomposition matches, another does not).

Evidence files live under `output_final/tables/`.

---

## A · Dataset and setup

| # | Manuscript claim | Notebook evidence | Status |
|---|---|---|---|
| A1 | "925,128 records" | `T19_claim_verification.csv` row A1 → 925,128 | **TRUE** |
| A2 | "441 hospitals" | `T19_claim_verification.csv` row A2 → 441 | **TRUE** |
| A3 | "740,102 train / 185,026 test (80/20 stratified)" | cell 13 stdout: `Train: 740,102  Test: 185,026` | **TRUE** |
| A4 | "55.0% normal / 45.0% extended LOS" | `T3_descriptive.csv` overall row | **TRUE (≈)** |
| A5 | "12 ML models" | cell 21 stdout enumerates 12 models | **TRUE** |
| A6 | "7 fairness metrics × 4 attrs = 28 cells" | `T19_claim_verification.csv` row B1 → 336 = 12·28 | **TRUE** |
| A7 | "11 features after excluding 4 protected attrs" | cell 13 stdout: `Final feature set (11)` | **TRUE** (after paper edit) |

## B · Best-model performance (Table 4)

| # | Manuscript claim | Notebook evidence | Status |
|---|---|---|---|
| B1 | Best model = **LGB-XGB Blend** | XGBoost is the canonical model (FIX 1, all downstream tables) | **FALSE — wrong model attribution** |
| B2 | Best AUC = 0.9536 | `T19` row E1: best AUROC = **0.9515** (XGBoost) | **TRUE (≈)** if reattributed to XGBoost |
| B3 | Best Accuracy = 0.8787 | `T19` row E2: best Accuracy = **0.8755** (XGBoost) | **TRUE (≈)** if reattributed |
| B4 | Best F1 = 0.8605 | `T15` Standard row F1 = 0.8604 | **TRUE** |
| B5 | XGBoost row in Table 4 (Acc 0.8753, AUC 0.9503) | XGBoost: 0.8755 / 0.9515 | **TRUE (≈)** |

## C · Baseline DI / Table 5 (single-point fairness, XGBoost)

| # | Manuscript claim | Notebook evidence | Status |
|---|---|---|---|
| C1 | RACE DI = 0.646 | `T4_best_model_landscape.csv` row RACE → 0.664 | **TRUE (≈)** |
| C2 | SEX DI = 0.731 | `T4` row SEX → 0.763 | **TRUE (≈)** |
| C3 | ETHNICITY DI = 0.802 | `T4` row ETHNICITY → 0.830 | **TRUE (≈)** |
| C4 | AGE_GROUP DI = 0.602 (3-bucket) | `T4` row AGE_GROUP → **0.295** (4-bucket) | **FALSE — paper used 3-bucket binning** |
| C5 | "no model satisfies all 7 metrics for any attribute" | `T20_unanimous_fair_matrix.csv` shows ETH = 6/7 (best); no model = 7/7 | **TRUE** |
| C6 | Fair-of-7 per attr (XGBoost): RACE 5/7, SEX 5/7, ETH 7/7, AGE 3/7 | `T4` Fair_k_over_7 column | **TRUE** |

## D · Intervention claims (§6.10, §8 conclusion)

| # | Manuscript claim | Notebook evidence | Status |
|---|---|---|---|
| D1 | "AFCE achieves DI ≥ 0.80 for **3 of 4** attributes" | `T15.csv` Fair col: RACE 0.864, SEX 0.990, ETH 0.961, AGE 0.809 → **4 of 4** | **FALSE-BETTER (notebook beats paper)** |
| D2 | "AGE = 0.774 remains below threshold" | `T15` AGE Fair = **0.809** (passes) | **FALSE-BETTER** |
| D3 | "RACE post-AFCE = 0.951" | 0.8638 | **FALSE** |
| D4 | "SEX post-AFCE = 0.952" | 0.9902 | **FALSE-BETTER** |
| D5 | "ETHNICITY post-AFCE = 0.950" | 0.9606 | **TRUE (≈)** |
| D6 | "1.2% accuracy trade-off (0.8787 → 0.8675)" | `T15`: 0.8755 → 0.8098 = **6.57pp drop** | **FALSE — paper severely understates cost** |

## E · Lambda reweighing (§6.10.2)

| # | Manuscript claim | Notebook evidence | Status |
|---|---|---|---|
| E1 | "λ = 5.0" | `T13_lambda_sweep.csv` tests 10 values; no single λ chosen | **PARTIAL** |
| E2 | "RACE 0.646 → 0.750 (+16.1%) at λ=5" | `T13` λ=5 row: RACE DI = **0.590** (slightly worse than baseline 0.634 at λ=0) | **FALSE** |
| E3 | "maintains accuracy above 0.86" | `T13` λ=5 Acc = 0.853 | **FALSE (just below)** |

## F · Cross-hospital (§6.6)

| # | Manuscript claim | Notebook evidence | Status |
|---|---|---|---|
| F1 | "DI range 0.18 to 1.12" across hospitals | `T16_per_cluster_xgboost.csv`: Std_DI_RACE min 0.127, max 0.852 | **PARTIAL — actual range 0.13–0.85** |
| F2 | "205 hospitals ≥ 500 patients" | `T16` uses 20 clusters of ~22 hospitals each (441 total / 20 = 22) | **DIFFERENT GROUPING** — paper hospital-level, notebook cluster-level |
| F3 | "Fleiss' κ = 0.22 for DI" | `T11_fleiss_kappa.csv`: DI κ = **0.208** | **TRUE (≈)** |
| F4 | "Fleiss' κ = 0.22 on RACE" | `T11`: per-attr RACE κ = 0.067 | **FALSE** if paper means RACE-decomposition |
| F5 | "Hospital DI median 0.69, IQR [0.54, 0.85]" | needs computation from `T16` | **VERIFY** (numbers plausible) |
| F6 | "Fair at 38% of hospitals, unfair at 62%" | `T16`: 11/20 = 55% pass all-4-DI | **DIFFERENT** — paper uses single-DI-on-RACE; notebook uses all-4-DI |
| F7 | "K = 5 fold DI 0.61–0.78" | `T17_k_sensitivity_real.csv`: only K=10/20/40 reported | **FALSE — K=5 not in notebook** |
| F8 | "K = 20 std-dev 0.09 for RACE" | `T11` K=20 DI κ = 0.208 (consistent with low CV) | **VERIFY** numerical |

## G · Sample size (§6.5)

| # | Manuscript claim | Notebook evidence (`T9_min_sample_size.csv`) | Status |
|---|---|---|---|
| G1 | "DI/SPD: n ≥ 5,000" | DI Sex=5k, Eth=5k; SPD Age=5k | **TRUE** for some attributes |
| G2 | "EOD/EOPP: n ≥ 15,000" | EOPP Age=50k, EOD Sex=100k | **FALSE — notebook says higher** |
| G3 | "PP: n ≥ 50,000" | PP all attrs = 185k | **FALSE — paper underclaims** |
| G4 | "TI: n ≥ 2,000" | TI = 5k–10k | **FALSE — slightly higher** |

## H · Stability claims (§6.4)

| # | Manuscript claim | Notebook evidence | Status |
|---|---|---|---|
| H1 | "33.6% of cells with VFR > 0" | `T19` row B2: **48.8%** (after B=500 update) | **FALSE — notebook now higher** |
| H2 | "Max VFR = 50%" | `T19` row B3: 50.0 | **TRUE** |
| H3 | "VFR ≤ 10% in 259 cells" | `T19` row B4: **252** | **TRUE (≈)** |
| H4 | "VFR = 0% in 226 cells" | `T19` row B5: **172** (after B=500) | **FALSE** — more variance with bigger B |
| H5 | "Bootstrap CV 0.05–0.08 for DI/SPD" | `T10_cross_hospital_cv.csv`: DI CV up to 0.30, SPD up to 0.60 | **FALSE — much higher** |
| H6 | "TI CV = 0.02 (most stable)" | `T10`: TI CV = 0.131 | **FALSE** |

## I · Per-cluster transferability (NEW manuscript section, FIX 7)

| # | Notebook fact (`T16_per_cluster_xgboost.csv`) | Verdict |
|---|---|---|
| I1 | 20 hospital clusters of ≈22 hospitals each | TRUE |
| I2 | Std all-4-DI pass = 0/20 (baseline always fails AGE) | TRUE |
| I3 | Fair all-4-DI pass = **11/20** | TRUE |
| I4 | Std DI-worst → Fair DI-worst improves: **19/20** clusters | TRUE |
| I5 | Accuracy within 5pp of standard: **18/20** clusters | TRUE |

## J · Conclusion claims (§8)

| # | Manuscript claim | Notebook evidence | Status |
|---|---|---|---|
| J1 | "DI ≥ 0.80 for 3 of 4 attrs with <1.3% trade-off" | 4/4 with 6.57pp | **FALSE on both halves** |
| J2 | "107 cells / 37 figures / 13 tables" | **48 cells (29 code) / 5 figures / 19 tables** | **FALSE** |
| J3 | "Stability tests confirm fairness improvements robust" | per-cluster: 11/20 all-4-DI, 19/20 worst-DI improved | **PARTIAL — only 55% transferability** |

---

## Summary recommendation for the .tex edits you'll do yourself

* Replace every "LGB-XGB Blend" with "XGBoost" (canonical model) — affects §6.1, §6.2, §6.4, §6.5, all baseline numbers.
* Update Table 5 to XGBoost numbers (`T4_best_model_landscape.csv`).
* Rewrite §6.10 AFCE paragraph: 4-of-4 DI, 6.6pp accuracy cost, AGE error-rate metrics constrained by impossibility theorem.
* Drop the §6.10.2 λ=5 specific claim — refer to T13 sweep instead.
* Drop or rephrase §6.6 "Hospital DI 38% fair / 62% unfair" — that exact number isn't in the notebook; use the 11/20 cluster-level number from `T16`.
* Drop "K=5" sentence in §6.6 — not in notebook (only K=10/20/40).
* Update §8 numbers: 4/4 DI, 6.6pp trade-off, 11/20 cluster pass, 19/20 worst-DI improved.
* Update reproducibility paragraph: "29 code cells, 5 figures, 19 tables".

Every claim with status **TRUE / TRUE (≈) / FALSE-BETTER** is safe to leave or upgrade. **FALSE / CONTRADICTION** claims must be changed for the paper to match what the supervisor will see in the notebook.
