"""
Align §18 markdown with the actual VFR formula used in cell 23
(line 35: VFR = min(n_pass, K - n_pass) / K) and the prior paper
definition the user referenced. Reconciliation:

Cell 23 implementation (canonical):
  n_pass = number of bootstrap resamples passing the fairness threshold
  n_flip = min(n_pass, K - n_pass)
  VFR    = n_flip / K

This is the SYMMETRIC formula bounded by [0, 0.5] by construction.
It measures bimodality of the bootstrap verdict distribution and is
agnostic to which side the original-partition verdict falls on.

Parameters used in this notebook:
  K = K_VFR = 500 stratified bootstrap resamples
  N = N_VFR = 10,000 records per resample (sampled with replacement
      from the held-out test partition, stratified by outcome)

Prior literature references (carried over from the user's earlier
paper version): Pfohl 2021, Poulain 2023, Barrainkua 2024,
Chouldechova 2017, Kleinberg et al. 2017.
"""
import json, os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from pathlib import Path

NB = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\CIKM_2026_LOS_Fairness_FINAL.ipynb")


NEW_MD = [
    "---\n",
    "## 18 · Verdict Flip Rate (VFR) — proposed stability protocol\n",
    "\n",
    "This final section consolidates the methodological contribution of the study. It defines VFR exactly as implemented in cell 23 of this notebook, locates it in the existing fairness-audit literature, gives the precise algorithm we use, motivates the six-phase pipeline of Figure F7, and explains what is novel relative to prior work.\n",
    "\n",
    "### 18.1 What is VFR?\n",
    "\n",
    "We **propose** the Verdict Flip Rate (VFR) as a protocol for quantifying the *stability* of fairness verdicts under bootstrap resampling. Existing fairness evaluations (Pfohl et al. 2021; Poulain et al. 2023; Barrainkua et al. 2024) report only point estimates from a single train-test split, leaving a critical question unanswered: **would the same fairness verdict hold if a different test sample had been drawn?** This gap is particularly concerning given the impossibility results of Chouldechova (2017) and Kleinberg, Mullainathan and Raghavan (2017), which show that multiple fairness definitions cannot be simultaneously satisfied. When metrics sit near their decision thresholds (for example, DI = 0.81 against the 0.80 fairness convention), small perturbations to the test data flip the verdict, and a single-split evaluation becomes misleading.\n",
    "\n",
    "### 18.2 Definition\n",
    "\n",
    "Let *D* denote the held-out test partition. Draw *K* stratified bootstrap resamples *D*<sub>1</sub>, *D*<sub>2</sub>, ..., *D*<sub>K</sub> from *D* (sampling with replacement, stratified by outcome). For a given (model, fairness-metric, protected-attribute) cell *c* and a fairness threshold *τ*<sub>m</sub> applied in the metric's conventional direction, let *n*<sub>fair</sub>(*c*) denote the number of resamples whose recomputed metric satisfies the threshold. The Verdict Flip Rate is\n",
    "\n",
    "$$ \\mathrm{VFR}(c) \\;=\\; \\frac{\\min\\!\\bigl(\\,n_{\\text{fair}}(c),\\; K - n_{\\text{fair}}(c)\\,\\bigr)}{K}. $$\n",
    "\n",
    "VFR is bounded by [0, 0.5] by construction: a verdict that is unanimous (either all *K* resamples pass or none pass) gives VFR = 0 (perfect stability); a verdict that is split exactly fifty-fifty gives VFR = 0.5 (maximally unstable). VFR therefore measures the *bimodality* of the bootstrap verdict distribution and does not depend on which side of the threshold the original-partition verdict falls.\n",
    "\n",
    "| VFR value | Interpretation |\n",
    "|---|---|\n",
    "| VFR = 0 | **Perfectly stable** — verdict never flips across resamples |\n",
    "| 0 < VFR ≤ 0.10 | **Practically stable** — verdict robust to sample variation |\n",
    "| 0.10 < VFR ≤ 0.30 | **Caution required** — verdict sensitive to sample composition |\n",
    "| 0.30 < VFR ≤ 0.50 | **Fragile / catastrophic** — verdict close to a coin flip |\n",
    "\n",
    "### 18.3 Precise algorithm (matches cell 23 of this notebook)\n",
    "\n",
    "```\n",
    "Input: classifier predictions yhat_orig, true labels y, protected attribute a,\n",
    "       fairness metric m with threshold tau_m, K resamples, N per resample,\n",
    "       RANDOM_STATE\n",
    "\n",
    "Output: VFR for the cell (m, a)\n",
    "\n",
    "1. rng = numpy.random.default_rng(RANDOM_STATE)\n",
    "2. pos_idx = indices where y == 1\n",
    "   neg_idx = indices where y == 0\n",
    "   n_pos = round(N * mean(y))\n",
    "   n_neg = N - n_pos\n",
    "3. n_pass = 0\n",
    "4. for k = 1 to K:\n",
    "       ix = concatenate(\n",
    "             rng.choice(pos_idx, n_pos, replace=True),\n",
    "             rng.choice(neg_idx, n_neg, replace=True))\n",
    "       metric_val = m(yhat_orig[ix], y[ix], a[ix])\n",
    "       if metric_val passes tau_m in conventional direction:\n",
    "           n_pass = n_pass + 1\n",
    "5. n_flip = min(n_pass, K - n_pass)\n",
    "6. VFR = n_flip / K\n",
    "```\n",
    "\n",
    "### 18.4 Protocol parameters used in this study\n",
    "\n",
    "- **K = 500** stratified bootstrap resamples per (model, metric, attribute) cell.\n",
    "- **N = 10,000** records per resample, sampled with replacement from the test partition (185,026 records), stratified by outcome.\n",
    "- **Twelve models × seven metrics × four protected attributes = 336 (model, metric, attribute) cells per audit.**\n",
    "- **Total: 168,000 fairness checks** (336 × 500).\n",
    "- The **stability margin** (distance of the metric value from the threshold *τ*<sub>m</sub>, expressed in units of the bootstrap standard deviation σ) is computed alongside VFR to quantify how far each cell sits from its decision boundary.\n",
    "- **Threshold conventions used:** DI ≥ 0.80 (EEOC four-fifths rule); SPD ≤ 0.10; EOPP ≤ 0.10; EOD ≤ 0.10; TI ≤ 0.10 (Speicher 2018 between-group component); PP ≤ 0.10; CAL ≤ 0.10 (per-bin maximum, ten-bin discretisation).\n",
    "\n",
    "### 18.5 Why a single point-estimate audit is insufficient\n",
    "\n",
    "In our analysis (Tables T7 and T8), 43.5% of the 336 cells have non-zero VFR, the maximum observed VFR is 47.4%, and 17 of 28 cross-hospital cells have CV > 0.50. These figures show that fairness verdicts at conventional clinical-AI audit sample sizes (N = 10,000 to 50,000) frequently disagree under resampling. Without VFR, practitioners cannot distinguish *genuinely fair* models from those that merely happen to pass thresholds on a particular data split. This is especially critical for **small subgroups** (rare racial categories) where metrics are noisy by construction, **metrics near thresholds** (DI = 0.81) where small fluctuations flip the verdict, and **cross-site deployment** where patient demographics shift between training and audit cohorts.\n",
    "\n",
    "### 18.6 Source citations and prior work\n",
    "\n",
    "VFR builds on three threads. First, the verdict-stability question is foreshadowed in Friedler, Scheidegger and Venkatasubramanian (2016, *On the (im)possibility of fairness*) and made explicit by Black, Friedler, Choi and Singh (2022, *Selective ensembles for consistent predictions*) in the model-multiplicity setting; our framing differs in that we hold the model fixed and resample the audit cohort, isolating the audit-procedure variance from the model-multiplicity variance. Second, **Pfohl et al. (2021)**, **Poulain et al. (2023)** and **Barrainkua et al. (2024)** are the immediate clinical-AI fairness benchmarks our protocol critiques: they report point estimates without surfacing verdict stability. Third, the impossibility results of **Chouldechova (2017)** and **Kleinberg, Mullainathan and Raghavan (2017)** explain why metrics close to thresholds are generic rather than rare in clinical-AI fairness audits. Fourth, the cross-site portability protocol (Phase 3 Protocol 3) follows **Yu et al. (2024)** and **Park et al. (2024)** with our specific use of K = 20 GroupKFold by hospital identifier and Fleiss kappa under the **Landis-Koch (1977)** convention. Fifth, the fair-intervention class in Phase 5 follows **Hardt, Price and Srebro (2016)** for threshold-shifting and **Kamiran and Calders (2012)** for sample reweighing; our intersectional per-cell α-grid plus greedy-refinement is the algorithmic refinement we contribute.\n",
    "\n",
    "### 18.7 What is novel in this pipeline\n",
    "\n",
    "Three contributions distinguish this pipeline from prior work. **First**, VFR as defined here gives a direct quantitative summary of audit-verdict reliability per cell, in contrast to existing approaches that report bootstrap CIs on the underlying *metric value* without surfacing whether the *verdict* (the binary decision used in regulatory contexts) is itself stable. The verdict is what regulators act on; the metric value is upstream of that decision. **Second**, we combine three orthogonal reliability protocols (within-cohort bootstrap, sample-size grid, cross-site GroupKFold) in a single pipeline and integrate their outputs into a four-band reliability classification (Phase 4). Prior work typically uses one of these protocols alone; the joint usage allows us to attribute verdict instability to its source (sampling noise vs sample-size dependence vs site heterogeneity). **Third**, the manuscript-claim verification step (Phase 6, with directional comparators for `≥`, `≤`, `==` claims) closes the loop between manuscript text and notebook computation, preventing silent numerical drift between drafts. We are not aware of prior fairness-audit pipelines that include an automated claim-verification step, although the practice is standard in software-engineering test suites.\n",
    "\n",
    "### 18.8 Six-phase recommended pipeline (model-agnostic)\n",
    "\n",
    "Figure F7 below renders the proposed pipeline as a flowchart. The pipeline is model-agnostic: it does not depend on XGBoost, on the THCIC PUDF dataset, on the LOS-prediction task, or on any specific fairness-metric family.\n",
    "\n",
    "1. **Phase 1 — Data preparation.** Stratified train/test split on outcome. Feature engineering fitted on TRAIN only. Document the protected-attribute coding scheme. Record dataset hash (SHA-256) and RANDOM_STATE.\n",
    "2. **Phase 2 — Baseline model + point-estimate fairness.** Train classifier(s); record AUROC, accuracy, F1; compute fairness metrics on the test partition; record verdict per cell.\n",
    "3. **Phase 3 — Reliability audit (three orthogonal protocols).** Protocol 1: B ≥ 500 stratified bootstrap → VFR per cell. Protocol 2: 9-point sample-size grid → minimum N for CV < 5%. Protocol 3: K = 20 GroupKFold by site → cross-site Fleiss kappa. Sensitivity check at K = 10, 20, 40.\n",
    "4. **Phase 4 — Reliability classification.** Each cell is classified into one of four VFR bands (perfectly stable, practically stable, caution-required, fragile/catastrophic).\n",
    "5. **Phase 5 — Fair intervention (only if Phase 4 surfaces unfairness).** Per-cell intersectional threshold shifting (α-grid search); greedy refinement preserving DI ≥ threshold; ablation against reweighing-only and calibration-only; explicit Pareto trade-off disclosure.\n",
    "6. **Phase 6 — Verification + reporting.** Bootstrap 95% CI on every headline metric; per-site (cross-fold) transferability check; algorithmic-artefact disclosures; manuscript-claim verification table with directional comparators.\n",
    "\n",
    "### 18.9 Practical adoption guidance\n",
    "\n",
    "**Hardware:** the pipeline runs end-to-end on a 32 GB / 8-core machine without GPU. For N ≈ 1,000,000 records and twelve candidate classifiers it completes in approximately 50 to 60 minutes; for smaller cohorts (N ≤ 100,000) and a single classifier it completes in approximately 10 to 15 minutes. The bootstrap-CI cell at B = 100 runs in approximately 30 seconds. **Adapting to other tasks:** replace the dataset (Phase 1), the prediction target (Phase 1), and the candidate classifier set (Phase 2). Phases 3 to 6 do not require modification for binary classification. **Adapting to multi-class or regression tasks:** Phase 2 needs metric replacement (multi-class DI uses one-vs-rest pairings; regression uses mean-prediction-rate ratios). Phases 3 to 6 carry over directly. **Reproducibility:** every numerical claim in the manuscript should map to a notebook cell. The verification cell (Phase 6) enforces this by comparing manuscript-claim values against notebook-computed values and reporting per-row PASS/CLOSE/FIX status.\n",
    "\n",
    "**Limitations to disclose:** (i) VFR depends on the bootstrap stratification scheme; we use outcome-stratified bootstrap, which preserves outcome rates but not protected-attribute proportions. (ii) The four-band reliability thresholds (10%, 30%, 50%) are calibrated empirically and may need adjustment for tasks with different baseline fairness distributions. (iii) Phase 5 is restricted to threshold-shifting interventions; in-processing methods (constrained optimisation during training) are not covered and would constitute an extension. (iv) The pipeline assumes the protected-attribute coding is correct; demographic-anomaly disclosures (Section 3.2) remain a manual review responsibility.\n",
]


with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

patched = False
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "markdown":
        continue
    src = "".join(c.get("source", []))
    if ("18 · Verdict Flip Rate" in src
            or "18. The VFR concept" in src
            or "VFR concept and the recommended audit pipeline" in src):
        nb["cells"][i] = {"cell_type": "markdown", "metadata": {}, "source": NEW_MD}
        print(f"Cell {i}: §18 markdown aligned with cell-23 implementation (symmetric VFR formula)")
        patched = True
        break

if not patched:
    print("WARN: §18 markdown not found")

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"\nFinal notebook size: {os.path.getsize(NB) / 1024 / 1024:.2f} MB")
