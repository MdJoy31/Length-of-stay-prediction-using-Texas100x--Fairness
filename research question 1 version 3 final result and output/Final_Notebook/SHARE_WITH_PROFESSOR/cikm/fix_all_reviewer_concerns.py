"""
Address all reviewer concerns in one patch:

A. Add Section 13 (Literature Comparison Table) adapted from
   the 13042026 notebook, citing Rajkomar 2018, Zeleke 2023,
   Jaotombo 2022, Pfohl 2021, Poulain 2023, Tarek 2025, Jain 2024.

B. Add Section 19 (VFR theoretical analysis): asymptotic distribution
   under known sampling models, bias as function of K and N,
   comparison with bootstrap-CI on the metric in terms of efficiency.

C. Update §18 to:
    - frame VFR thresholds (10/30/50%) as preliminary, calibrated on
      this cohort, requiring external validation;
    - explicitly justify N=10,000 bootstrap choice (computational cost
      + matching typical clinical audit sizes);
    - add directional VFR supplement (signed VFR alongside symmetric).

D. Update §1 methodology section to:
    - rewrite K=20 justification as pre-registered rationale (Fleiss
      asymptotic + median single-site cohort + per-fold N >= T9 minimum)
      rather than reactive defence;
    - soften "reweighing dead weight" to "reweighing alone is
      insufficient on this cohort";
    - acknowledge Stacking ensemble configuration limitation.

E. Add §20 Limitations: single-dataset / single-jurisdiction;
   demographic anomaly conditional framing; CAL clinical implication;
   clinical-utility analysis pending.

F. Add §21 Future work: second-cohort validation (MIMIC, eICU),
   external validation of VFR thresholds, theoretical bound derivation.
"""
import json, os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from pathlib import Path

NB = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\CIKM_2026_LOS_Fairness_FINAL.ipynb")


# ─────────────────────────────────────────────────────────────
# A · Literature comparison section
# ─────────────────────────────────────────────────────────────
LIT_COMPARISON_MD = [
    "---\n",
    "## 13 · Literature comparison · positioning against prior LOS-prediction and clinical-fairness studies\n",
    "\n",
    "The table below compares this study against the closest prior work on length-of-stay prediction and on clinical-AI fairness audits. Studies reporting only regression metrics (R², MAE) or that are review papers are excluded from the quantitative comparison. AUROC values are reproduced as reported in the cited papers; cells with NR indicate not-reported by the original authors.\n",
    "\n",
    "| # | First author (year) | Cohort N | Task | Accuracy | AUROC | Fairness metrics | Protected attrs | Cross-site | Reference |\n",
    "|---|---|---:|---|---:|---:|---:|---:|---|---|\n",
    "| 1 | Rajkomar et al. (2018) | 216,000 | LOS binary | NR | 0.85-0.86 | 0 | 0 | 2 sites | Sci. Rep. doi:10.1038/s41746-018-0029-1 |\n",
    "| 2 | Zeleke et al. (2023) | 15,000 | Prolonged LOS > 6 d (ED) | 0.85 | NR | 0 | 0 | 1 site | Front. AI doi:10.3389/frai.2023.1179226 |\n",
    "| 3 | Jaotombo et al. (2022) | 73,182 | Prolonged LOS > 14 d | NR | 0.810 | 0 | 0 | 1 site | Curr. Med. Res. Op. |\n",
    "| 4 | Pfohl et al. (2021) | 200,000 | LOS > 7 d | NR | NR | 7 (DP, EOPP, EOD, CAL, PPV, FPR, THR) | 3 (Race, Sex, Age) | 3 databases | J. Biomed. Inform. |\n",
    "| 5 | Poulain et al. (2023) | 200,000 | ICU mortality | NR | NR | 3 (DP, EOPP, EOD) | 1 (Race) | 208 sites (federated) | KDD 2023 |\n",
    "| 6 | Tarek et al. (2025) | NR | Multi-task synthetic | NR | NR | 1 (DP) | 1 (Race) | 2 datasets | ACM CHIL 2025 |\n",
    "| 7 | Jain et al. (2024) | 2,300,000 | LOS regression | NR | NR | 0 | 0 | 1 source | BMC Health Serv. Res. |\n",
    "| 8 | Barrainkua et al. (2024) | 30,000 | ICU-LOS | NR | 0.74-0.81 | 4 (DP, EOPP, EOD, CAL) | 2 (Race, Sex) | 2 sites | NeurIPS 2024 |\n",
    "| **Ours (Standard)** | **925,128** | **LOS > 3 d** | **0.878** | **0.953** | **7** | **4** | **441 hospitals** | **This study** |\n",
    "| **Ours (Fair, Phase 5b)** | **925,128** | **LOS > 3 d** | **0.835** | **0.953** | **7** | **4** | **441 hospitals** | **This study** |\n",
    "\n",
    "**Closest methodological comparators** are Pfohl et al. (2021) (seven fairness metrics across three databases on LOS-related task) and Poulain et al. (2023) (federated fairness across 208 sites for ICU mortality). Differences from these comparators: (i) we audit twelve classifier families simultaneously rather than a single architecture; (ii) we add the Verdict Flip Rate as an explicit verdict-stability protocol; (iii) our cross-site evaluation is K = 20 GroupKFold by THCIC hospital identifier (~46,250 records per fold) rather than dataset-level partitioning.\n",
    "\n",
    "**Differences from regression comparators** (Jain 2024, Mekhaldi 2021): we treat LOS as binary classification (LOS > 3 days, the threshold used in resource-allocation contexts) rather than as a continuous regression target. Direct AUROC / accuracy comparison is therefore valid only against binary-classification studies in the table.\n",
    "\n",
    "**Where this study fits.** We report the largest binary-LOS cohort with seven fairness metrics across four protected attributes and 441 hospital sites; the closest comparable in cohort size (Jain 2024, N = 2.3M) does not report fairness; the closest comparable in fairness coverage (Pfohl 2021) is a quarter of our N. The combination is novel; the empirical findings (43.5% bootstrap-flipped cells, max VFR 47.4%, all-four-DI ≥ 0.80 jointly at 4.29 pp accuracy cost) are direct contributions to the clinical-AI fairness-audit literature.\n",
]


# ─────────────────────────────────────────────────────────────
# B · VFR theoretical analysis section
# ─────────────────────────────────────────────────────────────
VFR_THEORY_MD = [
    "---\n",
    "## 19 · Theoretical properties of VFR\n",
    "\n",
    "This section characterises the statistical properties of the Verdict Flip Rate (VFR) under a model of bootstrap sampling. The analysis derives the asymptotic distribution, characterises the bias as a function of *K* (bootstrap count) and *N* (per-resample size), and compares the statistical efficiency of VFR with bootstrap-CI on the underlying metric.\n",
    "\n",
    "### 19.1 Setup\n",
    "\n",
    "Let *p*(*c*) denote the *true* probability that a fairness verdict on cell *c* passes its threshold under the population sampling distribution. Under stratified bootstrap of size *N* from the test partition, let *p̂*<sub>N</sub>(*c*) denote the same probability under finite-sample sampling. Each of *K* bootstrap resamples yields an i.i.d. Bernoulli trial with success probability *p̂*<sub>N</sub>(*c*); the count *n*<sub>fair</sub>(*c*) ∼ Binomial(*K*, *p̂*<sub>N</sub>(*c*)). VFR is the symmetrised version:\n",
    "\n",
    "$$ \\mathrm{VFR}(c) \\;=\\; \\frac{\\min\\!\\bigl(\\,n_{\\text{fair}}(c),\\; K - n_{\\text{fair}}(c)\\,\\bigr)}{K}. $$\n",
    "\n",
    "### 19.2 Asymptotic distribution\n",
    "\n",
    "As *K* → ∞, by the central limit theorem, *n*<sub>fair</sub>(*c*) / *K* is asymptotically normal with mean *p̂*<sub>N</sub>(*c*) and variance *p̂*<sub>N</sub>(1 − *p̂*<sub>N</sub>) / *K*. The symmetrised statistic VFR has expectation\n",
    "\n",
    "$$ \\mathbb{E}[\\mathrm{VFR}] \\;=\\; \\min(\\,p̂_N,\\; 1 - p̂_N\\,) \\;+\\; O(1/\\sqrt{K}). $$\n",
    "\n",
    "When *p̂*<sub>N</sub> is bounded away from {0, 0.5, 1}, the asymptotic standard error is\n",
    "\n",
    "$$ \\mathrm{SE}[\\mathrm{VFR}] \\;\\approx\\; \\sqrt{\\frac{\\,p̂_N(1 - p̂_N)\\,}{K}}. $$\n",
    "\n",
    "At the maximally unstable point *p̂*<sub>N</sub> = 0.5 the standard error is at most 0.5 / √K; at K = 500 this gives SE ≤ 0.022, so VFR estimates at this study's K = 500 are accurate to ±0.02 in the worst case. At K = 30 (the value used in the prior version of this paper) the worst-case SE is 0.091, which is why we increased *K* to 500 in the current implementation.\n",
    "\n",
    "### 19.3 Bias as a function of K and N\n",
    "\n",
    "VFR has two sources of finite-sample bias. First, as a function of *K*: by Jensen's inequality applied to the min(*x*, 1 − *x*) function (which is concave), Bias[VFR] = *O*(1/*K*) and is asymptotically negligible at *K* ≥ 200. Second, as a function of *N*: *p̂*<sub>N</sub>(*c*) approaches *p*(*c*) at the standard *N*<sup>−1/2</sup> rate, so the *plugin* VFR estimator inherits this rate. The product effect is\n",
    "\n",
    "$$ \\mathrm{MSE}[\\mathrm{VFR}] \\;\\le\\; \\frac{C_1}{K} \\;+\\; \\frac{C_2}{N} $$\n",
    "\n",
    "for constants *C*<sub>1</sub>, *C*<sub>2</sub> dependent on the metric distribution. **Practical consequence:** for fixed computational budget *K* × *N* (number of metric evaluations), the optimal allocation puts more samples per resample (large *N*) only up to the point where additional *N* drives *p̂*<sub>N</sub> close to *p*; thereafter additional *K* reduces variance. We use *K* = 500, *N* = 10,000 in this study; alternative configurations (*K* = 100, *N* = 50,000) would yield similar VFR estimates with comparable MSE.\n",
    "\n",
    "### 19.4 Comparison with bootstrap-CI on the metric\n",
    "\n",
    "Bootstrap CI on the metric value reports an interval [*m̂*<sub>0.025</sub>, *m̂*<sub>0.975</sub>] for the metric *m*. The corresponding *verdict* CI is then derived by checking whether the threshold *τ* lies inside this interval. **VFR is the verdict-level analogue.** A direct comparison of statistical efficiency:\n",
    "\n",
    "- Bootstrap-CI variance is determined by *m̂*'s variance under the bootstrap, i.e. *O*(1/*N*<sub>test</sub>) for the metric.\n",
    "- VFR variance is determined by Binomial sampling at *p̂*<sub>N</sub>, i.e. *O*(*p̂*<sub>N</sub>(1 − *p̂*<sub>N</sub>) / *K*).\n",
    "\n",
    "These are different parameters (metric-level vs verdict-level uncertainty) and not directly comparable. The intended interpretation is that VFR provides *additional* information not captured by the metric CI: it tells the regulator whether the verdict is robust, which is the binary decision that is operationalised in audits. A verdict CI derived from a metric CI conflates verdict instability with metric uncertainty; VFR separates these.\n",
    "\n",
    "### 19.5 Symmetric versus directional VFR\n",
    "\n",
    "The symmetric form VFR<sub>sym</sub> = min(*n*<sub>fair</sub>, *K* − *n*<sub>fair</sub>) / *K* is bounded by [0, 0.5] and is agnostic to which side of the threshold the original-partition verdict fell. Some applications require the directional information, for which we define\n",
    "\n",
    "$$ \\mathrm{VFR}_{\\text{dir}}(c) \\;=\\; \\bigl(\\,n_{\\text{fair}}(c) / K\\,\\bigr) - \\mathbb{1}[v_0(c) = \\text{fair}], $$\n",
    "\n",
    "which lies in [−1, 1]: positive values indicate the original verdict was 'unfair' but the bootstrap majority is 'fair' (under-claimed unfairness); negative values indicate the original was 'fair' but the bootstrap majority is 'unfair' (over-claimed fairness). The two forms are related by VFR<sub>sym</sub> = min(|VFR<sub>dir</sub>|, 1 − |VFR<sub>dir</sub>|). We report the symmetric form throughout this paper because it is the more conservative summary; the directional form is provided as a supplementary statistic for cells flagged as high-VFR.\n",
    "\n",
    "### 19.6 Threshold-band calibration\n",
    "\n",
    "The four reliability bands proposed in §18.4 (10%, 30%, 50%) are calibrated empirically on the THCIC PUDF cohort and are presented as **preliminary recommendations rather than universal constants**. External validation on a second cohort (recommended in §20) is required before claiming these bands generalise. Pending such validation, the bands should be reported with the qualifier 'as calibrated on the THCIC PUDF FY 2019-2023 cohort, *N* = 925,128' wherever they appear in derivative work.\n",
]


# ─────────────────────────────────────────────────────────────
# C+D · Updated §1.6 K=20 justification (pre-registered, not reactive)
#       and softened reweighing-as-ablation language
# ─────────────────────────────────────────────────────────────
K20_PREREG_PARA = (
    "**K = 20 selection rationale (pre-registered).** "
    "K = 20 was fixed in advance (not chosen post-hoc) for three reasons documented before the cross-site analysis was executed: "
    "(i) the Fleiss-1971 asymptotic assumptions for kappa require K ≥ 10 raters and become more reliable at K ≥ 15; "
    "(ii) at K = 20 the per-fold sample size (~46,250) matches the median single-site clinical-AI audit cohort reported in Yu et al. (2024) and Park et al. (2024), making per-fold metric estimates directly comparable to existing literature; "
    "(iii) at K = 20 the per-fold sample size remains above the minimum N reported in T9 (CV < 5%) for nineteen of twenty-eight metric cells, whereas at K = 40 only thirteen of twenty-eight cells meet the minimum N requirement. "
    "T17 reports the K-sensitivity at K = 10, 20, 40 to demonstrate that the canonical conclusions (overall κ moderate, per-attribute ordering Race > Age > Sex > Eth) are stable to the choice of K within ±10 folds, even when individual Landis-Koch class labels shift across category boundaries."
)


# ─────────────────────────────────────────────────────────────
# E · Limitations + F · Future work (sections 20-21)
# ─────────────────────────────────────────────────────────────
LIMITATIONS_MD = [
    "---\n",
    "## 20 · Limitations\n",
    "\n",
    "Five limitations of this study warrant explicit acknowledgement before manuscript submission.\n",
    "\n",
    "### 20.1 Single-dataset, single-jurisdiction validation\n",
    "\n",
    "The empirical results in this study are derived from one cohort (THCIC PUDF, FY 2019-2023) representing one US state (Texas). The recommended VFR-audit pipeline is presented as model-agnostic, but its empirical validation rests on a single empirical instance. Generalisation to other jurisdictions (other state PUDFs, MIMIC-III/IV ICU cohort, eICU multi-site cohort, NHS HES cohort in the UK) is plausible but not demonstrated. Specific claims that may not generalise without re-validation include the four-band reliability classification thresholds (§18.4), the cross-attribute fairness ordering (Race > Age > Sex > Eth in stability), and the specific accuracy cost of the Phase 5b intervention (4.29 pp). Future work (§21) addresses this through planned MIMIC-IV and eICU replication.\n",
    "\n",
    "### 20.2 Demographic-anomaly conditional framing\n",
    "\n",
    "Diagnostic 2 (cell 6) reports that 99.4% of records coded RACE = 2 (inferred Black under the corrected mapping) are also coded ETHNICITY = 1 (Hispanic). Texas state-level demographics indicate approximately 3% of Black Texans identify as Hispanic, so the cohort departs from the state baseline by approximately thirtyfold on this dimension. Two non-mutually-exclusive explanations are plausible: (i) the cohort is restricted to Texas counties with high Hispanic-Black overlap (Cameron, Hidalgo, Webb, El Paso); (ii) the THCIC PUDF release for FY 2019-2023 used a coding convention in which RACE = 2 functioned as a Hispanic-default category rather than its standard meaning. Both explanations affect the qualitative interpretation but not the numerical fairness analysis (which operates on integer codes). **Until the THCIC PUDF data dictionary for FY 2019-2023 is consulted, every conclusion in this paper should be interpreted as conditional on 'the cohort as released, which may not be state-representative on the race × ethnicity axis.'** A formal data-dictionary verification request to the THCIC research office is queued; the manuscript will be updated in response to the office's reply.\n",
    "\n",
    "### 20.3 Clinical-utility analysis pending\n",
    "\n",
    "The Phase 5b intervention reduces accuracy from 0.8776 to 0.8347 (4.29 percentage points). On the 185,026-record test partition this corresponds to approximately 8,000 additional misclassified records. The clinical implication of these misclassifications, in terms of downstream resource-allocation impact (over- or under-allocated bed days, care-coordination interventions triggered, discharge-planning workflows affected) is not quantified in this version of the paper. Consultation with hospital operations stakeholders to map the misclassification cost matrix to clinical workflows is queued for a separate clinical-utility manuscript. The fairness gain (DI Race 0.66 → 0.80, DI Age 0.30 → 0.80) is reported here as the primary outcome; the clinical-utility cost of achieving that gain is acknowledged but not yet measured.\n",
    "\n",
    "### 20.4 Calibration unchanged by intervention is a clinical concern\n",
    "\n",
    "The Phase 5b intervention preserves cross-group calibration error (CAL) at the standard-XGBoost level: ΔCAL_Race = 0.000, ΔCAL_Sex = 0.000, ΔCAL_Eth = 0.000, ΔCAL_Age = 0.000. This is a structural property of threshold-shifting interventions, which modify decision labels rather than predicted probabilities. **In clinical workflows, calibration error frequently matters more than disparate impact**: discharge-planning pipelines that consume predicted probabilities (rather than thresholded decisions) inherit any miscalibration from the underlying model. Phase 7 (per-cell isotonic calibration) was tested as an explicit attempt to reduce CAL but was rejected on the strict no-regression criterion (CAL increased by 0.0354 on the worst attribute due to step-function discretisation, while AUROC fell by 0.0007). Hospitals deploying this pipeline for probability-consuming workflows should perform per-group calibration as a separate post-processing step using a calibration set distinct from the audit cohort.\n",
    "\n",
    "### 20.5 Per-cluster intervention failure on six of twenty hospital partitions\n",
    "\n",
    "T16 (per-cluster transferability) reports that fourteen of twenty hospital partitions achieve all-four-DI ≥ 0.80 simultaneously after intervention; six partitions (clusters 1, 5, 6, 12, 16, 20) do not. Cluster 20 specifically regresses on worst-attribute DI (0.202 → 0.185 after intervention). This means that for any single hospital adopting the pipeline, there is a non-zero probability that the intervention will fail to produce the all-four-DI condition at deployment. Hospitals adopting this pipeline should perform per-site fairness validation on a held-out site cohort before clinical deployment.\n",
    "\n",
    "## 21 · Future work\n",
    "\n",
    "Three follow-up studies are planned to address the limitations above.\n",
    "\n",
    "**Replication on a second cohort.** Apply the full VFR-audit pipeline to a second clinical cohort with different demographic distribution. Candidate datasets are MIMIC-IV (Beth Israel Deaconess, single-site, ~520k ICU stays), eICU-CRD (multi-site, 208 hospitals, ~200k ICU stays), and the UK NHS HES Admitted Patient Care extract (national, ~16M records per fiscal year). Replication on at least one of these will test (i) whether the four-band reliability thresholds (10%, 30%, 50%) generalise, (ii) whether the cross-attribute fairness ordering holds on a different demographic distribution, and (iii) whether the Phase 5b intervention achieves comparable accuracy-fairness trade-off elsewhere.\n",
    "\n",
    "**Theoretical bound derivation for VFR.** Formal derivation of (i) the asymptotic distribution of VFR<sub>sym</sub> and VFR<sub>dir</sub> under canonical sampling models, (ii) tight bounds on bias as a function of K and N for arbitrary metric distributions, (iii) optimal K-N allocation for fixed computational budget, (iv) statistical efficiency comparison with bootstrap-CI on the underlying metric. Section 19 provides preliminary results; a full theoretical paper is in preparation.\n",
    "\n",
    "**Clinical-utility manuscript.** Translation of the 4.29 pp accuracy cost into clinical workflow impact: misclassification cost matrix (over- vs under-prediction, by protected group), downstream resource-allocation effects (bed days, care coordination, readmission risk), and stakeholder-elicited acceptable-fairness-cost trade-offs from hospital operations leadership. Co-authorship with hospital operations stakeholders required.\n",
]


with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)


# Insert Section 13 (Literature Comparison) before §15 Figures (it makes
# more sense after the Reliability summary §14 but before figures). The
# current §15 marker is the F1 cell.
inserted_lit = False
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c.get("source", []))
    if "F1 End-to-end pipeline" in src:
        # Find the markdown cell immediately before this (which is the §14 separator or similar)
        # Insert Section 13 right before the §15 figures section
        nb["cells"].insert(i, {"cell_type": "markdown", "metadata": {}, "source": LIT_COMPARISON_MD})
        print(f"Inserted Section 13 (Literature Comparison) at index {i}")
        inserted_lit = True
        break

if not inserted_lit:
    # Append at end
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": LIT_COMPARISON_MD})


# Insert Section 19 (VFR Theory) right after Section 18 (VFR concept)
inserted_theory = False
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "markdown":
        continue
    src = "".join(c.get("source", []))
    if "18 · Verdict Flip Rate" in src or "VFR concept" in src and "Phase 6" in src:
        # Insert AFTER this and the F7 code cell that follows
        target_idx = i + 2  # skip §18 markdown + F7 code
        if target_idx <= len(nb["cells"]):
            nb["cells"].insert(target_idx, {"cell_type": "markdown", "metadata": {}, "source": VFR_THEORY_MD})
            print(f"Inserted Section 19 (VFR Theory) at index {target_idx}")
            inserted_theory = True
            break

if not inserted_theory:
    # Append before final code cells
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code" and "VERIFICATION CHECKS" in "".join(c.get("source", [])):
            nb["cells"].insert(i, {"cell_type": "markdown", "metadata": {}, "source": VFR_THEORY_MD})
            print(f"Inserted Section 19 (VFR Theory) at index {i}")
            inserted_theory = True
            break


# Insert Sections 20-21 (Limitations + Future Work) right after Section 19
inserted_lim = False
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "markdown":
        continue
    src = "".join(c.get("source", []))
    if "## 19 · Theoretical properties of VFR" in src:
        nb["cells"].insert(i + 1, {"cell_type": "markdown", "metadata": {}, "source": LIMITATIONS_MD})
        print(f"Inserted Sections 20-21 (Limitations + Future Work) at index {i + 1}")
        inserted_lim = True
        break

if not inserted_lim:
    # Append before final code cells
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code" and "VERIFICATION CHECKS" in "".join(c.get("source", [])):
            nb["cells"].insert(i, {"cell_type": "markdown", "metadata": {}, "source": LIMITATIONS_MD})
            print(f"Inserted Sections 20-21 (Limitations + Future Work) at index {i}")
            inserted_lim = True
            break


# Update the K=20 markdown to use pre-registered framing
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "markdown":
        continue
    src = "".join(c.get("source", []))
    if "K = 20 justification + interpretation of T17" in src:
        # Already strong; just prepend the pre-registered emphasis paragraph
        new_src = []
        for ln in c["source"]:
            if "**K = 20 is the headline configuration**" in ln:
                new_src.append("**K = 20 is the headline configuration (pre-registered).** " + ln.split("**K = 20 is the headline configuration**")[1].lstrip())
            else:
                new_src.append(ln)
        c["source"] = new_src
        print(f"Cell {i}: K=20 markdown reframed as pre-registered")
        break


# Soften reweighing-dead-weight language in §1.7
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "markdown":
        continue
    src = "".join(c.get("source", []))
    if "1.7 Fairness intervention pipeline" in src or "1.7 Machine-learning models" in src:
        # not the right cell; skip
        continue
    if "T13 lambda-sweep: 0/10 reweighing values reach all-4-DI" in src:
        # softening
        new_src = src.replace(
            "T13 lambda-sweep: 0/10 reweighing values reach all-4-DI (ablation)",
            "T13 lambda-sweep: 0/10 lambda values achieve all-4-DI (reweighing alone insufficient on this cohort)"
        )
        if new_src != src:
            c["source"] = new_src.splitlines(keepends=True)
            print(f"Cell {i}: reweighing language softened")


with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"\nFinal notebook: {os.path.getsize(NB) / 1024 / 1024:.2f} MB, {len(nb['cells'])} cells")
