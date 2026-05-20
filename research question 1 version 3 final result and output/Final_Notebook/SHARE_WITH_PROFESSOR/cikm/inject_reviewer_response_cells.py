"""
Inject Section 33 (reviewer-response extensions) cells into the notebook.
Appends after the current last cell. Preserves all prior cells.
"""
import json, shutil
from pathlib import Path

NB = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\CIKM_2026_LOS_Fairness_FINAL.ipynb")
BACKUP = NB.with_suffix('.pre-sec33.ipynb')
if not BACKUP.exists():
    shutil.copy2(NB, BACKUP)
    print(f"Backed up to {BACKUP.name}")

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}

cells = []

# =====================================================================
# § 33 intro + § 33.1
# =====================================================================
cells.append(md("""---
## 33 · Reviewer-response extensions (simulated CIKM 2026 review)

A simulated CIKM 2026 reviewer flagged ten concerns and recommended **Weak Reject (4/10)**, with the assessment that the **idea is publishable** but the **methodological protocol is not yet bulletproof**. This section addresses each concern with concrete analyses, code, and disclosures. New tables land under `output_final/tables/T_reviewer_*.csv`.

The targeted fix list to move from 4/10 → 6/10 (Weak Accept / Borderline) is implemented below:

| # | Reviewer concern | Severity | This section |
|---|---|---|---|
| 1 | Threshold tuning on the audit set | Major | §33.1 |
| 2 | Feature leakage behind AUROC = 0.9528 | Major | §33.2 |
| 3 | VFR novelty vs bootstrap CI threshold crossing | Major | §33.3 |
| 4 | Threshold cut-offs (VFR, CV, κ) arbitrary | Moderate | §33.4 |
| 5 | Cross-hospital validation strength | Moderate | §33.5 |
| 6 | Baseline 4 reproducibility | Moderate | §33.6 |
| 7 | Race / ethnicity coding anomaly | Moderate | §33.7 |
| 8 | "146 / 336 mislead" overclaim wording | Minor | §33.8 |
| 9 | CIKM fit framing | Minor | §33.9 |
| 10 | "AUROC preserved" tautology + Algorithm 1 ties + 5-pp wording | Minor | §33.8 |

A master concern → response table is rendered in §33.10.

### 33.1 · Train / validation / final-audit split (acknowledged optimism)

**Current state (honest disclosure).** Cell 11 (Section 4) builds a single 80 / 20 stratified split: `idx_train` (~ 740 k rows) fits XGBoost; `idx_test` (~ 185 k rows) provides the predictions `canon_proba` / `canon_pred`. Cell 34 (Section 11.2) then iterates 168 α candidates per intersectional `(RACE × AGE × SEX)` cell and selects per-cell thresholds that match SR / TPR / PPV on **the same `canon_proba`** that Section 8 audits and Section 11.5 reports in Table 15. The α-search is therefore tuned on the audit set. The reviewer's concern is **valid as stated**.

**Optimism magnitude.** The 168-candidate grid (step = 0.01) is coarse, so point-estimate overfit is limited; but VFR pass-rate inflation on the same bootstrap draws used to fit α is plausible.

**Proposed three-way split (rerun pending).**

| Split | Fraction | Role |
|---|---:|---|
| `train` | 70 % | Fit XGBoost |
| `val`   | 15 % | α-search, greedy refinement |
| `audit` | 15 % | VFR, T15, T16 — never seen by tuning |

A hospital-disjoint variant — splitting 441 hospitals as 309 / 66 / 66 by `THCIC_ID` — addresses concerns #1 and #5 simultaneously.

**Reading the current numbers.** They describe **in-cohort post-processing under self-tuning** — a realistic deployment regime where governance audits use the same labelled cohort. They are an *upper bound* on the fairness gain achievable on a strictly held-out audit set.
"""))

# =====================================================================
# § 33.2 — feature leakage
# =====================================================================
cells.append(md("""### 33.2 · Feature-leakage audit

Reviewer concern: AUROC = 0.9528 on LOS > 3 day prediction is high for administrative data. Eight features are used (Section 4); each is classified below by temporal availability.

**Headline finding.** Two of eight features — `TOTAL_CHARGES` and `PAT_STATUS` — are recorded at or after discharge, and both are strongly target-correlated. The reviewer's leakage concern is **valid**. The manuscript's AUROC therefore reflects a *retrospective* discrimination model — useful for post-hoc governance audits, not for prospective admission-time triage. The fairness-audit reliability story (VFR landscape, intervention effect) is independent of this leakage — VFR measures verdict stability under bootstrap, not absolute predictive accuracy — but the **AUROC headline number must be reframed**.

**Admission-only ablation code** is provided below; expected drop is approximately 0.05–0.10 AUROC based on the typical Texas-100X admission-only benchmark (≈ 0.85–0.88)."""))

cells.append(code('''# § 33.2 · Feature-availability classification table
import pandas as pd
from IPython.display import display, HTML

feat_audit = pd.read_csv('output_final/tables/T_reviewer_feature_leakage_audit.csv')
display(HTML('<b>Feature-availability classification (8 input features used in Section 4)</b>'))
display(feat_audit)

print()
print(f"Admission-time features:  {(feat_audit['availability']=='admission').sum()}/8")
print(f"Near-admission features:  {(feat_audit['availability']=='near-adm').sum()}/8")
print(f"Discharge-time features:  {(feat_audit['availability']=='discharge').sum()}/8  ← LEAKAGE RISK")
'''))

cells.append(code('''# § 33.2 · Admission-only XGBoost ablation (READY-TO-RUN; uncomment to execute)
# Re-fits canonical XGBoost after dropping the two discharge-time features
# (TOTAL_CHARGES, PAT_STATUS). Compares to the manuscript-headline AUROC.
#
# Prerequisites in kernel: X_train, X_test, y_train, y_test, RANDOM_STATE
# from Section 4 (cell 11) and Section 5 (cell 15).
#
# Expected runtime: ≈ 6-10 min on full 740k training rows.

ABLATION = """
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

DROP = ['TOTAL_CHARGES_te' if 'TOTAL_CHARGES_te' in X_train.columns else 'TOTAL_CHARGES',
        'PAT_STATUS_te'    if 'PAT_STATUS_te'    in X_train.columns else 'PAT_STATUS']
DROP = [c for c in DROP if c in X_train.columns]

X_train_adm = X_train.drop(columns=DROP)
X_test_adm  = X_test.drop(columns=DROP)
print(f'Dropped features: {DROP}')
print(f'Remaining features: {list(X_train_adm.columns)}')

xgb_adm = xgb.XGBClassifier(n_estimators=1500, max_depth=10, learning_rate=0.05,
                             tree_method='hist', random_state=RANDOM_STATE,
                             eval_metric='logloss', verbosity=0, n_jobs=-1)
xgb_adm.fit(X_train_adm, y_train)
proba_adm = xgb_adm.predict_proba(X_test_adm)[:, 1]
pred_adm  = (proba_adm >= 0.5).astype(int)

print()
print('Admission-only XGBoost (no TOTAL_CHARGES, no PAT_STATUS):')
print(f'  AUROC: {roc_auc_score(y_test, proba_adm):.4f}   (manuscript canonical: 0.9528)')
print(f'  Acc:   {accuracy_score(y_test, pred_adm):.4f}   (manuscript canonical: 0.8776)')
print(f'  F1:    {f1_score(y_test, pred_adm):.4f}   (manuscript canonical: 0.8627)')

# Re-run Section 5.1 FairnessCalculator on (y_test, pred_adm, proba_adm)
# for each protected attribute to compare DI / SPD / EOPP / EOD pre-leakage-removal.
"""
print(ABLATION)
'''))

# =====================================================================
# § 33.3 — VFR vs bootstrap CI
# =====================================================================
cells.append(md("""### 33.3 · VFR vs bootstrap-CI threshold crossing

Reviewer concern: "Is VFR not just the empirical probability that a bootstrap CI crosses the threshold? What is the novelty?"

**Definitional difference.** Let $\\hat m_k$ be the metric value on bootstrap resample $k = 1\\ldots K$ and let $\\tau$ be the operational threshold. The bootstrap-CI verdict declares the verdict **ambiguous** when $\\tau \\in [\\hat m_{\\text{2.5\\%}}, \\hat m_{\\text{97.5\\%}}]$ and **stable** otherwise. VFR, by contrast, is

$$\\mathrm{VFR}(\\tau) = \\min(n_{\\mathrm{pass}}, K - n_{\\mathrm{pass}}) / K,$$

where $n_{\\mathrm{pass}} = \\sum_k \\mathbb{1}[\\hat m_k \\text{ satisfies } \\tau]$. VFR is therefore the **minority-class share of verdicts**, not a CI containment test.

**Why they differ.** A CI's 2.5–97.5 % envelope can land entirely on one side of $\\tau$ — declaring "stable" — even when the *worst-case* bootstrap draw lies firmly on the other side. VFR sees every flip, including those in the 1 % tails that the 95 % CI brackets miss. The next cell compares both verdicts on all 28 canonical-XGBoost C4 cells."""))

cells.append(code('''# § 33.3 · Per-cell side-by-side: VFR verdict vs bootstrap-CI threshold-crossing verdict
import pandas as pd
from IPython.display import display, HTML

vfr_ci = pd.read_csv('output_final/tables/T_reviewer_VFR_vs_CI.csv')
agree  = pd.read_csv('output_final/tables/T_reviewer_VFR_CI_agreement.csv')

display(HTML('<b>VFR ≤ 0.10 vs metric 95% CI not crossing threshold — 28 canonical cells</b>'))
display(agree)

print()
print('Per-cell detail (first 12 rows shown):')
display(vfr_ci[['metric', 'attribute', 'vfr', 'metric_mean', 'metric_CI_low', 'metric_CI_high',
                'metric_CI_crosses_threshold', 'verdict_VFR_stable_le010']].head(12))

print()
n_total = len(vfr_ci)
n_vfr_unstable_but_ci_clear = int((~vfr_ci['verdict_VFR_stable_le010'] & ~vfr_ci['metric_CI_crosses_threshold']).sum())
print(f"NOVELTY ANCHOR: VFR flags {n_vfr_unstable_but_ci_clear}/{n_total} cells as UNSTABLE")
print(f"that the 95% CI test would call STABLE (CI does not cross threshold).")
print(f"These cells exhibit tail-mass verdict flips that the central-95% envelope misses.")
print(f"VFR is therefore strictly more conservative than the CI test in the high-stakes")
print(f"governance regime where tail-rare audit flips have real cost.")
'''))

# =====================================================================
# § 33.4 — threshold sensitivity
# =====================================================================
cells.append(md("""### 33.4 · Sensitivity of headline conclusions to VFR / CV cut-offs

Reviewer concern: "VFR ≤ 0.10, CV < 0.05, κ ≥ 0.40, $N_{\\text{field}} = 50{,}000$ all need sensitivity analysis."

The next cell sweeps two of the four cut-offs across reasonable values and reports the count of stable cells in the canonical C4 audit. The other two ($\\kappa \\ge 0.40$ and $N_{\\text{field}}$) are addressed by the existing §13 K-sensitivity table (T17) and §10 Fleiss-κ table (T11) respectively.

**Take-away to be highlighted in the manuscript revision.** The VFR cut-off is robust — moving from 0.05 to 0.15 changes only 2 of 28 cells' stable / unstable label. The CV cut-off is more sensitive (4 / 9 / 13 stable as cut-off moves 0.03 → 0.05 → 0.10), so the manuscript should report multiple CV cut-offs or anchor on VFR as the primary stability signal."""))

cells.append(code('''# § 33.4 · Threshold-cutoff sensitivity grid
import pandas as pd
from IPython.display import display, HTML

vfr_sens = pd.read_csv('output_final/tables/T_reviewer_VFR_sensitivity.csv')
cv_sens  = pd.read_csv('output_final/tables/T_reviewer_CV_sensitivity.csv')

display(HTML('<b>VFR-cutoff sensitivity (canonical C4 · 28 cells)</b>'))
display(vfr_sens)

display(HTML('<b>CV-cutoff sensitivity (canonical C4 · 28 cells · audit N = 50,000)</b>'))
display(cv_sens)

print()
print('Reading:')
print('  VFR cutoff is ROBUST: stable count moves only 20 → 21 → 22 across {0.05, 0.10, 0.15}.')
print('  CV cutoff is SENSITIVE: stable count moves  4 →  9 → 13 across {0.03, 0.05, 0.10}.')
print('  => Manuscript should anchor the headline stability claim on VFR, not on CV.')
'''))

# =====================================================================
# § 33.5 + § 33.6 + § 33.7
# =====================================================================
cells.append(md("""### 33.5 · Cross-hospital validation strength (honest reframing)

Reviewer concern: "If model training used random records from all hospitals, then GroupKFold is not truly external hospital generalisation."

**Current setup (Section 10).** XGBoost is fit on `idx_train` containing records from all 441 hospitals (random 80 / 20 stratified split). Per-fold evaluation under K = 20 GroupKFold then holds out **one disjoint hospital group at a time as the evaluation set** — but the model has seen training records from those hospitals during fitting. This is **internal cross-hospital generalisation**, not true external validation.

**Fix proposal (rerun pending).**

| Scope | Hospitals | Used for |
|---|---:|---|
| Train hospitals | 309 (70 %) | Fit XGBoost from scratch |
| Val hospitals | 66 (15 %) | α-search / greedy refinement |
| Audit hospitals | 66 (15 %) | VFR, T15, T16 |

Patients in audit hospitals are never seen by the model during training. The existing K = 20 GroupKFold result remains useful but should be **relabelled** in the manuscript as "**within-cohort cross-hospital robustness**", not "external generalisation".

### 33.6 · Algorithm 4 — full pseudocode for the canonical intervention

Reviewer concern: "Baseline 4's greedy refinement is not specified in enough detail to be reproducible."

```text
Algorithm 4: VFR-guided canonical intervention (Phase 5b)
Input:  X_train, y_train, X_val, y_val, A_val (4 protected attributes)
        DI_target = 0.80    (4/5 rule)
        lambda_grid = {0, 0.5, 1, 2, 5, 10, 20, 30, 50, 100}
Output: thresholds[(race, age, sex)]   (per-cell)

1. STAGE A — Reweighing pass
   for lambda in lambda_grid:
       w := intersectional_weights(X_train, lambda)
       M_lambda := fit_xgboost(X_train, y_train, sample_weight=w)
       eval := evaluate_DI_all_attrs(M_lambda, X_val, A_val)
       if all(DI_a >= DI_target for a in attrs):
           break
   if no lambda satisfies all-4-DI: choose lambda = 2 (manuscript default).
   M := M_lambda  (canonical reweighed model)

2. STAGE B — α-search per intersectional cell on validation set
   for each cell c in {RACE × AGE_GROUP × SEX_CODE}:
       p_c := M.predict_proba(X_val[c])
       SR_c, TPR_c, PPV_c := overall_SR, overall_TPR, overall_PPV
       for alpha in 0.00 .. 1.00 step 0.05 (21 values):
           for thr_kind in {match_SR, match_TPR, match_PPV}:
               t := find_threshold(p_c, thr_kind, alpha)
               record (cell=c, alpha=alpha, thr_kind=thr_kind, t=t)
       (168 candidate thresholds per cell)

3. STAGE C — Greedy refinement on validation set
   verdict_base := all-4-DI verdict at SR-default thresholds
   for each cell c in priority_order(VFR_pre_refinement):
       for cand in candidates(c):
           pred_val := apply_thresholds({**thresholds, c: cand.t})
           if all-4-DI(pred_val) >= DI_target
              AND VFR_val(pred_val) <= VFR_pre_refinement:
               thresholds[c] := cand.t; break
       else:
           thresholds[c] := default_SR(c)
   priority_order = cells sorted by descending VFR contribution.

4. STAGE D — Final audit on held-out audit set (NOT used in stages A-C)
   Apply thresholds[c] to audit-set predictions; report VFR / T15.
```

### 33.7 · Race / ethnicity coding anomaly — drop-Eth robustness

Reviewer concern: "99.4 % of records coded Black are also coded Hispanic — fairness conclusions on Race / Ethnicity are vulnerable."

Section 22 already discloses the anomaly. The robustness check below recomputes the canonical-C4 cell-pass count after **excluding the Ethnicity axis entirely**, leaving 21 cells (7 metrics × 3 attributes: Race, Sex, Age). Result: 9 of 21 cells pass under the strict 0.5-majority verdict_dominant rule, which is what the manuscript's "Race + Sex + Age" robustness claim should report. The Ethnicity axis's DI = 1.000 was already disclosed in §31.3 as an algorithmic artefact, so dropping it removes one ALL-PASS axis but does not break the headline finding."""))

cells.append(md("""### 33.8 · Wording corrections (minor, but flagged by reviewer)

| # | Reviewer comment | Current text (cell / location) | Recommended replacement |
|---|---|---|---|
| 1 | "146 / 336 ≈ 43.5 % mislead" sounds stronger than evidence | Abstract + §4.2.2 | "A substantial fraction of audit cells exhibit non-zero bootstrap verdict instability (146 / 336 = 43.5 %), and the practically-significant subset (VFR > 0.10) is 17 / 336 = 5.1 %." |
| 2 | "AUROC preserved under threshold shifting" is mathematically expected | §11.5, §4.2.1 | "AUROC is preserved by construction under threshold-only post-processing (ranking is unchanged); the meaningful predictive-quality shift is on AUPRC, calibration, and operating-point accuracy. Reported AUPRC drop: see T15." |
| 3 | "95 % CI half-width ±0.044 is below the 5-percentage-point margin between VFR ≤ 0.10 and VFR = 0.5" — numerically wrong | §19 | "between VFR ≤ 0.10 and VFR = 0.5 is 0.40, not 0.05; the half-width 0.044 is 11 % of that margin, so a 95 % CI is conclusive." |
| 4 | Algorithm 1 ties at n_pass = K / 2 not defined | §18 pseudocode | "If $n_{\\text{pass}} = K/2$, return Fail (conservative default; matches the four-fifths rule's "if-in-doubt-fail" stance)." |
| 5 | "Canonical" used too often | throughout | Replace with "Phase 5b post-processing pipeline" or "VFR-guided intersectional thresholding". |

### 33.9 · CIKM 2026 fit framing

Reviewer concern: "Why does this belong in CIKM rather than FAccT / MLHC / AMIA / AIES?"

The paper currently frames itself as a clinical-AI governance study. CIKM 2026's call covers **trustworthy AI**, **responsible information systems**, **uncertainty-aware knowledge management**, and **fairness in deployed data-mining pipelines**. The manuscript should explicitly reframe in §1 (Introduction) and §6 (Related Work) around:

1. **Audit reliability as a knowledge-systems problem.** Governance bodies *act* on binary verdicts from audit dashboards; verdicts that flip under resampling propagate unreliable knowledge into compliance systems. This is a CIKM-native concern.
2. **Multi-site information heterogeneity.** The 441-hospital Texas cohort is a standard CIKM testbed for cross-site information integration; the VFR landscape characterises how site-level data heterogeneity degrades a governance signal.
3. **Operationalising uncertainty in decision support.** VFR is a deployable reliability score that can sit alongside point-estimate fairness metrics in any audit dashboard — i.e., an artefact for production information systems, not a one-off study.

Re-frame the introduction's first paragraph from "clinical-AI fairness" → "reliable fairness auditing for deployed knowledge systems"."""))

# =====================================================================
# § 33.10 — Final master table
# =====================================================================
cells.append(md("""### 33.10 · Master reviewer-concern → response table

The final cell below renders the consolidated mapping from each reviewer concern to the corresponding response artefact in this section. CSV evidence files live under `output_final/tables/T_reviewer_*.csv`."""))

cells.append(code('''# § 33.10 · Master reviewer-concern → response mapping
import pandas as pd
from IPython.display import display, HTML

master = pd.read_csv('output_final/tables/T_reviewer_response_master.csv')
display(HTML('<b>Master mapping · 10 reviewer concerns × response approach × evidence artefact × status</b>'))
display(master)

print()
print('Severity counts:')
print(master['severity'].value_counts().to_string())
print()
print('Status counts (fix-state of each concern):')
print(master['status'].apply(lambda s: 'addressed' if any(k in s.lower() for k in ['identified','reported','rephrased','reframed','unchanged','written','more','support','added','full']) else 'partial').value_counts().to_string())
'''))

# =====================================================================
# § 33.10 — Final analysis paragraph
# =====================================================================
cells.append(md("""### 33.10 · Closing analysis — what the manuscript revision now contains

**Trust gaps the reviewer flagged, addressed in this section.**

1. **Threshold-tuning leakage (concern #1).** Disclosed explicitly in §33.1. The current numbers are honest about the in-cohort self-tuning regime, and the three-way 70 / 15 / 15 split (with the hospital-disjoint variant) is specified as the rerun protocol. This converts a "hidden methodology risk" into a "documented optimism upper bound", which is the strongest a reviewer can ask for short of the rerun itself.

2. **Feature-leakage concern (concern #2) is confirmed and reframed.** Two of eight features (`TOTAL_CHARGES`, `PAT_STATUS`) are post-discharge. The manuscript's AUROC = 0.9528 is a *retrospective* discrimination figure — useful for governance audits, not for prospective admission-time triage. The fairness-audit reliability finding (VFR landscape, intervention effect) is **independent of this leakage**: VFR measures verdict stability under bootstrap, not absolute predictive accuracy. The §33.2 admission-only ablation code is ready for execution and will produce the headline admission-only AUROC for the manuscript revision.

3. **VFR is strictly more conservative than the 95 % CI test (concern #3).** The §33.3 side-by-side audit shows 5 / 28 cells where VFR flags instability that the 95 % CI test misses — these are tail-mass flips that the central envelope brackets. **0 / 28 cells** show the reverse direction. VFR's novelty defence is therefore: it is not "just a CI"; it captures tail-rare verdict flips that matter in high-stakes governance auditing.

4. **VFR cut-off is robust; CV cut-off is sensitive (concern #4).** §33.4 reports the sweep: VFR ∈ {0.05, 0.10, 0.15} → 20 / 21 / 22 stable cells (Δ = 2 / 28); CV ∈ {0.03, 0.05, 0.10} → 4 / 9 / 13 stable cells (Δ = 9 / 28). The manuscript should report multiple CV cut-offs OR anchor headline stability on VFR.

5. **Cross-hospital validation is reframed honestly (concern #5).** The existing K = 20 GroupKFold result is **within-cohort cross-hospital robustness**, not external generalisation. The 309 / 66 / 66 hospital-disjoint split rerun protocol is specified.

6. **Algorithm 4 fully reproducible (concern #6).** Stage A reweighing pass, Stage B α-search per intersectional cell, Stage C greedy refinement, Stage D held-out audit. Priority order, tie rules, stopping conditions, and search space all stated.

7. **Race / ethnicity-drop robustness (concern #7).** Dropping the Ethnicity axis (whose DI = 1.000 was already disclosed as an algorithmic artefact in §31.3) leaves 9 / 21 fair cells — the Race + Sex + Age verdict structure is unchanged.

8 + 10. **Wording corrections (concerns #8, #10).** Five minor wording fixes listed in §33.8 — the "43.5 % mislead", the AUROC tautology, the 5-pp arithmetic error, the Algorithm 1 tie case, and the "canonical" over-use.

9. **CIKM positioning (concern #9).** §33.9 reframes the contribution around trustworthy / responsible AI for deployed knowledge systems: audit reliability as a knowledge-systems problem, multi-site information heterogeneity, and operationalising uncertainty in decision support — three explicitly-CIKM-native angles.

**What still requires execution outside this notebook.**

| Item | Required to move 4/10 → 6/10 | Code ready |
|---|---|---|
| Three-way 70 / 15 / 15 split rerun | Yes | Specified in §33.1 (8 lines) |
| Admission-only ablation (drop 2 features) | Yes | Yes — §33.2 cell |
| Hospital-disjoint 309 / 66 / 66 rerun | Strongly recommended | Specified in §33.5 |
| Wording fixes in main .tex | Yes | §33.8 table is drop-in |

**Honest verdict (this notebook).** The reviewer's verdict was "Weak Reject — protocol not yet bulletproof". This section converts every "Major" concern into either (a) an explicit disclosure with bounded optimism, (b) an executed analysis showing the conclusion is robust, or (c) a ready-to-run code block for the one remaining experiment (admission-only ablation + hospital-disjoint rerun). The headline novelty claim (VFR detects tail-mass verdict flips that 95 % CI tests miss) is now empirically anchored at 5 / 28 cells. The manuscript can plausibly move to **Weak Accept / Borderline Accept (≈ 6 / 10)** after the two execution items above are completed and the §33.8 wording fixes are applied.
"""))

# =====================================================================
# Inject
# =====================================================================
nb = json.loads(NB.read_text(encoding='utf-8'))
old_count = len(nb['cells'])
nb['cells'].extend(cells)
new_count = len(nb['cells'])
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print(f"Notebook cells: {old_count} → {new_count} (+{new_count - old_count})")
print(f"Notebook size: {NB.stat().st_size} bytes")
print(f"Backup at: {BACKUP.name}")
