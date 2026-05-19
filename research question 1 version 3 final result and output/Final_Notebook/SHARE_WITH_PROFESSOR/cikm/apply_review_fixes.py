"""
Apply 6 surgical fixes flagged by the brutal CIKM-reviewer audit.

Issues addressed:
  1. Undisclosed 4.24 -> 6.21 pp accuracy-cost delta
  2. Missing Algorithm 1 pseudocode citation
  3. "0.08 leakage score" framing risk (clarify as 8.09 pp AUROC drop)
  4. Seed=123 reproducibility scope clarification
  5. AUROC 0.9528 (CIKM) vs 0.9410 (journal) reconciliation
  6. §35 drop-in trace table mapping paragraphs to source CSVs

Adds a new §35.7 cell to BOTH notebooks (CIKM submission + journal extended).
"""
import json
from pathlib import Path

ROOT = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm")
CIKM = ROOT / "CIKM_2026_LOS_Fairness_FINAL.ipynb"
JNL  = ROOT / "full_journal_paper" / "Journal_LOS_Fairness_FULL.ipynb"

fix_cell = {
    "cell_type": "markdown", "metadata": {},
    "source": ("""### 35.7 · Reviewer-audit fix-ups (six surgical fixes)

A brutal CIKM-reviewer audit of this notebook flagged six remaining attack surfaces. Each is addressed below with a drop-in correction that supersedes the corresponding §35.x paragraph if there is overlap.

#### Fix 1 — Accuracy-cost delta is now disclosed (supersedes earlier silence)

The CIKM submission's headline accuracy cost is **4.24 pp** (single 80/20 split, threshold tuning on test). The journal-rerun bulletproof three-way 70/15/15 split with α-search restricted to a separate VAL slice raises the honest cost to **6.21 pp** (seed=42) and **6.30 pp** (seed=123). The delta (≈ 2 pp) is the empirical bound on selection-on-test optimism in the CIKM accuracy-cost number.

**Drop-in sentence for Results / §5.1:**

> The reported accuracy cost of 4.24 pp is measured on the single 80/20 split with threshold tuning on the held-out test set; an independent bulletproof three-way 70/15/15 split with α-search restricted to a separate VAL partition raises the cost to 6.21 pp (journal §34.1), which bounds the selection-on-test optimism in the headline figure at ≈ 2 pp. The fairness contribution (all-four-DI ≥ 0.80) is preserved in both protocols.

#### Fix 2 — Algorithm 1 / Algorithm 4 reproducibility citation

The full pseudocode for the four-stage canonical pipeline (Stage A reweighing → Stage B α-search → Stage C greedy refinement → Stage D audit-only reporting) is in **§33.6 of this notebook** (markdown cell, ≈ 30 lines). The §35.4 manuscript drop-in paragraph references it explicitly: in the LaTeX source, replace the placeholder line "see notebook" with "see Algorithm 1 in Appendix C", and copy §33.6 verbatim into Appendix C.

#### Fix 3 — Leakage-score wording (clarify the 0–1 scale meaning)

The §35.3 framing of "leakage score = 0.08" is on the AUROC scale (0–1), where 0.08 means an 8.09-percentage-point drop in AUROC after removing the two end-of-encounter features. This is "low" relative to the worst-case scenario (≥ 0.20 drop is typically considered serious leakage) but **not** "no leakage". The corrected drop-in (supersedes §35.3 paragraph):

> **Feature-leakage diagnostic — strict wording.** Removing the two end-of-encounter features (`TOTAL_CHARGES`, `PAT_STATUS`) drops AUROC from 0.9410 to 0.8601 on the bulletproof audit partition, an **8.09-percentage-point AUROC drop** (i.e., a leakage diagnostic score of 0.0809 on the 0–1 scale). This is a *moderate* but *bounded* leakage signal under the implemented diagnostic: the admission-only model still attains AUROC = 0.8601 with all-four-DI ≥ 0.80 and 23 / 28 VFR-stable cells. The result is therefore consistent with retrospective audit-reliability evaluation; it should **not** be read as proof that the model is admission-time-deployable without further temporal-feature validation.

#### Fix 4 — Seed=123 reproducibility scope (clarify it is split-shuffle, not external)

The §34.3 result confirms reproducibility under **resampling of the patient-level 70/15/15 split**. It does *not* test cross-cohort or temporal external validity. The clearer drop-in caption:

> *§34.3 reproducibility check, strict scope.* This experiment shuffles the patient-level 70/15/15 partition (seed = 42 → 123) and verifies that all-four-DI and VFR landscape recover under independent random sub-sampling. It does **not** constitute external validation; cross-hospital and temporal-cohort generalisation are addressed in §34.5 (Limitations).

#### Fix 5 — AUROC headline reconciliation (manuscript 0.9528 vs journal 0.9410)

The CIKM manuscript uses AUROC = **0.9528** from the single 80/20 split with full 8-feature model. The journal bulletproof rerun reports AUROC = **0.9410** on the held-out audit partition. The 1.18-pp difference is attributable to (a) a slightly smaller training set (70 % vs 80 %), and (b) the loss of selection-on-test optimism. **Drop-in footnote for the manuscript Results section:**

> *AUROC = 0.9528 is measured on the CIKM-submission single 80/20 split. An independent three-way 70/15/15 rerun (journal §34.1) yields AUROC = 0.9410 on the held-out audit slice, a 1.18-pp drop attributable to the smaller training set and removal of selection-on-test optimism. Both figures correspond to the full 8-feature model; the admission-only ablation (§35.3) yields AUROC = 0.8601.*

#### Fix 6 — §35 drop-in trace table (paragraph → source artefact)

Every §35 drop-in paragraph below is anchored to a specific journal artefact. **Use this table when responding to a reviewer who asks "where in the artefacts is this number?":**

| §35 paragraph | Claim | Source CSV / cell |
|---|---|---|
| §35.1 (overclaim) | "146 / 336 cells reverse at least once; 17 / 336 reach VFR > 0.10" | `output_final/tables/T13_axis1_vfr_config1.csv` (146 = non-zero VFR rows × 12 models); §17 verification check `cv_gt_50_count_is_17` |
| §35.2 (protocol) | "Bulletproof three-way 70/15/15 confirms all-four-DI on held-out audit" | `full_journal_paper/output/tables/journal_summary.csv` row `70_15_15` → `all_4_DI_pass_audit = True`, journal §34.1 cell |
| §35.3 (leakage 0.08) | "AUROC drop = 0.0809 after removing TOTAL_CHARGES, PAT_STATUS" | `full_journal_paper/output/tables/journal_summary.csv` row `admission_only` AUROC=0.8601 vs row `70_15_15` AUROC=0.9410 |
| §35.4 (B4 pseudocode) | Algorithm 4 spec | §33.6 of this notebook (markdown, ≈ 30 lines) |
| §35.5 (VFR vs CI) | "5 / 28 cells where VFR flags instability that CI test misses" | `output_final/tables/T_reviewer_VFR_CI_agreement.csv` row "VFR unstable but CI clear (CI is more permissive)" = 5 |
| §35.6 (manifest) | Master fix list | (text only; no CSV) |

#### Closing note for reviewers / co-authors

The CIKM submission is unchanged in Sections 1–32. Sections 33 (reviewer-response disclosures), 35 (manuscript drop-ins), and the journal extension (`full_journal_paper/`) provide the empirical and textual material needed to respond to reviewer concerns *without retracting any headline claim*. Each numerical claim in §35 is now traceable to a specific CSV file and section; co-authors editing the .tex can cite either the notebook section or the underlying CSV row.
""").splitlines(keepends=True)
}

# Inject into both notebooks
for nb_path, label in [(CIKM, 'CIKM'), (JNL, 'Journal')]:
    nb = json.loads(nb_path.read_text(encoding='utf-8'))
    nb['cells'].append(fix_cell)
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f"{label}: appended §35.7 fix-ups, total cells now {len(nb['cells'])}")
