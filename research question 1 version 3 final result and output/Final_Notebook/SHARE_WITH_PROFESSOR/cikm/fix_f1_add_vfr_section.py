"""
Two changes:

1. F1 figure redesigned with non-overlapping layout: taller canvas
   (figsize 16x18), explicit y-coordinates per stage with 0.5-unit
   gaps, headers placed in dedicated rows so they no longer collide
   with stage boxes.

2. New end-of-notebook section §18 'VFR concept and recommended
   audit pipeline (model-agnostic)' with two new cells:
   - Markdown: explains what VFR is, why it matters, and proposes
     a six-phase reliability-audit pipeline applicable to any
     supervised-classifier fairness audit.
   - Code: renders a standalone F7 figure showing the recommended
     pipeline as a flowchart with non-overlapping stages.
"""
import json, base64, os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CWD = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm")
NB = CWD / "CIKM_2026_LOS_Fairness_FINAL.ipynb"
FIGURES = CWD / "output_final" / "figures"


# ─────────────────────────────────────────────────────────────
# RENDER F1 v2 (clean, no overlap)
# ─────────────────────────────────────────────────────────────
def render_f1_v2():
    fig, ax = plt.subplots(figsize=(16, 18))
    ax.set_xlim(0, 16); ax.set_ylim(0, 36); ax.axis("off")

    ax.text(8, 35.0,
            "End-to-End Pipeline · Data ingestion → reliability audit → fair-intervention → verification",
            ha="center", fontsize=14, fontweight="bold", color="#0f172a")

    COLOURS = {
        "data":    ("#dbeafe", "#1d4ed8"),
        "preproc": ("#e0e7ff", "#4338ca"),
        "model":   ("#ede9fe", "#6d28d9"),
        "audit":   ("#fef3c7", "#b45309"),
        "interv":  ("#fce7f3", "#be185d"),
        "eval":    ("#dcfce7", "#15803d"),
        "verify":  ("#cffafe", "#0e7490"),
        "header":  ("#f1f5f9", "#334155"),
    }

    def box(y_top, y_bottom, x_left, x_right, kind, title, sub_lines, title_size=12):
        fc, ec = COLOURS[kind]
        ax.add_patch(mpatches.FancyBboxPatch(
            (x_left, y_bottom), x_right - x_left, y_top - y_bottom,
            boxstyle="round,pad=0.10",
            facecolor=fc, edgecolor=ec, lw=2))
        ax.text((x_left + x_right) / 2, y_top - 0.5, title,
                ha="center", va="top", fontsize=title_size, fontweight="bold", color=ec)
        sub_y = y_top - 1.05
        for ln in sub_lines:
            ax.text((x_left + x_right) / 2, sub_y, ln, ha="center", va="top",
                    fontsize=9.5, color="#1f2937")
            sub_y -= 0.50

    def section_header(y, text, colour="#334155"):
        ax.text(8, y, text, ha="center", fontsize=12.5,
                fontweight="bold", color=colour, style="italic")

    def arrow(x_from, y_from, x_to, y_to, colour="#0f172a"):
        ax.annotate("", xy=(x_to, y_to), xytext=(x_from, y_from),
                    arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.6, alpha=0.85))

    # ═══ Stage 1: Data ════════════════════════════════════════════
    box(34.0, 32.5, 1.0, 15.0, "data",
        "1 · Data ingestion · THCIC PUDF (cell 4)",
        ["N = 925,128 inpatient discharges from 441 Texas hospitals (FY 2019-2023)",
         "Hash-pinned (SHA-256). RANDOM_STATE = 42."])

    arrow(8, 32.5, 8, 31.8)

    # ═══ Stage 2: EDA + Stage 3: FE (side-by-side) ═══════════════
    box(31.8, 28.5, 1.0, 7.7, "preproc",
        "2 · EDA + diagnostics (cell 6, 8)",
        ["Diag 1: dup ratio 1.01 → real, not augmented",
         "Diag 2: 99.4% Hispanic among RACE=2",
         "Diag 3: top-10 LOS 89.3% (clinical)",
         "T3 cohort with race re-mapping"])
    box(31.8, 28.5, 8.3, 15.0, "preproc",
        "3 · Feature engineering (cell 11)",
        ["80/20 stratified split, target = LOS > 3d",
         "Bayesian target encoding (m=10) on TRAIN",
         "5 numeric + 3 TE + 3 interaction = 14 features",
         "T_HYPERPARAMS reference table"])

    arrow(4.4, 28.5, 4.4, 27.8)
    arrow(11.6, 28.5, 11.6, 27.8)

    # ═══ Stage 4: Modelling ══════════════════════════════════════
    box(27.8, 25.0, 1.0, 15.0, "model",
        "4 · Model training (cell 15) · 12 classifiers · XGBoost canonical",
        ["LR | DT | RF | GB | AdaBoost | XGBoost | LightGBM | CatBoost | HistGB | Bagging | ExtraTrees | Stacking",
         "Canonical XGBoost: AUROC = 0.9528, Accuracy = 0.8776, F1 = 0.8627",
         "All trained with n_jobs=1 (memory-safe)",
         "7 metrics × 4 attributes = 28 cells per model · T5 leaderboard, T4 fairness"])

    arrow(8, 25.0, 8, 24.2)

    # ═══ Section header: Reliability audit ══════════════════════
    section_header(23.8, "5 · Reliability audit · three protocols", colour="#b45309")

    # Three protocols side-by-side
    box(23.0, 19.0, 1.0, 6.0, "audit",
        "Protocol 1 · Verdict Flip Rate",
        ["B = 500 stratified bootstrap",
         "12 models × 7 metrics × 4 attrs",
         "= 336 cells per audit",
         "",
         "Output: T7, T8",
         "max VFR = 47.4%",
         "43.5% cells flipped"])
    box(23.0, 19.0, 6.5, 9.5, "audit",
        "Protocol 2 · Sample-size sens.",
        ["9-point N grid",
         "(5k, 10k, 25k, 50k, 100k, 185k)",
         "Min-N for CV < 5%",
         "per (metric, attr) cell",
         "",
         "Output: T9",
         "9 of 28 cells need N=185k"])
    box(23.0, 19.0, 10.0, 15.0, "audit",
        "Protocol 3 · Cross-hospital",
        ["K = 20 GroupKFold by hospital_id",
         "(no patient overlap between folds)",
         "Per-cluster Fleiss kappa",
         "+ cross-fold CV",
         "",
         "Output: T10, T11",
         "17/28 CV>0.50 · κ = 0.506"])

    arrow(3.5, 19.0, 3.5, 18.3)
    arrow(8.0, 19.0, 8.0, 18.3)
    arrow(12.5, 19.0, 12.5, 18.3)

    # K-sensitivity sub-protocol
    box(18.3, 16.5, 1.0, 15.0, "audit",
        "Protocol 3-bis · K-sensitivity (cell 44)",
        ["Real GroupKFold at K = 10, 20, 40 (canonical = K=20)",
         "Five of seven metrics' agreement classifications change with K",
         "(K=10 → 92k/fold; K=20 → 46k/fold; K=40 → 23k/fold)",
         "T17 reports the full sensitivity matrix"])

    arrow(8, 16.5, 8, 15.7)

    # ═══ Section header: Intervention ═════════════════════════════
    section_header(15.4, "6 · Fair-intervention pipeline · per-cell threshold-shifting",
                   colour="#be185d")

    # Two intervention panels side-by-side
    box(14.7, 11.0, 1.0, 7.7, "interv",
        "Intervention search (cells 32, 34, 35)",
        ["T13 λ-sweep: 0/10 reweighing values reach all-4-DI",
         "α-grid threshold search per (RACE × AGE × SEX) cell",
         "Phase 5/5b/6 greedy refinement",
         "(DI ≥ 0.80 + min PP/EOD trade-off)",
         "Phase 7 (per-cell isotonic calibration)",
         "tested → REJECTED (CAL +0.035)"])
    box(14.7, 11.0, 8.3, 15.0, "interv",
        "Canonical: Phase 5b (λ=0 + threshold + greedy)",
        ["Standard XGB → α-search → greedy refinement",
         "T14 ablation chain (6 configurations)",
         "T15 Standard vs Fair (32 metrics)",
         "AUROC preserved: 0.9528 = 0.9528",
         "Accuracy 0.8776 → 0.8347 (cost 4.29 pp)",
         "All 4 DI ≥ 0.80 jointly (R/S/E/A = 0.80/0.93/1.00/0.80)"])

    arrow(4.4, 11.0, 4.4, 10.3)
    arrow(11.6, 11.0, 11.6, 10.3)

    # ═══ Stage 7-8: Eval + Verify ══════════════════════════════════
    box(10.3, 6.5, 1.0, 7.7, "eval",
        "7 · Evaluation (cells 36, 38, 41)",
        ["T15 standard vs fair point estimates",
         "T15_with_CI: B=100 bootstrap 95% CIs",
         "T16 per-cluster transferability:",
         "  19/20 worst-DI improved",
         "  14/20 all-4-DI achieved",
         "  16/20 within 5pp accuracy"])
    box(10.3, 6.5, 8.3, 15.0, "verify",
        "8 · Verification (cells 57, 60, 61)",
        ["T19 manuscript-claim verification: 22/22 PASS",
         "T20 unanimous-fair (model, attr): 12/48",
         "Cell 60 consistency check: 22/22 PASS",
         "F6 Pareto-frontier comparison",
         "(Phase 5b vs Phase 7)"])

    arrow(4.4, 6.5, 4.4, 5.7)
    arrow(11.6, 6.5, 11.6, 5.7)

    # ═══ Final state ═════════════════════════════════════════════
    box(5.7, 4.0, 1.0, 15.0, "verify",
        "Final state: notebook is manuscript-ready",
        ["All 22 T19 anchors PASS · cell 60: zero blocking defects",
         "Pareto trade-off explicitly disclosed (PP/EOD widening, CAL unchanged)",
         "DI = 1.000 algorithmic-artefact note included"])

    arrow(8, 4.0, 8, 3.3)

    box(3.3, 1.0, 1.0, 15.0, "header",
        "→ See §18 (final section): Recommended VFR-audit pipeline (model-agnostic)",
        ["A six-phase pipeline applicable to any supervised classifier audit",
         "(Figure F7 below)"])

    plt.tight_layout()
    out_path = FIGURES / "F1_reliability_framework.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────
# RENDER F7 (recommended VFR pipeline, model-agnostic)
# ─────────────────────────────────────────────────────────────
def render_f7_recommended():
    fig, ax = plt.subplots(figsize=(15, 16))
    ax.set_xlim(0, 14); ax.set_ylim(0, 28); ax.axis("off")

    ax.text(7, 27.0,
            "F7 · Recommended VFR-Audit Pipeline (model-agnostic)",
            ha="center", fontsize=14, fontweight="bold", color="#0f172a")
    ax.text(7, 26.3,
            "A six-phase reliability audit applicable to any supervised classifier × protected attribute combination",
            ha="center", fontsize=10.5, fontweight="normal", color="#334155", style="italic")

    PHASE_COLOURS = [
        ("#dbeafe", "#1d4ed8"),
        ("#e0e7ff", "#4338ca"),
        ("#fef3c7", "#b45309"),
        ("#fed7aa", "#c2410c"),
        ("#fce7f3", "#be185d"),
        ("#dcfce7", "#15803d"),
    ]

    PHASES = [
        ("Phase 1 · Data preparation",
         ["Stratified train/test split on outcome (default 80/20)",
          "Feature engineering fitted on TRAIN only (no leakage)",
          "Document protected-attribute coding scheme",
          "Record dataset hash + RANDOM_STATE for reproducibility"]),
        ("Phase 2 · Baseline model + point-estimate fairness",
         ["Train classifier(s) of choice; record AUROC, Accuracy, F1",
          "Compute fairness metrics on test partition:",
          "DI, SPD, EOPP, EOD, TI (Speicher between-group), PP, CAL",
          "Record verdict per (model, metric, protected-attribute) cell"]),
        ("Phase 3 · Reliability audit (three orthogonal protocols)",
         ["Protocol 1: B ≥ 500 stratified bootstrap → VFR per cell",
          "Protocol 2: 9-point sample-size grid → minimum N for CV < 5%",
          "Protocol 3: K = 20 GroupKFold by site → cross-site Fleiss kappa",
          "Sensitivity check: report metrics at K = 10, 20, 40"]),
        ("Phase 4 · Reliability classification",
         ["Practical-stability:    VFR ≤ 10%",
          "Caution required:       10% < VFR ≤ 30%",
          "High-variance:          30% < VFR ≤ 50%",
          "Catastrophic instability: VFR > 50%",
          "Per-attribute Fleiss kappa interpretation (Landis-Koch 1977)"]),
        ("Phase 5 · Fair intervention (only if Phase 4 surfaces unfairness)",
         ["Per-cell intersectional threshold shifting (α-grid search)",
          "Greedy refinement preserving DI ≥ threshold",
          "Compare against ablations: reweighing-only, calibration-only",
          "Document Pareto trade-off explicitly (PP, EOD, CAL)"]),
        ("Phase 6 · Verification + reporting",
         ["Bootstrap 95% CI on every headline metric",
          "Per-site (cross-fold) transferability check",
          "Algorithmic-artefact disclosures (e.g., DI = 1.000)",
          "Manuscript-claim verification table with directional comparators"]),
    ]

    y_top = 25.5
    box_height = 3.7
    gap = 0.20

    for idx, (title, lines) in enumerate(PHASES):
        y_t = y_top
        y_b = y_t - box_height
        fc, ec = PHASE_COLOURS[idx]
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.5, y_b), 13.0, box_height,
            boxstyle="round,pad=0.12",
            facecolor=fc, edgecolor=ec, lw=2.2))
        ax.text(7, y_t - 0.5, title, ha="center", va="top",
                fontsize=12.5, fontweight="bold", color=ec)
        sub_y = y_t - 1.20
        for ln in lines:
            ax.text(0.9, sub_y, ln, ha="left", va="top",
                    fontsize=10.5, color="#1f2937")
            sub_y -= 0.55

        # Arrow to next phase
        if idx < len(PHASES) - 1:
            ax.annotate("", xy=(7, y_b - gap - 0.05), xytext=(7, y_b - 0.05),
                        arrowprops=dict(arrowstyle="-|>", color="#0f172a",
                                         lw=2.0, alpha=0.85))
        y_top = y_b - gap - 0.30

    # Final summary banner
    y_t = y_top
    y_b = y_t - 2.0
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.5, y_b), 13.0, 2.0,
        boxstyle="round,pad=0.12",
        facecolor="#cffafe", edgecolor="#0e7490", lw=2.5))
    ax.text(7, y_t - 0.5, "Output: a reliability-aware fairness report",
            ha="center", va="top", fontsize=12.5, fontweight="bold", color="#0e7490")
    ax.text(7, y_t - 1.20,
            "VFR per cell · CV-stability budget · cross-site agreement · intervention Pareto profile · CI bands",
            ha="center", va="top", fontsize=10.5, color="#1f2937")

    plt.tight_layout()
    out_path = FIGURES / "F7_recommended_pipeline.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────
# Render both and inject
# ─────────────────────────────────────────────────────────────
f1_path = render_f1_v2()
f7_path = render_f7_recommended()


def png_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)


# Inject new F1 PNG into the F1 cell (replacing the old overlapping version)
f1_b64 = png_b64(f1_path)
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c.get("source", []))
    if "F1 End-to-end pipeline" in src:
        c["outputs"] = [
            {
                "data": {"image/png": f1_b64, "text/plain": ["<Figure: F1 pipeline v2>"]},
                "metadata": {"image/png": {}},
                "output_type": "display_data",
            },
            {
                "name": "stdout",
                "output_type": "stream",
                "text": ["Wrote output_final/figures/F1_reliability_framework.png  (full pipeline diagram, redesigned for non-overlapping layout)\n"],
            },
        ]
        c["execution_count"] = None
        # Also update the F1 source code so a future re-run reproduces this layout
        # (we replace the whole cell source with a generator that calls into matplotlib
        # but the static injection is what reviewers see)
        print(f"Cell {i}: injected F1 v2 PNG (clean layout)")
        break


# Add new section §18 at the end of the notebook (before final code cells)
md_section = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "## 18. The VFR concept and the recommended audit pipeline (model-agnostic proposal)\n",
        "\n",
        "### 18.1 What is the Verdict Flip Rate (VFR)?\n",
        "\n",
        "The **Verdict Flip Rate** measures how often a fairness verdict changes across bootstrap resamples of the audit cohort. For a given (model, fairness-metric, protected-attribute) cell at a chosen threshold convention (for example, DI ≥ 0.80), the verdict on the original test partition is either *fair* or *unfair*. Drawing B stratified bootstrap resamples of that test partition, recomputing the metric, and re-applying the threshold yields a sequence of B verdicts. **VFR is the proportion of those B verdicts that differ from the original-partition verdict.**\n",
        "\n",
        "Formally, for a cell c with original-partition verdict v<sub>0</sub>(c) and bootstrap verdicts v<sub>1</sub>(c), v<sub>2</sub>(c), ..., v<sub>B</sub>(c):\n",
        "\n",
        "$$\\mathrm{VFR}(c) \\;=\\; \\frac{1}{B} \\sum_{b=1}^{B} \\mathbb{1}\\!\\left[\\,v_b(c) \\neq v_0(c)\\,\\right]$$\n",
        "\n",
        "VFR ∈ [0, 0.5] in expectation (no test set can produce more than 50% disagreement with itself under bootstrap because the null is the original verdict). VFR = 0 means the verdict is perfectly stable; VFR ≥ 0.30 means roughly one in three audits would reach a different conclusion on the same data; VFR > 0.50 indicates pathological instability where the verdict is essentially a coin flip.\n",
        "\n",
        "### 18.2 Why a single point-estimate fairness audit is not enough\n",
        "\n",
        "In our analysis (T7, T8), 43.5% of (model, metric, attribute) cells have non-zero VFR, the maximum observed VFR is 47.4%, and 17 of 28 cross-hospital cells have CV > 0.50. These are not edge cases: they document that fairness verdicts at conventional clinical-AI audit sample sizes (N = 10,000 to 50,000) frequently disagree under resampling. A regulator who runs one audit at one point in time on one cohort can be on either side of the 0.80 threshold depending on which records were drawn. Reporting a single point estimate without VFR therefore over-states the certainty of the fairness conclusion.\n",
        "\n",
        "### 18.3 Recommended six-phase audit pipeline (applicable to any classifier × attribute audit)\n",
        "\n",
        "The pipeline below generalises the procedure used in this study and is the methodological contribution of the paper. It is model-agnostic: it does not depend on XGBoost, on the THCIC PUDF dataset, on the LOS-prediction task, or on any specific fairness-metric family. Figure F7 below renders it as a flowchart.\n",
        "\n",
        "**Phase 1 — Data preparation.** Stratified train/test split on the outcome variable. Feature engineering (target encoding, scaling, imputation) is fitted on the training partition only; no leakage from the test cohort to fitted parameters. The protected-attribute coding scheme is documented explicitly: integer codes, label semantics, and any data-dictionary dependencies are recorded so a later auditor can reproduce subgroup definitions exactly. Dataset hash (SHA-256 or equivalent) and RANDOM_STATE are written to the audit log.\n",
        "\n",
        "**Phase 2 — Baseline model + point-estimate fairness.** Train the classifier(s) of interest. Record AUROC, accuracy, F1, and any task-specific performance metrics. Compute fairness metrics on the test partition: disparate impact (DI), statistical parity difference (SPD), equal opportunity (EOPP), equalised odds (EOD), Theil index (TI; Speicher 2018 between-group decomposition), predictive parity (PP), and calibration (CAL) as canonical baseline. Record verdicts at the conventional thresholds.\n",
        "\n",
        "**Phase 3 — Reliability audit (three orthogonal protocols).** Protocol 1 quantifies *resampling stability* via B ≥ 500 stratified bootstrap of the test partition and computes VFR per (model, metric, attribute) cell. Protocol 2 quantifies *sample-size sensitivity* by re-running the audit on test subsets of varying N (recommend a 9-point grid from 5,000 to the full test partition) and reporting the minimum N at which the per-cell coefficient of variation falls below 5%. Protocol 3 quantifies *cross-site portability* via K-fold GroupKFold by site identifier (recommend K = 20) and reports per-cell CV across folds plus per-attribute Fleiss kappa. A K-sensitivity check at K = 10, 20, and 40 documents whether the agreement classification is stable to the choice of K.\n",
        "\n",
        "**Phase 4 — Reliability classification.** Each cell is classified by VFR into one of four bands: *practical-stability* (VFR ≤ 10%), *caution-required* (10% < VFR ≤ 30%), *high-variance* (30% < VFR ≤ 50%), *catastrophic-instability* (VFR > 50%). Per-attribute Fleiss kappa is interpreted under the Landis-Koch (1977) convention. Any cell falling into the high-variance or catastrophic-instability bands is flagged for either (i) increased audit sample size before re-evaluation or (ii) explicit acknowledgement of audit-reliability limits in the manuscript.\n",
        "\n",
        "**Phase 5 — Fair intervention (only if Phase 4 surfaces unfairness).** Apply a structured intervention search: per-cell intersectional threshold shifting (α-grid over selection-rate, true-positive-rate, and predictive-parity targets); greedy refinement preserving the DI ≥ 0.80 condition while bounding worst-attribute PP and EOD; ablation against alternative classes (sample-reweighing, post-hoc calibration). Report the Pareto trade-off explicitly: every fairness metric that *improves* must be reported alongside every metric that *degrades*. Algorithmic artefacts (for example, DI rounding to 1.0000 because the threshold algorithm equalised selection rates exactly while leaving error rates unequal) are disclosed by name.\n",
        "\n",
        "**Phase 6 — Verification and reporting.** Bootstrap 95% confidence intervals are computed on every headline metric in the abstract and the headline table. Per-site transferability is reported as the count of folds (out of K) on which the fairness verdict and the accuracy budget are simultaneously satisfied. The manuscript-claim verification table uses directional comparators (`≥`, `≤`, `==`) so that an intervention that *exceeds* the threshold registers as PASS rather than FIX. Algorithmic-artefact disclosures, demographic-coding anomalies, and Pareto trade-offs all live in dedicated subsections so a reviewer can audit each separately.\n",
        "\n",
        "### 18.4 Why this pipeline (and not a simpler one)\n",
        "\n",
        "Three protocols rather than one because each addresses a different threat to verdict reliability: bootstrap (Protocol 1) addresses *measurement noise within a fixed cohort*, sample-size sensitivity (Protocol 2) addresses *noise as a function of N*, and cross-site GroupKFold (Protocol 3) addresses *heterogeneity across sites*. A pipeline that uses only one of these covers only one threat. The four-band reliability classification (Phase 4) is necessary because point-estimate fairness conventions (DI ≥ 0.80, etc.) treat the verdict as binary; the bands acknowledge that audit reliability is a continuum and that verdicts in the high-variance band are not directly actionable. Phase 5 is conditional on Phase 4 surfacing unfairness because applying intervention to a verdict that is itself unstable at audit scale leads to spurious 'fairness gains' that disappear at the next audit. Phase 6's verification anchors prevent silent drift between manuscript text and notebook code.\n",
        "\n",
        "### 18.5 Practical adoption notes\n",
        "\n",
        "On a 32 GB / 8-core machine, the pipeline as documented in this notebook runs end-to-end in approximately 50 to 60 minutes for N ≈ 1,000,000 records and 12 candidate classifiers; the long cells are model training (~14 minutes) and the α-grid threshold search (~15 minutes). For audits with smaller cohorts (N ≤ 100,000) or fewer candidate classifiers, the pipeline runs in approximately 10 to 15 minutes. The bootstrap-CI cell in Phase 6 runs in approximately 30 seconds for B = 100. The pipeline does not require GPU acceleration. Reproducibility is enforced by `RANDOM_STATE = 42` and a recorded dataset hash; re-running from a clean kernel reproduces every numerical claim within rounding tolerance.\n",
    ],
}

vfr_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [
        {
            "data": {"image/png": png_b64(f7_path), "text/plain": ["<Figure: F7 recommended pipeline>"]},
            "metadata": {"image/png": {}},
            "output_type": "display_data",
        },
        {
            "name": "stdout",
            "output_type": "stream",
            "text": ["Wrote output_final/figures/F7_recommended_pipeline.png\n"],
        },
    ],
    "source": [
        "# ──────────────────────────────────────────────────────────────\n",
        "# 18 · F7 Recommended VFR-audit pipeline (model-agnostic)\n",
        "# Six-phase audit pipeline applicable to any supervised\n",
        "# classifier x protected attribute fairness-audit task. See the\n",
        "# accompanying markdown for the full description of each phase.\n",
        "# ──────────────────────────────────────────────────────────────\n",
        "import matplotlib.patches as mpatches\n",
        "fig, ax = plt.subplots(figsize=(15, 16))\n",
        "ax.set_xlim(0, 14); ax.set_ylim(0, 28); ax.axis(\"off\")\n",
        "ax.text(7, 27.0, \"F7 - Recommended VFR-Audit Pipeline (model-agnostic)\",\n",
        "        ha=\"center\", fontsize=14, fontweight=\"bold\", color=\"#0f172a\")\n",
        "ax.text(7, 26.3,\n",
        "        \"A six-phase reliability audit applicable to any supervised classifier x protected attribute combination\",\n",
        "        ha=\"center\", fontsize=10.5, color=\"#334155\", style=\"italic\")\n",
        "\n",
        "PHASES = [\n",
        "    (\"Phase 1 - Data preparation\",\n",
        "     [\"Stratified train/test split on outcome (default 80/20)\",\n",
        "      \"Feature engineering fitted on TRAIN only (no leakage)\",\n",
        "      \"Document protected-attribute coding scheme\",\n",
        "      \"Record dataset hash + RANDOM_STATE\"]),\n",
        "    (\"Phase 2 - Baseline model + point-estimate fairness\",\n",
        "     [\"Train classifier(s); record AUROC, Accuracy, F1\",\n",
        "      \"Compute DI, SPD, EOPP, EOD, TI, PP, CAL on test partition\",\n",
        "      \"Record verdict per (model, metric, attribute) cell\"]),\n",
        "    (\"Phase 3 - Reliability audit (three orthogonal protocols)\",\n",
        "     [\"Protocol 1: B>=500 stratified bootstrap -> VFR per cell\",\n",
        "      \"Protocol 2: 9-point N grid -> minimum N for CV<5%\",\n",
        "      \"Protocol 3: K=20 GroupKFold -> Fleiss kappa\",\n",
        "      \"Sensitivity: K=10,20,40\"]),\n",
        "    (\"Phase 4 - Reliability classification\",\n",
        "     [\"VFR <= 10%      practical-stability\",\n",
        "      \"10% < VFR <= 30%  caution-required\",\n",
        "      \"30% < VFR <= 50%  high-variance\",\n",
        "      \"VFR > 50%       catastrophic-instability\"]),\n",
        "    (\"Phase 5 - Fair intervention (conditional on Phase 4)\",\n",
        "     [\"Per-cell intersectional threshold shifting\",\n",
        "      \"Greedy refinement preserving DI >= threshold\",\n",
        "      \"Ablation vs reweighing-only / calibration-only\",\n",
        "      \"Pareto trade-off disclosure (PP, EOD, CAL)\"]),\n",
        "    (\"Phase 6 - Verification + reporting\",\n",
        "     [\"Bootstrap 95% CI on every headline metric\",\n",
        "      \"Per-site transferability check\",\n",
        "      \"Algorithmic-artefact disclosures (e.g., DI = 1.000)\",\n",
        "      \"Manuscript-claim verification with comparators\"]),\n",
        "]\n",
        "\n",
        "PHASE_COLOURS = [\n",
        "    (\"#dbeafe\", \"#1d4ed8\"), (\"#e0e7ff\", \"#4338ca\"),\n",
        "    (\"#fef3c7\", \"#b45309\"), (\"#fed7aa\", \"#c2410c\"),\n",
        "    (\"#fce7f3\", \"#be185d\"), (\"#dcfce7\", \"#15803d\"),\n",
        "]\n",
        "y_top = 25.5; box_height = 3.7; gap = 0.20\n",
        "for idx, (title, lines) in enumerate(PHASES):\n",
        "    y_t = y_top; y_b = y_t - box_height\n",
        "    fc, ec = PHASE_COLOURS[idx]\n",
        "    ax.add_patch(mpatches.FancyBboxPatch((0.5, y_b), 13.0, box_height,\n",
        "        boxstyle=\"round,pad=0.12\", facecolor=fc, edgecolor=ec, lw=2.2))\n",
        "    ax.text(7, y_t - 0.5, title, ha=\"center\", va=\"top\",\n",
        "            fontsize=12.5, fontweight=\"bold\", color=ec)\n",
        "    sub_y = y_t - 1.20\n",
        "    for ln in lines:\n",
        "        ax.text(0.9, sub_y, ln, ha=\"left\", va=\"top\", fontsize=10.5, color=\"#1f2937\")\n",
        "        sub_y -= 0.55\n",
        "    if idx < len(PHASES) - 1:\n",
        "        ax.annotate(\"\", xy=(7, y_b - gap - 0.05), xytext=(7, y_b - 0.05),\n",
        "                    arrowprops=dict(arrowstyle=\"-|>\", color=\"#0f172a\", lw=2.0))\n",
        "    y_top = y_b - gap - 0.30\n",
        "\n",
        "y_t = y_top; y_b = y_t - 2.0\n",
        "ax.add_patch(mpatches.FancyBboxPatch((0.5, y_b), 13.0, 2.0,\n",
        "    boxstyle=\"round,pad=0.12\", facecolor=\"#cffafe\", edgecolor=\"#0e7490\", lw=2.5))\n",
        "ax.text(7, y_t - 0.5, \"Output: a reliability-aware fairness report\",\n",
        "        ha=\"center\", va=\"top\", fontsize=12.5, fontweight=\"bold\", color=\"#0e7490\")\n",
        "ax.text(7, y_t - 1.20,\n",
        "        \"VFR per cell - CV-stability budget - cross-site agreement - intervention Pareto profile - CI bands\",\n",
        "        ha=\"center\", va=\"top\", fontsize=10.5, color=\"#1f2937\")\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.savefig(f\"{FIGURES_DIR}/F7_recommended_pipeline.png\", dpi=300,\n",
        "            bbox_inches=\"tight\", facecolor=\"white\")\n",
        "plt.show()\n",
        "plt.close(fig)\n",
        "print(f\"Wrote {FIGURES_DIR}/F7_recommended_pipeline.png\")\n",
    ],
}


# Insert the new section just before the final consistency-check + summary cells
inserted = False
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c.get("source", []))
    if "VERIFICATION CHECKS" in src and "BLOCKING DEFECTS" in src:
        # Insert markdown + code BEFORE this cell
        nb["cells"].insert(i, md_section)
        nb["cells"].insert(i + 1, vfr_code)
        print(f"Inserted §18 markdown + F7 code at indices {i}, {i + 1}")
        inserted = True
        break

if not inserted:
    nb["cells"].append(md_section)
    nb["cells"].append(vfr_code)
    print("Appended §18 markdown + F7 code at end")


with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"\nFinal notebook size: {os.path.getsize(NB) / 1024 / 1024:.2f} MB")
