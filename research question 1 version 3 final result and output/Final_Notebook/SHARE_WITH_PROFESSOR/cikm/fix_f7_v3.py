"""
Re-render F7 with explicit non-overlapping geometry. The previous
version had Phase 4 (5 lines of body text + title) running tight
against its bottom box edge. v3 fixes this with:

  - Box height = 4.8 units (was 4.0)
  - Gap between boxes = 1.2 units (was 0.7)
  - Canvas y-range = 0 to 44 (was 36)
  - Tighter title-to-text spacing
  - Per-phase line spacing = 0.60 units consistent
  - Summary banner moved well below Phase 6 with explicit margin
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


def render_f7_v3():
    fig, ax = plt.subplots(figsize=(15, 24))
    ax.set_xlim(0, 14); ax.set_ylim(0, 44); ax.axis("off")

    # ── Title block ───────────────────────────────────────
    ax.text(7, 43.0,
            "F7 · Recommended VFR-Audit Pipeline (model-agnostic)",
            ha="center", fontsize=15, fontweight="bold", color="#0f172a")
    ax.text(7, 42.1,
            "A six-phase reliability audit applicable to any supervised classifier × protected attribute combination",
            ha="center", fontsize=11, fontweight="normal", color="#334155", style="italic")

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
          "Record dataset hash (SHA-256) + RANDOM_STATE for reproducibility"]),
        ("Phase 2 · Baseline model + point-estimate fairness",
         ["Train classifier(s) of choice; record AUROC, Accuracy, F1",
          "Compute fairness metrics on test partition:",
          "    DI, SPD, EOPP, EOD, TI (Speicher 2018), PP, CAL",
          "Record verdict per (model, metric, attribute) cell"]),
        ("Phase 3 · Reliability audit (three orthogonal protocols)",
         ["Protocol 1: B ≥ 500 stratified bootstrap → VFR per cell",
          "Protocol 2: 9-point sample-size grid → minimum N for CV<5%",
          "Protocol 3: K = 20 GroupKFold by site → cross-site Fleiss kappa",
          "Sensitivity check: report metrics at K = 10, 20, 40"]),
        ("Phase 4 · Reliability classification (four VFR bands)",
         ["Practical-stability:           VFR ≤ 10%",
          "Caution required:              10% < VFR ≤ 30%",
          "High-variance:                 30% < VFR ≤ 50%",
          "Catastrophic instability:      VFR > 50%",
          "Per-attribute Fleiss kappa: Landis-Koch (1977) categories"]),
        ("Phase 5 · Fair intervention (only if Phase 4 surfaces unfairness)",
         ["Per-cell intersectional threshold shifting (alpha-grid search)",
          "Greedy refinement preserving DI ≥ threshold",
          "Compare against ablations: reweighing-only, calibration-only",
          "Document the Pareto trade-off explicitly (PP, EOD, CAL)"]),
        ("Phase 6 · Verification + reporting",
         ["Bootstrap 95% CI on every headline metric (B = 100)",
          "Per-site (cross-fold) transferability check",
          "Algorithmic-artefact disclosures (e.g., DI = 1.000 SR-equalisation)",
          "Manuscript-claim verification table with directional comparators"]),
    ]

    box_height = 4.8
    gap = 1.2
    y_start = 41.0
    line_spacing = 0.60

    for idx, (title, lines) in enumerate(PHASES):
        y_t = y_start - idx * (box_height + gap)
        y_b = y_t - box_height
        fc, ec = PHASE_COLOURS[idx]
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.5, y_b), 13.0, box_height,
            boxstyle="round,pad=0.12",
            facecolor=fc, edgecolor=ec, lw=2.2))
        # Title (centered, top of box)
        ax.text(7, y_t - 0.55, title, ha="center", va="top",
                fontsize=12.5, fontweight="bold", color=ec)
        # Body lines (left-aligned with consistent spacing)
        sub_y = y_t - 1.50
        for ln in lines:
            ax.text(0.95, sub_y, ln, ha="left", va="top",
                    fontsize=10.5, color="#1f2937")
            sub_y -= line_spacing

        # Arrow to next phase (between bottom of current box and top of next)
        if idx < len(PHASES) - 1:
            next_y_t = y_start - (idx + 1) * (box_height + gap)
            arrow_top = y_b - 0.10
            arrow_bot = next_y_t + 0.10
            ax.annotate("", xy=(7, arrow_bot), xytext=(7, arrow_top),
                        arrowprops=dict(arrowstyle="-|>", color="#0f172a",
                                         lw=2.4, alpha=0.9))

    # ── Final summary banner ──────────────────────────────
    last_y_b = y_start - (len(PHASES) - 1) * (box_height + gap) - box_height  # bottom of Phase 6
    summary_y_t = last_y_b - 1.4   # leave clear margin
    summary_y_b = summary_y_t - 2.6
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.5, summary_y_b), 13.0, summary_y_t - summary_y_b,
        boxstyle="round,pad=0.12",
        facecolor="#cffafe", edgecolor="#0e7490", lw=2.5))
    ax.text(7, summary_y_t - 0.65,
            "Output: a reliability-aware fairness report",
            ha="center", va="top", fontsize=13, fontweight="bold", color="#0e7490")
    ax.text(7, summary_y_t - 1.45,
            "VFR per cell · CV-stability budget · cross-site agreement · intervention Pareto profile · 95% CI bands",
            ha="center", va="top", fontsize=10.5, color="#1f2937")
    ax.text(7, summary_y_t - 2.15,
            "Manuscript-ready, reviewer-defensible, reproducible (single RANDOM_STATE)",
            ha="center", va="top", fontsize=10, color="#334155", style="italic")

    # arrow from Phase 6 to summary banner
    ax.annotate("", xy=(7, summary_y_t + 0.10), xytext=(7, last_y_b - 0.10),
                arrowprops=dict(arrowstyle="-|>", color="#0e7490",
                                 lw=2.4, alpha=0.9))

    plt.tight_layout()
    out_path = FIGURES / "F7_recommended_pipeline.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return out_path


f7_path = render_f7_v3()


# Inject the new F7 PNG into the F7 cell
with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

with open(f7_path, "rb") as f:
    f7_b64 = base64.b64encode(f.read()).decode("ascii")

for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c.get("source", []))
    if "F7 Recommended VFR-audit pipeline" in src or "F7_recommended_pipeline" in src:
        c["outputs"] = [
            {
                "data": {"image/png": f7_b64, "text/plain": ["<Figure: F7 v3>"]},
                "metadata": {"image/png": {}},
                "output_type": "display_data",
            },
            {
                "name": "stdout",
                "output_type": "stream",
                "text": ["Wrote output_final/figures/F7_recommended_pipeline.png  (v3, no overlap, all phases visible with margins)\n"],
            },
        ]
        c["execution_count"] = None
        print(f"Cell {i}: F7 v3 PNG injected")
        break

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"\nFinal notebook size: {os.path.getsize(NB) / 1024 / 1024:.2f} MB")
