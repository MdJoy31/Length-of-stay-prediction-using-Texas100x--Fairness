"""
Render the new F1 end-to-end pipeline figure standalone (no kernel
required), then inject the resulting PNG into cell 49 of the notebook.
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
FIGURES.mkdir(parents=True, exist_ok=True)


def render_f1():
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_xlim(0, 16); ax.set_ylim(0, 22); ax.axis("off")

    ax.text(8, 21.1,
            "End-to-End Pipeline · Data ingestion to reliability audit to fair-intervention to verification",
            ha="center", fontsize=14, fontweight="bold", color="#0f172a")

    COLOURS = {
        "data":    ("#dbeafe", "#1d4ed8"),
        "preproc": ("#e0e7ff", "#4338ca"),
        "model":   ("#ede9fe", "#6d28d9"),
        "audit":   ("#fef3c7", "#b45309"),
        "interv":  ("#fce7f3", "#be185d"),
        "eval":    ("#dcfce7", "#15803d"),
        "verify":  ("#cffafe", "#0e7490"),
    }

    def stage(y_top, y_bottom, x_left, x_right, kind, title, sub_lines):
        fc, ec = COLOURS[kind]
        ax.add_patch(mpatches.FancyBboxPatch(
            (x_left, y_bottom), x_right - x_left, y_top - y_bottom,
            boxstyle="round,pad=0.10",
            facecolor=fc, edgecolor=ec, lw=2))
        ax.text((x_left + x_right) / 2, y_top - 0.35, title,
                ha="center", va="top", fontsize=11.5, fontweight="bold", color=ec)
        sub_y = y_top - 0.85
        for ln in sub_lines:
            ax.text((x_left + x_right) / 2, sub_y, ln, ha="center", va="top",
                    fontsize=9.5, color="#1f2937")
            sub_y -= 0.42

    def arrow(x_from, y_from, x_to, y_to, colour="#0f172a"):
        ax.annotate("", xy=(x_to, y_to), xytext=(x_from, y_from),
                    arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.6, alpha=0.85))

    stage(20.5, 19.2, 0.5, 15.5, "data",
          "1 · Data ingestion · THCIC PUDF · texas_100x.csv (cell 4)",
          ["N = 925,128 inpatient discharges from 441 Texas hospitals (FY 2019-2023)",
           "Hash-pinned (SHA-256). RANDOM_STATE = 42."])
    stage(18.9, 16.3, 0.5, 7.7, "preproc",
          "2 · EDA + diagnostics (cell 6, 8)",
          ["Diag 1: duplication ratio 1.01 -> real, not augmented",
           "Diag 2: RACE x ETHNICITY crosstab -> 99.4% Hispanic among RACE=2",
           "Diag 3: top-10 LOS coverage 89.3% -> clinical clustering",
           "T3 cohort table (race re-mapped)"])
    stage(18.9, 16.3, 8.3, 15.5, "preproc",
          "3 · Feature engineering (cell 11)",
          ["80/20 stratified split (random_state=42, target = LOS > 3 days)",
           "Bayesian target encoding (m=10) on TRAIN only - no leakage",
           "5 numeric + 3 TE + 3 interaction features = 14 total",
           "T_HYPERPARAMS reference table"])
    stage(16.0, 13.4, 0.5, 15.5, "model",
          "4 · Model training (cell 15) · 12 classifiers · XGBoost canonical",
          ["LR | DT | RF | GB | AdaBoost | XGBoost | LightGBM | CatBoost | HistGB | Bagging | ExtraTrees | Stacking",
           "Canonical XGBoost: AUROC = 0.9528, Accuracy = 0.8776, F1 = 0.8627",
           "All 12 with n_jobs=1 (memory-safe). 7 metrics x 4 attrs = 28 cells per model."])

    ax.text(8, 12.85, "5 · Reliability audit · three protocols",
            ha="center", fontsize=12, fontweight="bold", color="#b45309")

    stage(12.5, 9.8, 0.5, 5.4, "audit",
          "Protocol 1 · Verdict Flip Rate",
          ["B=500 stratified bootstrap on test partition",
           "12 models x 7 metrics x 4 attrs = 336 cells",
           "Output: T7, T8 (max VFR 47.4%, 43.5% flipped)"])
    stage(12.5, 9.8, 5.7, 10.3, "audit",
          "Protocol 2 · Sample-size sensitivity",
          ["9-point N grid (5k, 10k, 25k, 50k, 100k, 185k)",
           "Min-N for CV < 5% per (metric, attribute) cell",
           "Output: T9 (9 of 28 cells need N=185k)"])
    stage(12.5, 9.8, 10.6, 15.5, "audit",
          "Protocol 3 · Cross-hospital portability",
          ["K=20 GroupKFold by hospital_id (no patient overlap)",
           "Per-cluster Fleiss kappa + cross-fold CV",
           "Output: T10 (17/28 CV>0.50), T11 (kappa=0.506)"])
    stage(9.5, 8.2, 0.5, 15.5, "audit",
          "Protocol 3-bis · K-sensitivity (cell 44)",
          ["Real GroupKFold at K=10, K=20, K=40 (canonical = K=20)",
           "Five of seven metrics' agreement classifications change with K because per-fold sample size shrinks",
           "(K=10 -> 92k/fold; K=20 -> 46k/fold; K=40 -> 23k/fold). T17 reports the full sensitivity matrix."])

    ax.text(8, 7.65, "6 · Fair-intervention pipeline · per-cell threshold-shifting",
            ha="center", fontsize=12, fontweight="bold", color="#be185d")

    stage(7.3, 4.6, 0.5, 7.9, "interv",
          "Intervention search (cells 32, 34, 35)",
          ["T13 lambda-sweep: 0/10 reweighing values achieve all-4-DI (ablation)",
           "alpha-grid threshold search per (RACE x AGE x SEX) cell",
           "Phase 5/5b/6 greedy refinement (DI>=0.80 + min PP/EOD trade-off)",
           "Phase 7 (per-cell isotonic calibration) tested -> REJECTED (CAL +0.035)"])
    stage(7.3, 4.6, 8.1, 15.5, "interv",
          "Canonical: Phase 5b (lambda=0 + threshold + greedy)",
          ["Standard XGB -> alpha-search -> greedy refinement",
           "T14 ablation chain (6 configs); T15 Standard vs Fair (32 metrics)",
           "AUROC preserved 0.9528 = 0.9528. Accuracy 0.8776 -> 0.8347 (cost 4.29 pp)",
           "All 4 DI >= 0.80 jointly: True (Race 0.80 / Sex 0.93 / Eth 1.00 / Age 0.80)"])

    stage(4.3, 1.6, 0.5, 7.9, "eval",
          "7 · Evaluation (cells 36, 38, 41)",
          ["T15 standard vs fair point estimates",
           "T15_with_CI: B=100 bootstrap 95% CIs on headline metrics",
           "T16 per-cluster transferability: 19/20 worst-DI improved,",
           "14/20 all-4-DI achieved, 16/20 within 5pp"])
    stage(4.3, 1.6, 8.1, 15.5, "verify",
          "8 · Verification (cells 57, 60, 61)",
          ["T19 manuscript-claim verification: 22/22 PASS",
           "T20 unanimous-fair (model, attr) cells: 12/48",
           "Cell 60 consistency check: 22/22 PASS",
           "F6 Pareto-frontier comparison (Phase 5b vs Phase 7)"])

    stage(1.0, 0.05, 0.5, 15.5, "verify",
          "Final state: notebook is manuscript-ready · all anchors PASS · Pareto trade-off disclosed",
          [])

    arrow(8, 19.15, 8, 18.95)
    arrow(4, 16.25, 4, 16.05)
    arrow(11.9, 16.25, 11.9, 16.05)
    arrow(8, 13.35, 8, 13.0)
    arrow(2.95, 9.75, 2.95, 9.55)
    arrow(8, 9.75, 8, 9.55)
    arrow(13.05, 9.75, 13.05, 9.55)
    arrow(8, 8.15, 8, 7.85)
    arrow(4.2, 4.55, 4.2, 4.35)
    arrow(11.8, 4.55, 11.8, 4.35)
    arrow(4.2, 1.55, 4.2, 1.05)
    arrow(11.8, 1.55, 11.8, 1.05)

    plt.tight_layout()
    out_path = FIGURES / "F1_reliability_framework.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return out_path


png_path = render_f1()


# Inject the PNG into cell 49 of the notebook
with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

with open(png_path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("ascii")

png_output = {
    "data": {"image/png": b64, "text/plain": ["<Figure: F1 pipeline>"]},
    "metadata": {"image/png": {}},
    "output_type": "display_data",
}
stream_output = {
    "name": "stdout",
    "output_type": "stream",
    "text": [f"Wrote output_final/figures/F1_reliability_framework.png  (full pipeline diagram)\n"],
}

for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c.get("source", []))
    if "F1 End-to-end pipeline" in src or ("F1_reliability_framework.png" in src and "stage(" in src):
        c["outputs"] = [png_output, stream_output]
        c["execution_count"] = None
        print(f"Cell {i}: F1 PNG injected as cell output")
        break

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Final notebook size: {os.path.getsize(NB) / 1024 / 1024:.2f} MB")
