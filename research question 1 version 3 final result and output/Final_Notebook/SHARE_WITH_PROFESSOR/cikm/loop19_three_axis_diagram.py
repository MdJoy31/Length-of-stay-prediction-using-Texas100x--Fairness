"""
Loop 19: append a new code cell at the end of the notebook that renders
a half-page 3-axis evaluation-framework diagram (F9).

The three axes are:
  Axis 1 - Resampling stability   (within-cohort, B=500 bootstrap, VFR)
  Axis 2 - Sample-size sensitivity (9-point N grid, CV<5%)
  Axis 3 - Cross-hospital portability (K=20 GroupKFold, Fleiss kappa)

A central hub represents "Reliability-aware fairness audit (336 cells)".
Each axis arm carries: name, protocol, key statistic, headline finding.

Cell is appended as the new last cell. PNG saved to
output_final/figures/F9_three_axis_framework.png and base64-embedded.
"""
import json, sys, io, base64
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm")
NB = ROOT / "CIKM_2026_LOS_Fairness_FINAL.ipynb"
F9_PNG = ROOT / "output_final" / "figures" / "F9_three_axis_framework.png"

# --- Render F9 ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

fig, ax = plt.subplots(figsize=(10, 6.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off")

# Title
ax.text(5, 5.95, "F9 · Three-Axis Reliability-Aware Fairness Audit Framework",
        ha="center", fontsize=12.5, fontweight="bold", color="#0f172a")
ax.text(5, 5.62, "Each (model, metric, attribute) cell is evaluated on three orthogonal reliability axes",
        ha="center", fontsize=9.5, style="italic", color="#475569")

# Central hub
hub_x, hub_y = 5.0, 3.05
ax.add_patch(Circle((hub_x, hub_y), 0.85, facecolor="#fef3c7", edgecolor="#b45309", lw=2.5, zorder=3))
ax.text(hub_x, hub_y + 0.18, "Reliability-aware",
        ha="center", va="center", fontsize=10, fontweight="bold", color="#b45309")
ax.text(hub_x, hub_y - 0.05, "fairness audit",
        ha="center", va="center", fontsize=10, fontweight="bold", color="#b45309")
ax.text(hub_x, hub_y - 0.30, "336 cells",
        ha="center", va="center", fontsize=8.5, color="#475569")

# Three axis boxes - positioned around the hub
# Axis 1 (top) - Resampling stability
A1_C = ("#dbeafe", "#1d4ed8")
A1_BOX = (1.5, 4.4, 4.3, 1.2)  # x, y, w, h
ax.add_patch(FancyBboxPatch((A1_BOX[0], A1_BOX[1]), A1_BOX[2], A1_BOX[3],
                             boxstyle="round,pad=0.12",
                             facecolor=A1_C[0], edgecolor=A1_C[1], lw=2, zorder=2))
ax.text(A1_BOX[0] + A1_BOX[2]/2, A1_BOX[1] + A1_BOX[3] - 0.25,
        "Axis 1  ·  Resampling stability",
        ha="center", va="top", fontsize=10.5, fontweight="bold", color=A1_C[1])
ax.text(A1_BOX[0] + 0.12, A1_BOX[1] + A1_BOX[3] - 0.62,
        "Protocol: B = 500 stratified bootstrap, N = 10,000 per resample",
        ha="left", va="top", fontsize=8.5, color="#1f2937")
ax.text(A1_BOX[0] + 0.12, A1_BOX[1] + A1_BOX[3] - 0.82,
        "Statistic: VFR per cell + stability margin (sigma units)",
        ha="left", va="top", fontsize=8.5, color="#1f2937")
ax.text(A1_BOX[0] + 0.12, A1_BOX[1] + A1_BOX[3] - 1.02,
        "Finding: 43.5 % of 336 cells flipped (peak 47.4 %)",
        ha="left", va="top", fontsize=8.5, fontweight="bold", color="#dc2626")

# Axis 2 (bottom-left) - Sample-size sensitivity
A2_C = ("#dcfce7", "#15803d")
A2_BOX = (0.25, 0.55, 4.3, 1.55)
ax.add_patch(FancyBboxPatch((A2_BOX[0], A2_BOX[1]), A2_BOX[2], A2_BOX[3],
                             boxstyle="round,pad=0.12",
                             facecolor=A2_C[0], edgecolor=A2_C[1], lw=2, zorder=2))
ax.text(A2_BOX[0] + A2_BOX[2]/2, A2_BOX[1] + A2_BOX[3] - 0.25,
        "Axis 2  ·  Sample-size sensitivity",
        ha="center", va="top", fontsize=10.5, fontweight="bold", color=A2_C[1])
ax.text(A2_BOX[0] + 0.12, A2_BOX[1] + A2_BOX[3] - 0.62,
        "Protocol: 9-point N grid {1k ... 185k}",
        ha="left", va="top", fontsize=8.5, color="#1f2937")
ax.text(A2_BOX[0] + 0.12, A2_BOX[1] + A2_BOX[3] - 0.82,
        "Statistic: minimum N for CV < 5 % per cell",
        ha="left", va="top", fontsize=8.5, color="#1f2937")
ax.text(A2_BOX[0] + 0.12, A2_BOX[1] + A2_BOX[3] - 1.02,
        "Finding: 9 of 28 cells require N = 185,026",
        ha="left", va="top", fontsize=8.5, fontweight="bold", color="#dc2626")
ax.text(A2_BOX[0] + 0.12, A2_BOX[1] + A2_BOX[3] - 1.22,
        "(noisy metrics: EOPP, EOD, PP, CAL, TI)",
        ha="left", va="top", fontsize=8, style="italic", color="#475569")

# Axis 3 (bottom-right) - Cross-hospital portability
A3_C = ("#fce7f3", "#be185d")
A3_BOX = (5.45, 0.55, 4.3, 1.55)
ax.add_patch(FancyBboxPatch((A3_BOX[0], A3_BOX[1]), A3_BOX[2], A3_BOX[3],
                             boxstyle="round,pad=0.12",
                             facecolor=A3_C[0], edgecolor=A3_C[1], lw=2, zorder=2))
ax.text(A3_BOX[0] + A3_BOX[2]/2, A3_BOX[1] + A3_BOX[3] - 0.25,
        "Axis 3  ·  Cross-hospital portability",
        ha="center", va="top", fontsize=10.5, fontweight="bold", color=A3_C[1])
ax.text(A3_BOX[0] + 0.12, A3_BOX[1] + A3_BOX[3] - 0.62,
        "Protocol: K = 20 GroupKFold by THCIC_ID",
        ha="left", va="top", fontsize=8.5, color="#1f2937")
ax.text(A3_BOX[0] + 0.12, A3_BOX[1] + A3_BOX[3] - 0.82,
        "Statistic: Fleiss kappa (Landis-Koch 1977)",
        ha="left", va="top", fontsize=8.5, color="#1f2937")
ax.text(A3_BOX[0] + 0.12, A3_BOX[1] + A3_BOX[3] - 1.02,
        "Finding: overall kappa = 0.506 (moderate);",
        ha="left", va="top", fontsize=8.5, fontweight="bold", color="#dc2626")
ax.text(A3_BOX[0] + 0.12, A3_BOX[1] + A3_BOX[3] - 1.22,
        "EOPP = 0.674, EOD = 0.601, CAL = 0.016",
        ha="left", va="top", fontsize=8.5, fontweight="bold", color="#dc2626")

# Arrows from hub to each axis box
for (bx, by, bw, bh, color) in [
    (A1_BOX[0]+A1_BOX[2]/2, A1_BOX[1]+0.05, A1_BOX[2], A1_BOX[3], A1_C[1]),  # to Axis 1
    (A2_BOX[0]+A2_BOX[2]-0.3, A2_BOX[1]+A2_BOX[3]-0.05, A2_BOX[2], A2_BOX[3], A2_C[1]),  # to Axis 2
    (A3_BOX[0]+0.3, A3_BOX[1]+A3_BOX[3]-0.05, A3_BOX[2], A3_BOX[3], A3_C[1]),  # to Axis 3
]:
    arr = FancyArrowPatch((hub_x, hub_y), (bx, by),
                           arrowstyle="-|>", mutation_scale=14, color=color,
                           lw=1.6, alpha=0.85, zorder=1)
    ax.add_patch(arr)

# Output line
ax.text(5, 0.22,
        "Output:  per-cell reliability tier (Practical / Caution / High-variance / Catastrophic)",
        ha="center", fontsize=9.5, fontweight="bold", color="#0e7490",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#cffafe",
                  edgecolor="#0e7490", lw=1.5))

plt.tight_layout()
plt.savefig(F9_PNG, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Wrote F9 PNG: {F9_PNG.stat().st_size / 1024:.0f} KB")

# --- Build new code cell ---
cell_src = (
    "# ──────────────────────────────────────────────────────────────\n"
    "# §25 · F9 · Three-Axis Reliability-Aware Fairness Audit Framework\n"
    "# Half-page diagram visualising the three reliability axes (resampling\n"
    "# stability, sample-size sensitivity, cross-hospital portability)\n"
    "# that frame this study's contribution. Pre-rendered to F9_three_axis_framework.png.\n"
    "# ──────────────────────────────────────────────────────────────\n"
    "from IPython.display import Image, display\n"
    "display(Image(filename='output_final/figures/F9_three_axis_framework.png'))\n"
)

# Read PNG and base64-encode for the embedded output
with open(F9_PNG, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode('ascii')

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "source": cell_src.splitlines(keepends=True),
    "outputs": [
        {
            "data": {
                "image/png": img_b64,
                "text/plain": ["<IPython.core.display.Image object>"]
            },
            "metadata": {},
            "output_type": "display_data"
        }
    ]
}

# Append to notebook
with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Check idempotency: if last cell already has F9 marker, replace it instead
if nb['cells']:
    last_src = ''.join(nb['cells'][-1].get('source', []))
    if 'F9 · Three-Axis Reliability-Aware Fairness Audit' in last_src:
        nb['cells'][-1] = new_cell
        print("Replaced existing F9 cell at end")
    else:
        nb['cells'].append(new_cell)
        print(f"Appended new F9 cell at index {len(nb['cells']) - 1}")
else:
    nb['cells'].append(new_cell)

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"\nFinal notebook: {NB.stat().st_size / 1024 / 1024:.2f} MB, {len(nb['cells'])} cells")
