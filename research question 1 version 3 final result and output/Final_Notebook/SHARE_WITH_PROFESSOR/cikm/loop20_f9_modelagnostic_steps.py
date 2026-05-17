"""
Loop 20: redesign F9 as a TRUE model-agnostic 3-axis framework.

Drops all project-specific numbers (925k records, 43.5%, kappa=0.506, etc).
Each axis becomes a column listing the numbered pipeline steps a user follows.
A combined tier-output box at the bottom shows how the three axes feed into
the per-cell reliability classification.

Title: F9 - Three-Axis Reliability-Aware Fairness Framework (model-agnostic)
Half-page size: figsize=(11, 6.5)
"""
import json, sys, io, base64
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm")
NB = ROOT / "CIKM_2026_LOS_Fairness_FINAL.ipynb"
F9_PNG = ROOT / "output_final" / "figures" / "F9_three_axis_framework.png"

# --- Render F9 (redesigned) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(11, 6.5))
ax.set_xlim(0, 11); ax.set_ylim(0, 6.5); ax.axis("off")

# Title and subtitle
ax.text(5.5, 6.25,
        "F9 · Three-Axis Reliability-Aware Fairness Framework (model-agnostic)",
        ha="center", fontsize=12.5, fontweight="bold", color="#0f172a")
ax.text(5.5, 5.95,
        "Each (model, metric, attribute) cell is evaluated on three orthogonal reliability axes",
        ha="center", fontsize=9.5, style="italic", color="#475569")

# Three columns
col_w = 3.4
col_gap = 0.25
col_x = [0.30, 0.30 + col_w + col_gap, 0.30 + 2 * (col_w + col_gap)]
col_top = 5.55
col_bottom = 1.20
col_h = col_top - col_bottom

COLS = [
    ("Axis 1  ·  Resampling stability",
     "Within-cohort verdict variance",
     ("#dbeafe", "#1d4ed8"),
     [
         "1.  Stratify test partition by outcome label",
         "2.  Draw  K  bootstrap resamples of size  N",
         "3.  Recompute fairness metric  m  on each",
         "4.  n_pass  =  count satisfying threshold  tau_m",
         "5.  VFR  =  min(n_pass, K - n_pass) / K",
         "6.  Stability margin  sigma  =  (mean - tau) / SD",
     ],
     "Output:  VFR per cell  in  [0, 0.5]"),
    ("Axis 2  ·  Sample-size sensitivity",
     "Minimum reliable audit cohort",
     ("#dcfce7", "#15803d"),
     [
         "1.  Define audit-size grid  N1 < N2 < ... < N_max",
         "2.  For each  N:  draw  R  random subsamples",
         "3.  Compute coefficient of variation  CV(metric, N)",
         "4.  Find smallest  N  with  CV < 5 %  per cell",
         "5.  Flag cells whose min-N exceeds field standard",
     ],
     "Output:  min reliable audit  N  per cell"),
    ("Axis 3  ·  Cross-hospital portability",
     "Cross-site verdict agreement",
     ("#fce7f3", "#be185d"),
     [
         "1.  Group records by site / cluster identifier",
         "2.  K-fold GroupKFold split (no patient overlap)",
         "3.  Per-fold verdict per (metric, attribute) cell",
         "4.  Fleiss kappa across attributes  x  folds",
         "5.  Classify by Landis - Koch (1977) bands",
     ],
     "Output:  per-metric agreement classification"),
]

for i, (title, sub, (fc, ec), steps, out_text) in enumerate(COLS):
    x0 = col_x[i]
    # Column box
    ax.add_patch(FancyBboxPatch((x0, col_bottom), col_w, col_h,
                                 boxstyle="round,pad=0.10",
                                 facecolor=fc, edgecolor=ec, lw=2))
    # Header
    ax.text(x0 + col_w/2, col_top - 0.18, title,
            ha="center", va="top", fontsize=10.5, fontweight="bold", color=ec)
    ax.text(x0 + col_w/2, col_top - 0.50, sub,
            ha="center", va="top", fontsize=8.5, style="italic", color="#334155")
    # Numbered steps
    step_y = col_top - 0.95
    step_dy = 0.42
    for s in steps:
        ax.text(x0 + 0.18, step_y, s, ha="left", va="top",
                fontsize=8.5, color="#0f172a", family="DejaVu Sans")
        step_y -= step_dy
    # Output line at the bottom of the column
    ax.text(x0 + col_w/2, col_bottom + 0.20, out_text,
            ha="center", va="center", fontsize=8.5, fontweight="bold", color=ec)

# Combined tier output box at the bottom
tier_y_top = 0.95
tier_y_bot = 0.18
ax.add_patch(FancyBboxPatch((0.30, tier_y_bot), col_x[2] + col_w - 0.30, tier_y_top - tier_y_bot,
                             boxstyle="round,pad=0.12",
                             facecolor="#fef3c7", edgecolor="#b45309", lw=2.2))
ax.text(5.5, tier_y_top - 0.20,
        "Combined output  ·  per-cell reliability tier",
        ha="center", va="top", fontsize=10, fontweight="bold", color="#b45309")
ax.text(5.5, tier_y_top - 0.50,
        "Practical-stability  (VFR <= 10 %  ·  min-N <= field-standard  ·  kappa >= 0.4 )"
        "      Caution-required      High-variance      Catastrophic-instability",
        ha="center", va="top", fontsize=8.5, color="#1f2937")

# Arrows from each column down to the tier box
for x_arrow in [col_x[0] + col_w/2, col_x[1] + col_w/2, col_x[2] + col_w/2]:
    arr = FancyArrowPatch((x_arrow, col_bottom - 0.04),
                           (x_arrow, tier_y_top + 0.04),
                           arrowstyle="-|>", mutation_scale=12,
                           color="#475569", lw=1.4, alpha=0.9)
    ax.add_patch(arr)

plt.tight_layout()
plt.savefig(F9_PNG, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Wrote F9 PNG: {F9_PNG.stat().st_size / 1024:.0f} KB")

# --- Update the F9 cell in notebook (replace embedded PNG) ---
with open(F9_PNG, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode('ascii')

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find the F9 cell (last cell with §25 marker)
f9_idx = None
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    if 'F9 · Three-Axis Reliability-Aware Fairness' in src and c['cell_type'] == 'code':
        f9_idx = i
if f9_idx is None:
    raise RuntimeError("F9 cell not found")

# Update the source comments to clarify model-agnostic + pipeline-steps framing
new_src = (
    "# ──────────────────────────────────────────────────────────────\n"
    "# §25 · F9 · Three-Axis Reliability-Aware Fairness Framework (model-agnostic)\n"
    "# A half-page protocol diagram. Each axis is presented as a numbered\n"
    "# pipeline so a reviewer or practitioner reading F9 alone learns\n"
    "# exactly what to compute on every (model, metric, attribute) cell.\n"
    "# No dataset-specific numbers appear in this figure - those live in\n"
    "# T7 (resampling), T9 (sample-size), T11/T17 (cross-site).\n"
    "# ──────────────────────────────────────────────────────────────\n"
    "from IPython.display import Image, display\n"
    "display(Image(filename='output_final/figures/F9_three_axis_framework.png'))\n"
)

# Update embedded PNG and source
nb['cells'][f9_idx]['source'] = new_src.splitlines(keepends=True)
for o_idx, o in enumerate(nb['cells'][f9_idx].get('outputs', [])):
    if 'data' in o and 'image/png' in o.get('data', {}):
        nb['cells'][f9_idx]['outputs'][o_idx]['data']['image/png'] = img_b64
        print(f"Cell {f9_idx} output {o_idx}: PNG updated to {len(img_b64) * 3 / 4 / 1024:.0f} KB")

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"\nFinal notebook: {NB.stat().st_size / 1024 / 1024:.2f} MB, {len(nb['cells'])} cells")
