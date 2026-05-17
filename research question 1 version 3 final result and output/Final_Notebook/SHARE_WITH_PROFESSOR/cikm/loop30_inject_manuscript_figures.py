"""
Loop 30: inject the four manuscript figures (F2 cohort distribution, F3 VFR
heatmap, F4 CV-vs-N curves, F5 hospital-fold violin) into the notebook as a
new appendix section §30 at the end.
"""
import json, sys, io, base64
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm")
NB = ROOT / "CIKM_2026_LOS_Fairness_FINAL.ipynb"
FIG_DIR = ROOT / "output_final" / "figures" / "manuscript"

FIGS = [
    ("F2", "F2_cohort_distribution.png",  "Cohort composition (race / sex × ethnicity / age + LOS rate / per-hospital volume)"),
    ("F3", "F3_vfr_heatmap.png",          "Verdict-Flip-Rate heatmap on Configuration 4 (Real+VFR canonical)"),
    ("F4", "F4_cv_curves.png",            "CV-vs-N curves on standard XGBoost predictions"),
    ("F5", "F5_hospital_violin.png",      "Per-metric distribution across K=20 hospital folds × 4 attributes"),
]

# Markdown narrative
MD_30 = """---
## 30 · Manuscript figure suite (F2 / F3 / F4 / F5)

This appendix bundles the four manuscript-ready figures referenced in the paper draft. They are pre-rendered to `output_final/figures/manuscript/` by `compute_manuscript_figures.py` (F2, F3, F5) and `fix_f4_cv_curves.py` (F4 with bootstrap-with-replacement at every N to avoid the degenerate CV → 0 collapse at full test size).

- **F2** — four-panel cohort composition. Panel (c) reveals the dominant base-rate driver: the LOS-positive rate triples between Young Adult (20.7 %) and Elderly (60.6 %), fixing a structural lower bound on Age-DI that no reweighing scheme can cross.
- **F3** — 7 × 4 VFR heatmap on Configuration 4 (Real+VFR canonical). Race-axis cells dominate the high-instability quadrant; ethnicity-axis cells are uniformly stable.
- **F4** — coefficient-of-variation curves across the 8-point audit-size grid. Calibration and Equal Opportunity for race / ethnicity require the full test partition; sex-axis metrics stabilise at N ≤ 10,000.
- **F5** — per-hospital-fold distribution of metric values under K = 20 GroupKFold. Equal Opportunity and Equalised Odds show tight inter-fold agreement; Disparate Impact and Calibration show wide inter-fold spread.
"""

# Build the 5-cell sequence: 1 markdown narrative + 4 code cells (one per figure)
def make_fig_cell(label, png_filename, caption):
    full_path = FIG_DIR / png_filename
    with open(full_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode('ascii')
    src = (
        f"# ──────────────────────────────────────────────────────────────\n"
        f"# §30 · {label} · {caption}\n"
        f"# ──────────────────────────────────────────────────────────────\n"
        f"from IPython.display import Image, display, HTML\n"
        f"display(HTML('<h4>{label} · {caption}</h4>'))\n"
        f"display(Image(filename='output_final/figures/manuscript/{png_filename}'))\n"
    )
    outputs = [
        {"data": {"text/html": [f"<h4>{label} · {caption}</h4>"], "text/plain": ["<HTML>"]},
         "metadata": {}, "output_type": "display_data"},
        {"data": {"image/png": b64, "text/plain": [f"<{label}>"]}, "metadata": {}, "output_type": "display_data"},
    ]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"_cell_marker": f"manuscript_fig_{label}_marker"},
        "source": src.splitlines(keepends=True),
        "outputs": outputs,
    }

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Build new cells in order: markdown narrative + 4 figure cells
new_cells = [
    {"cell_type": "markdown",
     "metadata": {"_cell_marker": "manuscript_figs_narrative_marker"},
     "source": MD_30.splitlines(keepends=True)}
]
for label, fn, cap in FIGS:
    new_cells.append(make_fig_cell(label, fn, cap))

# Idempotent: replace existing block if already injected
existing_indices = []
markers_to_replace = ["manuscript_figs_narrative_marker"] + [f"manuscript_fig_F{i}_marker" for i in [2,3,4,5]]
for i, c in enumerate(nb['cells']):
    if c.get('metadata', {}).get('_cell_marker') in markers_to_replace:
        existing_indices.append(i)

if existing_indices:
    # Remove existing block (in reverse to preserve indices)
    for i in sorted(existing_indices, reverse=True):
        del nb['cells'][i]
    print(f"removed {len(existing_indices)} existing manuscript-figure cells")

# Append new block at end
nb['cells'].extend(new_cells)
print(f"appended {len(new_cells)} cells (1 markdown + 4 figures)")

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"\nFinal notebook: {NB.stat().st_size / 1024 / 1024:.2f} MB, {len(nb['cells'])} cells")

# Print figure sizes
total_kb = 0
for label, fn, _ in FIGS:
    sz = (FIG_DIR / fn).stat().st_size / 1024
    total_kb += sz
    print(f"  {label}: {sz:.0f} KB ({fn})")
print(f"  total embedded PNG: {total_kb:.0f} KB")
