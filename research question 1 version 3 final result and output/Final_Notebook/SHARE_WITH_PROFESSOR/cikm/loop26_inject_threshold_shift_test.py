"""
Loop 26: inject the threshold-shift-only verification test (Configuration 3
of the ablation, λ=0 + α-search, NO greedy refinement) into the notebook
as a new code cell at the end. Cell embeds:
  - T_threshold_shift_only.csv (1-row headline)
  - T_threshold_shift_only_vfr.csv (28-row K=500 VFR per cell)
  - Narrative paragraph explaining the finding
"""
import json, sys, io, base64
from pathlib import Path
import pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm")
NB = ROOT / "CIKM_2026_LOS_Fairness_FINAL.ipynb"

T_HEAD = pd.read_csv(ROOT / "output_final" / "tables" / "T_threshold_shift_only.csv")
T_VFR  = pd.read_csv(ROOT / "output_final" / "tables" / "T_threshold_shift_only_vfr.csv")

cell_src = (
    "# ──────────────────────────────────────────────────────────────\n"
    "# §27 · Configuration 3 verification — Threshold-Shift only (λ=0 + α-search)\n"
    "# Tests whether the canonical Phase 5b greedy-refinement step is necessary\n"
    "# or whether α-search alone achieves a STABLE all-4-DI pass.\n"
    "# Pre-computed by compute_threshold_shift_only.py with K=500 bootstrap VFR\n"
    "# on the full 925,128-record cohort (canonical n_est=1500 XGBoost).\n"
    "# ──────────────────────────────────────────────────────────────\n"
    "import pandas as pd\n"
    "from pathlib import Path\n"
    "from IPython.display import HTML, display\n"
    "T_HEAD = pd.read_csv('output_final/tables/T_threshold_shift_only.csv')\n"
    "T_VFR  = pd.read_csv('output_final/tables/T_threshold_shift_only_vfr.csv')\n"
    "display(HTML('<h4>Configuration (3) Threshold-Shift only · headline</h4>'))\n"
    "display(T_HEAD.T)\n"
    "display(HTML('<h4>K=500 bootstrap VFR per (metric, attribute) cell</h4>'))\n"
    "display(T_VFR)\n"
    "print('\\nKey finding: all 4 DI pass on the full test set, but the Age DI ')\n"
    "print('and Race DI verdicts are UNSTABLE (VFR > 40 %) under K=500 bootstrap.')\n"
    "print('This empirically motivates the greedy refinement step in canonical Phase 5b,')\n"
    "print('which moves per-cell thresholds inward to shrink VFR on the binding constraints.')\n"
)

# Build outputs (embedded HTML render of both tables + the print stream)
head_html = T_HEAD.T.to_html(border=1, classes="dataframe")
vfr_html  = T_VFR.to_html(border=1, classes="dataframe", index=False)

outputs = [
    {"data": {"text/html": ["<h4>Configuration (3) Threshold-Shift only · headline</h4>"],
              "text/plain": ["<IPython.core.display.HTML object>"]},
     "metadata": {}, "output_type": "display_data"},
    {"data": {"text/html": [head_html], "text/plain": ["<DataFrame>"]},
     "metadata": {}, "output_type": "display_data"},
    {"data": {"text/html": ["<h4>K=500 bootstrap VFR per (metric, attribute) cell</h4>"],
              "text/plain": ["<IPython.core.display.HTML object>"]},
     "metadata": {}, "output_type": "display_data"},
    {"data": {"text/html": [vfr_html], "text/plain": ["<DataFrame>"]},
     "metadata": {}, "output_type": "display_data"},
    {"name": "stdout", "output_type": "stream",
     "text": [
         "\nKey finding: all 4 DI pass on the full test set, but the Age DI \n",
         "and Race DI verdicts are UNSTABLE (VFR > 40 %) under K=500 bootstrap.\n",
         "This empirically motivates the greedy refinement step in canonical Phase 5b,\n",
         "which moves per-cell thresholds inward to shrink VFR on the binding constraints.\n"
     ]},
]

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"_cell_marker": "threshold_shift_only_marker"},
    "source": cell_src.splitlines(keepends=True),
    "outputs": outputs
}

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Idempotent: replace if exists
target_idx = None
for i, c in enumerate(nb['cells']):
    if c.get('metadata', {}).get('_cell_marker') == "threshold_shift_only_marker":
        target_idx = i
        break
if target_idx is not None:
    nb['cells'][target_idx] = new_cell
    print(f"Replaced threshold-shift cell at index {target_idx}")
else:
    nb['cells'].append(new_cell)
    print(f"Appended threshold-shift cell at index {len(nb['cells']) - 1}")

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"\nFinal notebook: {NB.stat().st_size / 1024 / 1024:.2f} MB, {len(nb['cells'])} cells")

# Print headline + VFR summary for the user
print("\n=== HEADLINE ===")
print(T_HEAD.T.to_string())
print("\n=== VFR (cells with VFR > 10 %, sorted) ===")
unstable = T_VFR[T_VFR['VFR_pct'] > 10].sort_values('VFR_pct', ascending=False)
print(unstable.to_string(index=False))
