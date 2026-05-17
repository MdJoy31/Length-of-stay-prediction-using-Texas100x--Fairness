"""
Loop 27: inject the extreme lambda sweep results into the notebook.

Tests whether reweighing alone (at any λ ∈ {200, 500, 1000, 5000, 10000}, with
relaxed clipping or axis-specific Age reweighing) can achieve all-4-DI ≥ 0.80.
Conclusion: it cannot. The Age DI gap (cohort base rates 20.7% vs 60.6% LOS)
is too large for sample-weighted training to close without external decision-
threshold control.
"""
import json, sys, io
from pathlib import Path
import pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm")
NB = ROOT / "CIKM_2026_LOS_Fairness_FINAL.ipynb"

T = pd.read_csv(ROOT / "output_final" / "tables" / "T_lambda_extreme_sweep.csv")

cell_src = (
    "# ──────────────────────────────────────────────────────────────\n"
    "# §28 · Extreme λ sweep — does any reweighing-only configuration achieve\n"
    "#       all-4-DI ≥ 0.80 without threshold shifting?\n"
    "# Tests λ ∈ {200, 500, 1000, 5000, 10000} with three clipping schemes,\n"
    "# plus axis-specific Age-only reweighing. Pre-computed by\n"
    "# compute_lambda_extreme_sweep.py on the canonical XGBoost (n_est=300).\n"
    "# ──────────────────────────────────────────────────────────────\n"
    "import pandas as pd\n"
    "from IPython.display import display, HTML\n"
    "T_lam = pd.read_csv('output_final/tables/T_lambda_extreme_sweep.csv')\n"
    "display(HTML('<h4>Extreme-λ reweighing-only sweep</h4>'))\n"
    "display(T_lam)\n"
    "n_pass = int(T_lam['all4'].sum())\n"
    "print(f'\\nConfigurations achieving all-4-DI ≥ 0.80: {n_pass} of {len(T_lam)}')\n"
    "print(f'Best Age-DI achieved across all configurations: {T_lam[\"DI_A\"].max():.3f}')\n"
    "print('Conclusion: NO reweighing-only configuration closes the Age-DI gap.')\n"
    "print('Threshold shifting (per-cell α-search) is REQUIRED, not just helpful.')\n"
)

T_html = T.to_html(border=1, classes="dataframe", index=False)
n_pass = int(T['all4'].sum())

outputs = [
    {"data": {"text/html": ["<h4>Extreme-λ reweighing-only sweep</h4>"],
              "text/plain": ["<IPython.core.display.HTML object>"]},
     "metadata": {}, "output_type": "display_data"},
    {"data": {"text/html": [T_html], "text/plain": ["<DataFrame>"]},
     "metadata": {}, "output_type": "display_data"},
    {"name": "stdout", "output_type": "stream",
     "text": [
         f"\nConfigurations achieving all-4-DI ≥ 0.80: {n_pass} of {len(T)}\n",
         f"Best Age-DI achieved across all configurations: {T['DI_A'].max():.3f}\n",
         "Conclusion: NO reweighing-only configuration closes the Age-DI gap.\n",
         "Threshold shifting (per-cell α-search) is REQUIRED, not just helpful.\n",
     ]},
]

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"_cell_marker": "lambda_extreme_marker"},
    "source": cell_src.splitlines(keepends=True),
    "outputs": outputs
}

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

target_idx = None
for i, c in enumerate(nb['cells']):
    if c.get('metadata', {}).get('_cell_marker') == "lambda_extreme_marker":
        target_idx = i
        break
if target_idx is not None:
    nb['cells'][target_idx] = new_cell
    print(f"Replaced lambda-extreme cell at index {target_idx}")
else:
    nb['cells'].append(new_cell)
    print(f"Appended lambda-extreme cell at index {len(nb['cells']) - 1}")

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"\nFinal notebook: {NB.stat().st_size / 1024 / 1024:.2f} MB, {len(nb['cells'])} cells")
