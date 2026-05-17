"""
Loop 28: inject the four-configuration baseline audit extension into the notebook.

Adds one new code cell at the end that loads + displays all 14 baseline-extension
CSVs (T13_axis1, T9_axis2, T10_axis3 across 4 configs + summary + diagnostics).

The canonical Phase 5b artefacts (T13_lambda_sweep, T14_ablation_xgboost,
T15_standard_vs_fair, T9_min_sample_size, T10_cross_hospital_cv, T11_fleiss_kappa)
are NOT modified.
"""
import json, sys, io, base64
from pathlib import Path
import pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm")
NB = ROOT / "CIKM_2026_LOS_Fairness_FINAL.ipynb"
TAB = ROOT / "output_final" / "tables"

T_SUM = pd.read_csv(TAB / "T_baseline_audit_summary.csv")
try:
    T_DIAG = pd.read_csv(TAB / "T_baseline_audit_diagnostics.csv")
except Exception:
    T_DIAG = pd.DataFrame()

# Load all per-config CSVs
T_VFR = {i: pd.read_csv(TAB / f"T13_axis1_vfr_config{i}.csv") for i in [1, 2, 3, 4]}
T_N   = {i: pd.read_csv(TAB / f"T9_axis2_minN_config{i}.csv") for i in [1, 2, 3, 4]}
T_K   = {i: pd.read_csv(TAB / f"T10_axis3_kappa_config{i}.csv") for i in [1, 2, 3, 4]}

cell_src = (
    "# ──────────────────────────────────────────────────────────────\n"
    "# §29 · Four-configuration baseline audit extension\n"
    "# Same audit instrument (K=500 VFR, 8-N grid × 30 reps min-N, K=20 GroupKFold)\n"
    "# applied to four configurations:\n"
    "#   (1) Real-Only             - Standard XGBoost, no intervention\n"
    "#   (2) Reweighing-only λ=2  - intersectional reweighed XGBoost, threshold 0.5\n"
    "#   (3) Threshold-Shift only  - Standard XGBoost + α-search thresholds (no greedy)\n"
    "#   (4) Real+VFR canonical    - Phase 5b: α-search + greedy refinement\n"
    "# Pre-computed by compute_baseline_audit_extension.py (random_state=42 throughout).\n"
    "# Canonical artefacts (T13_lambda_sweep, T14, T15, T9_min_sample_size, T11) are NOT modified.\n"
    "# ──────────────────────────────────────────────────────────────\n"
    "import pandas as pd\n"
    "from IPython.display import display, HTML\n"
    "T_SUM = pd.read_csv('output_final/tables/T_baseline_audit_summary.csv')\n"
    "display(HTML('<h4>Cross-config summary (4 rows × 12 cols)</h4>'))\n"
    "display(T_SUM)\n"
    "for cfg in [1, 2, 3, 4]:\n"
    "    name = T_SUM[T_SUM['config_id']==cfg]['Configuration'].iloc[0]\n"
    "    display(HTML(f'<h5>Config {cfg}: {name} — Axis 1 VFR (28 cells)</h5>'))\n"
    "    display(pd.read_csv(f'output_final/tables/T13_axis1_vfr_config{cfg}.csv'))\n"
    "    display(HTML(f'<h5>Config {cfg}: Axis 2 min-N (28 cells)</h5>'))\n"
    "    display(pd.read_csv(f'output_final/tables/T9_axis2_minN_config{cfg}.csv'))\n"
    "    display(HTML(f'<h5>Config {cfg}: Axis 3 Fleiss κ (7 metrics)</h5>'))\n"
    "    display(pd.read_csv(f'output_final/tables/T10_axis3_kappa_config{cfg}.csv'))\n"
)

# Build embedded outputs: summary HTML + first 3 per-config tables (truncate to keep cell size sane)
outputs = []
outputs.append({"data": {"text/html": ["<h4>Cross-config summary (4 rows × 12 cols)</h4>"],
                          "text/plain": ["<HTML>"]},
                "metadata": {}, "output_type": "display_data"})
outputs.append({"data": {"text/html": [T_SUM.to_html(index=False, border=1, classes="dataframe")],
                          "text/plain": ["<DataFrame>"]},
                "metadata": {}, "output_type": "display_data"})

for cfg in [1, 2, 3, 4]:
    name = T_SUM[T_SUM['config_id'] == cfg]['Configuration'].iloc[0]
    outputs.append({"data": {"text/html": [f"<h5>Config {cfg}: {name} — Axis 1 VFR</h5>"],
                              "text/plain": ["<HTML>"]},
                    "metadata": {}, "output_type": "display_data"})
    outputs.append({"data": {"text/html": [T_VFR[cfg].to_html(index=False, border=1, classes="dataframe")],
                              "text/plain": ["<DataFrame>"]},
                    "metadata": {}, "output_type": "display_data"})
    outputs.append({"data": {"text/html": [f"<h5>Config {cfg}: Axis 2 min-N</h5>"],
                              "text/plain": ["<HTML>"]},
                    "metadata": {}, "output_type": "display_data"})
    outputs.append({"data": {"text/html": [T_N[cfg].to_html(index=False, border=1, classes="dataframe")],
                              "text/plain": ["<DataFrame>"]},
                    "metadata": {}, "output_type": "display_data"})
    outputs.append({"data": {"text/html": [f"<h5>Config {cfg}: Axis 3 Fleiss κ</h5>"],
                              "text/plain": ["<HTML>"]},
                    "metadata": {}, "output_type": "display_data"})
    outputs.append({"data": {"text/html": [T_K[cfg].to_html(index=False, border=1, classes="dataframe")],
                              "text/plain": ["<DataFrame>"]},
                    "metadata": {}, "output_type": "display_data"})

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"_cell_marker": "baseline_audit_extension_marker"},
    "source": cell_src.splitlines(keepends=True),
    "outputs": outputs
}

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

target_idx = None
for i, c in enumerate(nb['cells']):
    if c.get('metadata', {}).get('_cell_marker') == "baseline_audit_extension_marker":
        target_idx = i
        break
if target_idx is not None:
    nb['cells'][target_idx] = new_cell
    print(f"Replaced baseline-audit cell at index {target_idx}")
else:
    nb['cells'].append(new_cell)
    print(f"Appended baseline-audit cell at index {len(nb['cells']) - 1}")

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"Final notebook: {NB.stat().st_size / 1024 / 1024:.2f} MB, {len(nb['cells'])} cells")

print()
print("=== ONE-PARAGRAPH SUMMARY (verifiable against CSVs) ===")
parts = []
s = T_SUM.set_index('Configuration')
for cfg_name in T_SUM['Configuration']:
    r = s.loc[cfg_name]
    parts.append(
        f"{cfg_name} achieved DI-pass on {int(r['n_DI_pass'])}/4 attributes "
        f"with VFR mean {r['vfr_mean_across_28_cells']:.4f} (max {r['vfr_max']:.4f}) "
        f"and cross-site mean κ {r['kappa_mean']:.4f}"
    )
print("Across four configurations: " + ". ".join(parts) + ".")
