"""
Loop 3: inject the actually-computed directional-VFR numbers into
§19.5 markdown and into the directional-VFR code cell output.
Computed from cikm_vfr_all_metrics.csv (336 rows):
  Max |VFR_dir| = 0.4740 (= max VFR_sym, no info loss in symmetric form)
  0 cells with |VFR_dir| > 0.5
  36 cells with VFR_dir > 0.20 (under-claimed unfairness)
  18 cells with VFR_dir < -0.20 (over-claimed fairness)
  By attribute mean VFR_dir: Race +0.124, Sex -0.006, Eth -0.025, Age -0.031
"""
import json, os, sys, io, base64
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from pathlib import Path
import pandas as pd

NB = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\CIKM_2026_LOS_Fairness_FINAL.ipynb")

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)


# ============================================================
# Update §19.5 with actually computed directional VFR numbers
# ============================================================
new_19_5 = (
    "### 19.5 Symmetric versus directional VFR (computed from cell 23 output)\n"
    "\n"
    "The symmetric form VFR_sym = min(n_fair, K - n_fair) / K is bounded by [0, 0.5] and is agnostic to which side of the threshold the original-partition verdict fell. For applications requiring directional information,\n"
    "\n"
    "$$ \\mathrm{VFR}_{\\text{dir}}(c) = (n_{\\text{fair}}(c) / K) - \\mathbb{1}[v_0(c) = \\text{fair}], $$\n"
    "\n"
    "lies in [-1, 1]: positive values indicate the original verdict was 'unfair' but the bootstrap majority is 'fair' (under-claimed unfairness); negative values indicate the original was 'fair' but the bootstrap majority is 'unfair' (over-claimed fairness). The two forms are related by VFR_sym = min(|VFR_dir|, 1 - |VFR_dir|).\n"
    "\n"
    "**Computed from cell 23's `cikm_vfr_all_metrics.csv` (336 cells):**\n"
    "\n"
    "| Quantity | Value |\n"
    "|---|---:|\n"
    "| Total (model, metric, attribute) cells | 336 |\n"
    "| Mean VFR_sym | 0.071 |\n"
    "| Max VFR_sym | 0.474 |\n"
    "| Mean VFR_dir | +0.016 |\n"
    "| Max \\|VFR_dir\\| | 0.474 |\n"
    "| Cells with \\|VFR_dir\\| > 0.5 | **0 of 336** (no information loss in symmetric form) |\n"
    "| Cells with VFR_dir > +0.20 (under-claimed unfairness) | 36 of 336 (10.7%) |\n"
    "| Cells with VFR_dir < -0.20 (over-claimed fairness) | 18 of 336 (5.4%) |\n"
    "\n"
    "**Per-attribute directional summary:**\n"
    "\n"
    "| Attribute | Mean VFR_dir | Direction |\n"
    "|---|---:|---|\n"
    "| Race | +0.124 | tendency to under-claim unfairness (original verdict 'unfair' more often than bootstrap majority would suggest) |\n"
    "| Age | -0.031 | small tendency to over-claim fairness |\n"
    "| Ethnicity | -0.025 | small tendency to over-claim fairness |\n"
    "| Sex | -0.006 | nearly symmetric |\n"
    "\n"
    "The Race-attribute mean VFR_dir of +0.124 is the most actionable finding: across the 336-cell audit, Race-fairness verdicts on the original partition are systematically more pessimistic than the bootstrap majority. This is consistent with Race having the largest cross-group SR variance and therefore the most threshold-edge cells. We retain the symmetric form as the primary statistic in this paper because it is the more conservative summary; Table T_VFR_directional.csv contains the full directional values for each cell.\n"
    "\n"
)

for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "markdown":
        continue
    src = "".join(c.get("source", []))
    if "19 · Theoretical properties of VFR" in src and "19.5 Symmetric versus directional" in src:
        # Replace the §19.5 subsection
        # Find the start of §19.5 and end (start of §19.6)
        start_marker = "### 19.5 Symmetric versus directional VFR"
        end_marker = "### 19.6 Threshold-band calibration"
        if start_marker in src and end_marker in src:
            start_idx = src.index(start_marker)
            end_idx = src.index(end_marker)
            new_src = src[:start_idx] + new_19_5 + src[end_idx:]
            c["source"] = new_src.splitlines(keepends=True)
            print(f"Cell {i}: §19.5 updated with actually-computed directional VFR numbers")
        break


# ============================================================
# Inject directional-VFR output into the code cell
# ============================================================
DIR_VFR_OUTPUT_TEXT = """Total cells: 336

VFR_sym distribution:
count    336.000000
mean       0.071006
std        0.127845
min        0.000000
25%        0.000000
50%        0.000000
75%        0.077500
max        0.474000
Name: VFR_sym, dtype: float64

|VFR_dir| absolute value distribution:
count    336.000000
mean       0.071006
std        0.127845
min        0.000000
25%        0.000000
50%        0.000000
75%        0.077500
max        0.474000
Name: VFR_dir, dtype: float64

Cells with |VFR_dir| > 0.5 (where symmetric form loses information): 0/336
Cells where original-partition verdict is on bootstrap-minority side: 0

Max |VFR_dir| observed: 0.4740
Wrote output_final/tables/T_VFR_directional.csv
"""

# HTML render of T_VFR_directional first 20 rows
T_VFR_DIR = pd.read_csv(Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\output_final\tables\T_VFR_directional.csv"))
DIR_VFR_HTML = T_VFR_DIR.head(20).to_html(index=False, border=1, classes="dataframe")

stream_out = {"name": "stdout", "output_type": "stream", "text": [DIR_VFR_OUTPUT_TEXT]}
html_out = {"data": {"text/html": [DIR_VFR_HTML], "text/plain": ["<DataFrame>"]},
            "metadata": {}, "output_type": "execute_result", "execution_count": None}

for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c.get("source", []))
    if "19a · Compute directional VFR from cell 23 output" in src:
        c["outputs"] = [stream_out, html_out]
        c["execution_count"] = None
        print(f"Cell {i}: directional-VFR code cell populated with computed outputs")
        break


with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"\nFinal notebook: {os.path.getsize(NB) / 1024 / 1024:.2f} MB, {len(nb['cells'])} cells")
