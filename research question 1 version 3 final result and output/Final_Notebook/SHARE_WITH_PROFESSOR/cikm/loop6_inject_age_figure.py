"""
Loop 6: inject the F8 age-group figure and T_age_group_analysis table into
the notebook as a new code cell immediately after cell 69 (the markdown
section that explains the THCIC PUDF age-bucket coding).

Cell content:
- Imports + read T_age_group_analysis.csv
- Display the table inline
- Display the F8 PNG via base64
- Print verification numbers
"""
import json, base64, sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm")
NB = ROOT / "CIKM_2026_LOS_Fairness_FINAL.ipynb"
F8 = ROOT / "output_final" / "figures" / "F8_age_group_analysis.png"
T_AG = ROOT / "output_final" / "tables" / "T_age_group_analysis.csv"

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Read T_age_group_analysis.csv text
import pandas as pd
T = pd.read_csv(T_AG, index_col=0)
T_html = T.to_html(border=1, classes="dataframe")

# Read F8 PNG as base64
with open(F8, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode('ascii')

# Build the new code cell
code_src = (
    "# ──────────────────────────────────────────────────────────────\n"
    "# §24.0a · Age-group distribution figure (F8) + table — verifies fix for PAT_AGE\n"
    "# ──────────────────────────────────────────────────────────────\n"
    "# Pre-computed by standalone script `compute_age_group_analysis.py`.\n"
    "# Confirms PAT_AGE values 0-21 (22 distinct THCIC PUDF buckets) and that\n"
    "# the cell-5 `_age_grp` function aggregates correctly.\n"
    "import pandas as pd\n"
    "from pathlib import Path\n"
    "from IPython.display import Image, HTML, display\n"
    "T_age = pd.read_csv(Path('output_final/tables/T_age_group_analysis.csv'), index_col=0)\n"
    "display(HTML('<h4>T_age_group_analysis (computed from full 925,128-record cohort)</h4>'))\n"
    "display(T_age)\n"
    "print(f'Sum N: {int(T_age[\"N\"].sum()):,} (target 925,128)')\n"
    "print(f'Sum Cohort_pct: {T_age[\"Cohort_pct\"].sum():.2f}% (target 100%)')\n"
    "display(HTML('<h4>F8 · Age-group distribution and clinical signal</h4>'))\n"
    "display(Image(filename='output_final/figures/F8_age_group_analysis.png'))\n"
)

# Build the table HTML output
table_html_out = {
    "data": {"text/html": [T_html], "text/plain": ["<DataFrame>"]},
    "metadata": {}, "output_type": "display_data"
}

# Build verification text output
verify_lines = (
    f"Sum N: {int(T['N'].sum()):,} (target 925,128)\n"
    f"Sum Cohort_pct: {T['Cohort_pct'].sum():.2f}% (target 100%)\n"
)
verify_out = {"name": "stdout", "output_type": "stream", "text": [verify_lines]}

# Build image output
img_out = {
    "data": {"image/png": img_b64, "text/plain": ["<IPython.core.display.Image object>"]},
    "metadata": {}, "output_type": "display_data"
}

# Heading HTML for the table
heading_table = {
    "data": {"text/html": ["<h4>T_age_group_analysis (computed from full 925,128-record cohort)</h4>"], "text/plain": ["<IPython.core.display.HTML object>"]},
    "metadata": {}, "output_type": "display_data"
}
heading_fig = {
    "data": {"text/html": ["<h4>F8 · Age-group distribution and clinical signal</h4>"], "text/plain": ["<IPython.core.display.HTML object>"]},
    "metadata": {}, "output_type": "display_data"
}

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "source": code_src.splitlines(keepends=True),
    "outputs": [
        heading_table,
        table_html_out,
        verify_out,
        heading_fig,
        img_out,
    ]
}

# Find cell 69 (§24.0 markdown) and insert immediately after
TARGET_HEADER = "### 24.0 · Age-group distribution"
target_idx = None
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'markdown':
        continue
    src = ''.join(c.get('source', []))
    if TARGET_HEADER in src:
        target_idx = i
        break

if target_idx is None:
    raise RuntimeError(f"Could not find §24.0 markdown cell with header '{TARGET_HEADER}'")

# Check if a cell with the same heading already exists right after (idempotency)
already_inserted = False
if target_idx + 1 < len(nb['cells']):
    next_cell = nb['cells'][target_idx + 1]
    if next_cell['cell_type'] == 'code':
        next_src = ''.join(next_cell.get('source', []))
        if "§24.0a · Age-group distribution figure" in next_src:
            already_inserted = True
            print(f"Cell {target_idx + 1} already contains the F8 injection. Replacing outputs only.")
            nb['cells'][target_idx + 1] = new_cell

if not already_inserted:
    nb['cells'].insert(target_idx + 1, new_cell)
    print(f"Inserted new code cell at index {target_idx + 1} (after §24.0 markdown at cell {target_idx})")

# Write back
with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"\nFinal notebook: {NB.stat().st_size / 1024 / 1024:.2f} MB, {len(nb['cells'])} cells")
print(f"F8 PNG embedded: {len(img_b64):,} base64 chars (~{len(img_b64) * 3 // 4 // 1024} KB binary)")
