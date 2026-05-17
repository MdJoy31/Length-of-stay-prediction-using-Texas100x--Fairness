"""
Loop 24: clean up the F1 pipeline diagram year-string formatting and
re-render the embedded PNG so the visible figure matches the corrected
year (FY 2006 Q1-Q4) and Texas-100X attribution.
"""
import json, sys, io, base64, re
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm")
NB = ROOT / "CIKM_2026_LOS_Fairness_FINAL.ipynb"
F1_PNG = ROOT / "output_final" / "figures" / "F1_reliability_framework.png"

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

src = ''.join(nb['cells'][49].get('source', []))

# Clean up nested parens
src = src.replace(
    "N = 925,128 inpatient discharges from 441 Texas hospitals (FY 2006 (Q1-Q4))",
    "N = 925,128 discharges, 441 Texas hospitals, FY 2006 Q1-Q4 (Texas-100X benchmark)"
)
nb['cells'][49]['source'] = src.splitlines(keepends=True)

# Re-render F1
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "axes.titleweight": "bold",
})
FIGURES_DIR = str(ROOT / "output_final" / "figures")

exec_src = re.sub(r"^import matplotlib\.patches as mpatches\n", "", src, flags=re.MULTILINE)
ns = {
    "plt": plt, "mpl": mpl, "mpatches": mpatches,
    "FancyBboxPatch": FancyBboxPatch, "np": np,
    "FIGURES_DIR": FIGURES_DIR,
}
exec(exec_src, ns)
print(f"Re-rendered F1: {F1_PNG.stat().st_size / 1024:.0f} KB")

# Update embedded PNG in cell 49
with open(F1_PNG, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode('ascii')
for o_idx, o in enumerate(nb['cells'][49].get('outputs', [])):
    if 'data' in o and 'image/png' in o.get('data', {}):
        nb['cells'][49]['outputs'][o_idx]['data']['image/png'] = img_b64
        print(f"Cell 49 output {o_idx}: PNG updated to {len(img_b64) * 3 / 4 / 1024:.0f} KB")

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"\nFinal notebook: {NB.stat().st_size / 1024 / 1024:.2f} MB")
