"""
Loop 18: remove "manuscript" reference from F7 (model-agnostic pipeline).

Cell 61 Phase 6 currently lists:
  "Manuscript-claim verification table with directional comparators"

Since F7 is the model-agnostic recommended pipeline, replace with a generic
phrasing that doesn't tie to "manuscript":
  "Numerical-claim verification table with directional comparators"

Also update the bottom output line if it mentions 'manuscript'.

Then re-render the F7 PNG and update the embedded output in the notebook.
"""
import json, sys, io, base64, re
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm")
NB = ROOT / "CIKM_2026_LOS_Fairness_FINAL.ipynb"
F7_PNG = ROOT / "output_final" / "figures" / "F7_recommended_pipeline.png"

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

src = ''.join(nb['cells'][61].get('source', []))

# Replace manuscript references in F7 source
replacements = [
    ('"Manuscript-claim verification with comparators"',
     '"Numerical-claim verification with directional comparators"'),
    ('Manuscript-claim verification table with directional comparators',
     'Numerical-claim verification table with directional comparators'),
    ('Manuscript-ready, reviewer-defensible',
     'Auditable, reviewer-defensible'),
    ('manuscript-ready', 'audit-ready'),
]

n_changes = 0
for old, new in replacements:
    if old in src:
        src = src.replace(old, new)
        n_changes += 1
        print(f"  Replaced: {old[:60]}... -> {new[:60]}...")

nb['cells'][61]['source'] = src.splitlines(keepends=True)
print(f"\nCell 61 source updates: {n_changes}")
print(f"Manuscript mentions in cell 61 after: {src.lower().count('manuscript')}")

# Now re-render the F7 PNG
print("\nRe-rendering F7 PNG ...")
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

# Strip the matplotlib import from the source for re-execution
exec_src = re.sub(r"^import matplotlib\.patches as mpatches\n", "", src, flags=re.MULTILINE)
ns = {
    "plt": plt, "mpl": mpl, "mpatches": mpatches,
    "FancyBboxPatch": FancyBboxPatch, "np": np,
    "FIGURES_DIR": FIGURES_DIR,
}
try:
    exec(exec_src, ns)
    new_size = F7_PNG.stat().st_size
    print(f"Re-rendered F7 PNG: {new_size / 1024:.0f} KB")
except Exception as e:
    print(f"Render failed: {e}")
    raise

# Update embedded F7 PNG in cell 61 outputs
with open(F7_PNG, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode('ascii')

for o_idx, o in enumerate(nb['cells'][61].get('outputs', [])):
    if 'data' in o and 'image/png' in o.get('data', {}):
        nb['cells'][61]['outputs'][o_idx]['data']['image/png'] = img_b64
        print(f"Cell 61 output {o_idx}: PNG updated to {len(img_b64) * 3 / 4 / 1024:.0f} KB")

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"\nFinal notebook: {NB.stat().st_size / 1024 / 1024:.2f} MB")
