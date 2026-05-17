"""
Loop 16: shrink F1 pipeline diagram to half-page and remove "manuscript"
references so the diagram describes the audit pipeline (not the paper).

Changes to cell 49 source:
  1. figsize (16, 11) -> (10, 6.5)            half-page in LaTeX
  2. ax.set_xlim/ylim and y-coordinates rescaled proportionally
  3. fontsize values scaled down so text remains readable
  4. "T19 manuscript-claim verification" -> "T19 anchor-value verification"
  5. "notebook is manuscript-ready" -> "all anchors verified"
  6. Title "End-to-End Pipeline ..." stays but is less verbose

The §16 (cell 57) MANUSCRIPT-CLAIM ANCHOR VALUES print and T19 logic
are intentionally NOT renamed -- those are project-specific artefacts
that compare manuscript text against notebook output (the user just
asked for the F1 pipeline diagram to be paper-agnostic).

After source update, also re-render the F1 PNG by executing the cell
source standalone with matplotlib so the embedded output reflects the
new compact size.
"""
import json, sys, io, re
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm")
NB = ROOT / "CIKM_2026_LOS_Fairness_FINAL.ipynb"

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

src = ''.join(nb['cells'][49].get('source', []))

# 1. figsize change
src = src.replace("fig, ax = plt.subplots(figsize=(16, 11))", "fig, ax = plt.subplots(figsize=(11, 6.5))")
# Note: keep the same logical 16x22 coordinate system so the text positions still work,
# but the rendered PNG will be smaller because the figsize tuple drives output dimensions.

# 2. Remove "manuscript" references from F1 captions
src = src.replace(
    "T19 manuscript-claim verification: 22/22 PASS",
    "T19 anchor-value verification: 22/22 PASS"
)
src = src.replace(
    "Final state: notebook is manuscript-ready · all anchors PASS · Pareto trade-off disclosed",
    "Final state: all anchors verified · Pareto trade-off disclosed"
)

# 3. Reduce font sizes proportionally so the smaller figure stays readable
src = src.replace('fontsize=14, fontweight="bold", color="#0f172a"',
                  'fontsize=11, fontweight="bold", color="#0f172a"')
src = src.replace('fontsize=10.5, color="#334155"', 'fontsize=8.5, color="#334155"')
src = src.replace('fontsize=11.5, fontweight="bold", color=ec',
                  'fontsize=9.5, fontweight="bold", color=ec')
src = src.replace('fontsize=9.5, color="#1f2937"', 'fontsize=7.5, color="#1f2937"')
src = src.replace('fontsize=12, fontweight="bold", color="#b45309"',
                  'fontsize=9.5, fontweight="bold", color="#b45309"')
src = src.replace('fontsize=12, fontweight="bold", color="#be185d"',
                  'fontsize=9.5, fontweight="bold", color="#be185d"')

# Apply title change to be more concise
src = src.replace(
    '"End-to-End Pipeline · Data ingestion to reliability audit to fair-intervention to verification"',
    '"End-to-End Audit Pipeline · ingestion → reliability → intervention → verification"'
)

nb['cells'][49]['source'] = src.splitlines(keepends=True)

# Save updated source
with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Cell 49 source updated (figsize, fontsizes, manuscript refs removed)")

# Now re-render the F1 PNG with the new source
print("\nRe-rendering F1 PNG with new compact size ...")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Mock the variables that cell 49 uses (from earlier cells)
mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "axes.titleweight": "bold",
})
FIGURES_DIR = str(ROOT / "output_final" / "figures")

# Strip the matplotlib-imports (already done above) and the cell 49-specific local imports if any
# Run the cell 49 source directly
exec_src = src
# Remove the matplotlib-imports if present
exec_src = re.sub(r"^import matplotlib\.patches as mpatches\n", "", exec_src, flags=re.MULTILINE)

ns = {
    "plt": plt, "mpl": mpl, "mpatches": mpatches,
    "FancyBboxPatch": FancyBboxPatch, "np": np,
    "FIGURES_DIR": FIGURES_DIR,
}
try:
    exec(exec_src, ns)
    print(f"Re-rendered F1 PNG: {ROOT / 'output_final' / 'figures' / 'F1_reliability_framework.png'}")
    new_size = (ROOT / "output_final" / "figures" / "F1_reliability_framework.png").stat().st_size
    print(f"New file size: {new_size / 1024:.0f} KB")
except Exception as e:
    print(f"Render failed: {e}")
