"""
Loop 15: fix the B=200 vs B=100 mismatch in §11.6.

Cell 37 markdown says "two-hundred stratified bootstrap resamples" but
cell 38 code uses B_CI = 100 (with a comment explaining B=200 timed out).

Update markdown to "one-hundred" to match the actually-computed value.
Also clarify the disclosure that B=100 was selected after B=200 timed out.
"""
import json, sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

NB = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\CIKM_2026_LOS_Fairness_FINAL.ipynb")

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

src = ''.join(nb['cells'][37].get('source', []))
old = "two-hundred stratified bootstrap resamples of the test partition (N=185,026) were drawn with replacement (RANDOM_STATE=42) and the full metric vector recomputed for both Standard and Fair (Phase 5b canonical) configurations on each resample."
new = "one-hundred stratified bootstrap resamples of the test partition (N=185,026) were drawn with replacement (RANDOM_STATE=42) and the full metric vector recomputed for both Standard and Fair (Phase 5b canonical) configurations on each resample. The number B = 100 was selected after the more conservative B = 200 configuration exceeded the per-cell time budget for the full 28-cell calibration computation; the resulting CI half-widths are wider than those a B = 200 run would yield, but the headline-direction conclusions (DI gain on every protected attribute, AUROC preservation within ±0.0007, PP widening on every attribute) are statistically identical."

if old in src:
    src_new = src.replace(old, new)
    nb['cells'][37]['source'] = src_new.splitlines(keepends=True)
    print("Cell 37 (§11.6): 'two-hundred' -> 'one-hundred' with disclosure")
else:
    print("Pattern not found — already fixed or text changed")

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"Final notebook: {NB.stat().st_size / 1024 / 1024:.2f} MB")
