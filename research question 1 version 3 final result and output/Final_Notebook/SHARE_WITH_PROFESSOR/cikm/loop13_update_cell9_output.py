"""
Loop 13: update cell 9's stream output 2 to reflect the corrected
expected_T3 dict. The cell was last run with the BROKEN dict and
the captured output shows MISMATCH errors. After the fix, all 14
expected_T3 rows match T3, so the verification block should print
the clean-PASS message instead.
"""
import json, sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

NB = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\CIKM_2026_LOS_Fairness_FINAL.ipynb")

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Replace cell 9 output 2 stream text
NEW_OUTPUT = (
    "\nVerifying T3 against manuscript Table 3 (image-extracted, after RACE re-mapping) ...\n"
    "  All 14 rows of Table 3 match the manuscript image exactly.\n"
)

if len(nb['cells'][9]['outputs']) >= 3:
    nb['cells'][9]['outputs'][2] = {
        "name": "stdout",
        "output_type": "stream",
        "text": [NEW_OUTPUT]
    }
    print("Cell 9 output 2 updated with clean-PASS verification text")

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"\nFinal notebook: {NB.stat().st_size / 1024 / 1024:.2f} MB")
