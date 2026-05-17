"""
Loop 9: cleanup remaining section-numbering duplications.

Cell 1 contains only the divider "---\n## 1. Setup & Methodology\n" which
duplicates §1 (cell 2 has the substantive §1 content). Delete cell 1 to
remove the duplicate.

After this loop the notebook should have exactly one §1 ... §24 (no gaps,
no duplicates).
"""
import json, sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

NB = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\CIKM_2026_LOS_Fairness_FINAL.ipynb")

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Verify cell 1 is the duplicate header before deleting
target_src = ''.join(nb['cells'][1].get('source', []))
EXPECTED = '---\n## 1. Setup & Methodology\n'
if target_src.strip() == EXPECTED.strip():
    del nb['cells'][1]
    print(f"Deleted cell 1 (duplicate §1 header)")
else:
    print(f"Cell 1 content unexpected ({len(target_src)} chars), not deleting:")
    print(repr(target_src[:200]))

# Now check what cell 1 has become (was cell 2, the real §1)
new_cell_1_src = ''.join(nb['cells'][1].get('source', []))
print(f"\nNew cell 1 first 100 chars: {new_cell_1_src[:100]!r}")

# Save
with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"\nFinal notebook: {NB.stat().st_size / 1024 / 1024:.2f} MB, {len(nb['cells'])} cells")
