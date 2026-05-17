"""
Loop 10: move §21 (Comparison against prior Q1/A* studies) to its
correct numerical position between §20 (Limitations) and §22
(Demographic-anomaly resolution).

Currently cell 49 (§21) is interrupting the §15 figures sequence
(cells 48-56). Move it to after §20 cell.
"""
import json, sys, io, re
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

NB = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\CIKM_2026_LOS_Fairness_FINAL.ipynb")

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find §21 cell
sec21_idx = None
sec22_idx = None
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'markdown':
        continue
    src = ''.join(c.get('source', []))
    if "## 21 · Comparison against prior Q1" in src:
        sec21_idx = i
    if "## 22 · Demographic-anomaly resolution" in src:
        sec22_idx = i

print(f"§21 currently at cell {sec21_idx}")
print(f"§22 currently at cell {sec22_idx}")

if sec21_idx is None or sec22_idx is None:
    raise RuntimeError("Could not locate §21 or §22 cells")

# Pop §21 cell
sec21_cell = nb['cells'].pop(sec21_idx)

# §22 index decreases by 1 since we removed an earlier cell
new_sec22_idx = sec22_idx - 1 if sec22_idx > sec21_idx else sec22_idx

# Insert §21 immediately before §22 (i.e., at new_sec22_idx)
nb['cells'].insert(new_sec22_idx, sec21_cell)

# Verify new position
print(f"\nAfter reorder:")
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'markdown':
        continue
    src = ''.join(c.get('source', []))
    for line in src.split('\n')[:3]:
        m = re.match(r'^##\s+(\d+)\s*[·\.]', line.strip())
        if m and int(m.group(1)) >= 19 and int(m.group(1)) <= 24:
            print(f"  Cell {i:>2}: {line.strip()[:90]}")
            break

# Save
with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"\nFinal notebook: {NB.stat().st_size / 1024 / 1024:.2f} MB, {len(nb['cells'])} cells")
