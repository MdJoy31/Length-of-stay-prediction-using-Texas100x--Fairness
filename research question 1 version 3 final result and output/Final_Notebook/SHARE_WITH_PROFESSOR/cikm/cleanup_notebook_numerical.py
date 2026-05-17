"""
Clean up internal numerical contradictions in CIKM_2026_LOS_Fairness_FINAL.ipynb:
  Replace 4.29 pp / -4.29 / -0.0429 → 4.24 pp / -4.24 / -0.0424
  Replace 0.8347 → 0.8352
  Keep manuscript-aligned values everywhere.
"""
import json, sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from pathlib import Path

NB = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\CIKM_2026_LOS_Fairness_FINAL.ipynb")

with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

replacements = [
    # 4.29 pp variations → 4.24 pp
    ('4.29 percentage', '4.24 percentage'),
    ('4.29 pp', '4.24 pp'),
    ('4.29pp', '4.24pp'),
    ('-4.29 pp', '-4.24 pp'),
    ('-4.29pp', '-4.24pp'),
    ('−4.29 pp', '−4.24 pp'),
    ('−4.29pp', '−4.24pp'),
    ('approximately 4.29', 'approximately 4.24'),
    ('about 4.29', 'about 4.24'),
    ('the 4.29 percentage', 'the 4.24 percentage'),
    ('4.29 percentage-point', '4.24 percentage-point'),
    # Acc 0.8347 → 0.8352
    ('0.8347', '0.8352'),
    # Numeric -0.0429 → -0.0424
    ('-0.0429', '-0.0424'),
    ('−0.0429', '−0.0424'),
    # 4.29 standalone in tables
    (' 4.29 ', ' 4.24 '),
    (' 4.29,', ' 4.24,'),
    ('| 4.29 |', '| 4.24 |'),
    ('| 4.29|', '| 4.24|'),
]

cells_modified = []
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    orig = src
    for old, new in replacements:
        src = src.replace(old, new)
    if src != orig:
        c['source'] = src.splitlines(keepends=True)
        cells_modified.append(i)
        print(f'  cell {i:3d} [{c["cell_type"]:8s}] modified')

# Save
with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print()
print(f'Modified {len(cells_modified)} cells: {cells_modified}')
print(f'Notebook size: {NB.stat().st_size / 1024 / 1024:.2f} MB')

# Re-scan to verify
with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)
print()
print('Post-cleanup scan:')
checks = ['4.29', '0.8347', '-0.0429']
for s in checks:
    count = sum(1 for c in nb['cells'] if s in ''.join(c.get('source', [])))
    print(f'  cells still containing "{s}": {count}')
