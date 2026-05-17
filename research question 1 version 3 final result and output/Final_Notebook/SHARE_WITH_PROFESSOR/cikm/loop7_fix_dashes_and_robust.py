"""
Loop 7: fix all em-dash / en-dash occurrences in code cells, and fix
the 'robust' AI-cliche in cell 71 markdown.

Replacements:
  ' — ' (space em-dash space) -> ': ' (most common case in code comments)
  '—' (em-dash, anywhere else) -> '-' (display placeholder)
  '–' (en-dash, in numeric range strings) -> '-'
  'is robust to' -> 'is stable under'

Markdown cells are left alone except for cell 71 'robust' fix.
"""
import json, sys, io, re
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

NB = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\CIKM_2026_LOS_Fairness_FINAL.ipynb")

EM = "—"  # —
EN = "–"  # –

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

total_em_fixed = 0
total_en_fixed = 0
robust_fixed = 0

for i, c in enumerate(nb['cells']):
    src_lines = c.get('source', [])
    if not src_lines:
        continue
    src = ''.join(src_lines)
    new_src = src
    em_count_before = src.count(EM)
    en_count_before = src.count(EN)

    if c['cell_type'] == 'code':
        # In code cells: replace ' — ' with ': ' (covers most comment patterns)
        new_src = new_src.replace(f" {EM} ", ": ")
        # Lone em-dashes (e.g., return "—") -> "-"
        new_src = new_src.replace(EM, "-")
        # En-dashes (cell 48: numeric ranges) -> "-"
        new_src = new_src.replace(EN, "-")

    elif c['cell_type'] == 'markdown' and i == 71:
        # Replace "is robust to" with "is stable under"
        if "is robust to" in new_src:
            new_src = new_src.replace("is robust to", "is stable under")
            robust_fixed += 1

    if new_src != src:
        c['source'] = new_src.splitlines(keepends=True)
        em_after = new_src.count(EM)
        en_after = new_src.count(EN)
        em_fixed_here = em_count_before - em_after
        en_fixed_here = en_count_before - en_after
        total_em_fixed += em_fixed_here
        total_en_fixed += en_fixed_here
        if em_fixed_here or en_fixed_here:
            print(f"Cell {i:>2} ({c['cell_type']}): em-fixed={em_fixed_here}, en-fixed={en_fixed_here}")

print(f"\nTotal em-dashes fixed: {total_em_fixed}")
print(f"Total en-dashes fixed: {total_en_fixed}")
print(f"'robust' replacements: {robust_fixed}")

# Write back
with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"\nFinal notebook: {NB.stat().st_size / 1024 / 1024:.2f} MB, {len(nb['cells'])} cells")
