"""
Loop 8: fix duplicate §13 / missing §21.

Cell 50 currently is "## 13 · Comparison against prior Q1 / A* studies", which
duplicates §13 (K-Sensitivity, cell 44). Rename cell 50 to §21, filling the
gap between §20 Limitations and §22 Demographic-anomaly resolution.

Also rename internal "Table 13.1" -> "Table 21.1" in cell 50 only.

Cell 45's reference to "13.1" is a code comment about §13 K-Sensitivity and
remains correct — left untouched.
"""
import json, sys, io, re
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

NB = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\CIKM_2026_LOS_Fairness_FINAL.ipynb")

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

src = ''.join(nb['cells'][50].get('source', []))
new_src = src

# 1. Header rename: ## 13 · -> ## 21 ·
new_src = new_src.replace("## 13 · Comparison against prior Q1", "## 21 · Comparison against prior Q1")
# 2. Inline references: Table 13.1 -> Table 21.1 (table only, leave §13 K-sensitivity refs alone)
new_src = re.sub(r"\bTable 13\.1\b", "Table 21.1", new_src)

# Verify changes
n_changes = sum(1 for a, b in zip(src, new_src) if a != b)
print(f"Cell 50 character changes: {n_changes}")
print(f"## 13 -> ## 21: {('## 13 ·' in src and '## 13 ·' not in new_src)}")
print(f"Table 13.1 -> Table 21.1: {src.count('Table 13.1')} -> {new_src.count('Table 21.1')}")

nb['cells'][50]['source'] = new_src.splitlines(keepends=True)

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"\nFinal notebook: {NB.stat().st_size / 1024 / 1024:.2f} MB")
