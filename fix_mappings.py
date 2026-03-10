"""
fix_mappings.py — Fix the wrong EDA visualization mappings in all 3 notebooks.

Problem:
  EDA cells use 1-indexed maps: RACE_MAP={1:'White',...}, SEX_MAP={1:'Male', 2:'Female'}
  But data values are 0-indexed:  RACE=0..4, SEX_CODE=0..1, ETHNICITY=0..1

Fix:
  RACE_MAP for viz:  {0:'Other/Unknown', 1:'White', 2:'Black', 3:'Hispanic', 4:'Asian/PI'}
  SEX_MAP for viz:   {0:'Female', 1:'Male'}
"""
import json, os

BASE = r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\final_notebooks"

notebooks = [
    "LOS_Prediction_Standard.ipynb",
    "LOS_Prediction_Detailed.ipynb",
    "Fairness_Analysis_Complete.ipynb",
]

# Old → New replacements (applied to the source text of each code cell)
REPLACEMENTS = [
    # Fix EDA RACE_MAP (1-indexed → 0-indexed)
    (
        "RACE_MAP = {1:'White', 2:'Black', 3:'Hispanic', 4:'Asian/PI', 5:'Other'}",
        "RACE_MAP = {0:'Other/Unknown', 1:'White', 2:'Black', 3:'Hispanic', 4:'Asian/PI'}"
    ),
    # Fix EDA SEX_MAP (1-indexed → 0-indexed)
    (
        "SEX_MAP = {1:'Male', 2:'Female'}",
        "SEX_MAP = {0:'Female', 1:'Male'}"
    ),
]

for nb_name in notebooks:
    path = os.path.join(BASE, nb_name)
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    fix_count = 0
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        # Join source lines to check/replace
        src = ''.join(cell['source'])
        changed = False
        for old, new in REPLACEMENTS:
            if old in src:
                src = src.replace(old, new)
                changed = True
                fix_count += 1

        if changed:
            # Split back into source lines (preserving \n at end of each line)
            lines = src.split('\n')
            new_source = []
            for i, line in enumerate(lines):
                if i < len(lines) - 1:
                    new_source.append(line + '\n')
                else:
                    if line:  # Non-empty last line
                        new_source.append(line)
            cell['source'] = new_source
            # Clear execution count and outputs (will re-execute)
            cell['execution_count'] = None
            cell['outputs'] = []

    # Clear all cells' execution state for clean re-run
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            cell['execution_count'] = None
            cell['outputs'] = []

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    print(f"[FIXED] {nb_name}: {fix_count} replacements, {len(code_cells)} code cells cleared")

# Also fix root-level Complete notebook
root_complete = os.path.join(BASE, '..', 'Fairness_Analysis_Complete.ipynb')
if os.path.exists(root_complete):
    with open(root_complete, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    fix_count = 0
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        changed = False
        for old, new in REPLACEMENTS:
            if old in src:
                src = src.replace(old, new)
                changed = True
                fix_count += 1
        if changed:
            lines = src.split('\n')
            new_source = []
            for i, line in enumerate(lines):
                if i < len(lines) - 1:
                    new_source.append(line + '\n')
                else:
                    if line:
                        new_source.append(line)
            cell['source'] = new_source
        cell['execution_count'] = None
        cell['outputs'] = []
    with open(root_complete, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"[FIXED] ROOT Fairness_Analysis_Complete.ipynb: {fix_count} replacements")

print("\n✅ All mappings fixed. Ready for re-execution.")
