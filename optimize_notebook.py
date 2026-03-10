"""Optimize notebook for successful execution:
1. Replace slow GradientBoostingClassifier with HistGradientBoostingClassifier (100x faster)
2. Add gc.collect() between heavy cells
3. Reduce DNN max epochs from 100 to 50
4. Keep everything else the same
"""
import json

NB_PATH = 'LOS_Prediction_Detailed.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

fixes = 0

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = cell['source']

    for j, line in enumerate(src):
        # Fix 1: Replace GradientBoostingClassifier with HistGradientBoostingClassifier
        if "GradientBoostingClassifier(" in line and "Hist" not in line and "import" not in line:
            src[j] = line.replace("GradientBoostingClassifier(", "HistGradientBoostingClassifier(")
            print(f"  Cell {i}, line {j}: GBC → HistGBC")
            fixes += 1

        # Fix 2: Replace GradientBoostingClassifier import
        if "from sklearn.ensemble import" in line and "GradientBoostingClassifier" in line and "Hist" not in line:
            src[j] = line.replace("GradientBoostingClassifier", "HistGradientBoostingClassifier")
            print(f"  Cell {i}, line {j}: Updated import")
            fixes += 1

        # Fix 3: Remove random_state from HistGBC (different param names)
        # Actually HistGBC supports random_state. Keep it.

        # Fix 4: Reduce DNN epochs from 100 to 50
        if "for epoch in range(100):" in line:
            src[j] = line.replace("range(100)", "range(50)")
            print(f"  Cell {i}, line {j}: DNN epochs 100→50")
            fixes += 1

        # Fix 5: Add explicit gc.collect() after training loops
        # (We'll add this separately)

# Fix 6: Add gc.collect() at the start of heavy computation cells
heavy_markers = [
    "# Stacking Ensemble",
    "# Extended Subset Fairness Analysis",
    "# Fairness Intervention: Lambda-Scaled Reweighing",
    "# AFCE",
]

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    if not cell['source']:
        continue
    first_line = cell['source'][0]
    for marker in heavy_markers:
        if marker in first_line:
            # Add gc.collect() at the start
            cell['source'].insert(0, "import gc; gc.collect()\n")
            print(f"  Cell {i}: Added gc.collect() before {marker}")
            fixes += 1
            break

# Clear all outputs
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nApplied {fixes} optimizations. Outputs cleared.")
