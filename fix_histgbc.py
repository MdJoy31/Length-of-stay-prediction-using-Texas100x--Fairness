"""Fix HistGradientBoostingClassifier parameters"""
import json

NB_PATH = 'LOS_Prediction_Detailed.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

fixes = 0

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = cell['source']

    # Find HistGBC definitions and fix params
    in_histgbc = False
    for j, line in enumerate(src):
        if 'HistGradientBoostingClassifier(' in line:
            in_histgbc = True

        if in_histgbc:
            # Fix n_estimators → max_iter
            if 'n_estimators=' in line and 'HistGradientBoosting' not in line and 'XGB' not in line and 'LGBM' not in line and 'Random' not in line:
                src[j] = line.replace('n_estimators=', 'max_iter=')
                print(f"  Cell {i}, line {j}: n_estimators → max_iter")
                fixes += 1

            # Fix subsample (not supported) → remove
            if 'subsample=0.8' in line and 'XGB' not in line:
                # Remove subsample from the line
                src[j] = line.replace(', subsample=0.8', '').replace('subsample=0.8, ', '')
                print(f"  Cell {i}, line {j}: Removed subsample")
                fixes += 1

            # Fix min_samples_split → min_samples_leaf
            if 'min_samples_split=' in line:
                src[j] = line.replace('min_samples_split=', 'min_samples_leaf=')
                print(f"  Cell {i}, line {j}: min_samples_split → min_samples_leaf")
                fixes += 1

            # End of HistGBC definition
            if '),' in line or '),\n' in line:
                in_histgbc = False

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nApplied {fixes} parameter fixes.")
