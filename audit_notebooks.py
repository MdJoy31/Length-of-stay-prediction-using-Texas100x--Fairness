"""Audit all 3 notebooks for sex distribution maps, LightGBM config, and outputs."""
import json

for nb_name in ['final_notebooks/LOS_Prediction_Standard.ipynb',
                'final_notebooks/LOS_Prediction_Detailed.ipynb',
                'final_notebooks/Fairness_Analysis_Complete.ipynb']:
    with open(nb_name, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    total = len(nb['cells'])
    code_cells = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')
    has_outputs = sum(1 for c in nb['cells'] if c['cell_type'] == 'code' and len(c.get('outputs', [])) > 0)

    print(f"\n=== {nb_name} ===")
    print(f"  Total: {total}, Code: {code_cells}, With outputs: {has_outputs}")

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']

        # Check sex maps
        if 'SEX_MAP' in src or 'sex_map' in src or 'SEX_CODE' in src:
            for line in src.split('\n'):
                line_s = line.strip()
                if ('SEX_MAP' in line_s or 'sex_map' in line_s) and '=' in line_s and '{' in line_s:
                    print(f"  Cell {i}: {line_s[:120]}")

        # Check for 1-indexed sex maps (THE BUG)
        if "1: 'Male'" in src and "2: 'Female'" in src:
            print(f"  *** Cell {i}: STILL HAS 1-INDEXED SEX MAP (BUG!) ***")
        if "1:'Male'" in src and "2:'Female'" in src:
            print(f"  *** Cell {i}: STILL HAS 1-INDEXED SEX MAP (BUG!) ***")

        # Check LightGBM
        if 'LGBMClassifier' in src:
            for line in src.split('\n'):
                if 'device=' in line or 'n_jobs' in line:
                    print(f"  Cell {i} LGB: {line.strip()[:100]}")

        # Check RF n_jobs
        if 'RandomForest' in src and 'n_jobs' in src:
            for line in src.split('\n'):
                if 'n_jobs' in line:
                    print(f"  Cell {i} RF: {line.strip()[:80]}")

        # Check bootstrap iterations
        if 'N_BOOT' in src or 'range(B)' in src or 'range(200)' in src:
            for line in src.split('\n'):
                if 'N_BOOT' in line or 'B =' in line or 'range(200)' in line or 'range(B)' in line:
                    print(f"  Cell {i} BOOT: {line.strip()[:80]}")

print("\nDone!")
