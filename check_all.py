import json, os
for f in ['final_notebooks/LOS_Prediction_Standard.ipynb', 'final_notebooks/LOS_Prediction_Detailed.ipynb', 'Fairness_Analysis_Complete.ipynb']:
    with open(f, 'r', encoding='utf-8') as fp:
        nb = json.load(fp)
    code = [c for c in nb['cells'] if c['cell_type'] == 'code']
    executed = [c for c in code if c.get('execution_count') is not None]
    errors = sum(1 for c in code for o in c.get('outputs', []) if o.get('output_type') == 'error')
    sz = os.path.getsize(f)
    print(f"{f}: {len(nb['cells'])} total, {len(code)} code, {len(executed)} exec, {errors} err, {sz:,} bytes")
