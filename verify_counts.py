import json

for nb_name in ['final_notebooks/LOS_Prediction_Standard.ipynb',
                'final_notebooks/LOS_Prediction_Detailed.ipynb']:
    nb = json.load(open(nb_name, 'r', encoding='utf-8'))
    cells = nb['cells']
    code = [c for c in cells if c['cell_type'] == 'code']
    ex = [c for c in code if c.get('execution_count')]
    err = [i for i, c in enumerate(code) if any(o.get('output_type') == 'error' for o in c.get('outputs', []))]
    md = [c for c in cells if c['cell_type'] == 'markdown']
    short = nb_name.split('/')[-1]
    print(f"{short}: {len(cells)} total, {len(code)} code, {len(md)} md, {len(ex)} exec, {len(err)} err")
