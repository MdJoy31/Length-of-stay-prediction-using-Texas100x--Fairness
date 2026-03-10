import json

nbs = [
    'final_notebooks/LOS_Prediction_Standard.ipynb',
    'final_notebooks/LOS_Prediction_Detailed.ipynb',
    'final_notebooks/Fairness_Analysis_Complete.ipynb'
]
for nb_name in nbs:
    nb = json.load(open(nb_name, 'r', encoding='utf-8'))
    short = nb_name.split('/')[-1]
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        for line in cell['source']:
            if ('RACE_MAP' in line or 'SEX_MAP' in line or 'ETH_MAP' in line) and '=' in line and '{' in line:
                print(f"  {short}: {line.strip()}")
