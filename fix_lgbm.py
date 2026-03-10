import json

for nb_path in ['final_notebooks/LOS_Prediction_Standard.ipynb',
                'final_notebooks/LOS_Prediction_Detailed.ipynb',
                'final_notebooks/Fairness_Analysis_Complete.ipynb']:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    changed = False
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = cell['source']
        if isinstance(src, list):
            new_src = []
            for line in src:
                if "device='gpu'" in line:
                    line = line.replace("device='gpu'", "device='cpu'")
                    changed = True
                new_src.append(line)
            cell['source'] = new_src
        elif isinstance(src, str):
            if "device='gpu'" in src:
                cell['source'] = src.replace("device='gpu'", "device='cpu'")
                changed = True

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    status = "fixed device='gpu' -> device='cpu'" if changed else "no changes"
    print(f'{nb_path}: {status}')

print('Done.')
