"""Fix LightGBM threading: add n_jobs=1 to LGBMClassifier in all notebooks."""
import json

OLD = "device='cpu', random_state=42, verbose=-1"
NEW = "device='cpu', n_jobs=1, random_state=42, verbose=-1"

for nb_name in ['final_notebooks/LOS_Prediction_Detailed.ipynb',
                'final_notebooks/Fairness_Analysis_Complete.ipynb',
                'final_notebooks/LOS_Prediction_Standard.ipynb']:
    with open(nb_name, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    changed = False
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src = cell['source']
        is_list = isinstance(src, list)
        text = ''.join(src) if is_list else src

        if 'LGBMClassifier' in text and 'n_jobs=1' not in text:
            if is_list:
                new_src = [line.replace(OLD, NEW) for line in src]
                if new_src != src:
                    cell['source'] = new_src
                    changed = True
                    print(f"  Fixed (list): {nb_name} Cell {i}")
            else:
                new_text = text.replace(OLD, NEW)
                if new_text != text:
                    cell['source'] = new_text
                    changed = True
                    print(f"  Fixed (str): {nb_name} Cell {i}")
        elif 'LGBMClassifier' in text and 'n_jobs=1' in text:
            print(f"  Already fixed: {nb_name} Cell {i}")

    if changed:
        with open(nb_name, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"✅ Saved: {nb_name}")
    else:
        print(f"No changes: {nb_name}")

print("\nDone!")
