"""Reduce bootstrap iterations from 200 to 100 in all notebooks for faster execution.
Also ensure all outputs are cleared before re-execution."""
import json

for nb_name in ['final_notebooks/LOS_Prediction_Standard.ipynb',
                'final_notebooks/LOS_Prediction_Detailed.ipynb',
                'final_notebooks/Fairness_Analysis_Complete.ipynb']:
    with open(nb_name, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    changes = []
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            # Clear outputs
            cell['outputs'] = []
            cell['execution_count'] = None

            src = cell['source']
            is_list = isinstance(src, list)
            text = ''.join(src) if is_list else src

            # Reduce bootstrap from 200 to 100
            if 'B = 200' in text:
                new_text = text.replace('B = 200', 'B = 100')
                if is_list:
                    cell['source'] = [line.replace('B = 200', 'B = 100') for line in src]
                else:
                    cell['source'] = new_text
                changes.append(f"  Cell {i}: B = 200 -> B = 100")

    with open(nb_name, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"\n{nb_name}: Outputs cleared.")
    for c in changes:
        print(c)

print("\nDone!")
