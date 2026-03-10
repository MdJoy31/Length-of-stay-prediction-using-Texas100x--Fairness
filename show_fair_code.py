"""Extract fair model training and prediction storage code from Complete notebook."""
import json

with open("Fairness_Analysis_Complete.ipynb", 'r', encoding='utf-8') as f:
    nb = json.load(f)

code_idx = 0
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        source = ''.join(c['source'])
        code_idx += 1
        # Code cells 27-29 (fair model)
        if code_idx in [27, 28, 29]:
            print(f"=== CODE CELL {code_idx} (cell index {i}) ===")
            print(source[:2500])
            if len(source) > 2500:
                print("... (truncated)")
            print()
