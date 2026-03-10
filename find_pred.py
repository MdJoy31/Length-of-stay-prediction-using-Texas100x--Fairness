"""Find where predictions['Fairness_Derived'] is stored."""
import json

with open("Fairness_Analysis_Complete.ipynb", 'r', encoding='utf-8') as f:
    nb = json.load(f)

code_idx = 0
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        source = ''.join(c['source'])
        code_idx += 1
        if "Fairness_Derived" in source and "predictions" in source:
            print(f"=== CODE CELL {code_idx} (cell index {i}) ===")
            # Find the relevant lines
            for line in source.split('\n'):
                if 'Fairness_Derived' in line or 'predictions' in line:
                    print(f"  {line}")
            print()
