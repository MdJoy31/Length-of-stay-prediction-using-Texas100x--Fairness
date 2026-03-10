"""Check executed notebook for errors."""
import json, sys

nb_file = sys.argv[1] if len(sys.argv) > 1 else "final_notebooks/LOS_Prediction_Standard.ipynb"

with open(nb_file, 'r', encoding='utf-8') as f:
    nb = json.load(f)

errors = 0
code_cells = 0
executed = 0

for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        code_cells += 1
        if c.get('execution_count') is not None:
            executed += 1
        for o in c.get('outputs', []):
            if o.get('output_type') == 'error':
                errors += 1
                ename = o.get('ename', '?')
                evalue = str(o.get('evalue', '?'))[:120]
                print(f"  ERROR cell {i} (exec#{c.get('execution_count')}): {ename}: {evalue}")
                break

print(f"\nSummary: {len(nb['cells'])} total, {code_cells} code, {executed} executed, {errors} errors")

if errors == 0:
    print("SUCCESS: All code cells executed without errors!")
