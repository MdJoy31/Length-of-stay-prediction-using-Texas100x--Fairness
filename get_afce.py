"""Extract AFCE cell output from executed notebook."""
import json, sys

nb_file = sys.argv[1] if len(sys.argv) > 1 else "final_notebooks/LOS_Prediction_Standard.ipynb"
with open(nb_file, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        text_out = ""
        for o in c.get('outputs', []):
            if o.get('output_type') == 'stream':
                text_out += ''.join(o.get('text', []))
        if 'AFCE' in text_out and 'Step' in text_out:
            print(f"=== CELL {i}: AFCE Results ===")
            print(text_out)
            break
