"""Extract key metrics from executed notebook outputs."""
import json, sys, re

nb_file = sys.argv[1] if len(sys.argv) > 1 else "final_notebooks/LOS_Prediction_Standard.ipynb"

with open(nb_file, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        text_out = ""
        for o in c.get('outputs', []):
            if o.get('output_type') == 'stream':
                text_out += ''.join(o.get('text', []))

        # Look for key metrics
        if 'Best Model:' in text_out or 'BEST MODEL' in text_out:
            print(f"=== Cell {i} (Performance) ===")
            for line in text_out.split('\n'):
                if any(kw in line for kw in ['Best Model', 'Accuracy', 'AUC', 'F1', 'BEST']):
                    print(f"  {line.strip()}")

        if 'Fair model' in text_out or 'Fairness-Derived' in text_out or 'fair_acc' in text_out:
            if 'Accuracy:' in text_out:
                print(f"=== Cell {i} (Fair Model) ===")
                for line in text_out.split('\n'):
                    if any(kw in line for kw in ['Accuracy', 'F1', 'AUC', 'DI', 'WTPR', 'threshold']):
                        print(f"  {line.strip()}")

        if 'AFCE' in text_out and ('accuracy' in text_out.lower() or 'acc' in text_out.lower()):
            print(f"=== Cell {i} (AFCE) ===")
            for line in text_out.split('\n'):
                line = line.strip()
                if line and ('0.' in line or 'RACE' in line or 'Best' in line or 'α=' in line or 'fair' in line.lower()):
                    print(f"  {line}")
