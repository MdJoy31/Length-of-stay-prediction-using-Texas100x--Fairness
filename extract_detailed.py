"""Extract detailed metrics from executed notebook."""
import json, sys

nb_file = sys.argv[1]
with open(nb_file, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        text_out = ""
        for o in c.get('outputs', []):
            if o.get('output_type') == 'stream':
                text_out += ''.join(o.get('text', []))

        # Find Standard vs Fair comparison
        if 'Standard Model vs Fairness' in text_out:
            print(f"=== Cell {i} (Std vs Fair) ===")
            for line in text_out.split('\n'):
                if line.strip():
                    print(f"  {line.rstrip()}")
            print()

        # Find training results
        if 'MODEL PERFORMANCE' in text_out:
            print(f"=== Cell {i} (Model Performance) ===")
            for line in text_out.split('\n'):
                if 'Best' in line or 'MODEL' in line:
                    print(f"  {line.rstrip()}")
            print()

        # Find overfitting assessment
        if 'Overfitting Assessment' in text_out:
            print(f"=== Cell {i} (Overfitting) ===")
            for line in text_out.split('\n'):
                if 'Gap=' in line or 'Overfitting' in line:
                    print(f"  {line.rstrip()}")
            print()

        # Training results per model
        if 'Accuracy' in text_out and 'Train' in text_out and 'Test' in text_out and 'Gap' in text_out:
            # Check for training table
            lines = text_out.split('\n')
            for j, line in enumerate(lines):
                if 'Training:' in line and ('─' in lines[j+1] if j+1 < len(lines) else False):
                    print(f"=== Cell {i} (Training: {line.strip()}) ===")
                    for k in range(j, min(j+12, len(lines))):
                        if lines[k].strip():
                            print(f"  {lines[k].rstrip()}")
                    print()
                    break

