"""Extract specific code cells from notebook."""
import json, sys

nb_file = sys.argv[1]
with open(nb_file, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find fair model training cell and comparison cell
code_idx = 0
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        code_idx += 1
        source = ''.join(c['source'])

        # Fair model cells (~cells 37-39)
        if 'Fairness_Derived' in source and 'Per-Group' in source:
            print(f"=== CELL {i} (code #{code_idx}): Fair Model Threshold Optimization ===")
            print(source[:2000])
            print("..." if len(source) > 2000 else "")
            print()
            # Print output too
            text_out = ""
            for o in c.get('outputs', []):
                if o.get('output_type') == 'stream':
                    text_out += ''.join(o.get('text', []))
            if text_out:
                print(f">>> OUTPUT:")
                print(text_out[:2000])
                print()

        # Comparison cell
        if 'Standard Model vs Fairness' in source:
            print(f"=== CELL {i} (code #{code_idx}): Standard vs Fair Comparison ===")
            print(source[:2000])
            print()
            text_out = ""
            for o in c.get('outputs', []):
                if o.get('output_type') == 'stream':
                    text_out += ''.join(o.get('text', []))
            if text_out:
                print(f">>> OUTPUT:")
                print(text_out[:2000])
                print()
