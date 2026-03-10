"""Extract full output of specific cells."""
import json, sys

nb_file = sys.argv[1]
with open(nb_file, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 39 - Standard vs Fair comparison
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        text_out = ""
        for o in c.get('outputs', []):
            if o.get('output_type') == 'stream':
                text_out += ''.join(o.get('text', []))

        if 'Standard Model vs Fairness' in text_out:
            print(f"=== CELL {i}: Standard vs Fair (FULL) ===")
            print(text_out)
            print("=" * 60)

        # Also get the fairness-derived model details (per-group thresholds)
        if 'Per-Group Threshold Optimization' in text_out or 'group_results' in text_out or 'Fairness-Derived' in text_out:
            print(f"=== CELL {i}: Fair Model Details ===")
            for line in text_out.split('\n'):
                if any(kw in line for kw in ['DI', 'Threshold', 'Accuracy', 'F1', 'AUC', 'Group', 'RACE', 'SEX', 'ETH', 'AGE', 'TPR', '→', 'Overall']):
                    print(f"  {line.rstrip()}")
            print()

        # Also check AFCE results
        if 'AFCE' in text_out and 'α' in text_out:
            print(f"=== CELL {i}: AFCE Results ===")
            print(text_out[:3000])
            print()
