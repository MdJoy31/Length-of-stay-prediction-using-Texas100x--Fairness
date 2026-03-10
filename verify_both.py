"""Check both notebooks for errors and get key metrics."""
import json

for nb_name in ['LOS_Prediction_Standard.ipynb', 'LOS_Prediction_Detailed.ipynb']:
    path = f"final_notebooks/{nb_name}"
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    total = len(nb['cells'])
    code = [c for c in nb['cells'] if c['cell_type'] == 'code']
    executed = [c for c in code if c.get('execution_count') is not None]
    errors = []
    for c in code:
        for o in c.get('outputs', []):
            if o.get('output_type') == 'error':
                errors.append(o.get('ename', 'Unknown'))

    print(f"\n{'='*60}")
    print(f"{nb_name}")
    print(f"  Total cells: {total}")
    print(f"  Code cells: {len(code)}")
    print(f"  Executed: {len(executed)}")
    print(f"  Errors: {len(errors)}")
    if errors:
        for e in errors[:5]:
            print(f"    - {e}")
    else:
        print(f"  STATUS: ✅ ALL PASSED")

    # File size
    import os
    size = os.path.getsize(path)
    print(f"  File size: {size:,} bytes ({size/1024/1024:.1f} MB)")

    # Extract AFCE key result
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            text_out = ""
            for o in c.get('outputs', []):
                if o.get('output_type') == 'stream':
                    text_out += ''.join(o.get('text', []))
            if 'Best fair (3/4)' in text_out:
                for line in text_out.split('\n'):
                    if 'Best fair' in line or 'Accuracy:' in line or 'RACE DI' in line or 'cost' in line:
                        print(f"  AFCE: {line.strip()}")
                break
