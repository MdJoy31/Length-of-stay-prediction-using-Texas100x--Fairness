"""Extract cell 39 display outputs (HTML/formatted tables)."""
import json, sys

nb_file = sys.argv[1]
with open(nb_file, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Check ALL outputs from cells 37-42 (fair model + comparison)
code_idx = 0
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        code_idx += 1
        if 36 <= i <= 42:
            outputs = c.get('outputs', [])
            print(f"=== CELL {i} (code #{code_idx}) - outputs: {len(outputs)} ===")
            for j, o in enumerate(outputs):
                otype = o.get('output_type', 'unknown')
                if otype == 'stream':
                    txt = ''.join(o.get('text', []))
                    print(f"  [stream] {txt[:500]}")
                elif otype == 'display_data' or otype == 'execute_result':
                    data = o.get('data', {})
                    if 'text/plain' in data:
                        txt = ''.join(data['text/plain'])
                        print(f"  [{otype}/text/plain] {txt[:800]}")
                    if 'text/html' in data:
                        html = ''.join(data['text/html'])
                        # Strip HTML tags for readability
                        import re
                        clean = re.sub(r'<[^>]+>', ' ', html)
                        clean = re.sub(r'\s+', ' ', clean).strip()
                        print(f"  [{otype}/html] {clean[:1500]}")
                    if 'image/png' in data:
                        print(f"  [{otype}/image] (PNG figure)")
                else:
                    print(f"  [{otype}]")
            print()
