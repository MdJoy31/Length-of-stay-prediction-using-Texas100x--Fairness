import json
with open('test_kernel.ipynb','r') as f:
    nb = json.load(f)
for i,c in enumerate(nb['cells']):
    ec = c.get('execution_count')
    print(f'Cell {i}: exec_count={ec}')
    for o in c.get('outputs',[]):
        ot = o.get('output_type','')
        if ot == 'stream':
            txt = o["text"] if isinstance(o["text"], str) else ''.join(o["text"])
            print(f'  output: {txt.strip()}')
        elif ot == 'error':
            print(f'  ERROR: {o["ename"]}: {str(o["evalue"])[:80]}')
