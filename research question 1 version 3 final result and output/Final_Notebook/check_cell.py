import nbformat
nb = nbformat.read('RQ1_LOS_Fairness_Analysis.ipynb', as_version=4)
code_cells = [c for c in nb.cells if c.cell_type == 'code']
c = code_cells[15]
print(f"Cell 16 source (first 3 lines): {c.source.split(chr(10))[:3]}")
print(f"Outputs: {len(c.outputs)}")
for j, o in enumerate(c.outputs):
    otype = o.get('output_type', '?')
    if otype == 'stream':
        txt = o.get('text', '')[-800:]
        print(f"Output {j} ({otype}): ...{txt}")
    elif otype == 'error':
        ename = o.get('ename', '')
        evalue = o.get('evalue', '')[:500]
        tb = o.get('traceback', [])
        print(f"Output {j} ERROR: {ename}: {evalue}")
        for line in tb[-5:]:
            print(line[:200])
    else:
        print(f"Output {j}: type={otype}")
