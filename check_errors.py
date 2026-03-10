import nbformat
nb = nbformat.read('RQ1_LOS_Fairness_Complete.ipynb', as_version=4)
code_cells = [(i, c) for i, c in enumerate(nb.cells) if c.cell_type == 'code']
print(f'Total code cells: {len(code_cells)}')
for idx, cell in code_cells:
    has_output = len(cell.outputs) > 0
    has_error = any(o.get('output_type') == 'error' for o in cell.outputs)
    if not has_output or has_error:
        src = cell.source[:100].replace('\n', ' ')
        status = 'ERROR' if has_error else 'NO OUTPUT'
        print(f'  Cell {idx+1} [{status}]: {src}...')
        if has_error:
            for o in cell.outputs:
                if o.get('output_type') == 'error':
                    ename = o.get('ename', 'Unknown')
                    evalue = o.get('evalue', '')[:300]
                    print(f'    => {ename}: {evalue}')
                    # Print last 3 traceback lines
                    tb = o.get('traceback', [])
                    if tb:
                        for line in tb[-3:]:
                            # strip ANSI codes
                            import re
                            clean = re.sub(r'\x1b\[[0-9;]*m', '', str(line))
                            print(f'    TB: {clean[:200]}')
