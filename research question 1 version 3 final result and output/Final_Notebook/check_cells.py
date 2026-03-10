"""Check all notebook cells for syntax errors."""
import nbformat, ast, sys

nb = nbformat.read('RQ1_LOS_Fairness_Analysis.ipynb', as_version=4)
code_cells = [c for c in nb.cells if c.cell_type == 'code']
print(f'Total code cells: {len(code_cells)}')

errors = []
for i, c in enumerate(code_cells):
    src = c.source.strip()
    if not src:
        continue
    try:
        ast.parse(src)
    except SyntaxError as e:
        errors.append((i+1, str(e)[:150], src[:200]))

if errors:
    print(f'\nSYNTAX ERRORS in {len(errors)} cells:')
    for idx, err, preview in errors:
        print(f'\n  Cell {idx}: {err}')
        print(f'  Preview: {preview[:100]}...')
else:
    print('All cells parse OK')
