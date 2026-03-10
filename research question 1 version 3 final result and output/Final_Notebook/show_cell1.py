"""Print cell 1 source."""
import nbformat
nb = nbformat.read('RQ1_LOS_Fairness_Analysis.ipynb', as_version=4)
code_cells = [c for c in nb.cells if c.cell_type == 'code']
src = code_cells[0].source
print(f'Lines: {len(src.splitlines())}')
for i, line in enumerate(src.splitlines()[:30], 1):
    print(f'{i:3d}: {line}')
