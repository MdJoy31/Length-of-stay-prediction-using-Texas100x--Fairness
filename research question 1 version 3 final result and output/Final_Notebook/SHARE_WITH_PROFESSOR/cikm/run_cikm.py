import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import time, sys

nb_path = 'CIKM_2026_LOS_Fairness.ipynb'
print(f'Loading {nb_path}...')
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

code_cells = [i for i, c in enumerate(nb.cells) if c.cell_type == 'code']
print(f'Total cells: {len(nb.cells)}, Code cells: {len(code_cells)}')

ep = ExecutePreprocessor(timeout=7200, kernel_name='python3')
ep.allow_errors = False

ts = time.strftime("%H:%M:%S")
print(f'Starting execution at {ts}...')
sys.stdout.flush()

start = time.time()
try:
    ep.preprocess(nb, {'metadata': {'path': '.'}})
    elapsed = time.time() - start
    print(f'\nAll cells executed successfully in {elapsed/60:.1f} min')
except Exception as e:
    elapsed = time.time() - start
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code' and 'outputs' in cell:
            for out in cell.outputs:
                if out.get('output_type') == 'error':
                    print(f'\nError in cell {i+1}: {out["ename"]}: {out["evalue"]}')
                    break
    print(f'\nExecution failed after {elapsed/60:.1f} min: {e}')

with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print('Notebook saved with outputs.')
