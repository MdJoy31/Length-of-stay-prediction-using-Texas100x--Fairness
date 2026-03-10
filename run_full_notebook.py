"""
Execute RQ1_LOS_Fairness_Complete.ipynb from top to bottom,
saving all cell outputs back into the notebook file.
Uses nbconvert's ExecutePreprocessor with a generous timeout.
Progress is logged to run_full_notebook.log.
"""
import sys, os, time, traceback, logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    handlers=[
        logging.FileHandler('run_full_notebook.log', mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

NB_PATH = 'RQ1_LOS_Fairness_Complete.ipynb'
NB_OUT  = 'RQ1_LOS_Fairness_Complete.ipynb'   # overwrite in-place

log.info(f'Starting execution of {NB_PATH}')
log.info(f'Python: {sys.executable}')
log.info(f'CWD:    {os.getcwd()}')

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError

# Read notebook
with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

total_cells = len(nb.cells)
code_cells = [i for i, c in enumerate(nb.cells) if c.cell_type == 'code']
log.info(f'Total cells: {total_cells}, Code cells: {len(code_cells)}')

# Configure executor
ep = ExecutePreprocessor(
    timeout=7200,          # 2 hours max per cell (model training can be slow)
    kernel_name='python3',
    interrupt_on_timeout=True,
)

# Set the notebook metadata kernel to use our venv python
nb.metadata.setdefault('kernelspec', {})
nb.metadata['kernelspec']['name'] = 'python3'
nb.metadata['kernelspec']['display_name'] = 'Python 3'

t0 = time.time()
try:
    log.info('Launching kernel and executing all cells...')
    ep.preprocess(nb, {'metadata': {'path': os.getcwd()}})
    elapsed = time.time() - t0
    log.info(f'ALL CELLS EXECUTED SUCCESSFULLY in {elapsed/60:.1f} minutes')
except CellExecutionError as e:
    elapsed = time.time() - t0
    log.error(f'Cell execution error after {elapsed/60:.1f} minutes:')
    log.error(str(e)[:2000])
    log.info('Saving notebook with partial outputs...')
except Exception as e:
    elapsed = time.time() - t0
    log.error(f'Unexpected error after {elapsed/60:.1f} minutes:')
    log.error(traceback.format_exc()[:2000])
    log.info('Saving notebook with partial outputs...')

# Save notebook with outputs (even partial)
with open(NB_OUT, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
log.info(f'Notebook saved to {NB_OUT}')

# Count outputs
cells_with_output = sum(1 for c in nb.cells if c.cell_type == 'code' and c.outputs)
cells_with_images = sum(1 for c in nb.cells if c.cell_type == 'code'
                        and any('image/png' in o.get('data', {}) for o in c.outputs
                               if hasattr(o, 'get')))
log.info(f'Code cells with output: {cells_with_output}/{len(code_cells)}')
log.info(f'Code cells with images: {cells_with_images}')
log.info(f'Total time: {(time.time()-t0)/60:.1f} minutes')
log.info('Done.')
