"""Run the notebook end-to-end via nbconvert with progress logging."""
import nbformat, time, sys, os, asyncio, glob
from nbconvert.preprocessors import ExecutePreprocessor

# Fix Windows Proactor event loop / ZMQ compatibility
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Fix encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

NB_PATH = 'RQ1_LOS_Fairness_Analysis.ipynb'
LOG_FILE = 'execution_log_v5.txt'

os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

# Clear log
with open(LOG_FILE, 'w', encoding='utf-8') as f:
    f.write('')

log(f"Executing {NB_PATH} ...")

nb = nbformat.read(NB_PATH, as_version=4)

total_code = sum(1 for c in nb.cells if c.cell_type == 'code')
log(f"Total cells: {len(nb.cells)}, Code cells: {total_code}")

# Custom preprocessor that logs each cell
class LoggingExecutePreprocessor(ExecutePreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._code_idx = 0
        self._total = total_code

    def preprocess_cell(self, cell, resources, index):
        if cell.cell_type == 'code':
            self._code_idx += 1
            lines = cell.source.split('\n')
            header = lines[1].strip()[:70] if len(lines) > 1 else lines[0][:70]
            log(f"  [{self._code_idx}/{self._total}] Executing: {header} ...")
            t0 = time.time()
            result = super().preprocess_cell(cell, resources, index)
            elapsed = time.time() - t0
            log(f"  [{self._code_idx}/{self._total}] Done in {elapsed:.1f}s")
            return result
        return super().preprocess_cell(cell, resources, index)

ep = LoggingExecutePreprocessor(timeout=7200, kernel_name='python3')

t0 = time.time()
try:
    ep.preprocess(nb, {'metadata': {'path': '.'}})
    elapsed = time.time() - t0
    log(f"\n{'='*60}")
    log(f"  ALL CELLS EXECUTED SUCCESSFULLY in {elapsed/60:.1f} min")
    log(f"{'='*60}")
except Exception as e:
    elapsed = time.time() - t0
    log(f"\nERROR after {elapsed/60:.1f} min: {e}")

# Count outputs
code_cells = [c for c in nb.cells if c.cell_type == 'code']
cells_with_output = sum(1 for c in code_cells if c.outputs)
cells_with_images = sum(1 for c in code_cells
                        if any('image/png' in o.get('data', {}) for o in c.outputs
                               if o.get('output_type') in ('display_data','execute_result')))

log(f"Code cells: {len(code_cells)}")
log(f"Cells with output: {cells_with_output}")
log(f"Cells with images: {cells_with_images}")

# Save executed notebook (even partial)
nbformat.write(nb, NB_PATH)
log(f"Saved executed notebook: {NB_PATH}")

# Count saved files
figs = glob.glob('output/figures/*.png')
tabs = glob.glob('output/tables/*.csv')
log(f"Figures: {len(figs)}")
log(f"Tables: {len(tabs)}")
