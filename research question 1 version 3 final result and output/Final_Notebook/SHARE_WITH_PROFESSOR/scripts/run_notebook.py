"""Run the notebook end-to-end via nbconvert."""
import nbformat, time, sys, os
from nbconvert.preprocessors import ExecutePreprocessor

# Fix encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

NB_PATH = 'RQ1_LOS_Fairness_Analysis.ipynb'

# Use the venv kernel
VENV_PYTHON = r"D:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\.venv\Scripts\python.exe"
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

print(f"Executing {NB_PATH} ...")

nb = nbformat.read(NB_PATH, as_version=4)
ep = ExecutePreprocessor(timeout=7200, kernel_name='python3')

t0 = time.time()
try:
    ep.preprocess(nb, {'metadata': {'path': '.'}})
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  ALL CELLS EXECUTED SUCCESSFULLY in {elapsed/60:.1f} min")
    print(f"{'='*60}")
except Exception as e:
    elapsed = time.time() - t0
    print(f"\nERROR after {elapsed/60:.1f} min: {e}", file=sys.stderr)
    print(f"Error: {e}")

# Count outputs
code_cells = [c for c in nb.cells if c.cell_type == 'code']
cells_with_output = sum(1 for c in code_cells if c.outputs)
cells_with_images = sum(1 for c in code_cells
                        if any('image/png' in o.get('data', {}) for o in c.outputs
                               if o.get('output_type') in ('display_data','execute_result')))

print(f"Code cells: {len(code_cells)}")
print(f"Cells with output: {cells_with_output}")
print(f"Cells with images: {cells_with_images}")

# Save executed notebook
nbformat.write(nb, NB_PATH)
print(f"Saved executed notebook: {NB_PATH}")

# Count saved files
import glob
figs = glob.glob('output/figures/*.png')
tabs = glob.glob('output/tables/*.csv')
print(f"Figures: {len(figs)}")
print(f"Tables: {len(tabs)}")
