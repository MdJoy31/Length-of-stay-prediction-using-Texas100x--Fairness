"""
execute_notebook.py — Execute a notebook using nbconvert with progress logging.
Usage: .venv\Scripts\python.exe execute_notebook.py <notebook.ipynb>
Logs progress to execute_log.txt in the workspace root.
"""
import sys, os, time, json, gc

os.chdir(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\final_notebooks")
LOG = r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\execute_log.txt"

nb_file = sys.argv[1] if len(sys.argv) > 1 else "LOS_Prediction_Standard.ipynb"

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

# Clear log
with open(LOG, 'w') as f:
    f.write('')

log(f"Starting execution: {nb_file}")

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

log("Loading notebook...")
with open(nb_file, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

code_cells = [i for i, c in enumerate(nb.cells) if c.cell_type == 'code']
log(f"Total cells: {len(nb.cells)}, Code cells: {len(code_cells)}")

# Configure executor
ep = ExecutePreprocessor(
    timeout=1800,  # 30 min per cell
    kernel_name='fairness_env',
    allow_errors=True
)

log("Starting kernel and executing...")
start = time.time()

try:
    ep.preprocess(nb, {'metadata': {'path': os.getcwd()}})
    elapsed = time.time() - start
    log(f"Execution completed in {elapsed:.0f}s")
except Exception as e:
    elapsed = time.time() - start
    log(f"ERROR after {elapsed:.0f}s: {type(e).__name__}: {str(e)[:200]}")

# Count errors
err_count = 0
for i, c in enumerate(nb.cells):
    if c.cell_type != 'code':
        continue
    for o in c.get('outputs', []):
        if o.get('output_type') == 'error':
            ename = o.get('ename', '?')
            evalue = str(o.get('evalue', ''))[:80]
            log(f"  Cell {i}: {ename} - {evalue}")
            err_count += 1
            break

log(f"Errors: {err_count}/{len(code_cells)} cells")

# Save
with open(nb_file, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
log(f"Saved: {nb_file}")

if err_count == 0:
    log("SUCCESS: No errors!")
else:
    log(f"ISSUES: {err_count} cells had errors")

# Cleanup
gc.collect()
log("Done.")
