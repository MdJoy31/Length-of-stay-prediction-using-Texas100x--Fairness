"""Execute a notebook using nbconvert's ExecutePreprocessor."""
import sys, os, json, time

os.chdir(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\final_notebooks")

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

nb_file = sys.argv[1] if len(sys.argv) > 1 else "LOS_Prediction_Standard.ipynb"
output_file = nb_file  # overwrite in-place

print(f"Loading: {nb_file}")
with open(nb_file, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

print(f"Cells: {len(nb.cells)} ({sum(1 for c in nb.cells if c.cell_type=='code')} code)")

ep = ExecutePreprocessor(
    timeout=3600,
    kernel_name='fairness_env',
    allow_errors=True
)

start = time.time()
print(f"Executing... (this may take 10-20 minutes)")
try:
    ep.preprocess(nb, {'metadata': {'path': '.'}})
    elapsed = time.time() - start
    print(f"Execution completed in {elapsed:.0f}s")
except Exception as e:
    elapsed = time.time() - start
    print(f"Execution error after {elapsed:.0f}s: {e}")

# Save
with open(output_file, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print(f"Saved: {output_file}")

# Check for errors
error_count = 0
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code':
        for output in cell.get('outputs', []):
            if output.get('output_type') == 'error':
                error_count += 1
                ename = output.get('ename', '?')
                evalue = output.get('evalue', '?')
                print(f"  ERROR in cell {i}: {ename}: {evalue[:100]}")
                break

if error_count == 0:
    print("SUCCESS: No errors found!")
else:
    print(f"ISSUES: {error_count} cells had errors")
