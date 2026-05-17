"""
Execute Sections 17, 18, 19 in a Jupyter kernel and write the outputs
(tables, images, stdout) directly into the notebook JSON. Sections 1-16
are NOT re-run — they rely on hours of training that's already in the CSVs
the new sections read. The new cells are self-contained: they import
matplotlib/pandas, load CSVs from disk, and produce all outputs.
"""
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from pathlib import Path
import sys, time

NB = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\CIKM_2026_LOS_Fairness_13042026.ipynb")

print(f"Loading notebook: {NB}")
nb = nbformat.read(NB, as_version=4)
print(f"  total cells: {len(nb.cells)}")

# Find Section 17 start
start = None
for i, c in enumerate(nb.cells):
    src = "".join(c.get("source", []))
    if c.cell_type == "markdown" and "17. Paper-Ready Tables" in src:
        start = i
        break
print(f"  Section 17 starts at cell {start}")

# Build a slim "exec only sections 17-19" notebook so the kernel does not
# re-run the heavy training cells. We grab the new section cells and run
# them in their own temp notebook, then copy outputs back.
sub = nbformat.v4.new_notebook()
sub.metadata = nb.metadata
sub.cells = nb.cells[start:]
print(f"  cells to execute: {len(sub.cells)}")
print(f"  code cells:       {sum(1 for c in sub.cells if c.cell_type=='code')}")

t0 = time.time()
client = NotebookClient(sub, timeout=600, kernel_name="python3",
                       resources={"metadata": {"path": str(NB.parent)}})
print("\nExecuting...")
try:
    client.execute()
    print(f"  ok in {time.time()-t0:.1f}s")
except CellExecutionError as e:
    print(f"  CELL ERROR after {time.time()-t0:.1f}s: {e}")
    raise

# Write executed cells back into the original notebook
nb.cells[start:] = sub.cells
nbformat.write(nb, NB)
print(f"\nWrote executed notebook back to {NB}")

# Quick stats on what got written
n_with_outputs = sum(1 for c in nb.cells[start:] if c.cell_type=="code" and c.get("outputs"))
n_total_outputs = sum(len(c.get("outputs", [])) for c in nb.cells[start:] if c.cell_type=="code")
print(f"  code cells with outputs: {n_with_outputs}")
print(f"  total output items: {n_total_outputs}")
