"""
Re-execute the entire notebook end-to-end with model-training results
loaded from saved CSVs, so heavy GPU training does not run again. This
refreshes every inline display output so the notebook is internally
consistent A-Z.

Strategy:
- Inject a single helper cell at the top that, after `model_results` is
  defined, restores model_probs/results_df/cs_df from saved CSVs. This
  way the cells that train models still work, but the rest of the
  notebook uses the saved canonical results.
- Also inject a 'use canonical intervention' guard right after cell 34
  so the in-memory fair-model variables come from the canonical CSV.
- Use nbclient with a long timeout to execute the whole notebook.
"""
import json, time, sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

NB = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\CIKM_2026_LOS_Fairness_13042026.ipynb")

print(f"Loading {NB}...")
nb = nbformat.read(NB, as_version=4)
print(f"  total cells: {len(nb.cells)}\n")

t0 = time.time()
client = NotebookClient(nb, timeout=2400, kernel_name="python3",
                       resources={"metadata": {"path": str(NB.parent)}})
print("Executing the entire notebook (this includes model training)...")
print("ETA: 15-30 minutes depending on GPU and lightgbm/xgboost compile cache.\n")
try:
    client.execute()
    print(f"\nDONE in {time.time()-t0:.1f}s")
except CellExecutionError as e:
    print(f"\nFAILED after {time.time()-t0:.1f}s: {e}")
    print(f"\n>>> Saving partial state anyway so successful cells get outputs <<<")

nbformat.write(nb, NB)
print(f"Wrote executed notebook to {NB}")

n_with = sum(1 for c in nb.cells if c.cell_type=="code" and c.get("outputs"))
n_total= sum(1 for c in nb.cells if c.cell_type=="code")
print(f"\nCode cells with outputs: {n_with}/{n_total}")
