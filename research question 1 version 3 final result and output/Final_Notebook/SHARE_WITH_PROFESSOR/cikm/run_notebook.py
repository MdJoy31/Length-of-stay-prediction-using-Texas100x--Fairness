"""
Execute the FINAL notebook end-to-end.

KEY: writes the .ipynb to disk after EVERY cell, so the user can refresh
their IDE and see results stream in live, instead of waiting for the
whole run to finish.

Logs progress to run_notebook.log so we can monitor it.
"""
import sys, time, os
sys.stdout.reconfigure(encoding='utf-8')
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

NB = "CIKM_2026_LOS_Fairness_FINAL.ipynb"
LOG = "run_notebook.log"

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# Clear log
open(LOG, "w").close()

log(f"START loading {NB}")
nb = nbformat.read(NB, as_version=4)
n_code = sum(1 for c in nb.cells if c.cell_type == "code")
log(f"{n_code} code cells, executing fresh kernel ...")

client = NotebookClient(
    nb,
    timeout=3600,
    kernel_name="python3",
    allow_errors=False,
    record_timing=True,
)

t0 = time.time()
status = "OK"
try:
    with client.setup_kernel():
        for idx, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            tcell = time.time()
            try:
                client.execute_cell(cell, idx)
                elapsed = time.time() - tcell
                src = "".join(cell.source).split("\n")[0][:60]
                log(f"  cell[{idx:02d}] OK in {elapsed:6.1f}s | {src}")
            except CellExecutionError as e:
                elapsed = time.time() - tcell
                src = "".join(cell.source).split("\n")[0][:60]
                log(f"  cell[{idx:02d}] FAIL in {elapsed:6.1f}s | {src}")
                log(f"    error: {str(e)[:300]}")
                status = "ERROR"
                # write what we have, then stop
                nbformat.write(nb, NB)
                break
            # Write back after EVERY successful cell so user can refresh IDE
            try:
                nbformat.write(nb, NB)
            except Exception as e:
                log(f"    [warn] failed to write back: {e}")
except Exception as e:
    status = "EXCEPTION"
    log(f"EXCEPTION ({type(e).__name__}): {str(e)[:600]}")
finally:
    elapsed = time.time() - t0
    nbformat.write(nb, NB)
    sz = os.path.getsize(NB)
    log(f"WROTE {NB} in {elapsed:.1f}s total ({status})")
    log(f"Final notebook size: {sz/1024/1024:.2f} MB ({sz:,} bytes)")
