"""
exec_nb.py — Robust cell-by-cell notebook executor with file logging.
Usage: .venv\Scripts\python.exe exec_nb.py <notebook.ipynb>
Logs to exec_log.txt in workspace root.
"""
import sys, os, time, json, gc, asyncio

# Fix Windows event loop issue
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

WORKSPACE = r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1"
NB_DIR = os.path.join(WORKSPACE, "final_notebooks")
LOG = os.path.join(WORKSPACE, "exec_log.txt")

os.chdir(NB_DIR)

nb_file = sys.argv[1] if len(sys.argv) > 1 else "LOS_Prediction_Standard.ipynb"

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

# Clear log
with open(LOG, 'w') as f:
    f.write('')

log(f"=== Starting: {nb_file} ===")

import nbformat
from jupyter_client.manager import KernelManager

# Load notebook
log("Loading notebook...")
with open(nb_file, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Clear all outputs first
for cell in nb.cells:
    if cell.cell_type == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

code_cells = [(i, c) for i, c in enumerate(nb.cells) if c.cell_type == 'code']
log(f"Total cells: {len(nb.cells)}, Code cells: {len(code_cells)}")

# Start kernel
log("Starting kernel...")
start = time.time()
km = KernelManager(kernel_name='fairness_env')
km.start_kernel(cwd=NB_DIR)
kc = km.client()
kc.start_channels()
kc.wait_for_ready(timeout=120)
log("Kernel ready.")

err_count = 0
CELL_TIMEOUT = 1800  # 30 min per cell

for idx, (cell_idx, cell) in enumerate(code_cells):
    cell_start = time.time()
    src = cell.source
    first_line = src.split('\n')[0][:80] if src else '<empty>'
    log(f"[{idx+1}/{len(code_cells)}] Cell {cell_idx}: {first_line}")

    # Send execution request
    msg_id = kc.execute(src)

    # Collect outputs
    outputs = []
    has_error = False
    exec_count_val = idx + 1

    # Wait for execution to complete by reading iopub messages
    while True:
        try:
            msg = kc.get_iopub_msg(timeout=CELL_TIMEOUT)
        except Exception as e:
            log(f"  TIMEOUT after {CELL_TIMEOUT}s: {str(e)[:100]}")
            has_error = True
            break

        # Only process messages for this execution request
        if msg['parent_header'].get('msg_id') != msg_id:
            continue

        msg_type = msg['msg_type']
        content = msg['content']

        if msg_type == 'status' and content.get('execution_state') == 'idle':
            break
        elif msg_type == 'execute_input':
            exec_count_val = content.get('execution_count', exec_count_val)
        elif msg_type == 'execute_result':
            outputs.append(nbformat.v4.new_output(
                output_type='execute_result',
                data=content.get('data', {}),
                metadata=content.get('metadata', {}),
                execution_count=content.get('execution_count')
            ))
        elif msg_type == 'stream':
            outputs.append(nbformat.v4.new_output(
                output_type='stream',
                name=content.get('name', 'stdout'),
                text=content.get('text', '')
            ))
        elif msg_type == 'display_data':
            outputs.append(nbformat.v4.new_output(
                output_type='display_data',
                data=content.get('data', {}),
                metadata=content.get('metadata', {})
            ))
        elif msg_type == 'error':
            has_error = True
            outputs.append(nbformat.v4.new_output(
                output_type='error',
                ename=content.get('ename', 'Error'),
                evalue=content.get('evalue', ''),
                traceback=content.get('traceback', [])
            ))

    # Wait for shell reply
    try:
        reply = kc.get_shell_msg(timeout=60)
        if reply['content'].get('status') == 'error' and not has_error:
            has_error = True
    except:
        pass

    # Update cell
    cell['outputs'] = outputs
    cell['execution_count'] = exec_count_val

    if has_error:
        err_count += 1

    elapsed = time.time() - cell_start
    status = "ERR" if has_error else "OK"
    log(f"  {status} ({elapsed:.1f}s)")

    # Save checkpoint every 5 cells or on error
    if (idx + 1) % 5 == 0 or has_error:
        with open(nb_file, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        log(f"  [checkpoint saved: {idx+1}/{len(code_cells)}]")

# Shutdown kernel
log("Shutting down kernel...")
try:
    kc.stop_channels()
    km.shutdown_kernel(now=True)
except:
    pass

elapsed = time.time() - start
log(f"Completed in {elapsed:.0f}s")
log(f"Errors: {err_count}/{len(code_cells)} cells")

# Final save
with open(nb_file, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
log(f"Saved: {nb_file}")

if err_count == 0:
    log("SUCCESS!")
else:
    log(f"ISSUES: {err_count} cells had errors")
