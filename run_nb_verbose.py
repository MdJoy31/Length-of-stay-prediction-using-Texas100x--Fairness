"""
run_nb_verbose.py — Execute notebook cell by cell with progress reporting.
Usage: .venv\Scripts\python.exe run_nb_verbose.py <notebook.ipynb>
"""
import sys, os, json, time
os.chdir(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\final_notebooks")

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

nb_file = sys.argv[1] if len(sys.argv) > 1 else "LOS_Prediction_Standard.ipynb"

print(f"Loading: {nb_file}", flush=True)
with open(nb_file, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

code_cells = [(i, c) for i, c in enumerate(nb.cells) if c.cell_type == 'code']
print(f"Total cells: {len(nb.cells)}, Code cells: {len(code_cells)}", flush=True)

# Set up kernel
ep = ExecutePreprocessor(
    timeout=1800,  # 30 min per cell max
    kernel_name='fairness_env',
    allow_errors=True
)

# Start kernel using the proper API
print("Starting kernel...", flush=True)
start = time.time()
from jupyter_client.manager import KernelManager

km = KernelManager(kernel_name='fairness_env')
km.start_kernel()
kc = km.client()
kc.start_channels()
kc.wait_for_ready(timeout=60)
print("Kernel ready.", flush=True)

err_count = 0
exec_count = 0
for idx, (cell_idx, cell) in enumerate(code_cells):
    cell_start = time.time()
    first_line = cell.source.split('\n')[0][:80] if cell.source else '<empty>'
    print(f"  [{idx+1}/{len(code_cells)}] Cell {cell_idx}: {first_line}", flush=True)

    exec_count += 1
    msg_id = kc.execute(cell.source)

    # Collect outputs
    cell['outputs'] = []
    cell['execution_count'] = exec_count
    has_error = False

    while True:
        try:
            msg = kc.get_iopub_msg(timeout=1800)
        except Exception as e:
            print(f"    ⚠ Timeout/Error: {str(e)[:120]}", flush=True)
            has_error = True
            break

        msg_type = msg['msg_type']
        content = msg['content']

        if msg_type == 'status' and content.get('execution_state') == 'idle':
            break
        elif msg_type == 'execute_result':
            cell['outputs'].append(nbformat.v4.new_output(
                output_type='execute_result',
                data=content.get('data', {}),
                metadata=content.get('metadata', {}),
                execution_count=content.get('execution_count')
            ))
        elif msg_type == 'stream':
            cell['outputs'].append(nbformat.v4.new_output(
                output_type='stream',
                name=content.get('name', 'stdout'),
                text=content.get('text', '')
            ))
        elif msg_type == 'display_data':
            cell['outputs'].append(nbformat.v4.new_output(
                output_type='display_data',
                data=content.get('data', {}),
                metadata=content.get('metadata', {})
            ))
        elif msg_type == 'error':
            has_error = True
            cell['outputs'].append(nbformat.v4.new_output(
                output_type='error',
                ename=content.get('ename', 'Error'),
                evalue=content.get('evalue', ''),
                traceback=content.get('traceback', [])
            ))
            err_count += 1

    # Also get shell reply
    try:
        reply = kc.get_shell_msg(timeout=30)
        if reply['content'].get('status') == 'error' and not has_error:
            has_error = True
            err_count += 1
    except:
        pass

    elapsed = time.time() - cell_start
    status = "❌" if has_error else "✅"
    print(f"    {status} Done in {elapsed:.1f}s", flush=True)

# Shutdown kernel
try:
    kc.stop_channels()
    km.shutdown_kernel(now=True)
except:
    pass

elapsed = time.time() - start
print(f"\nExecution completed in {elapsed:.0f}s", flush=True)
print(f"Errors: {err_count}/{len(code_cells)} cells", flush=True)

# Save
with open(nb_file, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print(f"Saved: {nb_file}", flush=True)

if err_count == 0:
    print("SUCCESS: No errors!", flush=True)
else:
    print(f"ISSUES: {err_count} cells had errors", flush=True)
