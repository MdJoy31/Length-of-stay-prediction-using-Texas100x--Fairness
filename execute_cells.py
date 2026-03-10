"""Execute remaining notebook cells one by one using jupyter_client directly.
Resumes from cell 13 onwards (first 12 already executed by papermill).
"""
import json
import time
import sys
from jupyter_client import KernelManager

NB_PATH = 'LOS_Prediction_Detailed.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find code cells
code_cells = [(i, c) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']
print(f"Total code cells: {len(code_cells)}")

# Find first cell without output (resume point)
resume_from = 0
for idx, (i, c) in enumerate(code_cells):
    if not c['outputs']:
        resume_from = idx
        break

print(f"Resuming from code cell {resume_from + 1} (notebook index {code_cells[resume_from][0]})")

# First, we need to re-execute all prior cells to rebuild state
# But that would take too long. Instead, execute ALL cells from start.
# Actually, let's just execute from the beginning to ensure state is good.

# Start kernel
km = KernelManager(kernel_name='python3')
km.start_kernel()
kc = km.client()
kc.start_channels()
kc.wait_for_ready(timeout=60)
print("Kernel ready!")

def execute_cell(source, timeout=7200):
    """Execute a cell and return outputs."""
    msg_id = kc.execute(source)
    outputs = []

    deadline = time.time() + timeout
    while True:
        if time.time() > deadline:
            return outputs, "TIMEOUT"

        try:
            msg = kc.get_iopub_msg(timeout=10)
        except Exception:
            # Check if kernel is still alive
            if not km.is_alive():
                return outputs, "KERNEL_DEAD"
            continue

        msg_type = msg['msg_type']
        content = msg['content']
        parent_id = msg.get('parent_header', {}).get('msg_id', '')

        if parent_id != msg_id:
            continue

        if msg_type == 'stream':
            outputs.append({
                'output_type': 'stream',
                'name': content.get('name', 'stdout'),
                'text': content.get('text', '')
            })
            # Print progress
            text = content.get('text', '').strip()
            if text:
                for line in text.split('\n')[-3:]:  # Last 3 lines
                    print(f"    {line[:120]}")

        elif msg_type in ('display_data', 'execute_result'):
            out = {
                'output_type': msg_type,
                'data': content.get('data', {}),
                'metadata': content.get('metadata', {}),
            }
            if msg_type == 'execute_result':
                out['execution_count'] = content.get('execution_count')
            outputs.append(out)

        elif msg_type == 'error':
            outputs.append({
                'output_type': 'error',
                'ename': content.get('ename', ''),
                'evalue': content.get('evalue', ''),
                'traceback': content.get('traceback', [])
            })
            print(f"    ERROR: {content.get('ename')}: {content.get('evalue', '')[:200]}")
            return outputs, "ERROR"

        elif msg_type == 'status' and content.get('execution_state') == 'idle':
            return outputs, "OK"

    return outputs, "UNKNOWN"

# Execute all cells from the beginning
total = len(code_cells)
for idx, (cell_idx, cell) in enumerate(code_cells):
    code_num = idx + 1
    src = ''.join(cell['source'])
    first_line = cell['source'][0].strip()[:60] if cell['source'] else ''

    print(f"\n[{code_num}/{total}] Cell idx={cell_idx}: {first_line}")

    start = time.time()
    outputs, status = execute_cell(src, timeout=7200)
    elapsed = time.time() - start

    print(f"  => {status} ({elapsed:.1f}s, {len(outputs)} outputs)")

    # Save outputs to notebook
    cell['outputs'] = outputs
    cell['execution_count'] = code_num

    # Save notebook after each cell
    with open(NB_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    if status in ("ERROR", "KERNEL_DEAD", "TIMEOUT"):
        print(f"\n*** STOPPED at cell {code_num} due to {status} ***")
        if status == "KERNEL_DEAD":
            print("Kernel died! Cannot continue.")
            break
        # For errors, continue to next cell (some errors are recoverable)
        if status == "ERROR":
            continue
        break

# Cleanup
try:
    kc.stop_channels()
    km.shutdown_kernel(now=True)
except:
    pass

print(f"\nDone! Executed {total} cells.")
