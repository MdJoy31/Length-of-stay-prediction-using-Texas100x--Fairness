#!/usr/bin/env python3
"""Execute CIKM notebook cell-by-cell with progress reporting."""
import nbformat
from nbclient import NotebookClient
from datetime import datetime
import sys, time, traceback, re

NB = 'CIKM_2026_LOS_Fairness.ipynb'

def run():
    print(f"[{datetime.now():%H:%M:%S}] Loading {NB}...")
    nb = nbformat.read(NB, as_version=4)
    total = len(nb.cells)
    code_cells = [i for i, c in enumerate(nb.cells) if c.cell_type == 'code']
    print(f"Total cells: {total}, Code cells: {len(code_cells)}")
    sys.stdout.flush()

    client = NotebookClient(nb, timeout=7200, kernel_name='python3',
                            resources={'metadata': {'path': '.'}})

    print(f"[{datetime.now():%H:%M:%S}] Starting kernel...")
    sys.stdout.flush()

    client.create_kernel_manager()
    client.start_new_kernel()
    client.start_new_kernel_client()

    code_idx = 0
    try:
        for idx, cell in enumerate(nb.cells):
            cell_num = idx + 1
            if cell.cell_type == 'markdown':
                continue

            code_idx += 1
            first_line = cell.source.split('\n')[0][:80] if cell.source else ''
            print(f"[{datetime.now():%H:%M:%S}] Cell {cell_num}/{total} (code {code_idx}/{len(code_cells)}): {first_line}")
            sys.stdout.flush()

            try:
                t0 = time.time()
                client.execute_cell(cell, idx)
                elapsed = time.time() - t0

                has_error = any(o.get('output_type') == 'error' for o in cell.get('outputs', []))
                if has_error:
                    print(f"  ⚠ CELL ERROR after {elapsed:.1f}s")
                    for o in cell.get('outputs', []):
                        if o.get('output_type') == 'error':
                            print(f"  {o.get('ename', '?')}: {o.get('evalue', '?')}")
                            if o.get('traceback'):
                                for tb_line in o['traceback'][-3:]:
                                    clean = re.sub(r'\x1b\[[0-9;]*m', '', str(tb_line))
                                    print(f"    {clean}")
                    sys.stdout.flush()
                    raise RuntimeError(f"Cell {cell_num} had errors")
                else:
                    n_out = len(cell.get('outputs', []))
                    has_img = any('image/png' in str(o.get('data', {})) for o in cell.get('outputs', []))
                    status = f"✓ {elapsed:.1f}s | {n_out} outputs"
                    if has_img:
                        status += " (figure)"
                    print(f"  {status}")
                    sys.stdout.flush()

            except Exception as e:
                if 'Cell' in str(e) and 'had errors' in str(e):
                    raise
                print(f"  ✗ FAILED: {type(e).__name__}: {str(e)[:200]}")
                traceback.print_exc()
                sys.stdout.flush()
                raise

    except Exception as e:
        nbformat.write(nb, NB)
        print(f"\n[{datetime.now():%H:%M:%S}] Notebook saved (partial). Error at cell {cell_num}.")
        sys.exit(1)

    nbformat.write(nb, NB)
    print(f"\n[{datetime.now():%H:%M:%S}] ✓ ALL {len(code_cells)} CODE CELLS COMPLETED. Notebook saved with outputs.")
    sys.stdout.flush()

    try:
        client._cleanup_kernel()
    except Exception:
        pass

    print("\n" + "="*60)
    print("KEY RESULTS SUMMARY")
    print("="*60)
    for cell in nb.cells:
        for o in cell.get('outputs', []):
            text = o.get('text', '')
            if isinstance(text, list):
                text = ''.join(text)
            for line in text.split('\n'):
                if any(kw in line for kw in ['AUC', 'Accuracy', 'DI =', 'FAIR', 'UNFAIR',
                                              'Best model', 'Fair model', 'Trade-off', 'κ',
                                              'VFR', 'stability', 'candidates', 'selected']):
                    print(f"  {line.strip()}")

if __name__ == '__main__':
    run()
