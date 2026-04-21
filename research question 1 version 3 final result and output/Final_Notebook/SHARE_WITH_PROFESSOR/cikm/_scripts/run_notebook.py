"""Execute the notebook end-to-end using nbclient.

This is slow (full model training on 740K records). We redirect all output
to _scripts/notebook_run.log and exit codes are logged.
"""
import nbformat, sys, os, time, traceback
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

sys.stdout.reconfigure(encoding='utf-8')
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NB = 'CIKM_2026_LOS_Fairness.ipynb'
LOG = '_scripts/notebook_run.log'
OUT_NB = 'CIKM_2026_LOS_Fairness.executed.ipynb'

logf = open(LOG, 'w', encoding='utf-8')
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}\n"
    logf.write(line); logf.flush()
    print(line, end='')

log(f'Loading notebook {NB}')
with open(NB, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

log(f'Total cells: {len(nb.cells)}')
client = NotebookClient(nb, timeout=7200, kernel_name='python3', allow_errors=False,
                        resources={'metadata': {'path': '.'}})

start = time.time()
try:
    client.execute()
    log(f'Notebook executed successfully in {time.time()-start:.1f}s')
except CellExecutionError as e:
    log(f'CELL ERROR at {time.time()-start:.1f}s: {e}')
    traceback.print_exc(file=logf)
except Exception as e:
    log(f'EXCEPTION at {time.time()-start:.1f}s: {e}')
    traceback.print_exc(file=logf)
finally:
    with open(OUT_NB, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    log(f'Executed notebook saved to {OUT_NB}')
    logf.close()
