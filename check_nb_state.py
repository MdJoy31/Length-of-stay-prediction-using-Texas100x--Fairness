import nbformat
nb = nbformat.read('research question 1 version 3 final result and output/Final_Notebook/SHARE_WITH_PROFESSOR/cikm/CIKM_2026_LOS_Fairness.ipynb', 4)
for i, c in enumerate(nb.cells):
    if c.cell_type == 'code':
        ec = c.get('execution_count', None)
        n_out = len(c.get('outputs', []))
        print(f"Cell {i:2d} (code): exec_count={ec}, outputs={n_out}")
