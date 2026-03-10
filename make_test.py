import nbformat

nb = nbformat.v4.new_notebook()
nb.metadata['kernelspec'] = {'display_name': 'Python 3 (fairness)', 'language': 'python', 'name': 'fairness_env'}
nb.cells.append(nbformat.v4.new_code_cell('print("hello world")'))
nb.cells.append(nbformat.v4.new_code_cell('import pandas as pd; print(pd.__version__)'))
nb.cells.append(nbformat.v4.new_code_cell('import torch; print(torch.cuda.is_available())'))

with open('test_kernel.ipynb', 'w') as f:
    nbformat.write(nb, f)
print('test notebook created')
