import nbformat

with open('final_notebooks/LOS_Prediction_Standard.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

try:
    nbformat.validate(nb)
    print('Notebook is VALID')
except Exception as e:
    print(f'INVALID: {e}')

ks = nb.metadata.get('kernelspec', {})
print(f'Kernelspec: name={ks.get("name","?")} display={ks.get("display_name","?")}')

code_cells = [c for c in nb.cells if c.cell_type == 'code']
print(f'Code cells: {len(code_cells)}')

list_src = sum(1 for c in code_cells if isinstance(c.source, list))
str_src = sum(1 for c in code_cells if isinstance(c.source, str))
print(f'Source types: {list_src} list, {str_src} str')

# Check cell 11 (model config) for device setting
for i, c in enumerate(code_cells):
    src = c.source if isinstance(c.source, str) else ''.join(c.source)
    if "device=" in src:
        import re
        devices = re.findall(r"device='(\w+)'", src)
        print(f'Cell {i}: devices={devices}')
