import json, re
nb=json.load(open('LOS_Prediction_Detailed.ipynb','r',encoding='utf-8'))

for i, c in enumerate(nb['cells']):
    if c['cell_type']!='code': continue
    src = c.get('source','')
    if isinstance(src, list): src=''.join(src)
    # device settings
    for m in re.finditer(r"device\s*=\s*['\"](\w+)['\"]", src):
        print(f'Cell {i}: device={m.group(1)}')
    # n_jobs
    for m in re.finditer(r'n_jobs\s*=\s*(-?\d+)', src):
        print(f'Cell {i}: n_jobs={m.group(1)}')
    # File paths
    for m in re.finditer(r"read_csv\(['\"](.+?)['\"]", src):
        print(f'Cell {i}: read_csv={m.group(1)}')
    for m in re.finditer(r"savefig\(['\"](.+?)['\"]", src):
        print(f'Cell {i}: savefig={m.group(1)}')
    for m in re.finditer(r"open\(['\"](.+?)['\"]", src):
        print(f'Cell {i}: open()={m.group(1)}')
