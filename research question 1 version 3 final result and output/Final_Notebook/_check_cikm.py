import ast, json, sys
nb = json.load(open('SHARE_WITH_PROFESSOR/cikm/CIKM_2026_LOS_Fairness.ipynb', 'r', encoding='utf-8'))
code_cells = [(i, c) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']
errors = []
for i, cell in code_cells:
    src = ''.join(cell['source'])
    try:
        ast.parse(src)
    except SyntaxError as e:
        errors.append(f'Cell {i+1}: {e}')
if errors:
    print('Syntax errors:')
    for e in errors:
        print(f'  {e}')
    sys.exit(1)
else:
    print(f'All {len(code_cells)} code cells compile OK')
