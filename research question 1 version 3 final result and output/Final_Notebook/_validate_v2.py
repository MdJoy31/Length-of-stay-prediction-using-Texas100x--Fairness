import json
nb = json.load(open('SHARE_WITH_PROFESSOR/cikm/CIKM_2026_LOS_Fairness.ipynb'))
cells = nb['cells']
print(f"Valid notebook: {len(cells)} cells")
for i, c in enumerate(cells):
    ct = c['cell_type']
    raw = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
    src = raw[:80].strip().replace('\n', ' ')
    print(f"  Cell {i+1:2d}: {ct:8s} | {src}")
