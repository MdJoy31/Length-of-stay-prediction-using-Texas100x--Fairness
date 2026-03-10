import json
nb = json.load(open('LOS_Prediction_Detailed.ipynb'))
fixes = 0
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code': continue
    for j, line in enumerate(cell['source']):
        if 'n_estimators=' in line:
            prev = ''.join(cell['source'][max(0,j-3):j])
            if 'HistGradientBoosting' in prev:
                cell['source'][j] = line.replace('n_estimators=', 'max_iter=')
                fixed = cell['source'][j].strip()[:70]
                print(f'Fixed cell {i} line {j}: {fixed}')
                fixes += 1
json.dump(nb, open('LOS_Prediction_Detailed.ipynb','w'), indent=1, ensure_ascii=False)
print(f'Fixed {fixes} occurrences')
