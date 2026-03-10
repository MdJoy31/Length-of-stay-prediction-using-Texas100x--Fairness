import json
nb = json.load(open('LOS_Prediction_Detailed.ipynb'))
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code': continue
    for j, line in enumerate(cell['source']):
        if '              HistGradientBoostingClassifier)' in line:
            # This is the duplicate line - remove the duplicate. Keep just the closing paren
            # The previous line already imports it
            cell['source'][j] = line.replace('HistGradientBoostingClassifier)', ')')
            # Actually, look at what's on line j-1 - it has "RandomForestClassifier, HistGradientBoostingClassifier,\n"
            # So we want: line j-1: "...RandomForestClassifier, HistGradientBoostingClassifier)\n"
            # And remove line j entirely
            prev = cell['source'][j-1]
            if prev.rstrip().endswith(','):
                cell['source'][j-1] = prev.rstrip().rstrip(',') + ')\n'
                cell['source'].pop(j)
                print(f'Fixed duplicate import in cell {i}')
            break
json.dump(nb, open('LOS_Prediction_Detailed.ipynb','w'), indent=1, ensure_ascii=False)
print('Done')
