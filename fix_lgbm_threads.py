"""Fix LightGBM threading deadlock: add n_jobs=1 to LGBMClassifier in all notebooks."""
import json, glob, re

notebooks = glob.glob("final_notebooks/*.ipynb")
for nb_path in sorted(notebooks):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    changed = False
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src = cell['source'] if isinstance(cell['source'], list) else [cell['source']]
        new_src = []
        cell_changed = False
        for line in src:
            # Add n_jobs=1 to LGBMClassifier line
            if 'device=\'cpu\'' in line and 'verbose=-1' in line and 'LGBMClassifier' not in line:
                # This is the params line with device='cpu', add n_jobs=1
                new_line = line.replace("device='cpu'", "device='cpu', n_jobs=1")
                new_src.append(new_line)
                if new_line != line:
                    cell_changed = True
                    print(f"  [{nb_path}] Cell {i}: Added n_jobs=1 to LightGBM device line")
            else:
                new_src.append(line)

        if cell_changed:
            cell['source'] = new_src if isinstance(cell['source'], list) else ''.join(new_src)
            changed = True

    if changed:
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"✅ Fixed: {nb_path}")
    else:
        print(f"⚠️ No changes needed or pattern not found: {nb_path}")

print("\nDone!")
