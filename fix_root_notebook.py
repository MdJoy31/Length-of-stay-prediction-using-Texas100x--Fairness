"""Fix all issues in root LOS_Prediction_Detailed.ipynb"""
import json

NB_PATH = 'LOS_Prediction_Detailed.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

fixes_applied = 0

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = cell['source']  # list of strings

    for j, line in enumerate(src):
        # Fix 1: RACE_MAP_VIZ 1-indexed → 0-indexed
        if "RACE_MAP_VIZ = {1:'White', 2:'Black', 3:'Hispanic', 4:'Asian/PI', 5:'Other'}" in line:
            src[j] = line.replace(
                "RACE_MAP_VIZ = {1:'White', 2:'Black', 3:'Hispanic', 4:'Asian/PI', 5:'Other'}",
                "RACE_MAP_VIZ = {0:'Other/Unknown', 1:'White', 2:'Black', 3:'Hispanic', 4:'Asian/PI'}"
            )
            print(f"  Fixed RACE_MAP_VIZ in cell {i}, line {j}")
            fixes_applied += 1

        # Fix 2: RandomForest n_jobs=-1 → n_jobs=4
        if "class_weight='balanced', random_state=42, n_jobs=-1" in line:
            src[j] = line.replace("n_jobs=-1", "n_jobs=4")
            print(f"  Fixed RF n_jobs in cell {i}, line {j}")
            fixes_applied += 1

        # Fix 3: LightGBM device='gpu' → device='cpu', n_jobs=1
        if "device='gpu', random_state=42, verbose=-1" in line:
            src[j] = line.replace(
                "device='gpu', random_state=42, verbose=-1",
                "device='cpu', n_jobs=1, random_state=42, verbose=-1"
            )
            print(f"  Fixed LightGBM GPU→CPU in cell {i}, line {j}")
            fixes_applied += 1

    # Fix 4: Add torch.cuda.empty_cache() before DNN training
    if any("# PyTorch DNN with GPU" in line for line in src):
        # Add gc.collect() and torch.cuda.empty_cache() at the start
        new_lines = [
            "# PyTorch DNN with GPU\n",
            "import gc; gc.collect()\n",
            "if torch.cuda.is_available(): torch.cuda.empty_cache()\n",
        ]
        # Replace the first line that has the comment
        for j, line in enumerate(src):
            if "# PyTorch DNN with GPU" in line:
                src[j:j+1] = new_lines
                print(f"  Added CUDA cache clearing in DNN cell {i}")
                fixes_applied += 1
                break

# Clear all outputs
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nTotal fixes applied: {fixes_applied}")
print("All cell outputs cleared.")
print("Notebook saved.")
