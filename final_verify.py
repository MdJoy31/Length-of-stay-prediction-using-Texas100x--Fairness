"""Final verification of all 3 notebooks."""
import json, os

base = r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\final_notebooks"

notebooks = [
    "LOS_Prediction_Standard.ipynb",
    "LOS_Prediction_Detailed.ipynb",
    "Fairness_Analysis_Complete.ipynb",
]

print("=" * 80)
print("FINAL NOTEBOOK VERIFICATION")
print("=" * 80)

for nb_name in notebooks:
    path = os.path.join(base, nb_name)
    nb = json.load(open(path, 'r', encoding='utf-8'))
    cells = nb['cells']
    code = [c for c in cells if c['cell_type'] == 'code']
    md = [c for c in cells if c['cell_type'] == 'markdown']
    ex = [c for c in code if c.get('execution_count')]
    err_cells = []
    for i, c in enumerate(code):
        for o in c.get('outputs', []):
            if o.get('output_type') == 'error':
                err_cells.append(i)
                break

    sz = os.path.getsize(path)
    status = "PASS" if len(ex) == len(code) and len(err_cells) == 0 else "FAIL"

    print(f"\n{nb_name}")
    print(f"  Total cells:    {len(cells)}")
    print(f"  Code cells:     {len(code)}")
    print(f"  Markdown cells: {len(md)}")
    print(f"  Executed:       {len(ex)}/{len(code)}")
    print(f"  Errors:         {len(err_cells)}")
    print(f"  File size:      {sz:,} bytes")
    print(f"  Status:         [{status}]")

print("\n" + "=" * 80)
print("All files in final_notebooks/:")
print("=" * 80)
for root, dirs, files in os.walk(base):
    for f in sorted(files):
        if '__pycache__' in root:
            continue
        rel = os.path.relpath(os.path.join(root, f), base)
        sz = os.path.getsize(os.path.join(root, f))
        print(f"  {sz:>12,}  {rel}")
