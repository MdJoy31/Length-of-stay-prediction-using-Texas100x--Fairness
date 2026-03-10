"""Fix cell IDs and execute notebooks."""
import json, uuid, sys, os
os.chdir(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1")

# Step 1: Fix cell IDs
for nb_name in ['final_notebooks/LOS_Prediction_Standard.ipynb', 'final_notebooks/LOS_Prediction_Detailed.ipynb']:
    with open(nb_name, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for c in nb['cells']:
        if 'id' not in c:
            c['id'] = uuid.uuid4().hex[:8]
    nb['nbformat_minor'] = 5
    with open(nb_name, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Fixed IDs: {nb_name} ({len(nb['cells'])} cells)")

print("Done fixing IDs")
