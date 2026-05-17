"""
Loop 17: replace cell 49's embedded F1 PNG output with the new compact
PNG so the notebook display matches the on-disk figure.
"""
import json, sys, io, base64
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm")
NB = ROOT / "CIKM_2026_LOS_Fairness_FINAL.ipynb"
F1_PNG = ROOT / "output_final" / "figures" / "F1_reliability_framework.png"

with open(F1_PNG, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode('ascii')

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find cell 49 outputs and replace the PNG output
cell = nb['cells'][49]
print(f"Cell 49 has {len(cell.get('outputs', []))} outputs")

updated = False
for o_idx, o in enumerate(cell.get('outputs', [])):
    if 'data' in o and 'image/png' in o.get('data', {}):
        old_b64 = o['data']['image/png']
        old_size_kb = len(old_b64) * 3 / 4 / 1024
        new_size_kb = len(img_b64) * 3 / 4 / 1024
        cell['outputs'][o_idx]['data']['image/png'] = img_b64
        # Remove text/plain "<Figure: F1 v3>" if present so the new title shows
        if 'text/plain' in cell['outputs'][o_idx]['data']:
            cell['outputs'][o_idx]['data']['text/plain'] = ["<Figure size 1100x650 with 1 Axes>"]
        print(f"  Output {o_idx}: image/png updated ({old_size_kb:.0f} KB -> {new_size_kb:.0f} KB)")
        updated = True

if updated:
    # Also update the stream output to reflect the new caption
    for o_idx, o in enumerate(cell.get('outputs', [])):
        if o.get('output_type') == 'stream' and 'text' in o:
            text = ''.join(o['text']) if isinstance(o['text'], list) else o['text']
            if 'F1_reliability_framework.png' in text:
                new_text = "Wrote output_final/figures/F1_reliability_framework.png  (compact half-page size, 11x6.5 in)\n"
                cell['outputs'][o_idx]['text'] = [new_text]
                print(f"  Output {o_idx}: stream text updated")

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"\nFinal notebook: {NB.stat().st_size / 1024 / 1024:.2f} MB")
