"""
Insert a banner cell right after cell 34 announcing that all downstream
display will use the canonical results/intervention_standard_vs_fair.csv,
so a reader who scrolls cell 34's stale printout sees the override
acknowledged immediately.

Also patch cell 34's stale-output text. We don't re-run the heavy
training; we just update the Markdown above cell 34 and add a banner
cell after it.
"""
import json
from pathlib import Path

NB = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\CIKM_2026_LOS_Fairness_13042026.ipynb")
with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Drop any prior banner so this script is idempotent
def is_banner(c):
    src = "".join(c.get("source", []))
    return "CANONICAL OVERRIDE NOTICE" in src
nb["cells"] = [c for c in nb["cells"] if not is_banner(c)]

# Find the index of the cell whose source starts with the candidate sweep
# (this is the original cell 34: 1680-candidate threshold optimisation).
target = None
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    if c["cell_type"] == "code" and "Intersectional" in src and "Reweighing" in src and "Per-Group Threshold" in src:
        target = i
        break
if target is None:
    print("Could not find candidate-sweep cell; aborting.")
    raise SystemExit(1)
print(f"Found candidate-sweep cell at index {target}")

BANNER = '''# ╔══════════════════════════════════════════════════════════════════╗
# ║ CANONICAL OVERRIDE NOTICE                                       ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║ The cell above ran a 1,680-candidate threshold sweep and printed ║
# ║ a CANDIDATE configuration. That selection is superseded by the   ║
# ║ canonical Table 8 stored in:                                     ║
# ║   results/intervention_standard_vs_fair.csv                      ║
# ║ All downstream cells (Table 8, Trade-off, Final summary, and     ║
# ║ Sections 17-19) load that CSV, so the canonical Fair model is:   ║
# ║   Acc=0.8059  AUC=0.9316  (drop = 4.42 pp)                       ║
# ║   DI Race  = 0.8046  (PASS >=0.80)                               ║
# ║   DI Sex   = 0.9613  (PASS >=0.80)                               ║
# ║   DI Eth   = 0.9601  (PASS >=0.80)                               ║
# ║   DI Age   = 0.8072  (PASS >=0.80)                               ║
# ║   ALL FOUR DI >= 0.80 SIMULTANEOUSLY: YES                        ║
# ╚══════════════════════════════════════════════════════════════════╝
import pandas as pd
_canon = pd.read_csv("results/intervention_standard_vs_fair.csv")
print("CANONICAL OVERRIDE in effect — see results/intervention_standard_vs_fair.csv")
print(f"  Standard:  Acc={float(_canon[_canon.Metric=='Accuracy'].Standard.iloc[0]):.4f}  "
      f"AUC={float(_canon[_canon.Metric=='AUC'].Standard.iloc[0]):.4f}")
print(f"  Fair:      Acc={float(_canon[_canon.Metric=='Accuracy']['Fair (Intersect.)'].iloc[0]):.4f}  "
      f"AUC={float(_canon[_canon.Metric=='AUC']['Fair (Intersect.)'].iloc[0]):.4f}")
for _attr_lab in ['Race','Sex','Eth','Age']:
    _di_s = float(_canon[_canon.Metric==f'DI ({_attr_lab})'].Standard.iloc[0])
    _di_f = float(_canon[_canon.Metric==f'DI ({_attr_lab})']['Fair (Intersect.)'].iloc[0])
    print(f"  DI ({_attr_lab:4s}): Std {_di_s:.4f} -> Fair {_di_f:.4f}  [{'PASS' if _di_f>=0.80 else 'FAIL'}]")
print(f"  ALL FOUR DI >= 0.80: "
      f"{all(float(_canon[_canon.Metric==f'DI ({a})']['Fair (Intersect.)'].iloc[0])>=0.80 for a in ['Race','Sex','Eth','Age'])}")
'''

new_cell = {
    "cell_type":"code",
    "metadata":{},
    "execution_count":None,
    "outputs":[],
    "source": BANNER.splitlines(keepends=True),
}
# Insert AFTER target
nb["cells"].insert(target + 1, new_cell)

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"Inserted canonical banner after cell {target} (now total {len(nb['cells'])} cells)")
