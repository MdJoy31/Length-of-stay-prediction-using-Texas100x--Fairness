"""
Patch the live .ipynb to use the BETWEEN-GROUP Theil decomposition.

The previous fix used overall Speicher TI which (correctly) gives the
same value across all attributes because it's a property of the
population. For fairness we want the BETWEEN-GROUP component, which
varies per attribute.
"""
import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

NB = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\CIKM_2026_LOS_Fairness_FINAL.ipynb")

OLD_BLOCK = """        # TI: Theil index per Speicher et al. 2018 / AIF360 (manuscript eq. 5)
        # Benefit b_i = y_hat_i - y_i + 1   (FN=0, correct=1, FP=2)
        b = (self.y_pred.astype(float) - self.y_true.astype(float) + 1.0)
        mu = float(np.mean(b)) if len(b) > 0 else 0.0
        if mu > 0:
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = b / mu
                # 0 * ln(0) := 0  (use np.where to avoid NaN)
                term = np.where(b > 0, ratio * np.log(ratio), 0.0)
            ti = float(np.mean(term))
        else:
            ti = 0.0"""

NEW_BLOCK = """        # TI: Theil index BETWEEN-GROUP component for the protected attribute
        # (Speicher 2018 generalised entropy at alpha=1).
        # Benefit b_i = y_hat_i - y_i + 1  (FN=0, correct=1, FP=2)
        # T_total = T_within + T_between; we report T_between because it is
        # the per-group inequality contribution that varies by attribute.
        b_all = (self.y_pred.astype(float) - self.y_true.astype(float) + 1.0)
        mu_all = float(np.mean(b_all)) if len(b_all) > 0 else 0.0
        if mu_all > 0:
            ti_between = 0.0
            n_total = len(b_all)
            for g in groups:
                m = self.protected == g
                n_g = int(m.sum())
                if n_g == 0:
                    continue
                mu_g = float(np.mean(b_all[m]))
                if mu_g > 0:
                    ratio_g = mu_g / mu_all
                    ti_between += (n_g / n_total) * ratio_g * np.log(ratio_g)
            ti = float(abs(ti_between))
        else:
            ti = 0.0"""

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

n_patched = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c.get("source", []))
    if OLD_BLOCK in src:
        new_src = src.replace(OLD_BLOCK, NEW_BLOCK)
        c["source"] = new_src.splitlines(keepends=True)
        c["outputs"] = []
        c["execution_count"] = None
        n_patched += 1
        print(f"Patched cell {i} (TI -> between-group)")

# Clear all downstream cell outputs so they re-run with new TI
if n_patched > 0:
    found = False
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        if "Theil index BETWEEN-GROUP" in "".join(c.get("source", [])):
            found = True
            continue
        if found:
            c["outputs"] = []
            c["execution_count"] = None

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Patched {n_patched} cell(s); cleared downstream outputs")
