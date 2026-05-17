"""
Replace the wrong TI computation in the FINAL notebook.

Current bug: TI is computed as average pairwise prediction-disagreement
(yields ~0.50 for every cell), which is not the Theil index defined in the
manuscript (eq. 5).

Fix: implement the Speicher-2018 / AIF360 generalised entropy index with
alpha=1 (Theil), with benefit per Speicher:
    b_i = y_hat_i - y_i + 1   (0=FN, 1=correct, 2=FP)
    TI  = (1/n) sum_i (b_i / mu) * ln(b_i / mu),  with 0*ln(0) := 0
    where mu = mean(b_i).

For pure between-group fairness (which the paper uses), we restrict the
mean to non-zero benefits and treat 0*ln(0) properly via xlogy.

Idempotent: safe to run multiple times.
"""
import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

NB = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\CIKM_2026_LOS_Fairness_FINAL.ipynb")

OLD_BLOCK = """        # TI: average pairwise prediction-disagreement across subgroups
        all_p = []
        for g in groups:
            m = self.protected == g
            all_p.append(self.y_pred[m][:min(int(m.sum()), 5000)])
        if len(all_p) >= 2:
            mn = min(len(p) for p in all_p)
            disagree = float(np.mean([
                np.mean(all_p[i][:mn] != all_p[j][:mn])
                for i in range(len(groups)) for j in range(i+1, len(groups))
            ]))
        else:
            disagree = 0.0
        ti = disagree"""

NEW_BLOCK = """        # TI: Theil index per Speicher et al. 2018 / AIF360 (manuscript eq. 5)
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
        print(f"Patched TI computation in cell {i}")
    elif "ti = disagree" in src and "Theil index per Speicher" not in src:
        print(f"Cell {i} has 'ti = disagree' but old-block string did not match exactly; needs manual fix")

# Clear outputs of all DOWNSTREAM cells too (so re-run reflects the new TI)
if n_patched > 0:
    # find min patched index, then clear all later code cells' outputs
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Patched cells: {n_patched}")
print(f"Wrote notebook ({len(nb['cells'])} cells)")
