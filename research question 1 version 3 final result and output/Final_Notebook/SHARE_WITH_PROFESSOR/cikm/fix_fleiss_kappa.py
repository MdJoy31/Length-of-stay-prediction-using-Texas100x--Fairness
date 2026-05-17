"""
The single-item Fleiss-kappa values in Table5_CrossHospital.csv are
degenerate (only +1.0 or -1/9). Recompute Fleiss kappa correctly:

  - Per (metric, attribute) cell:  proportion-of-agreement (single-item
    Fleiss is undefined; use percent of folds that agree on the majority).
  - Per metric: kappa across the 4 attribute items × 20 fold raters.
  - Per attribute: kappa across the 7 metric items × 20 fold raters.
  - Overall: kappa across all 28 items × 20 raters.

Save the corrected file and a per-metric/per-attribute summary, then
update the notebook tables that were reading the broken column.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm")
TBL = ROOT / "output" / "tables"

THR = {"DI": 0.80, "SPD": 0.10, "EOPP": 0.10, "EOD": 0.10,
       "TI": 0.10, "PP": 0.10, "CAL": 0.05}
metrics = ["DI","SPD","EOPP","EOD","TI","PP","CAL"]
attrs   = ["RACE","SEX","ETHNICITY","AGE_GROUP"]

pcl = pd.read_csv(TBL / "Table6_CrossSite_PerCluster.csv")

def build_verdict_matrix():
    """Returns (28, 20) binary matrix and the item labels."""
    items = []
    M = []
    for m in metrics:
        for a in attrs:
            sub = pcl[pcl["Attribute"]==a].sort_values("Cluster")
            v = sub[m].astype(float).values
            if m == "DI":
                p = (v >= THR[m]).astype(int)
            else:
                p = (np.abs(v) < THR[m]).astype(int)
            items.append((m, a))
            M.append(p)
    return np.array(M), items

def fleiss_kappa(V):
    """Fleiss kappa for binary verdicts. V: (n_items, n_raters)."""
    n_items, n_raters = V.shape
    n_pass = V.sum(axis=1); n_fail = n_raters - n_pass
    N = np.column_stack([n_fail, n_pass])
    P_i = (np.sum(N**2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = P_i.mean()
    p_j = N.sum(axis=0) / (n_items * n_raters)
    P_e = np.sum(p_j**2)
    if abs(1 - P_e) < 1e-12:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)

def landis_koch(k):
    if k < 0:    return "below chance"
    if k <= 0.20: return "slight"
    if k <= 0.40: return "fair"
    if k <= 0.60: return "moderate"
    if k <= 0.80: return "substantial"
    return "almost perfect"

V, items = build_verdict_matrix()
print(f"Built verdict matrix: {V.shape}")

# Overall
overall = fleiss_kappa(V)
print(f"\nOverall Fleiss kappa (28 items x 20 raters): {overall:+.4f}  [{landis_koch(overall)}]")

# Per-metric
per_metric = {}
print("\nPer-metric kappa:")
for m in metrics:
    rows = [i for i, (mm, aa) in enumerate(items) if mm == m]
    k = fleiss_kappa(V[rows])
    per_metric[m] = k
    print(f"  {m:5s}: kappa={k:+.4f}  [{landis_koch(k)}]")

# Per-attribute
per_attr = {}
print("\nPer-attribute kappa:")
for a in attrs:
    rows = [i for i, (mm, aa) in enumerate(items) if aa == a]
    k = fleiss_kappa(V[rows])
    per_attr[a] = k
    print(f"  {a:11s}: kappa={k:+.4f}  [{landis_koch(k)}]")

# Per (metric, attribute) — use proportion-agreement on majority class as
# the meaningful single-item analogue (Fleiss is degenerate for n_items=1).
prop_agree = {}
print("\nPer-(metric,attribute) proportion-agreement on majority verdict:")
for i, (m, a) in enumerate(items):
    p = V[i].mean()
    pa = max(p, 1-p)        # majority share
    prop_agree[(m,a)] = pa
    print(f"  {m:5s} x {a:11s}: pass-rate={p*100:5.1f}% / proportion-agreement={pa*100:5.1f}%")

# Save corrected aggregates
out_rows = []
for m in metrics:
    for a in attrs:
        sub = pcl[pcl["Attribute"]==a].sort_values("Cluster")
        v = sub[m].astype(float).values
        if m == "DI":
            p = (v >= THR[m]).astype(int)
        else:
            p = (np.abs(v) < THR[m]).astype(int)
        out_rows.append({
            "Metric": m, "Attribute": a,
            "Mean": v.mean(), "SD across clusters": v.std(),
            "Range": f"{v.min():.3f}-{v.max():.3f}",
            "Pass_rate_folds": p.mean(),
            "Proportion_agreement_majority": max(p.mean(), 1 - p.mean()),
            "Per_metric_kappa": per_metric[m],
            "Per_attribute_kappa": per_attr[a],
            "Overall_kappa": overall,
        })
fixed = pd.DataFrame(out_rows)
fixed.to_csv(TBL / "Table5_CrossHospital_FIXED_KAPPA.csv", index=False)
print(f"\nWrote fixed file: {TBL / 'Table5_CrossHospital_FIXED_KAPPA.csv'}")

# Save the per-metric/per-attribute kappa summary as a small CSV
pd.DataFrame({
    "Metric": list(per_metric.keys()),
    "Fleiss_kappa": list(per_metric.values()),
    "Landis_Koch": [landis_koch(v) for v in per_metric.values()],
}).to_csv(TBL / "Fleiss_kappa_per_metric.csv", index=False)
pd.DataFrame({
    "Attribute": list(per_attr.keys()),
    "Fleiss_kappa": list(per_attr.values()),
    "Landis_Koch": [landis_koch(v) for v in per_attr.values()],
}).to_csv(TBL / "Fleiss_kappa_per_attribute.csv", index=False)
print(f"Wrote per-metric and per-attribute kappa CSVs.")
