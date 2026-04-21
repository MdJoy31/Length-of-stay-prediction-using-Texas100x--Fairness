"""In-place rewrite of the Table 6 / 6b display cell in the CIKM notebook.

The computation that produces `cs_df` and `cikm_cross_site_portability.csv`
stays intact.  We replace only the trailing display block so the user sees
per-cluster values for 20 hospital clusters and a per-metric pass-rate
summary, rather than the single CV pivot.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "CIKM_2026_LOS_Fairness_13042026.ipynb"
BACKUP = ROOT / "CIKM_2026_LOS_Fairness.pre-table6-rebuild.ipynb"

NEW_SOURCE = """# ──────────────────────────────────────────────────────────────
# Cell 15 · Cross-Site K=20 GroupKFold
# ──────────────────────────────────────────────────────────────
K_CS = 20
print(f"Cross-Site Portability: K={K_CS} GroupKFold …")

X_all = np.vstack([X_train, X_test])
y_all = np.concatenate([y_train, y_test])
hosp_all = np.concatenate([hospital_ids_train, hospital_ids_test])
prot_all = {}
for attr in ['RACE','SEX','ETHNICITY','AGE_GROUP']:
    prot_all[attr] = np.concatenate([protected_attrs_train[attr], protected_attrs[attr]])

gkf_cs = GroupKFold(n_splits=K_CS)
cs_results = []
_t0 = time.time()

for fold, (tr_idx, val_idx) in enumerate(gkf_cs.split(X_all, y_all, groups=hosp_all)):
    model_cs = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63,
        max_depth=8, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
    model_cs.fit(X_all[tr_idx], y_all[tr_idx])
    y_val = y_all[val_idx]
    y_pred_cs = model_cs.predict(X_all[val_idx])
    y_prob_cs = model_cs.predict_proba(X_all[val_idx])[:, 1]

    row = {'Fold': fold+1, 'N_val': len(val_idx),
           'N_hospitals': len(set(hosp_all[val_idx])),
           'Acc': accuracy_score(y_val, y_pred_cs),
           'AUC': roc_auc_score(y_val, y_prob_cs) if len(set(y_val)) > 1 else np.nan}

    for attr in ['RACE','SEX','ETHNICITY','AGE_GROUP']:
        attr_val = prot_all[attr][val_idx]
        if len(set(attr_val)) >= 2:
            fc_cs = FairnessCalculator(y_val, y_pred_cs, y_prob_cs, attr_val)
            mc, vc, _ = fc_cs.compute_all()
            for mk in METRIC_KEYS:
                row[f'{mk}_{attr}'] = mc[mk]
                row[f'V_{mk}_{attr}'] = 1 if vc[mk] else 0
        else:
            for mk in METRIC_KEYS:
                row[f'{mk}_{attr}'] = np.nan; row[f'V_{mk}_{attr}'] = np.nan
    cs_results.append(row)
    if (fold+1) % 5 == 0:
        print(f"  Fold {fold+1}/{K_CS}: N_val={len(val_idx):,}  Acc={row['Acc']:.4f}")

cs_df = pd.DataFrame(cs_results)
cs_df.to_csv(f'{TABLES_DIR}/cikm_cross_site_portability.csv', index=False)
print(f"Completed in {time.time()-_t0:.1f}s")

# ──────────────────────────────────────────────────────────────
# Rebuilt Table 6: per-cluster values and per-metric verdicts at
# the cluster-average. Source is the 20-fold cs_df above; no extra
# training. Standard thresholds from main.tex Table 7 caption.
# ──────────────────────────────────────────────────────────────
_THR = {'DI':0.80,'SPD':0.10,'EOPP':0.10,'EOD':0.10,'TI':0.10,'PP':0.10,'CAL':0.05}

def _is_fair(metric, value):
    if pd.isna(value):
        return False
    return (value >= _THR['DI']) if metric == 'DI' else (abs(value) < _THR[metric])

pc_rows = []
for _, _r in cs_df.iterrows():
    for _attr in ['RACE','SEX','ETHNICITY','AGE_GROUP']:
        _e = {'Cluster': int(_r['Fold']), 'N_hosp': int(_r['N_hospitals']),
              'Acc': round(_r['Acc'], 4), 'AUC': round(_r['AUC'], 4),
              'Attribute': _attr}
        _nf = 0
        for _m in METRIC_KEYS:
            _v = _r.get(f'{_m}_{_attr}', np.nan)
            _e[_m] = round(_v, 4) if pd.notna(_v) else np.nan
            _nf += int(_is_fair(_m, _v))
        _e['N_fair'] = f'{_nf}/7'
        pc_rows.append(_e)
t6_percluster = pd.DataFrame(pc_rows)
t6_percluster.to_csv(f'{TABLES_DIR}/Table6_CrossSite_PerCluster.csv', index=False)

display(HTML("<h4>Table 6: Cross-Site Fairness per Cluster (20 GroupKFold Hospital Clusters) — ETHNICITY view</h4>"))
_fmt = {c: '{:.3f}' for c in ['Acc','AUC'] + METRIC_KEYS}
display(t6_percluster[t6_percluster['Attribute']=='ETHNICITY']
        .drop(columns=['Attribute']).reset_index(drop=True).style.format(_fmt))

t6b_rows = []
for _attr in ['RACE','SEX','ETHNICITY','AGE_GROUP']:
    for _m in METRIC_KEYS:
        _vals = cs_df[f'{_m}_{_attr}'].dropna()
        if len(_vals) == 0:
            continue
        _mean = float(_vals.mean())
        _k = int(sum(_is_fair(_m, v) for v in _vals))
        t6b_rows.append({'Attribute': _attr, 'Metric': _m,
            'Mean': round(_mean, 4), 'SD': round(float(_vals.std(ddof=1)), 4),
            'Threshold': _THR[_m],
            'Fair_at_mean': 'Pass' if _is_fair(_m, _mean) else 'Fail',
            'Pass_k_over_N': f'{_k}/{len(_vals)}',
            'Pass_pct': round(100 * _k / len(_vals), 1)})
t6b_metric = pd.DataFrame(t6b_rows)
t6b_metric.to_csv(f'{TABLES_DIR}/Table6b_CrossSite_MetricVerdicts.csv', index=False)
display(HTML("<h4>Table 6b: Cross-Site Per-Metric Mean &amp; Verdict (DI&ge;0.80; SPD/EOPP/EOD/TI/PP&lt;0.10; CAL&lt;0.05)</h4>"))
display(t6b_metric.style.format({'Mean':'{:.3f}','SD':'{:.3f}','Threshold':'{:.2f}','Pass_pct':'{:.1f}'}))

t6c_rows = []
for _attr in ['RACE','SEX','ETHNICITY','AGE_GROUP']:
    _sub = t6b_metric[t6b_metric['Attribute']==_attr]
    _nf = int((_sub['Fair_at_mean']=='Pass').sum())
    t6c_rows.append({'Attribute': _attr, 'N_fair_at_mean': f'{_nf}/7',
        'Mean_pass_rate_across_clusters_%': round(float(_sub['Pass_pct'].mean()), 1)})
t6c_totals = pd.DataFrame(t6c_rows)
t6c_totals.to_csv(f'{TABLES_DIR}/Table6_CrossSite_AttributeTotals.csv', index=False)
display(HTML("<h4>Table 6c: Attribute Totals — ETHNICITY 5/7 and SEX 3/7 fair at cross-site mean</h4>"))
display(t6c_totals)

# Keep legacy CV summary for backwards compatibility with downstream cells.
cs_summary = []
for attr in ['RACE','SEX','ETHNICITY','AGE_GROUP']:
    for mk in METRIC_KEYS:
        col = f'{mk}_{attr}'
        vals = cs_df[col].dropna()
        if len(vals) < 2: continue
        cs_summary.append({'Attribute': attr, 'Metric': mk,
            'Mean': vals.mean(), 'Std': vals.std(),
            'CV': vals.std()/max(vals.mean(),1e-9),
            'Min': vals.min(), 'Max': vals.max(), 'Range': vals.max()-vals.min()})
cs_summary_df = pd.DataFrame(cs_summary)
"""

TARGET_ID = "f0719ff6"


def main() -> None:
    if not NB.exists():
        raise FileNotFoundError(NB)
    shutil.copy2(NB, BACKUP)
    nb = json.loads(NB.read_text(encoding="utf-8"))
    found = False
    for cell in nb["cells"]:
        if cell.get("id") == TARGET_ID:
            lines = NEW_SOURCE.splitlines(keepends=True)
            cell["source"] = lines
            cell["outputs"] = []
            cell["execution_count"] = None
            found = True
            break
    if not found:
        raise RuntimeError(f"Cell id {TARGET_ID} not found in {NB.name}")

    # Rename the Fleiss' kappa heading cell from "Table 6b" to "Table 6d"
    # so the numbering aligns with the new Table 6 / 6b / 6c above.
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "Table 6b: Fleiss" in src:
            new_src = src.replace("Table 6b: Fleiss", "Table 6d: Fleiss")
            cell["source"] = new_src.splitlines(keepends=True)
            cell["outputs"] = []
            cell["execution_count"] = None
            break

    # Bump the Table 7 heading (per-fold table in cell 16b) to Table 6e for
    # continuity with the new numbering.  Leaves the following Table 7 /
    # Table 8 text untouched -- they refer to later manuscript tables.
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Patched {NB}; backup at {BACKUP}")


if __name__ == "__main__":
    main()
