"""Insert the Fair-pipeline cross-site cell into the CIKM notebook
immediately after Cell 15 (id = f0719ff6).

Not committed to git; exists so we can script the edit rather than
ask the Read tool to load a 2 MB notebook.
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "CIKM_2026_LOS_Fairness_13042026.ipynb"
BACKUP = ROOT / "CIKM_2026_LOS_Fairness.pre-fair-cs.ipynb"
ANCHOR_ID = "f0719ff6"

NEW_SRC = '''# ──────────────────────────────────────────────────────────────
# Cell 15b · Fair-Pipeline Cross-Site Portability (K=20 GroupKFold)
# ──────────────────────────────────────────────────────────────
# Section 9 above ran the UN-intervened LightGBM across 20 hospital
# clusters (Tables 6 / 6b / 6c).  This cell reruns using the FULL Fair
# pipeline from Section 6, per fold:
#   Stage 1: Intersectional λ-reweigh  (RACE × AGE × SEX, λ = 1)
#   Stage 2: LightGBM classifier trained on the reweighed 19 clusters
#   Stage 3: Per-group selection-rate threshold optimisation (α_SR = 0.8)
# Tables 6e / 6f / 6g report the Fair model's cross-site fairness and
# the Fair-vs-Standard delta at the cluster mean — i.e., whether the
# paper's pooled-test-set "all 4 DI ≥ 0.80" finding survives site-level
# validation.

FAIR_LAMBDA = 1.0
FAIR_A_SR   = 0.8

def _fair_multi_weights(race_tr, age_tr, sex_tr, y_tr, lam):
    key = np.array([f"{r}|{a}|{s}" for r, a, s in zip(race_tr, age_tr, sex_tr)])
    uniq = sorted(set(key))
    n = len(y_tr)
    sw = np.ones(n, dtype=float)
    total0 = int((y_tr == 0).sum())
    total1 = int((y_tr == 1).sum())
    for g in uniq:
        mg = key == g
        ng = int(mg.sum())
        for lab, total_lab in ((0, total0), (1, total1)):
            mgl = mg & (y_tr == lab)
            ngl = int(mgl.sum())
            if ngl > 0:
                expected = (ng / n) * (total_lab / n)
                observed = ngl / n
                raw_w = expected / observed if observed > 0 else 1.0
                sw[mgl] = np.clip(1.0 + lam * (raw_w - 1.0), 0.1, 10.0)
    return sw

def _find_sr_threshold(probs, target, lo=0.01, hi=0.99, step=0.01):
    best_t = 0.5
    best_d = abs(float((probs >= 0.5).mean()) - target)
    for t in np.arange(lo, hi, step):
        d = abs(float((probs >= t).mean()) - target)
        if d < best_d:
            best_d, best_t = d, float(t)
    return best_t

print(f"Fair Cross-Site Portability: K={K_CS} GroupKFold  (λ={FAIR_LAMBDA}, α_SR={FAIR_A_SR}) …")
fair_cs_results = []
_t0 = time.time()
for fold, (tr_idx, val_idx) in enumerate(gkf_cs.split(X_all, y_all, groups=hosp_all)):
    X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
    race_tr = prot_all[\'RACE\'][tr_idx]
    age_tr  = prot_all[\'AGE_GROUP\'][tr_idx]
    sex_tr  = prot_all[\'SEX\'][tr_idx]
    sw = _fair_multi_weights(race_tr, age_tr, sex_tr, y_tr, FAIR_LAMBDA)

    mdl = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=63, max_depth=8,
        random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
    )
    mdl.fit(X_tr, y_tr, sample_weight=sw)

    X_val, y_val = X_all[val_idx], y_all[val_idx]
    race_val = prot_all[\'RACE\'][val_idx]
    age_val  = prot_all[\'AGE_GROUP\'][val_idx]
    sex_val  = prot_all[\'SEX\'][val_idx]
    y_prob = mdl.predict_proba(X_val)[:, 1]

    # Per-group α_SR threshold shift toward selection-rate parity.
    vkeys = np.array([f"{r}|{a}|{s}" for r, a, s in zip(race_val, age_val, sex_val)])
    overall_sr = float((y_prob >= 0.5).mean())
    y_pred = (y_prob >= 0.5).astype(int)
    for g in set(vkeys):
        mask = vkeys == g
        if int(mask.sum()) < 5:
            continue
        t_sr = _find_sr_threshold(y_prob[mask], overall_sr)
        t = float(np.clip(0.5 + FAIR_A_SR * (t_sr - 0.5), 0.01, 0.99))
        y_pred[mask] = (y_prob[mask] >= t).astype(int)

    row = {
        \'Fold\': fold + 1,
        \'N_val\': int(len(val_idx)),
        \'N_hospitals\': len(set(hosp_all[val_idx])),
        \'Acc\': accuracy_score(y_val, y_pred),
        \'AUC\': roc_auc_score(y_val, y_prob) if len(set(y_val)) > 1 else np.nan,
    }
    for attr in [\'RACE\',\'SEX\',\'ETHNICITY\',\'AGE_GROUP\']:
        attr_val = prot_all[attr][val_idx]
        if len(set(attr_val)) >= 2:
            fc = FairnessCalculator(y_val, y_pred, y_prob, attr_val)
            mc, vc, _ = fc.compute_all()
            for mk in METRIC_KEYS:
                row[f\'{mk}_{attr}\']  = mc[mk]
                row[f\'V_{mk}_{attr}\'] = 1 if vc[mk] else 0
        else:
            for mk in METRIC_KEYS:
                row[f\'{mk}_{attr}\'] = np.nan
                row[f\'V_{mk}_{attr}\'] = np.nan
    fair_cs_results.append(row)
    if (fold + 1) % 5 == 0:
        print(f"  Fold {fold+1}/{K_CS}: Acc={row[\'Acc\']:.4f}  "
              f"DI: R={row[\'DI_RACE\']:.3f} S={row[\'DI_SEX\']:.3f} "
              f"E={row[\'DI_ETHNICITY\']:.3f} A={row[\'DI_AGE_GROUP\']:.3f}")

fair_cs_df = pd.DataFrame(fair_cs_results)
fair_cs_df.to_csv(f\'{TABLES_DIR}/cikm_cross_site_portability_FAIR.csv\', index=False)
print(f"Completed in {time.time()-_t0:.1f}s")

# ---- Table 6e: Fair-model per-cluster (80 rows; ETHNICITY view shown)
fair_pc_rows = []
for _, _r in fair_cs_df.iterrows():
    for _attr in [\'RACE\',\'SEX\',\'ETHNICITY\',\'AGE_GROUP\']:
        _e = {\'Cluster\': int(_r[\'Fold\']), \'N_hosp\': int(_r[\'N_hospitals\']),
              \'Acc\': round(_r[\'Acc\'], 4), \'AUC\': round(_r[\'AUC\'], 4),
              \'Attribute\': _attr}
        _nf = 0
        for _m in METRIC_KEYS:
            _v = _r.get(f\'{_m}_{_attr}\', np.nan)
            _e[_m] = round(_v, 4) if pd.notna(_v) else np.nan
            _nf += int(_is_fair(_m, _v))
        _e[\'N_fair\'] = f\'{_nf}/7\'
        fair_pc_rows.append(_e)
t6fair_pc = pd.DataFrame(fair_pc_rows)
t6fair_pc.to_csv(f\'{TABLES_DIR}/Table6Fair_CrossSite_PerCluster.csv\', index=False)
display(HTML("<h4>Table 6e: <b>Fair model</b> cross-site per cluster "
             "(Reweigh λ=1 + α<sub>SR</sub>=0.8) — ETHNICITY view</h4>"))
_fmt_fair = {c: \'{:.3f}\' for c in [\'Acc\',\'AUC\'] + METRIC_KEYS}
display(t6fair_pc[t6fair_pc[\'Attribute\']==\'ETHNICITY\']
        .drop(columns=[\'Attribute\']).reset_index(drop=True).style.format(_fmt_fair))

# ---- Table 6f: Fair-model per-metric mean + verdict
t6fair_b_rows = []
for _attr in [\'RACE\',\'SEX\',\'ETHNICITY\',\'AGE_GROUP\']:
    for _m in METRIC_KEYS:
        _vals = fair_cs_df[f\'{_m}_{_attr}\'].dropna()
        if len(_vals) == 0:
            continue
        _mean = float(_vals.mean())
        _k = int(sum(_is_fair(_m, v) for v in _vals))
        t6fair_b_rows.append({
            \'Attribute\': _attr, \'Metric\': _m,
            \'Mean\': round(_mean, 4),
            \'SD\': round(float(_vals.std(ddof=1)), 4),
            \'Threshold\': _THR[_m],
            \'Fair_at_mean\': \'Pass\' if _is_fair(_m, _mean) else \'Fail\',
            \'Pass_k_over_N\': f\'{_k}/{len(_vals)}\',
            \'Pass_pct\': round(100 * _k / len(_vals), 1),
        })
t6fair_b = pd.DataFrame(t6fair_b_rows)
t6fair_b.to_csv(f\'{TABLES_DIR}/Table6Fair_CrossSite_MetricVerdicts.csv\', index=False)
display(HTML("<h4>Table 6f: <b>Fair model</b> per-metric cross-site mean &amp; verdict "
             "(thresholds: DI&ge;0.80; SPD/EOPP/EOD/TI/PP&lt;0.10; CAL&lt;0.05)</h4>"))
display(t6fair_b.style.format({\'Mean\':\'{:.3f}\',\'SD\':\'{:.3f}\',\'Threshold\':\'{:.2f}\',\'Pass_pct\':\'{:.1f}\'}))

# ---- Table 6g: Fair vs Standard attribute totals at cross-site mean
t6fair_c_rows = []
for _attr in [\'RACE\',\'SEX\',\'ETHNICITY\',\'AGE_GROUP\']:
    _std = t6b_metric[t6b_metric[\'Attribute\']==_attr]
    _frf = t6fair_b[t6fair_b[\'Attribute\']==_attr]
    _std_nf = int((_std[\'Fair_at_mean\']==\'Pass\').sum())
    _frf_nf = int((_frf[\'Fair_at_mean\']==\'Pass\').sum())
    _std_di = float(_std[_std[\'Metric\']==\'DI\'][\'Mean\'].iloc[0])
    _frf_di = float(_frf[_frf[\'Metric\']==\'DI\'][\'Mean\'].iloc[0])
    t6fair_c_rows.append({
        \'Attribute\': _attr,
        \'Standard_N_fair_at_mean\': f\'{_std_nf}/7\',
        \'Fair_N_fair_at_mean\':     f\'{_frf_nf}/7\',
        \'Delta\':                   _frf_nf - _std_nf,
        \'Standard_DI_mean\': round(_std_di, 3),
        \'Fair_DI_mean\':     round(_frf_di, 3),
        \'DI_improvement\':   round(_frf_di - _std_di, 3),
    })
t6fair_c = pd.DataFrame(t6fair_c_rows)
t6fair_c.to_csv(f\'{TABLES_DIR}/Table6Fair_StdVsFair_Totals.csv\', index=False)
display(HTML("<h4>Table 6g: <b>Fair vs Standard</b> — attribute totals at cross-site mean "
             "(20 hospital clusters)</h4>"))
display(t6fair_c)
'''


def main() -> None:
    shutil.copy2(NB, BACKUP)
    nb = json.loads(NB.read_text(encoding="utf-8"))
    target_index = None
    for i, cell in enumerate(nb["cells"]):
        if cell.get("id") == ANCHOR_ID:
            target_index = i
            break
    if target_index is None:
        raise RuntimeError(f"Anchor cell {ANCHOR_ID} not found")

    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": NEW_SRC.splitlines(keepends=True),
    }
    nb["cells"].insert(target_index + 1, new_cell)

    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Inserted Fair cross-site cell at position {target_index+1}  "
          f"(new cell id = {new_cell['id']})")
    print(f"Backup saved: {BACKUP}")


if __name__ == "__main__":
    main()
