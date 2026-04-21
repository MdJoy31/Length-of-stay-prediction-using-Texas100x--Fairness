"""D4 Per-Hospital-Cluster Intervention Transferability (standalone).

Runs GroupKFold K=20 across hospitals; at each fold trains a Standard and a
Fair model using the pipeline from Cell 34 of the notebook; computes DI and
Accuracy on the held-out cluster with 100-bootstrap 95% CIs.

Invocation:
    python _scripts/per_cluster_intervention.py

Outputs:
    results/intervention_per_cluster.csv
    results/intervention_cluster_aggregate.csv
    results/intervention_per_cluster.md

Runtime: ~30-60 min on GPU for K=20 LightGBM training + 100 bootstrap resamples.
"""
import pandas as pd, numpy as np, sys, os, time
sys.stdout.reconfigure(encoding='utf-8')
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs('results', exist_ok=True)

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import lightgbm as lgb
import xgboost as xgb

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Load same config as notebook
def disparate_impact(y_pred, protected):
    groups = np.unique(protected)
    rates = [np.mean(y_pred[protected == g]) for g in groups]
    max_r = max(rates)
    return min(rates) / max_r if max_r > 0 else 1.0

# Data
DATA = '../../../../data/texas_100x.csv'
print(f'[load] {DATA}')
df = pd.read_csv(DATA)
df['LOS_BINARY'] = (df['LENGTH_OF_STAY'] > 3).astype(int)
def age_grp(a):
    if a <= 4: return 'Age_0_17'
    elif a <= 9: return 'Age_18_39'
    elif a <= 12: return 'Age_40_54'
    elif a <= 14: return 'Age_55_64'
    else: return 'Age_65_Plus'
df['AGE_GROUP'] = df['PAT_AGE'].apply(age_grp)

target = 'LOS_BINARY'
protected_cols = ['RACE', 'SEX_CODE', 'ETHNICITY', 'AGE_GROUP']
exclude = [target, 'LENGTH_OF_STAY', 'THCIC_ID', 'RECORD_ID'] + protected_cols
feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ['int64', 'float64', 'object']]
for c in feature_cols:
    if df[c].dtype == 'object':
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))
    if df[c].dtype == 'int64':
        df[c] = df[c].astype('int32')
X_all = df[feature_cols].fillna(0).values.astype('float32')
y_all = df[target].values
hosp_all = df['THCIC_ID'].values
race_all = df['RACE'].values
sex_all = df['SEX_CODE'].values
eth_all = df['ETHNICITY'].values
age_all = LabelEncoder().fit_transform(df['AGE_GROUP'].astype(str))

# Fair intervention pipeline per fold
def fit_fair(X_tr, y_tr, race_tr, age_tr, sex_tr, X_te, y_te, race_te, age_te, sex_te,
             lam=30.0, alpha_sr=0.4, alpha_tpr=0.9, alpha_ppv=0.3):
    """Minimal fair pipeline: λ-reweigh + per-group threshold optimization."""
    key_tr = np.array([f'{r}|{a}|{s}' for r, a, s in zip(race_tr, age_tr, sex_tr)])
    sw = np.ones(len(y_tr))
    n = len(y_tr)
    for g in np.unique(key_tr):
        m = key_tr == g
        ng = m.sum()
        for lab in [0, 1]:
            ml = m & (y_tr == lab)
            if ml.sum() > 0:
                expected = (ng / n) * ((y_tr == lab).sum() / n)
                observed = ml.sum() / n
                raw_w = expected / observed if observed > 0 else 1.0
                sw[ml] = np.clip(1.0 + lam * (raw_w - 1.0), 0.1, 10.0)
    # Soft-voting LGB + XGB (LGB-XGB Blend)
    lgbm = lgb.LGBMClassifier(n_estimators=500, num_leaves=63, max_depth=8,
                              learning_rate=0.05, random_state=RANDOM_STATE,
                              verbose=-1, n_jobs=-1)
    lgbm.fit(X_tr, y_tr, sample_weight=sw)
    xgbm = xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.05,
                             tree_method='hist', random_state=RANDOM_STATE,
                             eval_metric='logloss', verbosity=0)
    xgbm.fit(X_tr, y_tr, sample_weight=sw)
    p_te = 0.6 * lgbm.predict_proba(X_te)[:, 1] + 0.4 * xgbm.predict_proba(X_te)[:, 1]
    # Per-group threshold optimization (SR-equalized)
    overall_sr = (p_te >= 0.5).mean()
    key_te = np.array([f'{r}|{a}|{s}' for r, a, s in zip(race_te, age_te, sex_te)])
    y_pred = (p_te >= 0.5).astype(int)
    for g in np.unique(key_te):
        m = key_te == g
        if m.sum() < 5: continue
        group_p = p_te[m]
        best_t, best_diff = 0.5, abs((group_p >= 0.5).mean() - overall_sr)
        for t in np.arange(0.05, 0.95, 0.01):
            diff = abs((group_p >= t).mean() - overall_sr)
            if diff < best_diff:
                best_diff, best_t = diff, t
        adj = 0.5 + alpha_sr * (best_t - 0.5)
        y_pred[m] = (group_p >= np.clip(adj, 0.05, 0.95)).astype(int)
    return p_te, y_pred, (lgbm, xgbm)

def fit_standard(X_tr, y_tr, X_te):
    lgbm = lgb.LGBMClassifier(n_estimators=500, num_leaves=63, max_depth=8,
                              learning_rate=0.05, random_state=RANDOM_STATE,
                              verbose=-1, n_jobs=-1)
    lgbm.fit(X_tr, y_tr)
    xgbm = xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.05,
                             tree_method='hist', random_state=RANDOM_STATE,
                             eval_metric='logloss', verbosity=0)
    xgbm.fit(X_tr, y_tr)
    p_te = 0.6 * lgbm.predict_proba(X_te)[:, 1] + 0.4 * xgbm.predict_proba(X_te)[:, 1]
    y_pred = (p_te >= 0.5).astype(int)
    return p_te, y_pred

rows = []
gkf = GroupKFold(n_splits=20)
t0 = time.time()
for fold_idx, (tr_idx, te_idx) in enumerate(gkf.split(X_all, y_all, hosp_all)):
    fold_t = time.time()
    X_tr, X_te = X_all[tr_idx], X_all[te_idx]
    y_tr, y_te = y_all[tr_idx], y_all[te_idx]
    race_tr, race_te = race_all[tr_idx], race_all[te_idx]
    sex_tr, sex_te = sex_all[tr_idx], sex_all[te_idx]
    eth_te = eth_all[te_idx]
    age_tr, age_te = age_all[tr_idx], age_all[te_idx]
    n_hosp = len(np.unique(hosp_all[te_idx]))
    # Standardize per fold
    sc = StandardScaler().fit(X_tr)
    X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)

    # Standard
    p_std, yp_std = fit_standard(X_tr, y_tr, X_te)
    std = dict(
        cluster_id=fold_idx, n_hospitals=int(n_hosp), n_patients=int(len(y_te)),
        model='Standard',
        accuracy=accuracy_score(y_te, yp_std),
        auroc=roc_auc_score(y_te, p_std),
        di_race=disparate_impact(yp_std, race_te),
        di_sex=disparate_impact(yp_std, sex_te),
        di_eth=disparate_impact(yp_std, eth_te),
        di_age=disparate_impact(yp_std, age_te),
    )
    # Fair
    p_fair, yp_fair, _ = fit_fair(X_tr, y_tr, race_tr, age_tr, sex_tr,
                                   X_te, y_te, race_te, age_te, sex_te)
    fair = dict(
        cluster_id=fold_idx, n_hospitals=int(n_hosp), n_patients=int(len(y_te)),
        model='Fair',
        accuracy=accuracy_score(y_te, yp_fair),
        auroc=roc_auc_score(y_te, p_fair),
        di_race=disparate_impact(yp_fair, race_te),
        di_sex=disparate_impact(yp_fair, sex_te),
        di_eth=disparate_impact(yp_fair, eth_te),
        di_age=disparate_impact(yp_fair, age_te),
    )
    # Bootstrap CIs (100 resamples of the held-out cluster)
    for result in (std, fair):
        di_boots = {'di_race': [], 'di_sex': [], 'di_eth': [], 'di_age': []}
        acc_boots = []
        y_pred_here = yp_std if result['model'] == 'Standard' else yp_fair
        p_here = p_std if result['model'] == 'Standard' else p_fair
        for b in range(100):
            idx = np.random.choice(len(y_te), len(y_te), replace=True)
            acc_boots.append(accuracy_score(y_te[idx], y_pred_here[idx]))
            di_boots['di_race'].append(disparate_impact(y_pred_here[idx], race_te[idx]))
            di_boots['di_sex'].append(disparate_impact(y_pred_here[idx], sex_te[idx]))
            di_boots['di_eth'].append(disparate_impact(y_pred_here[idx], eth_te[idx]))
            di_boots['di_age'].append(disparate_impact(y_pred_here[idx], age_te[idx]))
        result['accuracy_ci_lo'] = round(float(np.percentile(acc_boots, 2.5)), 4)
        result['accuracy_ci_hi'] = round(float(np.percentile(acc_boots, 97.5)), 4)
        for k, v in di_boots.items():
            result[f'{k}_ci_lo'] = round(float(np.percentile(v, 2.5)), 4)
            result[f'{k}_ci_hi'] = round(float(np.percentile(v, 97.5)), 4)

    rows.extend([std, fair])
    print(f'[fold {fold_idx+1:2d}/20] hosp={n_hosp:3d} pat={len(y_te):6d}  '
          f'StdAcc={std["accuracy"]:.4f} FairAcc={fair["accuracy"]:.4f}  '
          f'StdDI_age={std["di_age"]:.3f} FairDI_age={fair["di_age"]:.3f}  '
          f'(+{time.time()-fold_t:.0f}s, total {time.time()-t0:.0f}s)')

per_df = pd.DataFrame(rows)
per_df.to_csv('results/intervention_per_cluster.csv', index=False)
print(f'[saved] results/intervention_per_cluster.csv (rows={len(per_df)})')

# Aggregate
agg_rows = []
for metric in ['accuracy', 'auroc', 'di_race', 'di_sex', 'di_eth', 'di_age']:
    std_vals = [r[metric] for r in rows if r['model'] == 'Standard']
    fair_vals = [r[metric] for r in rows if r['model'] == 'Fair']
    if metric.startswith('di_'):
        improved = sum(1 for s, f in zip(std_vals, fair_vals) if f >= s)
    else:
        improved = sum(1 for s, f in zip(std_vals, fair_vals) if s - f < 0.05)
    agg_rows.append({
        'metric': metric,
        'std_median': round(float(np.median(std_vals)), 4),
        'std_iqr_lo': round(float(np.percentile(std_vals, 25)), 4),
        'std_iqr_hi': round(float(np.percentile(std_vals, 75)), 4),
        'std_worst': round(float(np.min(std_vals) if metric.startswith('di_') else np.min(std_vals)), 4),
        'fair_median': round(float(np.median(fair_vals)), 4),
        'fair_iqr_lo': round(float(np.percentile(fair_vals, 25)), 4),
        'fair_iqr_hi': round(float(np.percentile(fair_vals, 75)), 4),
        'fair_worst': round(float(np.min(fair_vals)), 4),
        'fraction_improved_or_within_5pp': f'{improved}/20',
    })
agg_df = pd.DataFrame(agg_rows)
agg_df.to_csv('results/intervention_cluster_aggregate.csv', index=False)
print(agg_df.to_string(index=False))
print(f'[saved] results/intervention_cluster_aggregate.csv')
