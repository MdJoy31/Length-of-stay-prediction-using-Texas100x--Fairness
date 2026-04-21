"""D2 lambda sweep (standalone).

Trains the Fair pipeline at each lambda in {0, 0.5, 1, 3, 5, 10, 15, 30, 50, 100}
on the full training split; evaluates Accuracy, AUROC, F1 and DI per attribute.

Outputs: results/intervention_lambda_sweep.csv (+.md)
Runtime: ~20-40 min on GPU (10 x LGB-XGB Blend training).
"""
import pandas as pd, numpy as np, sys, os, time
sys.stdout.reconfigure(encoding='utf-8')
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs('results', exist_ok=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import lightgbm as lgb
import xgboost as xgb

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def disparate_impact(y_pred, protected):
    groups = np.unique(protected)
    rates = [np.mean(y_pred[protected == g]) for g in groups]
    max_r = max(rates)
    return min(rates) / max_r if max_r > 0 else 1.0

df = pd.read_csv('../../../../data/texas_100x.csv')
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
X = df[feature_cols].fillna(0).values.astype('float32')
y = df[target].values

indices = np.arange(len(df))
tr_idx, te_idx = train_test_split(indices, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
X_tr, X_te = X[tr_idx], X[te_idx]
y_tr, y_te = y[tr_idx], y[te_idx]
sc = StandardScaler().fit(X_tr)
X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)

race_tr, race_te = df['RACE'].values[tr_idx], df['RACE'].values[te_idx]
sex_tr, sex_te = df['SEX_CODE'].values[tr_idx], df['SEX_CODE'].values[te_idx]
eth_te = df['ETHNICITY'].values[te_idx]
age_tr = LabelEncoder().fit_transform(df['AGE_GROUP'].astype(str))[tr_idx]
age_te = LabelEncoder().fit(df['AGE_GROUP'].astype(str)).transform(df['AGE_GROUP'].astype(str).values[te_idx])

def weights(lam):
    key = np.array([f'{r}|{a}|{s}' for r, a, s in zip(race_tr, age_tr, sex_tr)])
    sw = np.ones(len(y_tr))
    n = len(y_tr)
    for g in np.unique(key):
        m = key == g
        ng = m.sum()
        for lab in [0, 1]:
            ml = m & (y_tr == lab)
            if ml.sum() > 0:
                expected = (ng / n) * ((y_tr == lab).sum() / n)
                observed = ml.sum() / n
                raw_w = expected / observed if observed > 0 else 1.0
                sw[ml] = np.clip(1.0 + lam * (raw_w - 1.0), 0.1, 10.0)
    return sw

LAM_GRID = [0, 0.5, 1, 3, 5, 10, 15, 30, 50, 100]
rows = []
t0 = time.time()
baseline_acc = None
for lam in LAM_GRID:
    t = time.time()
    sw = weights(lam) if lam > 0 else np.ones(len(y_tr))
    lgbm = lgb.LGBMClassifier(n_estimators=500, num_leaves=63, max_depth=8,
                              learning_rate=0.05, random_state=RANDOM_STATE,
                              verbose=-1, n_jobs=-1)
    lgbm.fit(X_tr, y_tr, sample_weight=sw)
    xgbm = xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.05,
                             tree_method='hist', random_state=RANDOM_STATE,
                             eval_metric='logloss', verbosity=0)
    xgbm.fit(X_tr, y_tr, sample_weight=sw)
    p = 0.6 * lgbm.predict_proba(X_te)[:, 1] + 0.4 * xgbm.predict_proba(X_te)[:, 1]
    yp = (p >= 0.5).astype(int)
    acc = accuracy_score(y_te, yp)
    auc = roc_auc_score(y_te, p)
    f1 = f1_score(y_te, yp)
    di_r, di_s, di_e, di_a = [disparate_impact(yp, g) for g in (race_te, sex_te, eth_te, age_te)]
    all_pass = all(x >= 0.80 for x in (di_r, di_s, di_e, di_a))
    if baseline_acc is None and lam == 0:
        baseline_acc = acc
    rows.append({'lambda': lam, 'accuracy': round(acc, 4), 'auroc': round(auc, 4), 'f1': round(f1, 4),
                 'di_race': round(di_r, 3), 'di_sex': round(di_s, 3),
                 'di_eth': round(di_e, 3), 'di_age': round(di_a, 3),
                 'all_di_pass': all_pass,
                 'accuracy_drop_vs_baseline': round((baseline_acc or acc) - acc, 4) if baseline_acc is not None else 0.0})
    print(f'[lambda={lam:5}] Acc={acc:.4f} AUC={auc:.4f}  DI R/S/E/A={di_r:.3f}/{di_s:.3f}/{di_e:.3f}/{di_a:.3f}  AllPass={all_pass}  (+{time.time()-t:.0f}s)')

df_out = pd.DataFrame(rows)
# Mark selected: smallest lambda with all_di_pass and drop<0.05
selected = None
for _, r in df_out.iterrows():
    if r['all_di_pass'] and r['accuracy_drop_vs_baseline'] < 0.05:
        selected = r['lambda']; break
df_out['selected'] = df_out['lambda'] == selected
df_out.to_csv('results/intervention_lambda_sweep.csv', index=False)
print(df_out.to_string(index=False))
print(f'[selected] lambda = {selected} (smallest value with all_di_pass and acc_drop<5pp)' if selected is not None
      else '[selected] NONE — constraint infeasible across full lambda grid')
