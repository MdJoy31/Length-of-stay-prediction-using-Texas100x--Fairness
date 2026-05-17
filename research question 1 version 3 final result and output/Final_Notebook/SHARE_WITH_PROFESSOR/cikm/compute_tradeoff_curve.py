"""
Trade-off curve: for canonical XGBoost, sweep DI_target across {0.80, 0.82,
0.85, 0.88, 0.90} and measure:
  - Final DI Race, DI Age (point estimate)
  - Final accuracy
  - VFR Race, VFR Age (under K=500 bootstrap, N=10000)

Output: T_tradeoff_curve.csv
"""
import pandas as pd, numpy as np, sys, io, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

DATA = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\data\texas_100x.csv")
TAB  = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\output_final\tables")

RANDOM_STATE = 42
K_VFR = 500
N_VFR = 10_000

t0 = time.time()
def log(msg): print(f"[{time.time()-t0:>5.0f}s] {msg}", flush=True)

# Feature engineering (canonical)
log("load + features")
df = pd.read_csv(DATA)
df['LOS_BINARY'] = (df['LENGTH_OF_STAY'] > 3).astype(int)
def age_grp(a):
    if a <= 4: return 0
    if a <= 9: return 1
    if a <= 14: return 2
    return 3
df['AGE_GROUP'] = df['PAT_AGE'].apply(age_grp)
y_full = df['LOS_BINARY'].values.astype('int32')
idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.20, random_state=42, stratify=y_full)
train_mask = np.zeros(len(df), dtype=bool); train_mask[idx_tr] = True
m_smooth = 10.0
y_global_mean = float(df.loc[train_mask, 'LOS_BINARY'].mean())
for col in ["ADMITTING_DIAGNOSIS","PRINC_SURG_PROC_CODE","THCIC_ID"]:
    cs = df.loc[train_mask].groupby(col)['LOS_BINARY'].agg(['count','mean']).rename(columns={'count':'n','mean':'yk'})
    cs['mu_k'] = (cs['n']*cs['yk'] + m_smooth*y_global_mean)/(cs['n']+m_smooth)
    df[f'{col}_te'] = df[col].map(cs['mu_k']).fillna(y_global_mean).astype('float32')
KEEP = ["PAT_AGE","TOTAL_CHARGES","PAT_STATUS","TYPE_OF_ADMISSION","SOURCE_OF_ADMISSION"]
TE_COLS = ["ADMITTING_DIAGNOSIS_te","PRINC_SURG_PROC_CODE_te","THCIC_ID_te"]
df['AGE_X_DIAG_TE'] = (df['PAT_AGE'].astype('float32')*df['ADMITTING_DIAGNOSIS_te']).astype('float32')
df['ADMIT_X_SOURCE'] = (df['TYPE_OF_ADMISSION'].astype('float32')*10.0 + df['SOURCE_OF_ADMISSION'].astype('float32')).astype('float32')
hosp_vol = df.groupby('THCIC_ID').size()
df['HOSP_VOLUME_LOG'] = np.log1p(df['THCIC_ID'].map(hosp_vol).fillna(0)).astype('float32')
feature_cols = KEEP + TE_COLS + ["AGE_X_DIAG_TE","ADMIT_X_SOURCE","HOSP_VOLUME_LOG"]
X_full = df[feature_cols].fillna(0).astype('float32').values
y_tr, y_te = y_full[idx_tr], y_full[idx_te]
X_tr, X_te = X_full[idx_tr], X_full[idx_te]
prot_te = {
    'Race': df['RACE'].values[idx_te],
    'Sex':  df['SEX_CODE'].values[idx_te],
    'Eth':  df['ETHNICITY'].values[idx_te],
    'Age':  df['AGE_GROUP'].values[idx_te],
}
attr_keys = ['Race', 'Sex', 'Eth', 'Age']
log(f"  Test {len(X_te):,}")

# Train canonical XGBoost once
log("training canonical XGBoost (n_est=1500)")
mdl = xgb.XGBClassifier(
    n_estimators=1500, max_depth=10, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.85, min_child_weight=3, reg_lambda=1.0,
    tree_method='hist', random_state=42, seed=42,
    eval_metric='logloss', verbosity=0, n_jobs=1,
)
mdl.fit(X_tr, y_tr)
proba = mdl.predict_proba(X_te)[:, 1].astype('float32')
log(f"  trained · acc_std={accuracy_score(y_te, (proba>=0.5).astype(int)):.4f}")

# Helpers
def di(yp, prot):
    groups = np.unique(prot)
    rates = [yp[prot==g].mean() for g in groups if (prot==g).sum() > 0]
    if not rates or max(rates) <= 1e-9: return 1.0
    return float(min(rates) / max(rates))

def predict_with_thresh(proba, prot_dict, thresholds):
    n = len(proba)
    eff = np.full(n, 0.5, dtype='float32')
    for a in attr_keys:
        arr = prot_dict[a]
        for g, tau in thresholds[a].items():
            eff[arr == g] = np.minimum(eff[arr == g], tau)
    return (proba >= eff).astype(int)

def alpha_search(proba, prot, target_di):
    groups = np.unique(prot)
    thr = {int(g): 0.5 for g in groups}
    rates = {int(g): float((proba[prot==g] >= 0.5).mean()) for g in groups}
    max_rate = max(rates.values())
    target_min_sr = target_di * max_rate
    for g in groups:
        g = int(g)
        if rates[g] >= target_min_sr: continue
        for tau in np.linspace(0.05, 0.95, 91):
            sr_g = float((proba[prot==g] >= tau).mean())
            if sr_g >= target_min_sr:
                thr[g] = tau
        # pick the largest tau where sr still meets target
    return thr

def phase5b(proba, prot_dict, di_target, eps=0.002, max_iter=500):
    thresholds = {a: {int(g): 0.5 for g in np.unique(prot_dict[a])} for a in attr_keys}
    for a in ['Age', 'Race']:
        thresholds[a] = alpha_search(proba, prot_dict[a], target_di=di_target)
    yhat = predict_with_thresh(proba, prot_dict, thresholds)
    for step in range(max_iter):
        di_pa = {a: di(yhat, prot_dict[a]) for a in attr_keys}
        if all(d >= di_target for d in di_pa.values()): break
        below = {a: d for a, d in di_pa.items() if d < di_target}
        if not below: break
        worst = min(below, key=below.get)
        groups = np.unique(prot_dict[worst])
        rates = {int(g): float(yhat[prot_dict[worst]==g].mean()) for g in groups}
        min_g = min(rates, key=rates.get)
        old_t = thresholds[worst][min_g]
        new_t = max(0.02, old_t - eps)
        if new_t >= old_t: break
        thresholds[worst][min_g] = new_t
        yhat = predict_with_thresh(proba, prot_dict, thresholds)
    return yhat, thresholds

def bootstrap_vfr(yhat, prot_arr, y_te, threshold=0.80, K=K_VFR, N=N_VFR):
    rng = np.random.default_rng(42)
    y_te_pos = np.where(y_te == 1)[0]
    y_te_neg = np.where(y_te == 0)[0]
    pos_rate = y_te.mean()
    n_pos = int(N * pos_rate); n_neg = N - n_pos
    n_pass = 0
    for k in range(K):
        ix = np.concatenate([rng.choice(y_te_pos, n_pos, replace=True),
                             rng.choice(y_te_neg, n_neg, replace=True)])
        d = di(yhat[ix], prot_arr[ix])
        if d >= threshold: n_pass += 1
    return min(n_pass, K - n_pass) / K

# Sweep DI targets
DI_TARGETS = [0.80, 0.82, 0.85, 0.88, 0.90]
rows = []
acc_std = float(accuracy_score(y_te, (proba >= 0.5).astype(int)))
for di_t in DI_TARGETS:
    log(f"Phase 5b · target_di = {di_t}")
    yhat, thr = phase5b(proba, prot_te, di_t)
    di_post = {a: di(yhat, prot_te[a]) for a in attr_keys}
    acc_post = float(accuracy_score(y_te, yhat))
    all_pass = all(d >= 0.80 for d in di_post.values())
    log(f"  DI: Race={di_post['Race']:.3f} Sex={di_post['Sex']:.3f} Eth={di_post['Eth']:.3f} Age={di_post['Age']:.3f}  acc={acc_post:.4f}  cost={(acc_std-acc_post)*100:.2f} pp")
    # Bootstrap VFR for Race-DI and Age-DI
    log(f"  bootstrap VFR (K=500, N=10000) ...")
    vfr_race = bootstrap_vfr(yhat, prot_te['Race'], y_te)
    vfr_age  = bootstrap_vfr(yhat, prot_te['Age'],  y_te)
    log(f"  VFR Race={vfr_race:.3f}  VFR Age={vfr_age:.3f}")
    rows.append({
        'DI_target': di_t,
        'DI_Race_post': round(di_post['Race'], 4),
        'DI_Sex_post':  round(di_post['Sex'], 4),
        'DI_Eth_post':  round(di_post['Eth'], 4),
        'DI_Age_post':  round(di_post['Age'], 4),
        'Acc_after':    round(acc_post, 4),
        'Acc_cost_pp':  round((acc_std - acc_post) * 100, 2),
        'VFR_Race':     round(vfr_race, 4),
        'VFR_Age':      round(vfr_age, 4),
        'All4_pass':    bool(all_pass),
    })

T = pd.DataFrame(rows)
T.to_csv(TAB / 'T_tradeoff_curve.csv', index=False)
log(f"saved T_tradeoff_curve.csv ({len(T)} rows)")
print()
print(T.to_string(index=False))
log("DONE")
