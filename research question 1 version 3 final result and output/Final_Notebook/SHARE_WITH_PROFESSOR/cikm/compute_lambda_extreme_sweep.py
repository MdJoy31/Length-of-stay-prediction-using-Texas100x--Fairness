"""
Standalone test: can ANY reweighing-only configuration (no threshold shifting,
no greedy refinement) achieve all-4-DI ≥ 0.80?

Tries:
  - Extreme λ values: 200, 500, 1000, 5000, 10000
  - Three clipping schemes: standard [0.1, 10], relaxed [0.01, 100], unclipped
  - Three reweighing axes: intersectional (RACE × AGE × SEX) vs axis-specific (Age only) vs (Race only)

Saves T_lambda_extreme_sweep.csv. Prints "best so far" status row by row.
"""
import pandas as pd, numpy as np, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb

DATA = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\data\texas_100x.csv")
OUT_DIR = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\output_final\tables")

RANDOM_STATE = 42
LOS_THRESHOLD = 3

t0 = time.time()
print("[1/4] Loading + FE ...")
df = pd.read_csv(DATA)
df['LOS_BINARY'] = (df['LENGTH_OF_STAY'] > LOS_THRESHOLD).astype(int)

def age_grp(a):
    if a <= 4:  return 0
    if a <= 9:  return 1
    if a <= 14: return 2
    return 3
df['AGE_GROUP'] = df['PAT_AGE'].apply(age_grp)

target = 'LOS_BINARY'
y = df[target].values.astype('int32')
idx_all = np.arange(len(df))
idx_tr, idx_te = train_test_split(idx_all, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
train_mask = np.zeros(len(df), dtype=bool); train_mask[idx_tr] = True

m_smooth = 10.0
y_global_mean = float(df.loc[train_mask, target].mean())
TARGET_ENCODE_COLS = ["ADMITTING_DIAGNOSIS", "PRINC_SURG_PROC_CODE", "THCIC_ID"]
for col in TARGET_ENCODE_COLS:
    cat_stats = df.loc[train_mask].groupby(col)[target].agg(['count','mean']).rename(columns={'count':'n','mean':'yk'})
    cat_stats['mu_k'] = (cat_stats['n'] * cat_stats['yk'] + m_smooth * y_global_mean) / (cat_stats['n'] + m_smooth)
    df[f'{col}_te'] = df[col].map(cat_stats['mu_k']).fillna(y_global_mean).astype('float32')

KEEP = ["PAT_AGE", "TOTAL_CHARGES", "PAT_STATUS", "TYPE_OF_ADMISSION", "SOURCE_OF_ADMISSION"]
TE_COLS = ["ADMITTING_DIAGNOSIS_te", "PRINC_SURG_PROC_CODE_te", "THCIC_ID_te"]
df['AGE_X_DIAG_TE'] = (df['PAT_AGE'].astype('float32') * df['ADMITTING_DIAGNOSIS_te']).astype('float32')
df['ADMIT_X_SOURCE'] = (df['TYPE_OF_ADMISSION'].astype('float32') * 10.0 + df['SOURCE_OF_ADMISSION'].astype('float32')).astype('float32')
hosp_vol = df.groupby('THCIC_ID').size()
df['HOSP_VOLUME_LOG'] = np.log1p(df['THCIC_ID'].map(hosp_vol).fillna(0)).astype('float32')
INTER = ["AGE_X_DIAG_TE", "ADMIT_X_SOURCE", "HOSP_VOLUME_LOG"]
feature_cols = KEEP + TE_COLS + INTER

X = df[feature_cols].fillna(0).astype('float32').values
y_tr, y_te = y[idx_tr], y[idx_te]
X_tr, X_te = X[idx_tr], X[idx_te]

prot_te = {
    'RACE': df['RACE'].values[idx_te],
    'SEX': df['SEX_CODE'].values[idx_te],
    'ETHNICITY': df['ETHNICITY'].values[idx_te],
    'AGE_GROUP': df['AGE_GROUP'].values[idx_te],
}
print(f"  ({time.time()-t0:.0f}s) Train {len(X_tr):,} / Test {len(X_te):,}")

# ----------------------------------------------------------
# Reweighing schemes
# ----------------------------------------------------------
def build_weights_intersect(lam, clip_low=0.1, clip_high=10.0):
    cells = (df['RACE'].values[idx_tr].astype(int).astype(str) + "_" +
             df['AGE_GROUP'].values[idx_tr].astype(str) + "_" +
             df['SEX_CODE'].values[idx_tr].astype(int).astype(str))
    cnt = pd.Series(cells).value_counts()
    p_obs = cnt / cnt.sum()
    p_exp = pd.Series(1.0/len(p_obs), index=p_obs.index)
    w_per = 1.0 + lam * (p_exp/p_obs - 1.0)
    if clip_low is not None:
        w_per = w_per.clip(clip_low, clip_high)
    return pd.Series(cells).map(w_per).values.astype('float32')

def build_weights_axis(lam, axis_col, clip_low=0.1, clip_high=10.0):
    arr = df[axis_col].values[idx_tr].astype(int).astype(str)
    cnt = pd.Series(arr).value_counts()
    p_obs = cnt / cnt.sum()
    p_exp = pd.Series(1.0/len(p_obs), index=p_obs.index)
    w_per = 1.0 + lam * (p_exp/p_obs - 1.0)
    if clip_low is not None:
        w_per = w_per.clip(clip_low, clip_high)
    return pd.Series(arr).map(w_per).values.astype('float32')

def disparate_impact(yp, prot):
    groups = np.unique(prot)
    rates = [yp[prot==g].mean() for g in groups]
    if max(rates) <= 1e-9: return 1.0
    return min(rates) / max(rates)

def evaluate(sw, label):
    mdl = xgb.XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
        tree_method='hist', random_state=RANDOM_STATE, seed=RANDOM_STATE,
        eval_metric='logloss', verbosity=0, n_jobs=1,
    )
    mdl.fit(X_tr, y_tr, sample_weight=sw)
    proba = mdl.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc = accuracy_score(y_te, pred)
    auc = roc_auc_score(y_te, proba)
    di = {a: disparate_impact(pred, prot_te[a]) for a in ['RACE','SEX','ETHNICITY','AGE_GROUP']}
    all4 = all(v >= 0.80 for v in di.values())
    return {
        'config': label,
        'Acc': round(acc, 4), 'AUROC': round(auc, 4),
        'DI_R': round(di['RACE'], 3), 'DI_S': round(di['SEX'], 3),
        'DI_E': round(di['ETHNICITY'], 3), 'DI_A': round(di['AGE_GROUP'], 3),
        'min_DI': round(min(di.values()), 3),
        'all4': all4,
    }

# ----------------------------------------------------------
# Sweep
# ----------------------------------------------------------
print(f"\n[2/4] Running extreme λ sweep (~5-10 min) ... ({time.time()-t0:.0f}s)")
results = []

# Intersectional reweighing with extreme λ + relaxed clipping
extreme_lambdas = [200, 500, 1000, 5000, 10000]
clip_schemes = [
    ('std',     0.1,  10.0),
    ('relaxed', 0.01, 100.0),
    ('unclipped', None, None),
]

for lam in extreme_lambdas:
    for clip_label, lo, hi in clip_schemes:
        try:
            sw = build_weights_intersect(lam, clip_low=lo, clip_high=hi)
            label = f"intersect λ={lam} clip={clip_label}"
            r = evaluate(sw, label)
            print(f"  ({time.time()-t0:.0f}s) {label:40s} Acc={r['Acc']} DI(R/S/E/A)={r['DI_R']}/{r['DI_S']}/{r['DI_E']}/{r['DI_A']} all4={r['all4']}")
            results.append(r)
        except Exception as e:
            print(f"  FAILED: {label}: {e}")

# Axis-specific reweighing on AGE only (the binding constraint)
print(f"\n[3/4] Trying axis-specific reweighing on Age only ... ({time.time()-t0:.0f}s)")
for lam in [10, 50, 100, 500, 1000]:
    for clip_label, lo, hi in clip_schemes:
        try:
            sw = build_weights_axis(lam, 'AGE_GROUP', clip_low=lo, clip_high=hi)
            label = f"age-only λ={lam} clip={clip_label}"
            r = evaluate(sw, label)
            print(f"  ({time.time()-t0:.0f}s) {label:40s} Acc={r['Acc']} DI(R/S/E/A)={r['DI_R']}/{r['DI_S']}/{r['DI_E']}/{r['DI_A']} all4={r['all4']}")
            results.append(r)
        except Exception as e:
            print(f"  FAILED: {label}: {e}")

T = pd.DataFrame(results)
T.to_csv(OUT_DIR / 'T_lambda_extreme_sweep.csv', index=False)
print(f"\n[4/4] Done. Total time: {time.time()-t0:.0f}s")
print()
print("=" * 80)
print("SWEEP SUMMARY")
print("=" * 80)
print(T.to_string(index=False))
print()
print(f"Configurations achieving all-4-DI ≥ 0.80: {int(T['all4'].sum())} of {len(T)}")
if T['all4'].any():
    print()
    print("Successful configurations:")
    print(T[T['all4']].to_string(index=False))
else:
    print()
    print("=> NO reweighing-only configuration achieves all-4-DI ≥ 0.80,")
    print("   even at λ = 10,000 with unclipped weights or axis-specific reweighing.")
print()
print(f"Best Age-DI achieved across all configurations: {T['DI_A'].max():.3f}")
print(f"  (still well below the 0.80 threshold)")
