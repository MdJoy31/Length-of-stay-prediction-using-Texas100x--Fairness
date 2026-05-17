"""
Real F4 + F5 computation:
  F4 (Axis 2): per-N coefficient-of-variation curves for 7 metrics × 4 attributes
              N grid: {1k, 2k, 5k, 10k, 25k, 50k, 100k, 185k}
              R=30 random sub-samples per N, compute CV across them
  F5 (Axis 3): per-fold metric values for 7 metrics × 4 attributes
              K_hosp=20 GroupKFold by THCIC hospital identifier

Output:
  T_axis2_real_CV.csv  · per (metric, attribute, N): mean_metric, sd_metric, CV
  T_axis3_real_per_fold.csv  · per (metric, attribute, fold): metric value, verdict
"""
import pandas as pd, numpy as np, sys, io, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.model_selection import train_test_split, GroupKFold
import xgboost as xgb

DATA = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\data\texas_100x.csv")
TAB  = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\output_final\tables")

RANDOM_STATE = 42
N_GRID = [1_000, 2_000, 5_000, 10_000, 25_000, 50_000, 100_000, 185_026]
R_REPS = 30
K_HOSP = 20
ATTR_KEYS = ['RACE', 'SEX', 'ETHNICITY', 'AGE_GROUP']
METRICS = ['DI', 'SPD', 'EOPP', 'EOD', 'TI', 'PP', 'CAL']
METRIC_THRESHOLDS = {
    'DI': (0.80, 'above'), 'SPD': (0.10, 'below'), 'EOPP': (0.10, 'below'),
    'EOD': (0.10, 'below'), 'TI': (0.10, 'below'), 'PP': (0.10, 'below'),
    'CAL': (0.05, 'below'),
}

t0 = time.time()
def log(msg): print(f"[{time.time()-t0:>6.0f}s] {msg}", flush=True)

# Feature engineering
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
    'RACE': df['RACE'].values[idx_te],
    'SEX': df['SEX_CODE'].values[idx_te],
    'ETHNICITY': df['ETHNICITY'].values[idx_te],
    'AGE_GROUP': df['AGE_GROUP'].values[idx_te],
}
hosp_te = df['THCIC_ID'].values[idx_te]
log(f"  Test {len(X_te):,}")

# Train canonical XGBoost
log("training canonical XGBoost")
mdl = xgb.XGBClassifier(
    n_estimators=1500, max_depth=10, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.85, min_child_weight=3, reg_lambda=1.0,
    tree_method='hist', random_state=42, seed=42,
    eval_metric='logloss', verbosity=0, n_jobs=1,
)
mdl.fit(X_tr, y_tr)
proba = mdl.predict_proba(X_te)[:, 1].astype('float32')
yhat = (proba >= 0.5).astype(int)
log(f"  trained · acc={np.mean(yhat==y_te):.4f}")

# Fairness metrics
def disparate_impact(yp, prot):
    groups = np.unique(prot)
    rates = [yp[prot==g].mean() for g in groups if (prot==g).sum() > 0]
    if not rates or max(rates) <= 1e-9: return 1.0
    return float(min(rates) / max(rates))
def stat_parity(yp, prot):
    groups = np.unique(prot)
    rates = [yp[prot==g].mean() for g in groups if (prot==g).sum() > 0]
    return float(max(rates) - min(rates)) if rates else 0.0
def equal_opp(yp, prot, y_true):
    groups = np.unique(prot)
    tprs = []
    for g in groups:
        m = (prot == g) & (y_true == 1)
        if m.sum() < 1: continue
        tprs.append(float(yp[m].mean()))
    return (max(tprs) - min(tprs)) if tprs else 0.0
def equalised_odds(yp, prot, y_true):
    groups = np.unique(prot)
    diffs = []
    for label in [0, 1]:
        rates = []
        for g in groups:
            m = (prot == g) & (y_true == label)
            if m.sum() < 1: continue
            rates.append(float(yp[m].mean()))
        if rates: diffs.append(max(rates) - min(rates))
    return float(max(diffs)) if diffs else 0.0
def theil_idx(yp, prot):
    groups = np.unique(prot)
    rates = np.array([yp[prot==g].mean() for g in groups if (prot==g).sum() > 0])
    if len(rates) == 0: return 0.0
    mu = rates.mean()
    if mu <= 1e-9: return 0.0
    ratios = np.clip(rates / mu, 1e-9, None)
    return float(np.mean(ratios * np.log(ratios)))
def predictive_parity(yp, prot, y_true):
    groups = np.unique(prot)
    ppvs = []
    for g in groups:
        m = (prot == g) & (yp == 1)
        if m.sum() < 1: continue
        ppvs.append(float(y_true[m].mean()))
    return (max(ppvs) - min(ppvs)) if ppvs else 0.0
def calibration_gap(proba, prot, y_true):
    groups = np.unique(prot)
    gaps = []
    for g in groups:
        m = (prot == g)
        if m.sum() < 1: continue
        gaps.append(abs(float(proba[m].mean()) - float(y_true[m].mean())))
    return float(max(gaps)) if gaps else 0.0

def compute_metric(metric, yp, proba_local, prot, y_true):
    if metric == 'DI':   return disparate_impact(yp, prot)
    if metric == 'SPD':  return stat_parity(yp, prot)
    if metric == 'EOPP': return equal_opp(yp, prot, y_true)
    if metric == 'EOD':  return equalised_odds(yp, prot, y_true)
    if metric == 'TI':   return theil_idx(yp, prot)
    if metric == 'PP':   return predictive_parity(yp, prot, y_true)
    if metric == 'CAL':  return calibration_gap(proba_local, prot, y_true)
    raise ValueError(metric)

# ===================================================================
# Axis 2 · per-N CV curves (real bootstrap)
# ===================================================================
log("Axis 2 · per-N CV curves")
rng = np.random.default_rng(42)
axis2_rows = []
for attr in ATTR_KEYS:
    prot_arr = prot_te[attr]
    for metric in METRICS:
        log(f"  ({metric}, {attr}) · 8 N × 30 reps")
        for N in N_GRID:
            vals = []
            for r in range(R_REPS):
                if N >= len(y_te):
                    sub_idx = np.arange(len(y_te))
                else:
                    sub_idx = rng.choice(len(y_te), size=N, replace=False)
                v = compute_metric(metric, yhat[sub_idx], proba[sub_idx], prot_arr[sub_idx], y_te[sub_idx])
                vals.append(v)
            vals = np.array(vals)
            mean_v = float(vals.mean())
            sd_v = float(vals.std(ddof=1))
            cv = sd_v / abs(mean_v) if abs(mean_v) > 1e-9 else 0.0
            axis2_rows.append({
                'metric': metric, 'attribute': attr, 'N': N,
                'mean_metric': round(mean_v, 6),
                'sd_metric':   round(sd_v, 6),
                'CV':          round(cv, 6),
            })

T2 = pd.DataFrame(axis2_rows)
T2.to_csv(TAB / 'T_axis2_real_CV.csv', index=False)
log(f"saved T_axis2_real_CV.csv ({len(T2)} rows)")

# ===================================================================
# Axis 3 · per-fold metric values (real GroupKFold by THCIC hospital)
# ===================================================================
log("Axis 3 · K_hosp=20 GroupKFold per-fold metric values")
gkf = GroupKFold(n_splits=K_HOSP)
fold_indices = list(gkf.split(np.arange(len(y_te)), groups=hosp_te))

axis3_rows = []
for attr in ATTR_KEYS:
    prot_arr = prot_te[attr]
    log(f"  attr={attr}")
    for metric in METRICS:
        for fold_id, (_, fold_idx) in enumerate(fold_indices):
            v = compute_metric(metric, yhat[fold_idx], proba[fold_idx], prot_arr[fold_idx], y_te[fold_idx])
            thr, direction = METRIC_THRESHOLDS[metric]
            verdict = (v >= thr) if direction == 'above' else (v < thr)
            axis3_rows.append({
                'metric': metric, 'attribute': attr, 'fold': fold_id,
                'metric_value': round(v, 6),
                'verdict': 'Pass' if verdict else 'Fail',
            })
T3 = pd.DataFrame(axis3_rows)
T3.to_csv(TAB / 'T_axis3_real_per_fold.csv', index=False)
log(f"saved T_axis3_real_per_fold.csv ({len(T3)} rows)")

log("DONE")
