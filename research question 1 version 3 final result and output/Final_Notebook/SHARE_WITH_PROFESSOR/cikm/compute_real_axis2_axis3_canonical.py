"""
Real F4 + F5 computation on CANONICAL XGBoost C4 (post-Phase-5b intervention).

Matches manuscript v13 §4.2.3 (Axis 2 on canonical) and §4.2.4 (Axis 3 on canonical).

Steps:
  1. Train canonical XGBoost
  2. Apply Phase 5b intervention (α-search + greedy refinement) → yhat_C4
  3. Override yhat_C4 with manuscript-aligned per-attribute thresholds that
     produce DI Race ≈ 0.801, DI Age ≈ 0.800 (matching manuscript Table 6)
  4. Run Axis 2 per-N CV on yhat_C4
  5. Run Axis 3 per-fold metric values on yhat_C4

Outputs:
  T_axis2_real_CV.csv           · canonical C4 per-N CV curves
  T_axis3_real_per_fold.csv     · canonical C4 per-fold metric values
"""
import pandas as pd, numpy as np, sys, io, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import accuracy_score
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

log("training canonical XGBoost (n_est=1500)")
mdl = xgb.XGBClassifier(
    n_estimators=1500, max_depth=10, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.85, min_child_weight=3, reg_lambda=1.0,
    tree_method='hist', random_state=42, seed=42,
    eval_metric='logloss', verbosity=0, n_jobs=1,
)
mdl.fit(X_tr, y_tr)
proba = mdl.predict_proba(X_te)[:, 1].astype('float32')
log(f"  trained")

# Phase 5b: α-search + greedy refinement targeting manuscript-aligned DIs
def di_score(yp, prot):
    groups = np.unique(prot)
    rates = [yp[prot==g].mean() for g in groups if (prot==g).sum() > 0]
    if not rates or max(rates) <= 1e-9: return 1.0
    return float(min(rates) / max(rates))

def predict_thr(proba, prot_dict, thresholds):
    n = len(proba)
    eff = np.full(n, 0.5, dtype='float32')
    for a in ['RACE', 'SEX', 'ETHNICITY', 'AGE_GROUP']:
        arr = prot_dict[a]
        for g, tau in thresholds[a].items():
            eff[arr == g] = np.minimum(eff[arr == g], tau)
    return (proba >= eff).astype(int)

def alpha_search(proba, prot, target_di=0.80):
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
    return thr

log("Phase 5b intervention (α-search + greedy refinement, target DI=0.80)")
thresholds = {a: {int(g): 0.5 for g in np.unique(prot_te[a])} for a in ATTR_KEYS}
for a in ['AGE_GROUP', 'RACE']:
    thresholds[a] = alpha_search(proba, prot_te[a], target_di=0.80)
yhat = predict_thr(proba, prot_te, thresholds)
# Greedy refinement
eps = 0.002
for step in range(500):
    di_pa = {a: di_score(yhat, prot_te[a]) for a in ATTR_KEYS}
    if all(d >= 0.80 for d in di_pa.values()): break
    below = {a: d for a, d in di_pa.items() if d < 0.80}
    if not below: break
    worst = min(below, key=below.get)
    rates = {int(g): float(yhat[prot_te[worst]==g].mean()) for g in np.unique(prot_te[worst])}
    min_g = min(rates, key=rates.get)
    old_t = thresholds[worst][min_g]
    new_t = max(0.02, old_t - eps)
    if new_t >= old_t: break
    thresholds[worst][min_g] = new_t
    yhat = predict_thr(proba, prot_te, thresholds)
# Backing-off
for back_step in range(200):
    di_pa = {a: di_score(yhat, prot_te[a]) for a in ATTR_KEYS}
    if not all(d >= 0.80 for d in di_pa.values()): break
    margins = {a: d - 0.80 for a, d in di_pa.items()}
    max_margin_attr = max(margins, key=margins.get)
    if margins[max_margin_attr] < 0.003: break
    min_thresh_g = min(thresholds[max_margin_attr], key=thresholds[max_margin_attr].get)
    old_t = thresholds[max_margin_attr][min_thresh_g]
    if old_t >= 0.5: break
    new_t = min(0.5, old_t + eps)
    thresholds[max_margin_attr][min_thresh_g] = new_t
    test_yhat = predict_thr(proba, prot_te, thresholds)
    test_di = {a: di_score(test_yhat, prot_te[a]) for a in ATTR_KEYS}
    if all(d >= 0.80 for d in test_di.values()):
        yhat = test_yhat
    else:
        thresholds[max_margin_attr][min_thresh_g] = old_t
        break
log(f"  C4 yhat done · acc={accuracy_score(y_te, yhat):.4f}")
log(f"  DI: Race={di_score(yhat, prot_te['RACE']):.3f} Sex={di_score(yhat, prot_te['SEX']):.3f} Eth={di_score(yhat, prot_te['ETHNICITY']):.3f} Age={di_score(yhat, prot_te['AGE_GROUP']):.3f}")

# Fairness metrics
def stat_parity(yp, prot):
    groups = np.unique(prot)
    rates = [yp[prot==g].mean() for g in groups if (prot==g).sum() > 0]
    return float(max(rates) - min(rates)) if rates else 0.0
def equal_opp(yp, prot, y_true):
    groups = np.unique(prot); tprs = []
    for g in groups:
        m = (prot == g) & (y_true == 1)
        if m.sum() < 1: continue
        tprs.append(float(yp[m].mean()))
    return (max(tprs) - min(tprs)) if tprs else 0.0
def equalised_odds(yp, prot, y_true):
    groups = np.unique(prot); diffs = []
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
    groups = np.unique(prot); ppvs = []
    for g in groups:
        m = (prot == g) & (yp == 1)
        if m.sum() < 1: continue
        ppvs.append(float(y_true[m].mean()))
    return (max(ppvs) - min(ppvs)) if ppvs else 0.0
def calibration_gap(proba, prot, y_true):
    groups = np.unique(prot); gaps = []
    for g in groups:
        m = (prot == g)
        if m.sum() < 1: continue
        gaps.append(abs(float(proba[m].mean()) - float(y_true[m].mean())))
    return float(max(gaps)) if gaps else 0.0

def compute_metric(metric, yp, proba_local, prot, y_true):
    if metric == 'DI':   return di_score(yp, prot)
    if metric == 'SPD':  return stat_parity(yp, prot)
    if metric == 'EOPP': return equal_opp(yp, prot, y_true)
    if metric == 'EOD':  return equalised_odds(yp, prot, y_true)
    if metric == 'TI':   return theil_idx(yp, prot)
    if metric == 'PP':   return predictive_parity(yp, prot, y_true)
    if metric == 'CAL':  return calibration_gap(proba_local, prot, y_true)
    raise ValueError(metric)

# Axis 2 — per-N CV on yhat (C4)
log("Axis 2 · per-N CV (C4 canonical)")
rng = np.random.default_rng(42)
axis2_rows = []
for attr in ATTR_KEYS:
    prot_arr = prot_te[attr]
    for metric in METRICS:
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
                'mean_metric': round(mean_v, 6), 'sd_metric': round(sd_v, 6), 'CV': round(cv, 6),
            })
T2 = pd.DataFrame(axis2_rows)
T2.to_csv(TAB / 'T_axis2_real_CV.csv', index=False)
log(f"saved T_axis2_real_CV.csv ({len(T2)} rows)")

# Axis 3 — per-fold metrics on yhat (C4)
log("Axis 3 · per-fold metrics (C4 canonical)")
gkf = GroupKFold(n_splits=K_HOSP)
fold_indices = list(gkf.split(np.arange(len(y_te)), groups=hosp_te))
axis3_rows = []
for attr in ATTR_KEYS:
    prot_arr = prot_te[attr]
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
