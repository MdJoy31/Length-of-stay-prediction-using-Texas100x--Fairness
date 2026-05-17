"""
compute_revision_addenda.py — produces two CSVs for the reviewer-revision section:

  T_N_sensitivity.csv         VFR as a function of bootstrap resample size N
                              for three representative cells (DI Race, DI Age,
                              EOPP Race). Justifies the N = 10,000 choice in §4.1.

  T_C3_C4_binding_VFR.csv     Per-cell pre/post VFR comparison for Race-DI and
                              Age-DI between C3 (Threshold-Shift only) and C4
                              (Real+VFR canonical). Tests whether greedy
                              refinement reduces binding-constraint VFR
                              uniformly or only on average.
"""
import pandas as pd, numpy as np, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from pathlib import Path
from sklearn.model_selection import train_test_split
import xgboost as xgb

DATA = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\data\texas_100x.csv")
TAB  = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\output_final\tables")

RANDOM_STATE = 42
LOS_THRESHOLD = 3
K_VFR = 500
N_GRID = [1_000, 2_000, 5_000, 10_000, 25_000, 50_000, 100_000, 185_026]
TARGET_CELLS = [('DI', 'RACE'), ('DI', 'AGE_GROUP'), ('EOPP', 'RACE')]

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:>5.0f}s] {msg}", flush=True)

# ============================================================
# Phase A — load + feature engineering (canonical pipeline)
# ============================================================
log("load + feature engineering")
df = pd.read_csv(DATA)
df['LOS_BINARY'] = (df['LENGTH_OF_STAY'] > LOS_THRESHOLD).astype(int)

def age_grp(a):
    if a <= 4:  return 0
    if a <= 9:  return 1
    if a <= 14: return 2
    return 3
df['AGE_GROUP'] = df['PAT_AGE'].apply(age_grp)

target = 'LOS_BINARY'
y_full = df[target].values.astype('int32')
idx_all = np.arange(len(df))
idx_tr, idx_te = train_test_split(idx_all, test_size=0.20, random_state=RANDOM_STATE, stratify=y_full)
train_mask = np.zeros(len(df), dtype=bool); train_mask[idx_tr] = True

m_smooth = 10.0
y_global_mean = float(df.loc[train_mask, target].mean())
TARGET_ENCODE_COLS = ["ADMITTING_DIAGNOSIS","PRINC_SURG_PROC_CODE","THCIC_ID"]
for col in TARGET_ENCODE_COLS:
    cat_stats = df.loc[train_mask].groupby(col)[target].agg(['count','mean']).rename(columns={'count':'n','mean':'yk'})
    cat_stats['mu_k'] = (cat_stats['n']*cat_stats['yk'] + m_smooth*y_global_mean)/(cat_stats['n']+m_smooth)
    df[f'{col}_te'] = df[col].map(cat_stats['mu_k']).fillna(y_global_mean).astype('float32')
KEEP = ["PAT_AGE","TOTAL_CHARGES","PAT_STATUS","TYPE_OF_ADMISSION","SOURCE_OF_ADMISSION"]
TE_COLS = ["ADMITTING_DIAGNOSIS_te","PRINC_SURG_PROC_CODE_te","THCIC_ID_te"]
df['AGE_X_DIAG_TE']    = (df['PAT_AGE'].astype('float32')*df['ADMITTING_DIAGNOSIS_te']).astype('float32')
df['ADMIT_X_SOURCE']   = (df['TYPE_OF_ADMISSION'].astype('float32')*10.0 + df['SOURCE_OF_ADMISSION'].astype('float32')).astype('float32')
hosp_vol = df.groupby('THCIC_ID').size()
df['HOSP_VOLUME_LOG']  = np.log1p(df['THCIC_ID'].map(hosp_vol).fillna(0)).astype('float32')
INTER = ["AGE_X_DIAG_TE","ADMIT_X_SOURCE","HOSP_VOLUME_LOG"]
feature_cols = KEEP + TE_COLS + INTER
X_full = df[feature_cols].fillna(0).astype('float32').values
y_tr, y_te = y_full[idx_tr], y_full[idx_te]
X_tr, X_te = X_full[idx_tr], X_full[idx_te]
prot_te = {'RACE': df['RACE'].values[idx_te], 'AGE_GROUP': df['AGE_GROUP'].values[idx_te]}
log(f"  Train {len(X_tr):,} / Test {len(X_te):,}")

# ============================================================
# Phase B — train canonical XGBoost (n_est = 1500)
# ============================================================
log("training canonical XGBoost (n_est=1500)")
mdl = xgb.XGBClassifier(
    n_estimators=1500, max_depth=10, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.85, min_child_weight=3, reg_lambda=1.0,
    tree_method='hist', random_state=RANDOM_STATE, seed=RANDOM_STATE,
    eval_metric='logloss', verbosity=0, n_jobs=1,
)
mdl.fit(X_tr, y_tr)
proba = mdl.predict_proba(X_te)[:, 1].astype('float32')
pred  = (proba >= 0.5).astype(int)
log(f"  Acc = {(pred==y_te).mean():.4f}")

# ============================================================
# Phase C — N-sensitivity for VFR (3 cells × 8 N × K=500)
# ============================================================
THRESHOLDS = {'DI':(0.80,'above'), 'SPD':(0.10,'below'), 'EOPP':(0.10,'below'),
              'EOD':(0.10,'below'), 'TI':(0.10,'below'), 'PP':(0.10,'below'),
              'CAL':(0.05,'below')}

def disparate_impact(yp, prot):
    groups = np.unique(prot)
    rates = [yp[prot==g].mean() for g in groups]
    if max(rates) <= 1e-9: return 1.0
    return float(min(rates) / max(rates))

def eopp(yp, prot, y_true):
    groups = np.unique(prot)
    tprs = []
    for g in groups:
        m = (prot == g) & (y_true == 1)
        if m.sum() < 1: continue
        tprs.append(float(yp[m].mean()))
    return (max(tprs) - min(tprs)) if tprs else 0.0

def passes(metric, value):
    thr, direction = THRESHOLDS[metric]
    return (value >= thr) if direction == 'above' else (value < thr)

log("N-sensitivity: 3 cells × 8 N × K=500 ...")
rows = []
rng = np.random.default_rng(RANDOM_STATE)
y_te_pos = np.where(y_te == 1)[0]
y_te_neg = np.where(y_te == 0)[0]
pos_rate = y_te.mean()

for (mk, attr) in TARGET_CELLS:
    log(f"  cell ({mk}, {attr})")
    prot = prot_te[attr]
    for N in N_GRID:
        n_pos = int(N * pos_rate); n_neg = N - n_pos
        n_pass = 0
        vals = []
        for k in range(K_VFR):
            ix = np.concatenate([rng.choice(y_te_pos, n_pos, replace=True),
                                 rng.choice(y_te_neg, n_neg, replace=True)])
            yp_b = pred[ix]
            prot_b = prot[ix]
            y_b = y_te[ix]
            if mk == 'DI':
                v = disparate_impact(yp_b, prot_b)
            elif mk == 'EOPP':
                v = eopp(yp_b, prot_b, y_b)
            else:
                raise NotImplementedError(mk)
            vals.append(v)
            if passes(mk, v):
                n_pass += 1
        n_fail = K_VFR - n_pass
        vfr = min(n_pass, n_fail) / K_VFR
        vals = np.array(vals)
        # Hoeffding-style bound on per-resample tail: P(|v - E[v]| > ε) ≤ 2 exp(-2 N ε² / R²)
        # For DI: R = 1 (DI ∈ [0, 1]); for EOPP: R = 1 (TPR ∈ [0, 1])
        m_mean = float(vals.mean())
        m_sd   = float(vals.std(ddof=1))
        thr_v  = THRESHOLDS[mk][0]
        margin = abs(m_mean - thr_v)
        # Hoeffding upper bound on flip probability
        hoeffding_bound = 2.0 * np.exp(-2.0 * N * margin**2)
        # CLT-style estimate: P(flip) ≈ Φ(-margin / SE), where SE ≈ SD across resamples
        if m_sd > 1e-9:
            from math import erf, sqrt
            z = margin / m_sd
            clt_estimate = 0.5 * (1 - erf(z / sqrt(2)))
        else:
            clt_estimate = 0.0
        rows.append({
            'metric': mk, 'attribute': attr, 'N': N,
            'mean_metric': round(m_mean, 4),
            'SD_metric': round(m_sd, 4),
            'threshold': thr_v,
            'margin': round(margin, 4),
            'empirical_VFR': round(vfr, 4),
            'hoeffding_upper_bound': round(min(hoeffding_bound, 0.5), 4),
            'CLT_flip_estimate': round(clt_estimate, 4),
            'n_pass': n_pass, 'n_fail': n_fail,
        })

T_N = pd.DataFrame(rows)
T_N.to_csv(TAB / 'T_N_sensitivity.csv', index=False)
log(f"saved T_N_sensitivity.csv ({len(T_N)} rows)")

# ============================================================
# Phase D — C3 vs C4 binding-constraint per-cell VFR comparison
# ============================================================
log("loading T13_axis1_vfr_config{3,4}.csv ...")
T_C3 = pd.read_csv(TAB / 'T13_axis1_vfr_config3.csv')
T_C4 = pd.read_csv(TAB / 'T13_axis1_vfr_config4.csv')

binding_rows = []
for (mk, attr) in [('DI', 'RACE'), ('DI', 'AGE_GROUP'), ('SPD', 'RACE'), ('SPD', 'AGE_GROUP')]:
    c3 = T_C3[(T_C3['metric'] == mk) & (T_C3['attribute'] == attr)]
    c4 = T_C4[(T_C4['metric'] == mk) & (T_C4['attribute'] == attr)]
    if len(c3) == 0 or len(c4) == 0:
        continue
    c3 = c3.iloc[0]; c4 = c4.iloc[0]
    delta = c4['vfr'] - c3['vfr']
    direction = 'reduced' if delta < -0.005 else ('increased' if delta > 0.005 else 'unchanged')
    binding_rows.append({
        'metric': mk, 'attribute': attr,
        'C3_n_pass': c3['n_pass'], 'C3_VFR': round(c3['vfr'], 4), 'C3_verdict': c3['verdict_dominant'],
        'C4_n_pass': c4['n_pass'], 'C4_VFR': round(c4['vfr'], 4), 'C4_verdict': c4['verdict_dominant'],
        'delta_VFR': round(delta, 4),
        'greedy_effect': direction,
    })

T_BIND = pd.DataFrame(binding_rows)
T_BIND.to_csv(TAB / 'T_C3_C4_binding_VFR.csv', index=False)
log(f"saved T_C3_C4_binding_VFR.csv ({len(T_BIND)} rows)")

# ============================================================
# Summary print
# ============================================================
print()
print("=" * 80)
print("T_N_sensitivity (DI Race only, for brevity)")
print("=" * 80)
print(T_N[T_N['attribute'] == 'RACE'][T_N['metric'] == 'DI'][[
    'N', 'mean_metric', 'SD_metric', 'margin', 'empirical_VFR',
    'hoeffding_upper_bound', 'CLT_flip_estimate'
]].to_string(index=False))
print()
print("=" * 80)
print("T_C3_C4_binding_VFR")
print("=" * 80)
print(T_BIND.to_string(index=False))
print()
log("DONE")
