"""
End-to-end verification of the manuscript claim that the canonical XGBoost
under the Phase 5b pipeline has:
  - DI(Race) point estimate ~ 0.801
  - VFR(DI, Race) ~ 0.476 under K=500 stratified bootstrap with N=10,000

This script:
  1. Loads + feature-engineers the data identically to the canonical pipeline
  2. Trains canonical XGBoost (n_est=1500)
  3. Applies the canonical Phase 5b α-search + greedy refinement on the
     per-attribute thresholds to satisfy all-four-DI >= 0.80
  4. Computes DI Race point estimate
  5. Runs K=500 bootstrap at N=10,000 and computes VFR
  6. Compares against the manuscript's claim and against T13_axis1_vfr_config4.csv
"""
import pandas as pd, numpy as np, sys, io, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb

DATA = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\data\texas_100x.csv")
TAB  = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\output_final\tables")

RANDOM_STATE = 42
LOS_THRESHOLD = 3
K_VFR = 500
N_VFR = 10_000
DI_TARGET = 0.80

t0 = time.time()
def log(msg): print(f"[{time.time()-t0:>5.0f}s] {msg}", flush=True)

# ============================================================
# Phase A: load + features
# ============================================================
log("load + features")
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
for col in ["ADMITTING_DIAGNOSIS","PRINC_SURG_PROC_CODE","THCIC_ID"]:
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
prot_te = {
    'RACE': df['RACE'].values[idx_te],
    'SEX': df['SEX_CODE'].values[idx_te] if 'SEX_CODE' in df.columns else df['SEX'].values[idx_te],
    'ETHNICITY': df['ETHNICITY'].values[idx_te],
    'AGE_GROUP': df['AGE_GROUP'].values[idx_te],
}
log(f"  Test {len(X_te):,}")

# ============================================================
# Phase B: canonical XGBoost
# ============================================================
log("training canonical XGBoost")
mdl = xgb.XGBClassifier(
    n_estimators=1500, max_depth=10, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.85, min_child_weight=3, reg_lambda=1.0,
    tree_method='hist', random_state=RANDOM_STATE, seed=RANDOM_STATE,
    eval_metric='logloss', verbosity=0, n_jobs=1,
)
mdl.fit(X_tr, y_tr)
proba = mdl.predict_proba(X_te)[:, 1].astype('float32')
pred_std = (proba >= 0.5).astype(int)
log(f"  Standard acc={accuracy_score(y_te, pred_std):.4f}  auc={roc_auc_score(y_te, proba):.4f}")

def di(yp, prot):
    groups = np.unique(prot)
    rates = [yp[prot==g].mean() for g in groups if (prot==g).sum() > 0]
    if not rates or max(rates) <= 1e-9: return 1.0
    return float(min(rates) / max(rates))

# ============================================================
# Phase C: canonical Phase 5b — α-search per attribute + greedy refinement
# ============================================================
log("Phase 5b α-search + greedy refinement on per-(attr, group) thresholds")

# Initialize per-group thresholds at 0.5
attr_keys = ['RACE', 'SEX', 'ETHNICITY', 'AGE_GROUP']
thresholds = {a: {int(g): 0.5 for g in np.unique(prot_te[a])} for a in attr_keys}

def predict_with_thresh(proba, prot_te, thresholds):
    """Apply per-attribute thresholds: for each record, the effective threshold
    is the minimum across its protected attributes (lower threshold = more positives).
    This is the "intersectional union" rule: a record can flip positive if ANY of
    its protected-attribute thresholds is met."""
    yhat = (proba >= 0.5).astype(int)
    n = len(proba)
    eff_thresh = np.full(n, 0.5, dtype='float32')
    for a in attr_keys:
        arr = prot_te[a]
        for g, tau in thresholds[a].items():
            mask = (arr == g)
            eff_thresh[mask] = np.minimum(eff_thresh[mask], tau)
    return (proba >= eff_thresh).astype(int)

# α-search: for each attribute, find threshold per group that pushes DI to target
def alpha_search(proba, prot, target_di=0.82):
    groups = np.unique(prot)
    thr = {int(g): 0.5 for g in groups}
    rates = {int(g): float((proba[prot==g] >= 0.5).mean()) for g in groups}
    max_rate = max(rates.values())
    target_min_sr = target_di * max_rate
    for g in groups:
        g = int(g)
        if rates[g] >= target_min_sr:
            continue
        cand = np.linspace(0.05, 0.95, 91)
        best_tau = 0.5; best_diff = 1e9
        for tau in cand:
            sr_g = float((proba[prot==g] >= tau).mean())
            if sr_g >= target_min_sr:
                diff = abs(sr_g - target_min_sr)
                if diff < best_diff:
                    best_diff = diff; best_tau = tau
        thr[g] = best_tau
    return thr

# Apply α-search starting from Age (binding) then Race
for a in ['AGE_GROUP', 'RACE']:
    thresholds[a] = alpha_search(proba, prot_te[a], target_di=0.82)

yhat = predict_with_thresh(proba, prot_te, thresholds)
log(f"  After α-search: DI Race={di(yhat, prot_te['RACE']):.4f}  DI Age={di(yhat, prot_te['AGE_GROUP']):.4f}  acc={accuracy_score(y_te, yhat):.4f}")

# Greedy refinement: walk per-attribute thresholds inward until DI >= 0.80 + margin
log("greedy refinement ...")
EPS = 0.01
margin = 0.005  # walk DI to 0.80 + 0.005
for step in range(50):
    changed = False
    for a in ['RACE', 'AGE_GROUP']:
        cur_di = di(yhat, prot_te[a])
        if cur_di < DI_TARGET + margin:
            # Find min-rate group; lower its threshold by EPS
            groups = np.unique(prot_te[a])
            rates = {int(g): float(yhat[prot_te[a]==g].mean()) for g in groups}
            min_g = min(rates, key=rates.get)
            old_t = thresholds[a][min_g]
            new_t = max(0.05, old_t - EPS)
            if new_t < old_t:
                thresholds[a][min_g] = new_t
                yhat = predict_with_thresh(proba, prot_te, thresholds)
                changed = True
    if not changed:
        break

log(f"  After greedy: DI Race={di(yhat, prot_te['RACE']):.4f}  DI Age={di(yhat, prot_te['AGE_GROUP']):.4f}  acc={accuracy_score(y_te, yhat):.4f}")
log(f"  Final per-attr thresholds:")
for a in attr_keys:
    log(f"    {a}: {thresholds[a]}")

# ============================================================
# Phase D: K=500 stratified bootstrap VFR for (DI, Race)
# ============================================================
log(f"K={K_VFR} stratified bootstrap at N={N_VFR}")
rng = np.random.default_rng(RANDOM_STATE)
y_te_pos = np.where(y_te == 1)[0]
y_te_neg = np.where(y_te == 0)[0]
pos_rate = y_te.mean()
n_pos = int(N_VFR * pos_rate); n_neg = N_VFR - n_pos

# For DI Race, we resample N_VFR records and recompute DI
n_pass_DI_RACE = 0
n_pass_DI_AGE  = 0
di_race_vals = []
di_age_vals  = []
for k in range(K_VFR):
    ix = np.concatenate([rng.choice(y_te_pos, n_pos, replace=True),
                         rng.choice(y_te_neg, n_neg, replace=True)])
    yp_b = yhat[ix]
    prot_race_b = prot_te['RACE'][ix]
    prot_age_b  = prot_te['AGE_GROUP'][ix]
    v_race = di(yp_b, prot_race_b)
    v_age  = di(yp_b, prot_age_b)
    di_race_vals.append(v_race)
    di_age_vals.append(v_age)
    if v_race >= DI_TARGET: n_pass_DI_RACE += 1
    if v_age  >= DI_TARGET: n_pass_DI_AGE  += 1

VFR_DI_RACE = min(n_pass_DI_RACE, K_VFR - n_pass_DI_RACE) / K_VFR
VFR_DI_AGE  = min(n_pass_DI_AGE,  K_VFR - n_pass_DI_AGE)  / K_VFR

print()
print("=" * 78)
print("END-TO-END VERIFICATION RESULT")
print("=" * 78)
print(f"Canonical XGBoost · Phase 5b pipeline (α-search + greedy refinement)")
print(f"Test partition N = {len(y_te):,}")
print(f"Final accuracy   = {accuracy_score(y_te, yhat):.4f}")
print(f"AUROC            = {roc_auc_score(y_te, proba):.4f}")
print()
print(f"Point-estimate DI:")
print(f"  Race: {di(yhat, prot_te['RACE']):.4f}  ({'PASS' if di(yhat, prot_te['RACE']) >= 0.80 else 'FAIL'})")
print(f"  Age:  {di(yhat, prot_te['AGE_GROUP']):.4f}  ({'PASS' if di(yhat, prot_te['AGE_GROUP']) >= 0.80 else 'FAIL'})")
print()
print(f"Stratified-bootstrap VFR (K={K_VFR}, N={N_VFR}):")
print(f"  DI Race: n_pass={n_pass_DI_RACE}, n_fail={K_VFR - n_pass_DI_RACE}, VFR={VFR_DI_RACE:.4f}")
print(f"  DI Age:  n_pass={n_pass_DI_AGE},  n_fail={K_VFR - n_pass_DI_AGE},  VFR={VFR_DI_AGE:.4f}")
print()
print(f"Reference (T13_axis1_vfr_config4.csv, manuscript C4):")
T4 = pd.read_csv(TAB / 'T13_axis1_vfr_config4.csv')
ref_race = T4[(T4['metric']=='DI') & (T4['attribute']=='RACE')].iloc[0]
ref_age  = T4[(T4['metric']=='DI') & (T4['attribute']=='AGE_GROUP')].iloc[0]
print(f"  DI Race: n_pass={ref_race['n_pass']:.0f}, n_fail={ref_race['n_fail']:.0f}, VFR={ref_race['vfr']:.4f}")
print(f"  DI Age:  n_pass={ref_age['n_pass']:.0f},  n_fail={ref_age['n_fail']:.0f},  VFR={ref_age['vfr']:.4f}")
print()
print(f"Reproduction match:")
print(f"  DI Race VFR:  computed {VFR_DI_RACE:.4f}  vs  stored {ref_race['vfr']:.4f}  (|Δ| = {abs(VFR_DI_RACE - ref_race['vfr']):.4f})")
print(f"  DI Age VFR:   computed {VFR_DI_AGE:.4f}   vs  stored {ref_age['vfr']:.4f}   (|Δ| = {abs(VFR_DI_AGE - ref_age['vfr']):.4f})")

# Save the per-resample DI values for inspection
out = pd.DataFrame({'k': np.arange(K_VFR),
                    'DI_race': di_race_vals,
                    'DI_age': di_age_vals,
                    'pass_DI_race': [int(v >= 0.80) for v in di_race_vals],
                    'pass_DI_age':  [int(v >= 0.80) for v in di_age_vals]})
out.to_csv(TAB / 'T_DI_Race_VFR_verification.csv', index=False)
print(f"\nSaved per-resample DI values to T_DI_Race_VFR_verification.csv ({K_VFR} rows)")
log("DONE")
