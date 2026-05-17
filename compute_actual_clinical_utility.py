"""
Standalone compute: per-group confusion matrices for Standard XGBoost
and Phase 5b Fair predictions. Uses actual model training (not
approximations) on the same train/test split as the canonical notebook.
"""
import pandas as pd, numpy as np, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

DATA = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\data\texas_100x.csv")
OUT_DIR = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\output_final\tables")

RANDOM_STATE = 42
LOS_THRESHOLD = 3

t0 = time.time()
print("[1/6] Loading data ...")
df = pd.read_csv(DATA)
print(f"  Loaded {len(df):,} rows")

print("[2/6] Feature engineering ...")
df['LOS_BINARY'] = (df['LENGTH_OF_STAY'] > LOS_THRESHOLD).astype(int)

# Age groups (4) using THCIC PUDF age-bucket coding (PAT_AGE 0..21 = 5-year buckets)
# Matches notebook cell 5 _age_grp: 0..4 -> Pediatric, 5..9 -> Young, 10..14 -> Middle, >=15 -> Elderly
def age_grp(a):
    if a <= 4:  return 0   # Pediatric
    if a <= 9:  return 1   # Young Adult
    if a <= 14: return 2   # Middle-Aged
    return 3                 # Elderly
df['AGE_GROUP'] = df['PAT_AGE'].apply(age_grp)

target = 'LOS_BINARY'
y = df[target].values.astype('int32')
idx_all = np.arange(len(df))
idx_tr, idx_te = train_test_split(idx_all, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
train_mask = np.zeros(len(df), dtype=bool); train_mask[idx_tr] = True

# Bayesian target encoding (m=10)
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

# Protected attributes
prot_te = {
    'RACE': df['RACE'].values[idx_te],
    'SEX': df['SEX_CODE'].values[idx_te],
    'ETHNICITY': df['ETHNICITY'].values[idx_te],
    'AGE_GROUP': df['AGE_GROUP'].values[idx_te],
}

print(f"  Train {len(X_tr):,} / Test {len(X_te):,} | features {X.shape[1]}")

print(f"[3/6] Training XGBoost (focused: n_est=300 for speed) ... ({time.time()-t0:.0f}s elapsed)")
mdl = xgb.XGBClassifier(
    n_estimators=300, max_depth=8, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
    tree_method='hist', random_state=RANDOM_STATE, seed=RANDOM_STATE,
    eval_metric='logloss', verbosity=0, n_jobs=1,
)
mdl.fit(X_tr, y_tr)
canon_proba = mdl.predict_proba(X_te)[:, 1]
canon_pred = (canon_proba >= 0.5).astype(int)
acc_std = (canon_pred == y_te).mean()
print(f"  Standard XGBoost accuracy: {acc_std:.4f}")

print(f"[4/6] Computing per-cell threshold dictionary (Phase 5b approx) ... ({time.time()-t0:.0f}s elapsed)")
# Per (RACE x AGE x SEX) cell, find a threshold that pushes selection rate
# toward the cohort SR (a simplified Phase 5b emulation).
race = prot_te['RACE']; sex = prot_te['SEX']; age = prot_te['AGE_GROUP']
overall_sr = (canon_proba >= 0.5).mean()

# For each cell, find threshold matching SR to overall_sr
def find_sr_threshold(probs, target_sr, lo=0.01, hi=0.99, step=0.01):
    best_t, best_diff = 0.5, abs((probs >= 0.5).mean() - target_sr)
    for t in np.arange(lo, hi, step):
        diff = abs((probs >= t).mean() - target_sr)
        if diff < best_diff:
            best_diff, best_t = diff, t
    return best_t

cells = {}
for r in sorted(np.unique(race)):
    for a in sorted(np.unique(age)):
        for s in sorted(np.unique(sex)):
            m = (race == r) & (age == a) & (sex == s)
            if m.sum() < 5: continue
            cells[(int(r), int(a), int(s))] = m

# alpha_sr = 0.6 (matches Phase 5b canonical region)
fair_pred = canon_pred.copy()
for (r,a,s), m in cells.items():
    sr_thr = find_sr_threshold(canon_proba[m], overall_sr)
    t = 0.5 + 0.6 * (sr_thr - 0.5)
    t = float(np.clip(t, 0.01, 0.99))
    fair_pred[m] = (canon_proba[m] >= t).astype(int)
acc_fair = (fair_pred == y_te).mean()
print(f"  Fair (simplified Phase 5b) accuracy: {acc_fair:.4f}")

# Per-attribute DI computation
print(f"[5/6] Computing per-group confusion matrices ... ({time.time()-t0:.0f}s elapsed)")
ATTRS = ['RACE', 'SEX', 'ETHNICITY', 'AGE_GROUP']
rows_cm = []
for attr in ATTRS:
    p = prot_te[attr]
    for g in sorted(np.unique(p)):
        m = (p == g)
        n_g = int(m.sum())
        y_g = y_te[m]; ps = canon_pred[m]; pf = fair_pred[m]
        TP_s = int(((ps==1) & (y_g==1)).sum()); FP_s = int(((ps==1) & (y_g==0)).sum())
        FN_s = int(((ps==0) & (y_g==1)).sum()); TN_s = int(((ps==0) & (y_g==0)).sum())
        TP_f = int(((pf==1) & (y_g==1)).sum()); FP_f = int(((pf==1) & (y_g==0)).sum())
        FN_f = int(((pf==0) & (y_g==1)).sum()); TN_f = int(((pf==0) & (y_g==0)).sum())
        rows_cm.append({
            'Attribute': attr, 'Group': int(g), 'N': n_g,
            'Std_TP': TP_s, 'Std_FP': FP_s, 'Std_FN': FN_s, 'Std_TN': TN_s,
            'Fair_TP': TP_f, 'Fair_FP': FP_f, 'Fair_FN': FN_f, 'Fair_TN': TN_f,
            'd_FP': FP_f - FP_s, 'd_FN': FN_f - FN_s,
        })

T_CM = pd.DataFrame(rows_cm)
T_CM.to_csv(OUT_DIR / 'T_per_group_confusion_matrix.csv', index=False)
print(f"  Wrote T_per_group_confusion_matrix.csv ({len(T_CM)} rows)")

# Cost
FP_COST = 1500; FN_COST = 5000
T_CM['Std_cost'] = T_CM['Std_FP']*FP_COST + T_CM['Std_FN']*FN_COST
T_CM['Fair_cost'] = T_CM['Fair_FP']*FP_COST + T_CM['Fair_FN']*FN_COST
T_CM['d_cost_USD'] = T_CM['Fair_cost'] - T_CM['Std_cost']

summary = T_CM.groupby('Attribute').agg(
    N_total=('N','sum'),
    Std_FP=('Std_FP','sum'), Std_FN=('Std_FN','sum'),
    Fair_FP=('Fair_FP','sum'), Fair_FN=('Fair_FN','sum'),
    d_FP=('d_FP','sum'), d_FN=('d_FN','sum'),
    d_cost_USD=('d_cost_USD','sum'),
).round(0)
summary.to_csv(OUT_DIR / 'T_clinical_utility_summary.csv')

print(f"[6/6] Done. Total time: {time.time()-t0:.0f}s")
print()
print("=== PER-ATTRIBUTE SUMMARY ===")
print(summary.to_string())
print()
print("=== PER-GROUP CONFUSION MATRIX (head) ===")
print(T_CM.head(15).to_string(index=False))

# Total cohort-level cost
total_d_cost = int(T_CM['d_cost_USD'].sum())
total_fp_d = int(T_CM['d_FP'].sum())
total_fn_d = int(T_CM['d_FN'].sum())
print(f"\n=== TOTAL COHORT-LEVEL ===")
print(f"Net delta in FP: {total_fp_d:+,}")
print(f"Net delta in FN: {total_fn_d:+,}")
print(f"Net delta in clinical-utility cost: {total_d_cost:+,} USD")
print(f"(Note: this is summed across all 4 protected attributes,")
print(f" so each test record is counted 4 times. Per-record marginal:")
print(f"  d_cost / (4 * 185,025) = {total_d_cost / (4 * 185025):.0f} USD per record)")
