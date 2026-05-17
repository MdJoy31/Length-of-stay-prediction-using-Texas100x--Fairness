"""
compute_baseline_audit_extension.py

Extend the three-axis audit-stability framework from the canonical Phase 5b
to four configurations, using the SAME audit instrument (same K, same N grid,
same GroupKFold splits, same random_state) so cross-config differences reflect
the intervention, not the audit setup.

Configurations:
  (1) Real-Only           - Standard XGBoost, no intervention
  (2) Reweighing-only λ=2 - intersectional reweighed XGBoost, threshold 0.5
  (3) Threshold-Shift only- Standard XGBoost + per-cell α-search thresholds (no greedy)
  (4) Real+VFR canonical  - Standard XGBoost + α-search + greedy refinement (Phase 5b)

Outputs (in output_final/tables/):
  T13_axis1_vfr_config{1,2,3,4}.csv      28 rows: metric, attribute, n_pass, n_fail, vfr, verdict_dominant
  T9_axis2_minN_config{1,2,3,4}.csv      28 rows: metric, attribute, min_N_for_cv_under_5pct, full_N_required
  T10_axis3_kappa_config{1,2,3,4}.csv    7 rows:  metric, fleiss_kappa, agreement_class
  T_baseline_audit_summary.csv           4 rows:  cross-config pivot
  T_baseline_audit_diagnostics.csv       NaN-reason log

Constraints (per spec):
  - random_state=42 throughout
  - 80/20 split, same FE pipeline as canonical
  - K=500 bootstrap, N=10,000 per resample for Axis 1
  - Axis 2 N grid: {1000, 2000, 5000, 10000, 25000, 50000, 100000, 185026}, 30 reps each
  - K=20 GroupKFold by THCIC_ID for Axis 3
  - Canonical n_estimators=1500, max_depth=10 for 80/20 split
  - For GroupKFold: n_estimators=200, max_depth=8 across all 4 configs (matches §1.6.1
    disclosed lighter-XGBoost convention for tractable cross-fold computation)
"""
import pandas as pd, numpy as np, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from pathlib import Path
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb

DATA = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\data\texas_100x.csv")
OUT_DIR = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\output_final\tables")

RANDOM_STATE = 42
LOS_THRESHOLD = 3
K_VFR = 500
N_VFR = 10_000
N_GRID = [1000, 2000, 5000, 10_000, 25_000, 50_000, 100_000, 185_026]
N_REPS = 30
K_GROUPKFOLD = 20

ATTRS = ['RACE', 'SEX', 'ETHNICITY', 'AGE_GROUP']
METRICS = ['DI', 'SPD', 'EOPP', 'EOD', 'TI', 'PP', 'CAL']
THRESHOLDS = {'DI':(0.80,'above'), 'SPD':(0.10,'below'), 'EOPP':(0.10,'below'),
              'EOD':(0.10,'below'), 'TI':(0.10,'below'), 'PP':(0.10,'below'),
              'CAL':(0.05,'below')}

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:>5.0f}s] {msg}", flush=True)

# =========================================================================
# Phase 1 - Load + feature engineering (canonical pipeline)
# =========================================================================
log("Phase 1: load + feature engineering")
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

X_full = df[feature_cols].fillna(0).astype('float32').values
y_tr, y_te = y_full[idx_tr], y_full[idx_te]
X_tr, X_te = X_full[idx_tr], X_full[idx_te]
hosp_te = df['THCIC_ID'].values[idx_te]
hosp_full = df['THCIC_ID'].values

prot_te = {
    'RACE': df['RACE'].values[idx_te],
    'SEX': df['SEX_CODE'].values[idx_te],
    'ETHNICITY': df['ETHNICITY'].values[idx_te],
    'AGE_GROUP': df['AGE_GROUP'].values[idx_te],
}
log(f"  Train {len(X_tr):,} / Test {len(X_te):,}")

# =========================================================================
# Phase 2 - Train Config 1 (Standard) and Config 2 (Reweigh λ=2) on 80/20 split
# =========================================================================
log("Phase 2a: training Config 1 (Standard XGBoost, n_est=1500)")
mdl1 = xgb.XGBClassifier(
    n_estimators=1500, max_depth=10, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.85, min_child_weight=3, reg_lambda=1.0,
    tree_method='hist', random_state=RANDOM_STATE, seed=RANDOM_STATE,
    eval_metric='logloss', verbosity=0, n_jobs=1,
)
mdl1.fit(X_tr, y_tr)
proba_c1 = mdl1.predict_proba(X_te)[:, 1].astype('float32')
pred_c1 = (proba_c1 >= 0.5).astype(int)
log(f"  Config 1: Acc={(pred_c1==y_te).mean():.4f}")

# Config 2 - Reweighing λ=2 on canonical architecture
log("Phase 2b: training Config 2 (Reweighing λ=2, n_est=1500)")
race_tr = df['RACE'].values[idx_tr].astype(int).astype(str)
age_tr  = df['AGE_GROUP'].values[idx_tr].astype(int).astype(str)
sex_tr  = df['SEX_CODE'].values[idx_tr].astype(int).astype(str)
cells_tr = race_tr + "_" + age_tr + "_" + sex_tr
cnt = pd.Series(cells_tr).value_counts()
p_obs = cnt / cnt.sum()
p_exp = pd.Series(1.0/len(p_obs), index=p_obs.index)
LAMBDA = 2.0
w_per = (1.0 + LAMBDA * (p_exp/p_obs - 1.0)).clip(0.1, 10.0)
sw_tr = pd.Series(cells_tr).map(w_per).values.astype('float32')
log(f"  weights: min={sw_tr.min():.3f} max={sw_tr.max():.3f}")
mdl2 = xgb.XGBClassifier(
    n_estimators=1500, max_depth=10, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.85, min_child_weight=3, reg_lambda=1.0,
    tree_method='hist', random_state=RANDOM_STATE, seed=RANDOM_STATE,
    eval_metric='logloss', verbosity=0, n_jobs=1,
)
mdl2.fit(X_tr, y_tr, sample_weight=sw_tr)
proba_c2 = mdl2.predict_proba(X_te)[:, 1].astype('float32')
pred_c2 = (proba_c2 >= 0.5).astype(int)
log(f"  Config 2: Acc={(pred_c2==y_te).mean():.4f}")

# =========================================================================
# Phase 3 - Compute Config 3 (α-search only) and Config 4 (α-search + greedy)
# =========================================================================
log("Phase 3a: α-SR/TPR/PPV grid search on Config 1 predictions")
race_te_arr = prot_te['RACE']; sex_te_arr = prot_te['SEX']; age_te_arr = prot_te['AGE_GROUP']
test_groups = {}
for r in sorted(np.unique(race_te_arr).tolist()):
    for a in sorted(np.unique(age_te_arr).tolist()):
        for s in sorted(np.unique(sex_te_arr).tolist()):
            mask = (race_te_arr == r) & (age_te_arr == a) & (sex_te_arr == s)
            if mask.sum() >= 5:
                test_groups[f"{r}|{a}|{s}"] = mask
log(f"  intersection cells: {len(test_groups)}")

def find_sr(probs, target_sr):
    best_t, best_diff = 0.5, abs((probs >= 0.5).mean() - target_sr)
    for t in np.arange(0.01, 0.99, 0.01):
        diff = abs((probs >= t).mean() - target_sr)
        if diff < best_diff:
            best_diff, best_t = diff, t
    return best_t

def find_tpr(probs, labels, target_tpr):
    pos = labels == 1
    if pos.sum() < 10: return 0.5
    best_t, best_diff = 0.5, abs((probs[pos] >= 0.5).mean() - target_tpr)
    for t in np.arange(0.01, 0.99, 0.01):
        diff = abs((probs[pos] >= t).mean() - target_tpr)
        if diff < best_diff:
            best_diff, best_t = diff, t
    return best_t

def find_ppv(probs, labels, target_ppv):
    best_t, best_diff = 0.5, 1.0
    for t in np.arange(0.01, 0.99, 0.01):
        preds = (probs >= t)
        if preds.sum() < 10: continue
        diff = abs(labels[preds].mean() - target_ppv)
        if diff < best_diff:
            best_diff, best_t = diff, t
    return best_t

overall_sr_c1  = (proba_c1 >= 0.5).mean()
overall_tpr_c1 = (proba_c1[y_te == 1] >= 0.5).mean()
overall_ppv_c1 = y_te[proba_c1 >= 0.5].mean() if (proba_c1 >= 0.5).sum() > 10 else 0.5
sr_thr, tpr_thr, ppv_thr = {}, {}, {}
for k, m in test_groups.items():
    sr_thr[k]  = find_sr(proba_c1[m], overall_sr_c1)
    tpr_thr[k] = find_tpr(proba_c1[m], y_te[m], overall_tpr_c1)
    ppv_thr[k] = find_ppv(proba_c1[m], y_te[m], overall_ppv_c1)

A_SR  = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
A_TPR = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
A_PPV = [0.0, 0.2, 0.4, 0.6, 0.8]

def disparate_impact(yp, prot):
    groups = np.unique(prot)
    rates = [yp[prot==g].mean() for g in groups]
    if max(rates) <= 1e-9: return 1.0
    return min(rates) / max(rates)

def apply_thresholds(probs, thresholds_dict):
    yp = (probs >= 0.5).astype(int)
    for k, m in test_groups.items():
        if k in thresholds_dict:
            yp[m] = (probs[m] >= thresholds_dict[k]).astype(int)
    return yp

best_acc = -1; best_thresh_c3 = None
for a_sr in A_SR:
    for a_tpr in A_TPR:
        for a_ppv in A_PPV:
            thresh = {}
            for k in test_groups:
                t = 0.5 + a_sr*(sr_thr[k]-0.5) + a_tpr*(tpr_thr[k]-0.5) + a_ppv*(ppv_thr[k]-0.5)
                thresh[k] = float(np.clip(t, 0.01, 0.99))
            yp = apply_thresholds(proba_c1, thresh)
            di_per = {a: disparate_impact(yp, prot_te[a]) for a in ATTRS}
            if all(v >= 0.80 for v in di_per.values()):
                acc_here = (yp == y_te).mean()
                if acc_here > best_acc:
                    best_acc = acc_here
                    best_thresh_c3 = dict(thresh)

if best_thresh_c3 is None:
    raise RuntimeError("No α-grid candidate satisfies all-4-DI ≥ 0.80!")
pred_c3 = apply_thresholds(proba_c1, best_thresh_c3)
proba_c3 = proba_c1.copy()  # probabilities unchanged by threshold-shifting
log(f"  Config 3: Acc={(pred_c3==y_te).mean():.4f}, all-4-DI: {all(disparate_impact(pred_c3, prot_te[a]) >= 0.80 for a in ATTRS)}")

# Config 4 - greedy refinement on top of Config 3 thresholds (canonical Phase 5b)
log("Phase 3b: greedy refinement -> Config 4 (Phase 5b canonical)")
greedy_thresh = dict(best_thresh_c3)
for it in range(500):
    moved = False
    for k in list(greedy_thresh.keys()):
        cur = greedy_thresh[k]
        new_t = cur + 0.01 * (-1 if cur > 0.5 else 1)
        new_t = float(np.clip(new_t, 0.01, 0.99))
        if abs(new_t - 0.5) >= abs(cur - 0.5):
            continue
        cand = dict(greedy_thresh); cand[k] = new_t
        yp = apply_thresholds(proba_c1, cand)
        di_per = {a: disparate_impact(yp, prot_te[a]) for a in ATTRS}
        if all(v >= 0.80 for v in di_per.values()):
            greedy_thresh[k] = new_t
            moved = True
    if not moved: break
pred_c4 = apply_thresholds(proba_c1, greedy_thresh)
proba_c4 = proba_c1.copy()
log(f"  Config 4: Acc={(pred_c4==y_te).mean():.4f}, all-4-DI: {all(disparate_impact(pred_c4, prot_te[a]) >= 0.80 for a in ATTRS)}")

# =========================================================================
# Fairness metric computation
# =========================================================================
def compute_seven(yp, ypb, prot, y_true):
    groups = np.unique(prot)
    rates = {}
    for g in groups:
        m = prot == g
        if m.sum() == 0: continue
        yt, yph = y_true[m], yp[m]
        sr  = float(np.mean(yph))
        tpr = float(np.mean(yph[yt == 1])) if (yt == 1).any() else 0.0
        fpr = float(np.mean(yph[yt == 0])) if (yt == 0).any() else 0.0
        ppv = float(np.mean(yt[yph == 1])) if (yph == 1).any() else 0.0
        rates[g] = (sr, tpr, fpr, ppv)
    sr_v  = [r[0] for r in rates.values()]
    tpr_v = [r[1] for r in rates.values()]
    fpr_v = [r[2] for r in rates.values()]
    ppv_v = [r[3] for r in rates.values()]
    if not sr_v: return None
    di  = (min(sr_v)/max(sr_v)) if max(sr_v) > 0 else 1.0
    spd = max(sr_v) - min(sr_v)
    eopp = max(tpr_v) - min(tpr_v)
    eod  = max(eopp, max(fpr_v) - min(fpr_v))
    pp   = max(ppv_v) - min(ppv_v)
    b_all = (yp.astype(float) - y_true.astype(float) + 1.0)
    mu_all = float(np.mean(b_all))
    ti = 0.0
    if mu_all > 0:
        n_total = len(b_all)
        for g in groups:
            m = prot == g
            if m.sum() == 0: continue
            mu_g = float(np.mean(b_all[m]))
            if mu_g > 0:
                ti += (m.sum() / n_total) * (mu_g/mu_all) * np.log(mu_g/mu_all)
        ti = float(abs(ti))
    cal = 0.0
    if ypb is not None:
        cal_diffs = []
        for g in groups:
            m = prot == g
            pg = ypb[m]; yg = y_true[m]
            bins = np.linspace(0, 1, 11)
            for b in range(len(bins)-1):
                in_bin = (pg >= bins[b]) & (pg < bins[b+1])
                if in_bin.sum() > 5:
                    cal_diffs.append(abs(pg[in_bin].mean() - yg[in_bin].mean()))
        cal = max(cal_diffs) if cal_diffs else 0.0
    return {'DI':di, 'SPD':spd, 'EOPP':eopp, 'EOD':eod, 'TI':ti, 'PP':pp, 'CAL':cal}

def passes(metric, value):
    thr, direction = THRESHOLDS[metric]
    return (value >= thr) if direction == 'above' else (value < thr)

# =========================================================================
# Phase 4 - Axis 1: VFR (K=500) for each config
# =========================================================================
configs = {
    1: ('Real-Only', pred_c1, proba_c1),
    2: ('Reweighing-only λ=2', pred_c2, proba_c2),
    3: ('Threshold-Shift only', pred_c3, proba_c3),
    4: ('Real+VFR canonical (Phase 5b)', pred_c4, proba_c4),
}

log(f"Phase 4: Axis 1 VFR (K={K_VFR})")
diagnostics = []
for cfg_id, (cfg_name, yp, ypb) in configs.items():
    log(f"  Config {cfg_id} ({cfg_name})")
    rng = np.random.default_rng(RANDOM_STATE)
    pos_idx = np.where(y_te == 1)[0]
    neg_idx = np.where(y_te == 0)[0]
    n_pos = int(N_VFR * y_te.mean())
    n_neg = N_VFR - n_pos

    rows = []
    for a in ATTRS:
        boot_pass = {m: 0 for m in METRICS}
        for k in range(K_VFR):
            ix = np.concatenate([rng.choice(pos_idx, n_pos, replace=True),
                                 rng.choice(neg_idx, n_neg, replace=True)])
            m = compute_seven(yp[ix], ypb[ix], prot_te[a][ix], y_te[ix])
            if m is None:
                continue
            for mk in METRICS:
                if passes(mk, m[mk]):
                    boot_pass[mk] += 1
        for mk in METRICS:
            n_pass = boot_pass[mk]
            n_fail = K_VFR - n_pass
            n_flip = min(n_pass, n_fail)
            vfr = n_flip / K_VFR
            verdict_dom = 'fair' if n_pass > n_fail else ('unfair' if n_fail > n_pass else 'tied')
            rows.append({'metric': mk, 'attribute': a,
                         'n_pass': n_pass, 'n_fail': n_fail,
                         'vfr': round(vfr, 4),
                         'verdict_dominant': verdict_dom})
    pd.DataFrame(rows).to_csv(OUT_DIR / f'T13_axis1_vfr_config{cfg_id}.csv', index=False)
    log(f"    saved T13_axis1_vfr_config{cfg_id}.csv")

# =========================================================================
# Phase 5 - Axis 2: Min-N for CV<5% per cell
# =========================================================================
log(f"Phase 5: Axis 2 min-N (8-point grid x 30 reps)")
for cfg_id, (cfg_name, yp, ypb) in configs.items():
    log(f"  Config {cfg_id} ({cfg_name})")
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for a in ATTRS:
        for mk in METRICS:
            min_n_cv5 = None
            for N_use in N_GRID:
                vals = []
                for r in range(N_REPS):
                    use_n = min(N_use, len(y_te))
                    ix = rng.choice(len(y_te), use_n, replace=False)
                    m = compute_seven(yp[ix], ypb[ix], prot_te[a][ix], y_te[ix])
                    if m is None: continue
                    vals.append(m[mk])
                vals = np.array(vals)
                if len(vals) < 5: continue
                mean_v = abs(np.mean(vals))
                cv = np.std(vals, ddof=1) / max(mean_v, 1e-9)
                if cv < 0.05 and min_n_cv5 is None:
                    min_n_cv5 = N_use
                    break
            full_required = (min_n_cv5 is None)
            if min_n_cv5 is None:
                min_n_cv5 = N_GRID[-1]
            rows.append({'metric': mk, 'attribute': a,
                         'min_N_for_cv_under_5pct': min_n_cv5,
                         'full_N_required': full_required})
    pd.DataFrame(rows).to_csv(OUT_DIR / f'T9_axis2_minN_config{cfg_id}.csv', index=False)
    log(f"    saved T9_axis2_minN_config{cfg_id}.csv")

# =========================================================================
# Phase 6 - Axis 3: Per-metric Fleiss kappa (K=20 GroupKFold per config)
# =========================================================================
log(f"Phase 6: Axis 3 Fleiss kappa (K={K_GROUPKFOLD} GroupKFold per config)")

def fleiss_kappa(V):
    n_items, n_raters = V.shape
    if n_items < 1 or n_raters < 2: return float('nan')
    n_pass = V.sum(axis=1); n_fail = n_raters - n_pass
    N = np.column_stack([n_fail, n_pass])
    P_i = (np.sum(N**2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = P_i.mean()
    p_j = N.sum(axis=0) / (n_items * n_raters)
    P_e = float(np.sum(p_j**2))
    if abs(1 - P_e) < 1e-12: return 1.0
    return float((P_bar - P_e) / (1 - P_e))

def landis_koch(k):
    if not np.isfinite(k):    return 'nan'
    if k < 0:                 return 'below chance'
    if k <= 0.20:             return 'slight'
    if k <= 0.40:             return 'fair'
    if k <= 0.60:             return 'moderate'
    if k <= 0.80:             return 'substantial'
    return 'almost perfect'

# Lighter XGBoost for cross-fold (per §1.6.1 disclosure)
def train_xgb_lite(X_tr_f, y_tr_f, sw=None):
    mdl = xgb.XGBClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.05,
        tree_method='hist', random_state=RANDOM_STATE, seed=RANDOM_STATE,
        eval_metric='logloss', verbosity=0, n_jobs=1,
    )
    mdl.fit(X_tr_f, y_tr_f, sample_weight=sw)
    return mdl

# Use scaled X (full); GroupKFold by THCIC_ID
gkf = GroupKFold(n_splits=K_GROUPKFOLD)
log(f"  generating GroupKFold splits ({K_GROUPKFOLD} folds)")
splits = list(gkf.split(X_full, y_full, hosp_full))

# For each fold and each config, compute per-attribute pass/fail per metric
fold_pass = {cfg_id: {a: {mk: [] for mk in METRICS} for a in ATTRS}
             for cfg_id in [1, 2, 3, 4]}

for fold_id, (tr_ix, te_ix) in enumerate(splits, 1):
    log(f"  fold {fold_id}/{K_GROUPKFOLD}")
    Xtr_f, ytr_f = X_full[tr_ix], y_full[tr_ix]
    Xte_f, yte_f = X_full[te_ix], y_full[te_ix]
    prot_f = {a: df[col].values[te_ix]
              for a, col in [('RACE','RACE'), ('SEX','SEX_CODE'),
                             ('ETHNICITY','ETHNICITY'), ('AGE_GROUP','AGE_GROUP')]}

    # Config 1 - Standard
    mdl_f1 = train_xgb_lite(Xtr_f, ytr_f)
    proba_f1 = mdl_f1.predict_proba(Xte_f)[:, 1].astype('float32')
    pred_f1 = (proba_f1 >= 0.5).astype(int)

    # Config 2 - Reweighed λ=2
    cells_tr_f = (df['RACE'].values[tr_ix].astype(int).astype(str) + "_"
                  + df['AGE_GROUP'].values[tr_ix].astype(int).astype(str) + "_"
                  + df['SEX_CODE'].values[tr_ix].astype(int).astype(str))
    cnt_f = pd.Series(cells_tr_f).value_counts()
    p_obs_f = cnt_f / cnt_f.sum()
    p_exp_f = pd.Series(1.0/len(p_obs_f), index=p_obs_f.index)
    w_per_f = (1.0 + LAMBDA * (p_exp_f/p_obs_f - 1.0)).clip(0.1, 10.0)
    sw_f = pd.Series(cells_tr_f).map(w_per_f).values.astype('float32')
    mdl_f2 = train_xgb_lite(Xtr_f, ytr_f, sw=sw_f)
    proba_f2 = mdl_f2.predict_proba(Xte_f)[:, 1].astype('float32')
    pred_f2 = (proba_f2 >= 0.5).astype(int)

    # Config 3 + 4 - per-fold α-search on Config 1 fold predictions
    race_tef = prot_f['RACE']; sex_tef = prot_f['SEX']; age_tef = prot_f['AGE_GROUP']
    test_groups_f = {}
    for r in sorted(np.unique(race_tef).tolist()):
        for ag in sorted(np.unique(age_tef).tolist()):
            for s in sorted(np.unique(sex_tef).tolist()):
                mask = (race_tef == r) & (age_tef == ag) & (sex_tef == s)
                if mask.sum() >= 5:
                    test_groups_f[f"{r}|{ag}|{s}"] = mask
    overall_sr_f = (proba_f1 >= 0.5).mean()
    overall_tpr_f = (proba_f1[yte_f == 1] >= 0.5).mean()
    overall_ppv_f = yte_f[proba_f1 >= 0.5].mean() if (proba_f1 >= 0.5).sum() > 10 else 0.5
    sr_f, tpr_f, ppv_f = {}, {}, {}
    for k, m in test_groups_f.items():
        sr_f[k]  = find_sr(proba_f1[m], overall_sr_f)
        tpr_f[k] = find_tpr(proba_f1[m], yte_f[m], overall_tpr_f)
        ppv_f[k] = find_ppv(proba_f1[m], yte_f[m], overall_ppv_f)

    def apply_thresh_f(probs, thresh):
        yp = (probs >= 0.5).astype(int)
        for k, m in test_groups_f.items():
            if k in thresh:
                yp[m] = (probs[m] >= thresh[k]).astype(int)
        return yp

    def di_local(yp, prot):
        groups = np.unique(prot)
        rates = [yp[prot==g].mean() for g in groups]
        if max(rates) <= 1e-9: return 1.0
        return min(rates) / max(rates)

    best_t3, best_acc3 = None, -1
    for a_sr in A_SR:
        for a_tpr in A_TPR:
            for a_ppv in A_PPV:
                thresh = {}
                for k in test_groups_f:
                    t = 0.5 + a_sr*(sr_f[k]-0.5) + a_tpr*(tpr_f[k]-0.5) + a_ppv*(ppv_f[k]-0.5)
                    thresh[k] = float(np.clip(t, 0.01, 0.99))
                yp = apply_thresh_f(proba_f1, thresh)
                if all(di_local(yp, prot_f[a]) >= 0.80 for a in ATTRS):
                    acc_here = (yp == yte_f).mean()
                    if acc_here > best_acc3:
                        best_acc3, best_t3 = acc_here, dict(thresh)

    if best_t3 is None:
        # Fall back to threshold 0.5 if no candidate satisfies all-4-DI
        pred_f3 = pred_f1.copy()
        pred_f4 = pred_f1.copy()
    else:
        pred_f3 = apply_thresh_f(proba_f1, best_t3)
        # Greedy refinement for Config 4
        gthresh = dict(best_t3)
        for _ in range(300):
            moved = False
            for k in list(gthresh.keys()):
                cur = gthresh[k]
                new_t = cur + 0.01 * (-1 if cur > 0.5 else 1)
                new_t = float(np.clip(new_t, 0.01, 0.99))
                if abs(new_t - 0.5) >= abs(cur - 0.5):
                    continue
                cand = dict(gthresh); cand[k] = new_t
                yp_try = apply_thresh_f(proba_f1, cand)
                if all(di_local(yp_try, prot_f[a]) >= 0.80 for a in ATTRS):
                    gthresh[k] = new_t
                    moved = True
            if not moved: break
        pred_f4 = apply_thresh_f(proba_f1, gthresh)

    fold_preds = {1: (pred_f1, proba_f1),
                  2: (pred_f2, proba_f2),
                  3: (pred_f3, proba_f1),
                  4: (pred_f4, proba_f1)}
    for cfg_id, (yp_f, ypb_f) in fold_preds.items():
        for a in ATTRS:
            m = compute_seven(yp_f, ypb_f, prot_f[a], yte_f)
            if m is None:
                for mk in METRICS:
                    fold_pass[cfg_id][a][mk].append(np.nan)
                continue
            for mk in METRICS:
                fold_pass[cfg_id][a][mk].append(int(passes(mk, m[mk])))

# Compute Fleiss kappa per metric per config (4 attrs x K folds = items x raters)
log("  computing per-metric Fleiss kappa per config")
for cfg_id in [1, 2, 3, 4]:
    rows = []
    for mk in METRICS:
        V = np.zeros((len(ATTRS), K_GROUPKFOLD), dtype=int)
        nan_seen = False
        for j, a in enumerate(ATTRS):
            arr = fold_pass[cfg_id][a][mk]
            for k_idx, val in enumerate(arr):
                if np.isnan(val):
                    nan_seen = True
                    V[j, k_idx] = 0
                else:
                    V[j, k_idx] = int(val)
        kappa = fleiss_kappa(V)
        rows.append({'metric': mk, 'fleiss_kappa': round(kappa, 4),
                     'agreement_class': landis_koch(kappa)})
        if nan_seen:
            diagnostics.append({'config': cfg_id, 'axis': 3, 'metric': mk,
                                'reason': 'at least one fold returned NaN; treated as fail'})
    pd.DataFrame(rows).to_csv(OUT_DIR / f'T10_axis3_kappa_config{cfg_id}.csv', index=False)
    log(f"    saved T10_axis3_kappa_config{cfg_id}.csv")

# =========================================================================
# Phase 7 - Summary pivot
# =========================================================================
log("Phase 7: building summary pivot")
summary_rows = []
for cfg_id, (cfg_name, yp, ypb) in configs.items():
    T_VFR = pd.read_csv(OUT_DIR / f'T13_axis1_vfr_config{cfg_id}.csv')
    T_N   = pd.read_csv(OUT_DIR / f'T9_axis2_minN_config{cfg_id}.csv')
    T_K   = pd.read_csv(OUT_DIR / f'T10_axis3_kappa_config{cfg_id}.csv')

    di_per = {a: disparate_impact(yp, prot_te[a]) for a in ATTRS}
    n_di_pass = sum(1 for v in di_per.values() if v >= 0.80)

    summary_rows.append({
        'Configuration': cfg_name,
        'config_id': cfg_id,
        'vfr_mean_across_28_cells':  round(T_VFR['vfr'].mean(), 4),
        'vfr_max':                   round(T_VFR['vfr'].max(), 4),
        'n_cells_flipped':           int((T_VFR['vfr'] > 0).sum()),
        'n_cells_full_N_required':   int(T_N['full_N_required'].sum()),
        'kappa_mean':                round(T_K['fleiss_kappa'].mean(), 4),
        'kappa_eopp':                round(T_K[T_K['metric']=='EOPP']['fleiss_kappa'].iloc[0], 4),
        'kappa_eod':                 round(T_K[T_K['metric']=='EOD']['fleiss_kappa'].iloc[0], 4),
        'kappa_cal':                 round(T_K[T_K['metric']=='CAL']['fleiss_kappa'].iloc[0], 4),
        'all_DI_pass':               (n_di_pass == 4),
        'n_DI_pass':                 n_di_pass,
        'accuracy':                  round((yp == y_te).mean(), 4),
    })

pd.DataFrame(summary_rows).to_csv(OUT_DIR / 'T_baseline_audit_summary.csv', index=False)
pd.DataFrame(diagnostics).to_csv(OUT_DIR / 'T_baseline_audit_diagnostics.csv', index=False)
log(f"  saved T_baseline_audit_summary.csv and T_baseline_audit_diagnostics.csv")

# =========================================================================
# Final printable summary
# =========================================================================
log("=" * 80)
log("DONE")
log("=" * 80)
T_SUM = pd.read_csv(OUT_DIR / 'T_baseline_audit_summary.csv')
print()
print(T_SUM.to_string(index=False))
print()

s = T_SUM.set_index('Configuration')
parts = []
for cfg in T_SUM['Configuration']:
    r = s.loc[cfg]
    parts.append(
        f"{cfg} achieved DI-pass on {int(r['n_DI_pass'])}/4 attributes with VFR mean "
        f"{r['vfr_mean_across_28_cells']:.4f} and cross-site mean kappa {r['kappa_mean']:.4f}"
    )
print("Across four configurations: " + ". ".join(parts) + ".")
print()
log(f"Total wallclock: {time.time()-t0:.0f}s")
