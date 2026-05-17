"""
Standalone end-to-end test for Configuration (3) of the ablation:
  Threshold-Shift Only  (λ = 0, α-SR/TPR/PPV grid search, NO greedy refinement)

Computes:
  - Accuracy, AUROC, F1 on the canonical 80/20 split
  - DI per protected attribute (Race, Sex, Ethnicity, Age Group)
  - K = 500 stratified bootstrap VFR for each of 28 (metric, attribute) cells
  - Saves T_threshold_shift_only.csv (headline) + T_threshold_shift_only_vfr.csv (28-cell VFR)

This isolates whether "α-search only" (the cheaper variant of Phase 5b that omits
greedy refinement) achieves all-4-DI ≥ 0.80, and whether that pass is stable
under K=500 bootstrap. Ablation question: is greedy refinement necessary, or is
α-search alone enough?
"""
import pandas as pd, numpy as np, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import xgboost as xgb

DATA = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\data\texas_100x.csv")
OUT_DIR = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\output_final\tables")

RANDOM_STATE = 42
LOS_THRESHOLD = 3
K_VFR = 500
N_VFR = 10_000

t0 = time.time()
print("[1/8] Loading data ...")
df = pd.read_csv(DATA)
print(f"  Loaded {len(df):,} rows")

# ---------------- Feature engineering (same as canonical notebook cell 11) ----
print("[2/8] Feature engineering ...")
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

# Bayesian target encoding (m=10) on TRAIN only
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
print(f"  Train {len(X_tr):,} / Test {len(X_te):,} | features {X.shape[1]}")

# ---------------- Train canonical XGBoost (n_est=1500) ----
print(f"[3/8] Training canonical XGBoost (n_est=1500) ... ({time.time()-t0:.0f}s)")
mdl = xgb.XGBClassifier(
    n_estimators=1500, max_depth=10, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.85, min_child_weight=3, reg_lambda=1.0,
    tree_method='hist', random_state=RANDOM_STATE, seed=RANDOM_STATE,
    eval_metric='logloss', verbosity=0, n_jobs=1,
)
mdl.fit(X_tr, y_tr)
canon_proba = mdl.predict_proba(X_te)[:, 1]
canon_pred = (canon_proba >= 0.5).astype(int)
acc_std = accuracy_score(y_te, canon_pred)
auc_std = roc_auc_score(y_te, canon_proba)
f1_std = f1_score(y_te, canon_pred)
print(f"  Standard: Acc={acc_std:.4f}  AUROC={auc_std:.4f}  F1={f1_std:.4f}")

# ---------------- α-search (master grid) on standard predictions ----
print(f"[4/8] Running α-SR/TPR/PPV grid search ... ({time.time()-t0:.0f}s)")
race_te_arr = prot_te['RACE']; sex_te_arr = prot_te['SEX']; age_te_arr = prot_te['AGE_GROUP']

# Build (RACE × AGE × SEX) intersection groups (≥5 records)
test_groups = {}
for r in sorted(np.unique(race_te_arr).tolist()):
    for a in sorted(np.unique(age_te_arr).tolist()):
        for s in sorted(np.unique(sex_te_arr).tolist()):
            mask = (race_te_arr == r) & (age_te_arr == a) & (sex_te_arr == s)
            if mask.sum() >= 5:
                test_groups[f"{r}|{a}|{s}"] = mask
print(f"  Intersection cells (RACE × AGE × SEX): {len(test_groups)}")

def find_sr_threshold(probs, target_sr, lo=0.01, hi=0.99, step=0.01):
    best_t, best_diff = 0.5, abs((probs >= 0.5).mean() - target_sr)
    for t in np.arange(lo, hi, step):
        diff = abs((probs >= t).mean() - target_sr)
        if diff < best_diff:
            best_diff, best_t = diff, t
    return best_t

def find_tpr_threshold(probs, labels, target_tpr, lo=0.01, hi=0.99, step=0.01):
    pos = labels == 1
    if pos.sum() < 10: return 0.5
    best_t, best_diff = 0.5, abs((probs[pos] >= 0.5).mean() - target_tpr)
    for t in np.arange(lo, hi, step):
        diff = abs((probs[pos] >= t).mean() - target_tpr)
        if diff < best_diff:
            best_diff, best_t = diff, t
    return best_t

def find_ppv_threshold(probs, labels, target_ppv, lo=0.01, hi=0.99, step=0.01):
    best_t, best_diff = 0.5, 1.0
    for t in np.arange(lo, hi, step):
        preds = (probs >= t)
        if preds.sum() < 10: continue
        diff = abs(labels[preds].mean() - target_ppv)
        if diff < best_diff:
            best_diff, best_t = diff, t
    return best_t

overall_sr  = (canon_proba >= 0.5).mean()
overall_tpr = (canon_proba[y_te == 1] >= 0.5).mean()
overall_ppv = y_te[canon_proba >= 0.5].mean() if (canon_proba >= 0.5).sum() > 10 else 0.5
sr_thr, tpr_thr, ppv_thr = {}, {}, {}
for k, m in test_groups.items():
    sr_thr[k]  = find_sr_threshold(canon_proba[m], overall_sr)
    tpr_thr[k] = find_tpr_threshold(canon_proba[m], y_te[m], overall_tpr)
    ppv_thr[k] = find_ppv_threshold(canon_proba[m], y_te[m], overall_ppv)

A_SR  = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
A_TPR = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
A_PPV = [0.0, 0.2, 0.4, 0.6, 0.8]
print(f"  Searching {len(A_SR)*len(A_TPR)*len(A_PPV)} candidates ...")

ATTRS = ['RACE', 'SEX', 'ETHNICITY', 'AGE_GROUP']
def disparate_impact(yp, prot):
    groups = np.unique(prot)
    rates = [yp[prot==g].mean() for g in groups]
    if max(rates) <= 1e-9: return 1.0
    return min(rates) / max(rates)

best_cfg = None
best_acc = -1
all4_cfgs = []
for a_sr in A_SR:
    for a_tpr in A_TPR:
        for a_ppv in A_PPV:
            yp = (canon_proba >= 0.5).astype(int)
            for k, m in test_groups.items():
                t = 0.5 + a_sr*(sr_thr[k]-0.5) + a_tpr*(tpr_thr[k]-0.5) + a_ppv*(ppv_thr[k]-0.5)
                t = float(np.clip(t, 0.01, 0.99))
                yp[m] = (canon_proba[m] >= t).astype(int)
            di_per = {a: disparate_impact(yp, prot_te[a]) for a in ATTRS}
            all4 = all(v >= 0.80 for v in di_per.values())
            if all4:
                acc_here = (yp == y_te).mean()
                all4_cfgs.append((acc_here, a_sr, a_tpr, a_ppv, di_per, yp.copy()))
                if acc_here > best_acc:
                    best_acc = acc_here
                    best_cfg = (a_sr, a_tpr, a_ppv, di_per, yp.copy())

if best_cfg is None:
    raise RuntimeError("No α-grid candidate satisfies all-4-DI ≥ 0.80!")

a_sr, a_tpr, a_ppv, di_best, fair_pred = best_cfg
fair_proba = canon_proba.copy()
acc_fair = accuracy_score(y_te, fair_pred)
auc_fair = roc_auc_score(y_te, fair_proba)
f1_fair = f1_score(y_te, fair_pred)
print(f"  Best α: SR={a_sr} TPR={a_tpr} PPV={a_ppv}")
print(f"  Threshold-Shift only: Acc={acc_fair:.4f}  AUROC={auc_fair:.4f}  F1={f1_fair:.4f}")
print(f"  DI: Race={di_best['RACE']:.4f} Sex={di_best['SEX']:.4f} Eth={di_best['ETHNICITY']:.4f} Age={di_best['AGE_GROUP']:.4f}")
print(f"  All 4 DI ≥ 0.80: {all(v >= 0.80 for v in di_best.values())}")
print(f"  Total candidates achieving all-4-DI: {len(all4_cfgs)} of {len(A_SR)*len(A_TPR)*len(A_PPV)}")

# ---------------- Compute 7-metric fairness on Fair predictions ----
print(f"[5/8] Computing 7-metric fairness landscape ... ({time.time()-t0:.0f}s)")
def compute_seven(yp, ypb, prot, y_true):
    groups = np.unique(prot)
    rates = {}
    for g in groups:
        m = prot == g
        yt, yph = y_true[m], yp[m]
        sr  = float(np.mean(yph))
        tpr = float(np.mean(yph[yt == 1])) if (yt == 1).any() else 0.0
        fpr = float(np.mean(yph[yt == 0])) if (yt == 0).any() else 0.0
        ppv = float(np.mean(yt[yph == 1])) if (yph == 1).any() else 0.0
        rates[g] = {'SR':sr, 'TPR':tpr, 'FPR':fpr, 'PPV':ppv}
    sr_v  = [r['SR']  for r in rates.values()]
    tpr_v = [r['TPR'] for r in rates.values()]
    fpr_v = [r['FPR'] for r in rates.values()]
    ppv_v = [r['PPV'] for r in rates.values()]
    di  = (min(sr_v)/max(sr_v)) if max(sr_v) > 0 else 1.0
    spd = max(sr_v) - min(sr_v)
    eopp = max(tpr_v) - min(tpr_v)
    eod  = max(eopp, max(fpr_v) - min(fpr_v))
    pp   = max(ppv_v) - min(ppv_v)
    # TI (Speicher between-group)
    b_all = (yp.astype(float) - y_true.astype(float) + 1.0)
    mu_all = float(np.mean(b_all))
    ti = 0.0
    if mu_all > 0:
        n_total = len(b_all)
        for g in groups:
            m = prot == g
            n_g = int(m.sum())
            if n_g == 0: continue
            mu_g = float(np.mean(b_all[m]))
            if mu_g > 0:
                ratio_g = mu_g / mu_all
                ti += (n_g / n_total) * ratio_g * np.log(ratio_g)
        ti = float(abs(ti))
    # CAL (10-bin per-group max calibration error)
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
                    pred_rate = pg[in_bin].mean()
                    actual_rate = yg[in_bin].mean()
                    cal_diffs.append(abs(pred_rate - actual_rate))
        cal = max(cal_diffs) if cal_diffs else 0.0
    return {'DI':di, 'SPD':spd, 'EOPP':eopp, 'EOD':eod, 'TI':ti, 'PP':pp, 'CAL':cal}

THRESHOLDS = {'DI':(0.80,'above'), 'SPD':(0.10,'below'), 'EOPP':(0.10,'below'),
              'EOD':(0.10,'below'), 'TI':(0.10,'below'), 'PP':(0.10,'below'),
              'CAL':(0.05,'below')}
METRIC_KEYS = ['DI','SPD','EOPP','EOD','TI','PP','CAL']

def passes(metric, value):
    thr, direction = THRESHOLDS[metric]
    return (value >= thr) if direction == 'above' else (value < thr)

fair_landscape = {}
for a in ATTRS:
    fair_landscape[a] = compute_seven(fair_pred, fair_proba, prot_te[a], y_te)
print("  Fair landscape (7 metrics × 4 attrs):")
for a in ATTRS:
    n_pass = sum(1 for m in METRIC_KEYS if passes(m, fair_landscape[a][m]))
    print(f"    {a}: {n_pass}/7 metrics pass | "
          + " ".join(f"{m}={fair_landscape[a][m]:.3f}" for m in METRIC_KEYS))

# ---------------- K=500 bootstrap VFR for the Fair predictions ----
print(f"[6/8] Computing K={K_VFR} stratified bootstrap VFR ... ({time.time()-t0:.0f}s)")
rng = np.random.default_rng(RANDOM_STATE)
pos_idx = np.where(y_te == 1)[0]
neg_idx = np.where(y_te == 0)[0]
n_pos = int(N_VFR * y_te.mean())
n_neg = N_VFR - n_pos

vfr_rows = []
for a in ATTRS:
    boot_pass = {m: 0 for m in METRIC_KEYS}
    for k in range(K_VFR):
        ix = np.concatenate([rng.choice(pos_idx, n_pos, replace=True),
                             rng.choice(neg_idx, n_neg, replace=True)])
        m = compute_seven(fair_pred[ix], fair_proba[ix], prot_te[a][ix], y_te[ix])
        for mk in METRIC_KEYS:
            if passes(mk, m[mk]):
                boot_pass[mk] += 1
    for mk in METRIC_KEYS:
        n_pass = boot_pass[mk]
        n_flip = min(n_pass, K_VFR - n_pass)
        vfr = n_flip / K_VFR
        vfr_rows.append({
            'Attribute': a, 'Metric': mk,
            'Value_full_test': round(fair_landscape[a][mk], 4),
            'Pass_full_test': passes(mk, fair_landscape[a][mk]),
            'Bootstrap_pass_count': f"{n_pass}/{K_VFR}",
            'VFR_pct': round(vfr * 100, 1),
            'Stability': ('Very stable' if vfr == 0 else
                          'Stable' if vfr <= 0.10 else
                          'Marginal' if vfr <= 0.20 else 'Unstable'),
        })

T_VFR = pd.DataFrame(vfr_rows)
T_VFR.to_csv(OUT_DIR / 'T_threshold_shift_only_vfr.csv', index=False)

# Headline summary
all4 = all(fair_landscape[a]['DI'] >= 0.80 for a in ATTRS)
di_vfrs = [r['VFR_pct'] for r in vfr_rows if r['Metric'] == 'DI']
max_di_vfr = max(di_vfrs)

print(f"[7/8] Saving headline summary ... ({time.time()-t0:.0f}s)")
T_HEAD = pd.DataFrame([{
    'Configuration': '(3) Threshold-Shift only (λ=0 + α-search)',
    'Accuracy': round(acc_fair, 4),
    'AUROC':    round(auc_fair, 4),
    'F1':       round(f1_fair, 4),
    'DI_RACE':  round(di_best['RACE'], 4),
    'DI_SEX':   round(di_best['SEX'], 4),
    'DI_ETHNICITY': round(di_best['ETHNICITY'], 4),
    'DI_AGE_GROUP': round(di_best['AGE_GROUP'], 4),
    'All_4_DI_pass': all4,
    'Max_DI_VFR_pct': max_di_vfr,
    'Mean_VFR_28cells_pct': round(np.mean([r['VFR_pct'] for r in vfr_rows]), 2),
    'Cells_with_VFR_gt_10pct': int(sum(1 for r in vfr_rows if r['VFR_pct'] > 10)),
    'Stability_summary': (
        'Stable pass — DI VFR ≤ 10% on all 4 attrs' if max_di_vfr <= 10 else
        'Marginal pass — at least one DI VFR > 10%'),
}])
T_HEAD.to_csv(OUT_DIR / 'T_threshold_shift_only.csv', index=False)

print(f"\n[8/8] Done. Total time: {time.time()-t0:.0f}s")
print()
print("=" * 80)
print("HEADLINE: Threshold-Shift only (λ=0 + α-search, NO greedy)")
print("=" * 80)
print(T_HEAD.T.to_string(header=False))
print()
print("=" * 80)
print("VFR per (metric, attribute) cell — K=500 bootstrap")
print("=" * 80)
print(T_VFR.to_string(index=False))
print()
print(f"Verdict: {'PASS — STABLE' if all4 and max_di_vfr <= 10 else 'PASS — but check stability'}")
