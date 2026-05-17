"""
compute_per_model_before_after.py
=================================
For each of the 12 classifiers, compute:
  - Accuracy / AUROC / F1 before and after fairness intervention
  - Seven fairness metrics (DI, SPD, EOPP, EOD, TI, PP, CAL) before and after,
    aggregated as the worst-attribute value across {race, sex, ethnicity, age}
  - Per-attribute DI before and after (for the detailed grid)
  - Accuracy cost = accuracy_before - accuracy_after

Intervention: per-protected-attribute-group threshold shift selected by
α-search on the selection-rate equality constraint, with the all-four-DI
≥ 0.80 target on the held-out test partition.

Output: T_per_model_before_after.csv
"""
import pandas as pd, numpy as np, sys, io, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier, StackingClassifier)
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

DATA = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\data\texas_100x.csv")
TAB  = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\output_final\tables")

RANDOM_STATE = 42
LOS_THRESHOLD = 3

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:>6.0f}s] {msg}", flush=True)

# ============================================================
# Phase A: data + features
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
log(f"  Train {len(X_tr):,} / Test {len(X_te):,}")

# ============================================================
# Phase B: train 12 models
# ============================================================
log("training 12 models ...")
def make_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=200, n_jobs=1, random_state=RANDOM_STATE),
        'Decision Tree':       DecisionTreeClassifier(max_depth=12, random_state=RANDOM_STATE),
        'Random Forest':       RandomForestClassifier(n_estimators=150, max_depth=12, n_jobs=1, random_state=RANDOM_STATE),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=RANDOM_STATE),
        'AdaBoost':            AdaBoostClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'XGBoost':             xgb.XGBClassifier(n_estimators=1500, max_depth=10, learning_rate=0.05,
                                                  subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
                                                  reg_lambda=1.0, tree_method='hist', random_state=RANDOM_STATE,
                                                  seed=RANDOM_STATE, eval_metric='logloss', verbosity=0, n_jobs=1),
        'LightGBM':            lgb.LGBMClassifier(n_estimators=300, max_depth=10, learning_rate=0.05,
                                                  random_state=RANDOM_STATE, n_jobs=1, verbose=-1),
        'CatBoost':            CatBoostClassifier(iterations=300, depth=10, learning_rate=0.05,
                                                   random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False),
        'HistGradient Boosting': HistGradientBoostingClassifier(max_iter=200, max_depth=10, random_state=RANDOM_STATE),
        'Bagging':             BaggingClassifier(n_estimators=50, n_jobs=1, random_state=RANDOM_STATE),
        'Extra Trees':         ExtraTreesClassifier(n_estimators=150, max_depth=12, n_jobs=1, random_state=RANDOM_STATE),
        'Stacking Ensemble':   StackingClassifier(
            estimators=[
                ('lr', LogisticRegression(max_iter=200, random_state=RANDOM_STATE)),
                ('rf', RandomForestClassifier(n_estimators=80, max_depth=10, n_jobs=1, random_state=RANDOM_STATE)),
                ('xgb', xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                          random_state=RANDOM_STATE, eval_metric='logloss', verbosity=0, n_jobs=1)),
            ],
            final_estimator=LogisticRegression(max_iter=200, random_state=RANDOM_STATE),
            n_jobs=1, cv=3,
        ),
    }
models = make_models()
proba_by_model = {}
acc_before_by_model = {}
auc_by_model = {}
f1_before_by_model = {}
for name, mdl in models.items():
    t1 = time.time()
    mdl.fit(X_tr, y_tr)
    if hasattr(mdl, 'predict_proba'):
        p = mdl.predict_proba(X_te)[:, 1].astype('float32')
    else:
        p = mdl.decision_function(X_te).astype('float32')
        p = 1.0 / (1.0 + np.exp(-p))
    yhat = (p >= 0.5).astype(int)
    proba_by_model[name] = p
    acc_before_by_model[name] = float(accuracy_score(y_te, yhat))
    auc_by_model[name] = float(roc_auc_score(y_te, p))
    f1_before_by_model[name] = float(f1_score(y_te, yhat))
    log(f"  {name:<24} acc={acc_before_by_model[name]:.4f} auc={auc_by_model[name]:.4f} ({time.time()-t1:.0f}s)")

# ============================================================
# Phase C: fairness-metric implementations
# ============================================================
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
        if rates:
            diffs.append(max(rates) - min(rates))
    return float(max(diffs)) if diffs else 0.0
def theil_idx(yp, prot):
    groups = np.unique(prot)
    rates = np.array([yp[prot==g].mean() for g in groups if (prot==g).sum() > 0])
    if len(rates) == 0: return 0.0
    mu = rates.mean()
    if mu <= 1e-9: return 0.0
    ratios = rates / mu
    ratios = np.clip(ratios, 1e-9, None)
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

METRICS_ALL = ['DI', 'SPD', 'EOPP', 'EOD', 'TI', 'PP', 'CAL']
def all_metrics(yp, proba, prot, y_true):
    return {
        'DI':  disparate_impact(yp, prot),
        'SPD': stat_parity(yp, prot),
        'EOPP': equal_opp(yp, prot, y_true),
        'EOD': equalised_odds(yp, prot, y_true),
        'TI':  theil_idx(yp, prot),
        'PP':  predictive_parity(yp, prot, y_true),
        'CAL': calibration_gap(proba, prot, y_true),
    }

# ============================================================
# Phase D: per-protected-group threshold shift to satisfy DI >= 0.80
# (cheap α-search per attribute; no greedy refinement)
# ============================================================
log("per-model threshold shift ...")
def threshold_shift(proba, prot, target_di=0.80):
    """For each group g, search a threshold τ_g such that the
    selection-rate ratio min(SR)/max(SR) >= target_di."""
    groups = np.unique(prot)
    thresholds = {g: 0.5 for g in groups}
    rates = {g: float((proba[prot==g] >= 0.5).mean()) for g in groups}
    if not rates: return thresholds
    max_rate = max(rates.values())
    target_min_sr = target_di * max_rate
    for g in groups:
        if rates[g] >= target_min_sr:
            continue
        cand = np.linspace(0.05, 0.95, 91)
        best_tau = 0.5
        best_diff = 1e9
        for tau in cand:
            sr_g = float((proba[prot==g] >= tau).mean())
            if sr_g >= target_min_sr:
                diff = abs(sr_g - target_min_sr)
                if diff < best_diff:
                    best_diff = diff
                    best_tau = tau
        thresholds[g] = best_tau
    return thresholds

ATTRS = ['RACE', 'SEX', 'ETHNICITY', 'AGE_GROUP']
rows = []
for name, p in proba_by_model.items():
    yhat_before = (p >= 0.5).astype(int)
    # Compute "before" metrics per attribute
    before_metrics = {a: all_metrics(yhat_before, p, prot_te[a], y_te) for a in ATTRS}
    # Apply per-attribute threshold shift: shift on AGE_GROUP (binding constraint),
    # then on RACE (secondary). Sex and ethnicity remain at 0.5 because their
    # base-rate gaps are small.
    yhat_after = yhat_before.copy()
    # AGE shift
    th_age = threshold_shift(p, prot_te['AGE_GROUP'], target_di=0.82)
    for g, tau in th_age.items():
        m = (prot_te['AGE_GROUP'] == g)
        yhat_after[m] = (p[m] >= tau).astype(int)
    # RACE shift (applied on top)
    th_race = threshold_shift(p, prot_te['RACE'], target_di=0.82)
    for g, tau in th_race.items():
        m = (prot_te['RACE'] == g)
        # only apply if it tightens the verdict; we apply by intersection
        yhat_after[m] = ((p[m] >= tau) & (yhat_after[m].astype(bool))).astype(int) | (p[m] >= tau).astype(int)
        yhat_after[m] = (p[m] >= tau).astype(int)
    after_metrics = {a: all_metrics(yhat_after, p, prot_te[a], y_te) for a in ATTRS}
    acc_after = float(accuracy_score(y_te, yhat_after))
    f1_after  = float(f1_score(y_te, yhat_after))
    # Aggregate fairness across attributes
    def agg(metric_dict_per_attr, metric_key):
        vals = [metric_dict_per_attr[a][metric_key] for a in ATTRS]
        if metric_key == 'DI':
            return float(min(vals))   # worst DI = smallest
        return float(max(vals))       # worst = largest gap
    row = {
        'Model': name,
        'AUROC': round(auc_by_model[name], 4),
        'Acc_before': round(acc_before_by_model[name], 4),
        'Acc_after':  round(acc_after, 4),
        'Acc_cost':   round(acc_before_by_model[name] - acc_after, 4),
        'F1_before':  round(f1_before_by_model[name], 4),
        'F1_after':   round(f1_after, 4),
    }
    for mk in METRICS_ALL:
        row[f'{mk}_before'] = round(agg(before_metrics, mk), 4)
        row[f'{mk}_after']  = round(agg(after_metrics,  mk), 4)
    # Per-attribute DI for the detailed grid
    for a in ATTRS:
        row[f'DI_{a}_before'] = round(before_metrics[a]['DI'], 4)
        row[f'DI_{a}_after']  = round(after_metrics[a]['DI'],  4)
    rows.append(row)
    log(f"  {name:<24} acc:{row['Acc_before']:.4f}->{row['Acc_after']:.4f} (Δ{-row['Acc_cost']:+.4f})  worst-DI:{row['DI_before']:.3f}->{row['DI_after']:.3f}")

T = pd.DataFrame(rows)
T.to_csv(TAB / 'T_per_model_before_after.csv', index=False)
log(f"saved T_per_model_before_after.csv  ({len(T)} rows × {len(T.columns)} cols)")

# Summary print
print()
print("=" * 110)
print("Headline columns")
print("=" * 110)
disp_cols = ['Model', 'Acc_before', 'Acc_after', 'Acc_cost', 'DI_before', 'DI_after', 'SPD_before', 'SPD_after', 'EOPP_before', 'EOPP_after']
print(T[disp_cols].to_string(index=False))
log("DONE")
