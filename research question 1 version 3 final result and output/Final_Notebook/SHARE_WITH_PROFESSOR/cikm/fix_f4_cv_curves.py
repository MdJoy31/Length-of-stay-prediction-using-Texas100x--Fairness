"""
Regenerate F4 only, fixing the degenerate CV → 0 collapse at N=185,026
(subsampling without replacement at the full test size → identical sample).
Use bootstrap-with-replacement at every N so CV is well-defined throughout.
"""
import pandas as pd, numpy as np, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb

ROOT = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm")
DATA = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\data\texas_100x.csv")
FIG_DIR = ROOT / "output_final" / "figures" / "manuscript"

RANDOM_STATE = 42
LOS_THRESHOLD = 3
ATTRS = ['RACE', 'SEX', 'ETHNICITY', 'AGE_GROUP']
METRICS = ['DI', 'SPD', 'EOPP', 'EOD', 'TI', 'PP', 'CAL']
N_GRID = [1000, 2000, 5000, 10_000, 25_000, 50_000, 100_000, 185_026]
N_REPS = 30

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:>5.0f}s] {msg}", flush=True)

log("loading + FE")
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
    cat_stats['mu_k'] = (cat_stats['n']*cat_stats['yk'] + m_smooth*y_global_mean)/(cat_stats['n']+m_smooth)
    df[f'{col}_te'] = df[col].map(cat_stats['mu_k']).fillna(y_global_mean).astype('float32')
KEEP = ["PAT_AGE","TOTAL_CHARGES","PAT_STATUS","TYPE_OF_ADMISSION","SOURCE_OF_ADMISSION"]
TE_COLS = ["ADMITTING_DIAGNOSIS_te","PRINC_SURG_PROC_CODE_te","THCIC_ID_te"]
df['AGE_X_DIAG_TE'] = (df['PAT_AGE'].astype('float32')*df['ADMITTING_DIAGNOSIS_te']).astype('float32')
df['ADMIT_X_SOURCE'] = (df['TYPE_OF_ADMISSION'].astype('float32')*10.0 + df['SOURCE_OF_ADMISSION'].astype('float32')).astype('float32')
hosp_vol = df.groupby('THCIC_ID').size()
df['HOSP_VOLUME_LOG'] = np.log1p(df['THCIC_ID'].map(hosp_vol).fillna(0)).astype('float32')
INTER = ["AGE_X_DIAG_TE","ADMIT_X_SOURCE","HOSP_VOLUME_LOG"]
feature_cols = KEEP + TE_COLS + INTER
X_full = df[feature_cols].fillna(0).astype('float32').values
X_tr, X_te = X_full[idx_tr], X_full[idx_te]
y_tr, y_te = y_full[idx_tr], y_full[idx_te]
prot_te = {'RACE': df['RACE'].values[idx_te], 'SEX': df['SEX_CODE'].values[idx_te],
           'ETHNICITY': df['ETHNICITY'].values[idx_te], 'AGE_GROUP': df['AGE_GROUP'].values[idx_te]}

log("training XGBoost (n_est=300)")
mdl = xgb.XGBClassifier(n_estimators=300, max_depth=10, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.85, min_child_weight=3, reg_lambda=1.0,
    tree_method='hist', random_state=RANDOM_STATE, seed=RANDOM_STATE,
    eval_metric='logloss', verbosity=0, n_jobs=1)
mdl.fit(X_tr, y_tr)
proba = mdl.predict_proba(X_te)[:, 1].astype('float32')
pred = (proba >= 0.5).astype(int)
log(f"  Acc={(pred==y_te).mean():.4f}")

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
    sr_v = [r[0] for r in rates.values()]; tpr_v = [r[1] for r in rates.values()]
    fpr_v = [r[2] for r in rates.values()]; ppv_v = [r[3] for r in rates.values()]
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
                ti += (m.sum()/n_total) * (mu_g/mu_all) * np.log(mu_g/mu_all)
        ti = float(abs(ti))
    cal_diffs = []
    if ypb is not None:
        for g in groups:
            m = prot == g
            pg = ypb[m]; yg = y_true[m]
            for b in range(10):
                in_bin = (pg >= b/10) & (pg < (b+1)/10)
                if in_bin.sum() > 5:
                    cal_diffs.append(abs(pg[in_bin].mean() - yg[in_bin].mean()))
    cal = max(cal_diffs) if cal_diffs else 0.0
    return {'DI':di, 'SPD':spd, 'EOPP':eopp, 'EOD':eod, 'TI':ti, 'PP':pp, 'CAL':cal}

log("F4 CV grid (with-replacement bootstrap throughout)")
cv_table = {(m, a): [] for m in METRICS for a in ATTRS}
rng = np.random.default_rng(RANDOM_STATE)
n_test = len(y_te)
for N_use in N_GRID:
    log(f"  N={N_use}")
    for a in ATTRS:
        for m in METRICS:
            vals = []
            for r in range(N_REPS):
                # WITH-replacement bootstrap (so CV is well-defined even at full N)
                ix = rng.choice(n_test, N_use, replace=True)
                metrics = compute_seven(pred[ix], proba[ix], prot_te[a][ix], y_te[ix])
                if metrics: vals.append(metrics[m])
            vals = np.array(vals)
            mean_v = abs(np.mean(vals)) if len(vals) else 1.0
            cv = (np.std(vals, ddof=1) / max(mean_v, 1e-9)) if len(vals) >= 5 else np.nan
            cv_table[(m, a)].append(cv)

# Plot F4
fig, ax = plt.subplots(figsize=(11, 6))
metric_colors = {'DI':"#1d4ed8", 'SPD':"#0891b2", 'EOPP':"#16a34a", 'EOD':"#84cc16",
                 'TI':"#a855f7", 'PP':"#f59e0b", 'CAL':"#dc2626"}
attr_styles = {'RACE':'-', 'SEX':'--', 'ETHNICITY':':', 'AGE_GROUP':'-.'}
for m in METRICS:
    for a in ATTRS:
        ys = np.array(cv_table[(m, a)])
        ax.plot(N_GRID, ys, attr_styles[a], color=metric_colors[m], lw=1.6, alpha=0.85)
ax.axhline(0.05, color="black", ls="--", lw=1.5)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(1e-3, 5)
ax.set_xlabel("audit cohort N (log)"); ax.set_ylabel("coefficient of variation (log)")
ax.set_title("F4 · CV vs audit size (28 metric × attribute cells, with-replacement bootstrap)",
             fontsize=12, fontweight="bold", loc="left")
from matplotlib.lines import Line2D
metric_handles = [Line2D([0],[0], color=metric_colors[m], lw=2.5, label=m) for m in METRICS]
attr_handles = [Line2D([0],[0], color="#475569", ls=attr_styles[a], lw=2, label=a[:4])
                for a in ATTRS]
metric_handles.append(Line2D([0],[0], color="black", ls="--", lw=1.5, label="CV = 0.05"))
leg1 = ax.legend(handles=metric_handles, fontsize=8, ncol=4, loc="upper right", title="Metric")
ax.add_artist(leg1)
ax.legend(handles=attr_handles, fontsize=8, ncol=4, loc="lower left", title="Attribute")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "F4_cv_curves.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
log(f"saved F4 ({(FIG_DIR / 'F4_cv_curves.png').stat().st_size/1024:.0f} KB)")
