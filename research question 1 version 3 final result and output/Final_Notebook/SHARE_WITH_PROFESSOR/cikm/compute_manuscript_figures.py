"""
Generates the four manuscript figures (F2, F3, F4, F5) referenced in main_cikm.pdf
and saves PNGs to output_final/figures/manuscript/.

  F2_cohort_distribution.png  - 4-panel cohort composition (race, sex×eth, age, hospitals)
  F3_vfr_heatmap.png          - 7×4 VFR heatmap on Config 4 (canonical Phase 5b)
  F4_cv_curves.png            - 28 CV-vs-N curves on standard XGBoost predictions
  F5_hospital_violin.png      - 7-metric violin across 20 GroupKFold hospital folds
"""
import pandas as pd, numpy as np, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupKFold
import xgboost as xgb

ROOT = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm")
DATA = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\data\texas_100x.csv")
TAB = ROOT / "output_final" / "tables"
FIG_DIR = ROOT / "output_final" / "figures" / "manuscript"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
LOS_THRESHOLD = 3
ATTRS = ['RACE', 'SEX', 'ETHNICITY', 'AGE_GROUP']
METRICS = ['DI', 'SPD', 'EOPP', 'EOD', 'TI', 'PP', 'CAL']
N_GRID = [1000, 2000, 5000, 10_000, 25_000, 50_000, 100_000, 185_026]
N_REPS = 30
K_FOLD = 20

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:>5.0f}s] {msg}", flush=True)

# =========================================================================
# Phase 1 - load + FE (canonical pipeline)
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
KEEP = ["PAT_AGE","TOTAL_CHARGES","PAT_STATUS","TYPE_OF_ADMISSION","SOURCE_OF_ADMISSION"]
TE_COLS = ["ADMITTING_DIAGNOSIS_te","PRINC_SURG_PROC_CODE_te","THCIC_ID_te"]
df['AGE_X_DIAG_TE'] = (df['PAT_AGE'].astype('float32') * df['ADMITTING_DIAGNOSIS_te']).astype('float32')
df['ADMIT_X_SOURCE'] = (df['TYPE_OF_ADMISSION'].astype('float32') * 10.0 + df['SOURCE_OF_ADMISSION'].astype('float32')).astype('float32')
hosp_vol = df.groupby('THCIC_ID').size()
df['HOSP_VOLUME_LOG'] = np.log1p(df['THCIC_ID'].map(hosp_vol).fillna(0)).astype('float32')
INTER = ["AGE_X_DIAG_TE","ADMIT_X_SOURCE","HOSP_VOLUME_LOG"]
feature_cols = KEEP + TE_COLS + INTER
X_full = df[feature_cols].fillna(0).astype('float32').values
y_tr, y_te = y_full[idx_tr], y_full[idx_te]
X_tr, X_te = X_full[idx_tr], X_full[idx_te]
hosp_full = df['THCIC_ID'].values
prot_te = {
    'RACE': df['RACE'].values[idx_te],
    'SEX': df['SEX_CODE'].values[idx_te],
    'ETHNICITY': df['ETHNICITY'].values[idx_te],
    'AGE_GROUP': df['AGE_GROUP'].values[idx_te],
}
log(f"  Train {len(X_tr):,} / Test {len(X_te):,}")

# =========================================================================
# F2 - cohort distribution (4-panel, no model needed)
# =========================================================================
log("F2: cohort distribution")
fig, axes = plt.subplots(2, 2, figsize=(12, 6))
fig.suptitle("F2 · Texas-100X cohort composition", fontsize=12.5, fontweight="bold", y=1.02)

# Panel (a) - race code histogram (5 bars)
ax = axes[0, 0]
race_counts = df['RACE'].value_counts().sort_index()
race_labels = {0:"AI/AN", 1:"Asian/PI", 2:"Black", 3:"White", 4:"Other"}
ax.bar([race_labels[i] for i in race_counts.index], race_counts.values,
       color="#3b82f6", edgecolor="black")
ax.set_title("(a) Race code distribution", fontsize=10, fontweight="bold", loc="left")
ax.set_ylabel("records")
for i, v in enumerate(race_counts.values):
    ax.text(i, v + 5000, f"{v:,}", ha="center", fontsize=8)

# Panel (b) - sex × ethnicity stacked bar
ax = axes[0, 1]
ct = pd.crosstab(df['SEX_CODE'], df['ETHNICITY'])
ct.index = ['Female','Male']
ct.columns = ['Non-Hispanic','Hispanic']
ct.plot(kind='bar', stacked=True, ax=ax, color=['#94a3b8','#f59e0b'], edgecolor="black")
ax.set_title("(b) Sex × Ethnicity", fontsize=10, fontweight="bold", loc="left")
ax.set_xticklabels(['Female','Male'], rotation=0)
ax.set_xlabel(""); ax.set_ylabel("records")
ax.legend(fontsize=8, loc="upper left")

# Panel (c) - age group histogram with positive rate overlay
ax = axes[1, 0]
ag_counts = df['AGE_GROUP'].value_counts().sort_index()
ag_labels = {0:"Pediatric (<18)", 1:"Young (18-39)", 2:"Middle (40-64)", 3:"Elderly (≥65)"}
xs = np.arange(4)
bars = ax.bar(xs, ag_counts.values, color="#3b82f6", edgecolor="black", alpha=0.8)
ax.set_xticks(xs); ax.set_xticklabels([ag_labels[i] for i in range(4)], fontsize=8)
ax.set_ylabel("records", color="#1d4ed8")
ax.set_title("(c) Age group + positive rate (LOS > 3 d)", fontsize=10, fontweight="bold", loc="left")
ax2 = ax.twinx()
pos_rate = df.groupby('AGE_GROUP')['LOS_BINARY'].mean()
ax2.plot(xs, pos_rate.values, '-o', color="#dc2626", markersize=10, lw=2)
for i, p in enumerate(pos_rate.values):
    ax2.text(i, p + 0.03, f"{p*100:.1f}%", ha="center", fontsize=8, fontweight="bold", color="#dc2626")
ax2.set_ylabel("LOS > 3 d rate", color="#dc2626")
ax2.set_ylim(0, 0.85)

# Panel (d) - per-hospital record-count histogram
ax = axes[1, 1]
hosp_counts = df.groupby('THCIC_ID').size()
ax.hist(hosp_counts.values, bins=60, color="#3b82f6", edgecolor="black", alpha=0.8)
ax.set_yscale("log")
median_v = hosp_counts.median()
ax.axvline(median_v, color="#dc2626", ls="--", lw=2, label=f"median = {int(median_v):,}")
ax.set_xlabel("records per hospital"); ax.set_ylabel("hospitals (log)")
ax.set_title("(d) Per-hospital volume", fontsize=10, fontweight="bold", loc="left")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(FIG_DIR / "F2_cohort_distribution.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
log(f"  saved F2 ({(FIG_DIR / 'F2_cohort_distribution.png').stat().st_size/1024:.0f} KB)")

# =========================================================================
# F3 - VFR heatmap on Config 4 (canonical Phase 5b) using existing CSV
# =========================================================================
log("F3: VFR heatmap (Config 4)")
T_VFR4 = pd.read_csv(TAB / "T13_axis1_vfr_config4.csv")
mat = np.zeros((len(METRICS), len(ATTRS)))
verdict = np.full((len(METRICS), len(ATTRS)), "", dtype=object)
for i, m in enumerate(METRICS):
    for j, a in enumerate(ATTRS):
        row = T_VFR4[(T_VFR4['metric']==m) & (T_VFR4['attribute']==a)]
        if len(row) == 0:
            mat[i, j] = np.nan; verdict[i, j] = "-"
        else:
            mat[i, j] = row['vfr'].iloc[0]
            verdict[i, j] = "P" if row['verdict_dominant'].iloc[0] == 'fair' else "F"

fig, ax = plt.subplots(figsize=(7, 6))
fig.suptitle("F3 · Verdict-Flip-Rate heatmap (Real+VFR canonical, Config 4)",
             fontsize=12, fontweight="bold", y=1.00)
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("vfr",
    ["#16a34a", "#a3e635", "#fef08a", "#f59e0b", "#dc2626"], N=256)
im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=0.5, aspect="equal")
ax.set_xticks(range(len(ATTRS))); ax.set_xticklabels(['Race','Sex','Eth','Age'], fontsize=10, fontweight="bold")
ax.set_yticks(range(len(METRICS))); ax.set_yticklabels(METRICS, fontsize=10, fontweight="bold")
for i in range(len(METRICS)):
    for j in range(len(ATTRS)):
        if not np.isnan(mat[i, j]):
            color = "white" if mat[i, j] > 0.25 else "black"
            ax.text(j, i, f"{verdict[i,j]}\n{mat[i,j]:.2f}",
                    ha="center", va="center", fontsize=9, fontweight="bold", color=color)
plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="VFR")
ax.grid(False)
plt.tight_layout()
plt.savefig(FIG_DIR / "F3_vfr_heatmap.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
log(f"  saved F3 ({(FIG_DIR / 'F3_vfr_heatmap.png').stat().st_size/1024:.0f} KB)")

# =========================================================================
# F4 - CV-vs-N curves (need standard XGBoost predictions, then subsample grid)
# =========================================================================
log("F4: training light XGBoost (n_est=300) for CV curves")
mdl = xgb.XGBClassifier(
    n_estimators=300, max_depth=10, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.85, min_child_weight=3, reg_lambda=1.0,
    tree_method='hist', random_state=RANDOM_STATE, seed=RANDOM_STATE,
    eval_metric='logloss', verbosity=0, n_jobs=1,
)
mdl.fit(X_tr, y_tr)
proba = mdl.predict_proba(X_te)[:, 1].astype('float32')
pred = (proba >= 0.5).astype(int)
log(f"  Acc={(pred==y_te).mean():.4f}")

THRESHOLDS = {'DI':(0.80,'above'), 'SPD':(0.10,'below'), 'EOPP':(0.10,'below'),
              'EOD':(0.10,'below'), 'TI':(0.10,'below'), 'PP':(0.10,'below'), 'CAL':(0.05,'below')}
def passes(metric, value):
    thr, direction = THRESHOLDS[metric]
    return (value >= thr) if direction == 'above' else (value < thr)

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

log("F4: computing CV-vs-N curves (28 cells × 8 N × 30 reps)")
cv_table = {(m, a): [] for m in METRICS for a in ATTRS}
rng = np.random.default_rng(RANDOM_STATE)
for N_use in N_GRID:
    log(f"  N={N_use}")
    for a in ATTRS:
        for m in METRICS:
            vals = []
            for r in range(N_REPS):
                use_n = min(N_use, len(y_te))
                ix = rng.choice(len(y_te), use_n, replace=False)
                metrics = compute_seven(pred[ix], proba[ix], prot_te[a][ix], y_te[ix])
                if metrics: vals.append(metrics[m])
            vals = np.array(vals)
            if len(vals) < 5:
                cv_table[(m, a)].append(np.nan)
            else:
                mean_v = abs(np.mean(vals))
                cv = np.std(vals, ddof=1) / max(mean_v, 1e-9)
                cv_table[(m, a)].append(cv)

# Plot F4
fig, ax = plt.subplots(figsize=(11, 6))
metric_colors = {'DI':"#1d4ed8", 'SPD':"#0891b2", 'EOPP':"#16a34a", 'EOD':"#84cc16",
                 'TI':"#a855f7", 'PP':"#f59e0b", 'CAL':"#dc2626"}
attr_styles = {'RACE':'-', 'SEX':'--', 'ETHNICITY':':', 'AGE_GROUP':'-.'}
for m in METRICS:
    for a in ATTRS:
        ys = np.array(cv_table[(m, a)])
        ax.plot(N_GRID, ys, attr_styles[a], color=metric_colors[m], lw=1.6, alpha=0.85,
                label=f"{m}·{a[:3]}" if a == 'RACE' else None)
ax.axhline(0.05, color="black", ls="--", lw=1.5, label="CV = 0.05")
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel("audit cohort N (log)"); ax.set_ylabel("coefficient of variation (log)")
ax.set_title("F4 · CV vs audit size (28 metric × attribute cells, standard XGBoost)",
             fontsize=12, fontweight="bold", loc="left")
# Legend by metric color (one per metric)
from matplotlib.lines import Line2D
handles = [Line2D([0],[0], color=metric_colors[m], lw=2, label=m) for m in METRICS]
handles.append(Line2D([0],[0], color="black", ls="--", lw=1.5, label="CV = 0.05"))
ax.legend(handles=handles, fontsize=9, ncol=4, loc="upper right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "F4_cv_curves.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
log(f"  saved F4 ({(FIG_DIR / 'F4_cv_curves.png').stat().st_size/1024:.0f} KB)")

# =========================================================================
# F5 - hospital-fold violin (per-fold metric values across K=20 GroupKFold)
# =========================================================================
log("F5: K=20 GroupKFold for hospital-fold metric distributions (~10 min)")
gkf = GroupKFold(n_splits=K_FOLD)
splits = list(gkf.split(X_full, y_full, hosp_full))
fold_metrics = {(m, a): [] for m in METRICS for a in ATTRS}
for fold_id, (tr_ix, te_ix) in enumerate(splits, 1):
    log(f"  fold {fold_id}/{K_FOLD}")
    Xtr_f, ytr_f = X_full[tr_ix], y_full[tr_ix]
    Xte_f, yte_f = X_full[te_ix], y_full[te_ix]
    prot_f = {a: df[col].values[te_ix]
              for a, col in [('RACE','RACE'),('SEX','SEX_CODE'),
                             ('ETHNICITY','ETHNICITY'),('AGE_GROUP','AGE_GROUP')]}
    mdl_f = xgb.XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.05,
                               tree_method='hist', random_state=RANDOM_STATE, seed=RANDOM_STATE,
                               eval_metric='logloss', verbosity=0, n_jobs=1)
    mdl_f.fit(Xtr_f, ytr_f)
    proba_f = mdl_f.predict_proba(Xte_f)[:, 1].astype('float32')
    pred_f = (proba_f >= 0.5).astype(int)
    for a in ATTRS:
        m_dict = compute_seven(pred_f, proba_f, prot_f[a], yte_f)
        if m_dict is None: continue
        for mk in METRICS:
            fold_metrics[(mk, a)].append(m_dict[mk])

# Plot F5 - per-metric violin (one violin per metric, distribution across all 4 attrs × 20 folds)
fig, ax = plt.subplots(figsize=(11, 6))
positions = np.arange(len(METRICS))
data = []
for m in METRICS:
    vals = []
    for a in ATTRS:
        vals.extend(fold_metrics[(m, a)])
    data.append(vals)
parts = ax.violinplot(data, positions=positions, showmeans=False, showmedians=True, widths=0.85)
metric_color_list = [metric_colors[m] for m in METRICS]
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(metric_color_list[i]); pc.set_alpha(0.65); pc.set_edgecolor("black")
# Threshold markers per metric
for i, m in enumerate(METRICS):
    thr, direction = THRESHOLDS[m]
    ax.plot([i-0.4, i+0.4], [thr, thr], "k--", lw=1.5)
    ax.text(i+0.45, thr, f"τ={thr}", fontsize=8, va="center")
ax.set_xticks(positions); ax.set_xticklabels(METRICS, fontsize=10, fontweight="bold")
ax.set_ylabel("metric value")
ax.set_title(f"F5 · Per-metric distribution across K={K_FOLD} hospital folds × 4 attributes (80 verdicts each)",
             fontsize=12, fontweight="bold", loc="left")
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(FIG_DIR / "F5_hospital_violin.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
log(f"  saved F5 ({(FIG_DIR / 'F5_hospital_violin.png').stat().st_size/1024:.0f} KB)")

log(f"DONE  total: {time.time()-t0:.0f}s")
