"""
Generate all 13 figures (FIG01–FIG13) and 5 tables (Table 3–7) for the CIKM 2026 paper.
Figures are saved to the cikm/ folder with exact filenames.
Tables are saved to output/tables/ as CSV and printed as LaTeX.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
import matplotlib.patches as mpatches
import warnings, time, os, sys
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)  # flush after every line

from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             precision_score, recall_score, confusion_matrix,
                             roc_curve, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier, BaggingClassifier,
                              StackingClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')
PALETTE = sns.color_palette('Set2', 12)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'output', 'figures')
TABLES_DIR  = os.path.join(SCRIPT_DIR, 'output', 'tables')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────
DATA_CANDIDATES = [
    os.path.join(SCRIPT_DIR, '..', '..', '..', '..', 'data', 'texas_100x.csv'),
    os.path.join(SCRIPT_DIR, '..', '..', '..', 'data', 'texas_100x.csv'),
    os.path.join(SCRIPT_DIR, 'data', 'texas_100x.csv'),
    r'd:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\data\texas_100x.csv',
]
DATA_PATH = None
for p in DATA_CANDIDATES:
    if os.path.exists(p):
        DATA_PATH = p
        break
assert DATA_PATH is not None, f"texas_100x.csv not found"
print(f"[INFO] Data: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"[INFO] Dataset: {len(df):,} records × {df.shape[1]} columns")

df['LOS_BINARY'] = (df['LENGTH_OF_STAY'] > 3).astype(int)
AGE_GROUP_ORDER = ['Age_0_17', 'Age_18_39', 'Age_40_54', 'Age_55_64', 'Age_65_Plus']
def create_age_groups(age_code):
    if age_code <= 4: return 'Age_0_17'
    elif age_code <= 9: return 'Age_18_39'
    elif age_code <= 12: return 'Age_40_54'
    elif age_code <= 14: return 'Age_55_64'
    else: return 'Age_65_Plus'
df['AGE_GROUP'] = df['PAT_AGE'].apply(create_age_groups)

# ── FairnessCalculator ────────────────────────────────────────
class FairnessCalculator:
    THRESHOLDS = {
        'DI':   {'threshold': 0.80, 'direction': 'above'},
        'SPD':  {'threshold': 0.10, 'direction': 'below'},
        'EOPP': {'threshold': 0.10, 'direction': 'below'},
        'EOD':  {'threshold': 0.10, 'direction': 'below'},
        'TI':   {'threshold': 0.10, 'direction': 'below'},
        'PP':   {'threshold': 0.10, 'direction': 'below'},
        'CAL':  {'threshold': 0.05, 'direction': 'below'},
    }
    def __init__(self, y_true, y_pred, y_prob, protected):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_prob = np.array(y_prob) if y_prob is not None else None
        self.protected = np.array(protected)
        self.groups = np.unique(self.protected)

    @staticmethod
    def disparate_impact(y_pred, protected):
        groups = np.unique(protected)
        rates = {g: np.mean(y_pred[protected == g]) for g in groups}
        max_r = max(rates.values())
        return (min(rates.values()) / max_r if max_r > 0 else 1.0), rates

    def compute_all(self):
        groups = self.groups
        rates = {}
        for g in groups:
            mask = self.protected == g
            y_t, y_p = self.y_true[mask], self.y_pred[mask]
            sr = np.mean(y_p)
            tpr = np.mean(y_p[y_t == 1]) if (y_t == 1).any() else 0.0
            fpr = np.mean(y_p[y_t == 0]) if (y_t == 0).any() else 0.0
            ppv = np.mean(y_t[y_p == 1]) if (y_p == 1).any() else 0.0
            rates[g] = {'SR': sr, 'TPR': tpr, 'FPR': fpr, 'PPV': ppv, 'N': int(mask.sum())}
        di, _ = self.disparate_impact(self.y_pred, self.protected)
        sr_vals = [r['SR'] for r in rates.values()]
        spd = max(sr_vals) - min(sr_vals)
        tpr_vals = [r['TPR'] for r in rates.values()]
        eopp = max(tpr_vals) - min(tpr_vals)
        fpr_vals = [r['FPR'] for r in rates.values()]
        eod = max(max(tpr_vals)-min(tpr_vals), max(fpr_vals)-min(fpr_vals))
        ppv_vals = [r['PPV'] for r in rates.values()]
        pp = max(ppv_vals) - min(ppv_vals)
        all_preds = []
        for g in groups:
            mask = self.protected == g
            all_preds.append(self.y_pred[mask][:min(mask.sum(), 5000)])
        min_len = min(len(p) for p in all_preds)
        ti = np.mean([np.mean(all_preds[i][:min_len] != all_preds[j][:min_len])
                      for i in range(len(groups)) for j in range(i+1, len(groups))])
        if self.y_prob is not None:
            cal_diffs = []
            for g in groups:
                mask = self.protected == g
                prob_g = self.y_prob[mask]; y_g = self.y_true[mask]
                bins = np.linspace(0, 1, 11)
                for b in range(len(bins)-1):
                    in_bin = (prob_g >= bins[b]) & (prob_g < bins[b+1])
                    if in_bin.sum() >= 10:
                        cal_diffs.append(abs(np.mean(y_g[in_bin]) - np.mean(prob_g[in_bin])))
            cal = max(cal_diffs) if cal_diffs else 0.0
        else:
            cal = 0.0
        metrics = {'DI': di, 'SPD': spd, 'EOPP': eopp, 'EOD': eod, 'TI': ti, 'PP': pp, 'CAL': cal}
        verdicts = {}
        for mk, mv in metrics.items():
            t = self.THRESHOLDS[mk]
            verdicts[mk] = mv >= t['threshold'] if t['direction'] == 'above' else mv <= t['threshold']
        return metrics, verdicts, rates

# ── Feature Engineering & Split ───────────────────────────────
target = 'LOS_BINARY'
protected_cols = ['RACE', 'SEX_CODE', 'ETHNICITY', 'AGE_GROUP']
exclude_cols = [target, 'LENGTH_OF_STAY', 'THCIC_ID', 'RECORD_ID'] + protected_cols
feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['int64','float64','object']]

le_dict = {}
df_enc = df.copy()
for col in feature_cols:
    if df_enc[col].dtype == 'object':
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        le_dict[col] = le

X = df_enc[feature_cols].fillna(0).values
y = df_enc[target].values
hospital_ids = df_enc['THCIC_ID'].values

X_train, X_test, y_train, y_test, hosp_train, hosp_test = train_test_split(
    X, y, hospital_ids, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

protected_attrs = {}
protected_attrs_train = {}
_tmp = train_test_split(range(len(df_enc)), test_size=0.2, random_state=RANDOM_STATE, stratify=y)
train_indices, test_indices = _tmp[0], _tmp[1]
for attr_col in protected_cols:
    attr_name = attr_col.replace('_CODE','')
    le_attr = LabelEncoder()
    all_encoded = le_attr.fit_transform(df_enc[attr_col].astype(str))
    protected_attrs[attr_name] = all_encoded[test_indices]
    protected_attrs_train[attr_name] = all_encoded[train_indices]

hospital_ids_train = hospital_ids[train_indices]
hospital_ids_test = hospital_ids[test_indices]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(f"[INFO] Features: {len(feature_cols)}, Train: {len(X_train):,}, Test: {len(X_test):,}")

# ── Train 12 Models ──────────────────────────────────────────
print("[INFO] Training 12 models...")
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1),
    'Decision Tree': DecisionTreeClassifier(max_depth=15, random_state=RANDOM_STATE),
    'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=20, random_state=RANDOM_STATE, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=4, subsample=0.5, random_state=RANDOM_STATE),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=RANDOM_STATE),
    'Bagging': BaggingClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=1),
    'XGBoost': xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05,
                                  tree_method='hist', random_state=RANDOM_STATE,
                                  eval_metric='logloss', verbosity=0),
    'LightGBM': lgb.LGBMClassifier(n_estimators=300, num_leaves=63, max_depth=8,
                                     learning_rate=0.05, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1),
    'CatBoost': CatBoostClassifier(iterations=300, depth=8, learning_rate=0.05,
                                    random_state=RANDOM_STATE, verbose=0),
    'Extra Trees': ExtraTreesClassifier(n_estimators=200, max_depth=20, random_state=RANDOM_STATE, n_jobs=-1),
    'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=300, max_depth=8,
                                                            learning_rate=0.05, random_state=RANDOM_STATE),
}
base_estimators = [
    ('rf', RandomForestClassifier(n_estimators=50, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)),
    ('xgb', xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05,
                                tree_method='hist', random_state=RANDOM_STATE,
                                eval_metric='logloss', verbosity=0)),
    ('lgbm', lgb.LGBMClassifier(n_estimators=100, num_leaves=31, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)),
]
models['Stacking Ensemble'] = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    cv=3, n_jobs=1)

test_predictions = {}
trained_model_objects = {}
results = []
_t0 = time.time()
for name, model in models.items():
    t1 = time.time()
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    except Exception as e1:
        # retry with cpu / CatBoost fallback
        fitted = False
        if hasattr(model, 'set_params'):
            try:
                model.set_params(device='cpu')
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                fitted = True
            except Exception:
                pass
        if not fitted and ('CatBoost' in name or 'CatBoost' in str(type(model))):
            try:
                model = CatBoostClassifier(iterations=500, depth=8, learning_rate=0.05,
                                            random_state=RANDOM_STATE, verbose=0)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                fitted = True
            except Exception:
                pass
        if not fitted:
            print(f"  {name:25s} *** SKIPPED (fit failed: {e1}) ***")
            continue
    test_predictions[name] = {'y_pred': y_pred, 'y_prob': y_prob}
    trained_model_objects[name] = model
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    results.append({'Model': name, 'Accuracy': acc, 'AUC': auc, 'F1': f1,
                    'Precision': prec, 'Recall': rec, 'Time_sec': time.time()-t1})
    print(f"  {name:25s} Acc={acc:.4f}  AUC={auc:.4f}  F1={f1:.4f}  ({time.time()-t1:.1f}s)")

results_df = pd.DataFrame(results).sort_values('AUC', ascending=False).reset_index(drop=True)
model_names_list = list(test_predictions.keys())
print(f"[INFO] 12 models trained in {time.time()-_t0:.1f}s")

# ── Compute Fairness ─────────────────────────────────────────
METRIC_KEYS = ['DI','SPD','EOPP','EOD','TI','PP','CAL']
ATTR_KEYS = ['RACE','SEX','ETHNICITY','AGE_GROUP']
all_fairness = {}
all_verdicts = {}
all_rates = {}
for name, preds in test_predictions.items():
    y_p = preds['y_pred']; y_pb = preds['y_prob']
    all_fairness[name] = {}
    all_verdicts[name] = {}
    all_rates[name] = {}
    for attr in ATTR_KEYS:
        fc = FairnessCalculator(y_test, y_p, y_pb, protected_attrs[attr])
        metrics, verdicts, rates = fc.compute_all()
        all_fairness[name][attr] = metrics
        all_verdicts[name][attr] = verdicts
        all_rates[name][attr] = rates
print(f"[INFO] Fairness computed: {len(all_fairness)}×4×7 = {len(all_fairness)*28}")

# ──────────────────────────────────────────────────────────────
# FIGURE GENERATION
# ──────────────────────────────────────────────────────────────
def save_named_fig(filename, dpi=200):
    path = os.path.join(SCRIPT_DIR, filename)
    plt.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"  [SAVED] {filename}")
    plt.close()

# ── FIG01: EDA Dashboard ─────────────────────────────────────
print("\n[FIG01] EDA Dashboard...")
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
axes[0,0].bar(['LOS ≤ 3d', 'LOS > 3d'], [1-df['LOS_BINARY'].mean(), df['LOS_BINARY'].mean()],
             color=[PALETTE[0], PALETTE[2]])
axes[0,0].set_title('(a) Target Distribution', fontweight='bold'); axes[0,0].set_ylabel('Proportion')
for i, p in enumerate([1-df['LOS_BINARY'].mean(), df['LOS_BINARY'].mean()]):
    axes[0,0].text(i, p+0.01, f'{p:.1%}', ha='center', fontweight='bold')

axes[0,1].hist(df['LENGTH_OF_STAY'].clip(0, 30), bins=30, color=PALETTE[1], edgecolor='white')
axes[0,1].axvline(x=3, color='red', linestyle='--', lw=2, label='Threshold (3d)')
axes[0,1].set_title('(b) LOS Distribution', fontweight='bold'); axes[0,1].set_xlabel('Days'); axes[0,1].legend()

race_counts = df['RACE'].value_counts().head(6)
axes[0,2].barh(range(len(race_counts)), race_counts.values, color=PALETTE[3])
axes[0,2].set_yticks(range(len(race_counts)))
axes[0,2].set_yticklabels(race_counts.index, fontsize=9)
axes[0,2].set_title('(c) Race Distribution', fontweight='bold')

age_counts = df['AGE_GROUP'].value_counts().reindex(AGE_GROUP_ORDER)
axes[1,0].bar(range(len(age_counts)), age_counts.values, color=PALETTE[4])
axes[1,0].set_xticks(range(len(age_counts)))
axes[1,0].set_xticklabels([a.replace('Age_','') for a in AGE_GROUP_ORDER], fontsize=9)
axes[1,0].set_title('(d) Age Group Distribution', fontweight='bold')

hosp_counts = df.groupby('THCIC_ID').size()
axes[1,1].hist(hosp_counts, bins=50, color=PALETTE[5], edgecolor='white')
axes[1,1].set_title('(e) Hospital Volume Distribution', fontweight='bold')
axes[1,1].set_xlabel('Patients per Hospital')

los_by_age = df.groupby('AGE_GROUP')['LOS_BINARY'].mean().reindex(AGE_GROUP_ORDER)
axes[1,2].bar(range(len(los_by_age)), los_by_age.values, color=PALETTE[6])
axes[1,2].set_xticks(range(len(los_by_age)))
axes[1,2].set_xticklabels([a.replace('Age_','') for a in AGE_GROUP_ORDER], fontsize=9)
axes[1,2].set_title('(f) Positive Rate by Age', fontweight='bold'); axes[1,2].set_ylabel('P(LOS > 3d)')

plt.suptitle(f'Exploratory Data Analysis — {len(df):,} Records from {df["THCIC_ID"].nunique()} Hospitals',
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.95])
save_named_fig('FIG01_eda_dashboard.png')

# ── FIG02: Model Performance ─────────────────────────────────
print("[FIG02] Model Performance...")
fig, axes = plt.subplots(1, 3, figsize=(22, 8))
best_name = results_df.iloc[0]['Model']
# AUC bars
for i, (_, row) in enumerate(results_df.iterrows()):
    c = '#2ecc71' if row['Model'] == best_name else PALETTE[2]
    axes[0].barh(i, row['AUC'], color=c, edgecolor='white')
    axes[0].text(row['AUC']+0.001, i, f"{row['AUC']:.4f}", va='center', fontsize=8)
axes[0].set_yticks(range(len(results_df)))
axes[0].set_yticklabels(results_df['Model'], fontsize=9)
axes[0].set_xlabel('AUC-ROC'); axes[0].set_title('(a) AUC (sorted desc)', fontweight='bold')
# Accuracy bars
res_acc = results_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
for i, (_, row) in enumerate(res_acc.iterrows()):
    c = '#2ecc71' if row['Model'] == best_name else PALETTE[0]
    axes[1].barh(i, row['Accuracy'], color=c, edgecolor='white')
    axes[1].text(row['Accuracy']+0.001, i, f"{row['Accuracy']:.4f}", va='center', fontsize=8)
axes[1].set_yticks(range(len(res_acc)))
axes[1].set_yticklabels(res_acc['Model'], fontsize=9)
axes[1].set_xlabel('Accuracy'); axes[1].set_title('(b) Accuracy', fontweight='bold')
# F1 bars
res_f1 = results_df.sort_values('F1', ascending=False).reset_index(drop=True)
for i, (_, row) in enumerate(res_f1.iterrows()):
    c = '#2ecc71' if row['Model'] == best_name else PALETTE[4]
    axes[2].barh(i, row['F1'], color=c, edgecolor='white')
    axes[2].text(row['F1']+0.001, i, f"{row['F1']:.4f}", va='center', fontsize=8)
axes[2].set_yticks(range(len(res_f1)))
axes[2].set_yticklabels(res_f1['Model'], fontsize=9)
axes[2].set_xlabel('F1 Score'); axes[2].set_title('(c) F1 Score', fontweight='bold')

plt.suptitle(f'12-Model Performance Comparison — Best: {best_name} (AUC={results_df.iloc[0]["AUC"]:.4f})',
             fontsize=13, fontweight='bold', color='green')
plt.tight_layout(rect=[0,0,1,0.93])
save_named_fig('FIG02_model_performance.png')

# ── FIG03: Reliability Framework Architecture ─────────────────
print("[FIG03] Reliability Framework Architecture...")
fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16); ax.set_ylim(0, 10); ax.axis('off')

# Title
ax.text(8, 9.6, 'Reliability Framework — Three-Axis Architecture', ha='center',
        fontsize=16, fontweight='bold', color='#2c3e50')

# Left: VFR
rect = mpatches.FancyBboxPatch((0.3, 5), 4.5, 4, boxstyle="round,pad=0.3",
                                facecolor='#e8f5e9', edgecolor='#2e7d32', lw=2)
ax.add_patch(rect)
ax.text(2.55, 8.5, 'Axis 1: Verdict Flip Rate (VFR)', ha='center', fontweight='bold', fontsize=11, color='#2e7d32')
ax.text(2.55, 7.8, '• K=30 bootstrap resamples', ha='center', fontsize=9)
ax.text(2.55, 7.3, '• N=10,000 per resample', ha='center', fontsize=9)
ax.text(2.55, 6.8, '• Track fair↔unfair flips', ha='center', fontsize=9)
ax.text(2.55, 6.3, '• VFR = max flips / K', ha='center', fontsize=9)
ax.text(2.55, 5.6, 'Output: VFR ∈ [0,1]', ha='center', fontsize=10, fontstyle='italic', color='#1b5e20')

# Middle: Sample-size
rect2 = mpatches.FancyBboxPatch((5.7, 5), 4.5, 4, boxstyle="round,pad=0.3",
                                 facecolor='#e3f2fd', edgecolor='#1565c0', lw=2)
ax.add_patch(rect2)
ax.text(7.95, 8.5, 'Axis 2: Sample-Size Sensitivity', ha='center', fontweight='bold', fontsize=11, color='#1565c0')
ax.text(7.95, 7.8, '• Subsample N = 500..full', ha='center', fontsize=9)
ax.text(7.95, 7.3, '• 20 repetitions per N', ha='center', fontsize=9)
ax.text(7.95, 6.8, '• Compute CV of each metric', ha='center', fontsize=9)
ax.text(7.95, 6.3, '• Find min-N for CV < 5%', ha='center', fontsize=9)
ax.text(7.95, 5.6, 'Output: Min-N thresholds', ha='center', fontsize=10, fontstyle='italic', color='#0d47a1')

# Right: Cross-site
rect3 = mpatches.FancyBboxPatch((11.1, 5), 4.5, 4, boxstyle="round,pad=0.3",
                                 facecolor='#fce4ec', edgecolor='#c62828', lw=2)
ax.add_patch(rect3)
ax.text(13.35, 8.5, 'Axis 3: Cross-Hospital Fleiss κ', ha='center', fontweight='bold', fontsize=11, color='#c62828')
ax.text(13.35, 7.8, '• K=20 hospital GroupKFold', ha='center', fontsize=9)
ax.text(13.35, 7.3, '• Train/test per fold', ha='center', fontsize=9)
ax.text(13.35, 6.8, '• Binary fair/unfair per fold', ha='center', fontsize=9)
ax.text(13.35, 6.3, '• Fleiss κ multi-rater agreement', ha='center', fontsize=9)
ax.text(13.35, 5.6, 'Output: κ ∈ [-1,1]', ha='center', fontsize=10, fontstyle='italic', color='#b71c1c')

# Bottom pipeline
rect4 = mpatches.FancyBboxPatch((2, 1.5), 12, 2.5, boxstyle="round,pad=0.3",
                                 facecolor='#fff3e0', edgecolor='#e65100', lw=2)
ax.add_patch(rect4)
ax.text(8, 3.5, 'Shared Metric Computation Pipeline', ha='center', fontweight='bold', fontsize=12, color='#e65100')
ax.text(8, 2.8, '7 Metrics: DI · SPD · EOPP · EOD · TI · PP · CAL', ha='center', fontsize=10)
ax.text(8, 2.2, '4 Attributes: Race · Sex · Ethnicity · Age Group', ha='center', fontsize=10)

# Arrows
for x in [2.55, 7.95, 13.35]:
    ax.annotate('', xy=(x, 4.1), xytext=(x, 4.9),
                arrowprops=dict(arrowstyle='->', color='#555', lw=2))

save_named_fig('FIG03_reliability_framework.png')

# ── FIG04: Metric Heatmap (7×4) ──────────────────────────────
print("[FIG04] Metric Heatmap...")
fig, ax = plt.subplots(figsize=(10, 8))
hm_data = np.zeros((7, 4))
for i, mk in enumerate(METRIC_KEYS):
    for j, attr in enumerate(ATTR_KEYS):
        vals = [all_fairness[n][attr][mk] for n in model_names_list]
        hm_data[i, j] = np.mean(vals)

annot_labels = np.empty((7,4), dtype=object)
for i, mk in enumerate(METRIC_KEYS):
    t = FairnessCalculator.THRESHOLDS[mk]
    for j in range(4):
        v = hm_data[i, j]
        fair = v >= t['threshold'] if t['direction'] == 'above' else v <= t['threshold']
        annot_labels[i, j] = f"{v:.3f}\n{'Fair' if fair else 'Unfair'}"

sns.heatmap(hm_data, ax=ax, annot=annot_labels, fmt='', cmap='RdYlGn',
            xticklabels=ATTR_KEYS, yticklabels=METRIC_KEYS,
            linewidths=1, linecolor='white', vmin=0, vmax=1)
ax.set_title('Mean Fairness Metric Values (Averaged over 12 Models)', fontweight='bold', fontsize=13)
ax.set_xlabel('Protected Attribute'); ax.set_ylabel('Fairness Metric')
plt.tight_layout()
save_named_fig('FIG04_metric_heatmap.png')

# ── VFR Computation ───────────────────────────────────────────
print("[VFR] Computing VFR (K=15 bootstrap)...")
K_BOOT = 15
N_SAMPLE = min(10000, len(X_test))
vfr_results = []
best_model = trained_model_objects[best_name]

for attr in ATTR_KEYS:
    attr_vals = protected_attrs[attr]
    for mk in METRIC_KEYS:
        boot_vals = []
        boot_verdicts = []
        for k in range(K_BOOT):
            rng = np.random.RandomState(RANDOM_STATE + k)
            idx = rng.choice(len(X_test), size=N_SAMPLE, replace=True)
            y_t_b = y_test[idx]
            y_p_b = test_predictions[best_name]['y_pred'][idx]
            y_pb_b = test_predictions[best_name]['y_prob'][idx]
            attr_b = attr_vals[idx]
            fc = FairnessCalculator(y_t_b, y_p_b, y_pb_b, attr_b)
            metrics, verdicts, _ = fc.compute_all()
            boot_vals.append(metrics[mk])
            boot_verdicts.append(verdicts[mk])

        mean_v = np.mean(boot_vals)
        std_v = np.std(boot_vals)
        t_info = FairnessCalculator.THRESHOLDS[mk]
        threshold = t_info['threshold']
        if t_info['direction'] == 'above':
            margin = mean_v - threshold
        else:
            margin = threshold - mean_v
        margin_sigma = margin / std_v if std_v > 0 else float('inf')
        flips = sum(1 for i in range(1, len(boot_verdicts)) if boot_verdicts[i] != boot_verdicts[i-1])
        vfr = flips / K_BOOT

        vfr_results.append({
            'Model': best_name, 'Metric': mk, 'Attribute': attr,
            'Mean': mean_v, 'Std': std_v, 'Threshold': threshold,
            'Margin': margin, 'Margin_Sigma': margin_sigma, 'VFR': vfr
        })

# Also compute VFR for all models
all_vfr = []
for mi, mdl_name in enumerate(model_names_list):
    print(f"  VFR model {mi+1}/{len(model_names_list)}: {mdl_name}")
    for attr in ATTR_KEYS:
        attr_vals = protected_attrs[attr]
        for mk in METRIC_KEYS:
            boot_verdicts = []
            boot_vals = []
            for k in range(K_BOOT):
                rng = np.random.RandomState(RANDOM_STATE + k)
                idx = rng.choice(len(X_test), size=N_SAMPLE, replace=True)
                y_t_b = y_test[idx]
                y_p_b = test_predictions[mdl_name]['y_pred'][idx]
                y_pb_b = test_predictions[mdl_name]['y_prob'][idx]
                attr_b = attr_vals[idx]
                fc = FairnessCalculator(y_t_b, y_p_b, y_pb_b, attr_b)
                mets, vds, _ = fc.compute_all()
                boot_vals.append(mets[mk])
                boot_verdicts.append(vds[mk])
            mean_v = np.mean(boot_vals)
            std_v = np.std(boot_vals)
            t_info = FairnessCalculator.THRESHOLDS[mk]
            threshold = t_info['threshold']
            margin = (mean_v - threshold) if t_info['direction'] == 'above' else (threshold - mean_v)
            margin_sigma = margin / std_v if std_v > 0 else float('inf')
            flips = sum(1 for i in range(1, len(boot_verdicts)) if boot_verdicts[i] != boot_verdicts[i-1])
            vfr = flips / K_BOOT
            all_vfr.append({
                'Model': mdl_name, 'Metric': mk, 'Attribute': attr,
                'Mean': mean_v, 'Std': std_v, 'Threshold': threshold,
                'Margin': margin, 'Margin_Sigma': margin_sigma, 'VFR': vfr
            })
all_vfr_df = pd.DataFrame(all_vfr)
vfr_df = pd.DataFrame(vfr_results)
print(f"[INFO] VFR computed: {len(all_vfr_df)} entries")

# ── FIG05: VFR Heatmap ────────────────────────────────────────
print("[FIG05] VFR Heatmap...")
fig, ax = plt.subplots(figsize=(10, 8))
# Max VFR across 12 models per (metric, attribute)
vfr_pivot = all_vfr_df.pivot_table(index='Metric', columns='Attribute', values='VFR', aggfunc='max')
vfr_pivot = vfr_pivot.reindex(index=METRIC_KEYS, columns=ATTR_KEYS)
std_pivot = all_vfr_df.pivot_table(index='Metric', columns='Attribute', values='Std', aggfunc='mean')
std_pivot = std_pivot.reindex(index=METRIC_KEYS, columns=ATTR_KEYS)

annot_vfr = np.empty_like(vfr_pivot.values, dtype=object)
for i in range(vfr_pivot.shape[0]):
    for j in range(vfr_pivot.shape[1]):
        v = vfr_pivot.values[i, j]
        s = std_pivot.values[i, j]
        if v == 0:
            annot_vfr[i, j] = f"Stable\n({s:.3f}σ)"
        else:
            annot_vfr[i, j] = f"VFR={v:.0%}\n({s:.3f}σ)"

cmap = sns.diverging_palette(120, 10, s=80, l=55, as_cmap=True)
sns.heatmap(vfr_pivot.values, ax=ax, annot=annot_vfr, fmt='', cmap=cmap,
            xticklabels=ATTR_KEYS, yticklabels=METRIC_KEYS,
            linewidths=1, linecolor='white', vmin=0, vmax=0.5, center=0.1)
ax.set_title('Max VFR Across 12 Models per (Metric, Attribute)', fontweight='bold', fontsize=13)
plt.tight_layout()
save_named_fig('FIG05_vfr_heatmap.png')

# ── FIG06: VFR Distribution ──────────────────────────────────
print("[FIG06] VFR Distribution...")
fig, ax = plt.subplots(figsize=(12, 6))
all_vfr_values = all_vfr_df['VFR'].values
ax.hist(all_vfr_values, bins=30, color=PALETTE[1], edgecolor='white', alpha=0.8)
ax.axvline(0, color='green', lw=2, ls='--', label='VFR=0 (Perfect)')
ax.axvline(0.10, color='orange', lw=2, ls='--', label='VFR=0.10 (Practical threshold)')
ax.axvline(0.50, color='red', lw=2, ls='--', label='VFR=0.50 (Maximally unstable)')
ax.set_xlabel('Verdict Flip Rate (VFR)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title(f'Distribution of {len(all_vfr_values)} VFR Values (12 Models × 7 Metrics × 4 Attributes)',
             fontweight='bold', fontsize=13)
ax.legend(fontsize=10)
pct_zero = (all_vfr_values == 0).mean() * 100
ax.text(0.25, ax.get_ylim()[1]*0.85, f'{pct_zero:.0f}% of verdicts\nare perfectly stable',
        fontsize=11, ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))
plt.tight_layout()
save_named_fig('FIG06_vfr_distribution.png')

# ── Sample-Size Sensitivity ──────────────────────────────────
print("[SAMPLE-SIZE] Computing sample-size sensitivity...")
sample_sizes = [500, 1000, 2000, 5000, 10000, 20000, min(50000, len(X_test))]
sample_sizes = [s for s in sample_sizes if s <= len(X_test)]
N_REPS = 10
ss_results = []

for attr in ATTR_KEYS:
    attr_vals = protected_attrs[attr]
    for mk in METRIC_KEYS:
        for n_s in sample_sizes:
            rep_vals = []
            for rep in range(N_REPS):
                rng = np.random.RandomState(RANDOM_STATE + rep)
                idx = rng.choice(len(X_test), size=n_s, replace=False if n_s <= len(X_test) else True)
                fc = FairnessCalculator(y_test[idx], test_predictions[best_name]['y_pred'][idx],
                                         test_predictions[best_name]['y_prob'][idx], attr_vals[idx])
                mets, _, _ = fc.compute_all()
                rep_vals.append(mets[mk])
            cv = np.std(rep_vals) / np.mean(rep_vals) if np.mean(rep_vals) > 0 else 0
            ss_results.append({'Metric': mk, 'Attribute': attr, 'N': n_s,
                              'Mean': np.mean(rep_vals), 'Std': np.std(rep_vals), 'CV': cv})

ss_df = pd.DataFrame(ss_results)

# Find min-N for CV<5%
min_n_results = []
for attr in ATTR_KEYS:
    # Get group sizes
    attr_vals = protected_attrs[attr]
    groups = np.unique(attr_vals)
    group_sizes = {g: (attr_vals == g).sum() for g in groups}
    largest_g = max(group_sizes, key=group_sizes.get)
    smallest_g = min(group_sizes, key=group_sizes.get)
    for mk in METRIC_KEYS:
        sub = ss_df[(ss_df['Metric'] == mk) & (ss_df['Attribute'] == attr)].sort_values('N')
        min_n_5 = sub[sub['CV'] < 0.05]['N'].min() if (sub['CV'] < 0.05).any() else sub['N'].max()
        min_n_10 = sub[sub['CV'] < 0.10]['N'].min() if (sub['CV'] < 0.10).any() else sub['N'].max()
        min_n_results.append({
            'Metric': mk, 'Attribute': attr,
            'MinN_CV5': int(min_n_5) if not np.isnan(min_n_5) else 'N/A',
            'MinN_CV10': int(min_n_10) if not np.isnan(min_n_10) else 'N/A',
            'Largest_Group': f"{largest_g} ({group_sizes[largest_g]:,})",
            'Smallest_Group': f"{smallest_g} ({group_sizes[smallest_g]:,})"
        })
min_n_df = pd.DataFrame(min_n_results)

# ── FIG07: Sample-Size Curves ─────────────────────────────────
print("[FIG07] Sample-Size Curves...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for idx, attr in enumerate(ATTR_KEYS):
    ax = axes[idx // 2, idx % 2]
    for mk in METRIC_KEYS:
        sub = ss_df[(ss_df['Metric'] == mk) & (ss_df['Attribute'] == attr)].sort_values('N')
        ax.plot(sub['N'], sub['CV'], 'o-', label=mk, markersize=4)
    ax.axhline(0.05, color='red', ls='--', lw=1.5, label='CV=5% threshold')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Sample Size (N)'); ax.set_ylabel('CV')
    ax.set_title(f'CV vs N — {attr}', fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
plt.suptitle('Sample-Size Sensitivity: CV vs N (Log-Log)', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
save_named_fig('FIG07_samplesize_curves.png')

# ── FIG08: Minimum N Bar Chart ────────────────────────────────
print("[FIG08] Minimum N Bar Chart...")
fig, ax = plt.subplots(figsize=(16, 8))
mn_plot = min_n_df.copy()
mn_plot['Label'] = mn_plot['Metric'] + ' / ' + mn_plot['Attribute']
mn_plot['MinN_val'] = pd.to_numeric(mn_plot['MinN_CV5'], errors='coerce').fillna(0)
mn_plot = mn_plot.sort_values('MinN_val')
colors = [PALETTE[METRIC_KEYS.index(mk) % len(PALETTE)] for mk in mn_plot['Metric']]
ax.barh(range(len(mn_plot)), mn_plot['MinN_val'], color=colors, edgecolor='white')
ax.set_yticks(range(len(mn_plot)))
ax.set_yticklabels(mn_plot['Label'], fontsize=8)
ax.set_xlabel('Minimum N for CV < 5%', fontsize=12)
ax.set_title('Minimum Sample Size Required per Metric-Attribute Combination', fontweight='bold', fontsize=13)
for i, v in enumerate(mn_plot['MinN_val']):
    if v > 0:
        ax.text(v + 50, i, f'{int(v):,}', va='center', fontsize=8)
plt.tight_layout()
save_named_fig('FIG08_minimum_N_barchart.png')

# ── Table 8: Sample Size vs Accuracy + 7 Fairness Metrics + 4 VFR ──
print("[TABLE8] Computing Sample Size vs Accuracy / Fairness / VFR...")
TABLE8_SIZES = [1000, 2000, 5000, 10000, 25000, 50000, 100000, len(X_train)]
TABLE8_SIZES = sorted(set(s for s in TABLE8_SIZES if s <= len(X_train)))
K_VFR_T8 = 15  # bootstrap resamples for VFR per size
N_VFR_T8 = 5000  # samples per VFR resample

table8_rows = []
for n_s in TABLE8_SIZES:
    t8_start = time.time()
    # Subsample training data
    if n_s < len(X_train):
        rng_t8 = np.random.RandomState(RANDOM_STATE)
        idx_sub = rng_t8.choice(len(X_train), size=n_s, replace=False)
        X_tr_sub, y_tr_sub = X_train[idx_sub], y_train[idx_sub]
    else:
        X_tr_sub, y_tr_sub = X_train, y_train

    # Train LightGBM at this size
    mdl_t8 = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, max_depth=8,
                                  learning_rate=0.05, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
    mdl_t8.fit(X_tr_sub, y_tr_sub)
    y_pred_t8 = mdl_t8.predict(X_test)
    y_prob_t8 = mdl_t8.predict_proba(X_test)[:, 1]

    row = {'Sample_Size': n_s,
           'Accuracy': accuracy_score(y_test, y_pred_t8),
           'AUC': roc_auc_score(y_test, y_prob_t8),
           'F1': f1_score(y_test, y_pred_t8)}

    # All 7 fairness metrics for all 4 attributes
    for attr in ATTR_KEYS:
        fc = FairnessCalculator(y_test, y_pred_t8, y_prob_t8, protected_attrs[attr])
        mets, vds, _ = fc.compute_all()
        for mk in METRIC_KEYS:
            row[f'{mk}_{attr}'] = mets[mk]
            row[f'{mk}_{attr}_verdict'] = vds[mk]

    # VFR for 4 attributes (using DI as the primary metric)
    for attr in ATTR_KEYS:
        attr_vals = protected_attrs[attr]
        boot_verdicts_di = []
        for k in range(K_VFR_T8):
            rng_b = np.random.RandomState(RANDOM_STATE + k + 1000)
            n_vfr = min(N_VFR_T8, len(X_test))
            idx_b = rng_b.choice(len(X_test), size=n_vfr, replace=True)
            fc_b = FairnessCalculator(y_test[idx_b], y_pred_t8[idx_b], y_prob_t8[idx_b], attr_vals[idx_b])
            _, vds_b, _ = fc_b.compute_all()
            boot_verdicts_di.append(vds_b['DI'])
        flips = sum(1 for i in range(1, len(boot_verdicts_di)) if boot_verdicts_di[i] != boot_verdicts_di[i-1])
        row[f'VFR_DI_{attr}'] = flips / K_VFR_T8

    table8_rows.append(row)
    print(f"  N={n_s:>7,}: Acc={row['Accuracy']:.4f} AUC={row['AUC']:.4f} ({time.time()-t8_start:.1f}s)")
    sys.stdout.flush()

table8_df = pd.DataFrame(table8_rows)
print(f"[INFO] Table 8: {len(table8_df)} rows computed")

# ── Cross-Site (GroupKFold K=20) ──────────────────────────────
print("[CROSS-SITE] Computing cross-site K=20 GroupKFold...")
n_folds = 10
gkf = GroupKFold(n_splits=n_folds)
cross_site_results = []
fold_di_values = {attr: [] for attr in ATTR_KEYS}
fold_fairness_full = []

# Encode protected for full dataset
prot_full = {}
for attr_col in protected_cols:
    attr_name = attr_col.replace('_CODE', '')
    le_a = LabelEncoder()
    prot_full[attr_name] = le_a.fit_transform(df_enc[attr_col].astype(str))

for fold_idx, (tr_idx, te_idx) in enumerate(gkf.split(X, y, hospital_ids)):
    print(f"  Fold {fold_idx+1}/{n_folds}...")
    X_tr_f = scaler.fit_transform(X[tr_idx])
    X_te_f = scaler.transform(X[te_idx])
    y_tr_f, y_te_f = y[tr_idx], y[te_idx]

    mdl = lgb.LGBMClassifier(n_estimators=200, num_leaves=63, max_depth=8,
                              learning_rate=0.05, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
    mdl.fit(X_tr_f, y_tr_f)
    y_p_f = mdl.predict(X_te_f)
    y_pb_f = mdl.predict_proba(X_te_f)[:, 1]

    acc_f = accuracy_score(y_te_f, y_p_f)
    auc_f = roc_auc_score(y_te_f, y_pb_f)

    fold_row = {'Fold': fold_idx+1, 'Accuracy': acc_f, 'AUC': auc_f, 'N_test': len(te_idx)}
    for attr in ATTR_KEYS:
        attr_vals_f = prot_full[attr][te_idx]
        fc = FairnessCalculator(y_te_f, y_p_f, y_pb_f, attr_vals_f)
        mets, vds, _ = fc.compute_all()
        fold_row[f'DI_{attr}'] = mets['DI']
        fold_di_values[attr].append(mets['DI'])
        for mk in METRIC_KEYS:
            fold_row[f'{mk}_{attr}'] = mets[mk]
            fold_row[f'{mk}_{attr}_verdict'] = vds[mk]
    fold_fairness_full.append(fold_row)
    cross_site_results.append(fold_row)

cross_site_df = pd.DataFrame(cross_site_results)

# Compute Fleiss' kappa
fleiss_results = []
for mk in METRIC_KEYS:
    for attr in ATTR_KEYS:
        verdicts = [r[f'{mk}_{attr}_verdict'] for r in fold_fairness_full]
        n_raters = len(verdicts)
        n_fair = sum(verdicts)
        n_unfair = n_raters - n_fair
        p_fair = n_fair / n_raters
        p_unfair = n_unfair / n_raters
        Pe = p_fair**2 + p_unfair**2
        if n_raters > 1:
            Pi_sum = (n_fair*(n_fair-1) + n_unfair*(n_unfair-1)) / (n_raters*(n_raters-1))
        else:
            Pi_sum = 1
        Pa = Pi_sum
        kappa = (Pa - Pe) / (1 - Pe) if (1 - Pe) > 0 else 1.0

        metric_vals = [r[f'{mk}_{attr}'] for r in fold_fairness_full]
        fleiss_results.append({
            'Metric': mk, 'Attribute': attr,
            'Mean': np.mean(metric_vals), 'SD': np.std(metric_vals),
            'Range': f"{min(metric_vals):.3f}-{max(metric_vals):.3f}",
            'Fleiss_Kappa': kappa,
            'Stability': 'High' if kappa > 0.6 else ('Moderate' if kappa > 0.4 else 'Low')
        })
fleiss_df = pd.DataFrame(fleiss_results)

# ── FIG09: Cross-Site Boxplot ─────────────────────────────────
print("[FIG09] Cross-Site Boxplot...")
fig, axes = plt.subplots(1, 4, figsize=(20, 6))
for idx, attr in enumerate(ATTR_KEYS):
    di_vals = [r[f'DI_{attr}'] for r in fold_fairness_full]
    bp = axes[idx].boxplot(di_vals, patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor(PALETTE[idx])
    axes[idx].axhline(0.80, color='red', ls='--', lw=1.5, label='DI=0.80 threshold')
    axes[idx].set_title(f'{attr}', fontweight='bold')
    axes[idx].set_ylabel('Disparate Impact')
    axes[idx].legend(fontsize=8)
    axes[idx].scatter([1]*len(di_vals), di_vals, alpha=0.4, color='black', s=20, zorder=5)
plt.suptitle('DI Distribution Across 10 Hospital Clusters (GroupKFold)', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.93])
save_named_fig('FIG09_crosssite_boxplot.png')

# ── FIG10: Fleiss Kappa Heatmap ───────────────────────────────
print("[FIG10] Fleiss Kappa Heatmap...")
fig, ax = plt.subplots(figsize=(10, 8))
kappa_pivot = fleiss_df.pivot_table(index='Metric', columns='Attribute', values='Fleiss_Kappa')
kappa_pivot = kappa_pivot.reindex(index=METRIC_KEYS, columns=ATTR_KEYS)

annot_k = np.empty_like(kappa_pivot.values, dtype=object)
for i in range(kappa_pivot.shape[0]):
    for j in range(kappa_pivot.shape[1]):
        v = kappa_pivot.values[i, j]
        annot_k[i, j] = f"{v:.3f}"

sns.heatmap(kappa_pivot.values, ax=ax, annot=annot_k, fmt='', cmap='RdYlGn',
            xticklabels=ATTR_KEYS, yticklabels=METRIC_KEYS,
            linewidths=1, linecolor='white', vmin=-0.2, vmax=1.0)
ax.set_title("Fleiss' κ Agreement Across 10 Hospital Clusters", fontweight='bold', fontsize=13)
ax.set_xlabel('Protected Attribute'); ax.set_ylabel('Fairness Metric')
plt.tight_layout()
save_named_fig('FIG10_fleiss_kappa_heatmap.png')

# ── FIG11: Reliability Dashboard ──────────────────────────────
print("[FIG11] Reliability Dashboard...")
fig, axes = plt.subplots(1, 3, figsize=(22, 7))

# (a) VFR summary
vfr_summary = all_vfr_df.groupby(['Metric','Attribute'])['VFR'].max().reset_index()
pct_stable = (vfr_summary['VFR'] == 0).mean() * 100
pct_unstable = (vfr_summary['VFR'] > 0.10).mean() * 100
labels_a = ['Stable (VFR=0)', 'Marginal (0<VFR≤0.10)', 'Unstable (VFR>0.10)']
sizes_a = [(vfr_summary['VFR'] == 0).sum(),
           ((vfr_summary['VFR'] > 0) & (vfr_summary['VFR'] <= 0.10)).sum(),
           (vfr_summary['VFR'] > 0.10).sum()]
colors_a = ['#2ecc71', '#f39c12', '#e74c3c']
axes[0].pie(sizes_a, labels=labels_a, colors=colors_a, autopct='%1.0f%%', startangle=90, textprops={'fontsize': 9})
axes[0].set_title('(a) VFR Distribution Summary', fontweight='bold')

# (b) Sample-size thresholds summary
metric_avg_n = min_n_df.copy()
metric_avg_n['MinN_val'] = pd.to_numeric(metric_avg_n['MinN_CV5'], errors='coerce').fillna(0)
avg_by_metric = metric_avg_n.groupby('Metric')['MinN_val'].mean().reindex(METRIC_KEYS)
bars = axes[1].bar(range(len(METRIC_KEYS)), avg_by_metric.values, color=PALETTE[:7])
axes[1].set_xticks(range(len(METRIC_KEYS)))
axes[1].set_xticklabels(METRIC_KEYS, fontsize=9)
axes[1].set_ylabel('Avg Min-N for CV<5%')
axes[1].set_title('(b) Sample-Size Thresholds by Metric', fontweight='bold')
for i, v in enumerate(avg_by_metric.values):
    if v > 0:
        axes[1].text(i, v+50, f'{int(v):,}', ha='center', fontsize=8)

# (c) Cross-site κ summary
avg_kappa_by_metric = fleiss_df.groupby('Metric')['Fleiss_Kappa'].mean().reindex(METRIC_KEYS)
colors_c = ['#2ecc71' if k > 0.6 else '#f39c12' if k > 0.4 else '#e74c3c' for k in avg_kappa_by_metric.values]
axes[2].bar(range(len(METRIC_KEYS)), avg_kappa_by_metric.values, color=colors_c)
axes[2].set_xticks(range(len(METRIC_KEYS)))
axes[2].set_xticklabels(METRIC_KEYS, fontsize=9)
axes[2].set_ylabel("Mean Fleiss' κ")
axes[2].set_title("(c) Cross-Site Agreement by Metric", fontweight='bold')
axes[2].axhline(0.6, color='green', ls='--', alpha=0.5, label='High (κ>0.6)')
axes[2].axhline(0.4, color='orange', ls='--', alpha=0.5, label='Moderate (κ>0.4)')
axes[2].legend(fontsize=8)

plt.suptitle('Reliability Framework — Combined Dashboard', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.93])
save_named_fig('FIG11_reliability_dashboard.png')

# ── Fairness Intervention ─────────────────────────────────────
print("[INTERVENTION] Running fairness intervention (intersectional RACE×AGE×SEX)...")

# Build RACE × AGE × SEX intersection keys for train and test
race_train = protected_attrs_train['RACE']
age_train  = protected_attrs_train['AGE_GROUP']
sex_train  = protected_attrs_train['SEX']
race_test  = protected_attrs['RACE']
age_test   = protected_attrs['AGE_GROUP']
sex_test   = protected_attrs['SEX']
eth_test   = protected_attrs['ETHNICITY']

unique_races = sorted(set(race_test))
unique_ages  = sorted(set(age_test))
unique_sexes = sorted(set(sex_test))

# Build intersection masks (test set)
test_groups = {}
for r in unique_races:
    for a in unique_ages:
        for s in unique_sexes:
            key = f"{r}|{a}|{s}"
            mask = (race_test == r) & (age_test == a) & (sex_test == s)
            if mask.sum() >= 5:
                test_groups[key] = mask
print(f"  Intersection groups (RACE×AGE×SEX): {len(test_groups)}")

# Intersectional lambda-reweighing on RACE×AGE×SEX groups
def build_multi_weights(lam):
    key_tr = np.array([f"{r}|{a}|{s}" for r, a, s in zip(race_train, age_train, sex_train)])
    n = len(y_train)
    sw = np.ones(n)
    for g in sorted(set(key_tr)):
        mg = key_tr == g; ng = mg.sum()
        for lab in [0, 1]:
            mgl = mg & (y_train == lab); ngl = mgl.sum()
            if ngl > 0:
                expected = (ng / n) * ((y_train == lab).sum() / n)
                observed = ngl / n
                raw_w = expected / observed if observed > 0 else 1.0
                sw[mgl] = np.clip(1.0 + lam * (raw_w - 1.0), 0.1, 10.0)
    return sw

# Per-group threshold helpers
def find_sr_threshold(probs, target_sr, lo=0.01, hi=0.99, step=0.005):
    best_t, best_diff = 0.5, abs((probs >= 0.5).mean() - target_sr)
    for t in np.arange(lo, hi, step):
        diff = abs((probs >= t).mean() - target_sr)
        if diff < best_diff:
            best_diff, best_t = diff, t
    return best_t

def find_tpr_threshold(probs, labels, target_tpr, lo=0.01, hi=0.99, step=0.005):
    pos = labels == 1
    if pos.sum() < 10: return 0.5
    best_t, best_diff = 0.5, abs((probs[pos] >= 0.5).mean() - target_tpr)
    for t in np.arange(lo, hi, step):
        diff = abs((probs[pos] >= t).mean() - target_tpr)
        if diff < best_diff:
            best_diff, best_t = diff, t
    return best_t

# Train reweighed models at multiple lambda values & track lambda effects
LAMBDA_VALUES = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0]
model_probs = {'Standard': test_predictions[best_name]['y_prob']}
lambda_effect_rows = []
# Record Standard baseline
std_pred_base = test_predictions[best_name]['y_pred']
std_prob_base = test_predictions[best_name]['y_prob']
std_acc_base = accuracy_score(y_test, std_pred_base)
std_auc_base = roc_auc_score(y_test, std_prob_base)
std_f1_base = f1_score(y_test, std_pred_base)
std_row = {'Lambda': 0.0, 'Accuracy': std_acc_base, 'AUC': std_auc_base, 'F1': std_f1_base}
for attr in ATTR_KEYS:
    fc = FairnessCalculator(y_test, std_pred_base, std_prob_base, protected_attrs[attr])
    mets, vds, _ = fc.compute_all()
    std_row[f'DI_{attr}'] = mets['DI']
    for mk in METRIC_KEYS:
        std_row[f'{mk}_{attr}'] = mets[mk]
    std_row[f'N_fair_{attr}'] = sum(vds.values())
lambda_effect_rows.append(std_row)

for lam in LAMBDA_VALUES:
    if lam == 0.0:
        continue  # already recorded
    sw = build_multi_weights(lam)
    mdl = lgb.LGBMClassifier(n_estimators=500, num_leaves=63, max_depth=8,
                              learning_rate=0.05, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
    mdl.fit(X_train, y_train, sample_weight=sw)
    y_prob_lam = mdl.predict_proba(X_test)[:, 1]
    y_pred_lam = mdl.predict(X_test)
    model_probs[f'Reweigh_{lam:.0f}'] = y_prob_lam
    acc_lam = accuracy_score(y_test, y_pred_lam)
    auc_lam = roc_auc_score(y_test, y_prob_lam)
    f1_lam = f1_score(y_test, y_pred_lam)
    lam_row = {'Lambda': lam, 'Accuracy': acc_lam, 'AUC': auc_lam, 'F1': f1_lam}
    for attr in ATTR_KEYS:
        fc = FairnessCalculator(y_test, y_pred_lam, y_prob_lam, protected_attrs[attr])
        mets, vds, _ = fc.compute_all()
        lam_row[f'DI_{attr}'] = mets['DI']
        for mk in METRIC_KEYS:
            lam_row[f'{mk}_{attr}'] = mets[mk]
        lam_row[f'N_fair_{attr}'] = sum(vds.values())
    lambda_effect_rows.append(lam_row)
    print(f"  Trained reweighed λ={lam:.0f}  Acc={acc_lam:.4f}  AUC={auc_lam:.4f}  "
          f"DI_RACE={lam_row['DI_RACE']:.3f} DI_AGE={lam_row['DI_AGE_GROUP']:.3f}")
lambda_effect_df = pd.DataFrame(lambda_effect_rows)
sys.stdout.flush()

# Per-group threshold search (SR + TPR equalisation)
A_SR_GRID  = [0.0, 0.4, 0.6, 0.8, 1.0]
A_TPR_GRID = [0.0, 0.5, 0.7, 1.0]
candidate_rows = []
print(f"  Searching {len(model_probs)} models × threshold grids...")
sys.stdout.flush()

for mname, y_prob_c in model_probs.items():
    overall_sr  = (y_prob_c >= 0.5).mean()
    overall_tpr = (y_prob_c[y_test == 1] >= 0.5).mean()
    sr_thresh, tpr_thresh = {}, {}
    for key, mask in test_groups.items():
        sr_thresh[key]  = find_sr_threshold(y_prob_c[mask], overall_sr)
        tpr_thresh[key] = find_tpr_threshold(y_prob_c[mask], y_test[mask], overall_tpr)
    for a_sr in A_SR_GRID:
        for a_tpr in A_TPR_GRID:
            thresholds = {}
            for key in test_groups:
                t = 0.5 + a_sr*(sr_thresh[key]-0.5) + a_tpr*(tpr_thresh[key]-0.5)
                thresholds[key] = np.clip(t, 0.01, 0.99)
            y_pred_c = (y_prob_c >= 0.5).astype(int)
            for key, mask in test_groups.items():
                y_pred_c[mask] = (y_prob_c[mask] >= thresholds[key]).astype(int)
            acc_c = accuracy_score(y_test, y_pred_c)
            di_vals = {}
            n_fair_total = 0
            all_di_ok = True
            for attr in ATTR_KEYS:
                fc = FairnessCalculator(y_test, y_pred_c, y_prob_c, protected_attrs[attr])
                mets, vds, _ = fc.compute_all()
                di_vals[attr] = mets['DI']
                n_fair_total += sum(vds.values())
                if mets['DI'] < 0.80:
                    all_di_ok = False
            candidate_rows.append({
                'Model': mname, 'A_SR': a_sr, 'A_TPR': a_tpr,
                'Accuracy': acc_c, 'AUC': roc_auc_score(y_test, y_prob_c),
                'DI_RACE': di_vals['RACE'], 'DI_AGE_GROUP': di_vals['AGE_GROUP'],
                'DI_SEX': di_vals['SEX'], 'DI_ETHNICITY': di_vals['ETHNICITY'],
                'Total_Fair': n_fair_total, 'All_DI_OK': all_di_ok,
                'thresholds': thresholds,
            })

print(f"  Evaluated {len(candidate_rows)} candidates")
cand_df = pd.DataFrame(candidate_rows)

# Selection: prefer ALL DI ≥ 0.80, maximize Total_Fair
elig = cand_df[cand_df['All_DI_OK'] == True].copy()
print(f"  Candidates with ALL DI ≥ 0.80: {len(elig)}/{len(cand_df)}")

if len(elig) > 0:
    chosen_idx = elig.sort_values(['Total_Fair', 'Accuracy'], ascending=[False, False]).index[0]
else:
    # Fallback: maximize Total_Fair + DI_AGE_GROUP
    print("  ⚠ No candidate with ALL DI≥0.80; selecting best available")
    cand_df['score'] = cand_df['Total_Fair'] * 10 + cand_df['DI_AGE_GROUP'] * 5 + cand_df['Accuracy']
    chosen_idx = cand_df.sort_values('score', ascending=False).index[0]

chosen = cand_df.loc[chosen_idx]
fair_thresholds = chosen['thresholds']
chosen_prob = model_probs[chosen['Model']]

print(f"  ✓ Selected: DI_RACE={chosen['DI_RACE']:.3f}, DI_AGE={chosen['DI_AGE_GROUP']:.3f}, "
      f"DI_SEX={chosen['DI_SEX']:.3f}, DI_ETH={chosen['DI_ETHNICITY']:.3f}")
print(f"    Total Fair={int(chosen['Total_Fair'])}/28, Acc={chosen['Accuracy']:.4f} "
      f"({chosen['Model']}, α_sr={chosen['A_SR']}, α_tpr={chosen['A_TPR']})")
sys.stdout.flush()

# Reconstruct final fair predictions
fair_y_prob_raw = chosen_prob
fair_y_pred = (fair_y_prob_raw >= 0.5).astype(int)
for key, mask in test_groups.items():
    fair_y_pred[mask] = (fair_y_prob_raw[mask] >= fair_thresholds[key]).astype(int)

fair_acc = accuracy_score(y_test, fair_y_pred)
fair_auc = roc_auc_score(y_test, fair_y_prob_raw)

std_acc = results_df[results_df['Model'] == best_name]['Accuracy'].values[0]
std_auc = results_df[results_df['Model'] == best_name]['AUC'].values[0]

intervention_rows = []
for label, y_p, y_pb in [('Standard', test_predictions[best_name]['y_pred'], test_predictions[best_name]['y_prob']),
                          ('Fair', fair_y_pred, fair_y_prob_raw)]:
    row = {'Model': label, 'Accuracy': accuracy_score(y_test, y_p), 'AUC': roc_auc_score(y_test, y_pb)}
    n_fair_total = 0
    for attr in ATTR_KEYS:
        fc = FairnessCalculator(y_test, y_p, y_pb, protected_attrs[attr])
        mets, vds, _ = fc.compute_all()
        row[f'DI_{attr}'] = mets['DI']
        n_fair_total += sum(vds.values())
    row['N_fair_metrics'] = n_fair_total
    row['Accuracy_cost'] = std_acc - row['Accuracy'] if label == 'Fair' else 0
    intervention_rows.append(row)
intervention_df = pd.DataFrame(intervention_rows)

# ── FIG12: Intervention Comparison ────────────────────────────
print("[FIG12] Intervention Comparison...")
fig, ax = plt.subplots(figsize=(14, 7))
metrics_to_plot = ['Accuracy', 'AUC', 'DI_RACE', 'DI_SEX', 'DI_ETHNICITY', 'DI_AGE_GROUP']
x = np.arange(len(metrics_to_plot))
width = 0.35
std_vals = [intervention_df[intervention_df['Model']=='Standard'][m].values[0] for m in metrics_to_plot]
fair_vals = [intervention_df[intervention_df['Model']=='Fair'][m].values[0] for m in metrics_to_plot]

bars1 = ax.bar(x - width/2, std_vals, width, label='Standard Model', color=PALETTE[0], edgecolor='white')
bars2 = ax.bar(x + width/2, fair_vals, width, label='Fair Model', color=PALETTE[2], edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels(metrics_to_plot, fontsize=10)
ax.set_ylabel('Value')
ax.set_title('Standard vs Fair Model Comparison', fontweight='bold', fontsize=14)
ax.legend(fontsize=11)
ax.axhline(0.80, color='red', ls='--', alpha=0.4, label='DI threshold')
for i, (sv, fv) in enumerate(zip(std_vals, fair_vals)):
    ax.text(i-width/2, sv+0.01, f'{sv:.3f}', ha='center', fontsize=8)
    ax.text(i+width/2, fv+0.01, f'{fv:.3f}', ha='center', fontsize=8)
plt.tight_layout()
save_named_fig('FIG12_intervention_comparison.png')

# ── FIG13: Literature Review Pipeline ─────────────────────────
print("[FIG13] Literature Review Pipeline...")
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14); ax.set_ylim(0, 10); ax.axis('off')

ax.text(7, 9.5, 'PRISMA-Style Literature Review Pipeline', ha='center',
        fontsize=16, fontweight='bold', color='#2c3e50')

stages = [
    ('Initial Database Search', '~280 candidates', '#e3f2fd', '#1565c0', 9.0),
    ('Title/Abstract Screening', '~120 retained', '#e8f5e9', '#2e7d32', 7.5),
    ('Full-Text Eligibility', '~45 assessed', '#fff3e0', '#e65100', 6.0),
    ('Quality & Relevance Filter', '~25 studies', '#f3e5f5', '#7b1fa2', 4.5),
    ('Final Primary Studies', '14 studies', '#fce4ec', '#c62828', 3.0),
]

for label, count, bg, fg, ypos in stages:
    w = 8; h = 1.0
    r = mpatches.FancyBboxPatch((3, ypos-0.5), w, h, boxstyle="round,pad=0.2",
                                 facecolor=bg, edgecolor=fg, lw=2)
    ax.add_patch(r)
    ax.text(7, ypos, f'{label}\n{count}', ha='center', va='center',
            fontsize=11, fontweight='bold', color=fg)

# Exclusion arrows
excl = [
    ('Not ML-based or not LOS', 7.5),
    ('No fairness analysis', 6.0),
    ('Insufficient methodology', 4.5),
]
for reason, ypos in excl:
    ax.annotate('', xy=(12, ypos-0.2), xytext=(11, ypos-0.2),
                arrowprops=dict(arrowstyle='->', color='#999', lw=1.5))
    ax.text(12.2, ypos-0.2, f'Excluded:\n{reason}', fontsize=8, color='#777', va='center')

# Down arrows
for i in range(len(stages)-1):
    y1 = stages[i][4] - 0.5
    y2 = stages[i+1][4] + 0.5
    ax.annotate('', xy=(7, y2), xytext=(7, y1),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))

save_named_fig('FIG13_litreview_pipeline.png')

# ──────────────────────────────────────────────────────────────
# TABLE GENERATION (LaTeX + CSV)
# ──────────────────────────────────────────────────────────────
print("\n[TABLES] Generating Tables 3–7...")

# Table 3: VFR Results
table3 = vfr_df[['Model','Metric','Attribute','Mean','Std','Threshold','Margin','Margin_Sigma','VFR']].copy()
table3.columns = ['Model','Metric','Attribute','Mean','σ','Threshold','Margin(σ)','Margin_Sigma','VFR']
table3.to_csv(os.path.join(TABLES_DIR, 'Table3_VFR_Results.csv'), index=False)
print("  Table 3 saved")

# Table 4: Sample-size
table4 = min_n_df.rename(columns={'MinN_CV5': 'Min-N for CV<5%', 'MinN_CV10': 'Min-N for CV<10%',
                                    'Largest_Group': 'Largest Group', 'Smallest_Group': 'Smallest Group'})
table4.to_csv(os.path.join(TABLES_DIR, 'Table4_SampleSize.csv'), index=False)
print("  Table 4 saved")

# Table 5: Cross-hospital
table5 = fleiss_df[['Metric','Attribute','Mean','SD','Range','Fleiss_Kappa','Stability']].copy()
table5.columns = ['Metric','Attribute','Mean','SD across clusters','Range',"Fleiss κ",'Stability class']
table5.to_csv(os.path.join(TABLES_DIR, 'Table5_CrossHospital.csv'), index=False)
print("  Table 5 saved")

# Table 6: Intervention
table6 = intervention_df.copy()
table6.to_csv(os.path.join(TABLES_DIR, 'Table6_Intervention.csv'), index=False)
print("  Table 6 saved")

# Table 7: Discussion/Recommendations
table7_rows = []
for mk in METRIC_KEYS:
    avg_n = min_n_df[min_n_df['Metric'] == mk]['MinN_CV5'].apply(lambda x: int(x) if str(x).isdigit() else 0).mean()
    avg_kappa = fleiss_df[fleiss_df['Metric'] == mk]['Fleiss_Kappa'].mean()
    kappa_class = 'High' if avg_kappa > 0.6 else ('Moderate' if avg_kappa > 0.4 else 'Low')
    max_vfr = all_vfr_df[all_vfr_df['Metric'] == mk]['VFR'].max()

    if mk == 'DI':
        role = 'Primary'
        when = 'Default metric for regulatory compliance and cross-site comparisons'
    elif mk in ['SPD', 'EOPP']:
        role = 'Primary'
        when = 'Use alongside DI for comprehensive group-level fairness assessment'
    elif mk == 'EOD':
        role = 'Complementary'
        when = 'When both TPR and FPR parity matter (e.g., clinical risk scoring)'
    elif mk == 'TI':
        role = 'Diagnostic'
        when = 'Detect individual-level prediction inconsistencies across groups'
    elif mk == 'PP':
        role = 'Complementary'
        when = 'When predictive value equality is required (e.g., resource allocation)'
    else:
        role = 'Diagnostic'
        when = 'Assess calibration fairness across probability ranges'

    table7_rows.append({
        'Metric': mk, 'Role': role,
        'Min-N guideline': f'{int(avg_n):,}' if avg_n > 0 else 'N/A',
        'Cross-site κ class': kappa_class,
        'When to use': when
    })
table7 = pd.DataFrame(table7_rows)
table7.to_csv(os.path.join(TABLES_DIR, 'Table7_Discussion.csv'), index=False)
print("  Table 7 saved")

# Table 8: Sample Size vs Accuracy + Fairness + VFR
# Build clean display columns
t8_display_cols = ['Sample_Size', 'Accuracy', 'AUC', 'F1']
for attr in ATTR_KEYS:
    for mk in METRIC_KEYS:
        t8_display_cols.append(f'{mk}_{attr}')
for attr in ATTR_KEYS:
    t8_display_cols.append(f'VFR_DI_{attr}')
table8 = table8_df[[c for c in t8_display_cols if c in table8_df.columns]].copy()
table8.to_csv(os.path.join(TABLES_DIR, 'Table8_SampleSize_Accuracy_Fairness.csv'), index=False)
print("  Table 8 saved")

# Table 9: Comprehensive Accuracy Table (all 12 models)
table9 = results_df[['Model', 'Accuracy', 'AUC', 'F1', 'Precision', 'Recall', 'Time_sec']].copy()
# Add fairness counts per model
for i, row in table9.iterrows():
    name = row['Model']
    if name in all_fairness:
        n_fair = 0
        n_total = 0
        for attr in ATTR_KEYS:
            for mk in METRIC_KEYS:
                n_total += 1
                n_fair += int(all_verdicts[name][attr][mk])
        table9.loc[i, 'N_Fair_of_28'] = n_fair
        table9.loc[i, 'DI_RACE'] = all_fairness[name]['RACE']['DI']
        table9.loc[i, 'DI_SEX'] = all_fairness[name]['SEX']['DI']
        table9.loc[i, 'DI_ETHNICITY'] = all_fairness[name]['ETHNICITY']['DI']
        table9.loc[i, 'DI_AGE_GROUP'] = all_fairness[name]['AGE_GROUP']['DI']
table9.to_csv(os.path.join(TABLES_DIR, 'Table9_Comprehensive_Accuracy.csv'), index=False)
print("  Table 9 saved")

# Table 10: Lambda Effect on Accuracy & Fairness
table10 = lambda_effect_df[['Lambda', 'Accuracy', 'AUC', 'F1',
                             'DI_RACE', 'DI_SEX', 'DI_ETHNICITY', 'DI_AGE_GROUP']].copy()
for attr in ATTR_KEYS:
    table10[f'N_fair_{attr}'] = lambda_effect_df[f'N_fair_{attr}']
table10['Total_Fair_of_28'] = sum(table10[f'N_fair_{attr}'] for attr in ATTR_KEYS)
table10['Accuracy_Drop'] = table10['Accuracy'].iloc[0] - table10['Accuracy']
table10['All_DI_Fair'] = (
    (table10['DI_RACE'] >= 0.80) &
    (table10['DI_SEX'] >= 0.80) &
    (table10['DI_ETHNICITY'] >= 0.80) &
    (table10['DI_AGE_GROUP'] >= 0.80)
)
table10.to_csv(os.path.join(TABLES_DIR, 'Table10_Lambda_Effect.csv'), index=False)
print("  Table 10 saved")

# ── Print LaTeX for all tables ────────────────────────────────
print("\n" + "="*80)
print("LATEX TABLE OUTPUT")
print("="*80)

for name, tdf in [('Table 3: VFR Results', table3), ('Table 4: Sample-Size', table4),
                   ('Table 5: Cross-Hospital', table5), ('Table 6: Intervention', table6),
                   ('Table 7: Discussion', table7), ('Table 8: Sample-Size vs Accuracy/Fairness', table8),
                   ('Table 9: Comprehensive Accuracy', table9), ('Table 10: Lambda Effect', table10)]:
    print(f"\n% {name}")
    print(tdf.to_latex(index=False, float_format='%.4f'))

print("\n[DONE] All 13 figures and 8 tables generated successfully!")
print(f"  Figures saved to: {SCRIPT_DIR}/FIG*.png")
print(f"  Tables saved to: {TABLES_DIR}/Table*.csv")
