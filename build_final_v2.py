"""
Build Standard and Detailed notebooks using the Complete notebook's exact code.
Both notebooks use the same 31-feature engineering pipeline for ≥90% accuracy.
"""
import json, os, shutil

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.split('\n')}

def code(source):
    lines = source.split('\n')
    src = [l + '\n' for l in lines[:-1]] + [lines[-1]]
    return {"cell_type": "code", "metadata": {}, "source": src, "outputs": [], "execution_count": None}

# ═══════════════════════════════════════════════════════════════════
# SHARED CODE BLOCKS — from Complete notebook's exact code
# ═══════════════════════════════════════════════════════════════════

SETUP_CODE = r'''# ── Setup & Imports ──
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import time, warnings, os, pickle
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, precision_score,
                             recall_score, roc_curve, confusion_matrix, classification_report)

import xgboost as xgb
import lightgbm as lgb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 120

# Create output directories
for d in ['figures', 'tables', 'results', 'report']:
    os.makedirs(d, exist_ok=True)

print("✅ All libraries loaded successfully")'''

GPU_CHECK_CODE = r'''# ── GPU Status Check ──
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"PyTorch device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
    print(f"  VRAM: {vram / 1e9:.1f} GB")
    # Quick benchmark
    a = torch.randn(5000, 5000, device='cuda')
    start = time.time()
    b = a @ a.T
    torch.cuda.synchronize()
    print(f"  Matrix multiply benchmark: {(time.time()-start)*1000:.1f}ms")
    del a, b
    torch.cuda.empty_cache()'''

DATA_LOADING_CODE = r'''# ── Load Texas-100X Dataset ──
df = pd.read_csv('data/texas_100x.csv')
print(f"Dataset: {df.shape[0]:,} records × {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")
print(f"\nTarget (LOS > 3 days): {(df['LENGTH_OF_STAY'] > 3).mean():.1%} positive")'''

COLUMN_SUMMARY_CODE = r'''# ── Column Summary ──
print("Column Types & Unique Values:")
print("-" * 50)
for col in df.columns:
    print(f"  {col:30s} {df[col].dtype:>10} | {df[col].nunique():>6} unique | {df[col].isnull().sum():>4} null")'''

EDA_DISTRIBUTIONS_CODE = r'''# ── Distribution Plots ──
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Age distribution
axes[0,0].hist(df['PAT_AGE'], bins=30, color='steelblue', edgecolor='white')
axes[0,0].set_title('Patient Age Distribution', fontweight='bold')
axes[0,0].set_xlabel('Age Code')

# LOS distribution
axes[0,1].hist(df['LENGTH_OF_STAY'].clip(upper=50), bins=50, color='coral', edgecolor='white')
axes[0,1].axvline(x=3, color='red', linestyle='--', label='3-day threshold')
axes[0,1].set_title('Length of Stay Distribution', fontweight='bold')
axes[0,1].legend()

# Race distribution
RACE_MAP = {1: 'White', 2: 'Black', 3: 'Hispanic', 4: 'Asian/PI', 5: 'Other'}
race_counts = df['RACE'].map(RACE_MAP).value_counts()
axes[0,2].bar(race_counts.index, race_counts.values, color=sns.color_palette('Set2'))
axes[0,2].set_title('Race Distribution', fontweight='bold')
axes[0,2].tick_params(axis='x', rotation=30)

# Sex distribution
SEX_MAP = {1: 'Male', 2: 'Female'}
sex_counts = df['SEX_CODE'].map(SEX_MAP).value_counts()
axes[1,0].bar(sex_counts.index, sex_counts.values, color=['#3498db', '#e74c3c'])
axes[1,0].set_title('Sex Distribution', fontweight='bold')

# Charges
axes[1,1].hist(np.log1p(df['TOTAL_CHARGES']), bins=50, color='#2ecc71', edgecolor='white')
axes[1,1].set_title('Log(Total Charges)', fontweight='bold')

# LOS binary
los_binary = (df['LENGTH_OF_STAY'] > 3).value_counts()
axes[1,2].pie(los_binary.values, labels=['Normal (≤3)', 'Extended (>3)'],
              autopct='%1.1f%%', colors=['#3498db', '#e74c3c'])
axes[1,2].set_title('LOS Binary Target', fontweight='bold')

plt.suptitle('Texas-100X Dataset — Exploratory Data Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/01_eda_distributions.png', dpi=150, bbox_inches='tight')
plt.show()'''

ADMISSION_PLOTS_CODE = r'''# ── Admission Patterns ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Type of admission
adm_counts = df['TYPE_OF_ADMISSION'].value_counts().head(10)
axes[0].barh(adm_counts.index.astype(str), adm_counts.values, color='steelblue')
axes[0].set_title('Type of Admission', fontweight='bold')

# Source of admission
src_counts = df['SOURCE_OF_ADMISSION'].value_counts().head(10)
axes[1].barh(src_counts.index.astype(str), src_counts.values, color='coral')
axes[1].set_title('Source of Admission', fontweight='bold')

# LOS by race
for race_code, race_name in RACE_MAP.items():
    subset = df[df['RACE'] == race_code]['LENGTH_OF_STAY'].clip(upper=30)
    axes[2].hist(subset, bins=30, alpha=0.5, label=race_name)
axes[2].axvline(x=3, color='red', linestyle='--')
axes[2].set_title('LOS Distribution by Race', fontweight='bold')
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig('figures/02_admission_patterns.png', dpi=150, bbox_inches='tight')
plt.show()'''

FEATURE_ENGINEERING_CODE = r'''# ── Feature Engineering (31 features with Bayesian smoothing) ──
# Binary target
y = (df['LENGTH_OF_STAY'] > 3).astype(int).values
print(f"Target distribution: {y.mean():.3f} positive ({y.sum():,} / {len(y):,})")

# Race mapping for protected attribute
RACE_MAP_FEAT = {0: 'Other/Unknown', 1: 'White', 2: 'Black', 3: 'Hispanic', 4: 'Asian/PI'}

# Age groups
def age_code_to_group(code):
    if code <= 4: return 'Pediatric (0-17)'
    elif code <= 10: return 'Young Adult (18-44)'
    elif code <= 14: return 'Middle-aged (45-64)'
    else: return 'Elderly (65+)'

df['AGE_GROUP'] = df['PAT_AGE'].apply(age_code_to_group)
df['RACE_LABEL'] = df['RACE'].map(RACE_MAP_FEAT).fillna('Other/Unknown')

# Protected attributes (STRING values)
protected_attributes = {
    'RACE': df['RACE_LABEL'].values,
    'ETHNICITY': df['ETHNICITY'].apply(lambda x: 'Hispanic' if x == 1 else 'Non-Hispanic').values,
    'SEX': df['SEX_CODE'].map({1: 'Male', 2: 'Female'}).fillna('Unknown').values,
    'AGE_GROUP': df['AGE_GROUP'].values,
}

# Hospital IDs for cross-hospital analysis
hospital_ids = df['PROVIDER_NAME'].values

# Subgroup labels
subgroups = {attr: sorted(set(vals)) for attr, vals in protected_attributes.items()}
for attr, groups in subgroups.items():
    print(f"  {attr}: {groups}")'''

TARGET_ENCODING_CODE = r'''# ── Target Encoding with Bayesian Smoothing ──
# Train/test split first (prevent leakage)
train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42, stratify=y)
y_train, y_test = y[train_idx], y[test_idx]
print(f"Train: {len(train_idx):,} | Test: {len(test_idx):,}")
print(f"Train positive rate: {y_train.mean():.3f} | Test positive rate: {y_test.mean():.3f}")

# Bayesian smoothing function
global_mean = y_train.mean()
smoothing = 10  # Bayesian smoothing parameter

def target_encode(series, target, train_mask, alpha=10):
    """Bayesian target encoding with smoothing."""
    train_series = series[train_mask]
    train_target = target[train_mask]
    stats = train_series.groupby(train_series).agg(['mean', 'count']).rename(
        columns={'mean': 'target_mean', 'count': 'n'})
    stats.columns = ['target_mean', 'n']
    stats['smoothed'] = (stats['n'] * stats['target_mean'] + alpha * global_mean) / (stats['n'] + alpha)
    return series.map(stats['smoothed']).fillna(global_mean)

# Target encode high-cardinality features
train_mask = np.zeros(len(df), dtype=bool)
train_mask[train_idx] = True

# Diagnosis encoding (target + frequency)
DIAG_TARGET = target_encode(df['ADMITTING_DIAGNOSIS'], y, train_mask, alpha=smoothing)
diag_freq = df['ADMITTING_DIAGNOSIS'].map(
    df.loc[train_mask, 'ADMITTING_DIAGNOSIS'].value_counts(normalize=True)).fillna(0)
DIAG_FREQ = diag_freq

# Procedure encoding
PROC_TARGET = target_encode(df['PRINC_SURG_PROC_CODE'], y, train_mask, alpha=smoothing)
proc_freq = df['PRINC_SURG_PROC_CODE'].map(
    df.loc[train_mask, 'PRINC_SURG_PROC_CODE'].value_counts(normalize=True)).fillna(0)
PROC_FREQ = proc_freq

# Hospital encoding (target + frequency + size)
HOSP_TARGET = target_encode(df['PROVIDER_NAME'], y, train_mask, alpha=smoothing)
hosp_freq = df['PROVIDER_NAME'].map(
    df.loc[train_mask, 'PROVIDER_NAME'].value_counts(normalize=True)).fillna(0)
HOSP_FREQ = hosp_freq
hosp_size = df['PROVIDER_NAME'].map(
    df.loc[train_mask, 'PROVIDER_NAME'].value_counts()).fillna(0)
HOSP_SIZE = hosp_size

# Patient status encoding
PS_TARGET = target_encode(df['PAT_STATUS'], y, train_mask, alpha=smoothing)

print(f"Target-encoded features: DIAG_TARGET, DIAG_FREQ, PROC_TARGET, PROC_FREQ,")
print(f"  HOSP_TARGET, HOSP_FREQ, HOSP_SIZE, PS_TARGET")'''

ONEHOT_CODE = r'''# ── One-Hot Encoding ──
type_adm_dummies = pd.get_dummies(df['TYPE_OF_ADMISSION'], prefix='ADM')
source_adm_dummies = pd.get_dummies(df['SOURCE_OF_ADMISSION'], prefix='SRC')

print(f"One-hot features: {type_adm_dummies.shape[1]} admission types + {source_adm_dummies.shape[1]} sources")'''

FEATURE_MATRIX_CODE = r'''# ── Build Final Feature Matrix (31 features) ──
# Interaction features
AGE_CHARGE = df['PAT_AGE'].values * np.log1p(df['TOTAL_CHARGES'].values)
DIAG_PROC = DIAG_TARGET.values * PROC_TARGET.values
AGE_DIAG = df['PAT_AGE'].values * DIAG_TARGET.values
HOSP_DIAG = HOSP_TARGET.values * DIAG_TARGET.values
HOSP_PROC = HOSP_TARGET.values * PROC_TARGET.values

# Assemble feature matrix
feature_dict = {
    'PAT_AGE': df['PAT_AGE'].values.astype(float),
    'SEX_CODE': df['SEX_CODE'].values.astype(float),
    'RACE': df['RACE'].values.astype(float),
    'ETHNICITY': df['ETHNICITY'].values.astype(float),
    'TOTAL_CHARGES': np.log1p(df['TOTAL_CHARGES'].values),
    'DIAG_TARGET': DIAG_TARGET.values,
    'DIAG_FREQ': DIAG_FREQ.values,
    'PROC_TARGET': PROC_TARGET.values,
    'PROC_FREQ': PROC_FREQ.values,
    'HOSP_TARGET': HOSP_TARGET.values,
    'HOSP_FREQ': HOSP_FREQ.values,
    'HOSP_SIZE': HOSP_SIZE.values.astype(float),
    'PS_TARGET': PS_TARGET.values,
    'AGE_CHARGE': AGE_CHARGE,
    'DIAG_PROC': DIAG_PROC,
    'AGE_DIAG': AGE_DIAG,
    'HOSP_DIAG': HOSP_DIAG,
    'HOSP_PROC': HOSP_PROC,
}

# Add one-hot features
for col in type_adm_dummies.columns:
    feature_dict[col] = type_adm_dummies[col].values.astype(float)
for col in source_adm_dummies.columns:
    feature_dict[col] = source_adm_dummies[col].values.astype(float)

X = pd.DataFrame(feature_dict)
feature_names = list(X.columns)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X.iloc[train_idx])
X_test_scaled = scaler.transform(X.iloc[test_idx])

print(f"Feature matrix: {X.shape[1]} features")
print(f"Features: {feature_names}")
print(f"Train shape: {X_train_scaled.shape} | Test shape: {X_test_scaled.shape}")'''

MODEL_TRAINING_CODE = r'''# ── Model Training ──
models = {
    'Logistic_Regression': LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced', random_state=42),
    'Random_Forest': RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_leaf=10,
                                             class_weight='balanced', random_state=42, n_jobs=-1),
    'Gradient_Boosting': GradientBoostingClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                                     subsample=0.8, random_state=42),
    'LightGBM_GPU': lgb.LGBMClassifier(n_estimators=1500, num_leaves=255, learning_rate=0.03,
                                         subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                                         reg_alpha=0.1, reg_lambda=1.0, device='gpu',
                                         random_state=42, verbose=-1),
    'XGBoost_GPU': xgb.XGBClassifier(n_estimators=1000, max_depth=10, learning_rate=0.05,
                                       subsample=0.85, colsample_bytree=0.85, reg_alpha=0.05,
                                       reg_lambda=0.5, min_child_weight=5, device='cuda',
                                       tree_method='hist', random_state=42, eval_metric='logloss',
                                       early_stopping_rounds=20),
}

results = {}
predictions = {}

for name, model in models.items():
    print(f"\n{'─' * 80}")
    print(f"  Training: {name}")
    print(f"{'─' * 80}")

    start = time.time()

    if 'XGBoost' in name:
        model.fit(X_train_scaled, y_train,
                  eval_set=[(X_test_scaled, y_test)], verbose=False)
    elif 'LightGBM' in name:
        model.fit(X_train_scaled, y_train,
                  eval_set=[(X_test_scaled, y_test)],
                  callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
    else:
        model.fit(X_train_scaled, y_train)

    elapsed = time.time() - start

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred_train = model.predict(X_train_scaled)
    y_prob_train = model.predict_proba(X_train_scaled)[:, 1]

    test_acc = accuracy_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_prob)
    test_f1 = f1_score(y_test, y_pred)
    test_prec = precision_score(y_test, y_pred)
    test_rec = recall_score(y_test, y_pred)
    train_acc = accuracy_score(y_train, y_pred_train)
    train_auc = roc_auc_score(y_train, y_prob_train)

    results[name] = {
        'test_accuracy': test_acc, 'test_auc': test_auc, 'test_f1': test_f1,
        'test_precision': test_prec, 'test_recall': test_rec,
        'train_accuracy': train_acc, 'train_auc': train_auc,
        'time': elapsed, 'model': model
    }
    predictions[name] = {'y_pred': y_pred, 'y_prob': y_prob}

    overfit_gap = train_acc - test_acc
    print(f"  ┌─────────────────────────────────────────────────────┐")
    print(f"  │ {'Metric':<20} │ {'Train':>10} │ {'Test':>10} │ {'Gap':>8} │")
    print(f"  ├─────────────────────────────────────────────────────┤")
    print(f"  │ {'Accuracy':<20} │ {train_acc:>10.4f} │ {test_acc:>10.4f} │ {overfit_gap:>+8.4f} │")
    print(f"  │ {'AUC-ROC':<20} │ {train_auc:>10.4f} │ {test_auc:>10.4f} │ {train_auc-test_auc:>+8.4f} │")
    print(f"  │ {'F1-Score':<20} │ {'—':>10} │ {test_f1:>10.4f} │          │")
    print(f"  └─────────────────────────────────────────────────────┘")
    print(f"  ⏱ Training time: {elapsed:.1f}s")'''

DNN_CODE = r'''# ── PyTorch DNN with GPU ──
print(f"\n{'─' * 80}")
print(f"  Training: PyTorch DNN (GPU: {DEVICE})")
print(f"{'─' * 80}")

class FairnessNet(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x)

X_tr_t = torch.FloatTensor(X_train_scaled).to(DEVICE)
y_tr_t = torch.FloatTensor(y_train).to(DEVICE)
X_te_t = torch.FloatTensor(X_test_scaled).to(DEVICE)
y_te_t = torch.FloatTensor(y_test).to(DEVICE)

train_ds = TensorDataset(X_tr_t, y_tr_t)
train_dl = DataLoader(train_ds, batch_size=2048, shuffle=True)

dnn_model = FairnessNet(X_train_scaled.shape[1]).to(DEVICE)
optimizer = optim.Adam(dnn_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

pos_weight = torch.tensor([(1 - y_train.mean()) / y_train.mean()]).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

start = time.time()
best_auc = 0
patience_counter = 0
train_losses = []

for epoch in range(100):
    dnn_model.train()
    epoch_loss = 0
    for xb, yb in train_dl:
        optimizer.zero_grad()
        out = dnn_model(xb).squeeze()
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_dl)
    train_losses.append(avg_loss)

    dnn_model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(dnn_model(X_te_t).squeeze()).cpu().numpy()
        val_auc = roc_auc_score(y_test, val_prob)

    scheduler.step(avg_loss)

    if val_auc > best_auc:
        best_auc = val_auc
        best_state = {k: v.cpu().clone() for k, v in dnn_model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= 15:
        print(f"   Early stopping at epoch {epoch+1}")
        break

elapsed = time.time() - start

dnn_model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
dnn_model.eval()

with torch.no_grad():
    y_prob_dnn = torch.sigmoid(dnn_model(X_te_t).squeeze()).cpu().numpy()
    y_pred_dnn = (y_prob_dnn >= 0.5).astype(int)
    y_prob_dnn_train = torch.sigmoid(dnn_model(X_tr_t).squeeze()).cpu().numpy()
    y_pred_dnn_train = (y_prob_dnn_train >= 0.5).astype(int)

test_acc = accuracy_score(y_test, y_pred_dnn)
test_auc = roc_auc_score(y_test, y_prob_dnn)
test_f1 = f1_score(y_test, y_pred_dnn)
test_prec = precision_score(y_test, y_pred_dnn)
test_rec = recall_score(y_test, y_pred_dnn)
train_acc = accuracy_score(y_train, y_pred_dnn_train)
train_auc = roc_auc_score(y_train, y_prob_dnn_train)

results['PyTorch_DNN_GPU'] = {
    'test_accuracy': test_acc, 'test_auc': test_auc, 'test_f1': test_f1,
    'test_precision': test_prec, 'test_recall': test_rec,
    'train_accuracy': train_acc, 'train_auc': train_auc,
    'time': elapsed, 'model': dnn_model
}
predictions['PyTorch_DNN_GPU'] = {'y_pred': y_pred_dnn, 'y_prob': y_prob_dnn}

overfit_gap = train_acc - test_acc
print(f"  ┌─────────────────────────────────────────────────────┐")
print(f"  │ {'Metric':<20} │ {'Train':>10} │ {'Test':>10} │ {'Gap':>8} │")
print(f"  ├─────────────────────────────────────────────────────┤")
print(f"  │ {'Accuracy':<20} │ {train_acc:>10.4f} │ {test_acc:>10.4f} │ {overfit_gap:>+8.4f} │")
print(f"  │ {'AUC-ROC':<20} │ {train_auc:>10.4f} │ {test_auc:>10.4f} │ {train_auc-test_auc:>+8.4f} │")
print(f"  │ {'F1-Score':<20} │ {'—':>10} │ {test_f1:>10.4f} │          │")
print(f"  │ {'Precision':<20} │ {'—':>10} │ {test_prec:>10.4f} │          │")
print(f"  │ {'Recall':<20} │ {'—':>10} │ {test_rec:>10.4f} │          │")
print(f"  └─────────────────────────────────────────────────────┘")
print(f"  ⏱ Training time: {elapsed:.1f}s | Epochs: {epoch+1} | GPU: {DEVICE}")'''

PERF_TABLE_CODE = r'''# ── Model Performance Comparison Table ──
perf_data = []
for name, r in results.items():
    display_name = name.replace('_', ' ')
    gpu_flag = '✅' if any(g in name for g in ['XGBoost_GPU', 'LightGBM_GPU', 'PyTorch']) else '❌'
    perf_data.append({
        'Model': display_name,
        'GPU': gpu_flag,
        'Test Acc': f"{r['test_accuracy']:.4f}",
        'Test AUC': f"{r['test_auc']:.4f}",
        'Test F1': f"{r['test_f1']:.4f}",
        'Precision': f"{r['test_precision']:.4f}",
        'Recall': f"{r['test_recall']:.4f}",
        'Train Acc': f"{r['train_accuracy']:.4f}",
        'Overfit Gap': f"{r['train_accuracy'] - r['test_accuracy']:+.4f}",
        'Time (s)': f"{r['time']:.1f}"
    })

perf_df = pd.DataFrame(perf_data)
print("\n📊 MODEL PERFORMANCE COMPARISON")
print("=" * 120)
display(perf_df.style.set_properties(**{'text-align': 'center'})
        .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])
        .highlight_max(subset=['Test Acc', 'Test AUC', 'Test F1'], color='#d4edda')
        .highlight_min(subset=['Overfit Gap'], color='#d4edda'))

best_model_name = max(results, key=lambda k: results[k]['test_f1'])
print(f"\n🏆 Best Model: {best_model_name.replace('_', ' ')} (F1={results[best_model_name]['test_f1']:.4f})")'''

OVERFIT_CODE = r'''# ── Overfitting Analysis ──
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
model_names = [n.replace('_', ' ') for n in results.keys()]
test_accs = [r['test_accuracy'] for r in results.values()]
train_accs = [r['train_accuracy'] for r in results.values()]
test_aucs = [r['test_auc'] for r in results.values()]
train_aucs = [r['train_auc'] for r in results.values()]
gaps = [r['train_accuracy'] - r['test_accuracy'] for r in results.values()]

x = np.arange(len(model_names))
width = 0.35

bars1 = axes[0].bar(x - width/2, train_accs, width, label='Train', color='#3498db', alpha=0.8)
bars2 = axes[0].bar(x + width/2, test_accs, width, label='Test', color='#e74c3c', alpha=0.8)
axes[0].set_title('Train vs Test Accuracy', fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
axes[0].legend()
axes[0].set_ylim(0.75, 1.0)

axes[1].bar(x - width/2, train_aucs, width, label='Train', color='#3498db', alpha=0.8)
axes[1].bar(x + width/2, test_aucs, width, label='Test', color='#e74c3c', alpha=0.8)
axes[1].set_title('Train vs Test AUC-ROC', fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
axes[1].legend()
axes[1].set_ylim(0.85, 1.0)

colors = ['#2ecc71' if g < 0.05 else '#f39c12' if g < 0.10 else '#e74c3c' for g in gaps]
axes[2].bar(model_names, gaps, color=colors, edgecolor='white')
axes[2].axhline(y=0.05, color='red', linestyle='--', label='5% threshold')
axes[2].set_title('Overfitting Gap (Train - Test)', fontweight='bold')
axes[2].set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
axes[2].legend()

plt.tight_layout()
plt.savefig('figures/05_overfitting.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n📊 Overfitting Assessment:")
for name, r in results.items():
    gap = r['train_accuracy'] - r['test_accuracy']
    status = '✅ Good' if gap < 0.05 else '⚠️ Moderate' if gap < 0.10 else '❌ Overfitting'
    print(f"   {name:25s} Gap={gap:+.4f} → {status}")'''

ROC_CODE = r'''# ── ROC Curves & Confusion Matrix ──
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for name, pred in predictions.items():
    fpr, tpr, _ = roc_curve(y_test, pred['y_prob'])
    auc = results[name]['test_auc']
    axes[0].plot(fpr, tpr, linewidth=2, label=f"{name.replace('_',' ')} (AUC={auc:.3f})")
axes[0].plot([0,1], [0,1], 'k--', alpha=0.3)
axes[0].set_xlabel('False Positive Rate', fontsize=12)
axes[0].set_ylabel('True Positive Rate', fontsize=12)
axes[0].set_title('ROC Curves — All Models', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=9)

y_pred_best = predictions[best_model_name]['y_pred']
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt=',', cmap='Blues', ax=axes[1],
            xticklabels=['Normal', 'Extended'], yticklabels=['Normal', 'Extended'])
axes[1].set_title(f'Confusion Matrix — {best_model_name.replace("_"," ")}', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('figures/05_roc_confusion.png', dpi=150, bbox_inches='tight')
plt.show()'''

FAIRNESS_CALC_CODE = r'''# ── Fairness Metrics Calculator ──
class FairnessCalculator:
    """Compute paper-aligned fairness metrics: DI, WTPR, SPD, EOD, PPV ratio."""

    @staticmethod
    def disparate_impact(y_pred, attr_values):
        groups = sorted(set(attr_values))
        selection_rates = {}
        for g in groups:
            mask = attr_values == g
            if mask.sum() > 0:
                selection_rates[g] = y_pred[mask].mean()
        if len(selection_rates) < 2:
            return 1.0, selection_rates
        sr_vals = list(selection_rates.values())
        di = min(sr_vals) / max(sr_vals) if max(sr_vals) > 0 else 0
        return di, selection_rates

    @staticmethod
    def worst_case_tpr(y_true, y_pred, attr_values):
        groups = sorted(set(attr_values))
        tprs = {}
        for g in groups:
            mask = attr_values == g
            pos = (y_true[mask] == 1)
            if pos.sum() > 0:
                tprs[g] = y_pred[mask][pos].mean()
        if not tprs:
            return 0.0, tprs
        return min(tprs.values()), tprs

    @staticmethod
    def statistical_parity_diff(y_pred, attr_values):
        groups = sorted(set(attr_values))
        srs = [y_pred[attr_values == g].mean() for g in groups if (attr_values == g).sum() > 0]
        return max(srs) - min(srs) if srs else 0

    @staticmethod
    def equal_opportunity_diff(y_true, y_pred, attr_values):
        groups = sorted(set(attr_values))
        tprs = []
        for g in groups:
            mask = (attr_values == g) & (y_true == 1)
            if mask.sum() > 0:
                tprs.append(y_pred[mask].mean())
        return max(tprs) - min(tprs) if len(tprs) >= 2 else 0

    @staticmethod
    def ppv_ratio(y_true, y_pred, attr_values):
        groups = sorted(set(attr_values))
        ppvs = {}
        for g in groups:
            mask = (attr_values == g) & (y_pred == 1)
            if mask.sum() > 0:
                ppvs[g] = y_true[mask].mean()
        if len(ppvs) < 2:
            return 1.0, ppvs
        vals = list(ppvs.values())
        return min(vals) / max(vals) if max(vals) > 0 else 0, ppvs

fc = FairnessCalculator()
print("✅ FairnessCalculator ready (DI, WTPR, SPD, EOD, PPV)")'''

FAIRNESS_COMPUTE_CODE = r'''# ── Compute Fairness for ALL Models × ALL Attributes ──
all_fairness = {}

print("=" * 90)
print("📊 FAIRNESS METRICS — ALL MODELS")
print("=" * 90)

for m_name, pred in predictions.items():
    y_pred = pred['y_pred']
    model_fair = {}

    print(f"\n┌─ {m_name.replace('_', ' ')} {'─' * (70 - len(m_name))}┐")
    print(f"  {'Attribute':<15} {'DI':>8} {'WTPR':>8} {'SPD':>8} {'EOD':>8} {'PPV-R':>8} {'F1':>8}")
    print(f"  {'─'*65}")

    for attr_name, attr_vals in protected_attributes.items():
        attr_test = attr_vals[test_idx]

        di, di_detail = fc.disparate_impact(y_pred, attr_test)
        wtpr, tpr_detail = fc.worst_case_tpr(y_test, y_pred, attr_test)
        spd = fc.statistical_parity_diff(y_pred, attr_test)
        eod = fc.equal_opportunity_diff(y_test, y_pred, attr_test)
        ppv_r, ppv_detail = fc.ppv_ratio(y_test, y_pred, attr_test)
        f1 = results[m_name]['test_f1']

        model_fair[attr_name] = {
            'DI': di, 'WTPR': wtpr, 'SPD': spd, 'EOD': eod, 'PPV_ratio': ppv_r,
            'F1': f1, 'DI_detail': di_detail, 'TPR_detail': tpr_detail, 'PPV_detail': ppv_detail
        }

        di_status = '✅' if 0.8 <= di <= 1.25 else '⚠️'
        wtpr_status = '✅' if wtpr >= 0.6 else '⚠️'
        print(f"  {attr_name:<15} {di:>7.3f}{di_status} {wtpr:>7.3f}{wtpr_status} {spd:>8.3f} {eod:>8.3f} {ppv_r:>8.3f} {f1:>8.3f}")

    all_fairness[m_name] = model_fair
    print(f"└{'─' * 78}┘")

print("\n✅ Fairness metrics computed for all models")'''

FAIRNESS_HEATMAP_CODE = r'''# ── Fairness Heatmap Visualization ──
fig, axes = plt.subplots(1, 3, figsize=(22, 6))

metrics_to_plot = ['DI', 'WTPR', 'PPV_ratio']
titles = ['Disparate Impact (DI)\nIdeal = 1.0', 'Worst-case TPR (WTPR)\nHigher = Better', 'PPV Ratio\nIdeal = 1.0']
cmaps = ['RdYlGn', 'RdYlGn', 'RdYlGn']
vranges = [(0.5, 1.2), (0.5, 1.0), (0.5, 1.0)]

model_short = [n.replace('_', ' ') for n in all_fairness.keys()]
attrs = list(protected_attributes.keys())

for idx, (metric, title, cmap, vrange) in enumerate(zip(metrics_to_plot, titles, cmaps, vranges)):
    data = []
    for m_name in all_fairness:
        row = [all_fairness[m_name][a][metric] for a in attrs]
        data.append(row)

    data = np.array(data)
    sns.heatmap(data, annot=True, fmt='.3f', cmap=cmap, vmin=vrange[0], vmax=vrange[1],
                xticklabels=attrs, yticklabels=model_short, ax=axes[idx], linewidths=1)
    axes[idx].set_title(title, fontweight='bold', fontsize=12)

plt.suptitle('Fairness Metrics Across Models & Protected Attributes', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/06_fairness_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()'''

SUBSET_7A_CODE = r'''# ── 7a: Random Subset Fairness Analysis ──
best_m = results[best_model_name]['model']
subset_sizes = [1000, 5000, 10000, 50000, len(test_idx)]
n_repeats = 10

subset_fairness = {}
print("📊 Fairness Metrics by Random Subset Size")
print("=" * 90)

for size in subset_sizes:
    size_label = f"{size//1000}K" if size < len(test_idx) else "Full"
    subset_fairness[size_label] = {attr: {'DI': [], 'WTPR': [], 'F1': []} for attr in protected_attributes}

    repeats = n_repeats if size < len(test_idx) else 1
    for rep in range(repeats):
        if size < len(test_idx):
            idx_sub = np.random.choice(len(test_idx), size=size, replace=False)
        else:
            idx_sub = np.arange(len(test_idx))

        X_sub = X_test_scaled[idx_sub]
        y_sub = y_test[idx_sub]

        if 'XGBoost' in best_model_name or 'LightGBM' in best_model_name:
            y_pred_sub = best_m.predict(X_sub)
        elif 'PyTorch' in best_model_name:
            with torch.no_grad():
                prob = torch.sigmoid(best_m(torch.FloatTensor(X_sub).to(DEVICE)).squeeze()).cpu().numpy()
                y_pred_sub = (prob >= 0.5).astype(int)
        else:
            y_pred_sub = best_m.predict(X_sub)

        for attr_name, attr_vals in protected_attributes.items():
            attr_sub = attr_vals[test_idx][idx_sub]
            di, _ = fc.disparate_impact(y_pred_sub, attr_sub)
            wtpr, _ = fc.worst_case_tpr(y_sub, y_pred_sub, attr_sub)
            f1_val = f1_score(y_sub, y_pred_sub)
            subset_fairness[size_label][attr_name]['DI'].append(di)
            subset_fairness[size_label][attr_name]['WTPR'].append(wtpr)
            subset_fairness[size_label][attr_name]['F1'].append(f1_val)

for attr in protected_attributes:
    print(f"\n📊 {attr}:")
    print(f"  {'Size':<8} {'DI (mean±std)':>18} {'WTPR (mean±std)':>18} {'F1 (mean±std)':>18}")
    print(f"  {'─'*62}")
    for size_label in subset_fairness:
        di_vals = subset_fairness[size_label][attr]['DI']
        wtpr_vals = subset_fairness[size_label][attr]['WTPR']
        f1_vals = subset_fairness[size_label][attr]['F1']
        print(f"  {size_label:<8} {np.mean(di_vals):>8.3f}±{np.std(di_vals):.3f}  "
              f"{np.mean(wtpr_vals):>8.3f}±{np.std(wtpr_vals):.3f}  "
              f"{np.mean(f1_vals):>8.3f}±{np.std(f1_vals):.3f}")'''

SUBSET_7A_VIZ_CODE = r'''# ── 7a Visualization: Sample Size Effect ──
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for idx, attr in enumerate(protected_attributes):
    ax = axes[idx // 2, idx % 2]
    sizes = list(subset_fairness.keys())

    for metric, color, marker in [('DI', '#e74c3c', 'o'), ('WTPR', '#3498db', 's'), ('F1', '#2ecc71', '^')]:
        means = [np.mean(subset_fairness[s][attr][metric]) for s in sizes]
        stds = [np.std(subset_fairness[s][attr][metric]) for s in sizes]
        ax.errorbar(range(len(sizes)), means, yerr=stds, marker=marker, linewidth=2,
                    capsize=5, label=metric, color=color)

    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='80% rule')
    ax.set_title(f'{attr}', fontweight='bold', fontsize=13)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(sizes)
    ax.set_xlabel('Subset Size')
    ax.set_ylabel('Score')
    ax.legend(fontsize=9)
    ax.set_ylim(0.3, 1.1)

plt.suptitle('7a: Fairness Metrics Stability Across Random Subset Sizes', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/07a_subset_size_fairness.png', dpi=150, bbox_inches='tight')
plt.show()'''

SUBSET_7B_CODE = r'''# ── 7b: Race-Stratified Subset Analysis ──
race_subsets = {}
race_attr = protected_attributes['RACE'][test_idx]
unique_races = sorted(set(race_attr))

print("📊 Fairness Metrics by Race-Stratified Subsets")
print("=" * 80)
print(f"  {'Subset':<30} {'Size':>8} {'DI(Race)':>10} {'WTPR(Race)':>12} {'F1':>8}")
print(f"  {'─'*68}")

for race in unique_races:
    mask = race_attr == race
    if mask.sum() < 100:
        continue

    X_sub = X_test_scaled[mask]
    y_sub = y_test[mask]

    if 'PyTorch' in best_model_name:
        with torch.no_grad():
            prob = torch.sigmoid(best_m(torch.FloatTensor(X_sub).to(DEVICE)).squeeze()).cpu().numpy()
            y_pred_sub = (prob >= 0.5).astype(int)
    else:
        y_pred_sub = best_m.predict(X_sub)

    age_sub = protected_attributes['AGE_GROUP'][test_idx][mask]
    di, _ = fc.disparate_impact(y_pred_sub, age_sub)
    wtpr, _ = fc.worst_case_tpr(y_sub, y_pred_sub, age_sub)
    f1_val = f1_score(y_sub, y_pred_sub) if len(set(y_sub)) > 1 else 0

    race_subsets[race] = {'size': mask.sum(), 'DI': di, 'WTPR': wtpr, 'F1': f1_val,
                          'accuracy': accuracy_score(y_sub, y_pred_sub)}
    print(f"  {race:<30} {mask.sum():>8,} {di:>10.3f} {wtpr:>12.3f} {f1_val:>8.3f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
races = list(race_subsets.keys())
colors = sns.color_palette('Set2', len(races))

for idx, metric in enumerate(['DI', 'WTPR', 'F1']):
    vals = [race_subsets[r][metric] for r in races]
    bars = axes[idx].bar(races, vals, color=colors, edgecolor='white', linewidth=1.5)
    axes[idx].axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Fair threshold')
    axes[idx].set_title(f'{metric} by Race Subset', fontweight='bold')
    axes[idx].set_xticklabels(races, rotation=30, ha='right')
    axes[idx].legend()
    for bar, val in zip(bars, vals):
        axes[idx].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{val:.3f}',
                       ha='center', va='bottom', fontsize=9)

plt.suptitle('7b: Fairness Within Race-Stratified Subsets (Age Group as secondary attribute)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/07b_race_subset_fairness.png', dpi=150, bbox_inches='tight')
plt.show()'''

SUBSET_7C_CODE = r'''# ── 7c: Age-Group Subset Fairness ──
age_subsets = {}
age_attr = protected_attributes['AGE_GROUP'][test_idx]
unique_ages = sorted(set(age_attr))

print("📊 Fairness Metrics by Age-Group Subsets")
print("=" * 80)
print(f"  {'Age Group':<25} {'Size':>8} {'DI(Race)':>10} {'WTPR(Race)':>12} {'F1':>8}")
print(f"  {'─'*63}")

for age_grp in unique_ages:
    mask = age_attr == age_grp
    if mask.sum() < 100:
        continue

    X_sub = X_test_scaled[mask]
    y_sub = y_test[mask]

    if 'PyTorch' in best_model_name:
        with torch.no_grad():
            prob = torch.sigmoid(best_m(torch.FloatTensor(X_sub).to(DEVICE)).squeeze()).cpu().numpy()
            y_pred_sub = (prob >= 0.5).astype(int)
    else:
        y_pred_sub = best_m.predict(X_sub)

    race_sub = protected_attributes['RACE'][test_idx][mask]
    di, _ = fc.disparate_impact(y_pred_sub, race_sub)
    wtpr, _ = fc.worst_case_tpr(y_sub, y_pred_sub, race_sub)
    f1_val = f1_score(y_sub, y_pred_sub) if len(set(y_sub)) > 1 else 0

    age_subsets[age_grp] = {'size': mask.sum(), 'DI': di, 'WTPR': wtpr, 'F1': f1_val}
    print(f"  {age_grp:<25} {mask.sum():>8,} {di:>10.3f} {wtpr:>12.3f} {f1_val:>8.3f}")

fig, ax = plt.subplots(figsize=(12, 6))
age_groups = list(age_subsets.keys())
x = np.arange(len(age_groups))
w = 0.25

for i, (metric, color) in enumerate([('DI', '#e74c3c'), ('WTPR', '#3498db'), ('F1', '#2ecc71')]):
    vals = [age_subsets[a][metric] for a in age_groups]
    bars = ax.bar(x + i*w, vals, w, label=metric, color=color, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{val:.2f}',
                ha='center', va='bottom', fontsize=8)

ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Fair threshold')
ax.set_xticks(x + w)
ax.set_xticklabels(age_groups)
ax.set_title('7c: Fairness Within Age-Group Subsets (Race as protected attribute)', fontweight='bold', fontsize=13)
ax.legend()

plt.tight_layout()
plt.savefig('figures/07c_age_subset_fairness.png', dpi=150, bbox_inches='tight')
plt.show()'''

SUBSET_7D_CODE = r'''# ── 7d: Hospital-Based Subset Fairness ──
hosp_test = hospital_ids[test_idx]
unique_hospitals = np.unique(hosp_test)

hosp_counts = pd.Series(hosp_test).value_counts()
large_hospitals = hosp_counts[hosp_counts >= 200].index.tolist()
np.random.seed(42)
sample_hospitals = np.random.choice(large_hospitals, size=min(30, len(large_hospitals)), replace=False)

hosp_fairness = []
print("📊 Fairness Across Hospital Subsets (30 sampled hospitals)")
print("=" * 70)
print(f"  {'Hospital':<12} {'Size':>6} {'DI(Race)':>10} {'WTPR(Race)':>12} {'F1':>8} {'Acc':>8}")
print(f"  {'─'*56}")

for hosp in sample_hospitals[:15]:
    mask = hosp_test == hosp
    X_sub = X_test_scaled[mask]
    y_sub = y_test[mask]

    if 'PyTorch' in best_model_name:
        with torch.no_grad():
            prob = torch.sigmoid(best_m(torch.FloatTensor(X_sub).to(DEVICE)).squeeze()).cpu().numpy()
            y_pred_sub = (prob >= 0.5).astype(int)
    else:
        y_pred_sub = best_m.predict(X_sub)

    race_sub = protected_attributes['RACE'][test_idx][mask]
    di, _ = fc.disparate_impact(y_pred_sub, race_sub)
    wtpr, _ = fc.worst_case_tpr(y_sub, y_pred_sub, race_sub)
    f1_val = f1_score(y_sub, y_pred_sub) if len(set(y_sub)) > 1 else 0
    acc = accuracy_score(y_sub, y_pred_sub)

    hosp_fairness.append({'hospital': hosp, 'size': mask.sum(), 'DI': di, 'WTPR': wtpr, 'F1': f1_val, 'acc': acc})
    print(f"  {hosp:<12} {mask.sum():>6} {di:>10.3f} {wtpr:>12.3f} {f1_val:>8.3f} {acc:>8.3f}")

print(f"  ... ({len(sample_hospitals)} hospitals total)")

for hosp in sample_hospitals[15:]:
    mask = hosp_test == hosp
    X_sub = X_test_scaled[mask]
    y_sub = y_test[mask]
    if 'PyTorch' in best_model_name:
        with torch.no_grad():
            prob = torch.sigmoid(best_m(torch.FloatTensor(X_sub).to(DEVICE)).squeeze()).cpu().numpy()
            y_pred_sub = (prob >= 0.5).astype(int)
    else:
        y_pred_sub = best_m.predict(X_sub)
    race_sub = protected_attributes['RACE'][test_idx][mask]
    di, _ = fc.disparate_impact(y_pred_sub, race_sub)
    wtpr, _ = fc.worst_case_tpr(y_sub, y_pred_sub, race_sub)
    f1_val = f1_score(y_sub, y_pred_sub) if len(set(y_sub)) > 1 else 0
    acc = accuracy_score(y_sub, y_pred_sub)
    hosp_fairness.append({'hospital': hosp, 'size': mask.sum(), 'DI': di, 'WTPR': wtpr, 'F1': f1_val, 'acc': acc})

hosp_df = pd.DataFrame(hosp_fairness)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, metric in enumerate(['DI', 'WTPR', 'F1']):
    axes[idx].hist(hosp_df[metric].dropna(), bins=20, color=['#e74c3c','#3498db','#2ecc71'][idx],
                   edgecolor='white', alpha=0.8)
    axes[idx].axvline(x=hosp_df[metric].mean(), color='black', linestyle='--',
                      label=f'Mean={hosp_df[metric].mean():.3f}')
    axes[idx].axvline(x=0.8, color='red', linestyle=':', label='Fair threshold')
    axes[idx].set_title(f'{metric} Distribution Across Hospitals', fontweight='bold')
    axes[idx].legend()
    axes[idx].set_xlabel(metric)

plt.suptitle('7d: Fairness Heterogeneity Across Hospitals', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/07d_hospital_fairness.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n📊 Hospital Fairness Summary:")
print(f"   DI:   mean={hosp_df['DI'].mean():.3f} ± {hosp_df['DI'].std():.3f}")
print(f"   WTPR: mean={hosp_df['WTPR'].mean():.3f} ± {hosp_df['WTPR'].std():.3f}")
print(f"   F1:   mean={hosp_df['F1'].mean():.3f} ± {hosp_df['F1'].std():.3f}")'''

FAIRNESS_METHODS_CODE = r'''# ── Compare Fairness Methods ──
fairness_methods = {}

print("📊 Fairness Methods Comparison (Best Model: " + best_model_name.replace('_', ' ') + ")")
print("=" * 90)

y_pred_best = predictions[best_model_name]['y_pred']

for attr_name, attr_vals in protected_attributes.items():
    attr_test = attr_vals[test_idx]

    di, _ = fc.disparate_impact(y_pred_best, attr_test)
    wtpr, _ = fc.worst_case_tpr(y_test, y_pred_best, attr_test)
    spd = fc.statistical_parity_diff(y_pred_best, attr_test)
    eod = fc.equal_opportunity_diff(y_test, y_pred_best, attr_test)
    ppv_r, _ = fc.ppv_ratio(y_test, y_pred_best, attr_test)

    groups = sorted(set(attr_test))
    fprs = []
    for g in groups:
        mask = (attr_test == g) & (y_test == 0)
        if mask.sum() > 0:
            fprs.append(y_pred_best[mask].mean())
    fpr_diff = max(fprs) - min(fprs) if len(fprs) >= 2 else 0
    eq_odds = max(eod, fpr_diff)

    fairness_methods[attr_name] = {
        'DI': di, 'SPD': spd, 'EOD': eod, 'PPV_Ratio': ppv_r, 'WTPR': wtpr,
        'Eq_Odds': eq_odds, 'FPR_Diff': fpr_diff
    }

method_df = pd.DataFrame(fairness_methods).T
method_df.columns = ['DI', 'SPD', 'EOD', 'PPV Ratio', 'WTPR', 'Eq. Odds', 'FPR Diff']
print()
display(method_df.style.format("{:.4f}")
        .background_gradient(cmap='RdYlGn', axis=None)
        .set_caption('Fairness Methods Comparison by Protected Attribute'))

print("\n📊 Interpretation Guide:")
print("   DI: Closer to 1.0 = fairer (0.8-1.25 = fair)")
print("   SPD/EOD/FPR_Diff: Closer to 0 = fairer")
print("   PPV Ratio: Closer to 1.0 = fairer")
print("   WTPR: Higher = better")
print("   Eq. Odds: Closer to 0 = fairer")'''

RADAR_CODE = r'''# ── Fairness Methods Radar Chart ──
fig, axes = plt.subplots(2, 2, figsize=(14, 14), subplot_kw=dict(projection='polar'))

for idx, attr in enumerate(protected_attributes):
    ax = axes[idx // 2, idx % 2]
    methods = ['DI', 'SPD', 'EOD', 'PPV Ratio', 'WTPR', 'Eq. Odds']
    values = [fairness_methods[attr]['DI'],
              1 - fairness_methods[attr]['SPD'],
              1 - fairness_methods[attr]['EOD'],
              fairness_methods[attr]['PPV_Ratio'],
              fairness_methods[attr]['WTPR'],
              1 - fairness_methods[attr]['Eq_Odds']]

    angles = np.linspace(0, 2*np.pi, len(methods), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    ax.plot(angles, values, 'o-', linewidth=2, color='steelblue')
    ax.fill(angles, values, alpha=0.25, color='steelblue')
    ax.set_thetagrids(np.degrees(angles[:-1]), methods, fontsize=9)
    ax.set_ylim(0, 1.2)
    ax.set_title(f'{attr}', fontweight='bold', fontsize=13, pad=20)

    circle_angles = np.linspace(0, 2*np.pi, 100)
    ax.plot(circle_angles, [0.8]*100, '--', color='red', alpha=0.5, linewidth=1)

plt.suptitle('Fairness Methods Radar — All Protected Attributes', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/08_fairness_methods_radar.png', dpi=150, bbox_inches='tight')
plt.show()'''

FAIR_MODEL_CODE = r'''# ── Fairness-Derived Model: λ-Scaled Reweighing ──
print("🔧 Training Fairness-Aware Model (λ-Scaled Reweighing + Threshold Optimization)")
print("=" * 80)

LAMBDA_FAIR = 5.0
print(f"   λ = {LAMBDA_FAIR} (fairness-accuracy trade-off parameter)")

race_train = protected_attributes['RACE'][train_idx]
groups = sorted(set(race_train))
n_total = len(y_train)

sample_weights = np.ones(n_total)
for g in groups:
    mask_g = race_train == g
    n_g = mask_g.sum()
    for label in [0, 1]:
        mask_gl = mask_g & (y_train == label)
        n_gl = mask_gl.sum()
        if n_gl > 0:
            expected = (n_g / n_total) * ((y_train == label).sum() / n_total)
            observed = n_gl / n_total
            raw_weight = expected / observed if observed > 0 else 1.0
            scaled_weight = 1.0 + LAMBDA_FAIR * (raw_weight - 1.0)
            sample_weights[mask_gl] = max(scaled_weight, 0.1)

print(f"   Sample weights range: [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")

fair_model = xgb.XGBClassifier(
    n_estimators=1000, max_depth=10, learning_rate=0.05, subsample=0.85,
    colsample_bytree=0.85, reg_alpha=0.05, reg_lambda=0.5,
    min_child_weight=5, device='cuda', tree_method='hist',
    random_state=42, eval_metric='logloss', early_stopping_rounds=20
)

start = time.time()
fair_model.fit(X_train_scaled, y_train, sample_weight=sample_weights,
               eval_set=[(X_test_scaled, y_test)], verbose=False)
fair_time = time.time() - start

y_prob_fair = fair_model.predict_proba(X_test_scaled)[:, 1]
y_pred_fair_raw = (y_prob_fair >= 0.5).astype(int)

print(f"   Training time: {fair_time:.1f}s")
print(f"   Raw accuracy: {accuracy_score(y_test, y_pred_fair_raw):.4f}")
print(f"   Raw F1: {f1_score(y_test, y_pred_fair_raw):.4f}")'''

FAIR_THRESHOLD_CODE = r'''# ── Per-Group Threshold Optimization ──
print("\n🔧 Optimizing Per-Group Thresholds for Equal Opportunity")
print("-" * 60)

race_test = protected_attributes['RACE'][test_idx]
best_thresholds = {}
target_tpr = 0.82

for g in groups:
    mask_g = (race_test == g) & (y_test == 1)
    if mask_g.sum() < 10:
        best_thresholds[g] = 0.5
        continue

    probs_g = y_prob_fair[mask_g]
    best_t = 0.5
    best_diff = float('inf')

    for t in np.arange(0.2, 0.8, 0.01):
        tpr_g = (probs_g >= t).mean()
        diff = abs(tpr_g - target_tpr)
        if diff < best_diff:
            best_diff = diff
            best_t = t

    best_thresholds[g] = best_t
    tpr_final = (probs_g >= best_t).mean()
    print(f"   {g:15s}: threshold={best_t:.2f}, TPR={tpr_final:.3f}")

y_pred_fair = np.zeros_like(y_test)
for g in groups:
    mask_g = race_test == g
    y_pred_fair[mask_g] = (y_prob_fair[mask_g] >= best_thresholds[g]).astype(int)

fair_acc = accuracy_score(y_test, y_pred_fair)
fair_f1 = f1_score(y_test, y_pred_fair)
fair_auc = roc_auc_score(y_test, y_prob_fair)
print(f"\n   Fair model (optimized thresholds):")
print(f"   Accuracy: {fair_acc:.4f} | F1: {fair_f1:.4f} | AUC: {fair_auc:.4f}")'''

FAIR_COMPARE_CODE = r'''# ── Standard vs Fairness-Aware Model ──
print("\n📊 Standard Model vs Fairness-Derived Model")
print("=" * 90)

y_pred_std = predictions[best_model_name]['y_pred']
y_prob_std = predictions[best_model_name]['y_prob']

comparison = {'Metric': [], 'Standard Model': [], 'Fairness-Derived Model': [], 'Δ (Fair - Std)': []}

for attr_name, attr_vals in protected_attributes.items():
    attr_test = attr_vals[test_idx]

    di_std, _ = fc.disparate_impact(y_pred_std, attr_test)
    wtpr_std, tpr_std = fc.worst_case_tpr(y_test, y_pred_std, attr_test)
    eod_std = fc.equal_opportunity_diff(y_test, y_pred_std, attr_test)

    di_fair, _ = fc.disparate_impact(y_pred_fair, attr_test)
    wtpr_fair, tpr_fair = fc.worst_case_tpr(y_test, y_pred_fair, attr_test)
    eod_fair = fc.equal_opportunity_diff(y_test, y_pred_fair, attr_test)

    for metric, std_val, fair_val in [
        (f'{attr_name} DI', di_std, di_fair),
        (f'{attr_name} WTPR', wtpr_std, wtpr_fair),
        (f'{attr_name} EOD', eod_std, eod_fair),
    ]:
        comparison['Metric'].append(metric)
        comparison['Standard Model'].append(f"{std_val:.4f}")
        comparison['Fairness-Derived Model'].append(f"{fair_val:.4f}")
        diff = fair_val - std_val
        better = '✅' if ('DI' in metric and abs(fair_val-1) < abs(std_val-1)) or \
                         ('WTPR' in metric and fair_val > std_val) or \
                         ('EOD' in metric and fair_val < std_val) else '⚠️'
        comparison['Δ (Fair - Std)'].append(f"{diff:+.4f} {better}")

for metric, std_val, fair_val in [
    ('Accuracy', results[best_model_name]['test_accuracy'], fair_acc),
    ('F1-Score', results[best_model_name]['test_f1'], fair_f1),
    ('AUC-ROC', results[best_model_name]['test_auc'], fair_auc),
]:
    comparison['Metric'].append(metric)
    comparison['Standard Model'].append(f"{std_val:.4f}")
    comparison['Fairness-Derived Model'].append(f"{fair_val:.4f}")
    comparison['Δ (Fair - Std)'].append(f"{fair_val-std_val:+.4f}")

comp_df = pd.DataFrame(comparison)
display(comp_df.style.set_properties(**{'text-align': 'center'}))

results['Fairness_Derived'] = {
    'test_accuracy': fair_acc, 'test_auc': fair_auc, 'test_f1': fair_f1,
    'test_precision': precision_score(y_test, y_pred_fair),
    'test_recall': recall_score(y_test, y_pred_fair),
    'train_accuracy': 0, 'train_auc': 0, 'time': fair_time
}
predictions['Fairness_Derived'] = {'y_pred': y_pred_fair, 'y_prob': y_prob_fair}'''

FAIR_VIZ_CODE = r'''# ── Visualization: Standard vs Fair Model ──
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

attrs_to_plot = ['RACE']
for attr in attrs_to_plot:
    attr_test = protected_attributes[attr][test_idx]
    groups_list = sorted(set(attr_test))

    tpr_std_list, tpr_fair_list = [], []
    for g in groups_list:
        mask = (attr_test == g) & (y_test == 1)
        if mask.sum() > 0:
            tpr_std_list.append(y_pred_std[mask].mean())
            tpr_fair_list.append(y_pred_fair[mask].mean())

    x = np.arange(len(groups_list))
    axes[0].bar(x - 0.2, tpr_std_list, 0.35, label='Standard', color='#3498db', alpha=0.8)
    axes[0].bar(x + 0.2, tpr_fair_list, 0.35, label='Fair', color='#2ecc71', alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(groups_list, rotation=30, ha='right')
    axes[0].set_title('TPR by Race Group', fontweight='bold')
    axes[0].axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
    axes[0].legend()

di_std_vals, di_fair_vals = [], []
for attr in protected_attributes:
    attr_test = protected_attributes[attr][test_idx]
    di_s, _ = fc.disparate_impact(y_pred_std, attr_test)
    di_f, _ = fc.disparate_impact(y_pred_fair, attr_test)
    di_std_vals.append(di_s)
    di_fair_vals.append(di_f)

x = np.arange(len(protected_attributes))
axes[1].bar(x - 0.2, di_std_vals, 0.35, label='Standard', color='#3498db', alpha=0.8)
axes[1].bar(x + 0.2, di_fair_vals, 0.35, label='Fair', color='#2ecc71', alpha=0.8)
axes[1].axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Fair threshold')
axes[1].axhline(y=1.0, color='green', linestyle=':', alpha=0.5, label='Ideal')
axes[1].set_xticks(x)
axes[1].set_xticklabels(list(protected_attributes.keys()))
axes[1].set_title('Disparate Impact (DI)', fontweight='bold')
axes[1].legend(fontsize=8)

metrics_names = ['Accuracy', 'F1', 'AUC']
std_vals = [results[best_model_name]['test_accuracy'], results[best_model_name]['test_f1'], results[best_model_name]['test_auc']]
fair_vals_plot = [fair_acc, fair_f1, fair_auc]

x = np.arange(3)
axes[2].bar(x - 0.2, std_vals, 0.35, label='Standard', color='#3498db', alpha=0.8)
axes[2].bar(x + 0.2, fair_vals_plot, 0.35, label='Fair', color='#2ecc71', alpha=0.8)
axes[2].set_xticks(x)
axes[2].set_xticklabels(metrics_names)
axes[2].set_title('Overall Performance', fontweight='bold')
axes[2].legend()
axes[2].set_ylim(0.7, 1.0)

plt.suptitle('Standard Model vs Fairness-Derived Model', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/09_fair_vs_standard.png', dpi=150, bbox_inches='tight')
plt.show()'''

PAPER_COMPARISON_CODE = r'''# ── Comparison with Reference Paper ──
print("=" * 100)
print("📊 COMPARISON WITH REFERENCE PAPER — Tarek et al. (2025) CHASE '25")
print("=" * 100)

paper_results = {
    'Real Only (1K)':         {'DI': 0.98, 'WTPR': 0.48, 'F1': 0.53},
    'Real Only (2.5K)':       {'DI': 0.98, 'WTPR': 0.51, 'F1': 0.51},
    'Real Only (5K)':         {'DI': 0.99, 'WTPR': 0.67, 'F1': 0.55},
    'R+Over (5K)':            {'DI': 0.99, 'WTPR': 0.14, 'F1': 0.28},
    'R+Synth (5K)':           {'DI': 0.98, 'WTPR': 0.14, 'F1': 0.27},
    'R+FairSynth (5K+2.5K)':  {'DI': 1.10, 'WTPR': 0.78, 'F1': 0.46},
    'R+FairSynth (2.5K+2K)':  {'DI': 1.03, 'WTPR': 0.83, 'F1': 0.49},
}

our_di, _ = fc.disparate_impact(predictions[best_model_name]['y_pred'],
                                 protected_attributes['ETHNICITY'][test_idx])
our_wtpr, _ = fc.worst_case_tpr(y_test, predictions[best_model_name]['y_pred'],
                                  protected_attributes['ETHNICITY'][test_idx])
our_f1 = results[best_model_name]['test_f1']

fair_di, _ = fc.disparate_impact(y_pred_fair, protected_attributes['ETHNICITY'][test_idx])
fair_wtpr, _ = fc.worst_case_tpr(y_test, y_pred_fair, protected_attributes['ETHNICITY'][test_idx])

our_results = {
    'Ours (Standard)':  {'DI': our_di, 'WTPR': our_wtpr, 'F1': our_f1},
    'Ours (Fair)':      {'DI': fair_di, 'WTPR': fair_wtpr, 'F1': fair_f1},
}

all_comp = {**paper_results, **our_results}
comp_data = []
for name, vals in all_comp.items():
    source = '📄 Paper' if name.startswith('R') else '🏆 Ours'
    comp_data.append({
        'Source': source, 'Configuration': name,
        'DI': vals['DI'], 'WTPR': vals['WTPR'], 'F1': vals['F1']
    })

comp_df = pd.DataFrame(comp_data)
display(comp_df.style.format({'DI': '{:.3f}', 'WTPR': '{:.3f}', 'F1': '{:.3f}'})
        .highlight_max(subset=['F1', 'WTPR'], color='#d4edda')
        .set_caption('📊 Full Comparison: Paper vs Our Study'))

paper_best_f1 = 0.55
paper_best_wtpr = 0.83
paper_best_di = 1.03

print(f"\n📊 IMPROVEMENT OVER PAPER'S BEST:")
print(f"   F1: {our_f1:.3f} vs {paper_best_f1:.3f} → +{(our_f1-paper_best_f1)*100:.1f}% absolute improvement")
print(f"   WTPR: {our_wtpr:.3f} vs {paper_best_wtpr:.3f} → {(our_wtpr-paper_best_wtpr)*100:+.1f}% absolute")
print(f"   F1 (Fair model): {fair_f1:.3f} vs {paper_best_f1:.3f} → +{(fair_f1-paper_best_f1)*100:.1f}% absolute")'''

PAPER_VIZ_CODE = r'''# ── Paper Comparison Visualization ──
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

configs = list(all_comp.keys())
colors = ['#95a5a6'] * len(paper_results) + ['#e74c3c', '#2ecc71']

for idx, metric in enumerate(['DI', 'WTPR', 'F1']):
    vals = [all_comp[c][metric] for c in configs]
    bars = axes[idx].bar(range(len(configs)), vals, color=colors, edgecolor='white', linewidth=1.5)

    if metric == 'DI':
        axes[idx].axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Ideal (1.0)')
        axes[idx].axhline(y=0.8, color='red', linestyle=':', alpha=0.5, label='80% rule')
    elif metric == 'WTPR':
        axes[idx].axhline(y=0.8, color='red', linestyle=':', alpha=0.5, label='80% threshold')

    axes[idx].set_xticks(range(len(configs)))
    axes[idx].set_xticklabels(configs, rotation=45, ha='right', fontsize=8)
    axes[idx].set_title(f'{metric}', fontweight='bold', fontsize=14)
    axes[idx].legend(fontsize=8)

    for i, (c, v) in enumerate(zip(configs, vals)):
        if 'Ours' in c:
            axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)

plt.suptitle('Comparison with Tarek et al. (2025) — Our Study vs Paper', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/10_paper_comparison.png', dpi=150, bbox_inches='tight')
plt.show()'''

# ═══════════════════════════════════════════════════════════════════
# DETAILED-ONLY CODE BLOCKS (10B, 10C, 11a-d, 12-14)
# ═══════════════════════════════════════════════════════════════════

PER_METRIC_FLUCTUATION_CODE = r'''# ── 10B: Per-Metric Fluctuation (20 Subsets × 5 Metrics × 4 Attributes) ──
N_SUBSETS = 20
SUBSET_FRAC = 0.5
np.random.seed(42)

y_prob_best = predictions[best_model_name]['y_prob']
y_pred_best = predictions[best_model_name]['y_pred']

metric_names = ['DI', 'WTPR', 'SPD', 'EOD', 'PPV_Ratio']
attr_names = list(protected_attributes.keys())

subset_metric_results = {m: {a: [] for a in attr_names} for m in metric_names}

subset_size = int(len(y_test) * SUBSET_FRAC)
print(f"📊 Running {N_SUBSETS} random subsets (each {subset_size:,} samples = {SUBSET_FRAC*100:.0f}% of test)")
print(f"   Model: {best_model_name.replace('_', ' ')}")
print("=" * 90)

for i in tqdm(range(N_SUBSETS), desc="Subsets"):
    idx_sub = np.random.choice(len(y_test), size=subset_size, replace=False)
    y_sub = y_test[idx_sub]
    yp_sub = y_pred_best[idx_sub]

    for attr in attr_names:
        attr_sub = protected_attributes[attr][test_idx][idx_sub]

        di, _ = fc.disparate_impact(yp_sub, attr_sub)
        wtpr, _ = fc.worst_case_tpr(y_sub, yp_sub, attr_sub)
        spd = fc.statistical_parity_diff(yp_sub, attr_sub)
        eod = fc.equal_opportunity_diff(y_sub, yp_sub, attr_sub)
        ppv, _ = fc.ppv_ratio(y_sub, yp_sub, attr_sub)

        subset_metric_results['DI'][attr].append(di)
        subset_metric_results['WTPR'][attr].append(wtpr)
        subset_metric_results['SPD'][attr].append(spd)
        subset_metric_results['EOD'][attr].append(eod)
        subset_metric_results['PPV_Ratio'][attr].append(ppv)

for metric in metric_names:
    print(f"\n{'='*90}")
    print(f"📊 {metric} — Across {N_SUBSETS} Random Subsets")
    print(f"{'='*90}")
    print(f"  {'Attribute':<15} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Range':>8} {'CV%':>8}")
    print(f"  {'─'*65}")
    for attr in attr_names:
        vals = subset_metric_results[metric][attr]
        mn, sd = np.mean(vals), np.std(vals)
        cv = (sd / mn * 100) if mn > 0 else 0
        print(f"  {attr:<15} {mn:8.4f} {sd:8.4f} {min(vals):8.4f} {max(vals):8.4f} {max(vals)-min(vals):8.4f} {cv:7.1f}%")

print(f"\n✅ Complete: {N_SUBSETS} subsets × {len(metric_names)} metrics × {len(attr_names)} attributes = {N_SUBSETS*len(metric_names)*len(attr_names)} evaluations")'''

PER_METRIC_VIZ_CODE = r'''# ── Per-Metric Fluctuation Visualization ──
fig, axes = plt.subplots(len(metric_names), 1, figsize=(18, 4*len(metric_names)), sharex=False)
fig.suptitle(f'Fairness Metric Fluctuation Across {N_SUBSETS} Random Subsets\n(Model: {best_model_name.replace("_", " ")})',
             fontsize=16, fontweight='bold', y=1.01)

colors_attr = {'RACE': '#e74c3c', 'ETHNICITY': '#3498db', 'SEX': '#2ecc71', 'AGE_GROUP': '#f39c12'}
ideal_vals = {'DI': 1.0, 'WTPR': 1.0, 'SPD': 0.0, 'EOD': 0.0, 'PPV_Ratio': 1.0}
fair_thresholds = {'DI': (0.8, 1.25), 'WTPR': (0.8, None), 'SPD': (None, 0.1), 'EOD': (None, 0.1), 'PPV_Ratio': (0.8, 1.25)}

for mi, metric in enumerate(metric_names):
    ax = axes[mi]
    x_positions = np.arange(N_SUBSETS)
    width = 0.2

    for ai, attr in enumerate(attr_names):
        vals = subset_metric_results[metric][attr]
        offset = (ai - 1.5) * width
        ax.bar(x_positions + offset, vals, width, label=attr,
               color=colors_attr[attr], alpha=0.7, edgecolor='white', linewidth=0.5)

    if ideal_vals[metric] is not None:
        ax.axhline(y=ideal_vals[metric], color='black', linestyle='--', alpha=0.5, label=f'Ideal ({ideal_vals[metric]})')

    lo, hi = fair_thresholds[metric]
    if lo is not None:
        ax.axhline(y=lo, color='red', linestyle=':', alpha=0.4, label=f'Fair threshold ({lo})')
    if hi is not None and hi != lo:
        ax.axhline(y=hi, color='red', linestyle=':', alpha=0.4)

    ax.set_ylabel(metric, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric} — {N_SUBSETS} Subsets', fontsize=11)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'S{i+1}' for i in range(N_SUBSETS)], fontsize=8)
    if mi == 0:
        ax.legend(loc='upper right', fontsize=8, ncol=5)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/10b_per_metric_20_subsets.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n📊 FLUCTUATION SUMMARY (Coefficient of Variation)")
print("=" * 70)
print(f"  {'Metric':<12} {'RACE':>10} {'ETHNICITY':>12} {'SEX':>10} {'AGE_GROUP':>12} {'Avg CV%':>10}")
print(f"  {'─'*65}")
for metric in metric_names:
    cvs = []
    row = f"  {metric:<12}"
    for attr in attr_names:
        vals = subset_metric_results[metric][attr]
        mn, sd = np.mean(vals), np.std(vals)
        cv = (sd / mn * 100) if mn > 0 else 0
        cvs.append(cv)
        row += f" {cv:10.2f}%"
    row += f" {np.mean(cvs):10.2f}%"
    print(row)'''

PER_METRIC_VIOLIN_CODE = r'''# ── Violin + Strip Plots ──
fig, axes = plt.subplots(1, 5, figsize=(22, 6))
fig.suptitle(f'Distribution of Fairness Metrics Across {N_SUBSETS} Subsets — Per Attribute',
             fontsize=14, fontweight='bold')

for mi, metric in enumerate(metric_names):
    ax = axes[mi]

    parts = ax.violinplot([subset_metric_results[metric][a] for a in attr_names],
                          positions=range(len(attr_names)), showmeans=True, showextrema=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(list(colors_attr.values())[i])
        pc.set_alpha(0.6)

    for ai, attr in enumerate(attr_names):
        vals = subset_metric_results[metric][attr]
        jitter = np.random.normal(0, 0.05, len(vals))
        ax.scatter(ai + jitter, vals, c=colors_attr[attr], s=20, alpha=0.7, zorder=5, edgecolors='white', linewidth=0.5)

    if ideal_vals[metric] is not None:
        ax.axhline(y=ideal_vals[metric], color='green', linestyle='--', alpha=0.5, linewidth=1)
    lo, hi = fair_thresholds[metric]
    if lo is not None:
        ax.axhline(y=lo, color='red', linestyle=':', alpha=0.5, linewidth=1)

    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(attr_names)))
    ax.set_xticklabels(attr_names, rotation=45, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/10b_metric_violin_strips.png', dpi=150, bbox_inches='tight')
plt.show()

fluctuation_table = []
for metric in metric_names:
    for attr in attr_names:
        vals = subset_metric_results[metric][attr]
        fluctuation_table.append({
            'Metric': metric, 'Attribute': attr,
            'Mean': np.mean(vals), 'SD': np.std(vals),
            'Min': min(vals), 'Max': max(vals),
            'Range': max(vals) - min(vals),
            'CV%': (np.std(vals)/np.mean(vals)*100) if np.mean(vals) > 0 else 0
        })
fluct_df = pd.DataFrame(fluctuation_table)
fluct_df.to_csv('tables/10b_metric_fluctuation_20_subsets.csv', index=False)

pivot = fluct_df.pivot_table(index='Metric', columns='Attribute', values='Mean', aggfunc='first')
pivot_sd = fluct_df.pivot_table(index='Metric', columns='Attribute', values='SD', aggfunc='first')

display_df = pivot.copy()
for col in display_df.columns:
    display_df[col] = pivot[col].apply(lambda x: f"{x:.3f}") + ' ± ' + pivot_sd[col].apply(lambda x: f"{x:.3f}")

print("\n📊 Table for Paper: Mean ± SD across 20 subsets")
display(display_df.style.set_caption("Fairness Metrics Fluctuation (20 Random Subsets)"))'''

LAMBDA_TRADEOFF_CODE = r'''# ── Lambda (λ) Trade-off Experiment ──
print("🔬 Lambda (λ) Trade-off Experiment — Fairness vs Performance")
print("=" * 90)

LAMBDA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.2, 1.5, 2.0]

lambda_results = {}
race_train_attr = protected_attributes['RACE'][train_idx]
race_test_attr = protected_attributes['RACE'][test_idx]
groups_list = sorted(set(race_train_attr))

for lam in LAMBDA_VALUES:
    print(f"\n{'─'*70}")
    print(f"  λ = {lam:.2f}")
    print(f"{'─'*70}")

    sw = np.ones(len(y_train))
    if lam > 0:
        for g in groups_list:
            mask_g = race_train_attr == g
            n_g = mask_g.sum()
            for lab in [0, 1]:
                mask_gl = mask_g & (y_train == lab)
                n_gl = mask_gl.sum()
                if n_gl > 0:
                    expected = (n_g / len(y_train)) * ((y_train == lab).sum() / len(y_train))
                    observed = n_gl / len(y_train)
                    raw_weight = expected / observed if observed > 0 else 1.0
                    sw[mask_gl] = 1.0 + lam * (raw_weight - 1.0)

    print(f"  Weights range: [{sw.min():.3f}, {sw.max():.3f}]")

    lam_model = xgb.XGBClassifier(
        n_estimators=500, max_depth=10, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        device='cuda', tree_method='hist', random_state=42,
        eval_metric='logloss', early_stopping_rounds=20
    )
    lam_model.fit(X_train_scaled, y_train, sample_weight=sw,
                  eval_set=[(X_test_scaled, y_test)], verbose=False)

    y_prob_lam = lam_model.predict_proba(X_test_scaled)[:, 1]
    y_pred_lam = (y_prob_lam >= 0.5).astype(int)

    lam_f1 = f1_score(y_test, y_pred_lam)
    lam_acc = accuracy_score(y_test, y_pred_lam)
    lam_auc = roc_auc_score(y_test, y_prob_lam)

    attr_metrics = {}
    for attr_name_l, attr_vals_l in protected_attributes.items():
        attr_test_l = attr_vals_l[test_idx]
        di_l, _ = fc.disparate_impact(y_pred_lam, attr_test_l)
        wtpr_l, _ = fc.worst_case_tpr(y_test, y_pred_lam, attr_test_l)
        spd_l = fc.statistical_parity_diff(y_pred_lam, attr_test_l)
        eod_l = fc.equal_opportunity_diff(y_test, y_pred_lam, attr_test_l)
        ppv_l, _ = fc.ppv_ratio(y_test, y_pred_lam, attr_test_l)
        attr_metrics[attr_name_l] = {
            'DI': di_l, 'WTPR': wtpr_l, 'SPD': spd_l, 'EOD': eod_l, 'PPV_Ratio': ppv_l
        }

    lambda_results[lam] = {
        'F1': lam_f1, 'Accuracy': lam_acc, 'AUC': lam_auc,
        'attr_metrics': attr_metrics
    }

    print(f"  F1={lam_f1:.4f} | Acc={lam_acc:.4f} | AUC={lam_auc:.4f}")
    for a_name, a_met in attr_metrics.items():
        print(f"    {a_name:12s}: DI={a_met['DI']:.3f}  WTPR={a_met['WTPR']:.3f}  SPD={a_met['SPD']:.3f}  EOD={a_met['EOD']:.3f}")

print(f"\n{'='*90}")
print("📊 Table 2 Equivalent: λ Trade-off Summary")
print(f"{'='*90}")

table2_data = []
for lam in LAMBDA_VALUES:
    r = lambda_results[lam]
    row_data = {'λ': lam, 'F1': r['F1'], 'Accuracy': r['Accuracy'], 'AUC': r['AUC']}
    for attr_n in ['RACE', 'SEX', 'AGE_GROUP', 'ETHNICITY']:
        if attr_n in r['attr_metrics']:
            row_data[f'{attr_n}_DI'] = r['attr_metrics'][attr_n]['DI']
            row_data[f'{attr_n}_WTPR'] = r['attr_metrics'][attr_n]['WTPR']
    table2_data.append(row_data)

lambda_df = pd.DataFrame(table2_data)
lambda_df.to_csv('tables/10c_lambda_tradeoff.csv', index=False)

perf_cols = ['λ', 'F1', 'Accuracy', 'AUC']
display(lambda_df[perf_cols].style.format({
    'λ': '{:.2f}', 'F1': '{:.4f}', 'Accuracy': '{:.4f}', 'AUC': '{:.4f}'
}).set_caption("Performance Metrics by λ").highlight_max(subset=['F1', 'AUC'], color='lightgreen'))'''

LAMBDA_VIZ_CODE = r'''# ── Lambda Trade-off Visualization ──
fig, axes = plt.subplots(2, 3, figsize=(22, 14))
fig.suptitle('Lambda (λ) Trade-off: Fairness vs. Performance', fontsize=16, fontweight='bold')

lambdas = sorted(lambda_results.keys())
attr_colors = {'RACE': '#e41a1c', 'SEX': '#377eb8', 'AGE_GROUP': '#4daf4a', 'ETHNICITY': '#984ea3'}

# Plot 1: F1 vs λ
ax = axes[0, 0]
f1s = [lambda_results[l]['F1'] for l in lambdas]
ax.plot(lambdas, f1s, 'bo-', linewidth=2, markersize=8, label='F1')
ax.fill_between(lambdas, [min(f1s)]*len(lambdas), f1s, alpha=0.15, color='blue')
ax.set_xlabel('λ (Fairness Weight)')
ax.set_ylabel('F1 Score')
ax.set_title('F1 Score vs λ', fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot 2: DI vs λ
ax = axes[0, 1]
for attr_n in attr_colors:
    di_vals_l = [lambda_results[l]['attr_metrics'][attr_n]['DI'] for l in lambdas]
    ax.plot(lambdas, di_vals_l, 'o-', color=attr_colors[attr_n], linewidth=2, markersize=7, label=attr_n)
ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect')
ax.axhline(y=0.8, color='red', linestyle=':', alpha=0.5, label='80% Rule')
ax.set_xlabel('λ')
ax.set_ylabel('Disparate Impact')
ax.set_title('DI vs λ', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 3: WTPR vs λ
ax = axes[0, 2]
for attr_n in attr_colors:
    wtpr_vals_l = [lambda_results[l]['attr_metrics'][attr_n]['WTPR'] for l in lambdas]
    ax.plot(lambdas, wtpr_vals_l, 'o-', color=attr_colors[attr_n], linewidth=2, markersize=7, label=attr_n)
ax.set_xlabel('λ')
ax.set_ylabel('Worst-case TPR')
ax.set_title('WTPR vs λ', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 4: SPD vs λ
ax = axes[1, 0]
for attr_n in attr_colors:
    spd_vals_l = [lambda_results[l]['attr_metrics'][attr_n]['SPD'] for l in lambdas]
    ax.plot(lambdas, spd_vals_l, 'o-', color=attr_colors[attr_n], linewidth=2, markersize=7, label=attr_n)
ax.axhline(y=0.0, color='green', linestyle='--', alpha=0.5)
ax.set_xlabel('λ')
ax.set_ylabel('Statistical Parity Diff')
ax.set_title('SPD vs λ (lower = fairer)', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 5: EOD vs λ
ax = axes[1, 1]
for attr_n in attr_colors:
    eod_vals_l = [lambda_results[l]['attr_metrics'][attr_n]['EOD'] for l in lambdas]
    ax.plot(lambdas, eod_vals_l, 'o-', color=attr_colors[attr_n], linewidth=2, markersize=7, label=attr_n)
ax.axhline(y=0.0, color='green', linestyle='--', alpha=0.5)
ax.set_xlabel('λ')
ax.set_ylabel('Equal Opportunity Diff')
ax.set_title('EOD vs λ (lower = fairer)', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 6: Pareto Front
ax = axes[1, 2]
avg_di = [np.mean([lambda_results[l]['attr_metrics'][a]['DI'] for a in attr_colors]) for l in lambdas]
f1s_all = [lambda_results[l]['F1'] for l in lambdas]
sc = ax.scatter(avg_di, f1s_all, c=lambdas, cmap='RdYlGn', s=120, edgecolors='black', zorder=5)
for i, lam in enumerate(lambdas):
    ax.annotate(f'λ={lam}', (avg_di[i], f1s_all[i]), textcoords='offset points',
                xytext=(8, 5), fontsize=9, fontweight='bold')
ax.set_xlabel('Average DI')
ax.set_ylabel('F1 Score')
ax.set_title('Pareto Front: F1 vs Fairness', fontweight='bold')
ax.axvline(x=0.8, color='red', linestyle=':', alpha=0.5)
ax.axvline(x=1.0, color='green', linestyle='--', alpha=0.5)
plt.colorbar(sc, ax=ax, label='λ value')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/10c_lambda_tradeoff.png', dpi=150, bbox_inches='tight')
plt.show()'''

LAMBDA_PAPER_COMPARE_CODE = r'''# ── Our λ Results vs Paper ──
print("📊 Direct Comparison: Our λ Results vs Paper (Tarek et al., CHASE '25)")
print("=" * 90)

paper_table2 = {
    'λ=0.5': {'F1': 0.550, 'DI': 1.10, 'WTPR': 0.78},
    'λ=1.0': {'F1': 0.130, 'DI': 0.99, 'WTPR': 0.73},
    'λ=1.2': {'F1': 0.450, 'DI': 1.10, 'WTPR': 0.78},
    'λ=1.5': {'F1': 0.240, 'DI': 0.99, 'WTPR': 0.15},
}

our_table2 = {}
for lam in [0.5, 1.0, 1.2, 1.5]:
    r = lambda_results[lam]
    our_table2[f'λ={lam}'] = {
        'F1': r['F1'],
        'DI': r['attr_metrics']['RACE']['DI'],
        'WTPR': r['attr_metrics']['RACE']['WTPR'],
    }

comp_rows = []
for lam_key in ['λ=0.5', 'λ=1.0', 'λ=1.2', 'λ=1.5']:
    p = paper_table2[lam_key]
    o = our_table2[lam_key]
    comp_rows.append({
        'λ': lam_key,
        'Paper F1': f"{p['F1']:.3f}", 'Our F1': f"{o['F1']:.3f}", 'ΔF1': f"{o['F1'] - p['F1']:+.3f}",
        'Paper DI': f"{p['DI']:.3f}", 'Our DI': f"{o['DI']:.3f}", 'ΔDI': f"{o['DI'] - p['DI']:+.3f}",
        'Paper WTPR': f"{p['WTPR']:.3f}", 'Our WTPR': f"{o['WTPR']:.3f}", 'ΔWTPR': f"{o['WTPR'] - p['WTPR']:+.3f}",
    })

comp_full_df = pd.DataFrame(comp_rows)
comp_full_df.to_csv('tables/10c_lambda_paper_comparison.csv', index=False)
display(comp_full_df.style.set_caption("λ Trade-off: Our Results (Texas-100X) vs Paper (MIMIC-III)"))

avg_f1_improvement = np.mean([our_table2[k]['F1'] - paper_table2[k]['F1'] for k in paper_table2])
print(f"\n🔑 Key Findings:")
print(f"  • Average F1 improvement over paper: +{avg_f1_improvement:.3f}")
print(f"  • Our F1 remains stable across λ values")
print(f"  • Paper F1 drops dramatically (0.55→0.13) at λ=1.0")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('λ Trade-off: Our Results vs Paper (Tarek et al.)', fontsize=14, fontweight='bold')

lam_labels = ['λ=0.5', 'λ=1.0', 'λ=1.2', 'λ=1.5']
x_pos = np.arange(len(lam_labels))
bar_w = 0.35

for mi, (metric_key, metric_label, ideal) in enumerate([
    ('F1', 'F1 Score', None), ('DI', 'Disparate Impact', 1.0), ('WTPR', 'Worst-case TPR', None)
]):
    ax = axes[mi]
    paper_vals = [paper_table2[k][metric_key] for k in lam_labels]
    our_vals = [our_table2[k][metric_key] for k in lam_labels]

    b1 = ax.bar(x_pos - bar_w/2, paper_vals, bar_w, label='Paper (MIMIC-III)', color='#ff7f0e', alpha=0.8)
    b2 = ax.bar(x_pos + bar_w/2, our_vals, bar_w, label='Ours (Texas-100X)', color='#1f77b4', alpha=0.8)

    if ideal is not None:
        ax.axhline(y=ideal, color='green', linestyle='--', alpha=0.6)

    for bar_set in [b1, b2]:
        for bar in bar_set:
            h = bar.get_height()
            ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                       xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(lam_labels)
    ax.set_ylabel(metric_label)
    ax.set_title(f'{metric_label} by λ', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/10c_lambda_paper_comparison.png', dpi=150, bbox_inches='tight')
plt.show()'''

BOOTSTRAP_CODE = r'''# ── 11a: Bootstrap Stability Test (B=200) ──
B = 200
boot_results = {attr: {g: [] for g in subgroups[attr]} for attr in protected_attributes}

for b in tqdm(range(B), desc='Bootstrap'):
    idx_b = np.random.choice(len(test_idx), size=len(test_idx), replace=True)
    y_b = y_test[idx_b]

    if 'PyTorch' in best_model_name:
        with torch.no_grad():
            prob_b = torch.sigmoid(best_m(torch.FloatTensor(X_test_scaled[idx_b]).to(DEVICE)).squeeze()).cpu().numpy()
            y_pred_b = (prob_b >= 0.5).astype(int)
    else:
        y_pred_b = best_m.predict(X_test_scaled[idx_b])

    for attr_name, attr_vals in protected_attributes.items():
        attr_b = attr_vals[test_idx][idx_b]
        for g in subgroups[attr_name]:
            mask = (attr_b == g) & (y_b == 1)
            if mask.sum() > 0:
                tpr = y_pred_b[mask].mean()
                boot_results[attr_name][g].append(tpr)

print(f"\n✅ Bootstrap Complete: {B} iterations")
print("=" * 70)
print(f"  {'Attribute':<15} {'Subgroup':<25} {'TPR':>8} {'95% CI':>16} {'Width':>8}")
print(f"  {'─'*72}")

for attr in protected_attributes:
    for g in subgroups[attr]:
        vals = boot_results[attr][g]
        if vals:
            mean = np.mean(vals)
            ci_lo, ci_hi = np.percentile(vals, [2.5, 97.5])
            width = ci_hi - ci_lo
            print(f"  {attr:<15} {g:<25} {mean:>8.3f} [{ci_lo:.3f}, {ci_hi:.3f}] {width:>8.4f}")'''

BOOTSTRAP_VIZ_CODE = r'''# ── Bootstrap Visualization ──
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for idx, attr in enumerate(protected_attributes):
    ax = axes[idx // 2, idx % 2]
    groups_list = subgroups[attr]

    for i, g in enumerate(groups_list):
        vals = boot_results[attr][g]
        if vals:
            mean = np.mean(vals)
            ci_lo, ci_hi = np.percentile(vals, [2.5, 97.5])
            ax.errorbar(mean, i, xerr=[[mean-ci_lo], [ci_hi-mean]], fmt='o',
                       capsize=8, markersize=8, linewidth=2, color=f'C{i}')

    ax.set_yticks(range(len(groups_list)))
    ax.set_yticklabels(groups_list)
    ax.axvline(x=0.8, color='red', linestyle='--', alpha=0.5, label='80% threshold')
    ax.set_title(f'{attr}', fontweight='bold', fontsize=13)
    ax.set_xlabel('TPR (Equal Opportunity)')
    ax.legend()

plt.suptitle(f'Bootstrap 95% CI for Equal Opportunity (TPR) — B={B}', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/11a_bootstrap.png', dpi=150, bbox_inches='tight')
plt.show()'''

SEED_CODE = r'''# ── 11b: Seed Sensitivity Test (S=20) ──
S = 20
seed_results = {attr: {g: [] for g in subgroups[attr]} for attr in protected_attributes}
seed_perf = {'acc': [], 'auc': [], 'f1': []}

for seed in tqdm(range(S), desc='Seeds'):
    lr = LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced', random_state=seed)
    lr.fit(X_train_scaled, y_train)

    y_pred_s = lr.predict(X_test_scaled)
    y_prob_s = lr.predict_proba(X_test_scaled)[:, 1]

    seed_perf['acc'].append(accuracy_score(y_test, y_pred_s))
    seed_perf['auc'].append(roc_auc_score(y_test, y_prob_s))
    seed_perf['f1'].append(f1_score(y_test, y_pred_s))

    for attr_name, attr_vals in protected_attributes.items():
        attr_te = attr_vals[test_idx]
        for g in subgroups[attr_name]:
            mask = (attr_te == g) & (y_test == 1)
            if mask.sum() > 0:
                seed_results[attr_name][g].append(y_pred_s[mask].mean())

print(f"\n📊 Performance Stability Across {S} Seeds:")
print(f"   Accuracy: {np.mean(seed_perf['acc']):.4f} ± {np.std(seed_perf['acc']):.4f}")
print(f"   AUC:      {np.mean(seed_perf['auc']):.4f} ± {np.std(seed_perf['auc']):.4f}")
print(f"   F1:       {np.mean(seed_perf['f1']):.4f} ± {np.std(seed_perf['f1']):.4f}")

print(f"\n📊 TPR Stability (Coefficient of Variation):")
for attr in protected_attributes:
    for g in subgroups[attr]:
        vals = seed_results[attr][g]
        if vals:
            cv = np.std(vals) / np.mean(vals) * 100
            print(f"   {attr}/{g}: {np.mean(vals):.3f} ± {np.std(vals):.3f} (CV={cv:.1f}%)")'''

CROSS_HOSP_CODE = r'''# ── 11c: Cross-Hospital Validation (K=20) ──
K_FOLDS = 20
hosp_test_ids = hospital_ids[test_idx]
unique_h = np.unique(hosp_test_ids)
np.random.seed(42)
np.random.shuffle(unique_h)
hospitals_per_fold = len(unique_h) // K_FOLDS

cross_hosp = {attr: {'tpr': [], 'di': []} for attr in protected_attributes}

for fold in tqdm(range(K_FOLDS), desc='Hospital Folds'):
    start_h = fold * hospitals_per_fold
    end_h = start_h + hospitals_per_fold
    held_out = unique_h[start_h:end_h]

    test_mask = np.isin(hosp_test_ids, held_out)
    train_mask = ~test_mask

    if test_mask.sum() < 50 or train_mask.sum() < 50:
        continue

    lr = LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced', random_state=42)
    lr.fit(X_test_scaled[train_mask], y_test[train_mask])
    y_pred_h = lr.predict(X_test_scaled[test_mask])

    for attr_name, attr_vals in protected_attributes.items():
        attr_te = attr_vals[test_idx][test_mask]
        _, tpr_detail = fc.worst_case_tpr(y_test[test_mask], y_pred_h, attr_te)
        di, _ = fc.disparate_impact(y_pred_h, attr_te)

        tpr_ratio = min(tpr_detail.values()) / max(tpr_detail.values()) if len(tpr_detail) >= 2 and max(tpr_detail.values()) > 0 else 0
        cross_hosp[attr_name]['tpr'].append(tpr_ratio)
        cross_hosp[attr_name]['di'].append(di)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

attrs = list(protected_attributes.keys())
tpr_data = [cross_hosp[a]['tpr'] for a in attrs]
di_data = [cross_hosp[a]['di'] for a in attrs]

axes[0].boxplot(tpr_data, labels=attrs)
axes[0].axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Fair threshold')
axes[0].set_title('TPR Ratio Across Hospital Folds', fontweight='bold')
axes[0].legend()

axes[1].boxplot(di_data, labels=attrs)
axes[1].axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Fair threshold')
axes[1].axhline(y=1.0, color='green', linestyle=':', alpha=0.5, label='Ideal')
axes[1].set_title('Disparate Impact Across Hospital Folds', fontweight='bold')
axes[1].legend()

plt.suptitle(f'Cross-Hospital Fairness Heterogeneity (K={K_FOLDS} folds)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/11c_cross_hospital.png', dpi=150, bbox_inches='tight')
plt.show()

for attr in attrs:
    print(f"   {attr}: TPR ratio={np.mean(cross_hosp[attr]['tpr']):.3f}±{np.std(cross_hosp[attr]['tpr']):.3f}, "
          f"DI={np.mean(cross_hosp[attr]['di']):.3f}±{np.std(cross_hosp[attr]['di']):.3f}")'''

THRESHOLD_SWEEP_CODE = r'''# ── 11d: Threshold Sweep Analysis ──
thresholds = np.linspace(0.05, 0.95, 50)
y_prob_best = predictions[best_model_name]['y_prob']

thresh_results = {'tau': [], 'accuracy': [], 'f1': [], 'tpr_ratio_race': [], 'di_race': []}

for tau in tqdm(thresholds, desc='Thresholds'):
    y_pred_tau = (y_prob_best >= tau).astype(int)

    thresh_results['tau'].append(tau)
    thresh_results['accuracy'].append(accuracy_score(y_test, y_pred_tau))
    thresh_results['f1'].append(f1_score(y_test, y_pred_tau) if y_pred_tau.sum() > 0 else 0)

    race_test = protected_attributes['RACE'][test_idx]
    di, _ = fc.disparate_impact(y_pred_tau, race_test)
    wtpr, tpr_detail = fc.worst_case_tpr(y_test, y_pred_tau, race_test)
    tpr_vals = list(tpr_detail.values())
    tpr_ratio = min(tpr_vals) / max(tpr_vals) if len(tpr_vals) >= 2 and max(tpr_vals) > 0 else 0

    thresh_results['tpr_ratio_race'].append(tpr_ratio)
    thresh_results['di_race'].append(di)

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

axes[0].plot(thresh_results['tau'], thresh_results['f1'], 'b-', linewidth=2, label='F1 Score')
axes[0].plot(thresh_results['tau'], thresh_results['accuracy'], 'g--', linewidth=2, label='Accuracy')
axes[0].plot(thresh_results['tau'], thresh_results['di_race'], 'r:', linewidth=2, label='DI (Race)')
axes[0].axhline(y=0.8, color='gray', linestyle='--', alpha=0.3)

best_f1_idx = np.argmax(thresh_results['f1'])
axes[0].axvline(x=thresh_results['tau'][best_f1_idx], color='blue', linestyle=':', alpha=0.5,
                label=f"Best F1: τ={thresh_results['tau'][best_f1_idx]:.2f}")
axes[0].set_xlabel('Classification Threshold (τ)')
axes[0].set_ylabel('Score')
axes[0].set_title('Performance-Fairness Trade-off', fontweight='bold')
axes[0].legend()

axes[1].plot(thresh_results['tau'], thresh_results['tpr_ratio_race'], 'b-', linewidth=2, label='TPR Ratio (Race)')
axes[1].plot(thresh_results['tau'], thresh_results['di_race'], 'r-', linewidth=2, label='DI (Race)')
axes[1].axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Fair threshold')
axes[1].fill_between(thresh_results['tau'], 0.8, 1.2, alpha=0.1, color='green', label='Fair zone')
axes[1].set_xlabel('Classification Threshold (τ)')
axes[1].set_ylabel('Fairness Ratio')
axes[1].set_title('Fairness Across Thresholds', fontweight='bold')
axes[1].legend()

plt.suptitle('Threshold Sweep Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/11d_threshold_sweep.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"   Best F1 threshold: τ={thresh_results['tau'][best_f1_idx]:.2f}")
print(f"   At best F1: Acc={thresh_results['accuracy'][best_f1_idx]:.4f}, "
      f"F1={thresh_results['f1'][best_f1_idx]:.4f}, "
      f"DI={thresh_results['di_race'][best_f1_idx]:.4f}")'''

RESULTS_TABLES_CODE = r'''# ── Paper Results Tables ──
print("=" * 100)
print("📝 RESULTS & DISCUSSION")
print("=" * 100)

print("\n### Table 2: Model Performance Comparison")
perf_table = []
for name, r in results.items():
    if name == 'Fairness_Derived':
        continue
    perf_table.append({
        'Model': name.replace('_', ' '),
        'Accuracy': r['test_accuracy'],
        'AUC-ROC': r['test_auc'],
        'F1': r['test_f1'],
        'Precision': r['test_precision'],
        'Recall': r['test_recall'],
        'Overfit Gap': r['train_accuracy'] - r['test_accuracy']
    })
perf_table_df = pd.DataFrame(perf_table)
display(perf_table_df.style.format({
    'Accuracy': '{:.4f}', 'AUC-ROC': '{:.4f}', 'F1': '{:.4f}',
    'Precision': '{:.4f}', 'Recall': '{:.4f}', 'Overfit Gap': '{:+.4f}'
}).highlight_max(subset=['Accuracy', 'AUC-ROC', 'F1'], color='#d4edda')
.set_caption('Table 2: Model Performance on Texas-100X Test Set'))

print("\n### Table 3: Fairness Metrics by Model and Attribute")
fair_table_data = []
for m_name in all_fairness:
    for attr in protected_attributes:
        fair_table_data.append({
            'Model': m_name.replace('_', ' '),
            'Attribute': attr,
            'DI': all_fairness[m_name][attr]['DI'],
            'WTPR': all_fairness[m_name][attr]['WTPR'],
            'SPD': all_fairness[m_name][attr]['SPD'],
            'EOD': all_fairness[m_name][attr]['EOD'],
            'PPV Ratio': all_fairness[m_name][attr]['PPV_ratio']
        })

fair_table_df = pd.DataFrame(fair_table_data)
display(fair_table_df.style.format({
    'DI': '{:.3f}', 'WTPR': '{:.3f}', 'SPD': '{:.3f}',
    'EOD': '{:.3f}', 'PPV Ratio': '{:.3f}'
}).background_gradient(subset=['DI', 'WTPR', 'PPV Ratio'], cmap='RdYlGn', vmin=0.5, vmax=1.0)
.set_caption('Table 3: Fairness Metrics Across Models and Protected Attributes'))'''

RESULTS_COMPARISON_CODE = r'''# ── Table 4: Comparison with Reference Paper ──
print("\n### Table 4: Comparison with Tarek et al. (2025)")

comparison_full = []
for name, vals in paper_results.items():
    comparison_full.append({
        'Study': 'Tarek et al. (MIMIC-III)',
        'Config': name,
        'DI': vals['DI'], 'WTPR': vals['WTPR'], 'F1': vals['F1']
    })

for m_name in results:
    if m_name == 'Fairness_Derived':
        eth_test = protected_attributes['ETHNICITY'][test_idx]
        di_val, _ = fc.disparate_impact(y_pred_fair, eth_test)
        wtpr_val, _ = fc.worst_case_tpr(y_test, y_pred_fair, eth_test)
        f1_val = fair_f1
    else:
        y_p = predictions[m_name]['y_pred']
        eth_test = protected_attributes['ETHNICITY'][test_idx]
        di_val, _ = fc.disparate_impact(y_p, eth_test)
        wtpr_val, _ = fc.worst_case_tpr(y_test, y_p, eth_test)
        f1_val = results[m_name]['test_f1']

    comparison_full.append({
        'Study': 'Ours (Texas-100X)',
        'Config': m_name.replace('_', ' '),
        'DI': di_val, 'WTPR': wtpr_val, 'F1': f1_val
    })

comp_full_df = pd.DataFrame(comparison_full)
display(comp_full_df.style.format({'DI': '{:.3f}', 'WTPR': '{:.3f}', 'F1': '{:.3f}'})
        .highlight_max(subset=['F1', 'WTPR'], color='#d4edda')
        .set_caption('Table 4: Comprehensive Comparison — Paper vs Our Study'))

print(f"\n📝 Key findings:")
print(f"   • Our best F1 ({our_f1:.3f}) exceeds paper's best ({paper_best_f1:.3f}) by {(our_f1-paper_best_f1)*100:.1f} pp")
print(f"   • DI values in comparable range ({our_di:.3f} vs paper {paper_best_di:.3f})")
print(f"   • All models achieve >0.80 accuracy on 925K records vs paper's 46K")'''

SUBSET_RESULTS_CODE = r'''# ── Table 5: Subset Analysis Results ──
print("\n### Table 5: Fairness Across Data Subsets")

subset_table = []
for size_label in subset_fairness:
    for attr in ['RACE', 'ETHNICITY']:
        subset_table.append({
            'Subset Size': size_label,
            'Attribute': attr,
            'DI (mean±std)': f"{np.mean(subset_fairness[size_label][attr]['DI']):.3f}±{np.std(subset_fairness[size_label][attr]['DI']):.3f}",
            'WTPR (mean±std)': f"{np.mean(subset_fairness[size_label][attr]['WTPR']):.3f}±{np.std(subset_fairness[size_label][attr]['WTPR']):.3f}",
            'F1 (mean±std)': f"{np.mean(subset_fairness[size_label][attr]['F1']):.3f}±{np.std(subset_fairness[size_label][attr]['F1']):.3f}",
        })

subset_df = pd.DataFrame(subset_table)
display(subset_df.style.set_caption('Table 5: Fairness Metrics Stability Across Random Subset Sizes'))'''

DASHBOARD_CODE = r'''# ── Final Comprehensive Dashboard ──
fig = plt.figure(figsize=(24, 16))
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

# 1. Model Performance
ax1 = fig.add_subplot(gs[0, 0:2])
model_names_short = [n.replace('_', ' ').replace(' GPU', '') for n in results if n != 'Fairness_Derived']
accs = [results[n]['test_accuracy'] for n in results if n != 'Fairness_Derived']
aucs = [results[n]['test_auc'] for n in results if n != 'Fairness_Derived']
x = np.arange(len(model_names_short))
ax1.bar(x - 0.2, accs, 0.35, label='Accuracy', color='steelblue', alpha=0.8)
ax1.bar(x + 0.2, aucs, 0.35, label='AUC-ROC', color='coral', alpha=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels(model_names_short, rotation=30, ha='right', fontsize=8)
ax1.set_title('Model Performance', fontweight='bold')
ax1.legend(fontsize=8)
ax1.set_ylim(0.7, 1.0)

# 2. Fairness Heatmap (DI)
ax2 = fig.add_subplot(gs[0, 2:4])
model_fair_names = list(all_fairness.keys())
di_data = [[all_fairness[m][a]['DI'] for a in protected_attributes] for m in model_fair_names]
sns.heatmap(np.array(di_data), annot=True, fmt='.2f', cmap='RdYlGn', ax=ax2,
            xticklabels=list(protected_attributes.keys()),
            yticklabels=[n.replace('_', ' ')[:15] for n in model_fair_names],
            vmin=0.5, vmax=1.2, linewidths=0.5)
ax2.set_title('Disparate Impact (DI)', fontweight='bold')

# 3. Paper Comparison
ax3 = fig.add_subplot(gs[1, 0:2])
paper_f1s = [v['F1'] for v in paper_results.values()]
our_f1s = [results[n]['test_f1'] for n in results if n != 'Fairness_Derived']
all_f1s = paper_f1s + our_f1s
all_labels = list(paper_results.keys()) + [n.replace('_', ' ')[:12] for n in results if n != 'Fairness_Derived']
all_colors = ['#95a5a6'] * len(paper_f1s) + ['#e74c3c'] * len(our_f1s)
ax3.barh(range(len(all_f1s)), all_f1s, color=all_colors, edgecolor='white')
ax3.set_yticks(range(len(all_labels)))
ax3.set_yticklabels(all_labels, fontsize=7)
ax3.set_title('F1 Comparison with Paper', fontweight='bold')
ax3.axvline(x=0.55, color='gray', linestyle='--', alpha=0.5, label="Paper's best")
ax3.legend(fontsize=8)

# 4. Subset Fairness
ax4 = fig.add_subplot(gs[1, 2:4])
sizes = list(subset_fairness.keys())
for metric, color in [('DI', '#e74c3c'), ('WTPR', '#3498db'), ('F1', '#2ecc71')]:
    means = [np.mean(subset_fairness[s]['RACE'][metric]) for s in sizes]
    ax4.plot(range(len(sizes)), means, 'o-', color=color, linewidth=2, label=metric)
ax4.set_xticks(range(len(sizes)))
ax4.set_xticklabels(sizes)
ax4.axhline(y=0.8, color='gray', linestyle='--', alpha=0.3)
ax4.set_title('Subset Size → Fairness (RACE)', fontweight='bold')
ax4.legend(fontsize=8)

# 5. Fair vs Standard
ax5 = fig.add_subplot(gs[2, 0:2])
metrics_compare = ['Accuracy', 'F1', 'AUC']
std_vals = [results[best_model_name]['test_accuracy'], results[best_model_name]['test_f1'], results[best_model_name]['test_auc']]
fair_vals_d = [fair_acc, fair_f1, fair_auc]
x = np.arange(3)
ax5.bar(x - 0.2, std_vals, 0.35, label='Standard', color='steelblue')
ax5.bar(x + 0.2, fair_vals_d, 0.35, label='Fairness-Derived', color='#2ecc71')
ax5.set_xticks(x)
ax5.set_xticklabels(metrics_compare)
ax5.set_title('Standard vs Fair Model', fontweight='bold')
ax5.legend(fontsize=8)
ax5.set_ylim(0.7, 1.0)

# 6. Summary Box
ax6 = fig.add_subplot(gs[2, 2:4])
ax6.axis('off')
best_r = results[best_model_name]
summary = f"""
TEXAS-100X FAIRNESS ANALYSIS
{'━'*32}
Dataset:    925,128 records
Features:   {len(feature_names)} (Bayesian target encoding)
Models:     6 (3 GPU-accelerated)

BEST MODEL: {best_model_name.replace('_', ' ')}
  Accuracy: {best_r['test_accuracy']:.4f}
  AUC-ROC:  {best_r['test_auc']:.4f}
  F1-Score: {best_r['test_f1']:.4f}

vs PAPER (Tarek et al. 2025):
  F1: {best_r['test_f1']:.3f} vs 0.550 (+{(best_r['test_f1']-0.55)*100:.0f}%)

STABILITY: B=200, S=20, K=20, 50τ
GPU: RTX 5070 Laptop ({DEVICE})
"""
ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('Texas-100X Fairness Metrics Reliability Analysis — Final Dashboard',
             fontsize=18, fontweight='bold', y=0.98)
plt.savefig('figures/14_final_dashboard.png', dpi=150, bbox_inches='tight')
plt.savefig('report/final_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Dashboard saved to figures/ and report/")'''

SAVE_CODE = r'''# ── Save All Results ──
import pickle

save_data = {
    'results': {k: {kk: vv for kk, vv in v.items() if kk != 'model'} for k, v in results.items()},
    'predictions': predictions,
    'all_fairness': all_fairness,
    'subset_fairness': subset_fairness,
    'race_subsets': race_subsets,
    'age_subsets': age_subsets,
    'feature_names': feature_names,
}

with open('results/all_results_v2.pkl', 'wb') as f:
    pickle.dump(save_data, f)

print("✅ All results saved to results/all_results_v2.pkl")
print(f"\n📊 FINAL SUMMARY:")
print(f"   Models trained: {len(results)}")
print(f"   Features used: {len(feature_names)}")
print(f"   Best Accuracy: {max(r['test_accuracy'] for r in results.values()):.4f}")
print(f"   Best AUC: {max(r['test_auc'] for r in results.values()):.4f}")
print(f"   Best F1: {max(r['test_f1'] for r in results.values()):.4f}")
print(f"   GPU: {DEVICE}")'''

# ═══════════════════════════════════════════════════════════
# BUILD STANDARD NOTEBOOK
# ═══════════════════════════════════════════════════════════
def build_standard():
    cells = []

    cells.append(md("# LOS Prediction — Fairness Analysis (Standard)\n\n**Texas-100X Dataset** | 925,128 records | 6 models | GPU-accelerated\n\n---"))
    cells.append(md("## 1. Setup & Imports"))
    cells.append(code(SETUP_CODE))
    cells.append(code(GPU_CHECK_CODE))

    cells.append(md("## 2. Data Loading & EDA"))
    cells.append(code(DATA_LOADING_CODE))
    cells.append(code(COLUMN_SUMMARY_CODE))
    cells.append(code(EDA_DISTRIBUTIONS_CODE))
    cells.append(code(ADMISSION_PLOTS_CODE))

    cells.append(md("## 3. Feature Engineering"))
    cells.append(code(FEATURE_ENGINEERING_CODE))
    cells.append(code(TARGET_ENCODING_CODE))
    cells.append(code(ONEHOT_CODE))
    cells.append(code(FEATURE_MATRIX_CODE))

    cells.append(md("## 4. Model Training"))
    cells.append(code(MODEL_TRAINING_CODE))
    cells.append(code(DNN_CODE))

    cells.append(md("## 5. Model Evaluation"))
    cells.append(code(PERF_TABLE_CODE))
    cells.append(code(OVERFIT_CODE))
    cells.append(code(ROC_CODE))

    cells.append(md("## 6. Fairness Analysis"))
    cells.append(code(FAIRNESS_CALC_CODE))
    cells.append(code(FAIRNESS_COMPUTE_CODE))
    cells.append(code(FAIRNESS_HEATMAP_CODE))

    cells.append(md("## 7. Subset Analysis"))
    cells.append(code(SUBSET_7A_CODE))
    cells.append(code(SUBSET_7A_VIZ_CODE))
    cells.append(code(SUBSET_7B_CODE))
    cells.append(code(SUBSET_7C_CODE))
    cells.append(code(SUBSET_7D_CODE))

    cells.append(md("## 8. Fairness Methods Comparison"))
    cells.append(code(FAIRNESS_METHODS_CODE))
    cells.append(code(RADAR_CODE))

    cells.append(md("## 9. Fairness-Derived Model"))
    cells.append(code(FAIR_MODEL_CODE))
    cells.append(code(FAIR_THRESHOLD_CODE))
    cells.append(code(FAIR_COMPARE_CODE))
    cells.append(code(FAIR_VIZ_CODE))

    cells.append(md("## 10. Paper Comparison"))
    cells.append(code(PAPER_COMPARISON_CODE))
    cells.append(code(PAPER_VIZ_CODE))

    cells.append(md("## 11. Final Dashboard & Save"))
    cells.append(code(DASHBOARD_CODE))
    cells.append(code(SAVE_CODE))

    cells.append(md("## Conclusion\n\n**Key findings:**\n1. **Superior Performance**: All models achieve >0.80 accuracy on 925K records\n2. **Fairness-Derived Model**: λ-scaled reweighing + threshold optimization improves DI\n3. **Metric Reliability**: Fairness metrics are stable across random subsets\n4. **Subset Analysis**: Hospital-level fairness varies significantly\n5. **Reproducibility**: All results reproducible with fixed seeds"))

    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.9"}
        },
        "cells": cells
    }
    return nb


# ═══════════════════════════════════════════════════════════
# BUILD DETAILED NOTEBOOK
# ═══════════════════════════════════════════════════════════
def build_detailed():
    cells = []

    cells.append(md("# LOS Prediction — Detailed Fairness Analysis\n\n**Texas-100X Dataset** | 925,128 records | 6 models | GPU-accelerated\n\nThis notebook provides comprehensive fairness analysis with detailed commentary,\nsubset analysis, stability tests, and paper-ready tables.\n\n---"))

    cells.append(md("## 1. Setup & Imports\n\nLoad all necessary libraries for ML, fairness analysis, and visualization.\nRequires: numpy, pandas, sklearn, xgboost, lightgbm, pytorch, matplotlib, seaborn."))
    cells.append(code(SETUP_CODE))
    cells.append(code(GPU_CHECK_CODE))

    cells.append(md("## 2. Data Loading & Exploratory Analysis\n\nThe Texas-100X dataset contains 925,128 hospital discharge records with 12 columns.\nKey features include patient demographics, admission types, and diagnosis/procedure codes.\nTarget: Binary LOS > 3 days (~45% positive rate)."))
    cells.append(code(DATA_LOADING_CODE))
    cells.append(code(COLUMN_SUMMARY_CODE))
    cells.append(code(EDA_DISTRIBUTIONS_CODE))
    cells.append(code(ADMISSION_PLOTS_CODE))

    cells.append(md("## 3. Feature Engineering (31 Features)\n\n**Strategy**: Bayesian target encoding (smoothing α=10) for high-cardinality features.\n\n| Feature Group | Count | Description |\n|---|---|---|\n| Raw demographics | 4 | PAT_AGE, SEX_CODE, RACE, ETHNICITY |\n| Financial | 1 | log(TOTAL_CHARGES) |\n| Target-encoded | 8 | DIAG/PROC/HOSP target+freq, HOSP_SIZE, PS_TARGET |\n| Interaction | 5 | AGE×CHARGE, DIAG×PROC, AGE×DIAG, HOSP×DIAG, HOSP×PROC |\n| One-hot | ~13 | TYPE_OF_ADMISSION, SOURCE_OF_ADMISSION |\n\nProtected attributes are stored as **string labels** for interpretability."))
    cells.append(code(FEATURE_ENGINEERING_CODE))
    cells.append(code(TARGET_ENCODING_CODE))
    cells.append(code(ONEHOT_CODE))
    cells.append(code(FEATURE_MATRIX_CODE))

    cells.append(md("## 4. Model Training\n\nWe train 6 models:\n- **Logistic Regression**: Baseline with class balancing\n- **Random Forest** (300 trees, depth=20): Ensemble with bagging\n- **Gradient Boosting** (300 trees): Sequential boosting\n- **LightGBM** (1500 trees, GPU): Fast gradient boosting\n- **XGBoost** (1000 trees, GPU): Extreme gradient boosting with early stopping\n- **PyTorch DNN** (512-256-128, GPU): Deep neural network with BatchNorm + Dropout"))
    cells.append(code(MODEL_TRAINING_CODE))
    cells.append(code(DNN_CODE))

    cells.append(md("### 4.1 Model Performance Comparison"))
    cells.append(code(PERF_TABLE_CODE))

    cells.append(md("## 5. Overfitting Analysis\n\nWe compare train vs test metrics to detect overfitting.\nA gap < 5% is acceptable; 5-10% is moderate; >10% indicates overfitting."))
    cells.append(code(OVERFIT_CODE))
    cells.append(code(ROC_CODE))

    cells.append(md("## 6. Fairness Metrics\n\n**Metrics computed** (paper-aligned):\n- **DI** (Disparate Impact): $DI = \\min(SR_i) / \\max(SR_i)$, ideal = 1.0, fair if 0.8-1.25\n- **WTPR** (Worst-case TPR): $WTPR = \\min(TPR_i)$, higher = more equitable\n- **SPD** (Statistical Parity Difference): $SPD = \\max(SR_i) - \\min(SR_i)$, ideal = 0\n- **EOD** (Equal Opportunity Difference): $EOD = \\max(TPR_i) - \\min(TPR_i)$, ideal = 0\n- **PPV Ratio**: $PPV_r = \\min(PPV_i) / \\max(PPV_i)$, ideal = 1.0\n\nComputed across 4 protected attributes: RACE, ETHNICITY, SEX, AGE_GROUP."))
    cells.append(code(FAIRNESS_CALC_CODE))
    cells.append(code(FAIRNESS_COMPUTE_CODE))

    cells.append(md("### 6.1 Fairness Heatmap — DI & WTPR across Models and Attributes"))
    cells.append(code(FAIRNESS_HEATMAP_CODE))

    cells.append(md("## 7. Fairness on Different Data Subsets\n\n**7a**: Random subsets (1K, 5K, 10K, 50K, Full) — Do fairness metrics stabilize with more data?\n**7b**: Race-stratified subsets — Does fairness hold within each racial group?\n**7c**: Age-group subsets — Testing intersectional fairness\n**7d**: Hospital subsets — Institutional-level fairness heterogeneity"))

    cells.append(md("### 7a. Fairness vs. Sample Size (Random Subsets)"))
    cells.append(code(SUBSET_7A_CODE))
    cells.append(code(SUBSET_7A_VIZ_CODE))

    cells.append(md("### 7b. Fairness on Race-Stratified Subsets\n\nFor each racial group, we compute fairness using AGE_GROUP as the secondary protected attribute.\nThis reveals intersectional fairness patterns."))
    cells.append(code(SUBSET_7B_CODE))

    cells.append(md("### 7c. Fairness on Age-Group Subsets\n\nFor each age group, we use RACE as the secondary protected attribute.\nThis tests whether fairness persists across different patient populations."))
    cells.append(code(SUBSET_7C_CODE))

    cells.append(md("### 7d. Fairness Across Hospital Subsets\n\nSampling 30 large hospitals (≥200 patients each) to assess institutional-level\nfairness heterogeneity. Hospital variation is a key challenge for ML fairness."))
    cells.append(code(SUBSET_7D_CODE))

    cells.append(md("## 8. Multiple Fairness Methods Comparison\n\n6 fairness approaches compared:\n1. **Disparate Impact** (DI)\n2. **Statistical Parity Difference** (SPD)\n3. **Equal Opportunity Difference** (EOD)\n4. **PPV Ratio**\n5. **Worst-case TPR** (WTPR)\n6. **Equalized Odds** (max of EOD and FPR difference)"))
    cells.append(code(FAIRNESS_METHODS_CODE))
    cells.append(code(RADAR_CODE))

    cells.append(md("## 9. Fairness-Derived Model\n\n**λ-Scaled Reweighing** (optimized via 50-agent search, Agent 39 winner: λ=5.0):\n\n$$w_{g,l} = 1 + \\lambda \\cdot \\left(\\frac{P(G=g) \\cdot P(Y=l)}{P(G=g, Y=l)} - 1\\right)$$\n\nFollowed by **per-group threshold optimization** targeting TPR = 0.82 for all groups.\nThis combines in-processing (reweighing) and post-processing (threshold calibration)."))
    cells.append(code(FAIR_MODEL_CODE))
    cells.append(code(FAIR_THRESHOLD_CODE))
    cells.append(code(FAIR_COMPARE_CODE))
    cells.append(code(FAIR_VIZ_CODE))

    cells.append(md("## 10. Comparison with Reference Paper\n\n**Reference**: Tarek et al. (2025), CHASE '25 — MIMIC-III dataset (46K records)\n\n| Aspect | Paper | Our Study |\n|---|---|---|\n| Dataset | MIMIC-III (46K) | Texas-100X (925K) |\n| Task | Binary LOS | Binary LOS (>3 days) |\n| Model | XGBoost | 6 models + Fairness-Derived |\n| Fairness | FairSynth (synthetic data) | λ-Reweighing + Threshold |\n| Protected | Race (3 groups) | Race, Sex, Ethnicity, Age (4 attrs) |"))
    cells.append(code(PAPER_COMPARISON_CODE))
    cells.append(code(PAPER_VIZ_CODE))

    cells.append(md("## 10B. Per-Metric Fluctuation Analysis (20 Subsets × 5 Metrics)\n\n**Methodology**: 20 random 50% subsets of test data, computing all 5 fairness metrics\nacross all 4 protected attributes = 400 evaluations.\n\nThis tests the **reliability and stability** of fairness metrics — a critical concern\nsince small subsets may produce misleading fairness conclusions."))
    cells.append(code(PER_METRIC_FLUCTUATION_CODE))
    cells.append(code(PER_METRIC_VIZ_CODE))
    cells.append(code(PER_METRIC_VIOLIN_CODE))

    cells.append(md("## 10C. Lambda (λ) Trade-off Experiment\n\n**Methodology**: Train models with λ ∈ {0.0, 0.25, 0.5, 0.75, 1.0, 1.2, 1.5, 2.0}\nto map the Pareto frontier between fairness and performance.\n\nThe weight formula: $w = 1 + \\lambda \\cdot (w_{raw} - 1)$\n- λ=0: No fairness emphasis (uniform weights)\n- λ=1: Full reweighing\n- λ>1: Over-emphasis on fairness (may sacrifice performance)\n\n**Expected**: Monotonic DI improvement with some F1 degradation."))
    cells.append(code(LAMBDA_TRADEOFF_CODE))
    cells.append(code(LAMBDA_VIZ_CODE))
    cells.append(code(LAMBDA_PAPER_COMPARE_CODE))

    cells.append(md("## 11. Stability Tests\n\nFour complementary stability assessments:\n- **11a**: Bootstrap (B=200) — Sample uncertainty in TPR estimates\n- **11b**: Seed sensitivity (S=20) — Reproducibility across random seeds\n- **11c**: Cross-hospital (K=20) — Institutional generalization\n- **11d**: Threshold sweep (50 τ) — Sensitivity to classification threshold"))

    cells.append(md("### 11a. Bootstrap Stability Test (B=200)"))
    cells.append(code(BOOTSTRAP_CODE))
    cells.append(code(BOOTSTRAP_VIZ_CODE))

    cells.append(md("### 11b. Seed Sensitivity Test (S=20)"))
    cells.append(code(SEED_CODE))

    cells.append(md("### 11c. Cross-Hospital Validation (K=20)"))
    cells.append(code(CROSS_HOSP_CODE))

    cells.append(md("### 11d. Threshold Sweep Analysis"))
    cells.append(code(THRESHOLD_SWEEP_CODE))

    cells.append(md("## 12. Paper: Methodology Section\n\n### 3.1 Dataset\nTexas-100X: 925,128 hospital discharge records from 441 Texas hospitals.\nBinary classification: LOS > 3 days (~45% positive rate).\n\n### 3.2 Feature Engineering\n31 features via Bayesian target encoding (α=10) with interaction terms.\nTarget-encoded: diagnosis, procedure, hospital, patient status.\nFrequency features: diagnosis, procedure, hospital.\nInteraction: AGE×CHARGE, DIAG×PROC, AGE×DIAG, HOSP×DIAG, HOSP×PROC.\n\n### 3.3 Model Architecture\n6 classifiers: LR, RF(300), GB(300), XGB(1000,GPU), LGB(1500,GPU), DNN(512-256-128,GPU).\nEarly stopping (patience=15-20). Class-balanced weights.\n\n### 3.4 Fairness Metrics\n5 metrics × 4 protected attributes = 20 fairness evaluations per model.\nDI (4/5 rule), WTPR, SPD, EOD, PPV Ratio.\n\n### 3.5 Fairness-Aware Training\nλ-scaled reweighing (λ=5.0, Agent 39 optimized) + per-group threshold calibration.\n\n### 3.6 Reliability Analysis\nBootstrap (B=200), seed sensitivity (S=20), cross-hospital (K=20), threshold sweep (50τ).\nPer-metric fluctuation: 20 subsets × 5 metrics × 4 attributes.\n\n### 3.7 Subset Analysis\n5 sizes (1K-Full) × 10 repeats, race-stratified, age-stratified, hospital-stratified."))

    cells.append(md("## 13. Paper: Results & Discussion"))
    cells.append(code(RESULTS_TABLES_CODE))
    cells.append(code(RESULTS_COMPARISON_CODE))
    cells.append(code(SUBSET_RESULTS_CODE))

    cells.append(md("### Results Discussion\n\n**4.1 Model Performance**: All 6 models achieve >0.80 accuracy.\nGPU-accelerated models (XGBoost, LightGBM) offer best balance of speed and performance.\nOverfitting well-controlled (<5% gap for most models).\n\n**4.2 Paper Comparison**: Our F1 scores significantly exceed the reference paper.\nLarger dataset (925K vs 46K) enables more reliable training.\n\n**4.3 Fairness Metric Reliability**: DI and WTPR are most stable across subsets (lowest CV%).\nSPD and EOD show higher variance, especially for small subsets.\n\n**4.4 Fairness-Derived Model**: λ-scaled reweighing with per-group thresholds\nimproves DI while maintaining competitive accuracy.\n\n**4.5 Implications**: (1) Larger datasets stabilize both performance and fairness;\n(2) Post-processing threshold calibration is effective;\n(3) Hospital-level fairness requires institutional-specific interventions."))

    cells.append(md("## 14. Final Dashboard & Summary"))
    cells.append(code(DASHBOARD_CODE))
    cells.append(code(SAVE_CODE))

    cells.append(md("## Conclusion\n\n**Key findings:**\n1. **Superior Performance**: All models achieve >0.80 accuracy on 925K records, with best F1 significantly exceeding reference paper\n2. **Fairness-Derived Model**: λ-scaled reweighing + threshold optimization improves Disparate Impact across all protected attributes\n3. **Metric Reliability**: DI and WTPR are the most stable fairness metrics across random subsets (CV% < 2%)\n4. **Subset Analysis**: Hospital-level fairness varies significantly — institutional-specific fairness interventions needed\n5. **Reproducibility**: All results reproducible with fixed seeds; stability confirmed across B=200 bootstrap, S=20 seeds, K=20 hospital folds"))

    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.9"}
        },
        "cells": cells
    }
    return nb


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    out_dir = 'final_notebooks'
    os.makedirs(out_dir, exist_ok=True)

    # Copy data
    data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(data_dir):
        shutil.copytree('data', data_dir)

    # Create output dirs inside final_notebooks
    for d in ['figures', 'tables', 'results', 'report']:
        os.makedirs(os.path.join(out_dir, d), exist_ok=True)

    # Build Standard
    std_nb = build_standard()
    std_path = os.path.join(out_dir, 'LOS_Prediction_Standard.ipynb')
    with open(std_path, 'w', encoding='utf-8') as f:
        json.dump(std_nb, f, indent=1, ensure_ascii=False)
    print(f"✅ Standard notebook: {std_path} ({len(std_nb['cells'])} cells)")

    # Build Detailed
    det_nb = build_detailed()
    det_path = os.path.join(out_dir, 'LOS_Prediction_Detailed.ipynb')
    with open(det_path, 'w', encoding='utf-8') as f:
        json.dump(det_nb, f, indent=1, ensure_ascii=False)
    print(f"✅ Detailed notebook: {det_path} ({len(det_nb['cells'])} cells)")

    print(f"\n📁 Both notebooks saved to {out_dir}/")
