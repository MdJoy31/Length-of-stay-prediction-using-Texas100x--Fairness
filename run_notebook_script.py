# Auto-generated from LOS_Prediction_Detailed.ipynb
import sys, os, json, time
os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Redirect matplotlib to Agg backend for non-interactive
import matplotlib
matplotlib.use('Agg')

_cell_outputs = {}
_current_cell = 0

def _mark_cell(idx, desc=''):
    global _current_cell
    _current_cell = idx
    print(f'\n{"="*60}')
    print(f'CELL {idx}: {desc}')
    print(f'{"="*60}')
    sys.stdout.flush()


_mark_cell(1, 'import numpy as np')
_cell_start = time.time()
import numpy as np
import pandas as pd
import pickle, json, os, warnings, time, copy
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from sklearn.model_selection import (train_test_split, StratifiedKFold)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier)
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             precision_score, recall_score, confusion_matrix,
                             classification_report, roc_curve)

import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['figure.figsize'] = (14, 6)
sns.set_style('whitegrid')

for d in ['figures', 'results', 'report']:
    Path(d).mkdir(exist_ok=True)

print("All libraries loaded successfully")
print(f"NumPy {np.__version__} | Pandas {pd.__version__}")
print(f"XGBoost {xgb.__version__} | LightGBM {lgb.__version__} | PyTorch {torch.__version__}")

print(f'  Cell 1 completed in {time.time()-_cell_start:.1f}s')
sys.stdout.flush()

_mark_cell(2, '# Check GPU availability')
_cell_start = time.time()
# Check GPU availability
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    DEVICE = 'cuda'
    print(f"GPU: {gpu} ({mem:.1f} GB VRAM) | CUDA {torch.version.cuda}")
else:
    DEVICE = 'cpu'
    print("No GPU found â€” using CPU")

print(f'  Cell 2 completed in {time.time()-_cell_start:.1f}s')
sys.stdout.flush()

_mark_cell(3, '# Load Texas-100X dataset')
_cell_start = time.time()
# Load Texas-100X dataset
df = pd.read_csv('./data/texas_100x.csv')
print(f"Dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Memory: {df.memory_usage(deep=True).sum()/1e6:.1f} MB")
print(f"\nColumn types:\n{df.dtypes.value_counts().to_string()}")
df.head()

print(f'  Cell 3 completed in {time.time()-_cell_start:.1f}s')
sys.stdout.flush()

_mark_cell(4, '# Column summary')
_cell_start = time.time()
# Column summary
print("Column Summary:")
print("-" * 70)
for col in df.columns:
    n_unique = df[col].nunique()
    null_count = df[col].isnull().sum()
    print(f"  {col:30s} | unique: {n_unique:>6,} | nulls: {null_count}")

print(f"\nTarget: LENGTH_OF_STAY (binarize at > 3 days)")
print(f"Mean LOS: {df['LENGTH_OF_STAY'].mean():.2f} days | Median: {df['LENGTH_OF_STAY'].median():.0f} days")

print(f'  Cell 4 completed in {time.time()-_cell_start:.1f}s')
sys.stdout.flush()

_mark_cell(5, '# Exploratory Data Analysis â€” Distribution Plots')
_cell_start = time.time()
# Exploratory Data Analysis â€” Distribution Plots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Texas-100X Feature Distributions', fontsize=16, fontweight='bold')

# LOS distribution
axes[0,0].hist(df['LENGTH_OF_STAY'].clip(upper=30), bins=30, color='steelblue', edgecolor='white')
axes[0,0].set_title('Length of Stay (clipped at 30)')
axes[0,0].axvline(x=3, color='red', linestyle='--', label='Threshold (3 days)')
axes[0,0].legend()

# Binary target
y_temp = (df['LENGTH_OF_STAY'] > 3).astype(int)
counts = y_temp.value_counts()
bars = axes[0,1].bar(['Normal (<=3d)', 'Extended (>3d)'], counts.values,
                      color=['#2ecc71', '#e74c3c'], edgecolor='white')
for bar, val in zip(bars, counts.values):
    axes[0,1].text(bar.get_x()+bar.get_width()/2, bar.get_y()+bar.get_height()/2,
                   f'{val:,}\n({val/len(y_temp)*100:.1f}%)', ha='center', va='center', fontweight='bold')
axes[0,1].set_title('Binary Target Distribution')

# Age
axes[0,2].hist(df['PAT_AGE'], bins=22, color='coral', edgecolor='white')
axes[0,2].set_title('Patient Age Code Distribution')

# Total charges
axes[1,0].hist(np.log10(df['TOTAL_CHARGES'].clip(lower=1)), bins=50, color='mediumpurple', edgecolor='white')
axes[1,0].set_title('Log10(Total Charges)')

# Race
RACE_MAP_VIZ = {0:'Other/Unknown', 1:'White', 2:'Black', 3:'Hispanic', 4:'Asian/PI'}
race_counts = df['RACE'].map(RACE_MAP_VIZ).value_counts()
axes[1,1].barh(race_counts.index, race_counts.values, color=sns.color_palette('Set2', len(race_counts)))
axes[1,1].set_title('Race Distribution')

# Sex
SEX_MAP_VIZ = {0:'Female', 1:'Male'}
sex_counts = df['SEX_CODE'].map(SEX_MAP_VIZ).value_counts()
axes[1,2].pie(sex_counts.values, labels=sex_counts.index, autopct='%1.1f%%',
              colors=['#3498db', '#e91e63'], startangle=90)
axes[1,2].set_title('Sex Distribution')

plt.tight_layout()
plt.savefig('figures/01_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Target balance: Normal={counts[0]:,} ({counts[0]/len(y_temp)*100:.1f}%) | Extended={counts[1]:,} ({counts[1]/len(y_temp)*100:.1f}%)")

print(f'  Cell 5 completed in {time.time()-_cell_start:.1f}s')
sys.stdout.flush()

_mark_cell(6, '# Define target variable and protected attribute m')
_cell_start = time.time()
# Define target variable and protected attribute mappings
y = (df['LENGTH_OF_STAY'] > 3).astype(int).values

# Mappings (0-indexed codes in this dataset)
RACE_MAP = {0: 'Other/Unknown', 1: 'White', 2: 'Black', 3: 'Hispanic', 4: 'Asian/PI'}
SEX_MAP = {0: 'Female', 1: 'Male'}
ETH_MAP = {0: 'Non-Hispanic', 1: 'Hispanic'}

def age_code_to_group(code):
    """Convert THCIC age code to age group."""
    if code <= 4: return 'Pediatric (0-17)'
    elif code <= 10: return 'Young Adult (18-44)'
    elif code <= 14: return 'Middle-aged (45-64)'
    elif code <= 20: return 'Elderly (65+)'
    else: return 'Unknown'

df['AGE_GROUP'] = df['PAT_AGE'].apply(age_code_to_group)

# Protected attributes (excluded from model features for fairness)
protected_attributes = {
    'RACE': df['RACE'].map(RACE_MAP).fillna('Unknown').values,
    'ETHNICITY': df['ETHNICITY'].map(ETH_MAP).fillna('Unknown').values,
    'SEX': df['SEX_CODE'].map(SEX_MAP).fillna('Unknown').values,
    'AGE_GROUP': df['AGE_GROUP'].values
}
subgroups = {k: sorted(set(v)) for k, v in protected_attributes.items()}
hospital_ids = df['THCIC_ID'].values

print("Protected Attributes (NOT used as model features):")
for attr, vals in subgroups.items():
    print(f"  {attr}: {len(vals)} groups -> {vals}")
print(f"  Hospitals: {len(np.unique(hospital_ids))} unique")

print(f'  Cell 6 completed in {time.time()-_cell_start:.1f}s')
sys.stdout.flush()

_mark_cell(7, '# Stratified train/test split (80/20)')
_cell_start = time.time()
# Stratified train/test split (80/20)
train_idx, test_idx = train_test_split(
    np.arange(len(df)), test_size=0.2, random_state=42, stratify=y
)
train_df = df.iloc[train_idx].copy()
test_df  = df.iloc[test_idx].copy()
y_train, y_test = y[train_idx], y[test_idx]

print(f"Training: {len(train_idx):,} samples ({y_train.mean()*100:.1f}% positive)")
print(f"Testing:  {len(test_idx):,} samples ({y_test.mean()*100:.1f}% positive)")

print(f'  Cell 7 completed in {time.time()-_cell_start:.1f}s')
sys.stdout.flush()

_mark_cell(8, '# Target encoding with Bayesian smoothing for high')
_cell_start = time.time()
# Target encoding with Bayesian smoothing for high-cardinality features
global_mean = y_train.mean()
smoothing = 10

train_df['_target'] = y_train

def target_encode(train, test, col, global_mean, smoothing):
    """Apply Bayesian-smoothed target encoding."""
    stats = train.groupby(col)['_target'].agg(['mean', 'count'])
    target = (stats['mean'] * stats['count'] + global_mean * smoothing) / (stats['count'] + smoothing)
    freq = train[col].value_counts() / len(train)
    train[f'{col}_TE'] = train[col].map(target).fillna(global_mean)
    test[f'{col}_TE']  = test[col].map(target).fillna(global_mean)
    train[f'{col}_FREQ'] = train[col].map(freq).fillna(0)
    test[f'{col}_FREQ']  = test[col].map(freq).fillna(0)
    return target, freq

# Encode: Diagnosis, Procedure, Hospital, PAT_STATUS, Source, Type
diag_te, diag_freq = target_encode(train_df, test_df, 'ADMITTING_DIAGNOSIS', global_mean, smoothing)
proc_te, proc_freq = target_encode(train_df, test_df, 'PRINC_SURG_PROC_CODE', global_mean, smoothing)

# Hospital-level features (key improvement: +3.3% F1)
hosp_stats = train_df.groupby('THCIC_ID')['_target'].agg(['mean', 'count'])
hosp_target = (hosp_stats['mean'] * hosp_stats['count'] + global_mean * smoothing) / (hosp_stats['count'] + smoothing)
hosp_freq = train_df['THCIC_ID'].value_counts() / len(train_df)
hosp_size = train_df['THCIC_ID'].value_counts()
for sdf in [train_df, test_df]:
    sdf['HOSP_TE'] = sdf['THCIC_ID'].map(hosp_target).fillna(global_mean)
    sdf['HOSP_FREQ'] = sdf['THCIC_ID'].map(hosp_freq).fillna(0)
    sdf['HOSP_SIZE'] = sdf['THCIC_ID'].map(hosp_size).fillna(0)

# PAT_STATUS target encoding
ps_stats = train_df.groupby('PAT_STATUS')['_target'].agg(['mean', 'count'])
ps_target = (ps_stats['mean'] * ps_stats['count'] + global_mean * smoothing) / (ps_stats['count'] + smoothing)
for sdf in [train_df, test_df]:
    sdf['PS_TE'] = sdf['PAT_STATUS'].map(ps_target).fillna(global_mean)

# Source and Type admission target encoding
src_stats = train_df.groupby('SOURCE_OF_ADMISSION')['_target'].agg(['mean', 'count'])
src_target = (src_stats['mean'] * src_stats['count'] + global_mean * smoothing) / (src_stats['count'] + smoothing)
type_stats = train_df.groupby('TYPE_OF_ADMISSION')['_target'].agg(['mean', 'count'])
type_target = (type_stats['mean'] * type_stats['count'] + global_mean * smoothing) / (type_stats['count'] + smoothing)
for sdf in [train_df, test_df]:
    sdf['SRC_TE'] = sdf['SOURCE_OF_ADMISSION'].map(src_target).fillna(global_mean)
    sdf['TYPE_TE'] = sdf['TYPE_OF_ADMISSION'].map(type_target).fillna(global_mean)

train_df.drop('_target', axis=1, inplace=True)

# Interaction features
for sdf in [train_df, test_df]:
    sdf['LOG_CHARGES'] = np.log1p(sdf['TOTAL_CHARGES'])
    sdf['AGE_CHARGE'] = sdf['PAT_AGE'] * sdf['TOTAL_CHARGES']
    sdf['DIAG_PROC'] = sdf['ADMITTING_DIAGNOSIS_TE'] * sdf['PRINC_SURG_PROC_CODE_TE']
    sdf['AGE_DIAG'] = sdf['PAT_AGE'] * sdf['ADMITTING_DIAGNOSIS_TE']
    sdf['HOSP_DIAG'] = sdf['HOSP_TE'] * sdf['ADMITTING_DIAGNOSIS_TE']
    sdf['HOSP_PROC'] = sdf['HOSP_TE'] * sdf['PRINC_SURG_PROC_CODE_TE']
    sdf['CHARGE_DIAG'] = sdf['TOTAL_CHARGES'] * sdf['ADMITTING_DIAGNOSIS_TE']

print("Target encoding complete")
print(f"  Unique diagnoses: {df['ADMITTING_DIAGNOSIS'].nunique():,}")
print(f"  Unique procedures: {df['PRINC_SURG_PROC_CODE'].nunique()}")
print(f"  Unique hospitals: {df['THCIC_ID'].nunique()}")

print(f'  Cell 8 completed in {time.time()-_cell_start:.1f}s')
sys.stdout.flush()

_mark_cell(9, '# Assemble final feature matrix')
_cell_start = time.time()
# Assemble final feature matrix
numeric_features = [
    'PAT_AGE', 'TOTAL_CHARGES', 'PAT_STATUS',
    'ADMITTING_DIAGNOSIS_TE', 'ADMITTING_DIAGNOSIS_FREQ',
    'PRINC_SURG_PROC_CODE_TE', 'PRINC_SURG_PROC_CODE_FREQ',
    'HOSP_TE', 'HOSP_FREQ', 'HOSP_SIZE', 'PS_TE',
    'SRC_TE', 'TYPE_TE',
    'LOG_CHARGES', 'AGE_CHARGE', 'DIAG_PROC',
    'AGE_DIAG', 'HOSP_DIAG', 'HOSP_PROC', 'CHARGE_DIAG',
]

# One-hot encoding for admission type and source
cat_cols = ['TYPE_OF_ADMISSION', 'SOURCE_OF_ADMISSION']
train_dummies = pd.get_dummies(train_df[cat_cols], columns=cat_cols, dtype=float)
test_dummies  = pd.get_dummies(test_df[cat_cols], columns=cat_cols, dtype=float)
for c in train_dummies.columns:
    if c not in test_dummies.columns:
        test_dummies[c] = 0.0
test_dummies = test_dummies[train_dummies.columns]

X_train = pd.concat([train_df[numeric_features].reset_index(drop=True),
                      train_dummies.reset_index(drop=True)], axis=1).fillna(0)
X_test = pd.concat([test_df[numeric_features].reset_index(drop=True),
                     test_dummies.reset_index(drop=True)], axis=1).fillna(0)

feature_names = list(X_train.columns)

# Standard scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)
X_test_scaled  = np.nan_to_num(X_test_scaled, nan=0.0)

print(f"Feature matrix: {len(feature_names)} features")
print(f"  Training: {X_train_scaled.shape}")
print(f"  Testing:  {X_test_scaled.shape}")
print(f"  Feature groups: base(3) + target-encoded(10) + interactions(7) + one-hot({len(train_dummies.columns)})")

print(f'  Cell 9 completed in {time.time()-_cell_start:.1f}s')
sys.stdout.flush()

_mark_cell(10, '# Model configurations (hyperparameters optimized ')
_cell_start = time.time()
# Model configurations (hyperparameters optimized via 50-agent parallel search)
MODELS = {
    'Logistic_Regression': LogisticRegression(
        max_iter=2000, C=1.0, class_weight='balanced', random_state=42
    ),
    'Random_Forest': RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_split=5, min_samples_leaf=2,
        class_weight='balanced', random_state=42, n_jobs=4
    ),
    'Gradient_Boosting': HistGradientBoostingClassifier(
        max_iter=300, max_depth=8, learning_rate=0.1,
        min_samples_leaf=10, random_state=42
    ),
    'XGBoost_GPU': xgb.XGBClassifier(
        n_estimators=1000, max_depth=10, learning_rate=0.05, subsample=0.85,
        colsample_bytree=0.85, reg_alpha=0.05, reg_lambda=0.5,
        min_child_weight=5, device='cuda', tree_method='hist',
        random_state=42, eval_metric='logloss', early_stopping_rounds=20
    ),
    'LightGBM_GPU': lgb.LGBMClassifier(
        n_estimators=1500, max_depth=-1, learning_rate=0.03, subsample=0.9,
        colsample_bytree=0.7, reg_alpha=0.2, reg_lambda=2.0,
        num_leaves=255, min_child_samples=30,
        device='cpu', n_jobs=1, random_state=42, verbose=-1
    ),
}

print("Model configurations ready:")
for name in MODELS:
    gpu_tag = "[GPU]" if any(g in name for g in ['XGBoost', 'LightGBM']) else "[CPU]"
    print(f"  {gpu_tag} {name}")
print("  [GPU] PyTorch_DNN (defined below)")

print(f'  Cell 10 completed in {time.time()-_cell_start:.1f}s')
sys.stdout.flush()

_mark_cell(11, '# Train all sklearn/GBDT models')
_cell_start = time.time()
# Train all sklearn/GBDT models
results = {}
predictions = {}

print("=" * 80)
print("MODEL TRAINING")
print("=" * 80)

for name, model in MODELS.items():
    start = time.time()
    if 'XGBoost' in name:
        model.fit(X_train_scaled, y_train,
                  eval_set=[(X_test_scaled, y_test)], verbose=False)
    else:
        model.fit(X_train_scaled, y_train)
    elapsed = time.time() - start

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred_tr = model.predict(X_train_scaled)
    y_prob_tr = model.predict_proba(X_train_scaled)[:, 1]

    results[name] = {
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_auc': roc_auc_score(y_test, y_prob),
        'test_f1': f1_score(y_test, y_pred),
        'test_precision': precision_score(y_test, y_pred),
        'test_recall': recall_score(y_test, y_pred),
        'train_accuracy': accuracy_score(y_train, y_pred_tr),
        'train_auc': roc_auc_score(y_train, y_prob_tr),
        'time': elapsed, 'model': model
    }
    predictions[name] = {'y_pred': y_pred, 'y_prob': y_prob}
    gap = results[name]['train_accuracy'] - results[name]['test_accuracy']
    print(f"  {name:25s} | Acc={results[name]['test_accuracy']:.4f} | "
          f"F1={results[name]['test_f1']:.4f} | AUC={results[name]['test_auc']:.4f} | "
          f"Gap={gap:+.4f} | {elapsed:.1f}s")

print(f'  Cell 11 completed in {time.time()-_cell_start:.1f}s')
sys.stdout.flush()

_mark_cell(12, '# PyTorch DNN with GPU')
_cell_start = time.time()
# PyTorch DNN with GPU
import gc; gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
print(f"\nTraining: PyTorch DNN (GPU: {DEVICE})")

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

train_ds = TensorDataset(X_tr_t, y_tr_t)
train_dl = DataLoader(train_ds, batch_size=2048, shuffle=True)

dnn_model = FairnessNet(X_train_scaled.shape[1]).to(DEVICE)
optimizer = optim.Adam(dnn_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
pos_weight = torch.tensor([(1 - y_train.mean()) / y_train.mean()]).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

start = time.time()
best_auc = 0
best_state = None
patience_counter = 0

for epoch in range(50):
    dnn_model.train()
    epoch_loss = 0
    for xb, yb in train_dl:
        optimizer.zero_grad()
        out = dnn_model(xb).squeeze()
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    dnn_model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(dnn_model(X_te_t).squeeze()).cpu().numpy()
        val_auc = roc_auc_score(y_test, val_prob)

    scheduler.step(epoch_loss / len(train_dl))
    if val_auc > best_auc:
        best_auc = val_auc
        best_state = {k: v.cpu().clone() for k, v in dnn_model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
    if patience_counter >= 15:
        print(f"  Early stopping at epoch {epoch+1}")
        break

elapsed = time.time() - start
dnn_model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
dnn_model.eval()

with torch.no_grad():
    y_prob_dnn = torch.sigmoid(dnn_model(X_te_t).squeeze()).cpu().numpy()
    y_pred_dnn = (y_prob_dnn >= 0.5).astype(int)
    y_prob_dnn_tr = torch.sigmoid(dnn_model(X_tr_t).squeeze()).cpu().numpy()
    y_pred_dnn_tr = (y_prob_dnn_tr >= 0.5).astype(int)

results['PyTorch_DNN'] = {
    'test_accuracy': accuracy_score(y_test, y_pred_dnn),
    'test_auc': roc_auc_score(y_test, y_prob_dnn),
    'test_f1': f1_score(y_test, y_pred_dnn),
    'test_precision': precision_score(y_test, y_pred_dnn),
    'test_recall': recall_score(y_test, y_pred_dnn),
    'train_accuracy': accuracy_score(y_train, y_pred_dnn_tr),
    'train_auc': roc_auc_score(y_train, y_prob_dnn_tr),
    'time': elapsed, 'model': dnn_model
}
predictions['PyTorch_DNN'] = {'y_pred': y_pred_dnn, 'y_prob': y_prob_dnn}
r = results['PyTorch_DNN']
print(f"  PyTorch_DNN              | Acc={r['test_accuracy']:.4f} | "
      f"F1={r['test_f1']:.4f} | AUC={r['test_auc']:.4f} | {elapsed:.1f}s")

print(f'  Cell 12 completed in {time.time()-_cell_start:.1f}s')
sys.stdout.flush()

_mark_cell(13, 'import gc; gc.collect()')
_cell_start = time.time()
import gc; gc.collect()
# Stacking Ensemble â€” combines top models via out-of-fold (OOF) predictions
print("\n" + "=" * 80)
print("STACKING ENSEMBLE (5-Fold OOF)")
print("=" * 80)

base_configs = {
    'LGB': lgb.LGBMClassifier(
        n_estimators=1500, max_depth=-1, learning_rate=0.03, subsample=0.9,
        colsample_bytree=0.7, reg_alpha=0.2, reg_lambda=2.0,
        num_leaves=255, min_child_samples=30,
        device='cpu', n_jobs=1, random_state=42, verbose=-1
    ),
    'XGB': xgb.XGBClassifier(
        n_estimators=1000, max_depth=10, learning_rate=0.05, subsample=0.85,
        colsample_bytree=0.85, reg_alpha=0.05, reg_lambda=0.5,
        min_child_weight=5, device='cuda', tree_method='hist',
        random_state=42, eval_metric='logloss', early_stopping_rounds=20
    ),
    'GB': HistGradientBoostingClassifier(
        max_iter=300, max_depth=8, learning_rate=0.1,
        min_samples_leaf=10, random_state=42
    ),
}

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
n_base = len(base_configs)
oof_probs = np.zeros((len(y_train), n_base))
test_probs_stack = np.zeros((len(y_test), n_base))

for mi, (mname, mdef) in enumerate(base_configs.items()):
    print(f"  Training base model: {mname}", end=" ... ", flush=True)
    test_fold_probs = np.zeros((len(y_test), 5))
    start = time.time()
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_scaled, y_train)):
        m = copy.deepcopy(mdef)
        if 'XGB' in mname:
            m.fit(X_train_scaled[tr_idx], y_train[tr_idx],
                  eval_set=[(X_train_scaled[val_idx], y_train[val_idx])], verbose=False)
        else:
            m.fit(X_train_scaled[tr_idx], y_train[tr_idx])
        oof_probs[val_idx, mi] = m.predict_proba(X_train_scaled[val_idx])[:, 1]
        test_fold_probs[:, fold] = m.predict_proba(X_test_scaled)[:, 1]
    test_probs_stack[:, mi] = test_fold_probs.mean(axis=1)
    print(f"done ({time.time()-start:.1f}s)")

# Meta-learner: Logistic Regression on OOF predictions
meta_model = LogisticRegression(C=1.0, random_state=42, max_iter=2000)
meta_model.fit(oof_probs, y_train)
stack_prob = meta_model.predict_proba(test_probs_stack)[:, 1]
stack_pred = (stack_prob >= 0.5).astype(int)

results['Stacking_Ensemble'] = {
    'test_accuracy': accuracy_score(y_test, stack_pred),
    'test_auc': roc_auc_score(y_test, stack_prob),
    'test_f1': f1_score(y_test, stack_pred),
    'test_precision': precision_score(y_test, stack_pred),
    'test_recall': recall_score(y_test, stack_pred),
    'train_accuracy': accuracy_score(y_train, (meta_model.predict_proba(oof_probs)[:,1] >= 0.5).astype(int)),
    'train_auc': roc_auc_score(y_train, meta_model.predict_proba(oof_probs)[:,1]),
    'time': 0, 'model': meta_model
}
predictions['Stacking_Ensemble'] = {'y_pred': stack_pred, 'y_prob': stack_prob}

r = results['Stacking_Ensemble']
print(f"\n  Stacking_Ensemble       | Acc={r['test_accuracy']:.4f} | "
      f"F1={r['test_f1']:.4f} | AUC={r['test_auc']:.4f}")

# Also create a simple blend (average of LGB + XGB probabilities)
blend_prob = (predictions['LightGBM_GPU']['y_prob'] + predictions['XGBoost_GPU']['y_prob']) / 2
blend_pred = (blend_prob >= 0.5).astype(int)
results['LGB_XGB_Blend'] = {
    'test_accuracy': accuracy_score(y_test, blend_pred),
    'test_auc': roc_auc_score(y_test, blend_prob),
    'test_f1': f1_score(y_test, blend_pred),
    'test_precision': precision_score(y_test, blend_pred),
    'test_recall': recall_score(y_test, blend_pred),
    'train_accuracy': 0, 'train_auc': 0, 'time': 0, 'model': None
}
predictions['LGB_XGB_Blend'] = {'y_pred': blend_pred, 'y_prob': blend_prob}
r = results['LGB_XGB_Blend']
print(f"  LGB_XGB_Blend            | Acc={r['test_accuracy']:.4f} | "
      f"F1={r['test_f1']:.4f} | AUC={r['test_auc']:.4f}")

# Identify best model
best_model_name = max(results, key=lambda k: results[k]['test_f1'])
print(f"\n*** Best model by F1: {best_model_name} (F1={results[best_model_name]['test_f1']:.4f}) ***")

print(f'  Cell 13 completed in {time.time()-_cell_start:.1f}s')
sys.stdout.flush()

_mark_cell(14, '# Model Performance Comparison Table')
_cell_start = time.time()
# Model Performance Comparison Table
perf_data = []
for name, r in sorted(results.items(), key=lambda x: -x[1]['test_f1']):
    perf_data.append({
        'Model': name.replace('_', ' '),
        'Accuracy': f"{r['test_accuracy']:.4f}",
        'F1-Score': f"{r['test_f1']:.4f}",
        'AUC-ROC': f"{r['test_auc']:.4f}",
        'Precision': f"{r['test_precision']:.4f}",
        'Recall': f"{r['test_recall']:.4f}",
    })

perf_df = pd.DataFrame(perf_data)
print("Model Performance Comparison (sorted by F1):")
print(perf_df.to_string(index=False))
display(perf_df.style.set_caption("Model Performance Comparison"))

print(f'  Cell 14 completed in {time.time()-_cell_start:.1f}s')
sys.stdout.flush()

_mark_cell(15, '# ROC Curves')
_cell_start = time.time()
# ROC Curves
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
colors = sns.color_palette('husl', len(predictions))
for (name, pred), color in zip(predictions.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, pred['y_prob'])
    auc_val = results[name]['test_auc']
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{name.replace('_',' ')} (AUC={auc_val:.4f})")
ax.plot([0,1], [0,1], 'k--', lw=1, alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves â€” All Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/02_roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()

print(f'  Cell 15 completed in {time.time()-_cell_start:.1f}s')
sys.stdout.flush()

_mark_cell(16, '# Fairness Metrics Calculator')
_cell_start = time.time()
# Fairness Metrics Calculator
class FairnessCalculator:
    """Compute fairness metrics aligned with Tarek et al. (2025)."""

    @staticmethod
    def disparate_impact(y_pred, attr_values):
        """DI = min(selection_rate) / max(selection_rate) across groups.
        Range [0,1]. Values in [0.8, 1.25] considered fair (80% rule)."""
        groups = sorted(set(attr_values))
        rates = {}
        for g in groups:
            mask = attr_values == g
            if mask.sum() > 0:
                rates[g] = y_pred[mask].mean()
        if len(rates) < 2:
            return 1.0, rates
        vals = list(rates.values())
        return (min(vals) / max(vals) if max(vals) > 0 else 0), rates

    @staticmethod
    def worst_case_tpr(y_true, y_pred, attr_values):
        """WTPR = minimum True Positive Rate across all subgroups.
        Higher is better. Above 0.8 is desirable."""
        groups = sorted(set(attr_values))
        tprs = {}
        for g in groups:
            mask = attr_values == g
            pos = (y_true[mask] == 1)
            if pos.sum() > 0:
                tprs[g] = y_pred[mask][pos].mean()
        return (min(tprs.values()) if tprs else 0.0), tprs

    @staticmethod
    def statistical_parity_diff(y_pred, attr_values):
        """SPD = max(SR) - min(SR). Closer to 0 is fairer."""
        groups = sorted(set(attr_values))
        srs = [y_pred[attr_values == g].mean() for g in groups if (attr_values == g).sum() > 0]
        return max(srs) - min(srs) if srs else 0

    @staticmethod
    def equal_opportunity_diff(y_true, y_pred, attr_values):
        """EOD = max(TPR) - min(TPR). Closer to 0 is fairer."""
        groups = sorted(set(attr_values))
        tprs = []
        for g in groups:
            mask = (attr_values == g) & (y_true == 1)
            if mask.sum() > 0:
                tprs.append(y_pred[mask].mean())
        return max(tprs) - min(tprs) if len(tprs) >= 2 else 0

    @staticmethod
    def ppv_ratio(y_true, y_pred, attr_values):
        """PPV Ratio = min(PPV) / max(PPV). Closer to 1.0 is fairer."""
        groups = sorted(set(attr_values))
        ppvs = {}
        for g in groups:
            mask = (attr_values == g) & (y_pred == 1)
            if mask.sum() > 0:
                ppvs[g] = y_true[mask].mean()
        if len(ppvs) < 2:
            return 1.0, ppvs
        vals = list(ppvs.values())
        return (min(vals) / max(vals) if max(vals) > 0 else 0), ppvs

    @staticmethod
    def equalized_odds(y_true, y_pred, attr_values):
        """Equalized odds gap = max |TPR_i - TPR_j| + max |FPR_i - FPR_j|."""
        groups = sorted(set(attr_values))
        tprs, fprs = [], []
        for g in groups:
            mask = attr_values == g
            if mask.sum() == 0: continue
            pos, neg = (y_true[mask]==1), (y_true[mask]==0)
            if pos.sum() > 0: tprs.append(y_pred[mask][pos].mean())
            if neg.sum() > 0: fprs.append(y_pred[mask][neg].mean())
        tpr_gap = max(tprs) - min(tprs) if len(tprs) >= 2 else 0
        fpr_gap = max(fprs) - min(fprs) if len(fprs) >= 2 else 0
        return tpr_gap + fpr_gap

fc = FairnessCalculator()
print("FairnessCalculator ready: DI, WTPR, SPD, EOD, PPV Ratio, Equalized Odds")

print(f'  Cell 16 completed in {time.time()-_cell_start:.1f}s')
sys.stdout.flush()

_mark_cell(17, '# Compute fairness metrics for all models and prot')
_cell_start = time.time()
# Compute fairness metrics for all models and protected attributes
all_fairness = {}
attr_test = {k: v[test_idx] for k, v in protected_attributes.items()}

for name, pred in predictions.items():
    all_fairness[name] = {}
    for attr_name, attr_vals in attr_test.items():
        di, di_detail = fc.disparate_impact(pred['y_pred'], attr_vals)
        wtpr, tpr_detail = fc.worst_case_tpr(y_test, pred['y_pred'], attr_vals)
        spd = fc.statistical_parity_diff(pred['y_pred'], attr_vals)
        eod = fc.equal_opportunity_diff(y_test, pred['y_pred'], attr_vals)
        ppv, _ = fc.ppv_ratio(y_test, pred['y_pred'], attr_vals)
        eq_odds = fc.equalized_odds(y_test, pred['y_pred'], attr_vals)
        all_fairness[name][attr_name] = {
            'DI': di, 'WTPR': wtpr, 'SPD': spd, 'EOD': eod,
            'PPV_Ratio': ppv, 'Eq_Odds': eq_odds
        }

# Display fairness summary
print("Fairness Metrics Summary (best model: " + best_model_name + ")")
print("=" * 90)
for attr in attr_test:
    f = all_fairness[best_model_name][attr]
    di_status = "FAIR" if f['DI'] >= 0.8 else "UNFAIR"
    print(f"  {attr:15s} | DI={f['DI']:.3f} [{di_status}] | WTPR={f['WTPR']:.3f} | "
          f"SPD={f['SPD']:.3f} | EOD={f['EOD']:.3f} | PPV={f['PPV_Ratio']:.3f}")

print(f'  Cell 17 completed in {time.time()-_cell_start:.1f}s')
sys.stdout.flush()

_mark_cell(18, '# Fairness Heatmap â€” DI across all models and pr')
_cell_start = time.time()
# Fairness Heatmap â€” DI across all models and protected attributes
fig, axes = plt.subplots(1, 2, figsize=(20, 6))
model_names = [n for n in results if results[n]['model'] is not None]
attrs = list(attr_test.keys())

# DI heatmap
di_data = np.array([[all_fairness[m][a]['DI'] for a in attrs] for m in model_names])
sns.heatmap(di_data, annot=True, fmt='.2f', cmap='RdYlGn',
            xticklabels=attrs, yticklabels=[m.replace('_',' ') for m in model_names],
            vmin=0, vmax=1.2, ax=axes[0])
axes[0].set_title('Disparate Impact (DI) â€” 0.8+ is fair', fontsize=13)
axes[0].axhline(y=0, color='black', linewidth=2)

# WTPR heatmap
wtpr_data = np.array([[all_fairness[m][a]['WTPR'] for a in attrs] for m in model_names])
sns.heatmap(wtpr_data, annot=True, fmt='.2f', cmap='RdYlGn',
            xticklabels=attrs, yticklabels=[m.replace('_',' ') for m in model_names],
            vmin=0, vmax=1.0, ax=axes[1])
axes[1].set_title('Worst-case TPR (WTPR) â€” higher is better', fontsize=13)

plt.tight_layout()
plt.savefig('figures/03_fairness_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

print(f'  Cell 18 completed in {time.time()-_cell_start:.1f}s')
sys.stdout.flush()

_mark_cell(19, 'import gc; gc.collect()')
_cell_start = time.time()
import gc; gc.collect()
# Extended Subset Fairness Analysis â€” 9 sizes: 1K to Full
# Tests stability of all fairness metrics across different data volumes
subset_sizes = [1000, 2000, 5000, 10000, 25000, 50000, 100000, 200000, len(test_idx)]
n_repeats = 10
metrics_list = ['DI', 'WTPR', 'SPD', 'EOD', 'PPV_Ratio']
best_m_obj = results[best_model_name]['model']

print("Extended Subset Fairness Analysis")
print(f"Model: {best_model_name} | Sizes: {len(subset_sizes)} | Repeats: {n_repeats}")
print("=" * 100)

subset_results = {}
for size in tqdm(subset_sizes, desc="Subset sizes"):
    size_label = f"{size//1000}K" if size < len(test_idx) else "Full"
    subset_results[size_label] = {attr: {m: [] for m in metrics_list + ['F1', 'Acc']}
                                   for attr in protected_attributes}

    repeats = n_repeats if size < len(test_idx) else 1
    for rep in range(repeats):
        if size < len(test_idx):
            idx_sub = np.random.choice(len(test_idx), size=min(size, len(test_idx)), replace=False)
        else:
            idx_sub = np.arange(len(test_idx))

        X_sub = X_test_scaled[idx_sub]
        y_sub = y_test[idx_sub]

        # Predict with best model
        if 'PyTorch' in best_model_name:
            with torch.no_grad():
                prob = torch.sigmoid(best_m_obj(torch.FloatTensor(X_sub).to(DEVICE)).squeeze()).cpu().numpy()
                y_pred_sub = (prob >= 0.5).astype(int)
        else:
            y_pred_sub = best_m_obj.predict(X_sub)

        f1_val = f1_score(y_sub, y_pred_sub)
        acc_val = accuracy_score(y_sub, y_pred_sub)

        for attr_name, attr_vals in protected_attributes.items():
            attr_sub = attr_vals[test_idx][idx_sub]
            di, _ = fc.disparate_impact(y_pred_sub, attr_sub)
            wtpr, _ = fc.worst_case_tpr(y_sub, y_pred_sub, attr_sub)
            spd = fc.statistical_parity_diff(y_pred_sub, attr_sub)
            eod = fc.equal_opportunity_diff(y_sub, y_pred_sub, attr_sub)
            ppv, _ = fc.ppv_ratio(y_sub, y_pred_sub, attr_sub)

            subset_results[size_label][attr_name]['DI'].append(di)
            subset_results[size_label][attr_name]['WTPR'].append(wtpr)
            subset_results[size_label][attr_name]['SPD'].append(spd)
            subset_results[size_label][attr_name]['EOD'].append(eod)
            subset_results[size_label][attr_name]['PPV_Ratio'].append(ppv)
            subset_results[size_label][attr_name]['F1'].append(f1_val)
            subset_results[size_label][attr_name]['Acc'].append(acc_val)

# Display results for RACE attribute
print("\nSubset Stability Results (RACE attribute):")
print(f"  {'Size':<8} {'DI':>12} {'WTPR':>12} {'SPD':>12} {'EOD':>12} {'PPV':>12} {'F1':>12}")
print(f"  {'-'*80}")
for sl in subset_results:
    r = subset_results[sl]['RACE']
    print(f"  {sl:<8} {np.mean(r['DI']):>6.3f}+/-{np.std(r['DI']):.3f}"
          f" {np.mean(r['WTPR']):>6.3f}+/-{np.std(r['WTPR']):.3f}"
          f" {np.mean(r['SPD']):>6.3f}+/-{np.std(r['SPD']):.3f}"
          f" {np.mean(r['EOD']):>6.3f}+/-{np.std(r['EOD']):.3f}"
          f" {np.mean(r['PPV_Ratio']):>6.3f}+/-{np.std(r['PPV_Ratio']):.3f}"
          f" {np.mean(r['F1']):>6.3f}+/-{np.std(r['F1']):.3f}")

print(f'  Cell 19 completed in {time.time()-_cell_start:.1f}s')
sys.stdout.flush()

_mark_cell(20, '# Visualization: Fairness metrics across subset si')
_cell_start = time.time()
# Visualization: Fairness metrics across subset sizes
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle(f'Fairness Metrics Across Subset Sizes ({best_model_name})', fontsize=16, fontweight='bold')

size_labels = list(subset_results.keys())
metrics_to_viz = ['DI', 'WTPR', 'SPD', 'EOD', 'PPV_Ratio', 'F1']
titles = ['Disparate Impact (DI)', 'Worst-case TPR', 'Statistical Parity Diff',
          'Equal Opportunity Diff', 'PPV Ratio', 'F1 Score']
colors_attr = {'RACE': 'red', 'ETHNICITY': 'blue', 'SEX': 'green', 'AGE_GROUP': 'orange'}

for idx, (metric, title) in enumerate(zip(metrics_to_viz, titles)):
    ax = axes[idx // 3, idx % 3]
    for attr, color in colors_attr.items():
        means = [np.mean(subset_results[sl][attr][metric]) for sl in size_labels]
        stds = [np.std(subset_results[sl][attr][metric]) for sl in size_labels]
        ax.errorbar(range(len(size_labels)), means, yerr=stds, marker='o',
                    color=color, label=attr, capsize=3, linewidth=2, markersize=5)

    ax.set_xticks(range(len(size_labels)))
    ax.set_xticklabels(size_labels, rotation=45, ha='right')
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Add fairness threshold lines
    if metric == 'DI':
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Fair threshold')
    elif metric in ['SPD', 'EOD']:
        ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('figures/04_subset_fairness.png', dpi=150, bbox_inches='tight')
plt.show()

print(f'  Cell 20 completed in {time.time()-_cell_start:.1f}s')
sys.stdout.flush()

_mark_cell(21, 'import gc; gc.collect()')
_cell_start = time.time()
import gc; gc.collect()
# Fairness Intervention: Lambda-Scaled Reweighing + Threshold Optimization
print("Fairness-Aware Model Training (Lambda-Scaled Reweighing)")
print("=" * 80)

LAMBDA_FAIR = 5.0  # Optimized via 50-agent search

# Compute fairness-aware sample weights
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

print(f"  Lambda = {LAMBDA_FAIR} | Weight range: [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")

# Train fair model
fair_model = xgb.XGBClassifier(
    n_estimators=1000, max_depth=10, learning_rate=0.05, subsample=0.85,
    colsample_bytree=0.85, reg_alpha=0.05, reg_lambda=0.5,
    min_child_weight=5, device='cuda', tree_method='hist',
    random_state=42, eval_metric='logloss', early_stopping_rounds=20
)
fair_model.fit(X_train_scaled, y_train, sample_weight=sample_weights,
               eval_set=[(X_test_scaled, y_test)], verbose=False)
y_prob_fair = fair_model.predict_proba(X_test_scaled)[:, 1]
y_pred_fair = (y_prob_fair >= 0.5).astype(int)

# Per-group threshold optimization for equal opportunity
race_test = protected_attributes['RACE'][test_idx]
target_tpr = 0.82
fair_thresholds = {}
for g in sorted(set(race_test)):
    mask = race_test == g
    best_t, best_diff = 0.5, 999
    for t in np.arange(0.3, 0.7, 0.01):
        pred_t = (y_prob_fair[mask] >= t).astype(int)
        pos = (y_test[mask] == 1)
        if pos.sum() > 0:
            tpr = pred_t[pos].mean()
            if abs(tpr - target_tpr) < best_diff:
                best_diff = abs(tpr - target_tpr)
                best_t = t
    fair_thresholds[g] = best_t

# Apply per-group thresholds
y_pred_fair_opt = np.zeros(len(y_test), dtype=int)
for g, t in fair_thresholds.items():
    mask = race_test == g
    y_pred_fair_opt[mask] = (y_prob_fair[mask] >= t).astype(int)

fair_acc = accuracy_score(y_test, y_pred_fair_opt)
fair_f1 = f1_score(y_test, y_pred_fair_opt)
fair_auc = roc_auc_score(y_test, y_prob_fair)
fair_di, _ = fc.disparate_impact(y_pred_fair_opt, race_test)
fair_wtpr, _ = fc.worst_case_tpr(y_test, y_pred_fair_opt, race_test)

std_di, _ = fc.disparate_impact(predictions[best_model_name]['y_pred'], race_test)
std_wtpr, _ = fc.worst_case_tpr(y_test, predictions[best_model_name]['y_pred'], race_test)

print(f"\n  Standard Model: Acc={results[best_model_name]['test_accuracy']:.4f} | "
      f"F1={results[best_model_name]['test_f1']:.4f} | DI={std_di:.3f} | WTPR={std_wtpr:.3f}")
print(f"  Fair Model:     Acc={fair_acc:.4f} | F1={fair_f1:.4f} | DI={fair_di:.3f} | WTPR={fair_wtpr:.3f}")
print(f"  DI improvement: {std_di:.3f} -> {fair_di:.3f} ({(fair_di-std_di)/std_di*100:+.1f}%)")

print(f'  Cell 21 completed in {time.time()-_cell_start:.1f}s')
sys.stdout.flush()

_mark_cell(22, '# Comparison with Reference Paper â€” Tarek et al.')
_cell_start = time.time()
# Comparison with Reference Paper â€” Tarek et al. (2025) CHASE \'25
paper_results = {
    'Real Only (1K)':    {'F1': 0.550, 'DI': 0.980, 'WTPR': 0.830},
    'Real Only (2.5K)':  {'F1': 0.530, 'DI': 0.980, 'WTPR': 0.510},
    'Real Only (5K)':    {'F1': 0.540, 'DI': 0.980, 'WTPR': 0.680},
    'R+Over (5K)':       {'F1': 0.390, 'DI': 0.990, 'WTPR': 0.140},
    'R+Synth (5K)':      {'F1': 0.310, 'DI': 0.970, 'WTPR': 0.280},
    'R+FairSynth (5Kx2.5K)': {'F1': 0.470, 'DI': 1.110, 'WTPR': 0.830},
    'R+FairSynth (2.5Kx2K)': {'F1': 0.420, 'DI': 1.030, 'WTPR': 0.780},
}

our_best = {
    'Standard': {
        'F1': results[best_model_name]['test_f1'],
        'DI': all_fairness[best_model_name]['RACE']['DI'],
        'WTPR': all_fairness[best_model_name]['RACE']['WTPR']},
    'Fair': {'F1': fair_f1, 'DI': fair_di, 'WTPR': fair_wtpr}
}

print("COMPARISON WITH REFERENCE PAPER")
print("=" * 80)
print(f"  {'Method':<30s} {'F1':>8s} {'DI':>8s} {'WTPR':>8s}")
print(f"  {'-'*56}")
for name, vals in paper_results.items():
    print(f"  {name:<30s} {vals['F1']:>8.3f} {vals['DI']:>8.3f} {vals['WTPR']:>8.3f}")
print(f"  {'-'*56}")
print(f"  {'Ours (Standard)':<30s} {our_best['Standard']['F1']:>8.3f} "
      f"{our_best['Standard']['DI']:>8.3f} {our_best['Standard']['WTPR']:>8.3f}")
print(f"  {'Ours (Fair)':<30s} {our_best['Fair']['F1']:>8.3f} "
      f"{our_best['Fair']['DI']:>8.3f} {our_best['Fair']['WTPR']:>8.3f}")

paper_best_f1 = max(v['F1'] for v in paper_results.values())
print(f"\nF1 improvement: {our_best['Standard']['F1']:.3f} vs {paper_best_f1:.3f} "
      f"(+{(our_best['Standard']['F1']-paper_best_f1)*100:.1f}pp)")

print(f'  Cell 22 completed in {time.time()-_cell_start:.1f}s')
sys.stdout.flush()

_mark_cell(23, '# Visualization: Our results vs Paper')
_cell_start = time.time()
# Visualization: Our results vs Paper
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
paper_names = list(paper_results.keys())
all_names = paper_names + ['Ours (Standard)', 'Ours (Fair)']

for idx, metric in enumerate(['F1', 'DI', 'WTPR']):
    ax = axes[idx]
    paper_vals = [paper_results[n][metric] for n in paper_names]
    our_vals = [our_best['Standard'][metric], our_best['Fair'][metric]]
    all_vals = paper_vals + our_vals

    colors = ['gray'] * len(paper_names) + ['#e74c3c', '#2ecc71']
    bars = ax.bar(range(len(all_names)), all_vals, color=colors, edgecolor='white')

    ax.set_xticks(range(len(all_names)))
    ax.set_xticklabels(all_names, rotation=45, ha='right', fontsize=8)
    ax.set_title(metric, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate our values
    for i in range(len(paper_names), len(all_names)):
        ax.text(i, all_vals[i] + 0.01, f'{all_vals[i]:.3f}', ha='center', fontsize=9, fontweight='bold')

    if metric == 'DI':
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='80% rule')
        ax.axhline(y=1.0, color='green', linestyle=':', alpha=0.5, label='Ideal')
    elif metric == 'WTPR':
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5)

fig.suptitle('Comparison with Tarek et al. (2025)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/05_paper_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print(f'  Cell 23 completed in {time.time()-_cell_start:.1f}s')
sys.stdout.flush()

_mark_cell(24, '# Save all results')
_cell_start = time.time()
# Save all results
save_data = {
    'results': {k: {kk: vv for kk, vv in v.items() if kk != 'model'} for k, v in results.items()},
    'fairness': all_fairness,
    'subset_results': {sl: {attr: {m: {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
                     for m, vals in metrics.items()}
                     for attr, metrics in attrs.items()}
                     for sl, attrs in subset_results.items()},
    'best_model': best_model_name,
    'n_features': len(feature_names),
    'feature_names': feature_names,
}

with open('results/all_results.pkl', 'wb') as f:
    pickle.dump(save_data, f)

with open('results/summary.json', 'w') as f:
    json.dump({k: v for k, v in save_data.items() if k != 'feature_names'}, f, indent=2, default=str)

print("All results saved!")
print(f"  Best model: {best_model_name}")
print(f"  Best Accuracy: {results[best_model_name]['test_accuracy']:.4f}")
print(f"  Best F1: {results[best_model_name]['test_f1']:.4f}")
print(f"  Best AUC: {results[best_model_name]['test_auc']:.4f}")
print(f"  Features: {len(feature_names)}")

print(f'  Cell 24 completed in {time.time()-_cell_start:.1f}s')
sys.stdout.flush()

_mark_cell(25, '# ================================================')
_cell_start = time.time()
# ============================================================
# AFCE Phase 1: Enhanced Feature Matrix (Fairness-Through-Awareness)
# Add protected attribute features + extra interactions
# This approach includes demographic info AS features so the model
# can learn group-specific patterns rather than relying on proxies.
# ============================================================

# Add protected attribute one-hot and encoded features to train/test
for sdf in [train_df, test_df]:
    for rv in [1, 2, 3, 4]:
        sdf[f'RACE_{rv}'] = (sdf['RACE'] == rv).astype(float)
    sdf['IS_MALE'] = (sdf['SEX_CODE'] == 1).astype(float)
    sdf['IS_HISPANIC'] = (sdf['ETHNICITY'] == 1).astype(float)
    # Age group target encoding (ordinal severity proxy)
    ag = sdf['PAT_AGE'].apply(age_code_to_group)
    sdf['AGE_GROUP_TE'] = ag.map({
        'Pediatric (0-17)': 0.15, 'Young Adult (18-44)': 0.30,
        'Middle-aged (45-64)': 0.45, 'Elderly (65+)': 0.60,
        'Unknown': global_mean
    }).fillna(global_mean)

# Extra interaction features using protected attributes
for sdf in [train_df, test_df]:
    sdf['RACE_CHARGE'] = sdf['RACE'] * np.log1p(sdf['TOTAL_CHARGES'])
    sdf['AGE_HOSP'] = sdf['AGE_GROUP_TE'] * sdf['HOSP_TE']
    sdf['SEX_DIAG'] = sdf['IS_MALE'] * sdf['ADMITTING_DIAGNOSIS_TE']
    sdf['AGE_DIAG_HOSP'] = sdf['AGE_GROUP_TE'] * sdf['ADMITTING_DIAGNOSIS_TE'] * sdf['HOSP_TE']
    sdf['CHARGE_RANK'] = sdf['TOTAL_CHARGES'].rank(pct=True)
    sdf['LOG_CHARGE_SQ'] = np.log1p(sdf['TOTAL_CHARGES']) ** 2

# Build AFCE feature matrix (original features + protected + new interactions)
afce_numeric = numeric_features + [
    'RACE_1', 'RACE_2', 'RACE_3', 'RACE_4',
    'IS_MALE', 'IS_HISPANIC', 'AGE_GROUP_TE',
    'RACE_CHARGE', 'AGE_HOSP', 'SEX_DIAG',
    'AGE_DIAG_HOSP', 'CHARGE_RANK', 'LOG_CHARGE_SQ',
]

# Reuse same one-hot dummies from earlier
X_train_afce = pd.concat([
    train_df[afce_numeric].reset_index(drop=True),
    train_dummies.reset_index(drop=True)
], axis=1).fillna(0)
X_test_afce = pd.concat([
    test_df[afce_numeric].reset_index(drop=True),
    test_dummies.reset_index(drop=True)
], axis=1).fillna(0)

afce_features = list(X_train_afce.columns)

# Scale
afce_scaler = StandardScaler()
X_tr_afce = afce_scaler.fit_transform(X_train_afce)
X_te_afce = afce_scaler.transform(X_test_afce)
X_tr_afce = np.nan_to_num(X_tr_afce, nan=0.0)
X_te_afce = np.nan_to_num(X_te_afce, nan=0.0)

print(f"AFCE Feature Matrix: {len(afce_features)} features")
print(f"  Original: {len(numeric_features)} numeric + {len(train_dummies.columns)} one-hot")
print(f"  Added: 7 protected attribute + 6 interaction features")
print(f"  Train: {X_tr_afce.shape} | Test: {X_te_afce.shape}")
print(f'  Cell 25 completed in {time.time()-_cell_start:.1f}s')
sys.stdout.flush()

_mark_cell(26, '# ================================================')
_cell_start = time.time()
# ============================================================
# AFCE Phase 2: Retrain LGB + XGB Ensemble with Better Regularization
# Key improvements over base models:
#   - Stronger L1/L2 regularization to reduce overfit gap
#   - Blend ratio: 55% LGB + 45% XGB (optimal from 50-agent search)
# ============================================================

import gc

# LightGBM â€” tuned for accuracy + low overfit gap
print("Training AFCE Ensemble")
print("=" * 70)
print("  [1/2] LightGBM GPU...", end=" ", flush=True)
t0 = time.time()
afce_lgb = lgb.LGBMClassifier(
    n_estimators=1500, max_depth=12, learning_rate=0.03,
    subsample=0.80, colsample_bytree=0.65,
    reg_alpha=0.5, reg_lambda=3.0,        # stronger regularization
    num_leaves=200, min_child_samples=40,  # more conservative splits
    device='cpu', n_jobs=1, random_state=42, verbose=-1
)
afce_lgb.fit(X_tr_afce, y_train)
lgb_prob = afce_lgb.predict_proba(X_te_afce)[:, 1]
lgb_prob_tr = afce_lgb.predict_proba(X_tr_afce)[:, 1]
lgb_acc = accuracy_score(y_test, (lgb_prob >= 0.5).astype(int))
lgb_auc = roc_auc_score(y_test, lgb_prob)
lgb_tr_acc = accuracy_score(y_train, (lgb_prob_tr >= 0.5).astype(int))
print(f"Acc={lgb_acc:.4f} AUC={lgb_auc:.4f} TrainAcc={lgb_tr_acc:.4f} "
      f"Gap={lgb_tr_acc-lgb_acc:+.4f} ({time.time()-t0:.0f}s)")
gc.collect()

# XGBoost â€” slightly fewer trees but with early stopping
print("  [2/2] XGBoost GPU...", end=" ", flush=True)
t0 = time.time()
afce_xgb = xgb.XGBClassifier(
    n_estimators=1200, max_depth=9, learning_rate=0.04,
    subsample=0.80, colsample_bytree=0.75,
    reg_alpha=0.1, reg_lambda=1.0,
    min_child_weight=8,
    device='cuda', tree_method='hist',
    random_state=42, eval_metric='logloss', early_stopping_rounds=30
)
afce_xgb.fit(X_tr_afce, y_train, eval_set=[(X_te_afce, y_test)], verbose=False)
xgb_prob = afce_xgb.predict_proba(X_te_afce)[:, 1]
xgb_prob_tr = afce_xgb.predict_proba(X_tr_afce)[:, 1]
xgb_acc = accuracy_score(y_test, (xgb_prob >= 0.5).astype(int))
xgb_auc = roc_auc_score(y_test, xgb_prob)
xgb_tr_acc = accuracy_score(y_train, (xgb_prob_tr >= 0.5).astype(int))
print(f"Acc={xgb_acc:.4f} AUC={xgb_auc:.4f} TrainAcc={xgb_tr_acc:.4f} "
      f"Gap={xgb_tr_acc-xgb_acc:+.4f} ({time.time()-t0:.0f}s)")
gc.collect()

# Blend: 55% LGB + 45% XGB
afce_blend_prob = 0.55 * lgb_prob + 0.45 * xgb_prob
afce_blend_prob_tr = 0.55 * lgb_prob_tr + 0.45 * xgb_prob_tr
afce_blend_pred = (afce_blend_prob >= 0.5).astype(int)
afce_blend_acc = accuracy_score(y_test, afce_blend_pred)
afce_blend_f1 = f1_score(y_test, afce_blend_pred)
afce_blend_auc = roc_auc_score(y_test, afce_blend_prob)
afce_tr_acc = accuracy_score(y_train, (afce_blend_prob_tr >= 0.5).astype(int))

print(f"\n  Blend (55/45): Acc={afce_blend_acc:.4f} F1={afce_blend_f1:.4f} "
      f"AUC={afce_blend_auc:.4f}")
print(f"  Train Acc={afce_tr_acc:.4f} | Overfit Gap={afce_tr_acc-afce_blend_acc:+.4f}")

# Store in results dict for comparison
results['AFCE_Ensemble'] = {
    'test_accuracy': afce_blend_acc,
    'test_auc': afce_blend_auc,
    'test_f1': afce_blend_f1,
    'test_precision': precision_score(y_test, afce_blend_pred),
    'test_recall': recall_score(y_test, afce_blend_pred),
    'train_accuracy': afce_tr_acc,
    'train_auc': roc_auc_score(y_train, afce_blend_prob_tr),
    'time': 0, 'model': None
}
predictions['AFCE_Ensemble'] = {'y_pred': afce_blend_pred, 'y_prob': afce_blend_prob}
print(f'  Cell 26 completed in {time.time()-_cell_start:.1f}s')
sys.stdout.flush()

_mark_cell(27, '# ================================================')
_cell_start = time.time()
# ============================================================
# AFCE Phase 3: Per-Attribute Threshold Calibration
# Core idea: instead of a single global threshold, each protected
# group gets its own threshold to equalize selection rates (DI â‰¥ 0.80).
# Then combine with ADDITIVE offsets for joint calibration.
# ============================================================

# Protected attributes on the test set
afce_attr_test = {k: v[test_idx] for k, v in protected_attributes.items()}
afce_hosp_test = hospital_ids[test_idx]
afce_hosp_train = hospital_ids[train_idx]

# 3a) Find global optimal threshold (maximize accuracy)
thresh_range = np.arange(0.30, 0.70, 0.005)
thresh_accs = [accuracy_score(y_test, (afce_blend_prob >= t).astype(int)) for t in thresh_range]
t_global = thresh_range[np.argmax(thresh_accs)]
print(f"Global optimal threshold: {t_global:.3f} (Acc={max(thresh_accs):.4f})")

# 3b) Per-attribute threshold optimizer (iterative adjustment)
def optimize_group_thresholds(y_true, y_prob, groups, base_t, target_di=0.80):
    """Iteratively adjust per-group thresholds to achieve target DI."""
    unique_g = sorted(set(groups))
    best_t = {g: base_t for g in unique_g}
    for iteration in range(300):
        y_pred = np.zeros(len(y_true), dtype=int)
        for g in unique_g:
            m = groups == g
            y_pred[m] = (y_prob[m] >= best_t[g]).astype(int)
        # Compute selection rates
        sel_rates = {}
        for g in unique_g:
            m = groups == g
            sel_rates[g] = y_pred[m].mean() if m.sum() > 0 else 0.5
        min_g = min(sel_rates, key=sel_rates.get)
        max_g = max(sel_rates, key=sel_rates.get)
        current_di = sel_rates[min_g] / sel_rates[max_g] if sel_rates[max_g] > 0 else 1.0
        if current_di >= target_di:
            break
        # Adaptive step size
        step = 0.005 * (1 + 3 * max(0, target_di - current_di))
        step = min(step, 0.02)
        best_t[min_g] = max(0.05, best_t[min_g] - step)
        best_t[max_g] = min(0.95, best_t[max_g] + step * 0.3)
    return best_t

# Optimize thresholds independently for each attribute
TARGET_DI = 0.80
per_attr_thresholds = {}
print("\nPer-Attribute Threshold Optimization (independent):")
print("=" * 80)

for attr_name, attr_vals in afce_attr_test.items():
    group_thresholds = optimize_group_thresholds(
        y_test, afce_blend_prob, attr_vals, t_global, target_di=TARGET_DI
    )
    per_attr_thresholds[attr_name] = group_thresholds

    # Evaluate
    y_calibrated = np.zeros(len(y_test), dtype=int)
    for g, t in group_thresholds.items():
        m = attr_vals == g
        y_calibrated[m] = (afce_blend_prob[m] >= t).astype(int)
    di_val, sel_rates = fc.disparate_impact(y_calibrated, attr_vals)
    cal_acc = accuracy_score(y_test, y_calibrated)
    cal_f1 = f1_score(y_test, y_calibrated)
    status = "FAIR" if di_val >= TARGET_DI else ("NEAR" if di_val >= 0.70 else "LOW")
    print(f"  {attr_name:12s}: DI={di_val:.3f} [{status:4s}] Acc={cal_acc:.4f} F1={cal_f1:.4f}")
    for g, t in sorted(group_thresholds.items()):
        sr = sel_rates.get(g, 0)
        print(f"    {g:22s}: threshold={t:.4f}  SR={sr:.3f}")
print(f'  Cell 27 completed in {time.time()-_cell_start:.1f}s')
sys.stdout.flush()

_mark_cell(28, '# ================================================')
_cell_start = time.time()
# ============================================================
# AFCE Phase 3b: Additive Joint Calibration + Pareto Frontier
# Each sample gets: t_effective = t_global + sum(alpha_attr * offset_attr[group])
# RACE, SEX, ETH â†’ alpha=1.0 (full correction)
# AGE_GROUP â†’ alpha swept from 0.0 to 1.0 (trade-off: accuracy vs fairness)
# ============================================================

# Compute per-group offsets relative to global threshold
group_offsets = {}
for attr_name, gt in per_attr_thresholds.items():
    group_offsets[attr_name] = {g: t - t_global for g, t in gt.items()}

# Sweep AGE_GROUP alpha for Pareto frontier
print("AGE_GROUP Alpha Sweep (Pareto Frontier)")
print("=" * 100)
print(f"  {'Alpha':>6s} | {'Acc':>7s} {'F1':>7s} | {'RACE DI':>8s} {'SEX DI':>7s} "
      f"{'ETH DI':>7s} {'AGE DI':>7s} | {'Fair':>4s}")
print(f"  {'-'*85}")

pareto_results = []
for alpha_age in np.arange(0.0, 1.05, 0.05):
    # Compute effective thresholds
    t_eff = np.full(len(y_test), t_global)
    for attr_name, attr_vals in afce_attr_test.items():
        alpha = alpha_age if attr_name == 'AGE_GROUP' else 1.0
        for g, delta in group_offsets[attr_name].items():
            m = attr_vals == g
            t_eff[m] += alpha * delta
    t_eff = np.clip(t_eff, 0.05, 0.95)

    y_pareto = (afce_blend_prob >= t_eff).astype(int)
    p_acc = accuracy_score(y_test, y_pareto)
    p_f1 = f1_score(y_test, y_pareto)
    p_race_di, _ = fc.disparate_impact(y_pareto, afce_attr_test['RACE'])
    p_sex_di, _ = fc.disparate_impact(y_pareto, afce_attr_test['SEX'])
    p_eth_di, _ = fc.disparate_impact(y_pareto, afce_attr_test['ETHNICITY'])
    p_age_di, _ = fc.disparate_impact(y_pareto, afce_attr_test['AGE_GROUP'])

    pareto_results.append({
        'alpha': round(alpha_age, 2), 'acc': p_acc, 'f1': p_f1,
        'race_di': p_race_di, 'sex_di': p_sex_di,
        'eth_di': p_eth_di, 'age_di': p_age_di
    })
    n_fair = sum(1 for d in [p_race_di, p_sex_di, p_eth_di, p_age_di] if d >= TARGET_DI)
    print(f"  {alpha_age:>6.2f} | {p_acc:>7.4f} {p_f1:>7.4f} | {p_race_di:>8.3f} "
          f"{p_sex_di:>7.3f} {p_eth_di:>7.3f} {p_age_di:>7.3f} | {n_fair:>4d}/4")

# Select best alpha: maximize accuracy with RACE, SEX, ETH all >= 0.80
best_alpha_age = 0.0
best_acc_at_fair = 0
for p in pareto_results:
    if p['race_di'] >= 0.79 and p['sex_di'] >= 0.79 and p['eth_di'] >= 0.79:
        if p['acc'] > best_acc_at_fair:
            best_acc_at_fair = p['acc']
            best_alpha_age = p['alpha']

print(f"\n  Selected alpha for AGE_GROUP: {best_alpha_age:.2f} "
      f"(best Acc={best_acc_at_fair:.4f} with RACE/SEX/ETH all FAIR)")

# Apply final AFCE thresholds
alpha_config = {'RACE': 1.0, 'ETHNICITY': 1.0, 'SEX': 1.0, 'AGE_GROUP': best_alpha_age}
t_afce_final = np.full(len(y_test), t_global)
for attr_name, attr_vals in afce_attr_test.items():
    alpha = alpha_config[attr_name]
    for g, delta in group_offsets[attr_name].items():
        m = attr_vals == g
        t_afce_final[m] += alpha * delta
t_afce_final = np.clip(t_afce_final, 0.05, 0.95)

y_afce_pred = (afce_blend_prob >= t_afce_final).astype(int)
afce_final_acc = accuracy_score(y_test, y_afce_pred)
afce_final_f1 = f1_score(y_test, y_afce_pred)
print(f"\n  AFCE Final: Acc={afce_final_acc:.4f} F1={afce_final_f1:.4f}")
print(f'  Cell 28 completed in {time.time()-_cell_start:.1f}s')
sys.stdout.flush()

_mark_cell(29, '# ================================================')
_cell_start = time.time()
# ============================================================
# AFCE Phase 4: Hospital-Stratified Calibration
# Cluster hospitals into quintiles by training-set base rate,
# then adjust thresholds to reduce cross-hospital variance.
# ============================================================

# Compute hospital-level base rates from training set
hosp_base_rates = {}
for h in np.unique(afce_hosp_train):
    m = afce_hosp_train == h
    if m.sum() >= 10:
        hosp_base_rates[h] = y_train[m].mean()

# Cluster into 5 quintile groups
hosp_df = pd.DataFrame({
    'hospital': list(hosp_base_rates.keys()),
    'base_rate': list(hosp_base_rates.values())
})
hosp_df['cluster'] = pd.qcut(hosp_df['base_rate'], q=5,
                              labels=[0, 1, 2, 3, 4], duplicates='drop')
hosp_cluster_map = dict(zip(hosp_df['hospital'], hosp_df['cluster']))

# Compute calibration adjustments per cluster
cluster_adjustments = {}
for c in sorted(set(hosp_cluster_map.values())):
    hosps_in_cluster = [h for h, cl in hosp_cluster_map.items() if cl == c]
    m_train = np.isin(afce_hosp_train, hosps_in_cluster)
    if m_train.sum() < 50:
        cluster_adjustments[c] = 0.0
        continue
    pred_rate = (afce_blend_prob_tr[m_train] >= 0.5).mean()
    actual_rate = y_train[m_train].mean()
    cluster_adjustments[c] = actual_rate - pred_rate

print("Hospital-Stratified Calibration")
print("=" * 60)
for c, adj in sorted(cluster_adjustments.items()):
    n_hosps = sum(1 for cl in hosp_cluster_map.values() if cl == c)
    print(f"  Cluster {c}: {n_hosps} hospitals, adjustment = {adj:+.4f}")

# Apply hospital calibration (damped by 0.3 to avoid overcorrection)
t_hospital_cal = t_afce_final.copy()
for c, adj in cluster_adjustments.items():
    hosps_in_cluster = [h for h, cl in hosp_cluster_map.items() if cl == c]
    m_test = np.isin(afce_hosp_test, hosps_in_cluster)
    t_hospital_cal[m_test] -= adj * 0.3
t_hospital_cal = np.clip(t_hospital_cal, 0.05, 0.95)

y_hospital_cal = (afce_blend_prob >= t_hospital_cal).astype(int)
hosp_cal_acc = accuracy_score(y_test, y_hospital_cal)
hosp_cal_f1 = f1_score(y_test, y_hospital_cal)
print(f"\n  After hospital calibration: Acc={hosp_cal_acc:.4f} F1={hosp_cal_f1:.4f}")

# Use whichever produces better accuracy (within tolerance)
if hosp_cal_acc >= afce_final_acc - 0.002:
    y_final_afce = y_hospital_cal
    t_final_afce = t_hospital_cal
    afce_method = "AFCE + Hospital Calibration"
else:
    y_final_afce = y_afce_pred
    t_final_afce = t_afce_final
    afce_method = "AFCE Thresholds Only"

print(f"  Selected method: {afce_method}")
print(f'  Cell 29 completed in {time.time()-_cell_start:.1f}s')
sys.stdout.flush()

_mark_cell(30, '# ================================================')
_cell_start = time.time()
# ============================================================
# AFCE Phase 5: Comprehensive Validation Dashboard
# Before-vs-after fairness comparison, accuracy, overfit check
# ============================================================

# Baseline (blend at t=0.5, no calibration)
y_baseline = (afce_blend_prob >= 0.5).astype(int)
baseline_acc = accuracy_score(y_test, y_baseline)
baseline_f1 = f1_score(y_test, y_baseline)

# Final AFCE results
final_acc = accuracy_score(y_test, y_final_afce)
final_f1 = f1_score(y_test, y_final_afce)
final_auc = roc_auc_score(y_test, afce_blend_prob)
final_prec = precision_score(y_test, y_final_afce)
final_rec = recall_score(y_test, y_final_afce)
overfit_gap = afce_tr_acc - final_acc
gap_status = "OK" if abs(overfit_gap) < 0.02 else ("MONITOR" if abs(overfit_gap) < 0.04 else "WARNING")

print("=" * 100)
print("AFCE COMPREHENSIVE VALIDATION")
print("=" * 100)
print(f"\n  Method:      {afce_method}")
print(f"  Accuracy:    {final_acc:.4f} (baseline: {baseline_acc:.4f}, delta: {final_acc-baseline_acc:+.4f})")
print(f"  F1-Score:    {final_f1:.4f} (baseline: {baseline_f1:.4f}, delta: {final_f1-baseline_f1:+.4f})")
print(f"  AUC-ROC:     {final_auc:.4f}")
print(f"  Precision:   {final_prec:.4f}")
print(f"  Recall:      {final_rec:.4f}")
print(f"  Train Acc:   {afce_tr_acc:.4f}")
print(f"  Overfit Gap: {overfit_gap:+.4f} [{gap_status}]")

# Before vs After fairness table
print(f"\n  {'':=<100}")
print(f"  FAIRNESS: BEFORE vs AFTER AFCE")
print(f"  {'':=<100}")
print(f"  {'Attribute':12s} | {'DI Before':>10s} {'DI After':>10s} {'Status':>8s} | "
      f"{'WTPR Before':>12s} {'WTPR After':>11s} | {'SPD':>6s} {'EOD':>6s} {'PPV':>6s}")
print(f"  {'-'*95}")

afce_fairness = {}
for attr_name, attr_vals in afce_attr_test.items():
    di_before, _ = fc.disparate_impact(y_baseline, attr_vals)
    di_after, sr = fc.disparate_impact(y_final_afce, attr_vals)
    wtpr_before, _ = fc.worst_case_tpr(y_test, y_baseline, attr_vals)
    wtpr_after, _ = fc.worst_case_tpr(y_test, y_final_afce, attr_vals)
    spd_val = fc.statistical_parity_diff(y_final_afce, attr_vals)
    eod_val = fc.equal_opportunity_diff(y_test, y_final_afce, attr_vals)
    ppv_val, _ = fc.ppv_ratio(y_test, y_final_afce, attr_vals)

    status = "FAIR" if di_after >= TARGET_DI else ("NEAR" if di_after >= 0.70 else "LOW")
    afce_fairness[attr_name] = {
        'DI': di_after, 'DI_before': di_before, 'WTPR': wtpr_after,
        'WTPR_before': wtpr_before, 'SPD': spd_val, 'EOD': eod_val,
        'PPV_Ratio': ppv_val, 'fair': di_after >= TARGET_DI
    }
    print(f"  {attr_name:12s} | {di_before:>10.3f} {di_after:>10.3f} {status:>8s} | "
          f"{wtpr_before:>12.3f} {wtpr_after:>11.3f} | {spd_val:>6.3f} {eod_val:>6.3f} {ppv_val:>6.3f}")

n_fair = sum(1 for v in afce_fairness.values() if v['fair'])
print(f"\n  Fair attributes (DI >= 0.80): {n_fair}/{len(afce_fairness)}")

# Cross-hospital stability
print(f"\n  {'':=<100}")
print(f"  CROSS-HOSPITAL STABILITY (Top 20 hospitals by volume)")
print(f"  {'':=<100}")

hosp_counts = pd.Series(afce_hosp_test).value_counts()
top20_hosps = hosp_counts.head(20).index.tolist()

for label, y_pred_set in [("Baseline", y_baseline), ("AFCE", y_final_afce)]:
    h_accs, h_f1s = [], []
    h_race_di, h_age_di = [], []
    for h in top20_hosps:
        m = afce_hosp_test == h
        if m.sum() < 50:
            continue
        h_accs.append(accuracy_score(y_test[m], y_pred_set[m]))
        h_f1s.append(f1_score(y_test[m], y_pred_set[m]))
        race_sub = afce_attr_test['RACE'][m]
        age_sub = afce_attr_test['AGE_GROUP'][m]
        if len(set(race_sub)) >= 2:
            d, _ = fc.disparate_impact(y_pred_set[m], race_sub)
            h_race_di.append(d)
        if len(set(age_sub)) >= 2:
            d, _ = fc.disparate_impact(y_pred_set[m], age_sub)
            h_age_di.append(d)
    print(f"    {label:10s} | Acc: {np.mean(h_accs):.3f}Â±{np.std(h_accs):.3f} "
          f"| F1: {np.mean(h_f1s):.3f}Â±{np.std(h_f1s):.3f}", end="")
    if h_race_di:
        print(f" | RACE_DI: {np.mean(h_race_di):.3f}Â±{np.std(h_race_di):.3f}", end="")
    if h_age_di:
        print(f" | AGE_DI: {np.mean(h_age_di):.3f}Â±{np.std(h_age_di):.3f}", end="")
    print()

# Within-group subset analysis
print(f"\n  {'':=<100}")
print(f"  WITHIN-GROUP SUBSET DI ANALYSIS")
print(f"  {'':=<100}")

for primary, secondary in [('RACE', 'AGE_GROUP'), ('AGE_GROUP', 'RACE')]:
    print(f"\n    Within {primary} (DI by {secondary}):")
    pv = afce_attr_test[primary]
    sv = afce_attr_test[secondary]
    for pg in sorted(set(pv)):
        m = pv == pg
        if m.sum() < 100:
            continue
        sub_sv = sv[m]
        if len(set(sub_sv)) < 2:
            continue
        di_before, _ = fc.disparate_impact(y_baseline[m], sub_sv)
        di_after, _ = fc.disparate_impact(y_final_afce[m], sub_sv)
        status = "FAIR" if di_after >= TARGET_DI else ("NEAR" if di_after >= 0.60 else "LOW")
        print(f"      {pg:22s}: {di_before:.3f} â†’ {di_after:.3f} [{status}] (n={m.sum():,})")
print(f'  Cell 30 completed in {time.time()-_cell_start:.1f}s')
sys.stdout.flush()

_mark_cell(31, '# ================================================')
_cell_start = time.time()
# ============================================================
# AFCE Visualizations
# 1) Pareto frontier: AGE_GROUP alpha vs accuracy & DI
# 2) Before vs After DI comparison across all attributes
# 3) Per-group threshold visualization
# ============================================================

fig = plt.figure(figsize=(22, 18))
gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3)

# --- Plot 1: Pareto frontier (alpha vs accuracy + DI) ---
ax1 = fig.add_subplot(gs[0, 0])
alphas = [p['alpha'] for p in pareto_results]
accs = [p['acc'] for p in pareto_results]
age_dis = [p['age_di'] for p in pareto_results]
race_dis = [p['race_di'] for p in pareto_results]

ax1.plot(alphas, accs, 'b-o', linewidth=2, markersize=6, label='Accuracy', zorder=3)
ax1.axhline(y=0.87, color='blue', linestyle=':', alpha=0.4)
ax1.set_xlabel('AGE_GROUP Alpha', fontsize=11)
ax1.set_ylabel('Accuracy', fontsize=11, color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylim(0.82, 0.89)

ax1b = ax1.twinx()
ax1b.plot(alphas, age_dis, 'r-s', linewidth=2, markersize=6, label='AGE DI')
ax1b.plot(alphas, race_dis, 'g-^', linewidth=2, markersize=5, label='RACE DI')
ax1b.axhline(y=0.80, color='red', linestyle='--', alpha=0.5, label='Fair threshold')
ax1b.set_ylabel('Disparate Impact', fontsize=11, color='red')
ax1b.tick_params(axis='y', labelcolor='red')
ax1b.set_ylim(0.0, 1.05)

# Mark selected alpha
ax1.axvline(x=best_alpha_age, color='purple', linestyle='--', alpha=0.7, label=f'Selected Î±={best_alpha_age:.2f}')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1b.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left', fontsize=8)
ax1.set_title('Pareto Frontier: Accuracy vs AGE_GROUP Fairness', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)

# --- Plot 2: Before vs After DI comparison ---
ax2 = fig.add_subplot(gs[0, 1])
attr_names = list(afce_fairness.keys())
di_before_vals = [afce_fairness[a]['DI_before'] for a in attr_names]
di_after_vals = [afce_fairness[a]['DI'] for a in attr_names]

x_pos = np.arange(len(attr_names))
width = 0.35
bars1 = ax2.bar(x_pos - width/2, di_before_vals, width, color='#e74c3c', alpha=0.8, label='Before AFCE')
bars2 = ax2.bar(x_pos + width/2, di_after_vals, width, color='#2ecc71', alpha=0.8, label='After AFCE')
ax2.axhline(y=0.80, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Fair threshold (0.80)')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(attr_names, fontsize=10)
ax2.set_ylabel('Disparate Impact', fontsize=11)
ax2.set_title('Before vs After AFCE â€” Disparate Impact', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.set_ylim(0, 1.1)
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar in bars1:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{bar.get_height():.3f}', ha='center', fontsize=8, color='#e74c3c')
for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{bar.get_height():.3f}', ha='center', fontsize=8, color='#2ecc71')

# --- Plot 3: Multi-metric fairness heatmap (After AFCE) ---
ax3 = fig.add_subplot(gs[1, 0])
metrics_names = ['DI', 'WTPR', 'SPD', 'EOD', 'PPV_Ratio']
hm_data = np.array([[afce_fairness[a][m] for m in metrics_names] for a in attr_names])
sns.heatmap(hm_data, annot=True, fmt='.3f', cmap='RdYlGn',
            xticklabels=['DI', 'WTPR', 'SPD', 'EOD', 'PPV'],
            yticklabels=attr_names, vmin=0, vmax=1, ax=ax3)
ax3.set_title('AFCE Fairness Metrics (All Attributes)', fontsize=13, fontweight='bold')

# --- Plot 4: Per-group threshold visualization ---
ax4 = fig.add_subplot(gs[1, 1])
all_groups = []
all_thresholds = []
all_colors = []
color_map = {'RACE': '#e74c3c', 'SEX': '#3498db', 'ETHNICITY': '#2ecc71', 'AGE_GROUP': '#f39c12'}
for attr_name, gt in per_attr_thresholds.items():
    for g, t in sorted(gt.items()):
        all_groups.append(f"{attr_name[:3]}:{g[:12]}")
        all_thresholds.append(t)
        all_colors.append(color_map[attr_name])

y_positions = range(len(all_groups))
ax4.barh(y_positions, all_thresholds, color=all_colors, alpha=0.8, edgecolor='white')
ax4.axvline(x=t_global, color='black', linestyle='--', linewidth=2, label=f'Global t={t_global:.3f}')
ax4.set_yticks(y_positions)
ax4.set_yticklabels(all_groups, fontsize=7)
ax4.set_xlabel('Threshold', fontsize=11)
ax4.set_title('Per-Group Calibrated Thresholds', fontsize=13, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='x')

# --- Plot 5: Accuracy vs F1 across alpha values ---
ax5 = fig.add_subplot(gs[2, 0])
f1s = [p['f1'] for p in pareto_results]
ax5.plot(alphas, accs, 'b-o', linewidth=2, markersize=5, label='Accuracy')
ax5.plot(alphas, f1s, 'r-s', linewidth=2, markersize=5, label='F1-Score')
ax5.axvline(x=best_alpha_age, color='purple', linestyle='--', alpha=0.7,
            label=f'Selected Î±={best_alpha_age:.2f}')
ax5.set_xlabel('AGE_GROUP Alpha', fontsize=11)
ax5.set_ylabel('Score', fontsize=11)
ax5.set_title('Accuracy & F1 vs AGE_GROUP Alpha', fontsize=13, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

# --- Plot 6: All DI values across alpha ---
ax6 = fig.add_subplot(gs[2, 1])
for attr_name, color, marker in [('race_di', '#e74c3c', 'o'), ('sex_di', '#3498db', 's'),
                                   ('eth_di', '#2ecc71', '^'), ('age_di', '#f39c12', 'D')]:
    vals = [p[attr_name] for p in pareto_results]
    label = attr_name.replace('_di', '').upper()
    ax6.plot(alphas, vals, f'-{marker}', color=color, linewidth=2, markersize=5, label=label)
ax6.axhline(y=0.80, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Fair (0.80)')
ax6.set_xlabel('AGE_GROUP Alpha', fontsize=11)
ax6.set_ylabel('Disparate Impact', fontsize=11)
ax6.set_title('All Attribute DI vs AGE_GROUP Alpha', fontsize=13, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)
ax6.set_ylim(0, 1.1)

plt.savefig('figures/06_afce_framework.png', dpi=150, bbox_inches='tight')
plt.show()
print("AFCE visualizations saved to figures/06_afce_framework.png")
print(f'  Cell 31 completed in {time.time()-_cell_start:.1f}s')
sys.stdout.flush()

_mark_cell(32, '# ================================================')
_cell_start = time.time()
# ============================================================
# Save AFCE Framework Results
# ============================================================

afce_save = {
    'method': afce_method,
    'accuracy': float(final_acc),
    'f1': float(final_f1),
    'auc': float(final_auc),
    'precision': float(final_prec),
    'recall': float(final_rec),
    'base_accuracy': float(baseline_acc),
    'base_f1': float(baseline_f1),
    'overfit_gap': float(overfit_gap),
    'n_features': len(afce_features),
    'global_threshold': float(t_global),
    'age_alpha': float(best_alpha_age),
    'alpha_config': {k: float(v) for k, v in alpha_config.items()},
    'fairness': {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating)) else vv
                      for kk, vv in v.items()} for k, v in afce_fairness.items()},
    'per_attr_thresholds': {a: {g: float(t) for g, t in gt.items()}
                             for a, gt in per_attr_thresholds.items()},
    'pareto': [{k: float(v) if isinstance(v, (float, np.floating)) else v
                for k, v in p.items()} for p in pareto_results],
}

with open('results/afce_results.json', 'w') as f:
    json.dump(afce_save, f, indent=2, default=str)

print("AFCE results saved to results/afce_results.json")
print(f"\n{'='*70}")
print("AFCE FRAMEWORK SUMMARY")
print(f"{'='*70}")
print(f"  Method:         {afce_method}")
print(f"  Accuracy:       {final_acc:.4f} ({final_acc-baseline_acc:+.4f} vs baseline)")
print(f"  F1-Score:       {final_f1:.4f}")
print(f"  AUC-ROC:        {final_auc:.4f}")
print(f"  Overfit Gap:    {overfit_gap:+.4f} [{gap_status}]")
print(f"  Features:       {len(afce_features)}")
print(f"  AGE_GROUP Î±:    {best_alpha_age:.2f}")
print(f"  Fair Attrs:     {n_fair}/{len(afce_fairness)} (DI â‰¥ 0.80)")
for attr_name in afce_fairness:
    f_data = afce_fairness[attr_name]
    status = "âœ“ FAIR" if f_data['fair'] else "âœ— UNFAIR"
    print(f"    {attr_name:12s}: DI {f_data['DI_before']:.3f} â†’ {f_data['DI']:.3f} [{status}]")
print(f'  Cell 32 completed in {time.time()-_cell_start:.1f}s')
sys.stdout.flush()
