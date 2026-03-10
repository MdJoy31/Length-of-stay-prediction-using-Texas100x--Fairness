"""Quick test of new TI formula on actual data."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

DATA = r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\data\texas_100x.csv"
print("Loading data...")
df = pd.read_csv(DATA)
print(f"Rows: {len(df):,}")

# Quick AGE_GROUP mapping (same as notebook)
def create_age_groups(age_code):
    if age_code <= 4:    return 'Pediatric'
    elif age_code <= 9:  return 'Young_Adult'
    elif age_code <= 14: return 'Middle_Aged'
    else:                return 'Elderly'

RACE_LABELS = {0:'Other/Unknown', 1:'Native American', 2:'Asian/PI', 3:'Black', 4:'White'}
SEX_LABELS = {0:'Female', 1:'Male'}
ETH_LABELS = {0:'Non-Hispanic', 1:'Hispanic'}

df['RACE_LABEL'] = df['RACE'].map(RACE_LABELS)
df['SEX_LABEL'] = df['SEX_CODE'].map(SEX_LABELS)
df['ETH_LABEL'] = df['ETHNICITY'].map(ETH_LABELS)
df['AGE_GROUP'] = df['PAT_AGE'].apply(create_age_groups)
df['LOS_BINARY'] = (df['LENGTH_OF_STAY'] > 3).astype(int)

feature_cols = [c for c in df.columns if c not in ['LOS_BINARY','RACE_LABEL','SEX_LABEL','ETH_LABEL','AGE_GROUP','THCIC_ID','LENGTH_OF_STAY']]
X = df[feature_cols].values
y = df['LOS_BINARY'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
idx_train, idx_test = train_test_split(range(len(df)), test_size=0.2, random_state=42, stratify=y)

print("Training quick HistGBC...")
model = HistGradientBoostingClassifier(max_iter=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]
print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")

protected_attrs = {
    'RACE': df['RACE_LABEL'].values[idx_test],
    'SEX': df['SEX_LABEL'].values[idx_test],
    'ETHNICITY': df['ETH_LABEL'].values[idx_test],
    'AGE_GROUP': df['AGE_GROUP'].values[idx_test],
}

# OLD: Between-group TI
def ti_between(y_true, y_pred, y_prob, attr):
    benefits = 1.0 - np.abs(y_true.astype(float) - y_prob)
    benefits = np.clip(benefits, 1e-10, None)
    mu = benefits.mean()
    if mu <= 0: return 0.0
    groups = sorted(set(attr))
    ti_b = 0.0
    N = len(attr)
    for g in groups:
        mask = attr == g
        n_g = mask.sum()
        if n_g == 0: continue
        mu_g = benefits[mask].mean()
        ratio = mu_g / mu
        if ratio > 0:
            ti_b += (n_g / N) * ratio * np.log(ratio)
    return max(0, float(ti_b))

# NEW: Within-group TI disparity
def ti_disparity(y_true, y_pred, y_prob, attr):
    benefits = 1.0 - np.abs(y_true.astype(float) - y_prob)
    benefits = np.clip(benefits, 1e-10, None)
    groups = sorted(set(attr))
    group_ti = []
    for g in groups:
        mask = attr == g
        n_g = mask.sum()
        if n_g < 10: continue
        b_g = benefits[mask]
        mu_g = b_g.mean()
        if mu_g <= 0: continue
        ratios_g = b_g / mu_g
        ti_g = float(np.mean(ratios_g * np.log(ratios_g + 1e-10)))
        group_ti.append(max(0.0, ti_g))
    if len(group_ti) < 2:
        return 0.0
    return max(group_ti) - min(group_ti)

print("\n=== TI Comparison: Old (Between-Group) vs New (Within-Group Disparity) ===")
print(f"{'Attribute':15s} {'Old TI':>12s} {'New TI':>12s}")
print("-" * 42)
for attr_name, attr_vals in protected_attrs.items():
    old = ti_between(y_test, y_pred, y_prob, attr_vals)
    new = ti_disparity(y_test, y_pred, y_prob, attr_vals)
    print(f"{attr_name:15s} {old:12.8f} {new:12.6f}")
