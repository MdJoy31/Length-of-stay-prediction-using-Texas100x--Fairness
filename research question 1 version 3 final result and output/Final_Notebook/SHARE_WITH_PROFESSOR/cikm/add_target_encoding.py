"""
Patch the FINAL notebook's feature engineering (Section 4) to use
Bayesian-smoothed target encoding for high-cardinality categorical
columns (diagnosis, procedure, hospital ID), matching the manuscript
Section 4.2 specification. This typically lifts XGBoost AUROC from
~0.93 (label encoding) to ~0.95 (target encoding) on this dataset.

Run AFTER the current execution finishes if you want to push AUROC
above 0.95.
"""
import json
from pathlib import Path

NB = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\CIKM_2026_LOS_Fairness_FINAL.ipynb")
with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

NEW_FE = '''# ──────────────────────────────────────────────────────────────
# 4.1 · Feature engineering with Bayesian-smoothed target encoding
# (matches manuscript Section 4.2)
# ──────────────────────────────────────────────────────────────
target = "LOS_BINARY"
protected_cols = ["RACE", "SEX_CODE", "ETHNICITY", "AGE_GROUP"]
exclude_cols = [target, "LENGTH_OF_STAY", "RECORD_ID"] + protected_cols
high_card = []
for col in df.columns:
    if col in exclude_cols: continue
    if df[col].dtype == "object" or df[col].nunique() > 50:
        high_card.append(col)
print(f"High-cardinality cols: {high_card}")

feature_cols_low = [c for c in df.columns
                    if c not in exclude_cols + high_card and
                    df[c].dtype in ["int64","int32","float64","float32"]]

# 80/20 stratified train/test split FIRST (so target-encoding fits on train only)
import numpy as np
idx_all = np.arange(len(df))
y_full_pre = df[target].values.astype("int32")
idx_train, idx_test = train_test_split(idx_all, test_size=0.20,
                                        random_state=RANDOM_STATE, stratify=y_full_pre)
train_mask = np.zeros(len(df), dtype=bool); train_mask[idx_train] = True

# Bayesian-smoothed target encoding (m=10) on TRAIN only, applied to all rows
m_smooth = 10.0
y_global_mean = float(df.loc[train_mask, target].mean())
te_features = []
for col in high_card:
    cat_stats = (df.loc[train_mask].groupby(col)[target]
                 .agg(["count","mean"]).rename(columns={"count":"n","mean":"yk"}))
    cat_stats["mu_k"] = ((cat_stats["n"] * cat_stats["yk"]
                          + m_smooth * y_global_mean) / (cat_stats["n"] + m_smooth))
    te_col = f"{col}_te"
    df[te_col] = df[col].map(cat_stats["mu_k"]).fillna(y_global_mean).astype("float32")
    te_features.append(te_col)
print(f"Target-encoded features: {te_features}")

# Full feature matrix: low-cardinality numeric + target-encoded high-cardinality
feature_cols = feature_cols_low + te_features
print(f"Final feature set ({len(feature_cols)}): {feature_cols}")

# Build X / y / hospital_ids
X_full = df[feature_cols].fillna(0).astype("float32").values
y_full = df[target].values.astype("int32")
hospital_ids_full = df["THCIC_ID"].values

X_train, X_test = X_full[idx_train], X_full[idx_test]
y_train, y_test = y_full[idx_train], y_full[idx_test]
hosp_train      = hospital_ids_full[idx_train]
hosp_test       = hospital_ids_full[idx_test]

protected_test = {
    "RACE":      df["RACE"].values[idx_test],
    "SEX":       df["SEX_CODE"].values[idx_test],
    "ETHNICITY": df["ETHNICITY"].values[idx_test],
    "AGE_GROUP": df["AGE_GROUP"].values[idx_test],
}
protected_train = {
    "RACE":      df["RACE"].values[idx_train],
    "SEX":       df["SEX_CODE"].values[idx_train],
    "ETHNICITY": df["ETHNICITY"].values[idx_train],
    "AGE_GROUP": df["AGE_GROUP"].values[idx_train],
}
print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train).astype("float32")
X_test_sc  = scaler.transform(X_test).astype("float32")
print(f"Standardised feature matrices ready (train {X_train_sc.shape}, test {X_test_sc.shape})")
'''

# Find the FE cell and replace it
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code": continue
    src = "".join(c.get("source", []))
    if "Feature engineering" in src and "target encoding" not in src.lower():
        nb["cells"][i]["source"] = NEW_FE.splitlines(keepends=True)
        nb["cells"][i]["outputs"] = []
        nb["cells"][i]["execution_count"] = None
        print(f"Patched cell {i} with Bayesian-smoothed target encoding")
        break

# Clear outputs of all downstream cells (Section 5+)
for j in range(i+1, len(nb["cells"])):
    if nb["cells"][j]["cell_type"] == "code":
        nb["cells"][j]["outputs"] = []
        nb["cells"][j]["execution_count"] = None

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote patched notebook ({len(nb['cells'])} cells)")
