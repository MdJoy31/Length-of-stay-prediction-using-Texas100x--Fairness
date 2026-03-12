"""Diagnose AGE_GROUP DI, VFR, and other issues."""
import pandas as pd
import numpy as np

# 1. Load the data
data_path = r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\data\texas_100x.csv"
df = pd.read_csv(data_path)
print(f"Data shape: {df.shape}")

# 2. Check PAT_AGE distribution
print("\n=== PAT_AGE value counts (top 25) ===")
vc = df['PAT_AGE'].value_counts().sort_index()
print(vc.to_string())
print(f"\nPAT_AGE range: {df['PAT_AGE'].min()} to {df['PAT_AGE'].max()}")
print(f"PAT_AGE unique values: {sorted(df['PAT_AGE'].unique())}")

# 3. Create age groups with current mapping
def create_age_groups(age_code):
    if age_code <= 4:    return 'Pediatric'
    elif age_code <= 9:  return 'Young_Adult'
    elif age_code <= 14: return 'Middle_Aged'
    else:                return 'Elderly'

df['AGE_GROUP'] = df['PAT_AGE'].apply(create_age_groups)
df['LOS_BINARY'] = (df['LENGTH_OF_STAY'] > 3).astype(int)

# 4. Age group sizes and base rates
print("\n=== AGE_GROUP Distribution ===")
for g in ['Pediatric', 'Young_Adult', 'Middle_Aged', 'Elderly']:
    mask = df['AGE_GROUP'] == g
    n = mask.sum()
    rate = df.loc[mask, 'LOS_BINARY'].mean()
    print(f"  {g:15s}: N={n:>8,}  LOS>3_rate={rate:.3f}  ({n/len(df)*100:.1f}%)")

# 5. Check DI calculation
# DI = min(sel_rate) / max(sel_rate) across groups
# For now, let's compute DI on the ACTUAL outcomes (not predictions)
print("\n=== Actual outcome rate by group (base rate DI) ===")
for attr in ['RACE', 'SEX_CODE', 'ETHNICITY', 'AGE_GROUP']:
    col = attr
    rates = {}
    for g in sorted(df[col].unique()):
        mask = df[col] == g
        if mask.sum() > 0:
            rates[g] = df.loc[mask, 'LOS_BINARY'].mean()
    vals = list(rates.values())
    di = min(vals)/max(vals) if max(vals) > 0 else 0
    print(f"  {attr}: DI={di:.4f}  rates={rates}")

# 6. Check fairness comparison table
print("\n=== Checking fairness_comparison.csv ===")
try:
    fdf = pd.read_csv('output/tables/06_fairness_comparison.csv')
    print(f"Columns: {list(fdf.columns)}")
    print(f"Shape: {fdf.shape}")
    # Show DI by attribute
    for attr in ['RACE', 'SEX', 'ETHNICITY', 'AGE_GROUP']:
        mask = fdf['Attribute'] == attr
        if mask.sum() > 0:
            dis = fdf.loc[mask, 'DI'].tolist()
            print(f"  {attr} DI across models: min={min(dis):.4f} max={max(dis):.4f} mean={np.mean(dis):.4f}")
except Exception as e:
    print(f"Error: {e}")

# 7. Check VFR table
print("\n=== Checking VFR table ===")
try:
    vdf = pd.read_csv('output/tables/09b_vfr_all_metrics.csv')
    print(f"VFR Columns: {list(vdf.columns)}")
    print(f"VFR Shape: {vdf.shape}")
    # Check non-zero VFR
    if 'VFR' in vdf.columns:
        nonzero = vdf[vdf['VFR'] > 0]
        print(f"Non-zero VFR entries: {len(nonzero)} / {len(vdf)}")
        if len(nonzero) > 0:
            print(nonzero[['Model','Attribute','Metric','VFR','Mean','Margin']].head(20).to_string())
except Exception as e:
    print(f"Error: {e}")
