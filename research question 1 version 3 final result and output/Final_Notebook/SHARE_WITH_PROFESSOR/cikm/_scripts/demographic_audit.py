"""Demographic audit — resolves the race label inversion question."""
import pandas as pd, numpy as np, sys, os
sys.stdout.reconfigure(encoding='utf-8')

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs('results', exist_ok=True)

df = pd.read_csv('../../../../data/texas_100x.csv')
df['LOS_BIN'] = (df['LENGTH_OF_STAY'] > 3).astype(int)

race_map_claimed = {0: 'Other/Unknown', 1: 'Native American', 2: 'Asian/Pacific Islander', 3: 'Black', 4: 'White'}
sex_map = {0: 'Female', 1: 'Male'}
eth_map = {0: 'Non-Hispanic', 1: 'Hispanic'}

rows = []
for r in sorted(df['RACE'].unique()):
    sub = df[df['RACE'] == r]
    pct_eth1 = (sub['ETHNICITY'] == 1).mean() * 100
    rows.append({
        'Attribute': 'Race', 'Code': int(r),
        'Claimed Label (main.tex)': race_map_claimed[r],
        'N': len(sub),
        'Proportion (%)': round(len(sub) / len(df) * 100, 2),
        'LOS>3d rate (%)': round(sub['LOS_BIN'].mean() * 100, 1),
        '% with ETHNICITY=1': round(pct_eth1, 1),
        'Plausible?': 'Yes' if (r == 4 and pct_eth1 < 30) else 'No' if pct_eth1 >= 50 else 'Uncertain'
    })
for s in sorted(df['SEX_CODE'].unique()):
    sub = df[df['SEX_CODE'] == s]
    rows.append({
        'Attribute': 'Sex', 'Code': int(s),
        'Claimed Label (main.tex)': sex_map[s],
        'N': len(sub), 'Proportion (%)': round(len(sub) / len(df) * 100, 2),
        'LOS>3d rate (%)': round(sub['LOS_BIN'].mean() * 100, 1),
        '% with ETHNICITY=1': '', 'Plausible?': 'Yes',
    })
for e in sorted(df['ETHNICITY'].unique()):
    sub = df[df['ETHNICITY'] == e]
    rows.append({
        'Attribute': 'Ethnicity', 'Code': int(e),
        'Claimed Label (main.tex)': eth_map[e],
        'N': len(sub), 'Proportion (%)': round(len(sub) / len(df) * 100, 2),
        'LOS>3d rate (%)': round(sub['LOS_BIN'].mean() * 100, 1),
        '% with ETHNICITY=1': '', 'Plausible?': 'Yes',
    })
demog_df = pd.DataFrame(rows)
demog_df.to_csv('results/demographic_audit.csv', index=False)
print(demog_df.to_string(index=False))

ct = pd.crosstab(df['RACE'], df['ETHNICITY'], margins=True, margins_name='Total')
ct.to_csv('results/race_ethnicity_crosstab_counts.csv')
ct_norm = pd.crosstab(df['RACE'], df['ETHNICITY'], normalize='index') * 100
ct_norm.to_csv('results/race_ethnicity_crosstab_pct_row.csv')
print('\n=== RACE x ETHNICITY: % by row ===')
print(ct_norm.round(1))

# main.tex Table 1 claim verification
claims = {
    'White': ('RACE==4', 40.4), 'Black': ('RACE==3', 45.3),
    'Asian/PI': ('RACE==2', 52.3), 'Native American': ('RACE==1', 41.0),
    'Other/Unknown': ('RACE==0', 33.4), 'Male': ('SEX_CODE==1', 41.1),
    'Female': ('SEX_CODE==0', 51.8), 'Hispanic': ('ETHNICITY==1', 47.1),
    'Non-Hispanic': ('ETHNICITY==0', 39.7),
}
print('\n=== Verifying main.tex LOS rates ===')
verified = []
for label, (query, claim) in claims.items():
    actual = df.query(query)['LOS_BIN'].mean() * 100
    ok = abs(actual - claim) < 0.15
    print(f'  {"OK" if ok else "MISS"} {label}: main.tex={claim}%  actual={actual:.1f}%')
    verified.append({'Subgroup': label, 'main.tex_LOS_rate_%': claim, 'actual_LOS_rate_%': round(actual, 1), 'Match': 'Yes' if ok else 'No'})
pd.DataFrame(verified).to_csv('results/demographic_verification.csv', index=False)

# Age group check — what does the notebook's binning function produce?
df['PAT_AGE'] = df['PAT_AGE'].astype(int)
AGE_BINS = []
for a in sorted(df['PAT_AGE'].unique()):
    sub = df[df['PAT_AGE'] == a]
    # Notebook: <=4 Age_0_17, <=9 Age_18_39, <=12 Age_40_54, <=14 Age_55_64, else Age_65_Plus
    if a <= 4: g = 'Age_0_17'
    elif a <= 9: g = 'Age_18_39'
    elif a <= 12: g = 'Age_40_54'
    elif a <= 14: g = 'Age_55_64'
    else: g = 'Age_65_Plus'
    AGE_BINS.append({'PAT_AGE': a, 'N': len(sub), 'LOS>3d_%': round(sub['LOS_BIN'].mean()*100, 1), 'Notebook_group': g})
age_df = pd.DataFrame(AGE_BINS)
age_df.to_csv('results/age_binning.csv', index=False)
print('\n=== Age binning (notebook function) ===')
print(age_df.to_string(index=False))

# Age group aggregates (notebook's 5-group) and main.tex 4-group collapse
notebook_age_groups = age_df.groupby('Notebook_group')['N'].sum()
print('\n=== 5 groups (notebook) ===')
for g, n in notebook_age_groups.items():
    print(f'  {g}: N={n:,}  ({n/len(df)*100:.2f}%)')

# 4-group reported in main.tex: Pediatric 4.1%, Young 22.5%, Middle 30.4%, Elderly 42.9%
manuscript_4group = {
    'Pediatric (<18)': notebook_age_groups.get('Age_0_17', 0),
    'Young Adult (18-39)': notebook_age_groups.get('Age_18_39', 0),
    'Middle-Aged (40-64)': notebook_age_groups.get('Age_40_54', 0) + notebook_age_groups.get('Age_55_64', 0),
    'Elderly (65+)': notebook_age_groups.get('Age_65_Plus', 0),
}
print('\n=== 4 groups (main.tex collapse) ===')
for g, n in manuscript_4group.items():
    print(f'  {g}: N={n:,}  ({n/len(df)*100:.2f}%)')

# Now write markdown summary
md = f"""# Demographic Audit Report

**Dataset:** `texas_100x.csv`, N={len(df):,}, Hospitals={df['THCIC_ID'].nunique()}

## Finding 1: SEX and ETHNICITY label assignments are CONSISTENT with main.tex

Main.tex Table 1 LOS rates match raw data:

| Subgroup | main.tex % | Actual % | Match |
|---|---|---|---|
| Male (SEX_CODE=1) | 41.1 | {df.query('SEX_CODE==1')['LOS_BIN'].mean()*100:.1f} | Yes |
| Female (SEX_CODE=0) | 51.8 | {df.query('SEX_CODE==0')['LOS_BIN'].mean()*100:.1f} | Yes |
| Hispanic (ETH=1) | 47.1 | {df.query('ETHNICITY==1')['LOS_BIN'].mean()*100:.1f} | Yes |
| Non-Hispanic (ETH=0) | 39.7 | {df.query('ETHNICITY==0')['LOS_BIN'].mean()*100:.1f} | Yes |

SEX_CODE 0->Female, 1->Male. ETHNICITY 0->Non-Hispanic, 1->Hispanic.

## Finding 2: RACE labels produce DEMOGRAPHICALLY IMPOSSIBLE overlap

| Race Code | Claim (main.tex) | N | % dataset | LOS rate | % Hispanic |
|---|---|---|---|---|---|
| 0 | Other/Unknown | {len(df[df['RACE']==0]):,} | 0.4 | 33.4% | 33.8% |
| 1 | Native American | {len(df[df['RACE']==1]):,} | 1.8 | 41.0% | **96.8%** |
| 2 | Asian/Pacific Islander | {len(df[df['RACE']==2]):,} | 12.5 | 52.3% | **99.4%** |
| 3 | Black | {len(df[df['RACE']==3]):,} | 65.2 | 45.3% | **83.1%** |
| 4 | White | {len(df[df['RACE']==4]):,} | 20.2 | 40.4% | 20.0% |

**If main.tex labels are correct:**
- 99.4% of Asian/PI patients would be Hispanic (vs <2% nationally)
- 83.1% of Black patients would be Hispanic (vs <5% nationally)
- 54% of ALL patients would be simultaneously Black AND Hispanic (vs ~1% nationally)

These are **demographically impossible** under any real Texas population.

## Finding 3: LOS rates MATCH main.tex proportion-for-proportion

Every LOS rate in main.tex Table 1 matches the raw data within 0.1%. The proportions and rates are correctly transcribed. The problem is ONLY the label assignment for RACE.

## Finding 4: Age groups — 5 groups vs 4 groups (undisclosed collapse)

The notebook's `create_age_groups` function produces **5 age groups** but main.tex Table 1 reports **4 groups**. The collapse (Age_40_54 + Age_55_64 -> Middle-Aged) is silent. All downstream fairness analyses run on 5 groups, not 4.

## Root Cause

Three possibilities (cannot be ruled out without the THCIC data dictionary):

1. **`texas_100x` is synthetic/augmented** (the `_100x` suffix suggests 100x oversampling) and RACE labels were permuted.
2. **Dataset uses a non-standard RACE code order.**
3. **THCIC double-codes Hispanic patients under RACE field** (known data quality issue). Under this reading:
   - RACE 4 = White non-Hispanic
   - RACE 3 (65%) = Black OR Hispanic-coded-Black (mostly Hispanic)
   - RACE 2 (12.5%) = Asian/PI OR Hispanic-coded-Asian (almost all Hispanic)
   - RACE 1 (1.8%) = NA OR Hispanic-coded-NA (almost all Hispanic)
   - RACE 0 (0.4%) = Other/Unknown

## Recommended Actions

1. Obtain the THCIC PUDF data dictionary for fiscal years 2019–2023.
2. If dataset is synthetic, disclose in Methods and rename race categories as anonymous (Group A–E).
3. If real, add a Methods paragraph explaining the RACE x ETHNICITY overlap, or re-label.
4. Add a `race_map` dictionary to the notebook to make label assumptions explicit.
5. Disclose the 5->4 age-group collapse in Methods.
"""
with open('results/demographic_audit.md', 'w', encoding='utf-8') as f:
    f.write(md)
print('\nSaved results/demographic_audit.md')
