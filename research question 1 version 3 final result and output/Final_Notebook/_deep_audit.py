import pandas as pd, os, glob

base = 'd:/Research study/Research question ML/fairness_project_v2/fairness_project_v1/research question 1 version 3 final result and output/Final_Notebook/output/tables'

print("=== DEEP 0-VALUE AUDIT ACROSS ALL CSV FILES ===\n")

found_issues = []
for csv_file in sorted(glob.glob(f'{base}/*.csv')):
    fname = os.path.basename(csv_file)
    df = pd.read_csv(csv_file)
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        zeros = df[df[col] == 0]
        if len(zeros) > 0:
            # Check if this is a naturally-zero column (like counts, or identifiers)
            pct = len(zeros) / len(df) * 100
            if pct < 100:  # Don't report if ALL values are zero (probably placeholder)
                found_issues.append((fname, col, len(zeros), pct))

print(f"Found {len(found_issues)} columns with zero values:\n")
for fname, col, cnt, pct in found_issues:
    print(f"  {fname} → {col}: {cnt} zeros ({pct:.1f}%)")

# Focus on fairness tables
print("\n\n=== FAIR MODEL DETAILED VERDICTS ===\n")
tradeoff = pd.read_csv(f'{base}/14d_fairness_accuracy_tradeoff.csv')
fair_row = tradeoff[tradeoff['Model'] == 'Fair (Reweigh+Thr)']
if len(fair_row):
    r = fair_row.iloc[0]
    thresholds = {
        'DI': 0.80,
        'SPD': 0.10,
        'EOPP': 0.10,
        'EOD': 0.10,
        'TI': 0.10,
        'PP': 0.10,
        'CAL': 0.05
    }

    for attr in ['RACE', 'SEX', 'ETHNICITY', 'AGE_GROUP']:
        fairs = 0
        unfairs = 0
        print(f"\n  {attr}:")
        for metric, thresh in thresholds.items():
            col = f'{metric}_{attr}'
            if col in r:
                val = r[col]
                if metric == 'DI':
                    is_fair = val >= thresh
                else:
                    is_fair = abs(val) < thresh
                status = '✓' if is_fair else '✗'
                if is_fair:
                    fairs += 1
                else:
                    unfairs += 1
                print(f"    {metric}: {val:.4f} (threshold: {'≥' if metric=='DI' else '<'}{thresh}) {status}")
        print(f"    → {fairs}/7 FAIR")

# Check for the candidate that was actually selected
print("\n\n=== SELECTED CANDIDATE DETAILS ===\n")
cand = pd.read_csv(f'{base}/18b_fairness_candidate_search.csv')
all_di = cand[(cand['DI_RACE']>=0.80) & (cand['DI_SEX']>=0.80) & (cand['DI_ETH']>=0.80) & (cand['DI_AGE']>=0.80)]
best = all_di.sort_values(['Total_Fair','Age_Fair','Race_Fair','Acc_Drop_pp'], ascending=[False,False,False,True]).iloc[0]
for col in cand.columns:
    print(f"  {col}: {best[col]}")
