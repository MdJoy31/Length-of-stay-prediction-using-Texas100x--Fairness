import pandas as pd, json, os

tables = 'output/tables'

# 1. Check fairness-accuracy tradeoff
df = pd.read_csv(f'{tables}/14d_fairness_accuracy_tradeoff.csv')
print('=== FAIRNESS-ACCURACY TRADEOFF ===')
print(df.to_string(index=False))
print()

# 2. Check candidate search
cand = pd.read_csv(f'{tables}/18b_fairness_candidate_search.csv')
print(f'Total candidates: {len(cand)}')
all_di = cand[(cand['DI_RACE']>=0.80) & (cand['DI_SEX']>=0.80) & (cand['DI_ETH']>=0.80) & (cand['DI_AGE']>=0.80)]
print(f'Candidates with ALL DI>=0.80: {len(all_di)}')
if len(all_di):
    best = all_di.sort_values(['Total_Fair','Age_Fair','Race_Fair','Acc_Drop_pp'], ascending=[False,False,False,True]).iloc[0]
    print(f'  SELECTED: {best["Model"]}, A_SR={best["A_SR"]}, A_TPR={best["A_TPR"]}, A_PPV={best["A_PPV"]}')
    print(f'  DI_RACE={best["DI_RACE"]:.4f}, DI_AGE={best["DI_AGE"]:.4f}, DI_SEX={best["DI_SEX"]:.4f}, DI_ETH={best["DI_ETH"]:.4f}')
    print(f'  Race_Fair={int(best["Race_Fair"])}/7, Age_Fair={int(best["Age_Fair"])}/7, Sex_Fair={int(best["Sex_Fair"])}/7, Eth_Fair={int(best["Eth_Fair"])}/7')
    print(f'  Total_Fair={int(best["Total_Fair"])}/28, Accuracy={best["Accuracy"]:.4f}')
print()

# 3. Check final results JSON
for path in ['output/results/final_results.json', 'results/final_results.json']:
    if os.path.exists(path):
        with open(path) as f:
            res = json.load(f)
        print('=== FINAL RESULTS JSON ===')
        for k,v in res.items():
            if isinstance(v, (int, float, str)):
                print(f'  {k}: {v}')
        break

print()

# 4. Check for any 0-values in the tradeoff table
print('=== 0-VALUE AUDIT (tradeoff table) ===')
numeric_cols = df.select_dtypes(include='number').columns
for col in numeric_cols:
    zeros = df[df[col] == 0]
    if len(zeros) > 0:
        print(f'  Column "{col}" has {len(zeros)} zero(s): models={list(zeros["Model"])}')

# 5. Check key DI values for fair model
print()
print('=== DI VALUES FOR FAIR MODEL ===')
fair_row = df[df['Model'].str.contains('Fair|Reweigh|Thr', case=False)]
if len(fair_row):
    for _, r in fair_row.iterrows():
        print(f'  {r["Model"]}:')
        for col in df.columns:
            if 'DI' in col:
                val = r[col]
                status = '✓ FAIR' if val >= 0.80 else '✗ UNFAIR'
                print(f'    {col} = {val:.4f} {status}')
