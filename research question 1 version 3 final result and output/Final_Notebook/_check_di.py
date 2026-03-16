import pandas as pd
import os

# Check fairness tradeoff
fpath = 'output/tables/14d_fairness_accuracy_tradeoff.csv'
if os.path.exists(fpath):
    df = pd.read_csv(fpath)
    print('=== 14d_fairness_accuracy_tradeoff.csv ===')
    print('Columns:', list(df.columns))
    print()
    di_cols = [c for c in df.columns if 'DI' in c.upper()]
    fair_cols = [c for c in df.columns if '_Fair' in c]
    for _, r in df.iterrows():
        model = r.get('Model', '?')
        lam = r.get('Lambda', '?')
        acc = r.get('Accuracy', '?')
        print(f'Model={model}, Lambda={lam}, Acc={acc}')
        for c in di_cols:
            print(f'  {c} = {r[c]}')
        for c in fair_cols:
            print(f'  {c} = {r[c]}')
        print()

# Check fairness comparison
fpath2 = 'output/tables/06_fairness_comparison.csv'
if os.path.exists(fpath2):
    df2 = pd.read_csv(fpath2)
    print('=== 06_fairness_comparison.csv ===')
    print('Columns:', list(df2.columns))
    print()
    for _, r in df2.iterrows():
        print(dict(r))
        print()

# Check candidate search
fpath3 = 'output/tables/18b_fairness_candidate_search.csv'
if os.path.exists(fpath3):
    df3 = pd.read_csv(fpath3)
    print('=== 18b_fairness_candidate_search.csv ===')
    print('Shape:', df3.shape)
    print('Columns:', list(df3.columns))
    # Show top candidates by Total_Fair
    if 'Total_Fair' in df3.columns:
        top = df3.nlargest(5, 'Total_Fair')
        for _, r in top.iterrows():
            di_cols3 = [c for c in df3.columns if 'DI' in c]
            fair_cols3 = [c for c in df3.columns if '_Fair' in c]
            print(f'Lambda={r.get("Lambda","?")}, a_sr={r.get("a_sr","?")}, a_tpr={r.get("a_tpr","?")}, a_ppv={r.get("a_ppv","?")}')
            print(f'  Acc={r.get("Accuracy","?")}, Total_Fair={r.get("Total_Fair","?")}')
            for c in di_cols3:
                print(f'  {c} = {r[c]}')
            for c in fair_cols3:
                print(f'  {c} = {r[c]}')
            print()
