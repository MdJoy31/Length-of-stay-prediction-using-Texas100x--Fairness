import pandas as pd
df = pd.read_csv('output/tables/18b_fairness_candidate_search.csv')

# Reweigh_3 has best native DI_AGE=0.946
for model in ['Standard', 'Reweigh_1', 'Reweigh_3', 'Reweigh_8', 'Reweigh_15', 'Reweigh_25', 'Reweigh_50']:
    base = df[(df['Model']==model) & (df['A_SR']==0.0) & (df['A_TPR']==0.0) & (df['A_PPV']==0.0)]
    if len(base):
        r = base.iloc[0]
        print(f"{model:12s} base: DI_AGE={r['DI_AGE']:.3f} Age={int(r['Age_Fair'])} Race={int(r['Race_Fair'])} "
              f"Sex={int(r['Sex_Fair'])} Eth={int(r['Eth_Fair'])} T={int(r['Total_Fair'])} Acc={r['Accuracy']:.4f}")

print()
# For each model, find best Age_Fair with DI_RACE >= 0.80
print("Best Age_Fair per model (DI_RACE >= 0.80):")
for model in ['Standard', 'Reweigh_1', 'Reweigh_3', 'Reweigh_8', 'Reweigh_15', 'Reweigh_25', 'Reweigh_50']:
    sub = df[(df['Model']==model) & (df['DI_RACE'] >= 0.80)]
    if len(sub):
        best = sub.sort_values(['Age_Fair', 'Total_Fair'], ascending=False).iloc[0]
        print(f"  {model:12s} Age={int(best['Age_Fair'])} sr={best['A_SR']:.1f} tpr={best['A_TPR']:.1f} ppv={best['A_PPV']:.1f} "
              f"| R={int(best['Race_Fair'])} T={int(best['Total_Fair'])} Acc={best['Accuracy']:.4f}")
    else:
        # Also check without the DI_RACE constraint
        sub_all = df[df['Model']==model]
        best = sub_all.sort_values(['Age_Fair', 'Total_Fair'], ascending=False).iloc[0]
        print(f"  {model:12s} (no elig) Age={int(best['Age_Fair'])} sr={best['A_SR']:.1f} tpr={best['A_TPR']:.1f} ppv={best['A_PPV']:.1f} "
              f"| Race DI={best['DI_RACE']:.3f} T={int(best['Total_Fair'])} Acc={best['Accuracy']:.4f}")
