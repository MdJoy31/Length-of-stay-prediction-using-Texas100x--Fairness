import pandas as pd
df = pd.read_csv('output/tables/18b_fairness_candidate_search.csv')
print(f"Total candidates: {len(df)}")
print(f"\nColumns: {list(df.columns)}")

print("\n=== DI >= 0.80 candidates ===")
hi = df[df['DI_RACE'] >= 0.80]
print(f"Count: {len(hi)}")
if len(hi):
    print(hi[['Model','Alpha','Accuracy','Acc_Drop_pp','DI_RACE','Race_Fair','DI_AGE','Age_Fair']].to_string())

print("\n=== Age_Fair >= 4 candidates ===")
af = df[df['Age_Fair'] >= 4]
print(f"Count: {len(af)}")
if len(af):
    print(af[['Model','Alpha','Accuracy','Acc_Drop_pp','DI_RACE','Race_Fair','DI_AGE','Age_Fair']].to_string())

print("\n=== Best DI per model ===")
for m in df['Model'].unique():
    sub = df[df['Model'] == m]
    best = sub.loc[sub['DI_RACE'].idxmax()]
    print(f"  {m}: DI={best['DI_RACE']:.3f} α={best['Alpha']:.2f} Acc={best['Accuracy']:.4f} "
          f"Acc_Drop={best['Acc_Drop_pp']:.1f}pp Age_Fair={int(best['Age_Fair'])}")

print("\n=== Top 10 by DI (sorted) ===")
top = df.nlargest(10, 'DI_RACE')
print(top[['Model','Alpha','Accuracy','Acc_Drop_pp','DI_RACE','Race_Fair','DI_AGE','Age_Fair']].to_string())

print("\n=== Pareto: DI vs Age_Fair at various alpha ===")
for alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    sub = df[df['Alpha'] == alpha]
    if len(sub):
        best = sub.loc[sub['DI_RACE'].idxmax()]
        print(f"  α={alpha:.1f}: DI={best['DI_RACE']:.3f} Age={int(best['Age_Fair'])} "
              f"Race={int(best['Race_Fair'])} Acc={best['Accuracy']:.4f} ({best['Model']})")
