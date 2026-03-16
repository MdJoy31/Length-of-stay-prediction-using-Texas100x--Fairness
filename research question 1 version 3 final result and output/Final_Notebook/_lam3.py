import pandas as pd

cand = pd.read_csv('output/tables/18b_fairness_candidate_search.csv')
lam3 = cand[cand['Model'] == 'Reweigh_3'].sort_values('Total_Fair', ascending=False).iloc[0]
print(f"Lambda=3 best: Acc={lam3['Accuracy']:.4f}, AUC={lam3['AUC']:.4f}")
print(f"  DI_RACE={lam3['DI_RACE']:.3f}, DI_AGE={lam3['DI_AGE']:.3f}")
print(f"  Total_Fair={int(lam3['Total_Fair'])}")
