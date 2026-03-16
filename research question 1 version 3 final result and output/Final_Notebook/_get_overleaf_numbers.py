import pandas as pd

tables = 'output/tables'

# Tradeoff table
df = pd.read_csv(f'{tables}/14d_fairness_accuracy_tradeoff.csv')

print("=== MODEL COMPARISON FOR OVERLEAF UPDATE ===\n")

# Standard model
std = df[df['Model'] == 'LGB-XGB Blend'].iloc[0]
fair = df[df['Model'] == 'Fair (Reweigh+Thr)'].iloc[0]
afce_lgb = df[df['Model'] == 'AFCE-LightGBM'].iloc[0]
afce_xgb = df[df['Model'] == 'AFCE-XGBoost'].iloc[0]

# Lambda=3 from candidate search
cand = pd.read_csv(f'{tables}/18b_fairness_candidate_search.csv')
# Lambda 3 best candidate
lam3 = cand[cand['Model'] == 'Reweigh_3']
if len(lam3):
    best_lam3 = lam3.sort_values('Total_Fair', ascending=False).iloc[0]
    print(f"Lambda=3 best: Total_Fair={int(best_lam3['Total_Fair'])}, Acc={best_lam3['Accuracy']:.4f}")
    print(f"  DI_RACE={best_lam3['DI_RACE']:.3f}, DI_AGE={best_lam3['DI_AGE']:.3f}, DI_SEX={best_lam3['DI_SEX']:.3f}, DI_ETH={best_lam3['DI_ETH']:.3f}")

print()
for name, row in [("Standard", std), ("AFCE-LightGBM", afce_lgb), ("AFCE-XGBoost", afce_xgb), ("Fair", fair)]:
    print(f"{name}:")
    print(f"  Acc={row['Accuracy']:.4f}, AUC={row['AUC']:.4f}, F1={row['F1']:.4f}")
    print(f"  Fair_Verdicts={int(row['Fair_Verdicts'])}, CFS={row['CFS']:.4f}")
    print(f"  DI_RACE={row['DI_RACE']:.3f}, DI_SEX={row['DI_SEX']:.3f}, DI_ETH={row['DI_ETHNICITY']:.3f}, DI_AGE={row['DI_AGE_GROUP']:.3f}")
    print()

# Acc drop
print(f"Acc drop: {(fair['Accuracy'] - std['Accuracy'])*100:.2f} pp")
print(f"AUC drop: {(fair['AUC'] - std['AUC'])*100:.2f} pp")
print(f"Verdict gain: {int(fair['Fair_Verdicts']) - int(std['Fair_Verdicts'])}")

# RACE verdicts for fair model
print("\n=== FAIR MODEL DETAILED VERDICTS FOR OVERLEAF TABLE ===")
thresholds_di = {'DI': (0.80, '>='), 'SPD': (0.10, '<'), 'EOPP': (0.10, '<'),
                 'EOD': (0.10, '<'), 'TI': (0.10, '<'), 'PP': (0.10, '<'), 'CAL': (0.05, '<')}

for attr, col_suffix in [('Race', 'RACE'), ('Sex', 'SEX'), ('Ethnicity', 'ETHNICITY'), ('Age Group', 'AGE_GROUP')]:
    vals = []
    for metric, (thresh, op) in thresholds_di.items():
        col = f"{metric}_{col_suffix}"
        val = fair[col]
        if metric == 'DI':
            is_fair = val >= thresh
        else:
            is_fair = abs(val) < thresh
        mark = "checkmark" if is_fair else "ding{55}"
        vals.append(f"{val:.3f}\\,\\{mark}")
    fair_count = sum(1 for v in vals if 'checkmark' in v)
    print(f"  {attr:12s} & {' & '.join(vals)} & {fair_count}/7")

# Standard DI values with 5 bins
print("\n=== STANDARD MODEL DI VALUES (for comparison) ===")
print(f"  DI_RACE={std['DI_RACE']:.3f}, DI_SEX={std['DI_SEX']:.3f}, DI_ETH={std['DI_ETHNICITY']:.3f}, DI_AGE={std['DI_AGE_GROUP']:.3f}")

# Base rates for age groups
print("\n=== AGE GROUP BASE RATES (for impossibility section) ===")
try:
    full_df = pd.read_csv('output/tables/01_descriptive_statistics.csv')
    print(full_df.to_string())
except:
    print("(descriptive stats not found)")
