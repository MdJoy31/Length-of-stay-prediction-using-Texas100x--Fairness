import pandas as pd
import numpy as np

# Deep analysis of candidates
df = pd.read_csv('output/tables/18b_fairness_candidate_search.csv')
print('=== All columns ===')
print(list(df.columns))
print(f'\nTotal candidates: {len(df)}')

# Find candidates where DI_AGE >= 0.80
age_fair = df[df['DI_AGE'] >= 0.80]
print(f'\nCandidates with DI_AGE >= 0.80: {len(age_fair)}')
if len(age_fair):
    for _, r in age_fair.iterrows():
        print(f'  Model={r["Model"]}, A_SR={r["A_SR"]}, A_TPR={r["A_TPR"]}, A_PPV={r["A_PPV"]}')
        print(f'  Acc={r["Accuracy"]:.4f}, Total_Fair={r["Total_Fair"]}')
        print(f'  DI_RACE={r["DI_RACE"]:.4f}, DI_AGE={r["DI_AGE"]:.4f}, DI_SEX={r["DI_SEX"]:.4f}, DI_ETH={r["DI_ETH"]:.4f}')
        print(f'  Race_Fair={r["Race_Fair"]}, Age_Fair={r["Age_Fair"]}, Sex_Fair={r["Sex_Fair"]}, Eth_Fair={r["Eth_Fair"]}')
        print()

# Find candidates where ALL DIs >= 0.80
all_di_fair = df[(df['DI_RACE'] >= 0.80) & (df['DI_AGE'] >= 0.80) & (df['DI_SEX'] >= 0.80) & (df['DI_ETH'] >= 0.80)]
print(f'\nCandidates with ALL DIs >= 0.80: {len(all_di_fair)}')
if len(all_di_fair):
    best = all_di_fair.nlargest(5, 'Total_Fair')
    for _, r in best.iterrows():
        print(f'  Model={r["Model"]}, A_SR={r["A_SR"]}, A_TPR={r["A_TPR"]}, A_PPV={r["A_PPV"]}')
        print(f'  Acc={r["Accuracy"]:.4f}, Total_Fair={r["Total_Fair"]}')
        print(f'  DI_RACE={r["DI_RACE"]:.4f}, DI_AGE={r["DI_AGE"]:.4f}, DI_SEX={r["DI_SEX"]:.4f}, DI_ETH={r["DI_ETH"]:.4f}')
        print(f'  Race_Fair={r["Race_Fair"]}, Age_Fair={r["Age_Fair"]}, Sex_Fair={r["Sex_Fair"]}, Eth_Fair={r["Eth_Fair"]}')
        print()

# What is the maximum DI_AGE achievable?
print(f'\nMax DI_AGE across all candidates: {df["DI_AGE"].max():.4f}')
best_age = df.loc[df['DI_AGE'].idxmax()]
print(f'  At: Model={best_age["Model"]}, A_SR={best_age["A_SR"]}, A_TPR={best_age["A_TPR"]}, Acc={best_age["Accuracy"]:.4f}')

# What happens at high a_sr (1.0)?
high_sr = df[df['A_SR'] == 1.0]
print(f'\nCandidates with A_SR=1.0: {len(high_sr)}')
print(f'  DI_AGE range: [{high_sr["DI_AGE"].min():.4f}, {high_sr["DI_AGE"].max():.4f}]')
print(f'  DI_RACE range: [{high_sr["DI_RACE"].min():.4f}, {high_sr["DI_RACE"].max():.4f}]')
print(f'  Max Total_Fair at A_SR=1.0: {high_sr["Total_Fair"].max()}')

# Check DI_AGE distribution by A_SR
for asr in sorted(df['A_SR'].unique()):
    sub = df[df['A_SR'] == asr]
    print(f'\n  A_SR={asr}: DI_AGE mean={sub["DI_AGE"].mean():.4f}, max={sub["DI_AGE"].max():.4f}')

# Check the selection rates per group from the main data
print('\n\n=== Checking base rates from 30-subset data ===')
sub_df = pd.read_csv('output/tables/16_30_random_subsets.csv')
print('Columns:', list(sub_df.columns))
age_sub = sub_df[sub_df['Attribute'] == 'AGE_GROUP']
print(f'\nAGE_GROUP rows: {len(age_sub)}')
if 'DI' in age_sub.columns:
    print(f'DI: mean={age_sub["DI"].mean():.4f}, min={age_sub["DI"].min():.4f}, max={age_sub["DI"].max():.4f}')
