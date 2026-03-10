"""Diagnose TI = 0 issue."""
import pandas as pd
import numpy as np

df = pd.read_csv('output/tables/06_fairness_comparison.csv')

print('=== All TI values ===')
for _, row in df.iterrows():
    print(f"  {row['Model']:25s} {row['Attribute']:12s} TI={row['TI']:.15f}")

print(f"\nTI range: {df['TI'].min():.15f} to {df['TI'].max():.15f}")
print(f"Exact zeros: {(df['TI'] == 0).sum()} / {len(df)}")
print(f"Near zeros (<1e-6): {(df['TI'] < 1e-6).sum()} / {len(df)}")

print("\n=== Comparison with other metrics ===")
metrics = ['DI','SPD','EOPP','EOD','TI','PP','CAL']
for m in metrics:
    vals = df[m]
    print(f"  {m:5s}: min={vals.min():.8f}  max={vals.max():.8f}  mean={vals.mean():.8f}")
