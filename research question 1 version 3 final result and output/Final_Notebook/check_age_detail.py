"""Check which specific AGE metrics pass/fail at different (a_sr, a_tpr) combos."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, pickle, json
from sklearn.metrics import accuracy_score, roc_auc_score

# Load data from the notebook's output
NB_DIR = os.path.dirname(__file__)
OUT_DIR = os.path.join(NB_DIR, 'output')

# We need to reproduce the candidate computation for specific (a_sr, a_tpr) values
# Load the executed notebook to get the variables
import nbformat
nb = nbformat.read(os.path.join(NB_DIR, 'RQ1_LOS_Fairness_Analysis.ipynb'), as_version=4)

# Extract key data from cells - we need test data
# Simpler: just load the processed data and model predictions
DATA_DIR = os.path.join(NB_DIR, 'output', 'models')

# Load the test data
import importlib.util
# We need the FairnessCalculator class. Let's extract it from the notebook.

# Actually let's just re-run the key computation directly
data_path = r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\data\texas_100x.csv"
print("Loading data...")

# Let me use a more efficient approach - load from the processed notebook outputs
# The notebook saves predictions. Let me just extract from the notebook execution.

# Instead, let me parse the candidate data and check specific cells
# Fine — let me just run the computation quickly using the notebook's saved models

# Actually, the simplest approach: execute the fairness check in the notebook context
# But that's complex. Let me instead check specific candidate metrics from the
# execution output by examining higher-correction points.

# Let me read the CSV and check cross-reference what we can
df = pd.read_csv(os.path.join(OUT_DIR, 'tables', '18b_fairness_candidate_search.csv'))

# For each (A_SR, A_TPR) combo, show the best Age_Fair and the associated DI_AGE
print("\n=== All (A_SR, A_TPR) combinations: max Age_Fair, DI_AGE, DI_RACE ===")
combos = df.groupby(['A_SR', 'A_TPR']).agg(
    max_Age_Fair=('Age_Fair', 'max'),
    max_DI_AGE=('DI_AGE', 'max'),
    best_Total=('Total_Fair', 'max'),
    n=('Age_Fair', 'count')
).reset_index()

for _, row in combos.iterrows():
    print(f"  A_SR={row.A_SR:.1f}, A_TPR={row.A_TPR:.1f}: max_Age_Fair={int(row.max_Age_Fair)}, "
          f"max_DI_AGE={row.max_DI_AGE:.3f}, best_Total={int(row.best_Total)}")

# Check the transition: at which point does Age_Fair=3 switch between
# DI/SPD/TI=3 and EOPP/EOD/TI=3?
print("\n=== Candidates with Age_Fair=3 at different settings ===")
age3 = df[df['Age_Fair']==3].copy()

# Get representative candidates
for a_sr in [0.0, 0.3, 0.6, 0.8, 1.0]:
    for a_tpr in [0.0, 0.3, 0.6, 0.8, 1.0]:
        sub = age3[(age3['A_SR']==a_sr) & (age3['A_TPR']==a_tpr)]
        if len(sub):
            best = sub.nlargest(1, 'Total_Fair').iloc[0]
            print(f"  A_SR={a_sr}, A_TPR={a_tpr}: Model={best.Model}, "
                  f"DI_AGE={best.DI_AGE:.3f}, DI_RACE={best.DI_RACE:.3f}, "
                  f"Race_Fair={int(best.Race_Fair)}, Sex_Fair={int(best.Sex_Fair)}, "
                  f"Acc={best.Accuracy:.4f}")
