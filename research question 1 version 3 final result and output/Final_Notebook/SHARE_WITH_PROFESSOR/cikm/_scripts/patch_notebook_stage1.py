"""Stage 1 notebook patcher — fixes that do NOT require retraining.

Changes:
  Cell 3 (setup):     pin seeds for numpy, random, os, torch (if available), xgboost
  Cell 4 (thresholds): EOPP 0.20->0.10, EOD 0.20->0.10, CAL 0.10->0.05
  Cell 6 (data):       add RACE_MAP/SEX_MAP/ETH_MAP dictionaries with provenance note
  Cell 8 (EDA):        apply race_map labels to the race chart
  Cell 34 (intervention): fix {lam:.0f} -> {lam:.1f} so lambda=0.5 prints correctly
  Cell 35 (results):   save comparison DataFrame to CSV
"""
import json, os, shutil, datetime, sys
sys.stdout.reconfigure(encoding='utf-8')

NB = 'CIKM_2026_LOS_Fairness.ipynb'
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Backup before patching
bkp = f"CIKM_2026_LOS_Fairness.pre-stage1.{datetime.datetime.now():%Y%m%d-%H%M%S}.ipynb"
shutil.copy(NB, bkp)
print(f"[backup] {bkp}")

with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def replace_in_cell(idx, old, new, must_find=True):
    src = ''.join(nb['cells'][idx]['source'])
    if old not in src:
        if must_find:
            raise RuntimeError(f"cell {idx}: expected text not found: {old!r}")
        return False
    new_src = src.replace(old, new)
    nb['cells'][idx]['source'] = new_src.splitlines(keepends=True)
    print(f"[patched] cell {idx}: {old!r} -> {new!r}"[:180])
    return True

# ----- Cell 3: seeds -----
old = """RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)"""
new = """RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
import random
random.seed(RANDOM_STATE)
os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)
try:
    import torch
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
except Exception:
    pass"""
replace_in_cell(3, old, new)

# ----- Cell 4: FairnessCalculator thresholds -----
old = """    # Thresholds follow the four-fifths rule (DI) and literature standards.
    # EOPP/EOD thresholds set at 0.20 following Agarwal et al. (2018) and IBM AIF360
    # recommendations for multi-group settings with heterogeneous base rates.
    THRESHOLDS = {
        'DI':   {'threshold': 0.80, 'direction': 'above'},
        'SPD':  {'threshold': 0.10, 'direction': 'below'},
        'EOPP': {'threshold': 0.20, 'direction': 'below'},
        'EOD':  {'threshold': 0.20, 'direction': 'below'},
        'TI':   {'threshold': 0.10, 'direction': 'below'},
        'PP':   {'threshold': 0.10, 'direction': 'below'},
        'CAL':  {'threshold': 0.10, 'direction': 'below'},
    }"""
new = """    # Thresholds follow the four-fifths rule (DI) and the standard fairness
    # thresholds stated in main.tex Section 4.2 and Table 7 caption.
    # (Historical values EOPP=0.20, EOD=0.20, CAL=0.10 inflated the fair-verdict
    # count by ~3/28 on this dataset; see results/consistency_audit.md.)
    THRESHOLDS = {
        'DI':   {'threshold': 0.80, 'direction': 'above'},
        'SPD':  {'threshold': 0.10, 'direction': 'below'},
        'EOPP': {'threshold': 0.10, 'direction': 'below'},
        'EOD':  {'threshold': 0.10, 'direction': 'below'},
        'TI':   {'threshold': 0.10, 'direction': 'below'},
        'PP':   {'threshold': 0.10, 'direction': 'below'},
        'CAL':  {'threshold': 0.05, 'direction': 'below'},
    }"""
replace_in_cell(4, old, new)

# ----- Cell 6: add race/sex/eth/age label maps with provenance note -----
old = """df['AGE_GROUP'] = df['PAT_AGE'].apply(create_age_groups)

# Display summary"""
new = """df['AGE_GROUP'] = df['PAT_AGE'].apply(create_age_groups)

# ──────────────────────────────────────────────────────────────
# Protected-attribute label maps (explicit, machine-readable)
# ──────────────────────────────────────────────────────────────
# WARNING: The RACE labels below match main.tex Table 1, but a RACE x ETHNICITY
# cross-tab on this dataset shows 83% of RACE=3 and 99% of RACE=2 are coded
# ETHNICITY=1 (Hispanic). Under the labels below, 54% of all patients would be
# simultaneously Black AND Hispanic, which is demographically implausible. The
# labels are either (a) artifacts of synthetic augmentation (texas_100x naming
# suggests 100x oversampling), or (b) a THCIC double-coding pattern where many
# Hispanic patients are also coded RACE=3 or RACE=2. The definitive resolution
# requires the THCIC PUDF data dictionary for FY 2019-2023; until then the
# labels in main.tex should be read with the caveat in Methods.
# See results/demographic_audit.md for the full analysis.
RACE_MAP = {
    0: 'Other/Unknown',
    1: 'Native American',
    2: 'Asian/Pacific Islander',
    3: 'Black',
    4: 'White',
}
SEX_MAP = {0: 'Female', 1: 'Male'}  # verified via LOS rate match
ETH_MAP = {0: 'Non-Hispanic', 1: 'Hispanic'}  # verified via LOS rate match
AGE_GROUP_LABEL_MAP = {
    'Age_0_17': 'Pediatric (<18)',
    'Age_18_39': 'Young Adult (18-39)',
    'Age_40_54': 'Middle-Aged (40-54)',
    'Age_55_64': 'Middle-Aged (55-64)',
    'Age_65_Plus': 'Elderly (65+)',
}
# Manuscript collapses Age_40_54 + Age_55_64 into a single Middle-Aged bucket
# for Table 1; the fairness analysis runs on all 5 groups.

# Display summary"""
replace_in_cell(6, old, new)

# ----- Cell 8: apply race_map in EDA plot -----
old = """race_counts = df['RACE'].value_counts().head(6)
axes[0,2].barh(range(len(race_counts)), race_counts.values, color=PALETTE[3])
axes[0,2].set_yticks(range(len(race_counts)))
axes[0,2].set_yticklabels(race_counts.index, fontsize=9)
axes[0,2].set_title('(c) Race Distribution')"""
new = """race_counts = df['RACE'].value_counts().sort_index()
axes[0,2].barh(range(len(race_counts)), race_counts.values, color=PALETTE[3])
axes[0,2].set_yticks(range(len(race_counts)))
axes[0,2].set_yticklabels([RACE_MAP.get(idx, f'Code {idx}') for idx in race_counts.index], fontsize=9)
axes[0,2].set_title('(c) Race Distribution')"""
replace_in_cell(8, old, new)

# ----- Cell 34: fix {lam:.0f} -> {lam:.1f} so lambda=0.5 prints correctly -----
old = """    print(f"  Trained reweighed λ={lam:.0f}  AUC={roc_auc_score(y_test, model_probs[f'Reweigh_{lam:.0f}']):.4f}")"""
new = """    print(f"  Trained reweighed lambda={lam:g}  AUC={roc_auc_score(y_test, model_probs[f'Reweigh_{lam:g}']):.4f}")"""
replace_in_cell(34, old, new)

# Also fix the dict keys to use :g consistently
old = """    model_probs[f'Reweigh_{lam:.0f}'] = mdl.predict_proba(X_test)[:, 1]
    reweigh_model_objects[f'Reweigh_{lam:.0f}'] = mdl"""
new = """    model_probs[f'Reweigh_{lam:g}'] = mdl.predict_proba(X_test)[:, 1]
    reweigh_model_objects[f'Reweigh_{lam:g}'] = mdl"""
replace_in_cell(34, old, new)

# Also pin the reweighed XGBoost seed explicitly
old = """for lam in [0.5, 1.0, 3.0, 5.0, 10.0, 15.0, 30.0, 50.0, 100.0]:
    sw = build_multi_weights(lam)
    mdl = xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.05,
        tree_method='hist', random_state=RANDOM_STATE,
        eval_metric='logloss', verbosity=0)"""
new = """for lam in [0.5, 1.0, 3.0, 5.0, 10.0, 15.0, 30.0, 50.0, 100.0]:
    sw = build_multi_weights(lam)
    mdl = xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.05,
        tree_method='hist', random_state=RANDOM_STATE, seed=RANDOM_STATE,
        eval_metric='logloss', verbosity=0)"""
replace_in_cell(34, old, new)

# ----- Cell 35: save comparison DataFrame to CSV -----
old = """compare_data = {'Metric': metric_labels, 'Standard': std_vals,
                'Fair (Intersect.)': fair_vals,
                'Change': [f-s for s, f in zip(std_vals, fair_vals)]}
cdf = pd.DataFrame(compare_data)"""
new = """compare_data = {'Metric': metric_labels, 'Standard': std_vals,
                'Fair (Intersect.)': fair_vals,
                'Change': [f-s for s, f in zip(std_vals, fair_vals)]}
cdf = pd.DataFrame(compare_data)
os.makedirs('results', exist_ok=True)
cdf.to_csv('results/intervention_standard_vs_fair.csv', index=False)
cdf.to_csv(f'{TABLES_DIR}/cikm_intervention_comparison.csv', index=False)
print(f"Saved: results/intervention_standard_vs_fair.csv  (rows={len(cdf)})")"""
replace_in_cell(35, old, new)

# Clear any stale outputs on patched cells to force re-execution
for idx in [3, 4, 6, 8, 34, 35]:
    if nb['cells'][idx].get('cell_type') == 'code':
        nb['cells'][idx]['outputs'] = []
        nb['cells'][idx]['execution_count'] = None

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\n[done] Stage 1 notebook patches applied.")
print("Next: run the notebook top-to-bottom to materialise the corrected verdicts and CSVs.")
