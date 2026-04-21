"""D0 Consistency Audit — Reconcile Tables 7, 10, 11 from main.tex under claimed thresholds."""
import pandas as pd, sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs('results', exist_ok=True)

# Standard thresholds per main.tex Section 4.2 and Table 7 caption
TH_STANDARD = {
    'DI':   ('above', 0.80),
    'SPD':  ('below', 0.10),
    'EOPP': ('below', 0.10),
    'EOD':  ('below', 0.10),
    'TI':   ('below', 0.10),
    'PP':   ('below', 0.10),
    'CAL':  ('below', 0.05),
}
# Non-standard thresholds actually present in the notebook's FairnessCalculator
TH_NOTEBOOK = {
    'DI':   ('above', 0.80),
    'SPD':  ('below', 0.10),
    'EOPP': ('below', 0.20),
    'EOD':  ('below', 0.20),
    'TI':   ('below', 0.10),
    'PP':   ('below', 0.10),
    'CAL':  ('below', 0.10),
}

def passes(value, metric, thresholds):
    direction, tau = thresholds[metric]
    if direction == 'above':
        return value >= tau
    return value <= tau

# main.tex Table 7 — LGB-XGB Blend fairness metric point values
TABLE7 = {
    ('Race', 'DI'):   0.652, ('Race', 'SPD'):   0.176, ('Race', 'EOPP'): 0.053, ('Race', 'EOD'): 0.053,
    ('Race', 'TI'):   0.005, ('Race', 'PP'):    0.095, ('Race', 'CAL'):  0.245,
    ('Sex', 'DI'):    0.761, ('Sex', 'SPD'):    0.124, ('Sex', 'EOPP'):  0.036, ('Sex', 'EOD'):  0.049,
    ('Sex', 'TI'):    0.005, ('Sex', 'PP'):     0.004, ('Sex', 'CAL'):   0.026,
    ('Ethnicity', 'DI'):  0.830, ('Ethnicity', 'SPD'): 0.078, ('Ethnicity', 'EOPP'): 0.019, ('Ethnicity', 'EOD'): 0.029,
    ('Ethnicity', 'TI'):  0.003, ('Ethnicity', 'PP'):  0.001, ('Ethnicity', 'CAL'):  0.034,
    ('Age Group', 'DI'):  0.291, ('Age Group', 'SPD'): 0.436, ('Age Group', 'EOPP'): 0.184, ('Age Group', 'EOD'): 0.184,
    ('Age Group', 'TI'):  0.014, ('Age Group', 'PP'):  0.070, ('Age Group', 'CAL'):  0.065,
}
# main.tex Table 11 — pass counts (x/30 subsets passing)
TABLE11_PASS_COUNT = {
    ('Race', 'DI'): 0, ('Race', 'SPD'): 0, ('Race', 'EOPP'): 16, ('Race', 'EOD'): 20,
    ('Race', 'TI'): 30, ('Race', 'PP'): 17, ('Race', 'CAL'): 0,
    ('Sex', 'DI'): 1, ('Sex', 'SPD'): 1, ('Sex', 'EOPP'): 30, ('Sex', 'EOD'): 30,
    ('Sex', 'TI'): 30, ('Sex', 'PP'): 30, ('Sex', 'CAL'): 27,
    ('Ethnicity', 'DI'): 28, ('Ethnicity', 'SPD'): 29, ('Ethnicity', 'EOPP'): 30, ('Ethnicity', 'EOD'): 30,
    ('Ethnicity', 'TI'): 30, ('Ethnicity', 'PP'): 30, ('Ethnicity', 'CAL'): 26,
    ('Age Group', 'DI'): 0, ('Age Group', 'SPD'): 0, ('Age Group', 'EOPP'): 0, ('Age Group', 'EOD'): 0,
    ('Age Group', 'TI'): 30, ('Age Group', 'PP'): 22, ('Age Group', 'CAL'): 0,
}
# main.tex Table 10 — reported VFR values (%)
TABLE10_VFR = {
    ('Race', 'DI'): 0.0, ('Race', 'SPD'): 0.0, ('Race', 'EOPP'): 46.7, ('Race', 'EOD'): 33.3,
    ('Race', 'TI'): 0.0, ('Race', 'PP'): 43.3, ('Race', 'CAL'): 0.0,
    ('Sex', 'DI'): 3.3, ('Sex', 'SPD'): 3.3, ('Sex', 'EOPP'): 0.0, ('Sex', 'EOD'): 0.0,
    ('Sex', 'TI'): 0.0, ('Sex', 'PP'): 0.0, ('Sex', 'CAL'): 10.0,
    ('Ethnicity', 'DI'): 6.7, ('Ethnicity', 'SPD'): 3.3, ('Ethnicity', 'EOPP'): 0.0, ('Ethnicity', 'EOD'): 0.0,
    ('Ethnicity', 'TI'): 0.0, ('Ethnicity', 'PP'): 0.0, ('Ethnicity', 'CAL'): 13.3,
    ('Age Group', 'DI'): 0.0, ('Age Group', 'SPD'): 0.0, ('Age Group', 'EOPP'): 0.0, ('Age Group', 'EOD'): 0.0,
    ('Age Group', 'TI'): 0.0, ('Age Group', 'PP'): 26.7, ('Age Group', 'CAL'): 0.0,
}

# Verdict Flip Rate from pass count
def vfr_from_count(pass_count, K=30):
    fail_count = K - pass_count
    return round(min(pass_count, fail_count) / K * 100, 1)

def _near_threshold(val, metric, th):
    direction, tau = th[metric]
    return abs(val - tau) / max(abs(tau), 1e-9) < 0.2

rows = []
for (attr, metric), val in TABLE7.items():
    pass_standard = passes(val, metric, TH_STANDARD)
    pass_notebook = passes(val, metric, TH_NOTEBOOK)
    th_std = TH_STANDARD[metric]
    th_nb = TH_NOTEBOOK[metric]
    pass_count = TABLE11_PASS_COUNT[(attr, metric)]
    vfr_reported = TABLE10_VFR[(attr, metric)]
    vfr_computed = vfr_from_count(pass_count)
    table10_11_agree = abs(vfr_reported - vfr_computed) < 0.1
    contradiction = (pass_standard and pass_count < 15) or ((not pass_standard) and pass_count > 15)
    direction, tau = TH_STANDARD[metric]
    near = abs(val - tau) / max(abs(tau), 1e-9) < 0.2
    rows.append({
        'Attribute': attr,
        'Metric': metric,
        'Value (Table 7)': val,
        'Threshold (standard)': f"{direction} {tau}",
        'Pass (standard thresholds)': 'Pass' if pass_standard else 'Fail',
        'Pass (notebook 0.20/0.10)': 'Pass' if pass_notebook else 'Fail',
        'Verdicts match across thresholds?': 'Yes' if pass_standard == pass_notebook else 'NO',
        'Table 11 pass count': f"{pass_count}/30",
        'Table 10 VFR reported (%)': vfr_reported,
        'VFR computed from Table 11 (%)': vfr_computed,
        'Table 10<->11 consistent?': 'Yes' if table10_11_agree else 'NO',
        'Near threshold (|v-tau|/tau<0.2)': 'Yes' if near else 'No',
        'Table7<->Table11 hard contradiction?': 'Yes' if contradiction else 'No',
    })

df = pd.DataFrame(rows)
df.to_csv('results/consistency_audit.csv', index=False)

# Summary stats
n_flipped_by_threshold = sum(1 for r in rows if r['Verdicts match across thresholds?'] == 'NO')
n_internal_contradictions = sum(1 for r in rows if r['Table7<->Table11 hard contradiction?'] == 'Yes')
n_table10_11_disagree = sum(1 for r in rows if r['Table 10<->11 consistent?'] == 'NO')
n_near_threshold = sum(1 for r in rows if r['Near threshold (|v-tau|/tau<0.2)'] == 'Yes')

print(f"Total rows: {len(rows)}")
print(f"Rows where standard vs notebook thresholds disagree: {n_flipped_by_threshold}/28")
print(f"Rows where Table 7 and Table 11 hard-contradict: {n_internal_contradictions}/28")
print(f"Rows where Table 10 and Table 11 are inconsistent: {n_table10_11_disagree}/28")
print(f"Rows where metric is near threshold (|v-tau|/tau<0.2): {n_near_threshold}/28")
print()

# Recompute main.tex's 4/7, 5/7, 7/7, 2/7 counts under STANDARD thresholds
for attr in ['Race', 'Sex', 'Ethnicity', 'Age Group']:
    n_pass_std = sum(1 for (a, m), v in TABLE7.items() if a == attr and passes(v, m, TH_STANDARD))
    n_pass_nb = sum(1 for (a, m), v in TABLE7.items() if a == attr and passes(v, m, TH_NOTEBOOK))
    print(f"{attr}: under STANDARD={n_pass_std}/7, under NOTEBOOK={n_pass_nb}/7")
print()

# Totals
total_std = sum(passes(v, m, TH_STANDARD) for (a, m), v in TABLE7.items())
total_nb = sum(passes(v, m, TH_NOTEBOOK) for (a, m), v in TABLE7.items())
print(f"Total 'fair' verdicts under STANDARD thresholds: {total_std}/28")
print(f"Total 'fair' verdicts under NOTEBOOK thresholds: {total_nb}/28")
print(f"Delta: notebook inflates by {total_nb - total_std} verdicts")
print()
print(df.to_string(index=False))

# Write markdown summary
md = f"""# D0 Consistency Audit — main.tex Tables 7, 10, 11

**Model:** LGB-XGB Blend (AUROC 0.953, Acc 0.878) as reported in main.tex Sec. 6.

## Headline findings

1. **Main.tex's own Tables 7, 10, 11 ARE internally consistent with each other** under the standard thresholds stated in Table 7's caption (DI>=0.80; |SPD|/|EOPP|/|EOD|/|PP|/TI<0.10; CAL<0.05). All pass counts in Table 11 map correctly onto the VFR values in Table 10 (e.g., 16/30 pass -> min(16,14)/30 = 46.7% VFR for Race x EOPP).

2. **The NOTEBOOK's `FairnessCalculator.THRESHOLDS` uses non-standard values** (EOPP=0.20, EOD=0.20, CAL=0.10). Running the notebook against the same metric values produces **DIFFERENT verdict counts from main.tex**:

   | Attribute | main.tex (standard) | Notebook (0.20/0.10) | Delta |
   |---|---|---|---|
   | Race | 4/7 | 4/7 | 0 |
   | Sex | 5/7 | 5/7 | 0 |
   | Ethnicity | 7/7 | 7/7 | 0 |
   | Age Group | **2/7** | **5/7** | **+3** |
   | **Total** | **18/28** | **21/28** | **+3** |

   Under the notebook's current threshold code, Age Group EOPP (0.184) and EOD (0.184) pass (< 0.20), and Age Group CAL (0.065) passes (< 0.10). Under the standard thresholds in main.tex's caption, all three fail.

3. **Near-threshold cells identified as unstable in Table 10 VFR match near-threshold cells in Table 7**:
   - Race x EOPP (0.053 vs 0.10 threshold): VFR 46.7%. Point estimate passes on full test set (16/30 pass in bootstraps) — verdict is genuinely a coin toss.
   - Race x EOD (0.053 vs 0.10): VFR 33.3%. Point estimate passes, 10/30 fail.
   - Race x PP (0.095 vs 0.10): VFR 43.3%. Extremely near threshold; 17/30 pass, 13/30 fail.
   - Age x PP (0.070 vs 0.10): VFR 26.7%. 22/30 pass, 8/30 fail.
   - Eth x CAL (0.034 vs 0.05): VFR 13.3%. 26/30 pass.
   - Sex x CAL (0.026 vs 0.05): VFR 10.0%. 27/30 pass.
   - Eth x DI (0.830 vs 0.80): VFR 6.7%. 28/30 pass.

   This is **correct behaviour, not a threshold-inversion bug**: the point estimate on the full test set passes, but the verdict is unreliable because the value sits close to the threshold.

4. **No hard Table 7 <-> Table 11 contradictions** (no cells where Table 7 says Pass AND fewer than half the subsets pass, or vice versa).

## Reconciled definition (single consistent threshold set)

Use the thresholds in main.tex Table 7's caption for all cells in the notebook:

- DI: Pass if `DI >= 0.80`
- SPD, EOPP, EOD, PP, TI: Pass if `|value| < 0.10`
- CAL: Pass if `value < 0.05`

Use **absolute values** uniformly (notebook already does; main.tex does too via the `|.|` notation).

## Impact of applying this definition

- Notebook verdict counts: Age Group drops from 5/7 to **2/7**. Total drops from **21/28 to 18/28**.
- Intervention Section 10 "fair" count would shift similarly — the notebook's current threshold code inflates fair verdicts post-intervention as well.
- VFR values for EOPP and EOD on Race are unaffected (the 0.20 notebook threshold was already above 0.10; the mean is 0.053 so EOPP with threshold 0.20 almost always passes, but with 0.10 is threshold-adjacent and flips 46.7% of the time — this is the value main.tex reports).
- VFR computed under notebook's 0.20 threshold would be near 0% for all EOPP/EOD (value 0.053 is far below 0.20); main.tex's 46.7% number **requires the 0.10 threshold** to be reproduced.

This means: **main.tex's headline claim that 33.6% of verdicts flip depends on standard thresholds being used; it cannot be reproduced by running the notebook as-is.**

## Root cause

The `FairnessCalculator.THRESHOLDS` class-attribute in Cell 4 of the notebook was set to EOPP=0.20, EOD=0.20, CAL=0.10 with an internal justification comment referencing "Agarwal et al. (2018)". Main.tex's Section 4.2 and Table 7 caption both state the standard thresholds (0.10, 0.10, 0.05). Either the notebook was edited to the non-standard values after the manuscript's Section 4.2 was drafted, or the values in main.tex Tables 7/10/11 were computed by a separate script using standard thresholds and never folded back into the notebook.

## Action to resolve

1. Update `FairnessCalculator.THRESHOLDS` (Cell 4) to EOPP=0.10, EOD=0.10, CAL=0.05.
2. Re-run the notebook end-to-end so Tables 3, 6, 10, etc. in the notebook match main.tex.
3. If any author wishes to argue for 0.20 thresholds, that choice must be disclosed in Methods; main.tex currently does not do this.

See `results/consistency_audit.csv` for the row-level audit.
"""
with open('results/consistency_audit.md', 'w', encoding='utf-8') as f:
    f.write(md)
print('\nSaved results/consistency_audit.csv and results/consistency_audit.md')
