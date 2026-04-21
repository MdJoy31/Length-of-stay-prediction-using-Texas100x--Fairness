"""D1 Fairness Reconciliation Table.

Produces two tables:
  D1a: Best model in the CURRENT notebook (XGBoost) — values and verdicts
       recomputed from the VFR bootstrap CSV under the corrected thresholds.
  D1b: Best model in main.tex (LGB-XGB Blend) — Table 7 / Table 10 / Table 11
       values reconciled with standard thresholds. The notebook does not yet
       contain a LGB-XGB Blend model (that is a Stage 2 change), so the std
       columns are borrowed from XGBoost as a first-order approximation.
"""
import pandas as pd, numpy as np, sys, os
from scipy.stats import norm
sys.stdout.reconfigure(encoding='utf-8')
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs('results', exist_ok=True)

# Corrected thresholds per main.tex Sec 4.2 / Table 7 caption
TAU = {'DI': 0.80, 'SPD': 0.10, 'EOPP': 0.10, 'EOD': 0.10, 'TI': 0.10, 'PP': 0.10, 'CAL': 0.05}
DIR = {'DI': 'above', 'SPD': 'below', 'EOPP': 'below', 'EOD': 'below',
       'TI': 'below', 'PP': 'below', 'CAL': 'below'}

def passes(val, m):
    return val >= TAU[m] if DIR[m] == 'above' else val <= TAU[m]

def margin_sigma(val, m, sigma):
    """Signed distance to threshold in sigmas. Positive = safely passing."""
    if sigma <= 0:
        return float('inf') if passes(val, m) else -float('inf')
    if DIR[m] == 'above':
        return (val - TAU[m]) / sigma
    return (TAU[m] - val) / sigma

def estimated_vfr_from_gaussian(mu, sigma, m):
    """Approximate VFR under Gaussian assumption with new thresholds."""
    if sigma <= 0:
        return 0.0
    if DIR[m] == 'above':
        p_pass = 1 - norm.cdf((TAU[m] - mu) / sigma)
    else:
        p_pass = norm.cdf((TAU[m] - mu) / sigma)
    return round(min(p_pass, 1 - p_pass) * 100, 1)

# ====== D1a: XGBoost (current notebook best model) ======
vfr_csv = pd.read_csv('output/tables/cikm_vfr_all_metrics.csv')
# Pick the best model by AUC-equivalent ranking — XGBoost per notebook output
# If 'XGBoost' exists in VFR CSV, use it; else fall back to the first model
xgb_df = vfr_csv[vfr_csv['Model'] == 'XGBoost'].copy()
if xgb_df.empty:
    xgb_df = vfr_csv[vfr_csv['Model'].str.contains('XGBoost|XGB', case=False, na=False)].copy()
print(f'[D1a] XGBoost rows from VFR CSV: {len(xgb_df)}')

ATTR_ORDER = ['RACE', 'SEX', 'ETHNICITY', 'AGE_GROUP']
METRIC_ORDER = ['DI', 'SPD', 'EOPP', 'EOD', 'TI', 'PP', 'CAL']

d1a_rows = []
for attr in ATTR_ORDER:
    for metric in METRIC_ORDER:
        r = xgb_df[(xgb_df['Attribute'] == attr) & (xgb_df['Metric'] == metric)]
        if r.empty:
            continue
        mu = float(r['Mean'].iloc[0])
        sigma = float(r['Std'].iloc[0])
        old_vfr = float(r['VFR'].iloc[0]) * 100 if r['VFR'].iloc[0] <= 1 else float(r['VFR'].iloc[0])
        # At the CORRECTED thresholds:
        ms = margin_sigma(mu, metric, sigma)
        new_vfr_est = estimated_vfr_from_gaussian(mu, sigma, metric)
        d1a_rows.append({
            'Attribute': attr,
            'Metric': metric,
            'Bootstrap mean (K=30)': round(mu, 4),
            'Bootstrap sigma': round(sigma, 4),
            'Threshold (corrected)': TAU[metric],
            'Pass (mean vs threshold)': 'Pass' if passes(mu, metric) else 'Fail',
            'Margin (sigma)': round(ms, 2),
            'Stability': ('Stable' if abs(ms) >= 2
                          else 'Marginal' if abs(ms) >= 1
                          else 'Unstable'),
            'VFR at old thresholds (from CSV, %)': round(old_vfr, 1),
            'VFR at corrected thresholds (Gaussian est., %)': new_vfr_est,
        })

d1a = pd.DataFrame(d1a_rows)
d1a.to_csv('results/fairness_reconciliation_XGBoost.csv', index=False)
print(d1a.to_string(index=False))

# ====== D1b: LGB-XGB Blend from main.tex (using exact Table 7/10/11 values) ======
TABLE7 = {
    ('Race', 'DI'): 0.652, ('Race', 'SPD'): 0.176, ('Race', 'EOPP'): 0.053,
    ('Race', 'EOD'): 0.053, ('Race', 'TI'): 0.005, ('Race', 'PP'): 0.095,
    ('Race', 'CAL'): 0.245,
    ('Sex', 'DI'): 0.761, ('Sex', 'SPD'): 0.124, ('Sex', 'EOPP'): 0.036,
    ('Sex', 'EOD'): 0.049, ('Sex', 'TI'): 0.005, ('Sex', 'PP'): 0.004,
    ('Sex', 'CAL'): 0.026,
    ('Ethnicity', 'DI'): 0.830, ('Ethnicity', 'SPD'): 0.078, ('Ethnicity', 'EOPP'): 0.019,
    ('Ethnicity', 'EOD'): 0.029, ('Ethnicity', 'TI'): 0.003, ('Ethnicity', 'PP'): 0.001,
    ('Ethnicity', 'CAL'): 0.034,
    ('Age Group', 'DI'): 0.291, ('Age Group', 'SPD'): 0.436, ('Age Group', 'EOPP'): 0.184,
    ('Age Group', 'EOD'): 0.184, ('Age Group', 'TI'): 0.014, ('Age Group', 'PP'): 0.070,
    ('Age Group', 'CAL'): 0.065,
}
TABLE11_PASS = {
    ('Race', 'DI'): 0, ('Race', 'SPD'): 0, ('Race', 'EOPP'): 16, ('Race', 'EOD'): 20,
    ('Race', 'TI'): 30, ('Race', 'PP'): 17, ('Race', 'CAL'): 0,
    ('Sex', 'DI'): 1, ('Sex', 'SPD'): 1, ('Sex', 'EOPP'): 30, ('Sex', 'EOD'): 30,
    ('Sex', 'TI'): 30, ('Sex', 'PP'): 30, ('Sex', 'CAL'): 27,
    ('Ethnicity', 'DI'): 28, ('Ethnicity', 'SPD'): 29, ('Ethnicity', 'EOPP'): 30,
    ('Ethnicity', 'EOD'): 30, ('Ethnicity', 'TI'): 30, ('Ethnicity', 'PP'): 30,
    ('Ethnicity', 'CAL'): 26,
    ('Age Group', 'DI'): 0, ('Age Group', 'SPD'): 0, ('Age Group', 'EOPP'): 0,
    ('Age Group', 'EOD'): 0, ('Age Group', 'TI'): 30, ('Age Group', 'PP'): 22,
    ('Age Group', 'CAL'): 0,
}
TABLE10_VFR = {
    ('Race', 'DI'): 0.0, ('Race', 'SPD'): 0.0, ('Race', 'EOPP'): 46.7, ('Race', 'EOD'): 33.3,
    ('Race', 'TI'): 0.0, ('Race', 'PP'): 43.3, ('Race', 'CAL'): 0.0,
    ('Sex', 'DI'): 3.3, ('Sex', 'SPD'): 3.3, ('Sex', 'EOPP'): 0.0, ('Sex', 'EOD'): 0.0,
    ('Sex', 'TI'): 0.0, ('Sex', 'PP'): 0.0, ('Sex', 'CAL'): 10.0,
    ('Ethnicity', 'DI'): 6.7, ('Ethnicity', 'SPD'): 3.3, ('Ethnicity', 'EOPP'): 0.0,
    ('Ethnicity', 'EOD'): 0.0, ('Ethnicity', 'TI'): 0.0, ('Ethnicity', 'PP'): 0.0,
    ('Ethnicity', 'CAL'): 13.3,
    ('Age Group', 'DI'): 0.0, ('Age Group', 'SPD'): 0.0, ('Age Group', 'EOPP'): 0.0,
    ('Age Group', 'EOD'): 0.0, ('Age Group', 'TI'): 0.0, ('Age Group', 'PP'): 26.7,
    ('Age Group', 'CAL'): 0.0,
}

# Infer sigma from VFR and pass count: for below-type, pass iff val<tau.
# If we observe x/K passes with mean mu and threshold tau, then
# approximately: the proportion below tau equals x/K, which = P(normal < tau).
# For a Gaussian with mean mu, sigma satisfies: tau = mu + sigma*Phi^{-1}(x/K).
# This gives us a point estimate of sigma, which we can cross-check against VFR.
d1b_rows = []
for attr_label, metric in TABLE7:
    val = TABLE7[(attr_label, metric)]
    tau = TAU[metric]
    pass_count = TABLE11_PASS[(attr_label, metric)]
    vfr = TABLE10_VFR[(attr_label, metric)]
    pass_rate = pass_count / 30
    # Invert: infer sigma from the empirical pass rate
    if DIR[metric] == 'above':
        # pass iff val>=tau; P(pass) = 1 - Phi((tau-mu)/sigma)
        p = pass_rate
        if 0 < p < 1:
            z = norm.ppf(1 - p)  # (tau - mu)/sigma
            sigma_inferred = (tau - val) / z if z != 0 else float('nan')
        else:
            sigma_inferred = float('nan')
    else:
        # pass iff val<=tau; P(pass) = Phi((tau-mu)/sigma)
        p = pass_rate
        if 0 < p < 1:
            z = norm.ppf(p)  # (tau - mu)/sigma
            sigma_inferred = (tau - val) / z if z != 0 else float('nan')
        else:
            sigma_inferred = float('nan')
    if not np.isfinite(sigma_inferred) or sigma_inferred <= 0:
        sigma_inferred = float('nan')
    # Margin
    if np.isfinite(sigma_inferred) and sigma_inferred > 0:
        ms = margin_sigma(val, metric, sigma_inferred)
    else:
        ms = float('inf') if passes(val, metric) else -float('inf')
    d1b_rows.append({
        'Attribute': attr_label,
        'Metric': metric,
        'Value (Table 7)': val,
        'Threshold': tau,
        'Pass?': 'Pass' if passes(val, metric) else 'Fail',
        'Bootstrap sigma (inferred from Table 11)': (round(sigma_inferred, 4)
                                                      if np.isfinite(sigma_inferred)
                                                      else 'NA (p=0 or 1)'),
        'Margin (sigma)': round(ms, 2) if np.isfinite(ms) else 'inf/-inf',
        'Stability': ('Stable' if np.isfinite(ms) and abs(ms) >= 2
                      else 'Marginal' if np.isfinite(ms) and abs(ms) >= 1
                      else 'Unstable' if np.isfinite(ms)
                      else 'Very stable (p=0 or 1)'),
        'Table 11 pass count (x/30)': pass_count,
        'Table 10 VFR (%)': vfr,
    })

d1b = pd.DataFrame(d1b_rows)
d1b.to_csv('results/fairness_reconciliation_LGB_XGB_Blend.csv', index=False)
print('\n[D1b] main.tex LGB-XGB Blend reconciliation:')
print(d1b.to_string(index=False))

# Aggregate counts
totals_xgb = sum(1 for r in d1a_rows if passes(r['Bootstrap mean (K=30)'], r['Metric']))
totals_blend = sum(1 for r in d1b_rows if r['Pass?'] == 'Pass')
print(f'\n[D1a] XGBoost total verdicts at corrected thresholds: {totals_xgb}/{len(d1a_rows)}')
print(f'[D1b] LGB-XGB Blend (main.tex) total verdicts at corrected thresholds: {totals_blend}/{len(d1b_rows)}')

# Markdown summary
md = f"""# D1 Fairness Reconciliation Table

## D1a — Current notebook best model (XGBoost)

{len(d1a_rows)}-row table from `output/tables/cikm_vfr_all_metrics.csv` (K=30 bootstrap of the XGBoost test set).

- Verdicts recomputed under **corrected thresholds** (EOPP=EOD=0.10, CAL=0.05).
- "Margin (sigma)" = distance from bootstrap mean to threshold in sigmas.
  Positive = safely passing; negative = safely failing; |margin| < 1 = unstable.
- "VFR at corrected thresholds" is a **Gaussian estimate**; exact recomputation
  requires the per-resample raw metric values (not in the CSV). For precise
  values the VFR cell (Cell 23/24) must be re-run after the threshold fix.

Total XGBoost verdicts at corrected thresholds: **{totals_xgb}/28**.

See `results/fairness_reconciliation_XGBoost.csv`.

## D1b — main.tex best model (LGB-XGB Blend)

28-row table using main.tex Table 7 point values, Table 11 pass counts, and
Table 10 VFR values. The bootstrap sigma is **inferred** from the empirical
pass rate (x/30) via the Gaussian inversion sigma = (tau - mu) / Phi^-1(p).
For cells where the pass rate is 0/30 or 30/30, sigma cannot be inferred
exactly — these cells are marked "Very stable".

Total LGB-XGB Blend verdicts at corrected thresholds: **{totals_blend}/28**.

See `results/fairness_reconciliation_LGB_XGB_Blend.csv`.

## Caveat

D1b is complete for the main.tex-reported numbers under the assumption that
they are internally consistent (verified in D0). The notebook itself does not
currently train a LGB-XGB Blend model — that is a Stage 2 change. Until the
notebook is re-run with the corrected model set, D1a (XGBoost) is the only
reconciliation that reflects the **actual notebook state**; D1b reflects the
**manuscript's reported state**.
"""
with open('results/fairness_reconciliation.md', 'w', encoding='utf-8') as f:
    f.write(md)
print('\nSaved:')
print('  results/fairness_reconciliation_XGBoost.csv')
print('  results/fairness_reconciliation_LGB_XGB_Blend.csv')
print('  results/fairness_reconciliation.md')
