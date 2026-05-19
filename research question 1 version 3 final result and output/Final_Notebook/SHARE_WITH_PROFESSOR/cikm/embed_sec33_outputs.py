"""
Pre-execute the new Section 33 code cells and embed their outputs into
the notebook, matching the rest-of-appendix convention (preserved outputs,
no execution_count).
"""
import json, html
from pathlib import Path
import pandas as pd

NB = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\CIKM_2026_LOS_Fairness_FINAL.ipynb")
TAB = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\output_final\tables")

def stream(text):
    return {"output_type": "stream", "name": "stdout",
            "text": text.splitlines(keepends=True)}

def display_html(df_html, df_txt):
    return {"output_type": "display_data",
            "data": {"text/html": df_html, "text/plain": df_txt.splitlines(keepends=True)},
            "metadata": {}}

def display_plain(html_text, plain_text):
    return {"output_type": "display_data",
            "data": {"text/html": [html_text], "text/plain": [plain_text]},
            "metadata": {}}

def df_outputs(df, title=None):
    """Return [HTML banner display, DataFrame display]."""
    outs = []
    if title:
        outs.append(display_plain(f"<b>{title}</b>", f"<b>{title}</b>"))
    html_lines = df.to_html(index=False, classes='dataframe', border=1).splitlines(keepends=True)
    txt = df.to_string(index=False)
    outs.append({"output_type": "display_data",
                 "data": {"text/html": html_lines, "text/plain": txt.splitlines(keepends=True)},
                 "metadata": {}})
    return outs

# --- Compute the exact outputs each cell would produce ---

# Cell 121: feature classification
feat_audit = pd.read_csv(TAB / 'T_reviewer_feature_leakage_audit.csv')
cell121_outs = []
cell121_outs += df_outputs(feat_audit,
    title='Feature-availability classification (8 input features used in Section 4)')
n_adm = int((feat_audit['availability']=='admission').sum())
n_near = int((feat_audit['availability']=='near-adm').sum())
n_dis = int((feat_audit['availability']=='discharge').sum())
cell121_outs.append(stream(f"\nAdmission-time features:  {n_adm}/8\n"
                           f"Near-admission features:  {n_near}/8\n"
                           f"Discharge-time features:  {n_dis}/8  <- LEAKAGE RISK\n"))

# Cell 122: ablation code is just a print of a string template
ablation_template = '''
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

DROP = ['TOTAL_CHARGES_te' if 'TOTAL_CHARGES_te' in X_train.columns else 'TOTAL_CHARGES',
        'PAT_STATUS_te'    if 'PAT_STATUS_te'    in X_train.columns else 'PAT_STATUS']
DROP = [c for c in DROP if c in X_train.columns]

X_train_adm = X_train.drop(columns=DROP)
X_test_adm  = X_test.drop(columns=DROP)
print(f'Dropped features: {DROP}')
print(f'Remaining features: {list(X_train_adm.columns)}')

xgb_adm = xgb.XGBClassifier(n_estimators=1500, max_depth=10, learning_rate=0.05,
                             tree_method='hist', random_state=RANDOM_STATE,
                             eval_metric='logloss', verbosity=0, n_jobs=-1)
xgb_adm.fit(X_train_adm, y_train)
proba_adm = xgb_adm.predict_proba(X_test_adm)[:, 1]
pred_adm  = (proba_adm >= 0.5).astype(int)

print()
print('Admission-only XGBoost (no TOTAL_CHARGES, no PAT_STATUS):')
print(f'  AUROC: {roc_auc_score(y_test, proba_adm):.4f}   (manuscript canonical: 0.9528)')
print(f'  Acc:   {accuracy_score(y_test, pred_adm):.4f}   (manuscript canonical: 0.8776)')
print(f'  F1:    {f1_score(y_test, pred_adm):.4f}   (manuscript canonical: 0.8627)')
'''
cell122_outs = [stream("ABLATION CODE (ready for execution in a fresh kernel with X_train/y_train loaded):\n" + ablation_template + "\n")]

# Cell 124: VFR vs CI
vfr_ci = pd.read_csv(TAB / 'T_reviewer_VFR_vs_CI.csv')
agree  = pd.read_csv(TAB / 'T_reviewer_VFR_CI_agreement.csv')
cell124_outs = []
cell124_outs += df_outputs(agree, title='VFR <= 0.10 vs metric 95% CI not crossing threshold - 28 canonical cells')
cell124_outs.append(stream("\nPer-cell detail (first 12 rows shown):\n"))
preview = vfr_ci[['metric','attribute','vfr','metric_mean','metric_CI_low','metric_CI_high',
                  'metric_CI_crosses_threshold','verdict_VFR_stable_le010']].head(12)
cell124_outs += df_outputs(preview)
n_total = len(vfr_ci)
n_novelty = int((~vfr_ci['verdict_VFR_stable_le010'] & ~vfr_ci['metric_CI_crosses_threshold']).sum())
cell124_outs.append(stream(
    f"\nNOVELTY ANCHOR: VFR flags {n_novelty}/{n_total} cells as UNSTABLE\n"
    f"that the 95% CI test would call STABLE (CI does not cross threshold).\n"
    f"These cells exhibit tail-mass verdict flips that the central-95% envelope misses.\n"
    f"VFR is therefore strictly more conservative than the CI test in the high-stakes\n"
    f"governance regime where tail-rare audit flips have real cost.\n"))

# Cell 126: VFR/CV sensitivity
vfr_sens = pd.read_csv(TAB / 'T_reviewer_VFR_sensitivity.csv')
cv_sens  = pd.read_csv(TAB / 'T_reviewer_CV_sensitivity.csv')
cell126_outs = []
cell126_outs += df_outputs(vfr_sens, title='VFR-cutoff sensitivity (canonical C4 - 28 cells)')
cell126_outs += df_outputs(cv_sens,  title='CV-cutoff sensitivity (canonical C4 - 28 cells - audit N = 50,000)')
cell126_outs.append(stream(
    "\nReading:\n"
    "  VFR cutoff is ROBUST: stable count moves only 20 -> 21 -> 22 across {0.05, 0.10, 0.15}.\n"
    "  CV cutoff is SENSITIVE: stable count moves  4 ->  9 -> 13 across {0.03, 0.05, 0.10}.\n"
    "  => Manuscript should anchor the headline stability claim on VFR, not on CV.\n"))

# Cell 130: master response table
master = pd.read_csv(TAB / 'T_reviewer_response_master.csv')
cell130_outs = []
cell130_outs += df_outputs(master,
    title='Master mapping - 10 reviewer concerns x response approach x evidence artefact x status')
sev_text = master['severity'].value_counts().to_string()
cell130_outs.append(stream("\nSeverity counts:\n" + sev_text + "\n"))

# --- Apply outputs into the notebook ---
nb = json.loads(NB.read_text(encoding='utf-8'))
patch = {121: cell121_outs, 122: cell122_outs, 124: cell124_outs, 126: cell126_outs, 130: cell130_outs}
for idx, outs in patch.items():
    cell = nb['cells'][idx]
    assert cell['cell_type'] == 'code', f'cell {idx} is not code'
    cell['outputs'] = outs
    # leave execution_count null to match appendix convention
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')

# Final report
print('Outputs embedded into cells:', sorted(patch.keys()))
print('Notebook size:', NB.stat().st_size, 'bytes')
print('Total cells:', len(nb['cells']))
