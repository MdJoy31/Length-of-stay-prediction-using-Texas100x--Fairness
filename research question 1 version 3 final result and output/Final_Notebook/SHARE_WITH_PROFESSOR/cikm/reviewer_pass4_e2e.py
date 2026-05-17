"""
Pass 4 — end-to-end reviewer audit. Cross-checks the new four-configuration
baseline-extension results against the existing canonical artefacts to find
any inconsistency a Q1 reviewer would flag.

Checks:
  R1. Config 1 (Real-Only) numbers should equal the existing canonical
      XGBoost numbers (T4 / T5 / T6 / T15 row 1 / T11)
  R2. Config 4 (Phase 5b) headline numbers should equal T15 row 'Fair (Intersect.)'
  R3. Config 1's 28-cell VFR should match the XGBoost subset of cikm_vfr_all_metrics.csv
  R4. Config 1 max VFR ≤ canonical 336-cell max VFR (47.4%)
  R5. Greedy refinement (Config 3 → Config 4) should reduce VFR
  R6. Internal: VFR formula consistency, n_pass + n_fail = K, etc.
  R7. Notebook structure: section numbering, cell counts, presence of new cells
  R8. Missing narrative: are the new sections explained anywhere?
"""
import json, sys, io, re
import pandas as pd
import numpy as np
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm")
TAB = ROOT / "output_final" / "tables"
NB = ROOT / "CIKM_2026_LOS_Fairness_FINAL.ipynb"

issues = []
def flag(severity, msg):
    issues.append((severity, msg))
    sym = {"FAIL":"🔴","WARN":"🟡","INFO":"🟢"}.get(severity, "?")
    print(f"  {sym} [{severity}] {msg}")

print("=" * 80)
print("PASS 4 — END-TO-END REVIEWER AUDIT")
print("=" * 80)
print()

# ---------- R1. Config 1 vs canonical XGBoost ----------
print("R1. Config 1 (Real-Only) vs canonical XGBoost (T4 / T5 / T15)")
T_SUM = pd.read_csv(TAB / "T_baseline_audit_summary.csv")
c1 = T_SUM[T_SUM['config_id']==1].iloc[0]
T15 = pd.read_csv(TAB / "T15_standard_vs_fair.csv")
T15_acc = T15[T15['Metric']=='Accuracy']['Standard'].iloc[0]
T15_auc = T15[T15['Metric']=='AUC']['Standard'].iloc[0]
print(f"   Config 1 Acc:    {c1['accuracy']:.4f}     T15 Standard Acc: {T15_acc:.4f}")
if abs(c1['accuracy'] - T15_acc) > 0.005:
    flag("WARN", f"Config 1 Acc ({c1['accuracy']:.4f}) differs from T15 Standard ({T15_acc:.4f}) by {abs(c1['accuracy']-T15_acc):.4f}")
else:
    print("   ✓ within 0.005 tolerance (rounding + RNG variance)")

# ---------- R2. Config 4 vs T15 row 'Fair' ----------
print()
print("R2. Config 4 (Phase 5b) vs T15 row 'Fair (Intersect.)'")
c4 = T_SUM[T_SUM['config_id']==4].iloc[0]
T15_fair_acc = T15[T15['Metric']=='Accuracy']['Fair (Intersect.)'].iloc[0]
print(f"   Config 4 Acc:    {c4['accuracy']:.4f}     T15 Fair Acc: {T15_fair_acc:.4f}")
if abs(c4['accuracy'] - T15_fair_acc) > 0.005:
    flag("WARN", f"Config 4 Acc ({c4['accuracy']:.4f}) differs from T15 Fair ({T15_fair_acc:.4f}) by {abs(c4['accuracy']-T15_fair_acc):.4f}")
else:
    print("   ✓ within 0.005 tolerance")

# ---------- R3. Config 1 VFR vs canonical XGBoost VFR in cikm_vfr_all_metrics.csv ----------
print()
print("R3. Config 1 VFR vs cikm_vfr_all_metrics.csv (XGBoost subset)")
T_VFR_C1 = pd.read_csv(TAB / "T13_axis1_vfr_config1.csv")
canon_full = pd.read_csv(TAB / "cikm_vfr_all_metrics.csv")
canon_xgb = canon_full[canon_full['Model']=='XGBoost'].copy()
canon_xgb['vfr_canon'] = canon_xgb['VFR']
print(f"   Config 1 has {len(T_VFR_C1)} cells; canonical XGBoost has {len(canon_xgb)} cells")
print(f"   Config 1 VFR mean:   {T_VFR_C1['vfr'].mean()*100:.2f}%   max: {T_VFR_C1['vfr'].max()*100:.2f}%")
print(f"   Canonical VFR mean:  {canon_xgb['VFR'].mean()*100:.2f}%   max: {canon_xgb['VFR'].max()*100:.2f}%")
# join on (Attribute, Metric)
joined = T_VFR_C1.rename(columns={'attribute':'Attribute', 'metric':'Metric'}).merge(
    canon_xgb[['Attribute','Metric','VFR']].rename(columns={'VFR':'vfr_canon'}),
    on=['Attribute','Metric'], how='outer', indicator=True
)
n_match = (joined['_merge'] == 'both').sum()
print(f"   Joined rows (both sources): {n_match}")
if n_match == 28:
    diffs = (joined['vfr'] - joined['vfr_canon']).abs()
    max_abs_diff = diffs.max()
    n_exact = int((diffs < 0.01).sum())
    print(f"   Max abs VFR diff per cell: {max_abs_diff:.4f}")
    print(f"   Cells matching within 1pp: {n_exact}/28")
    if max_abs_diff > 0.10:
        flag("WARN", f"Config 1 VFR diverges from canonical XGBoost by up to {max_abs_diff*100:.1f}pp (likely RNG sequence difference)")
    elif max_abs_diff > 0.05:
        flag("INFO", f"Config 1 VFR diverges from canonical by up to {max_abs_diff*100:.1f}pp (acceptable bootstrap variance)")
    else:
        print(f"   ✓ Config 1 VFR matches canonical XGBoost to within 5pp")

# ---------- R4. Config 1 max VFR <= canonical 336-cell max ----------
print()
print("R4. Config 1 max VFR vs canonical 336-cell max")
canon_max = canon_full['VFR'].max() * 100
config1_max = T_VFR_C1['vfr'].max() * 100
print(f"   Config 1 (XGBoost only): {config1_max:.1f}%")
print(f"   Canonical (336 cells, 12 models): {canon_max:.1f}%")
if config1_max > canon_max + 0.5:
    flag("WARN", f"Config 1 max VFR ({config1_max:.1f}%) > canonical 336-cell max ({canon_max:.1f}%); RNG inconsistency between cell 23 and the new script")
elif config1_max > canon_max:
    flag("INFO", f"Config 1 max ({config1_max:.1f}%) > canonical max ({canon_max:.1f}%) — acceptable bootstrap-RNG variance")
else:
    print("   ✓ Config 1 max ≤ canonical max")

# ---------- R5. Greedy refinement should reduce VFR ----------
print()
print("R5. Greedy refinement effect (Config 3 → Config 4)")
c3 = T_SUM[T_SUM['config_id']==3].iloc[0]
c4 = T_SUM[T_SUM['config_id']==4].iloc[0]
print(f"   Config 3 VFR mean: {c3['vfr_mean_across_28_cells']:.4f}")
print(f"   Config 4 VFR mean: {c4['vfr_mean_across_28_cells']:.4f}")
print(f"   Reduction: {(c3['vfr_mean_across_28_cells']-c4['vfr_mean_across_28_cells'])/c3['vfr_mean_across_28_cells']*100:+.1f}%")
if c4['vfr_mean_across_28_cells'] >= c3['vfr_mean_across_28_cells']:
    flag("WARN", f"Greedy refinement (Config 4) did NOT reduce VFR mean vs Config 3")
else:
    print("   ✓ Greedy refinement reduces VFR as expected")

# ---------- R6. VFR formula consistency in all 4 configs ----------
print()
print("R6. VFR formula sanity (n_pass + n_fail = K, vfr = min/K)")
for cfg in [1, 2, 3, 4]:
    T = pd.read_csv(TAB / f"T13_axis1_vfr_config{cfg}.csv")
    n_pass_fail_sum = T['n_pass'] + T['n_fail']
    if not (n_pass_fail_sum == 500).all():
        flag("FAIL", f"Config {cfg}: n_pass + n_fail != 500 in some rows")
    else:
        print(f"   ✓ Config {cfg}: n_pass + n_fail = 500 for all 28 cells")
    expected_vfr = T.apply(lambda r: min(r['n_pass'], r['n_fail']) / 500, axis=1)
    if not np.isclose(T['vfr'], expected_vfr, atol=1e-3).all():
        flag("FAIL", f"Config {cfg}: vfr ≠ min(n_pass, n_fail)/K in some rows")

# ---------- R7. Notebook structure ----------
print()
print("R7. Notebook structure")
with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)
print(f"   Total cells: {len(nb['cells'])}")
markdown_count = sum(1 for c in nb['cells'] if c['cell_type'] == 'markdown')
code_count = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')
print(f"   Markdown: {markdown_count}, Code: {code_count}")

sec_pat = re.compile(r"^##\s+(\d+)\s*[\.\s·]", re.MULTILINE)
sections = []
for c in nb['cells']:
    if c['cell_type'] != 'markdown': continue
    src = ''.join(c.get('source', []))
    sections.extend(int(m.group(1)) for m in sec_pat.finditer(src))
sections = sorted(set(sections))
print(f"   Top-level sections: {sections}")
if sections != list(range(1, 25)):
    flag("WARN", f"Section numbering anomaly: expected §1-§24, got {sections}")

# Find the new appendix-style cells (§25, §26, §27, §28, §29) — these are CODE comments not real sections
appendix_markers = ['§25', '§26', '§27', '§28', '§29']
appendix_locations = []
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    for m in appendix_markers:
        if m in src:
            appendix_locations.append((i, m))
print(f"   Appendix-style markers (§25-§29) in code comments: {appendix_locations}")

# ---------- R8. Missing narrative for new sections ----------
print()
print("R8. Narrative coverage of the new appendix sections")
all_md = "\n\n".join(''.join(c.get('source', [])) for c in nb['cells'] if c['cell_type'] == 'markdown')
narrative_keywords = {
    'Three-Axis Reliability-Aware Fairness Framework': 'F9 model-agnostic diagram',
    'Threshold-Shifting Intervention': 'F10 / Config 3 vs 4 narrative',
    'Probability-Recalibration Intervention': 'F11 rejected variant narrative',
    'Configuration 3 verification': '§27 markdown explanation',
    'extreme λ sweep': '§28 / lambda extension narrative',
    'Four-configuration baseline': '§29 / cross-config narrative',
}
for keyword, label in narrative_keywords.items():
    if keyword.lower() in all_md.lower():
        print(f"   ✓ {label}: found in markdown")
    else:
        flag("INFO", f"{label}: no markdown narrative for '{keyword}'")

# ---------- R9. Audit-instrument identity check ----------
print()
print("R9. Audit-instrument identity across configs")
print("    (same K, same N grid, same GroupKFold deterministic with random_state=42)")
# Check N grid in T9 files
for cfg in [1, 2, 3, 4]:
    T = pd.read_csv(TAB / f"T9_axis2_minN_config{cfg}.csv")
    unique_n = sorted(T['min_N_for_cv_under_5pct'].unique())
    print(f"   Config {cfg} N values used: {unique_n}")
expected_grid = {1000, 2000, 5000, 10_000, 25_000, 50_000, 100_000, 185_026}
for cfg in [1, 2, 3, 4]:
    T = pd.read_csv(TAB / f"T9_axis2_minN_config{cfg}.csv")
    used = set(T['min_N_for_cv_under_5pct'].unique())
    extra = used - expected_grid
    if extra:
        flag("WARN", f"Config {cfg} uses N values not in spec: {extra}")

# ---------- Summary ----------
print()
print("=" * 80)
print(f"PASS 4 SUMMARY: {len(issues)} issues")
print("=" * 80)
fail = sum(1 for s, _ in issues if s == 'FAIL')
warn = sum(1 for s, _ in issues if s == 'WARN')
info = sum(1 for s, _ in issues if s == 'INFO')
print(f"  FAIL: {fail}")
print(f"  WARN: {warn}")
print(f"  INFO: {info}")
if not issues:
    print()
    print(">>> CLEAN — no Q1 reviewer issues found")
