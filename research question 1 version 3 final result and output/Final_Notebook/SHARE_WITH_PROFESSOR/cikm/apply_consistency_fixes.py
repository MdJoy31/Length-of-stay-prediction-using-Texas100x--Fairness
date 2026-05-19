"""
Apply 4 critical consistency fixes flagged by the end-to-end audit.

D1+D7: '17 of 336' → '77 of 336' in §35.1 (the CV>0.50 count vs VFR>0.10
       count were conflated). Also fix the §35.7 trace-table row to cite
       cikm_vfr_all_metrics.csv (not T13_axis1_vfr_config1.csv).
D2:    §34.5 limitation: hospital-disjoint failure is on DI Race, not DI
       Age. Remove the contradicted "age case-mix drift" causal claim.
D3:    4.24 vs 4.29 pp footnote. Keep 4.24 as the canonical headline (it
       matches the manuscript narrative and T_4model_before_after.csv),
       but add a clarifying note pointing to T15 = 4.29 as the
       reproduction figure.

All fixes are applied to BOTH notebooks (CIKM submission + journal).
"""
import json
from pathlib import Path

ROOT = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm")
CIKM = ROOT / "CIKM_2026_LOS_Fairness_FINAL.ipynb"
JNL  = ROOT / "full_journal_paper" / "Journal_LOS_Fairness_FULL.ipynb"

# Build canonical D1+D7 source text (the cikm_vfr_all_metrics.csv count)
# Audit-verified counts (read once via stdlib csv to avoid pandas memory issues)
import csv as csvmod
n_total = 0; n_nonzero = 0; n_practical = 0
with open(ROOT / "output_final" / "tables" / "cikm_vfr_all_metrics.csv", encoding='utf-8') as f:
    rdr = csvmod.DictReader(f)
    # find the column named vfr (case insensitive)
    vfr_col = None
    for c in rdr.fieldnames:
        if c.lower() == 'vfr': vfr_col = c; break
    if vfr_col is None:
        for c in rdr.fieldnames:
            if 'vfr' in c.lower(): vfr_col = c; break
    for row in rdr:
        try:
            v = float(row[vfr_col])
        except (TypeError, ValueError):
            continue
        n_total += 1
        if v > 0: n_nonzero += 1
        if v > 0.10: n_practical += 1
print(f"cikm_vfr_all_metrics.csv: total={n_total}, non-zero VFR={n_nonzero}, VFR>0.10={n_practical}")

for nb_path, label in [(CIKM, 'CIKM'), (JNL, 'Journal')]:
    nb = json.loads(nb_path.read_text(encoding='utf-8'))

    fixes_made = []
    for i, cell in enumerate(nb['cells']):
        src = cell['source']
        if isinstance(src, list): src = ''.join(src)
        original = src

        # D1: §35.1 17/336 -> 77/336 (the practical-significance count)
        if '17 of 336' in src and 'practically meaningful' in src:
            src = src.replace('17 of 336', f'{n_practical} of {n_total}')
            fixes_made.append(f'  cell {i}: D1 (17/336 -> {n_practical}/{n_total})')

        # D6+D7: §35.7 Fix 6 trace table — change CSV reference
        if '§35.1 (overclaim)' in src and 'T13_axis1_vfr_config1.csv' in src:
            src = src.replace(
                '`output_final/tables/T13_axis1_vfr_config1.csv` (146 = non-zero VFR rows × 12 models); §17 verification check `cv_gt_50_count_is_17`',
                f'`output_final/tables/cikm_vfr_all_metrics.csv` (336 rows = 12 models × 28 cells); '
                f'non-zero VFR count is {n_nonzero}/{n_total}, practically-significant count (VFR > 0.10) is {n_practical}/{n_total}')
            fixes_made.append(f'  cell {i}: D6/D7 (trace-table source CSV fixed)')

        # D2: §34.5 "age case-mix drift" -> "Race-axis case-mix"
        # The hospital-disjoint failure is DI Race=0.77; DI Age=0.89 passes.
        if 'Age axis exhibits genuine cross-site case-mix drift' in src:
            src = src.replace(
                'the age axis exhibits genuine cross-site case-mix drift that pure threshold-shifting cannot overcome',
                'the Race axis exhibits cross-site case-mix drift (DI Race drops to ≈ 0.77; the other three protected attributes still pass) that pure threshold-shifting did not fully resolve')
            fixes_made.append(f'  cell {i}: D2 (age->Race causal claim corrected)')
        if 'DI Race fell to ≈ 0.77 even after field calibration, driven by site-level age case-mix drift' in src:
            src = src.replace(
                'DI Race fell to ≈ 0.77 even after field calibration, driven by site-level age case-mix drift that pure threshold-shifting cannot resolve.',
                'DI Race fell to ≈ 0.77 even after field calibration; the other three DI values (Sex 0.98, Eth 0.92, Age 0.89) still passed the 4/5 rule. The failure is therefore Race-axis-specific cross-site drift, not the age case-mix gap previously suspected.')
            fixes_made.append(f'  cell {i}: D2b (corrected hospital-disjoint failure description)')

        # D3 footnote: add reconciliation note where 4.24 pp is cited prominently
        # (only patch §35.7 Fix 1 paragraph which is the most-cited place)
        if 'The CIKM submission\'s headline accuracy cost is **4.24 pp**' in src:
            if 'reproduction artifact' not in src:
                src = src.replace(
                    'The CIKM submission\'s headline accuracy cost is **4.24 pp**',
                    'The CIKM submission\'s headline accuracy cost is **4.24 pp** (per `T_4model_before_after.csv`; the rebuilt `T15_standard_vs_fair.csv` rounds to 4.29 pp due to a 0.05-pp Acc_after reproduction artefact — keep the 4.24 figure consistent with the manuscript narrative and add a one-line `T15 → 4.29 pp` footnote if the rebuilt T15 is included)')
                fixes_made.append(f'  cell {i}: D3 (4.24 vs 4.29 reconciliation footnote added)')

        if src != original:
            cell['source'] = src.splitlines(keepends=True)

    if fixes_made:
        nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
        print(f"\n{label}: {len(fixes_made)} fixes applied")
        for f in fixes_made: print(f)
    else:
        print(f"\n{label}: no fixes applied (patterns not found)")
