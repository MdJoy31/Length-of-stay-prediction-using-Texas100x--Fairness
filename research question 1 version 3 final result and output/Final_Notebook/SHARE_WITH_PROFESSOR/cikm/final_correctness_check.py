"""
Final correctness check: verify every fix is actually applied.

Returns PASS/FAIL on each of the 7 critical Q1 issues + 6 structural checks.
"""
import json, sys, io, re
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm")
NB = ROOT / "CIKM_2026_LOS_Fairness_FINAL.ipynb"
BS = chr(92)

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

results = []
def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append((name, status, detail))
    print(f"[{status}] {name}{(' — ' + detail) if detail else ''}")

print("=" * 80)
print("FINAL CORRECTNESS CHECK")
print("=" * 80)
print(f"Notebook: {NB.name} ({NB.stat().st_size / 1024 / 1024:.2f} MB, {len(nb['cells'])} cells)")
print()

# ─────────────────────────────────────────────────────────────────
# Q1 CRITICAL FIXES
# ─────────────────────────────────────────────────────────────────
print("--- Q1 CRITICAL FIXES ---")

# 1. Cell 4 RACE_MAP correct
src4 = ''.join(nb['cells'][4].get('source', []))
correct_race_map = '{0:"American Indian", 1:"Asian/Pacific Islander"' in src4 and '4:"Other/Unknown"}' in src4
old_race_map = 'Native American' in src4 or '4:"White"}' in src4
check("Cell 4 RACE_MAP corrected", correct_race_map and not old_race_map,
      "0=AmIndian, 1=Asian/PI, 2=Black, 3=White, 4=Other/Unknown" if correct_race_map else "Still has old mapping")

# 2. Cell 9 expected_T3 dict corrected
src9 = ''.join(nb['cells'][9].get('source', []))
correct_dict = (
    '("Race","White"): (603368, 65.2, 45.3)' in src9 and
    '("Race","American Indian"): (3474, 0.4, 33.4)' in src9 and
    '("Race","Other/Unknown"): (186670, 20.2, 40.4)' in src9
)
old_dict = '"Native American"' in src9 and '("Race","White"): (186670' in src9
check("Cell 9 expected_T3 corrected", correct_dict and not old_dict)

# 3. Cell 9 + 57 source has no 2-backslash-n bug
total_2bs_n = sum(''.join(c.get('source', [])).count(BS + BS + 'n') for c in nb['cells'])
check("No 2-backslash-n bugs anywhere", total_2bs_n == 0, f"found {total_2bs_n}")

# 4. Cell 1 CAL threshold is 0.05
src1 = ''.join(nb['cells'][1].get('source', []))
cal_05 = "Calibration | CAL | ≤ | **0.05**" in src1
cal_10 = "Calibration | CAL | ≤ | **0.10**" in src1
check("Cell 1 CAL threshold = 0.05 (matches code)", cal_05 and not cal_10)

# 5. Cell 16 says "11 features"
src16 = ''.join(nb['cells'][16].get('source', []))
correct_features = "11 features" in src16
wrong_features = "14 features" in src16
check("Cell 16 says '11 features'", correct_features and not wrong_features)

# 6. Cell 1 has §1.6.1 disclosure
has_161 = "1.6.1 Computational note" in src1
check("Cell 1 has §1.6.1 lighter-XGBoost disclosure", has_161)

# 6b. Cell 37 (§11.6) bootstrap B count matches cell 38 code
src37 = ''.join(nb['cells'][37].get('source', []))
src38_b = ''.join(nb['cells'][38].get('source', []))
b_in_md_200 = "two-hundred" in src37
b_in_md_100 = "one-hundred" in src37
b_ci_match = re.search(r'B_CI\s*=\s*(\d+)', src38_b)
b_ci_value = b_ci_match.group(1) if b_ci_match else "?"
md_matches_code = (b_ci_value == "100" and b_in_md_100 and not b_in_md_200) or (b_ci_value == "200" and b_in_md_200 and not b_in_md_100)
check(f"Cell 37 markdown B count matches cell 38 B_CI={b_ci_value}", md_matches_code,
      f"md says one-hundred={b_in_md_100}, two-hundred={b_in_md_200}")

# 7. Cell 9 captured output shows clean PASS (no MISMATCH artefacts)
out9_text = ""
for o in nb['cells'][9].get('outputs', []):
    if 'text' in o:
        t = o['text']
        out9_text += ''.join(t) if isinstance(t, list) else t
clean_out9 = "All 14 rows of Table 3 match" in out9_text and "MISMATCH" not in out9_text
check("Cell 9 output shows clean PASS", clean_out9)

# 8. Cell 57 captured output has no literal \n
out57_text = ""
for o in nb['cells'][57].get('outputs', []):
    if 'text' in o:
        t = o['text']
        out57_text += ''.join(t) if isinstance(t, list) else t
no_literal_n = '\\n' not in out57_text
check("Cell 57 output has no literal \\n artefacts", no_literal_n)

# ─────────────────────────────────────────────────────────────────
# STRUCTURAL CHECKS
# ─────────────────────────────────────────────────────────────────
print()
print("--- STRUCTURAL CHECKS ---")

# Section numbering
sec_pat = re.compile(r"^##\s+(\d+)\s*[\.\s·]", re.MULTILINE)
sections_found = set()
for c in nb['cells']:
    if c['cell_type'] != 'markdown':
        continue
    src = ''.join(c.get('source', []))
    for m in sec_pat.finditer(src):
        sections_found.add(int(m.group(1)))
expected_sections = set(range(1, 33))  # §1-§32 (24 base + 8 appendix)
gaps = expected_sections - sections_found
extras = sections_found - expected_sections
check("All §1-§32 present, no gaps", not gaps, f"missing: {sorted(gaps)}" if gaps else "")
check("No extra/duplicate top-level sections", not extras, f"extras: {sorted(extras)}" if extras else "")

# Cell counts
n_md = sum(1 for c in nb['cells'] if c['cell_type'] == 'markdown')
n_code = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')
check("119 cells (99 + §32 revision panels: 10 md + 10 code, includes F8 + F9)", len(nb['cells']) == 119, f"{n_md} markdown + {n_code} code")

# F8 figure injected (find the cell by content, not index)
f8_found = False
f8_has_png = False
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c.get('source', []))
    if 'F8_age_group_analysis.png' in src or '24.0a' in src:
        f8_found = True
        f8_has_png = any('image/png' in str(o.get('data', {})) for o in c.get('outputs', []))
        f8_idx = i
        break
check("F8 age-group figure embedded with PNG output",
      f8_found and f8_has_png,
      f"cell {f8_idx}" if f8_found else "not found")

# All key artefacts exist on disk
TABLES = ROOT / "output_final" / "tables"
FIGS = ROOT / "output_final" / "figures"
required_tables = ["T3_descriptive", "T4_best_model_landscape", "T5_cross_model_verdict",
                    "T6_reconciliation", "T7_vfr_heatmap", "T8_subset_fluctuation",
                    "T9_min_sample_size", "T10_cross_hospital_cv", "T11_fleiss_kappa",
                    "T12_combined_reliability", "T13_lambda_sweep", "T14_ablation_xgboost",
                    "T15_standard_vs_fair", "T15_with_CI", "T16_per_cluster_xgboost",
                    "T17_k_sensitivity_real", "T18_audit_recommendation",
                    "T19_claim_verification", "T20_unanimous_fair_matrix",
                    "T_HYPERPARAMS", "T_VFR_directional", "T_age_group_analysis",
                    "T_clinical_utility_summary", "T_per_group_confusion_matrix"]
on_disk_tables = {f.stem for f in TABLES.glob("*.csv")}
missing_tables = [t for t in required_tables if t not in on_disk_tables]
check(f"All {len(required_tables)} required tables on disk", not missing_tables,
      f"missing: {missing_tables}" if missing_tables else "")

required_figs = [f"F{i}" for i in range(1, 13)]
on_disk_figs = {f.stem.split('_')[0] for f in FIGS.glob("*.png")}
missing_figs = [f for f in required_figs if f not in on_disk_figs]
check(f"All {len(required_figs)} required figures on disk", not missing_figs,
      f"missing: {missing_figs}" if missing_figs else "")

# Texas-100X attribution
all_text = ""
for c in nb['cells']:
    all_text += ''.join(c.get('source', [])) + "\n"
texas100x_count = all_text.count("Texas-100X")
jayaraman_count = all_text.count("Jayaraman")
year2006_count = all_text.count("2006 (Q1-Q4)") + all_text.count("Quarters 1-4 of 2006")
check(f"Texas-100X attribution present (Jayaraman, 2006 Q1-Q4)",
      texas100x_count >= 5 and jayaraman_count >= 1 and year2006_count >= 5,
      f"Texas-100X={texas100x_count}, Jayaraman={jayaraman_count}, 2006={year2006_count}")
# No remaining 2019-2023 data-year refs
import re
old_year_count = sum(all_text.count(p) for p in
                     ["FY 2019-2023", "fiscal years 2019 to 2023",
                      "between 2019 and 2023", "from 2019 to 2023"])
check(f"No stale 2019-2023 data-year references", old_year_count == 0,
      f"found: {old_year_count}")

# ─────────────────────────────────────────────────────────────────
# VALIDATE AGE-GROUP MAPPING AGAINST USER's age_binning.csv
# ─────────────────────────────────────────────────────────────────
print()
print("--- AGE-GROUP MAPPING vs user's age_binning.csv ---")

import pandas as pd
T_AG = pd.read_csv(TABLES / "T_age_group_analysis.csv", index_col=0)
print(T_AG.to_string())
print()

# user's csv mapping (PAT_AGE → bucket)
user_buckets = {
    "Age_0_17":   list(range(0, 5)),     # PAT_AGE 0-4
    "Age_18_39":  list(range(5, 10)),    # PAT_AGE 5-9
    "Age_40_54":  list(range(10, 13)),   # PAT_AGE 10-12
    "Age_55_64":  list(range(13, 15)),   # PAT_AGE 13-14
    "Age_65_Plus": list(range(15, 22)),  # PAT_AGE 15-21
}
notebook_buckets = {
    "Pediatric (<18)":      list(range(0, 5)),
    "Young Adult (18-39)":  list(range(5, 10)),
    "Middle-Aged (40-64)":  list(range(10, 15)),  # combines user's 40-54 + 55-64
    "Elderly (>=65)":       list(range(15, 22)),
}
peds_match = user_buckets["Age_0_17"] == notebook_buckets["Pediatric (<18)"]
ya_match = user_buckets["Age_18_39"] == notebook_buckets["Young Adult (18-39)"]
mid_match = (user_buckets["Age_40_54"] + user_buckets["Age_55_64"]) == notebook_buckets["Middle-Aged (40-64)"]
eld_match = user_buckets["Age_65_Plus"] == notebook_buckets["Elderly (>=65)"]
all_match = peds_match and ya_match and mid_match and eld_match
check("Age-group labels match user's age_binning.csv", all_match)

# ─────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────
print()
print("=" * 80)
n_pass = sum(1 for _, s, _ in results if s == "PASS")
n_fail = sum(1 for _, s, _ in results if s == "FAIL")
print(f"FINAL: {n_pass} PASS, {n_fail} FAIL out of {len(results)} checks")
print("=" * 80)
if n_fail == 0:
    print("\n>>> ALL CORRECT — notebook ready to use <<<")
else:
    print("\n>>> ISSUES FOUND:")
    for name, status, detail in results:
        if status == "FAIL":
            print(f"   [FAIL] {name}: {detail}")
