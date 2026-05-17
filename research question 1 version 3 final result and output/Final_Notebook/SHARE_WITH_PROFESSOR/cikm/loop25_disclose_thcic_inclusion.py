"""
Loop 25: disclose the THCIC_ID-inclusion deviation from the original
Texas-100X benchmark protocol.

The original Texas-100X README states "this feature is excluded from
model training". Our pipeline includes THCIC_ID as a target-encoded
feature (THCIC_ID_te) plus HOSP_VOLUME_LOG. This deviation should be
explicitly disclosed in §1.1 (Dataset) and §4 (Feature Engineering)
before submission.
"""
import json, sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

NB = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\CIKM_2026_LOS_Fairness_FINAL.ipynb")
with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

# ─────────────────────────────────────────────────────────────────
# Cell 1 §1.1: append a deviation note after the existing bullet list
# ─────────────────────────────────────────────────────────────────
src1 = ''.join(nb['cells'][1].get('source', []))
anchor = "**Intended-use disclaimer (from the Texas-100X repository):**"
disclosure_text = (
    "\n- **Deviation 1 from Texas-100X benchmark protocol  ·  task transformation:** the original benchmark predicts `PRINC_SURG_PROC_CODE` (100-class surgical procedure). We repurpose it as a feature (target-encoded, Section 4) and use a binary `LENGTH_OF_STAY > 3 days` flag as the label. Different task therefore admits different headline numbers (Texas-100X reports 46 % test accuracy on the 100-class task with a 50,000-record subset; we report AUROC = 0.9528 on the binary task with the full 925,128-record cohort).\n"
    "- **Deviation 2 from Texas-100X benchmark protocol  ·  hospital identifier inclusion:** the Texas-100X README states 'this feature is excluded from model training' for `THCIC_ID`. Our pipeline INCLUDES `THCIC_ID` as a Bayesian-smoothed target-encoded feature (`THCIC_ID_te`, m = 10) and additionally adds `HOSP_VOLUME_LOG = log1p(records-per-hospital)` (cell 11). This decision is justified for length-of-stay modelling because hospital-level case-mix and operational practices materially affect LOS and ignoring them would understate per-site fairness portability behaviour (Section 10). It is, however, a deviation from the original benchmark's privacy-leakage-analysis convention. Three consequences are explicitly disclosed: (i) part of the headline AUROC (0.9528) is attributable to per-hospital LOS-rate signal recovered through target encoding; (ii) for the canonical 80/20 split a hospital can appear in both partitions, so test-set encodings are fitted on training data from the SAME hospital; (iii) the K = 20 GroupKFold cross-site evaluation in Section 10 deliberately holds out entire hospitals, so encoded values for held-out hospitals fall back to the global mean and the cross-site portability numbers in T10/T11 are computed without hospital-identifier leakage."
)
new_disclosure = (
    "- **Deviation 1 from Texas-100X benchmark protocol  ·  task transformation:** the original benchmark predicts `PRINC_SURG_PROC_CODE` (100-class surgical procedure). We repurpose it as a feature (target-encoded, Section 4) and use a binary `LENGTH_OF_STAY > 3 days` flag as the label. Different task therefore admits different headline numbers (Texas-100X reports 46 % test accuracy on the 100-class task with a 50,000-record subset; we report AUROC = 0.9528 on the binary task with the full 925,128-record cohort).\n"
    "- **Deviation 2 from Texas-100X benchmark protocol  ·  hospital identifier inclusion:** the Texas-100X README states 'this feature is excluded from model training' for `THCIC_ID`. Our pipeline INCLUDES `THCIC_ID` as a Bayesian-smoothed target-encoded feature (`THCIC_ID_te`, m = 10) and additionally adds `HOSP_VOLUME_LOG = log1p(records-per-hospital)` (cell 11). This decision is justified for length-of-stay modelling because hospital-level case-mix and operational practices materially affect LOS and ignoring them would understate per-site fairness portability behaviour (Section 10). It is, however, a deviation from the original benchmark's privacy-leakage-analysis convention. Three consequences are explicitly disclosed: (i) part of the headline AUROC (0.9528) is attributable to per-hospital LOS-rate signal recovered through target encoding; (ii) for the canonical 80/20 split a hospital can appear in both partitions, so test-set encodings are fitted on training data from the SAME hospital; (iii) the K = 20 GroupKFold cross-site evaluation in Section 10 deliberately holds out entire hospitals, so encoded values for held-out hospitals fall back to the global mean and the cross-site portability numbers in T10/T11 are computed without hospital-identifier leakage.\n"
    "- **Intended-use disclaimer (from the Texas-100X repository):**"
)
if anchor in src1 and "Deviation 1 from Texas-100X benchmark protocol" not in src1:
    src1 = src1.replace(anchor, new_disclosure)
    nb['cells'][1]['source'] = src1.splitlines(keepends=True)
    print("Cell 1 §1.1: deviation disclosure inserted before intended-use disclaimer")
else:
    print(f"Cell 1: anchor not found OR disclosure already present (anchor in src: {anchor in src1})")

# ─────────────────────────────────────────────────────────────────
# Cell 11 (feature engineering code) - add a comment about the deviation
# ─────────────────────────────────────────────────────────────────
src11 = ''.join(nb['cells'][11].get('source', []))
old_comment = "# 4.1 · Feature engineering: Bayesian-smoothed target encoding (m=10)"
new_comment = (
    "# 4.1 · Feature engineering: Bayesian-smoothed target encoding (m=10)\n"
    "#\n"
    "# DEVIATION FROM TEXAS-100X PROTOCOL: the original benchmark excludes\n"
    "# THCIC_ID from model training (per the README). We INCLUDE it as a\n"
    "# target-encoded feature (THCIC_ID_te) and additionally add\n"
    "# HOSP_VOLUME_LOG. Justification: hospital-level case-mix materially\n"
    "# affects LOS; the K=20 GroupKFold protocol in Section 10 isolates the\n"
    "# leakage risk by holding out entire hospitals. See Section 1.1\n"
    "# 'Deviation 2' for the full disclosure."
)
if old_comment in src11 and "DEVIATION FROM TEXAS-100X PROTOCOL" not in src11:
    src11 = src11.replace(old_comment, new_comment)
    nb['cells'][11]['source'] = src11.splitlines(keepends=True)
    print("Cell 11: deviation comment added to feature-engineering code")

# Save
with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"\nFinal notebook: {NB.stat().st_size / 1024 / 1024:.2f} MB")

# Verify deviation disclosure now present
all_text = ""
for c in nb['cells']:
    all_text += ''.join(c.get('source', [])) + "\n"
print(f"\nDisclosure verification:")
print(f"  'Deviation 1 from Texas-100X' mentions: {all_text.count('Deviation 1 from Texas-100X')}")
print(f"  'Deviation 2 from Texas-100X' mentions: {all_text.count('Deviation 2 from Texas-100X')}")
print(f"  'DEVIATION FROM TEXAS-100X PROTOCOL' (code comment): {all_text.count('DEVIATION FROM TEXAS-100X PROTOCOL')}")
