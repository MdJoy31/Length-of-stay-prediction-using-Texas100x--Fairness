"""
Loop 23: finish replacing the §1.1 Dataset block in cell 1 (the previous
loop missed it due to whitespace differences). Use a regex-based match.
"""
import json, sys, io, re
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

NB = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\CIKM_2026_LOS_Fairness_FINAL.ipynb")

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

src = ''.join(nb['cells'][1].get('source', []))

# Match the existing §1.1 Dataset block flexibly (whitespace-tolerant)
old_pat = re.compile(
    r"### 1\.1 Dataset\s*"
    r"- \*\*Source:\*\* Texas Inpatient Public Use Data File \(PUDF\), fiscal year 2006 \(Quarters 1-4\)[^\n]*\n"
    r"- \*\*Volume:\*\* 925,128 inpatient discharge records[^\n]*\n"
    r"- \*\*Target variable:\*\* binary classification[^\n]*\n"
    r"- \*\*Provenance:\*\* publicly available via <https://www\.dshs\.texas\.gov/texas-health-care-information-collection/>;\s*SHA-256 hash recorded in the audit log\.",
    re.DOTALL
)

new_block = (
    "### 1.1 Dataset\n"
    "- **Source:** Texas-100X benchmark dataset (Jayaraman, <https://github.com/bargavj/Texas-100X>), built from the Texas Hospital Inpatient Discharge Public Use Data File (PUDF) for **Quarters 1-4 of 2006**, released by the Texas Department of State Health Services (DSHS) under Chapter 108 of the Texas Health and Safety Code.\n"
    "- **Volume:** 925,128 inpatient discharge records across 441 hospitals (THCIC_ID); the duplication ratio on a nine-field key is 1.01, indicating a real (not synthetically augmented) cohort.\n"
    "- **Original benchmark task:** 100-class prediction of `PRINC_SURG_PROC_CODE` (principal surgical procedure). The Texas-100X README reports a 2-layer MLP baseline of 46% test accuracy on a 50,000-record subset.\n"
    "- **Task in this study (repurposed):** binary classification, length of stay greater than three days (positive-class rate 45.0% at the cohort level). `PRINC_SURG_PROC_CODE` is repurposed as a target-encoded feature (Section 4) rather than the prediction label.\n"
    "- **Provenance:** Texas-100X is a publicly available academic benchmark; the underlying THCIC PUDF is accessible via <https://www.dshs.texas.gov/texas-health-care-information-collection/>. SHA-256 hash of the local CSV recorded in `output_final/audit/data_hash.txt`.\n"
    "- **Intended-use disclaimer (from the Texas-100X repository):** the dataset is intended for academic machine-learning benchmarking and privacy-leakage analysis. Records are anonymised and only a numeric Hospital ID is retained for sampling and demographic visualisation. No demographic conclusions or decisions that could harm the population should be drawn from the data."
)

m = old_pat.search(src)
if m:
    src_new = src[:m.start()] + new_block + src[m.end():]
    nb['cells'][1]['source'] = src_new.splitlines(keepends=True)
    print(f"Cell 1 §1.1 Dataset block replaced ({m.end() - m.start()} chars -> {len(new_block)} chars)")
else:
    print("Pattern not found — printing first 800 chars of cell 1 for debugging")
    print(src[:800])

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"\nFinal notebook: {NB.stat().st_size / 1024 / 1024:.2f} MB")
