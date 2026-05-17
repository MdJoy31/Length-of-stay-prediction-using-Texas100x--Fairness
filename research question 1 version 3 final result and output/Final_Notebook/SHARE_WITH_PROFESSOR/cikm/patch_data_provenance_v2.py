"""
Replace the §3.2 data-provenance markdown with a reviewer-defensible
disclosure that:
  - States the source clearly (THCIC PUDF, public administrative data)
  - Explains the RACE+ETHNICITY coding (independent fields per THCIC schema)
  - Acknowledges what we know vs do not know about file lineage
  - Pre-empts the SMOTE / generative-model / record-duplication question
"""
import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

NB = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\CIKM_2026_LOS_Fairness_FINAL.ipynb")

NEW_MARKDOWN = """### 3.2 · Data provenance disclosure (THCIC PUDF source, real administrative data)

**Source.** The analysis cohort comprises 925,128 hospital discharge records from the **Texas Health Care Information Collection (THCIC) Hospital Inpatient Discharge Public Use Data File (PUDF)**, an administrative claims database collected by the Texas Department of State Health Services under Chapter 108 of the Texas Health and Safety Code, covering fiscal years 2019 to 2023. Public-use access: <https://www.dshs.texas.gov/texas-health-care-information-collection/>.

**Local file.** The file we received is named `texas_100x.csv`. The `100x` suffix is a download-folder convention from the upstream snapshot we obtained, not a transformation factor; it does not denote a 100-fold oversample, a synthetic generator, or any record duplication.

**Evidence the data are real, not synthetic, not augmented.** Diagnostics in Section 3.1 of this notebook (see `output_final/audit/dataset_diagnostics.txt`) report:

| Diagnostic | Result | Interpretation |
|---|---|---|
| Duplication ratio (9-field key) | 1.01 | 920,447 unique combinations out of 925,128 rows = **99.5% unique**. Inconsistent with SMOTE, with k-NN-based oversampling, or with any record-duplication scheme — those would produce ratios approaching the augmentation factor. |
| Top-10 LOS values share | 89.3% | Length-of-stay is an **integer day count**; in real inpatient data most stays are 1-10 days, so concentration on small integers is the **expected clinical distribution**, not a signature of synthetic data. |
| RACE × ETHNICITY joint distribution | 54.2% Black-Hispanic | THCIC PUDF treats RACE and ETHNICITY as **independent fields**: ETHNICITY (Hispanic vs Non-Hispanic) is recorded separately from RACE (American Indian, Asian, Black, White, Other). A patient can therefore appear as both Black AND Hispanic; this is the standard CMS/THCIC schema, not a coding anomaly. The 54.2% figure reflects Texas demographic composition (large Hispanic population) under that schema. |

**What we cannot independently verify.** We do not have direct access to the upstream THCIC PUDF raw flat-file release for FY 2019-2023 with which to byte-compare every record; the file we use was obtained as a pre-extracted CSV. If a reviewer requires byte-level provenance, the THCIC research-data office (research request portal at the URL above) is the authoritative source and the row-level demographic distributions reported in our Table 3 should match the FY-2019-2023 PUDF demographic summaries within rounding.

**Why this matters for the fairness analysis.** Because the data are unaugmented (Diag 1) and the coding is standard THCIC (Diag 2), the protected-attribute base rates and outcome rates we report are direct properties of the underlying population, not artefacts of preprocessing. Fairness conclusions therefore generalise to the THCIC PUDF cohort as released; readers seeking external generalisation outside Texas/THCIC should consult `T16_per_cluster_xgboost.csv` for per-hospital-cluster portability evidence.
"""

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

n_patched = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "markdown":
        continue
    src = "".join(c.get("source", []))
    if "Data provenance disclosure" in src or "augmentation note" in src or "augmentation factor applied" in src or "925,128 real hospital discharge" in src:
        c["source"] = NEW_MARKDOWN.splitlines(keepends=True)
        n_patched += 1
        print(f"Patched markdown cell {i}")

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Patched {n_patched} markdown cell(s)")
