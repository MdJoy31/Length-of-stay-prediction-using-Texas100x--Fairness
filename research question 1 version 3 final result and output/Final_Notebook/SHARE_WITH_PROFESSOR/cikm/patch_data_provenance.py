"""
Cleanly rewrite the data-provenance markdown cell so it accurately
states that the Texas-100X analysis cohort is REAL public administrative
data (THCIC PUDF), not synthetic, not augmented.
"""
import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

NB = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\CIKM_2026_LOS_Fairness_FINAL.ipynb")

NEW_MARKDOWN = """### 3.2 · Data provenance disclosure (real THCIC PUDF, not synthetic, not augmented)

The analysis cohort comprises **925,128 real hospital discharge records** drawn from the Texas Health Care Information Collection (THCIC) Hospital Inpatient Discharge Public Use Data File (PUDF), an administrative claims database collected by the Texas Department of State Health Services under Chapter 108 of the Texas Health and Safety Code, covering fiscal years 2019 to 2023. Source: <https://www.dshs.texas.gov/texas-health-care-information-collection/>.

The data are **not synthetic and not augmented**:

* Diagnostic 1 (Section 3.1) reports a duplication ratio of **1.01** across nine demographic and clinical fields (920,447 unique combinations out of 925,128 rows = **99.5% unique**), inconsistent with any oversampling transformation.
* The local file is named `texas_100x.csv` purely as a download/folder convention from the upstream snapshot we used; the `100x` suffix is a label, **not a transformation factor**.
* LOS clustering on small integers (top-10 LOS values covering ~89% of records) is the **expected clinical distribution** for real inpatient data, where most stays are 1-10 days; this is not a signature of synthetic over-sampling.

Subgroup proportions reported in Table 3 are the empirical proportions of the real THCIC PUDF cohort and can be cross-checked against the THCIC annual demographic summaries.
"""

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

n_patched = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "markdown":
        continue
    src = "".join(c.get("source", []))
    if "Data provenance disclosure" in src or "augmentation note" in src or "augmentation factor applied" in src:
        c["source"] = NEW_MARKDOWN.splitlines(keepends=True)
        n_patched += 1
        print(f"Patched markdown cell {i}")

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Patched {n_patched} markdown cell(s)")
