"""
Loop 22: correct dataset year and attribution.

The data is NOT from FY 2019-2023. It is the Texas-100X benchmark
dataset (Bargav Jayaraman, https://github.com/bargavj/Texas-100X) which
processes the Texas Hospital Inpatient Discharge Public Use Data File
for Quarters 1-4 of 2006.

Critical updates:
  1. All "FY 2019-2023" / "fiscal years 2019 to 2023" / "between 2019
     and 2023" -> "FY 2006 (Quarters 1-4)" or equivalent.
  2. Add Texas-100X benchmark attribution (Jayaraman, GitHub link).
  3. Update demographic-baseline comparison to a 2006-appropriate
     reference (US Census 2000 + 2005-2009 ACS estimates, NOT Census 2020).
  4. Disclose ML-task transformation: PRINC_SURG_PROC_CODE was the
     original label in Texas-100X (100-class surgical procedure
     prediction); we repurpose it as a feature and use LENGTH_OF_STAY
     > 3 days as the binary label.
  5. Acknowledge data-set intended use (academic benchmark / privacy
     leakage analysis) and disclosure clause from the source repo.

Patterns that look like CITATION years (e.g., "et al. (2019)", "Daghistani 2019")
in cell 65 (§21 lit comparison) are LEFT UNTOUCHED. We only replace
data-year references.
"""
import json, sys, io, re
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

NB = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\CIKM_2026_LOS_Fairness_FINAL.ipynb")

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

# ─────────────────────────────────────────────────────────────────
# Year-range replacements
# These are SAFE because they don't match citation years.
# ─────────────────────────────────────────────────────────────────
YEAR_REPLACEMENTS = [
    ("fiscal years 2019 to 2023", "fiscal year 2006 (Quarters 1-4)"),
    ("FY 2019 to 2023", "FY 2006 (Q1-Q4)"),
    ("FY 2019-2023", "FY 2006 (Q1-Q4)"),
    ("between 2019 and 2023", "in 2006"),
    ("from 2019 to 2023", "from 2006"),
    ("FY-2019-2023", "FY 2006"),
    ("the 2019-2023 release", "the 2006 release"),
    ("(2019-2023)", "(2006)"),
    ("FY-2019 to FY-2023", "FY 2006"),
    ("Texas state demographics in 2006-2020", "Texas state demographics around 2006"),
    # Census reference: 2020 -> 2000 + 2005-2009 ACS (closer to 2006 data year)
    ("US Census 2020", "US Census 2000 (decennial), supplemented by 2005-2009 American Community Survey 5-year estimates"),
    ("[US Census 2020]", "[US Census 2000 / 2005-2009 ACS]"),
    ("2020 US Census Decennial Redistricting Data Table P2", "2000 US Census Summary File 1 Table P4 ('Hispanic or Latino, and Not Hispanic or Latino by Race')"),
    ("2020 Decennial Census Redistricting Data, Table P2", "2000 Decennial Census Summary File 1, Table P4"),
    ("data.census.gov/table/DECENNIALPL2020.P2", "www.census.gov/programs-surveys/decennial-census/decade/2000/decade-2000.html"),
    # Hospital baseline year corrections
    ("the FY 2019-2023 release", "the FY 2006 release"),
    ("FY 2019-2023 PUDF", "FY 2006 PUDF"),
    ("the standard FY 2019-2023 PUDF", "the standard FY 2006 PUDF (as packaged in the Texas-100X benchmark)"),
    ("FY 2019-2023 raw flat-file release", "FY 2006 raw flat-file release"),
    # Generic "2019-2023" -> "2006" when adjacent to data-year context
    ("2019-2023", "2006 (Q1-Q4)"),  # safe last-pass; citation years use parens not hyphen
]

CITATION_PROTECT = re.compile(r"\b(et al\.?\s*\(20\d{2}\)|\(20\d{2}\))")  # protect citation parens like (2019)

n_replacements = 0
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'markdown' and c['cell_type'] != 'code':
        continue
    src = ''.join(c.get('source', []))
    new_src = src
    for old, new in YEAR_REPLACEMENTS:
        # If 'old' contains a year that's also in a citation context, this replacement
        # is unsafe. The replacements above are designed to avoid that.
        if old in new_src:
            count = new_src.count(old)
            new_src = new_src.replace(old, new)
            n_replacements += count
            if count:
                print(f"  Cell {i}: '{old[:40]}' -> '{new[:40]}' x{count}")
    if new_src != src:
        nb['cells'][i]['source'] = new_src.splitlines(keepends=True)

print(f"\nTotal year/baseline replacements: {n_replacements}")

# ─────────────────────────────────────────────────────────────────
# Cell 1 (§1.1 Dataset): rewrite the dataset description
# ─────────────────────────────────────────────────────────────────
src1 = ''.join(nb['cells'][1].get('source', []))
old_dataset_block = (
    "### 1.1 Dataset\n"
    "- **Source:** Texas Inpatient Public Use Data File (PUDF), fiscal year 2006 (Quarters 1-4), provided by the Texas Health Care Information Collection (THCIC) under Chapter 108 of the Texas Health and Safety Code.\n"
    "- **Volume:** 925,128 inpatient discharge records across 441 hospitals; the duplication ratio on a nine-field key is 1.01, indicating a real (not augmented) cohort.\n"
    "- **Target variable:** binary classification, length of stay greater than three days (positive-class rate 45.0% at the cohort level).\n"
    "- **Provenance:** publicly available via <https://www.dshs.texas.gov/texas-health-care-information-collection/>; SHA-256 hash recorded in the audit log."
)
new_dataset_block = (
    "### 1.1 Dataset\n"
    "- **Source:** Texas-100X benchmark dataset (Jayaraman, <https://github.com/bargavj/Texas-100X>), built from the Texas Hospital Inpatient Discharge Public Use Data File (PUDF) for Quarters 1-4 of 2006, released by the Texas Department of State Health Services (DSHS) under Chapter 108 of the Texas Health and Safety Code.\n"
    "- **Volume:** 925,128 inpatient discharge records across 441 hospitals (THCIC_ID); the duplication ratio on a nine-field key is 1.01, indicating a real (not synthetically augmented) cohort.\n"
    "- **Original benchmark task:** 100-class prediction of `PRINC_SURG_PROC_CODE` (principal surgical procedure). The Texas-100X README reports a 2-layer MLP baseline of 46% test accuracy on a 50,000-record subset.\n"
    "- **Task in this study (repurposed):** binary classification, length of stay greater than three days (positive-class rate 45.0% at the cohort level). `PRINC_SURG_PROC_CODE` is repurposed as a target-encoded feature (Section 4) rather than the prediction label.\n"
    "- **Provenance:** Texas-100X is a publicly available academic benchmark; the underlying THCIC PUDF is accessible via <https://www.dshs.texas.gov/texas-health-care-information-collection/>. SHA-256 hash of the local CSV recorded in `output_final/audit/data_hash.txt`.\n"
    "- **Intended use disclaimer (from the Texas-100X repository):** the dataset is intended for academic machine-learning benchmarking and privacy-leakage analysis. Records are anonymised and only a numeric Hospital ID is retained for sampling and demographic visualisation. No demographic conclusions or decisions that could harm the population should be drawn from the data."
)
if old_dataset_block in src1:
    src1 = src1.replace(old_dataset_block, new_dataset_block)
    nb['cells'][1]['source'] = src1.splitlines(keepends=True)
    print("Cell 1 §1.1 Dataset block rewritten with Texas-100X attribution")
else:
    print(f"Cell 1 §1.1 expected block not found verbatim - manual check required")

# ─────────────────────────────────────────────────────────────────
# Cell 8 (§3.2 data provenance): rewrite the source paragraph
# ─────────────────────────────────────────────────────────────────
src8 = ''.join(nb['cells'][8].get('source', []))
old_source_para = "**Source.** The analysis cohort comprises 925,128 hospital discharge records from the **Texas Health Care Information Collection (THCIC) Hospital Inpatient Discharge Public Use Data File (PUDF)**, an administrative claims database collected by the Texas Department of State Health Services under Chapter 108 of the Texas Health and Safety Code, covering fiscal year 2006 (Quarters 1-4). Public-use access: <https://www.dshs.texas.gov/texas-health-care-information-collection/>."
new_source_para = "**Source.** The analysis cohort comprises 925,128 hospital discharge records obtained via the **Texas-100X academic benchmark** (Jayaraman, <https://github.com/bargavj/Texas-100X>), which packages the **Texas Hospital Inpatient Discharge Public Use Data File (PUDF)** for **Quarters 1-4 of 2006**, an administrative claims database collected by the Texas Department of State Health Services (DSHS) under Chapter 108 of the Texas Health and Safety Code. The original raw PUDF release is accessible via <https://www.dshs.texas.gov/texas-health-care-information-collection/>. Texas-100X distributes a pre-extracted CSV (`texas_100x.csv`) with 925,128 rows, 12 columns, hospital identifier, and 100 surgical procedure codes; we repurpose the surgical-procedure column as a feature (target-encoded, Section 4) and use a binary `LENGTH_OF_STAY > 3 days` flag as the prediction label."
if old_source_para in src8:
    src8 = src8.replace(old_source_para, new_source_para)
    nb['cells'][8]['source'] = src8.splitlines(keepends=True)
    print("Cell 8 §3.2 Source paragraph rewritten")
else:
    print("Cell 8 §3.2 Source paragraph not found verbatim - skipping")

# Find and replace the "What we cannot independently verify" paragraph
old_cannot = "**What we cannot independently verify.** We do not have direct access to the upstream THCIC PUDF raw flat-file release for FY 2006 (Q1-Q4) with which to byte-compare every record; the file we use was obtained as a pre-extracted CSV. If a reviewer requires byte-level provenance, the THCIC research-data office (research request portal at the URL above) is the authoritative source and the row-level demographic distributions reported in our Table 3 should match the FY 2006 (Q1-Q4) PUDF demographic summaries within rounding."
new_cannot = "**What we cannot independently verify.** The pre-extracted CSV `texas_100x.csv` we consume is produced by the Texas-100X processing pipeline (`process_texas.py` in the source repository) from the four 2006 quarterly raw flat-files; we have not byte-compared our local copy against a fresh extraction. If a reviewer requires byte-level provenance, the upstream raw PUDF (Quarters 1-4 of 2006) is available from the THCIC research-data office (research-request portal at the URL above) and the row-level demographic distributions reported in Table 3 should match the FY 2006 PUDF demographic summaries within rounding."
if old_cannot in src8:
    src8 = src8.replace(old_cannot, new_cannot)
    nb['cells'][8]['source'] = src8.splitlines(keepends=True)
    print("Cell 8 §3.2 'What we cannot verify' paragraph rewritten")

# ─────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────
with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

# Verify no remaining 2019-2023 data-year references
final_src = ""
for c in nb['cells']:
    final_src += ''.join(c.get('source', [])) + "\n"
remaining = []
for pat in ["FY 2019-2023", "fiscal years 2019 to 2023", "between 2019 and 2023",
            "from 2019 to 2023", "FY-2019-2023", "2019-2023"]:
    if pat in final_src:
        remaining.append(f"{pat} ({final_src.count(pat)})")
if remaining:
    print(f"\nWARNING: remaining year refs: {remaining}")
else:
    print(f"\nNo remaining year references — all data-year mentions cleaned.")

print(f"\nFinal notebook: {NB.stat().st_size / 1024 / 1024:.2f} MB, {len(nb['cells'])} cells")
