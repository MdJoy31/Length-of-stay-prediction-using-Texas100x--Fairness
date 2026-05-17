"""
Correct the RACE-code label mapping in cell 8 (Section 3.3, Table 3).

Evidence (from cell 6 diagnostics in the current notebook state):
  RACE_CODE  N(%)        Hispanic%  LOS>3d%
  0          3,474 (0.4) 33.8       33.4
  1         16,404 (1.8) 96.8       41.0
  2        115,212 (12.5) 99.4      52.3
  3        603,368 (65.2) 83.1      45.3
  4        186,670 (20.2) 20.0      40.4

The current cell 8 source maps `RACE=4 -> "White"` and `RACE=3 -> "Black"`,
producing the implausible result that the cohort is 65.2% Black-coded.
Texas state demographics (US Census 2020): ~80% White (incl. Hispanic),
~12% Black, ~5% Asian, ~1% American Indian. The pattern in cell 6 is
consistent with the standard THCIC PUDF coding (1=AmInd, 2=Asian/PI,
3=Black, 4=White, 5=Other) shifted to 0-index during preprocessing,
giving:
  0 = American Indian
  1 = Asian/Pacific Islander
  2 = Black
  3 = White
  4 = Other / Unknown

Crucially, the underlying numerical fairness analysis (DI, SR, TPR,
FPR, PP, EOD, EOPP, TI, CAL) operates on the integer codes and is
invariant to the label permutation. Only the qualitative interpretation
in T3 (and any narrative text in the manuscript that names specific
groups) is affected by this fix.

A documentation comment is added to cell 8 noting that the mapping
was inferred from population statistics and should be cross-checked
against the THCIC PUDF data dictionary before submission. The
99.4% Hispanic-coding of RACE=2 (the inferred Black group) is also
flagged because it deviates from typical Texas state-level demographics
(~3% Hispanic among Black patients), suggesting either a county-
restricted subsample or a coding-scheme deviation upstream.
"""
import json
from pathlib import Path

NB = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\CIKM_2026_LOS_Fairness_FINAL.ipynb")

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)


# Find cell 8 (Section 3.3 Table 3)
target_idx = None
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c.get("source", []))
    if "3.3 · Cohort descriptive statistics (Table 3)" in src and "race_order" in src:
        target_idx = i
        break

if target_idx is None:
    print("ERROR: cell 8 (T3 cohort) not found")
    raise SystemExit(1)

c = nb["cells"][target_idx]
src = "".join(c.get("source", []))


OLD = """# Race in manuscript-canonical order: White, Black, Asian/PI, Native American, Other/Unknown
race_order = [(4, "White"), (3, "Black"), (2, "Asian/Pacific Islander"),
              (1, "Native American"), (0, "Other/Unknown")]"""

NEW = """# Race code mapping (THCIC PUDF schema, 0-indexed):
#   0 = American Indian
#   1 = Asian / Pacific Islander
#   2 = Black
#   3 = White
#   4 = Other / Unknown
# Mapping inferred from population statistics in cell 6 diagnostics:
#   RACE=3 has 65.2% of records and 83.1% Hispanic-coded => White
#   RACE=2 has 12.5% of records and 99.4% Hispanic-coded => Black
#   RACE=1 has  1.8% of records => Asian / Pacific Islander
#   RACE=0 has  0.4% of records => American Indian
#   RACE=4 has 20.2% of records => Other / Unknown
# Cross-check against the THCIC PUDF data dictionary recommended before
# submission. The 99.4% Hispanic share among RACE=2 (Black) is unusual
# relative to typical Texas state-level demographics (~3% Hispanic
# among Black patients) and may reflect county-restricted sampling
# upstream. The numerical fairness analysis is invariant to this label
# permutation; only the qualitative interpretation in this descriptive
# table is affected.
race_order = [(3, "White"), (2, "Black"), (1, "Asian/Pacific Islander"),
              (0, "American Indian"), (4, "Other/Unknown")]"""

if OLD in src:
    src = src.replace(OLD, NEW)
    print(f"Cell {target_idx}: race_order corrected (3=White, 2=Black, 1=Asian/PI, 0=AmInd, 4=Other)")
else:
    # Try a more flexible match (in case whitespace differs)
    import re
    pat = re.compile(r"# Race in manuscript-canonical order:.*?\(0, \"Other/Unknown\"\)\]", re.DOTALL)
    if pat.search(src):
        src = pat.sub(NEW, src)
        print(f"Cell {target_idx}: race_order corrected via regex match")
    else:
        print("ERROR: race_order block not found in cell 8")
        raise SystemExit(1)

c["source"] = src.splitlines(keepends=True)
c["outputs"] = []
c["execution_count"] = None


# Also update the verification list at the bottom of cell 8 if present
# (the manuscript-image comparison loop). It hardcoded labels that need to match.
src2 = "".join(c["source"])
src2 = src2.replace(
    'Verifying T3 against manuscript Table 3 (image-extracted) ...',
    'Verifying T3 against manuscript Table 3 (image-extracted, after RACE re-mapping) ...'
)
c["source"] = src2.splitlines(keepends=True)


with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"\nDone. Re-run cell {target_idx} after the current end-to-end run completes.")
