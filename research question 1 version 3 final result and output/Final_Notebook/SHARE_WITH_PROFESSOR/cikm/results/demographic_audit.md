# Demographic Audit Report

**Dataset:** `texas_100x.csv`, N=925,128, Hospitals=441

## Finding 1: SEX and ETHNICITY label assignments are CONSISTENT with main.tex

Main.tex Table 1 LOS rates match raw data:

| Subgroup | main.tex % | Actual % | Match |
|---|---|---|---|
| Male (SEX_CODE=1) | 41.1 | 41.1 | Yes |
| Female (SEX_CODE=0) | 51.8 | 51.8 | Yes |
| Hispanic (ETH=1) | 47.1 | 47.1 | Yes |
| Non-Hispanic (ETH=0) | 39.7 | 39.7 | Yes |

SEX_CODE 0->Female, 1->Male. ETHNICITY 0->Non-Hispanic, 1->Hispanic.

## Finding 2: RACE labels produce DEMOGRAPHICALLY IMPOSSIBLE overlap

| Race Code | Claim (main.tex) | N | % dataset | LOS rate | % Hispanic |
|---|---|---|---|---|---|
| 0 | Other/Unknown | 3,474 | 0.4 | 33.4% | 33.8% |
| 1 | Native American | 16,404 | 1.8 | 41.0% | **96.8%** |
| 2 | Asian/Pacific Islander | 115,212 | 12.5 | 52.3% | **99.4%** |
| 3 | Black | 603,368 | 65.2 | 45.3% | **83.1%** |
| 4 | White | 186,670 | 20.2 | 40.4% | 20.0% |

**If main.tex labels are correct:**
- 99.4% of Asian/PI patients would be Hispanic (vs <2% nationally)
- 83.1% of Black patients would be Hispanic (vs <5% nationally)
- 54% of ALL patients would be simultaneously Black AND Hispanic (vs ~1% nationally)

These are **demographically impossible** under any real Texas population.

## Finding 3: LOS rates MATCH main.tex proportion-for-proportion

Every LOS rate in main.tex Table 1 matches the raw data within 0.1%. The proportions and rates are correctly transcribed. The problem is ONLY the label assignment for RACE.

## Finding 4: Age groups — 5 groups vs 4 groups (undisclosed collapse)

The notebook's `create_age_groups` function produces **5 age groups** but main.tex Table 1 reports **4 groups**. The collapse (Age_40_54 + Age_55_64 -> Middle-Aged) is silent. All downstream fairness analyses run on 5 groups, not 4.

## Root Cause

Three possibilities (cannot be ruled out without the THCIC data dictionary):

1. **`texas_100x` is synthetic/augmented** (the `_100x` suffix suggests 100x oversampling) and RACE labels were permuted.
2. **Dataset uses a non-standard RACE code order.**
3. **THCIC double-codes Hispanic patients under RACE field** (known data quality issue). Under this reading:
   - RACE 4 = White non-Hispanic
   - RACE 3 (65%) = Black OR Hispanic-coded-Black (mostly Hispanic)
   - RACE 2 (12.5%) = Asian/PI OR Hispanic-coded-Asian (almost all Hispanic)
   - RACE 1 (1.8%) = NA OR Hispanic-coded-NA (almost all Hispanic)
   - RACE 0 (0.4%) = Other/Unknown

## Recommended Actions

1. Obtain the THCIC PUDF data dictionary for fiscal years 2019–2023.
2. If dataset is synthetic, disclose in Methods and rename race categories as anonymous (Group A–E).
3. If real, add a Methods paragraph explaining the RACE x ETHNICITY overlap, or re-label.
4. Add a `race_map` dictionary to the notebook to make label assumptions explicit.
5. Disclose the 5->4 age-group collapse in Methods.
