# D0 Consistency Audit — main.tex Tables 7, 10, 11

**Model:** LGB-XGB Blend (AUROC 0.953, Acc 0.878) as reported in main.tex Sec. 6.

## Headline findings

1. **Main.tex's own Tables 7, 10, 11 ARE internally consistent with each other** under the standard thresholds stated in Table 7's caption (DI>=0.80; |SPD|/|EOPP|/|EOD|/|PP|/TI<0.10; CAL<0.05). All pass counts in Table 11 map correctly onto the VFR values in Table 10 (e.g., 16/30 pass -> min(16,14)/30 = 46.7% VFR for Race x EOPP).

2. **The NOTEBOOK's `FairnessCalculator.THRESHOLDS` uses non-standard values** (EOPP=0.20, EOD=0.20, CAL=0.10). Running the notebook against the same metric values produces **DIFFERENT verdict counts from main.tex**:

   | Attribute | main.tex (standard) | Notebook (0.20/0.10) | Delta |
   |---|---|---|---|
   | Race | 4/7 | 4/7 | 0 |
   | Sex | 5/7 | 5/7 | 0 |
   | Ethnicity | 7/7 | 7/7 | 0 |
   | Age Group | **2/7** | **5/7** | **+3** |
   | **Total** | **18/28** | **21/28** | **+3** |

   Under the notebook's current threshold code, Age Group EOPP (0.184) and EOD (0.184) pass (< 0.20), and Age Group CAL (0.065) passes (< 0.10). Under the standard thresholds in main.tex's caption, all three fail.

3. **Near-threshold cells identified as unstable in Table 10 VFR match near-threshold cells in Table 7**:
   - Race x EOPP (0.053 vs 0.10 threshold): VFR 46.7%. Point estimate passes on full test set (16/30 pass in bootstraps) — verdict is genuinely a coin toss.
   - Race x EOD (0.053 vs 0.10): VFR 33.3%. Point estimate passes, 10/30 fail.
   - Race x PP (0.095 vs 0.10): VFR 43.3%. Extremely near threshold; 17/30 pass, 13/30 fail.
   - Age x PP (0.070 vs 0.10): VFR 26.7%. 22/30 pass, 8/30 fail.
   - Eth x CAL (0.034 vs 0.05): VFR 13.3%. 26/30 pass.
   - Sex x CAL (0.026 vs 0.05): VFR 10.0%. 27/30 pass.
   - Eth x DI (0.830 vs 0.80): VFR 6.7%. 28/30 pass.

   This is **correct behaviour, not a threshold-inversion bug**: the point estimate on the full test set passes, but the verdict is unreliable because the value sits close to the threshold.

4. **No hard Table 7 <-> Table 11 contradictions** (no cells where Table 7 says Pass AND fewer than half the subsets pass, or vice versa).

## Reconciled definition (single consistent threshold set)

Use the thresholds in main.tex Table 7's caption for all cells in the notebook:

- DI: Pass if `DI >= 0.80`
- SPD, EOPP, EOD, PP, TI: Pass if `|value| < 0.10`
- CAL: Pass if `value < 0.05`

Use **absolute values** uniformly (notebook already does; main.tex does too via the `|.|` notation).

## Impact of applying this definition

- Notebook verdict counts: Age Group drops from 5/7 to **2/7**. Total drops from **21/28 to 18/28**.
- Intervention Section 10 "fair" count would shift similarly — the notebook's current threshold code inflates fair verdicts post-intervention as well.
- VFR values for EOPP and EOD on Race are unaffected (the 0.20 notebook threshold was already above 0.10; the mean is 0.053 so EOPP with threshold 0.20 almost always passes, but with 0.10 is threshold-adjacent and flips 46.7% of the time — this is the value main.tex reports).
- VFR computed under notebook's 0.20 threshold would be near 0% for all EOPP/EOD (value 0.053 is far below 0.20); main.tex's 46.7% number **requires the 0.10 threshold** to be reproduced.

This means: **main.tex's headline claim that 33.6% of verdicts flip depends on standard thresholds being used; it cannot be reproduced by running the notebook as-is.**

## Root cause

The `FairnessCalculator.THRESHOLDS` class-attribute in Cell 4 of the notebook was set to EOPP=0.20, EOD=0.20, CAL=0.10 with an internal justification comment referencing "Agarwal et al. (2018)". Main.tex's Section 4.2 and Table 7 caption both state the standard thresholds (0.10, 0.10, 0.05). Either the notebook was edited to the non-standard values after the manuscript's Section 4.2 was drafted, or the values in main.tex Tables 7/10/11 were computed by a separate script using standard thresholds and never folded back into the notebook.

## Action to resolve

1. Update `FairnessCalculator.THRESHOLDS` (Cell 4) to EOPP=0.10, EOD=0.10, CAL=0.05.
2. Re-run the notebook end-to-end so Tables 3, 6, 10, etc. in the notebook match main.tex.
3. If any author wishes to argue for 0.20 thresholds, that choice must be disclosed in Methods; main.tex currently does not do this.

See `results/consistency_audit.csv` for the row-level audit.
