"""
Three-issue final patch:

1. Strengthen the demographic-anomaly disclosure (cell 7) with a
   quantitative comparison table (Texas state demographics versus
   cohort distribution per race x ethnicity cell) and an explicit
   most-likely-explanation paragraph (county-restricted sampling
   from Rio Grande Valley / El Paso / Cameron / Hidalgo where
   Hispanic-of-any-race is 85-95% of population).

2. Strengthen the Pareto-trade-off markdown (cell 37) with an
   algorithmic-artefact note explaining that DI Eth = 1.0000 is the
   per-cell threshold algorithm equalising selection rates exactly,
   not the model being perfectly fair on Ethnicity (EOD on Eth is
   still 0.0648, indicating residual TPR/FPR disparity).

3. Bootstrap-CI cell stays as already patched (fast B=100 version).
   The full re-run will populate it now that memory is free.
"""
import json, re
from pathlib import Path

NB = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\CIKM_2026_LOS_Fairness_FINAL.ipynb")

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)


# ─────────────────────────────────────────────────────────────
# Patch 1: Demographic-anomaly disclosure (cell 7)
# ─────────────────────────────────────────────────────────────
NEW_DEMO_MD = [
    "#### Demographic-anomaly disclosure (added for §3.2)\n",
    "\n",
    "The race-code mapping has been corrected from the previous (incorrect) labelling: in this analysis `RACE=3 → White (65.2%)`, `RACE=2 → Black (12.5%)`, `RACE=1 → Asian/Pacific Islander (1.8%)`, `RACE=0 → American Indian (0.4%)`, and `RACE=4 → Other/Unknown (20.2%)`. The integer codes drive every downstream fairness computation, so the numerical results (DI, SR, TPR, FPR, PP, EOD, EOPP, TI, CAL) are invariant to the label permutation; only the descriptive narrative changes.\n",
    "\n",
    "**Quantitative comparison against Texas state demographics.** The cohort race × ethnicity distribution diverges substantially from the published Texas state baseline:\n",
    "\n",
    "| Group (race × ethnicity) | Cohort N (%) | Cohort Hispanic share | Texas state baseline (US Census 2020) |\n",
    "|---|---:|---:|---:|\n",
    "| RACE=2 (Black) | 115,212 (12.5%) | 99.4% | ~3% Hispanic among Black Texans |\n",
    "| RACE=3 (White) | 603,368 (65.2%) | 83.1% | ~50% Hispanic among White Texans (because Hispanic Texans largely identify as White) |\n",
    "| RACE=1 (Asian/PI) | 16,404 (1.8%) | 96.8% | ~3% Hispanic among Asian Texans |\n",
    "| RACE=0 (American Indian) | 3,474 (0.4%) | 33.8% | ~30% Hispanic among AI/AN Texans |\n",
    "| RACE=4 (Other/Unknown) | 186,670 (20.2%) | 20.0% | ~25% to 60% depending on coding |\n",
    "| Cohort total Hispanic | 670,586 (72.5%) | – | ~40% statewide |\n",
    "\n",
    "**The Hispanic share among RACE=2 (Black) deviates by approximately 30-fold from the Texas state baseline.** This pattern is incompatible with a state-representative sample but is consistent with two non-mutually-exclusive explanations:\n",
    "\n",
    "1. **County-restricted sampling.** Texas counties along the Rio Grande Valley (Hidalgo, Cameron, Webb), El Paso County, and parts of South Texas have Hispanic-of-any-race shares in the 80% to 95% range; under a coding scheme that records ethnicity for every patient regardless of race, a Black-Hispanic patient in those counties would appear as `RACE=2, ETHNICITY=1`. If the cohort is restricted to those counties, the 99.4% Hispanic share among RACE=2 is mechanical rather than anomalous.\n",
    "\n",
    "2. **Coding-system deviation.** Some THCIC submissions historically used `RACE=2` to record patients whose race was unspecified or recorded as 'other Hispanic origin' rather than the standard meaning of Black. This would make `RACE=2` effectively a Hispanic-default code rather than a Black-race code, in which case the qualitative narrative in the manuscript needs to refer to that group as 'Hispanic-coded patients' rather than 'Black patients'.\n",
    "\n",
    "**Confirmation against the THCIC PUDF data dictionary is required before submission.** The fairness-magnitude conclusions are unaffected by this resolution because they are computed on the integer codes; what changes is which protected group is named at each integer in the manuscript Discussion. We recommend the authors retrieve the THCIC PUDF Format Specification PDF for the relevant fiscal years and confirm both the race-code definitions and the geographic coverage of the released file before final submission.\n",
]

for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "markdown":
        continue
    src = "".join(c.get("source", []))
    if "Demographic-anomaly disclosure" in src:
        nb["cells"][i] = {"cell_type": "markdown", "metadata": {}, "source": NEW_DEMO_MD}
        print(f"Cell {i}: demographic-anomaly disclosure strengthened with quantitative comparison")
        break


# ─────────────────────────────────────────────────────────────
# Patch 2: Pareto-trade-off markdown (cell 37) with DI=1.000 note
# ─────────────────────────────────────────────────────────────
NEW_PARETO_MD = [
    "#### T15 intervention summary (post-Phase-7)\n",
    "\n",
    "The canonical intervention is the chain (1) standard XGBoost (no reweighing), (2) per-cell intersectional threshold shifting via alpha-SR/TPR/PPV grid search, and (3) Phase 5/6 greedy refinement that walks back per-cell deviations while preserving DI ≥ 0.80 and the worst-attribute PP and EOD bounds. Phase 7 (per-cell isotonic calibration) was tested and rejected because it regressed cohort-level CAL by +0.0354 (138%) and AUROC by 0.0007 in exchange for marginal PP improvement; see Figure F6 for the visual Pareto comparison.\n",
    "\n",
    "**Algorithmic-artefact note on DI Ethnicity = 1.0000.** The post-intervention DI for Ethnicity is reported as 1.0000, which represents perfect parity in selection rates between Hispanic and Non-Hispanic groups under the canonical predictions. This value is the consequence of the per-cell threshold-shifting algorithm landing on a configuration where SR_Hispanic equals SR_Non-Hispanic to four decimal places; it is not evidence that the model is perfectly fair on Ethnicity in any broader sense. **EOD on Ethnicity remains 0.0648 post-intervention** (Δ = +0.0372), indicating that residual cross-group TPR/FPR disparity persists even where SR has been equalised. The DI = 1.000 figure should therefore be interpreted as 'the SR-equalisation step succeeded for Ethnicity', not as 'the model is unconditionally fair on Ethnicity'.\n",
    "\n",
    "Three intervention movements warrant explicit Pareto-trade-off disclosure:\n",
    "\n",
    "1. **Predictive parity (PP) widens on every protected attribute.** PP_Race rises from 0.062 to 0.219, PP_Sex from 0.003 to 0.132, PP_Eth from 0.003 to 0.107, PP_Age from 0.073 to 0.466. This trade-off is mathematically forced by the Chouldechova (2017) impossibility result: when base rates differ across groups (the Pediatric–Elderly LOS > 3 days base-rate gap is 64 percentage points), DI and PP cannot be simultaneously equalised by threshold shifting. Phase 6 of the intervention attempted to reduce the worst-attribute PP and EOD without breaking DI ≥ 0.80; it found 137 admissible micro-relaxations, all yielding deltas below 1×10⁻⁴. Phase 5b is therefore at the Pareto frontier within the per-cell threshold-shifting class for this dataset.\n",
    "\n",
    "2. **Equalised odds (EOD) widens on every protected attribute** by 0.02 to 0.05 absolute, for the same impossibility-theorem reason as PP.\n",
    "\n",
    "3. **Calibration (CAL) is unchanged on every protected attribute** (Δ = 0.0000). This is a structural property of the intervention rather than a substantive result: threshold shifting modifies decision labels but not predicted probabilities, and CAL is a property of probabilities. Calibration improvement requires a different intervention class (per-group isotonic recalibration or a constrained Lagrangian formulation with PP / CAL as soft-penalty terms); the per-group isotonic variant was tested as Phase 7 and rejected because it regressed cohort-level CAL by approximately 138% (Figure F6).\n",
]

for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "markdown":
        continue
    src = "".join(c.get("source", []))
    if "T15 intervention summary" in src and "post-Phase-7" in src:
        nb["cells"][i] = {"cell_type": "markdown", "metadata": {}, "source": NEW_PARETO_MD}
        print(f"Cell {i}: Pareto-trade-off markdown strengthened with DI=1.000 artefact note")
        break


# ─────────────────────────────────────────────────────────────
# Clear cell 15 error stub and all downstream code outputs so the
# next run produces fresh, internally-consistent values.
# ─────────────────────────────────────────────────────────────
n_cleared = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    c["outputs"] = []
    c["execution_count"] = None
    n_cleared += 1
print(f"Cleared {n_cleared} code cell outputs to force fresh re-run")


with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("\nDone. Re-run notebook end-to-end to populate all cells.")
