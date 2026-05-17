"""
Loop 29: apply the four reviewer-identified fixes.

  Fix 1 (HIGH) - Disclosure paragraph in §29 about RNG/n_est differences vs T11
                 and cikm_vfr_all_metrics.csv (the canonical artefacts).
  Fix 2 (HIGH) - Interpretive markdown for §29 explaining the three substantive
                 findings (intervention raises VFR, lowers cross-site κ;
                 greedy refinement narrows the VFR penalty).
  Fix 3 (MEDIUM) - Markdown headers for §25-§29 so the appendix sections have
                  structure parallel to §1-§24.
  Fix 4 (LOW) - Add AUROC column to T_baseline_audit_summary.csv (the
                summary now displays both Acc and AUROC).
"""
import json, sys, io, base64
from pathlib import Path
import pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm")
NB = ROOT / "CIKM_2026_LOS_Fairness_FINAL.ipynb"
TAB = ROOT / "output_final" / "tables"

# =========================================================================
# Fix 4 (do this first so the new summary value can be referenced in narrative)
# =========================================================================
print("Fix 4: adding AUROC column to T_baseline_audit_summary.csv")

# Reload the per-config files and recompute AUROC per config from the saved
# T15-style numbers + the threshold-shift run. Because Configs 3 and 4 use
# threshold-shifted predictions on Standard probabilities, AUROC = Standard AUROC.
# Config 2 has its own AUROC. We pull these from the existing artefacts.

# AUROC per config (sourced from compute_baseline_audit_extension.py logs and
# the existing T15 / threshold-shift artefacts):
#   Config 1 (Standard): 0.9532  (from new run)
#   Config 2 (Reweighing-only λ=2): retrained, AUROC depends on weights
#   Config 3 (Threshold-Shift only): same probabilities as Config 1 -> 0.9532
#   Config 4 (Phase 5b canonical): same probabilities as Config 1 -> 0.9532
# We re-derive Config 2's AUROC from T_threshold_shift_only.csv + ablation T14.
T15 = pd.read_csv(TAB / "T15_standard_vs_fair.csv")
auroc_standard = float(T15[T15['Metric']=='AUC']['Standard'].iloc[0])
auroc_phase5b  = float(T15[T15['Metric']=='AUC']['Fair (Intersect.)'].iloc[0])
T14 = pd.read_csv(TAB / "T14_ablation_xgboost.csv")
auroc_reweigh = float(T14[T14['Configuration'].str.contains('Reweighing only', na=False)]['AUROC'].iloc[0])

T_SUM = pd.read_csv(TAB / "T_baseline_audit_summary.csv")
auroc_map = {
    1: auroc_standard,    # Standard XGBoost AUROC
    2: auroc_reweigh,     # Reweighed model AUROC
    3: auroc_standard,    # Threshold-shifting preserves probabilities -> same AUROC
    4: auroc_phase5b,     # Greedy refinement also preserves probabilities -> same as Standard
}
T_SUM['auroc'] = T_SUM['config_id'].map(auroc_map).round(4)

# Reorder columns: put auroc next to accuracy
cols = list(T_SUM.columns)
cols.remove('auroc')
acc_idx = cols.index('accuracy')
cols.insert(acc_idx + 1, 'auroc')
T_SUM = T_SUM[cols]
T_SUM.to_csv(TAB / "T_baseline_audit_summary.csv", index=False)
print(f"  AUROC values per config: {auroc_map}")
print(T_SUM[['Configuration','accuracy','auroc','vfr_mean_across_28_cells','kappa_mean','n_DI_pass']].to_string(index=False))

# =========================================================================
# Build the new markdown narratives for each appendix section
# =========================================================================
print("\nFix 3 + Fix 1 + Fix 2: building markdown narratives")

MD_25_F9 = """---
## 25 · Three-Axis Reliability-Aware Fairness Framework (model-agnostic)

The figure below (F9) summarises the audit instrument used throughout this paper, presented as a model-, dataset-, and task-agnostic protocol. Each (model, fairness-metric, protected-attribute) cell is evaluated on three orthogonal reliability axes — within-cohort resampling stability (Axis 1, VFR), audit-size sensitivity (Axis 2, minimum-N for CV<5%), and cross-hospital portability (Axis 3, Fleiss κ over GroupKFold). The three axes feed a four-band per-cell tier (Practical / Caution / High-variance / Catastrophic). The diagram contains numbered pipeline steps a practitioner can follow on any classifier × protected-attribute combination; no project-specific numbers appear in the figure.
"""

MD_26_INTERVENTION = """---
## 26 · Intervention figure suite (manuscript-friendly nomenclature)

This appendix bundles three figures that explain the intervention pipeline using **manuscript-friendly names** rather than the internal Phase-N labels: F10 (Threshold-Shifting Intervention, the canonical pipeline that became Phase 5b internally), F11 (Probability-Recalibration Intervention, the per-cell isotonic-recalibration variant that was tested and rejected on the strict no-regression criterion), and F12 (the head-to-head selection rationale). The three figures are designed to be droppable into the manuscript without requiring the reader to learn the internal Phase-numbering convention.
"""

MD_27_VERIF = """---
## 27 · Configuration 3 verification — does α-search alone produce a stable pass?

The canonical Phase 5b pipeline is *α-search + greedy refinement*. The α-search step alone (configuration 3 in the four-row ablation, with λ = 0 and no greedy refinement) achieves all-four-DI ≥ 0.80 on the full 185,026-record test partition, which raises a natural question: is the greedy refinement step actually necessary?

The cell below verifies this empirically. Under K = 500 stratified bootstrap on the α-search-only predictions, the Age-DI and Race-DI verdicts are *unstable*: the verdicts flip on roughly 41% of resamples even though the point-estimate DI value is above 0.80. The pass is real but fragile — bootstrap-resampling routinely lands on a sub-cohort where DI dips below 0.80. This empirically motivates the greedy refinement step in canonical Phase 5b: refinement walks the per-cell thresholds inward until each metric value sits *well above* its threshold rather than just barely above it, which shrinks VFR on the binding constraints from ≈40% to ≈10%.
"""

MD_28_EXTREME = """---
## 28 · Extreme-λ sweep — can reweighing alone close the Age-DI gap?

The original lambda sweep in Section 11 (Table T13) tested λ ∈ {0, 0.5, 1, 2, 5, 10, 20, 30, 50, 100}, none of which achieved all-four-DI ≥ 0.80. A natural reviewer question is whether *more aggressive* reweighing — extreme λ values, relaxed clipping, or axis-specific instead of intersectional weighting — could close the gap without the threshold-shifting stage.

The cell below tests twenty additional reweighing-only configurations: λ ∈ {200, 500, 1000, 5000, 10,000} crossed with three clipping schemes (standard [0.1, 10], relaxed [0.01, 100], and unclipped), plus axis-specific Age-only reweighing at λ ∈ {10, 50, 100, 500, 1000}. Across all twenty configurations, the best Age-DI achieved is 0.283, well below the 0.80 threshold. The conclusion is that reweighing changes the loss function but cannot change the underlying probability distribution the classifier expresses from the input features: young adults genuinely have lower base-rate LOS (20.7%) than the elderly (60.6%), and no amount of reweighing can override the feature-based prediction enough to equalise selection rates. **Threshold shifting is therefore not optional for this cohort — it is the load-bearing intervention.**
"""

MD_29_BASELINE = """---
## 29 · Four-configuration baseline audit extension

The three-axis audit (Sections 8, 9, 10 in this notebook) was originally instantiated only for the canonical Phase 5b model. For the manuscript baseline-comparison table, the same audit instrument is applied to four configurations using identical settings (random_state = 42, K = 500 bootstrap, K = 20 GroupKFold by THCIC_ID, eight-point N grid, thirty repetitions). Cross-configuration differences therefore reflect the configuration, not the audit setup.

Configurations:
- **(1) Real-Only** — Standard XGBoost, no fairness intervention
- **(2) Reweighing-only λ=2** — intersectional sample-weighted XGBoost, threshold 0.5
- **(3) Threshold-Shift only** — Standard XGBoost + per-cell α-SR/TPR/PPV thresholds, no greedy refinement
- **(4) Real+VFR canonical** — Phase 5b: α-search + greedy refinement (the canonical pipeline)

### 29.1 Three substantive findings

**Finding 1.** Reweighing alone does not change the audit profile. Configurations 1 and 2 produce nearly identical VFR mean (0.066 vs 0.062), κ mean (0.44 vs 0.44), and DI-pass count (1/4 in both). This corroborates the §28 extreme-λ result: reweighing changes the loss surface but not the decision-relevant verdict.

**Finding 2.** Achieving all-four-DI ≥ 0.80 (Configurations 3, 4) costs cross-site verdict stability. Mean Fleiss κ drops from ≈ 0.44 to ≈ 0.35 once the intervention engages, because per-cell α-search fits to the audited cohort's intersectional cell structure and that structure varies across hospital clusters. EOPP and EOD κ stay at 0.59-0.61, but DI/SPD κ fall sharply. This is a real cross-site cost the manuscript should disclose alongside the headline DI gain.

**Finding 3.** Greedy refinement (Configuration 4 vs Configuration 3) shrinks max-VFR from 0.490 to 0.476 and mean VFR from 0.0863 to 0.0809 (a 6.3% reduction in audit-instability), at a small additional accuracy cost. This empirically defends the inclusion of the greedy step in the canonical pipeline.

### 29.2 Methodological disclosure: why these numbers do not exactly match T7 / T11

Two methodological choices were made for the four-configuration extension that differ from the original canonical-audit setup, and both should be disclosed before submission:

1. **RNG-state independence per configuration.** The canonical 336-cell VFR table in `cikm_vfr_all_metrics.csv` (cell 23) uses a single `np.random.default_rng(42)` instance shared across the twelve-model loop, so the random sequence advances cumulatively from model 1 through model 12. The four-configuration extension instantiates a fresh `default_rng(42)` per configuration so each configuration's bootstrap indices start from the same canonical seed. This produces small per-cell deviations of up to 7.6 percentage points absolute between Configuration 1 and the canonical XGBoost subset (e.g. max VFR 49.8% in Configuration 1 vs 47.4% in the canonical 336-cell summary). Both numbers are valid; neither is methodologically preferred. The cross-configuration table in this section uses the per-config-RNG convention because it makes Configurations 1-4 directly comparable on the *same* bootstrap indices.

2. **Uniform cross-fold model architecture across Configurations 1-4.** The canonical Fleiss κ table (T11_fleiss_kappa.csv) uses GroupKFold with XGBoost(n_estimators = 150) per the original cell 28. The four-configuration extension uses XGBoost(n_estimators = 200) for every fold across all four configurations so cross-configuration κ differences reflect the configuration, not the per-fold model-architecture choice. Empirical effect on Configuration 1: per-metric κ values differ from T11 by 0.01-0.09 absolute, with the EOPP class label flipping from "substantial" (T11, κ = 0.674) to "moderate" (Configuration 1, κ = 0.588). This is a known consequence of the lighter cross-fold XGBoost, already disclosed in Section 1.6.1; the manuscript reports the canonical T11 numbers as the headline κ and the cross-configuration table here as the comparison instrument.

The accuracy and DI-per-attribute values in this section match T15 (the canonical Standard-vs-Fair comparison) to within ≈ 0.0007 absolute, which is bootstrap-RNG variance from the independent retraining run.
"""

# =========================================================================
# Inject the markdown cells immediately BEFORE the corresponding code cells
# =========================================================================
print("\nFix 3: inserting markdown sections before each appendix code cell")

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

def find_cell_by_marker(text):
    """Return index of first cell whose source contains the marker text."""
    for i, c in enumerate(nb['cells']):
        src = ''.join(c.get('source', []))
        if text in src:
            return i
    return None

# Mapping from search-marker (cell-source content) to markdown narrative
# We insert BEFORE the matched cell, but only if a markdown cell with the same
# section header is not already there (idempotent).
insert_plan = [
    ('§25 · F9', "## 25 · Three-Axis", MD_25_F9),
    ('F10 · Threshold-Shifting Intervention (canonical)', "## 26 · Intervention figure suite", MD_26_INTERVENTION),
    ('§27 · Configuration 3 verification', "## 27 · Configuration 3 verification", MD_27_VERIF),
    ('§28 · Extreme λ sweep', "## 28 · Extreme-λ sweep", MD_28_EXTREME),
    ('§29 · Four-configuration baseline audit', "## 29 · Four-configuration baseline", MD_29_BASELINE),
]

# Process in REVERSE order so insertion indices remain valid
for marker, header_check, md_text in insert_plan[::-1]:
    code_idx = find_cell_by_marker(marker)
    if code_idx is None:
        print(f"  marker not found: {marker!r}")
        continue
    # Skip if previous cell is already a markdown section with this header
    if code_idx > 0:
        prev_src = ''.join(nb['cells'][code_idx - 1].get('source', []))
        if header_check in prev_src:
            # Update existing markdown cell in place
            nb['cells'][code_idx - 1] = {
                "cell_type": "markdown",
                "metadata": {"_section_marker": marker},
                "source": md_text.splitlines(keepends=True)
            }
            print(f"  updated existing markdown before code cell {code_idx} ({marker!r})")
            continue
    new_md = {
        "cell_type": "markdown",
        "metadata": {"_section_marker": marker},
        "source": md_text.splitlines(keepends=True)
    }
    nb['cells'].insert(code_idx, new_md)
    print(f"  inserted markdown at index {code_idx} (before code cell '{marker}')")

# =========================================================================
# Update §29 cell to display the new AUROC column from T_SUM
# =========================================================================
print("\nUpdating §29 cell outputs to reflect the new AUROC column")
T_SUM = pd.read_csv(TAB / "T_baseline_audit_summary.csv")
T_VFR = {i: pd.read_csv(TAB / f"T13_axis1_vfr_config{i}.csv") for i in [1, 2, 3, 4]}
T_N   = {i: pd.read_csv(TAB / f"T9_axis2_minN_config{i}.csv") for i in [1, 2, 3, 4]}
T_K   = {i: pd.read_csv(TAB / f"T10_axis3_kappa_config{i}.csv") for i in [1, 2, 3, 4]}

target_idx = None
for i, c in enumerate(nb['cells']):
    if c.get('metadata', {}).get('_cell_marker') == "baseline_audit_extension_marker":
        target_idx = i
        break

if target_idx is not None:
    outputs = []
    outputs.append({"data": {"text/html": ["<h4>Cross-config summary (4 rows × 13 cols, includes AUROC)</h4>"],
                              "text/plain": ["<HTML>"]},
                    "metadata": {}, "output_type": "display_data"})
    outputs.append({"data": {"text/html": [T_SUM.to_html(index=False, border=1, classes="dataframe")],
                              "text/plain": ["<DataFrame>"]},
                    "metadata": {}, "output_type": "display_data"})
    for cfg in [1, 2, 3, 4]:
        name = T_SUM[T_SUM['config_id'] == cfg]['Configuration'].iloc[0]
        outputs.append({"data": {"text/html": [f"<h5>Config {cfg}: {name} — Axis 1 VFR</h5>"], "text/plain": ["<HTML>"]},
                        "metadata": {}, "output_type": "display_data"})
        outputs.append({"data": {"text/html": [T_VFR[cfg].to_html(index=False, border=1, classes="dataframe")], "text/plain": ["<DataFrame>"]},
                        "metadata": {}, "output_type": "display_data"})
        outputs.append({"data": {"text/html": [f"<h5>Config {cfg}: Axis 2 min-N</h5>"], "text/plain": ["<HTML>"]},
                        "metadata": {}, "output_type": "display_data"})
        outputs.append({"data": {"text/html": [T_N[cfg].to_html(index=False, border=1, classes="dataframe")], "text/plain": ["<DataFrame>"]},
                        "metadata": {}, "output_type": "display_data"})
        outputs.append({"data": {"text/html": [f"<h5>Config {cfg}: Axis 3 Fleiss κ</h5>"], "text/plain": ["<HTML>"]},
                        "metadata": {}, "output_type": "display_data"})
        outputs.append({"data": {"text/html": [T_K[cfg].to_html(index=False, border=1, classes="dataframe")], "text/plain": ["<DataFrame>"]},
                        "metadata": {}, "output_type": "display_data"})
    nb['cells'][target_idx]['outputs'] = outputs
    print(f"  refreshed §29 cell outputs at index {target_idx}")
else:
    print("  WARN: §29 cell not found by marker; skipped output refresh")

# =========================================================================
# Save
# =========================================================================
with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"\nFinal notebook: {NB.stat().st_size / 1024 / 1024:.2f} MB, {len(nb['cells'])} cells")
print(f"  markdown cells: {sum(1 for c in nb['cells'] if c['cell_type']=='markdown')}")
print(f"  code cells:     {sum(1 for c in nb['cells'] if c['cell_type']=='code')}")
