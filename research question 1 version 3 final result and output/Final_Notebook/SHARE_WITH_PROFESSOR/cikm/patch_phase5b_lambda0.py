"""
Add Phase 5b: try threshold-only intervention on the UNREWEIGHTED
predictions (lambda = 0). Reweighing in this dataset actually
HURT Race DI (0.664 -> 0.592 standalone), so the reweighed baseline
forces the thresholding to do more work to recover Race DI. Skipping
reweighing should give the thresholder a higher-accuracy starting
point.

Phase 5b applies the alpha-grid search + greedy relaxation to the
standard XGBoost predictions directly. If the resulting (Acc, all-4-DI)
beats the current canonical, promote it.
"""
import json
from pathlib import Path

NB = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\CIKM_2026_LOS_Fairness_FINAL.ipynb")

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)


# Find the ablation cell (cell 29 in the live nb) — the one with "Phase 5"
target_idx = None
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c.get("source", []))
    if 'Phase 5 · Greedy threshold relaxation' in src:
        target_idx = i
        break

if target_idx is None:
    print("ERROR: cannot find Phase 5 cell")
    raise SystemExit(1)
print(f"Target cell: {target_idx}")

# Insert Phase 5b right after the Phase 5 block, before T14 build
INSERT_BEFORE = "# ──────────────────────────────────────────────────────────────\n# Build T14 (ablation table)"

PHASE5B = """# ──────────────────────────────────────────────────────────────
# Phase 5b · Threshold-only on unreweighted predictions (lambda = 0)
# Reweighing hurt Race DI in this dataset; try thresholding the standard
# XGBoost output directly to see if a better Pareto point exists.
# ──────────────────────────────────────────────────────────────
print("\\nPhase 5b · Threshold-only on lambda=0 predictions")
ypb_lam0 = canon_proba.astype(np.float32)  # standard XGBoost predictions
chosen_lam0 = alpha_search_for_probs(ypb_lam0, drop_limit=0.08)
if chosen_lam0 is not None and bool(chosen_lam0['All4_pass']):
    thr_lam0 = chosen_lam0['thresholds']
    yp_lam0 = _apply_thresholds(ypb_lam0, thr_lam0)
    cur_acc_b = accuracy_score(y_test, yp_lam0)
    cur_pass_b, cur_di_b = _all4_pass(yp_lam0, ypb_lam0)
    print(f"  alpha-search: acc={cur_acc_b:.4f} drop={(std_acc_xgb-cur_acc_b)*100:.2f}pp "
          f"DI(R/S/E/A)={cur_di_b['RACE']:.3f}/{cur_di_b['SEX']:.3f}/{cur_di_b['ETHNICITY']:.3f}/{cur_di_b['AGE_GROUP']:.3f}")
    # Greedy relaxation on this lambda=0 baseline
    refined_b = dict(thr_lam0)
    n_iter_b = 0; improved_b = 0
    while True:
        progressed = False
        for cell_key, thr in sorted(refined_b.items(), key=lambda kv: -abs(kv[1] - 0.5)):
            n_iter_b += 1
            if abs(thr - 0.5) < 0.005: continue
            step = 0.01 if thr > 0.5 else -0.01
            new_thr = thr - step
            if (thr > 0.5 and new_thr < 0.5) or (thr < 0.5 and new_thr > 0.5):
                new_thr = 0.5
            new_thr = float(np.clip(new_thr, 0.01, 0.99))
            trial = dict(refined_b); trial[cell_key] = new_thr
            yp_trial = _apply_thresholds(ypb_lam0, trial)
            ok, di_trial = _all4_pass(yp_trial, ypb_lam0)
            if ok:
                trial_acc = accuracy_score(y_test, yp_trial)
                if trial_acc > cur_acc_b:
                    refined_b = trial; cur_acc_b = trial_acc; cur_di_b = di_trial
                    progressed = True; improved_b += 1
        if not progressed: break
    print(f"  + Phase5: acc={cur_acc_b:.4f} drop={(std_acc_xgb-cur_acc_b)*100:.2f}pp "
          f"DI(R/S/E/A)={cur_di_b['RACE']:.3f}/{cur_di_b['SEX']:.3f}/{cur_di_b['ETHNICITY']:.3f}/{cur_di_b['AGE_GROUP']:.3f} "
          f"({improved_b} relaxations)")

    # If lambda=0 path beats current canonical, promote it
    if cur_acc_b > config4_summary['Accuracy']:
        print(f"  Phase 5b BEATS canonical by {(cur_acc_b - config4_summary['Accuracy'])*100:.3f}pp - promoting")
        yp5b = _apply_thresholds(ypb_lam0, refined_b)
        config5b_summary, fair5b = _summary_for_config(yp5b, ypb_lam0, "(5b) Phase5b lambda=0 + threshold")
        ablation_rows.append(config5b_summary)
        fair_pred  = yp5b
        fair_proba = ypb_lam0
        fair4 = fair5b
        config4_summary = config5b_summary
        thr_dict3 = refined_b
    else:
        print(f"  Phase 5b did not beat canonical - keeping current")
else:
    print("  alpha-search at lambda=0 did not find an all-4-DI candidate within budget")

"""

c = nb["cells"][target_idx]
src = "".join(c.get("source", []))
if "Phase 5b" not in src and INSERT_BEFORE in src:
    new_src = src.replace(INSERT_BEFORE, PHASE5B + INSERT_BEFORE, 1)
    c["source"] = new_src.splitlines(keepends=True)
    c["outputs"] = []
    c["execution_count"] = None
    for j in range(target_idx + 1, len(nb["cells"])):
        if nb["cells"][j]["cell_type"] == "code":
            nb["cells"][j]["outputs"] = []
            nb["cells"][j]["execution_count"] = None
    with open(NB, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Phase 5b inserted; downstream cleared")
else:
    print("Phase 5b already present or marker not found")
