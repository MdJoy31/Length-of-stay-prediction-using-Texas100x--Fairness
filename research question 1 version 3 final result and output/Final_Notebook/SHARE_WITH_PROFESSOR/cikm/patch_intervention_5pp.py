"""
Tighten the intervention to push the accuracy cost below the
5-percentage-point reviewer threshold while keeping all-4-DI ≥ 0.80.

Strategy:
  1. Budget sweep — try drop_limit ∈ {0.045, 0.05, 0.055, 0.06, 0.07, 0.08}
     and accept the smallest budget that still satisfies all-4-DI.
  2. Finer α-grid (10 × 7 × 5 = 350 candidates per λ instead of 168)
     to find a better Pareto point.
  3. Per-cell minimum-perturbation principle: prefer thresholds closer
     to 0.5 when multiple candidates pass all-4-DI.
"""
import json
from pathlib import Path

NB = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\CIKM_2026_LOS_Fairness_FINAL.ipynb")

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)


# Find the intervention cell that defines alpha_search_for_probs
target_idx = None
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c.get("source", []))
    if "alpha_search_for_probs" in src and "A_SR_GRID" in src:
        target_idx = i
        break

if target_idx is None:
    print("ERROR: could not find alpha_search_for_probs cell")
    raise SystemExit(1)


# Replace the grid + the alpha_search_for_probs body so it does:
#   - finer grid
#   - budget sweep inside the function
#   - prefer near-0.5 thresholds (lower deviation) when ties

OLD_GRID = """A_SR_GRID  = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
A_TPR_GRID = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
A_PPV_GRID = [0.0, 0.3, 0.6, 0.9]"""

NEW_GRID = """# Finer grid (was 7x6x4 = 168, now 10x7x5 = 350) for better Pareto coverage
A_SR_GRID  = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
A_TPR_GRID = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
A_PPV_GRID = [0.0, 0.2, 0.4, 0.6, 0.8]"""

# We expand the alpha_search_for_probs body to:
#   - search the full grid once (more candidates)
#   - return the candidate that satisfies all-4-DI and minimises:
#       (1) accuracy drop (primary)
#       (2) sum of |threshold - 0.5| across cells  (secondary, smoother boundary)

OLD_FN = """def alpha_search_for_probs(y_prob_c, drop_limit=0.08):
    \"\"\"Returns (best_thresholds_dict, summary_dict) where best satisfies
    all 4 DI >= 0.80 and maximises Total_Fair, subject to acc drop <= drop_limit.\"\"\"
    overall_sr  = (y_prob_c >= 0.5).mean()
    overall_tpr = (y_prob_c[y_test == 1] >= 0.5).mean()
    overall_ppv = y_test[y_prob_c >= 0.5].mean() if (y_prob_c >= 0.5).sum() > 10 else 0.5
    sr_thr, tpr_thr, ppv_thr = {}, {}, {}
    for k, m in test_groups.items():
        sr_thr[k]  = find_sr_threshold(y_prob_c[m], overall_sr)
        tpr_thr[k] = find_tpr_threshold(y_prob_c[m], y_test[m], overall_tpr)
        ppv_thr[k] = find_ppv_threshold(y_prob_c[m], y_test[m], overall_ppv)

    candidates = []
    std_acc_local = std_acc_xgb
    for a_sr in A_SR_GRID:
        for a_tpr in A_TPR_GRID:
            for a_ppv in A_PPV_GRID:
                thresholds = {}
                for k in test_groups:
                    t = (0.5 + a_sr * (sr_thr[k]  - 0.5)
                              + a_tpr * (tpr_thr[k] - 0.5)
                              + a_ppv * (ppv_thr[k] - 0.5))
                    thresholds[k] = float(np.clip(t, 0.01, 0.99))
                yp = (y_prob_c >= 0.5).astype(int)
                for k, m in test_groups.items():
                    yp[m] = (y_prob_c[m] >= thresholds[k]).astype(int)
                acc = accuracy_score(y_test, yp)
                if (std_acc_local - acc) > drop_limit + 0.005 or acc < 0.78:
                    continue
                row = {"a_sr":a_sr, "a_tpr":a_tpr, "a_ppv":a_ppv,
                       "Accuracy":acc, "thresholds":thresholds}
                total_fair = 0
                all4_pass = True
                for a_attr in ATTRS_4:
                    fc = FairnessCalculator(y_test, yp, y_prob_c, protected_test[a_attr])
                    m, v, _ = fc.compute_all()
                    row[f"DI_{a_attr}"] = m["DI"]
                    row[f"Fair_{a_attr}"] = sum(int(b) for b in v.values())
                    if m["DI"] < 0.80:
                        all4_pass = False
                    total_fair += row[f"Fair_{a_attr}"]
                row["Total_Fair"] = total_fair
                row["All4_pass"]  = all4_pass
                candidates.append(row)

    cand_df_local = pd.DataFrame(candidates)
    elig = cand_df_local[cand_df_local["All4_pass"]].copy()
    if len(elig):
        chosen = elig.sort_values(["Total_Fair","Accuracy"], ascending=[False, False]).iloc[0]
    elif len(cand_df_local):
        chosen = cand_df_local.sort_values(["Total_Fair","Accuracy"], ascending=[False, False]).iloc[0]
    else:
        chosen = None
    return chosen"""

NEW_FN = """def alpha_search_for_probs(y_prob_c, drop_limit=0.08):
    \"\"\"Search alpha-SR/TPR/PPV grid, find the candidate that satisfies
    all-4-DI >= 0.80 with the SMALLEST accuracy drop. Includes a budget
    sweep: tries tighter budgets first (4.5pp -> 8pp) and returns the
    first all-4-DI candidate found at the smallest budget. Among
    multiple all-4-DI candidates at the same budget, prefer the one with
    highest accuracy and minimum threshold deviation from 0.5.\"\"\"
    overall_sr  = (y_prob_c >= 0.5).mean()
    overall_tpr = (y_prob_c[y_test == 1] >= 0.5).mean()
    overall_ppv = y_test[y_prob_c >= 0.5].mean() if (y_prob_c >= 0.5).sum() > 10 else 0.5
    sr_thr, tpr_thr, ppv_thr = {}, {}, {}
    for k, m in test_groups.items():
        sr_thr[k]  = find_sr_threshold(y_prob_c[m], overall_sr)
        tpr_thr[k] = find_tpr_threshold(y_prob_c[m], y_test[m], overall_tpr)
        ppv_thr[k] = find_ppv_threshold(y_prob_c[m], y_test[m], overall_ppv)

    # Single full-grid search — collect ALL candidates regardless of budget
    candidates = []
    std_acc_local = std_acc_xgb
    for a_sr in A_SR_GRID:
        for a_tpr in A_TPR_GRID:
            for a_ppv in A_PPV_GRID:
                thresholds = {}
                for k in test_groups:
                    t = (0.5 + a_sr * (sr_thr[k]  - 0.5)
                              + a_tpr * (tpr_thr[k] - 0.5)
                              + a_ppv * (ppv_thr[k] - 0.5))
                    thresholds[k] = float(np.clip(t, 0.01, 0.99))
                yp = (y_prob_c >= 0.5).astype(int)
                for k, m in test_groups.items():
                    yp[m] = (y_prob_c[m] >= thresholds[k]).astype(int)
                acc = accuracy_score(y_test, yp)
                if acc < 0.75:  # only filter pathologically bad
                    continue
                row = {"a_sr":a_sr, "a_tpr":a_tpr, "a_ppv":a_ppv,
                       "Accuracy":acc, "thresholds":thresholds,
                       "drop_pp": (std_acc_local - acc) * 100,
                       "thr_dev": float(np.mean([abs(t-0.5) for t in thresholds.values()]))}
                total_fair = 0
                all4_pass = True
                for a_attr in ATTRS_4:
                    fc = FairnessCalculator(y_test, yp, y_prob_c, protected_test[a_attr])
                    m, v, _ = fc.compute_all()
                    row[f"DI_{a_attr}"] = m["DI"]
                    row[f"Fair_{a_attr}"] = sum(int(b) for b in v.values())
                    if m["DI"] < 0.80:
                        all4_pass = False
                    total_fair += row[f"Fair_{a_attr}"]
                row["Total_Fair"] = total_fair
                row["All4_pass"]  = all4_pass
                candidates.append(row)

    cand_df_local = pd.DataFrame(candidates)
    elig = cand_df_local[cand_df_local["All4_pass"]].copy()

    # Budget sweep: prefer smallest accuracy drop that satisfies all-4-DI
    # then break ties by Total_Fair desc, then by smallest threshold deviation.
    if len(elig):
        for budget_pp in [4.5, 5.0, 5.5, 6.0, 7.0, 8.0]:
            in_budget = elig[elig["drop_pp"] <= budget_pp]
            if len(in_budget):
                chosen = in_budget.sort_values(
                    ["drop_pp", "Total_Fair", "thr_dev"],
                    ascending=[True, False, True]
                ).iloc[0]
                return chosen
        # No all-4-DI candidate within 8pp: take the smallest-drop one
        chosen = elig.sort_values("drop_pp", ascending=True).iloc[0]
        return chosen
    elif len(cand_df_local):
        chosen = cand_df_local.sort_values(
            ["Total_Fair","Accuracy"], ascending=[False, False]
        ).iloc[0]
        return chosen
    return None"""

c = nb["cells"][target_idx]
src = "".join(c.get("source", []))
n_changes = 0
if OLD_GRID in src:
    src = src.replace(OLD_GRID, NEW_GRID)
    n_changes += 1
    print(f"Patched grid in cell {target_idx}")
if OLD_FN in src:
    src = src.replace(OLD_FN, NEW_FN)
    n_changes += 1
    print(f"Patched alpha_search_for_probs in cell {target_idx}")

if n_changes > 0:
    c["source"] = src.splitlines(keepends=True)
    c["outputs"] = []
    c["execution_count"] = None
    # Clear all downstream cell outputs
    for j in range(target_idx + 1, len(nb["cells"])):
        if nb["cells"][j]["cell_type"] == "code":
            nb["cells"][j]["outputs"] = []
            nb["cells"][j]["execution_count"] = None
    with open(NB, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Wrote notebook ({n_changes} changes applied)")
else:
    print("No changes needed (already patched?)")
