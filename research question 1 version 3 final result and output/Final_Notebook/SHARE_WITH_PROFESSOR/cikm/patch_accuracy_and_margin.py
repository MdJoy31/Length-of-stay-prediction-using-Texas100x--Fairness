"""
Three changes in one patch:
1. Feature engineering: add 3 interaction features (AGE x DIAG_TE,
   ADMIT x SOURCE, hospital volume log) to push baseline accuracy higher.
2. Canonical XGBoost: bump n_estimators 1500 -> 3000, max_depth 10 -> 12,
   learning_rate 0.05 -> 0.03 (+ subsample/colsample) for better fit.
3. Phase 5b: seek all-4-DI >= 0.82 (margin); fall back to 0.80 only if
   0.82 not achievable. This gives F3's cross-hospital violins a buffer
   so DI Age doesn't appear borderline.
"""
import json
from pathlib import Path

NB = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\CIKM_2026_LOS_Fairness_FINAL.ipynb")

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)


# ──── PATCH 1: Feature engineering — add interaction features ────
OLD_FE = """feature_cols = low_card_cols + te_features
print(f"Final feature set ({len(feature_cols)}): {feature_cols}")"""

NEW_FE = """# Add interaction features to push baseline accuracy higher
# AGE x ADMITTING_DIAGNOSIS_te interaction
if "ADMITTING_DIAGNOSIS_te" in df.columns:
    df["AGE_X_DIAG_TE"] = (df["PAT_AGE"].astype("float32") *
                            df["ADMITTING_DIAGNOSIS_te"]).astype("float32")
# ADMIT_TYPE x ADMIT_SOURCE interaction
if "TYPE_OF_ADMISSION" in df.columns and "SOURCE_OF_ADMISSION" in df.columns:
    df["ADMIT_X_SOURCE"] = (df["TYPE_OF_ADMISSION"].astype("float32") * 10.0 +
                             df["SOURCE_OF_ADMISSION"].astype("float32")).astype("float32")
# Hospital volume (log)
if "THCIC_ID" in df.columns:
    hosp_vol_map = df.groupby("THCIC_ID").size()
    df["HOSP_VOLUME_LOG"] = np.log1p(df["THCIC_ID"].map(hosp_vol_map).fillna(0)).astype("float32")

interaction_features = [c for c in ["AGE_X_DIAG_TE", "ADMIT_X_SOURCE", "HOSP_VOLUME_LOG"]
                         if c in df.columns]
print(f"Interaction features added: {interaction_features}")

feature_cols = low_card_cols + te_features + interaction_features
print(f"Final feature set ({len(feature_cols)}): {feature_cols}")"""


# ──── PATCH 2: Bigger XGBoost ────
OLD_XGB = """    "XGBoost": xgb.XGBClassifier(n_estimators=1500, max_depth=10, learning_rate=0.05,"""
NEW_XGB = """    "XGBoost": xgb.XGBClassifier(n_estimators=3000, max_depth=12, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85,"""


# ──── PATCH 3: Phase 5b margin (0.82 with fallback to 0.80) ────
OLD_PHASE5B_PASS = """def _all4_pass(yp_check, ypb_check):
    di = _di_per_attr(yp_check, ypb_check)
    return all(v >= 0.80 for v in di.values()), di"""

NEW_PHASE5B_PASS = """def _all4_pass(yp_check, ypb_check, threshold=0.80):
    di = _di_per_attr(yp_check, ypb_check)
    return all(v >= threshold for v in di.values()), di"""

# We also want the relaxation loop to prefer 0.82 candidates. We modify
# the body to first try DI>=0.82, then fall back to 0.80.
OLD_PHASE5_BODY = """    n_iter = 0
    improved_total = 0
    # Cells sorted by current deviation from 0.5 (most aggressive first)
    while True:
        progressed = False
        # For each cell, try moving threshold 1pp closer to 0.5
        cells_sorted = sorted(refined.items(), key=lambda kv: -abs(kv[1] - 0.5))
        for cell_key, thr in cells_sorted:
            n_iter += 1
            if abs(thr - 0.5) < 0.005:
                continue
            step = 0.01 if thr > 0.5 else -0.01
            new_thr = thr - step  # move toward 0.5
            if (thr > 0.5 and new_thr < 0.5) or (thr < 0.5 and new_thr > 0.5):
                new_thr = 0.5
            new_thr = float(np.clip(new_thr, 0.01, 0.99))
            trial = dict(refined)
            trial[cell_key] = new_thr
            yp_trial = _apply_thresholds(ypb2, trial)
            ok, di_trial = _all4_pass(yp_trial, ypb2)"""

NEW_PHASE5_BODY = """    # Choose acceptance threshold: prefer 0.82 (margin) but accept 0.80 if needed
    cur_pass_82, _ = _all4_pass(yp_cur, ypb2, threshold=0.82)
    margin_threshold = 0.82 if cur_pass_82 else 0.80
    print(f"  margin threshold for relaxation: {margin_threshold}")

    n_iter = 0
    improved_total = 0
    # Cells sorted by current deviation from 0.5 (most aggressive first)
    while True:
        progressed = False
        # For each cell, try moving threshold 1pp closer to 0.5
        cells_sorted = sorted(refined.items(), key=lambda kv: -abs(kv[1] - 0.5))
        for cell_key, thr in cells_sorted:
            n_iter += 1
            if abs(thr - 0.5) < 0.005:
                continue
            step = 0.01 if thr > 0.5 else -0.01
            new_thr = thr - step  # move toward 0.5
            if (thr > 0.5 and new_thr < 0.5) or (thr < 0.5 and new_thr > 0.5):
                new_thr = 0.5
            new_thr = float(np.clip(new_thr, 0.01, 0.99))
            trial = dict(refined)
            trial[cell_key] = new_thr
            yp_trial = _apply_thresholds(ypb2, trial)
            ok, di_trial = _all4_pass(yp_trial, ypb2, threshold=margin_threshold)"""

# Same changes for Phase 5b's lambda=0 path
OLD_5B_BODY = """    refined_b = dict(thr_lam0)
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
            ok, di_trial = _all4_pass(yp_trial, ypb_lam0)"""

NEW_5B_BODY = """    # Choose acceptance threshold for relaxation: prefer 0.82, fall back to 0.80
    cur_pass_82_b, _ = _all4_pass(yp_lam0, ypb_lam0, threshold=0.82)
    margin_b = 0.82 if cur_pass_82_b else 0.80
    print(f"  margin threshold for 5b relaxation: {margin_b}")

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
            ok, di_trial = _all4_pass(yp_trial, ypb_lam0, threshold=margin_b)"""

# ──── Apply patches ────
n_changes = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c.get("source", []))
    new_src = src
    cell_changed = False

    if OLD_FE in new_src:
        new_src = new_src.replace(OLD_FE, NEW_FE)
        cell_changed = True
        print(f"Patch 1 (FE interactions) applied to cell {i}")
        n_changes += 1
    if OLD_XGB in new_src:
        new_src = new_src.replace(OLD_XGB, NEW_XGB)
        cell_changed = True
        print(f"Patch 2 (bigger XGB) applied to cell {i}")
        n_changes += 1
    if OLD_PHASE5B_PASS in new_src:
        new_src = new_src.replace(OLD_PHASE5B_PASS, NEW_PHASE5B_PASS)
        cell_changed = True
        print(f"Patch 3a (_all4_pass threshold arg) applied to cell {i}")
        n_changes += 1
    if OLD_PHASE5_BODY in new_src:
        new_src = new_src.replace(OLD_PHASE5_BODY, NEW_PHASE5_BODY)
        cell_changed = True
        print(f"Patch 3b (Phase 5 body margin) applied to cell {i}")
        n_changes += 1
    if OLD_5B_BODY in new_src:
        new_src = new_src.replace(OLD_5B_BODY, NEW_5B_BODY)
        cell_changed = True
        print(f"Patch 3c (Phase 5b body margin) applied to cell {i}")
        n_changes += 1

    if cell_changed:
        c["source"] = new_src.splitlines(keepends=True)
        c["outputs"] = []
        c["execution_count"] = None

# Clear all downstream cells
if n_changes > 0:
    # Find earliest changed cell
    earliest = None
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        if c.get("execution_count") is None and not c.get("outputs"):
            if earliest is None:
                earliest = i
                break
    if earliest is not None:
        for j in range(earliest + 1, len(nb["cells"])):
            if nb["cells"][j]["cell_type"] == "code":
                nb["cells"][j]["outputs"] = []
                nb["cells"][j]["execution_count"] = None

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"\nTotal changes: {n_changes}")
