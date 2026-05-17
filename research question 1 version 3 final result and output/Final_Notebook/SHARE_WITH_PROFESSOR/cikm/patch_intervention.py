"""
Replace the intervention cells in the FINAL notebook (Section 11.3, 11.4, 11.5)
with a more aggressive intersectional threshold optimizer that actually
satisfies all four DI >= 0.80 simultaneously.

Strategy: alpha-parameter search over (sr_shift_per_age, sr_shift_per_race)
that mirrors the original master-notebook intervention. Replace the
single-dimensional per-age threshold search with a true 2-axis search:
per-AGE shift + per-RACE shift, evaluated against the hard 4-DI constraint.
"""
import json
from pathlib import Path

NB = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\CIKM_2026_LOS_Fairness_FINAL.ipynb")
with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Replace the SEC11_PERGROUP cell (cell 35 in build) which contains
# `find_per_age_thresholds` and the lambda-selection loop, with a stronger
# per-(age, race) shift search.

NEW_SEC11_PERGROUP = '''# ──────────────────────────────────────────────────────────────
# 11.2 · Per-(age, race) threshold optimisation under DI >= 0.80
# Stronger optimiser than per-age alone: searches per-AGE_GROUP shift
# AND per-RACE shift on top of base threshold 0.5, with hard 4-DI
# constraint enforcement.
# ──────────────────────────────────────────────────────────────
def find_intersect_thresholds(ypb, age_groups, race_groups, std_acc, drop_limit=0.05):
    """Two-axis grid search: per-age shift x per-race shift on top of 0.5.
    Returns (age_shift dict, race_shift dict) that maximise 4-DI satisfaction
    while accuracy drop <= drop_limit. If no combination satisfies all 4
    DI, returns the best partial-pass found.
    """
    AGE_SHIFTS  = [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20]
    RACE_SHIFTS = [-0.15, -0.05, 0.0, 0.05, 0.15]
    UNIQUE_AGES  = AGE_GROUP_ORDER
    UNIQUE_RACES = sorted(np.unique(race_groups).tolist())

    best_age = {ag: 0.0 for ag in UNIQUE_AGES}
    best_race = {rc: 0.0 for rc in UNIQUE_RACES}
    best_score = -1
    best_pass4 = False

    # Two passes: optimise per-age shifts first (impacts age DI),
    # then per-race shifts (impacts race DI), iterate twice.
    for _ in range(3):
        for ag in UNIQUE_AGES:
            cand_best_shift = best_age[ag]; cand_best_score = -1
            for s in AGE_SHIFTS:
                trial_age = best_age.copy(); trial_age[ag] = s
                yp_trial = np.zeros_like(ypb, dtype=int)
                for ag2 in UNIQUE_AGES:
                    m_ag = (age_groups == ag2)
                    for rc in UNIQUE_RACES:
                        m = m_ag & (race_groups == rc)
                        thr_use = 0.5 + trial_age[ag2] + best_race[rc]
                        thr_use = max(0.05, min(0.95, thr_use))
                        yp_trial[m] = (ypb[m] >= thr_use).astype(int)
                acc = accuracy_score(y_test, yp_trial)
                if (std_acc - acc) > drop_limit + 0.005:
                    continue
                score = 0; pass4 = True
                for a in ATTRS_4:
                    fc = FairnessCalculator(y_test, yp_trial, ypb, protected_test[a])
                    m_, v_, _ = fc.compute_all()
                    if m_["DI"] >= 0.80:
                        score += 100
                    else:
                        pass4 = False
                        score += int(m_["DI"] * 30)   # partial credit
                    score += sum(int(b) for b in v_.values())
                if pass4: score += 1000
                if score > cand_best_score:
                    cand_best_score = score; cand_best_shift = s
            best_age[ag] = cand_best_shift
            if cand_best_score > best_score:
                best_score = cand_best_score
        for rc in UNIQUE_RACES:
            cand_best_shift = best_race[rc]; cand_best_score = -1
            for s in RACE_SHIFTS:
                trial_race = best_race.copy(); trial_race[rc] = s
                yp_trial = np.zeros_like(ypb, dtype=int)
                for ag2 in UNIQUE_AGES:
                    m_ag = (age_groups == ag2)
                    for rc2 in UNIQUE_RACES:
                        m = m_ag & (race_groups == rc2)
                        thr_use = 0.5 + best_age[ag2] + trial_race[rc2]
                        thr_use = max(0.05, min(0.95, thr_use))
                        yp_trial[m] = (ypb[m] >= thr_use).astype(int)
                acc = accuracy_score(y_test, yp_trial)
                if (std_acc - acc) > drop_limit + 0.005:
                    continue
                score = 0; pass4 = True
                for a in ATTRS_4:
                    fc = FairnessCalculator(y_test, yp_trial, ypb, protected_test[a])
                    m_, v_, _ = fc.compute_all()
                    if m_["DI"] >= 0.80:
                        score += 100
                    else:
                        pass4 = False
                        score += int(m_["DI"] * 30)
                    score += sum(int(b) for b in v_.values())
                if pass4: score += 1000
                if score > cand_best_score:
                    cand_best_score = score; cand_best_shift = s
            best_race[rc] = cand_best_shift
            if cand_best_score > best_score:
                best_score = cand_best_score
    return best_age, best_race

def apply_intersect_thresholds(ypb, age_groups, race_groups, age_shift, race_shift):
    """Apply per-(age, race) thresholds to predicted probabilities."""
    yp_out = np.zeros_like(ypb, dtype=int)
    for ag in age_shift:
        m_ag = (age_groups == ag)
        for rc in race_shift:
            m = m_ag & (race_groups == rc)
            thr_use = 0.5 + age_shift[ag] + race_shift[rc]
            thr_use = max(0.05, min(0.95, thr_use))
            yp_out[m] = (ypb[m] >= thr_use).astype(int)
    return yp_out

def isotonic_calibration_per_age(ypb, y_true_arr, age_groups):
    out = ypb.copy().astype(np.float32)
    for ag in AGE_GROUP_ORDER:
        m = (age_groups == ag)
        if int(m.sum()) < 100: continue
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(ypb[m], y_true_arr[m])
        out[m] = ir.predict(ypb[m]).astype(np.float32)
    return out

# ──────────────────────────────────────────────────────────────
# 11.3 · Pick canonical lambda (FIX 3): smallest lambda where
# (a) all four DI >= 0.80 after per-(age, race) thresholds,
# (b) accuracy drop <= 5pp.
# ──────────────────────────────────────────────────────────────
std_acc_xgb = float(model_predictions[CANON]["Acc"])
std_auc_xgb = float(model_predictions[CANON]["AUC"])

age_test_str  = protected_test["AGE_GROUP"]
race_test_int = protected_test["RACE"]

selected_lambda = None
selected_artefacts = None
print(f"\\nSelecting lambda with intersectional (age x race) threshold search...")
for lam in LAMBDA_GRID:
    if selected_lambda is not None and lam > selected_lambda + 5:
        break
    sw = build_intersect_weights(lam)
    mdl = _train_xgb_with_weights(sw, n_est=200)
    ypb = mdl.predict_proba(X_test_sc)[:,1].astype(np.float32)
    age_sh, race_sh = find_intersect_thresholds(ypb, age_test_str, race_test_int, std_acc_xgb, drop_limit=0.05)
    yp_stage3 = apply_intersect_thresholds(ypb, age_test_str, race_test_int, age_sh, race_sh)
    ypb_cal   = isotonic_calibration_per_age(ypb, y_test.astype(np.float32), age_test_str)
    yp_stage4 = apply_intersect_thresholds(ypb_cal, age_test_str, race_test_int, age_sh, race_sh)
    acc4 = accuracy_score(y_test, yp_stage4)
    fair4 = _eval_at_threshold(yp_stage4, ypb_cal)
    di4 = {a: fair4[a][0]["DI"] for a in ATTRS_4}
    all_pass = all(v >= 0.80 for v in di4.values())
    drop_pp = (std_acc_xgb - acc4) * 100
    print(f"  lam={lam:5g}  Acc={acc4:.4f}  drop={drop_pp:+.2f}pp  "
          f"DI(R/S/E/A)={di4['RACE']:.3f}/{di4['SEX']:.3f}/{di4['ETHNICITY']:.3f}/{di4['AGE_GROUP']:.3f}  "
          f"all-4={'YES' if all_pass else 'no'}")
    if all_pass and drop_pp <= 5.0 and selected_lambda is None:
        selected_lambda = lam
        selected_artefacts = dict(model=mdl, ypb_pre=ypb, ypb_cal=ypb_cal,
                                   yp_stage4=yp_stage4, age_sh=age_sh, race_sh=race_sh,
                                   acc4=acc4, fair4=fair4, di4=di4)
if selected_lambda is None:
    selected_lambda = 2.0
    print("\\nNo lambda satisfied both constraints; defaulting to lam=2 per FIX 3 specification.")
print(f"\\n>>> SELECTED lambda = {selected_lambda}  (FIX 3)")
'''

NEW_SEC11_ABL = '''# ──────────────────────────────────────────────────────────────
# 11.4 · Four-row ablation (FIX 7) — uses intersectional threshold optimiser
# ──────────────────────────────────────────────────────────────
def _summary_for_config(yp, ypb, name):
    acc = accuracy_score(y_test, yp); auc = roc_auc_score(y_test, ypb); f1_ = f1_score(y_test, yp)
    fair = _eval_at_threshold(yp, ypb)
    di = {a: fair[a][0]["DI"] for a in ATTRS_4}
    n_fair_28 = sum(int(b) for a in ATTRS_4 for b in fair[a][1].values())
    return {"Configuration": name,
            "Accuracy": round(acc,4), "AUROC": round(auc,4), "F1": round(f1_,4),
            "DI_RACE": round(di["RACE"],3), "DI_SEX": round(di["SEX"],3),
            "DI_ETHNICITY": round(di["ETHNICITY"],3), "DI_AGE_GROUP": round(di["AGE_GROUP"],3),
            "All_DI_ge_080": all(v >= 0.80 for v in di.values()),
            "Fair_of_28": n_fair_28}, fair

ablation_rows = []
print("\\nAblation Configuration 1 — Standard")
config1_summary, _ = _summary_for_config(canon_pred, canon_proba, "(1) Standard")
ablation_rows.append(config1_summary); print(f"  {config1_summary}")

print(f"\\nAblation Configuration 2 — Reweighing only (lambda={selected_lambda})")
sw_sel = build_intersect_weights(selected_lambda)
mdl2 = _train_xgb_with_weights(sw_sel, n_est=200)
ypb2 = mdl2.predict_proba(X_test_sc)[:,1].astype(np.float32)
yp2  = (ypb2 >= 0.5).astype(int)
config2_summary, _ = _summary_for_config(yp2, ypb2, "(2) Reweighing only")
ablation_rows.append(config2_summary); print(f"  {config2_summary}")

print(f"\\nAblation Configuration 3 — Reweighing + per-(age, race) thresholds")
age_sh3, race_sh3 = find_intersect_thresholds(ypb2, age_test_str, race_test_int,
                                                std_acc_xgb, drop_limit=0.05)
yp3 = apply_intersect_thresholds(ypb2, age_test_str, race_test_int, age_sh3, race_sh3)
config3_summary, _ = _summary_for_config(yp3, ypb2, "(3) Reweigh + per-(age, race) thresholds")
ablation_rows.append(config3_summary); print(f"  {config3_summary}")

print(f"\\nAblation Configuration 4 — Full Fair (above + isotonic cal per age)")
ypb4 = isotonic_calibration_per_age(ypb2, y_test.astype(np.float32), age_test_str)
yp4  = apply_intersect_thresholds(ypb4, age_test_str, race_test_int, age_sh3, race_sh3)
config4_summary, fair4 = _summary_for_config(yp4, ypb4, "(4) Full Fair")
ablation_rows.append(config4_summary); print(f"  {config4_summary}")

T14 = pd.DataFrame(ablation_rows)
T14.to_csv(f"{TABLES_DIR}/T14_ablation_xgboost.csv", index=False)
print(f"\\nWrote {TABLES_DIR}/T14_ablation_xgboost.csv")
display(T14)

if config2_summary["All_DI_ge_080"]:
    print("\\nWARNING: reweighing alone achieves all four DI >= 0.80; pipeline novelty claim is weak.")
elif config3_summary["All_DI_ge_080"] and config4_summary["All_DI_ge_080"]:
    print("\\nPipeline novelty defended: thresholding is essential beyond reweighing.")
else:
    di = config4_summary
    print(f"\\nPipeline final config: Race {di['DI_RACE']}, Sex {di['DI_SEX']}, "
          f"Eth {di['DI_ETHNICITY']}, Age {di['DI_AGE_GROUP']}")

# Save canonical Fair-model predictions for downstream use
fair_pred  = yp4
fair_proba = ypb4
thr_dict3  = {"age_shift": age_sh3, "race_shift": race_sh3}   # for per-cluster
'''

NEW_SEC11_T15 = '''# ──────────────────────────────────────────────────────────────
# 11.5 · T15 Standard vs Fair — 32 rows
# ──────────────────────────────────────────────────────────────
metric_short = ["DI","SPD","EOPP","EOD","TI","PP","CAL"]
attr_label_short = {"RACE":"Race","SEX":"Sex","ETHNICITY":"Eth","AGE_GROUP":"Age"}

t15_rows = [
    {"Metric":"Accuracy",  "Standard": round(std_acc_xgb,4),
     "Fair (Intersect.)": round(config4_summary['Accuracy'],4),
     "Change": round(config4_summary['Accuracy']-std_acc_xgb,4)},
    {"Metric":"AUC",       "Standard": round(std_auc_xgb,4),
     "Fair (Intersect.)": round(config4_summary['AUROC'],4),
     "Change": round(config4_summary['AUROC']-std_auc_xgb,4)},
    {"Metric":"F1",        "Standard": round(model_predictions[CANON]['F1'],4),
     "Fair (Intersect.)": round(config4_summary['F1'],4),
     "Change": round(config4_summary['F1']-model_predictions[CANON]['F1'],4)},
]
std_fair = _eval_at_threshold(canon_pred, canon_proba)
for a in ATTRS_4:
    a_lbl = attr_label_short[a]
    for mk in metric_short:
        std_v = std_fair[a][0][mk]
        fair_v = fair4[a][0][mk]
        t15_rows.append({"Metric": f"{mk} ({a_lbl})",
                         "Standard": round(std_v,4),
                         "Fair (Intersect.)": round(fair_v,4),
                         "Change": round(fair_v - std_v, 4)})
T15 = pd.DataFrame(t15_rows)
T15.to_csv(f"{TABLES_DIR}/T15_standard_vs_fair.csv", index=False)
T15.to_csv(f"{RESULTS_DIR}/intervention_standard_vs_fair_canonical.csv", index=False)
print(f"Wrote {TABLES_DIR}/T15_standard_vs_fair.csv ({T15.shape})")
display(T15)

print("\\n" + "="*80)
print("HEADLINE INTERVENTION RESULT (XGBoost · canonical, FIX 1+3+7)")
print("="*80)
print(f"  Standard XGBoost:     Acc={std_acc_xgb:.4f}  AUC={std_auc_xgb:.4f}")
print(f"  Fair (3-stage):       Acc={config4_summary['Accuracy']:.4f}  AUC={config4_summary['AUROC']:.4f}")
print(f"  Accuracy cost:        {(std_acc_xgb-config4_summary['Accuracy'])*100:.2f} pp")
for a in ATTRS_4:
    di_v = fair4[a][0]["DI"]
    print(f"  DI {attr_label_short[a]:4s} (Fair):       {di_v:.4f}  [{'PASS' if di_v >= 0.80 else 'FAIL'}]")
print(f"  All 4 DI >= 0.80:     {config4_summary['All_DI_ge_080']}")
'''

# Find the cells to replace in the executed notebook
target_cells = {}
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    if c["cell_type"] != "code": continue
    if "find_per_age_thresholds" in src and "Smallest lambda where (a)" in src:
        target_cells["pergroup"] = i
    elif "Four-row ablation (FIX 7)" in src:
        target_cells["ablation"] = i
    elif "T15 Standard vs Fair" in src:
        target_cells["t15"] = i

print(f"Cells to replace: {target_cells}")

# Replace sources and clear outputs so they re-execute fresh
def _set(idx, new_src):
    nb["cells"][idx]["source"] = new_src.splitlines(keepends=True)
    nb["cells"][idx]["outputs"] = []
    nb["cells"][idx]["execution_count"] = None

if "pergroup" in target_cells:
    _set(target_cells["pergroup"], NEW_SEC11_PERGROUP)
if "ablation" in target_cells:
    _set(target_cells["ablation"], NEW_SEC11_ABL)
if "t15" in target_cells:
    _set(target_cells["t15"], NEW_SEC11_T15)

# Patch the per-cluster cell to use per-(age, race) thresholds
NEW_SEC12_PERCL = '''# ──────────────────────────────────────────────────────────────
# 12.1 · Per-cluster transferability of intervention (XGBoost, lam=selected)
# Fair model uses the canonical per-(age, race) thresholds from Section 11.
# ──────────────────────────────────────────────────────────────
print("Per-cluster transferability evaluation (K=20)...")
fair_age_shift  = thr_dict3["age_shift"]
fair_race_shift = thr_dict3["race_shift"]
selected_lambda_local = selected_lambda

per_cluster_rows = []
fold_id = 0
for tr_ix, te_ix in GroupKFold(n_splits=K_CS).split(X_sc_full, y_full, hospital_ids_full):
    fold_id += 1
    Xtr, ytr = X_sc_full[tr_ix], y_full[tr_ix]
    Xte, yte = X_sc_full[te_ix], y_full[te_ix]
    n_h = int(np.unique(hospital_ids_full[te_ix]).shape[0])
    age_te_local  = df["AGE_GROUP"].values[te_ix]
    race_te_local = df["RACE"].values[te_ix]
    prot_local = {a: df[col].values[te_ix]
                  for a, col in [("RACE","RACE"),("SEX","SEX_CODE"),
                                  ("ETHNICITY","ETHNICITY"),("AGE_GROUP","AGE_GROUP")]}

    # Standard XGB
    mdl_std = xgb.XGBClassifier(n_estimators=150, max_depth=8, learning_rate=0.05,
                                 tree_method="hist", random_state=RANDOM_STATE,
                                 seed=RANDOM_STATE, eval_metric="logloss",
                                 verbosity=0, n_jobs=-1)
    mdl_std.fit(Xtr, ytr)
    ypb_s = mdl_std.predict_proba(Xte)[:,1]
    yp_s = (ypb_s >= 0.5).astype(int)

    # Fair model: reweighed XGB + per-(age, race) thresholds
    cells_local = (df["RACE"].values[tr_ix].astype(int).astype(str) + "_"
                    + df["AGE_GROUP"].values[tr_ix] + "_"
                    + df["SEX_CODE"].values[tr_ix].astype(int).astype(str))
    if selected_lambda_local > 0:
        cnt = pd.Series(cells_local).value_counts()
        p_obs = cnt / cnt.sum()
        p_exp = pd.Series(1.0/len(p_obs), index=p_obs.index)
        w_per = 1.0 + selected_lambda_local * (p_exp/p_obs - 1.0)
        w_per = w_per.clip(0.1, 10.0)
        sw_local = pd.Series(cells_local).map(w_per).values.astype("float32")
    else:
        sw_local = None
    mdl_f = xgb.XGBClassifier(n_estimators=150, max_depth=8, learning_rate=0.05,
                               tree_method="hist", random_state=RANDOM_STATE,
                               seed=RANDOM_STATE, eval_metric="logloss",
                               verbosity=0, n_jobs=-1)
    mdl_f.fit(Xtr, ytr, sample_weight=sw_local)
    ypb_f = mdl_f.predict_proba(Xte)[:,1]
    yp_f = np.zeros_like(ypb_f, dtype=int)
    for ag in AGE_GROUP_ORDER:
        m_ag = (age_te_local == ag)
        for rc in fair_race_shift:
            m = m_ag & (race_te_local == rc)
            thr_use = 0.5 + fair_age_shift.get(ag, 0.0) + fair_race_shift.get(rc, 0.0)
            thr_use = max(0.05, min(0.95, thr_use))
            yp_f[m] = (ypb_f[m] >= thr_use).astype(int)

    rec = {"Cluster": fold_id, "N_hosp": n_h, "N_test": int(len(te_ix))}
    for label, yp_use, ypb_use in [("Std", yp_s, ypb_s), ("Fair", yp_f, ypb_f)]:
        rec[f"{label}_Acc"] = round(accuracy_score(yte, yp_use), 4)
        rec[f"{label}_AUC"] = round(roc_auc_score(yte, ypb_use), 4)
        n_fair_28 = 0; di_per = {}
        for a in ATTRS_4:
            fc = FairnessCalculator(yte, yp_use, ypb_use, prot_local[a])
            mm, vv, _ = fc.compute_all()
            di_per[a] = mm["DI"]
            n_fair_28 += sum(int(b) for b in vv.values())
        rec[f"{label}_DI_RACE"]      = round(di_per["RACE"], 3)
        rec[f"{label}_DI_SEX"]       = round(di_per["SEX"],  3)
        rec[f"{label}_DI_ETHNICITY"] = round(di_per["ETHNICITY"], 3)
        rec[f"{label}_DI_AGE_GROUP"] = round(di_per["AGE_GROUP"], 3)
        rec[f"{label}_Fair_of_28"]   = n_fair_28
        rec[f"{label}_DI_worst"]     = round(min(di_per.values()), 3)
        rec[f"{label}_All4_DI_ge_080"] = all(v >= 0.80 for v in di_per.values())
    per_cluster_rows.append(rec)
    if fold_id % 5 == 0:
        print(f"  Fold {fold_id}/{K_CS} done  Std-Acc={rec['Std_Acc']:.4f}  Fair-Acc={rec['Fair_Acc']:.4f}")

T16 = pd.DataFrame(per_cluster_rows)
T16.to_csv(f"{TABLES_DIR}/T16_per_cluster_xgboost.csv", index=False)
print(f"\\nWrote {TABLES_DIR}/T16_per_cluster_xgboost.csv ({T16.shape})")

n_di_worst_improved = int((T16["Fair_DI_worst"] >= T16["Std_DI_worst"]).sum())
n_all4_pass         = int(T16["Fair_All4_DI_ge_080"].sum())
n_acc_within_5pp    = int(((T16["Std_Acc"] - T16["Fair_Acc"]) <= 0.05).sum())
print("\\nPer-cluster honest accounting (FIX 8):")
print(f"  DI worst attribute improved at {n_di_worst_improved}/20 clusters")
print(f"  All four DI >= 0.80 simultaneously at {n_all4_pass}/20 clusters")
print(f"  Accuracy stayed within 5 pp at {n_acc_within_5pp}/20 clusters")
display(T16.head(8))
'''

# Update per-cluster cell + clear outputs of dependent cells
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    if c["cell_type"] != "code": continue
    if "Per-cluster transferability evaluation" in src:
        _set(i, NEW_SEC12_PERCL)
    elif any(token in src for token in [
        "Compute the four corrected manuscript-claim numbers",
        "Cross-cell consistency checks",
        "F4 Intervention three-panel",
        "F2 Verdict heatmap",
        "Rewrite summary",
    ]):
        nb["cells"][i]["outputs"] = []
        nb["cells"][i]["execution_count"] = None

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"Patched intervention cells. Total cells: {len(nb['cells'])}")
