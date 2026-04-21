"""Fair-pipeline cross-site validation, standalone runner.

Produces the same CSV outputs that the new notebook cell writes.  Exists
only so we can verify the logic without waiting for a full notebook run;
it is intentionally NOT committed to the repo.

Usage:  python _scripts/fair_cross_site.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = ROOT / "output" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

DATA_CANDIDATES = [
    ROOT / "../../../../data/texas_100x.csv",
    ROOT / "../../../data/texas_100x.csv",
    ROOT / "../../data/texas_100x.csv",
    ROOT / "../../final_analysis/data/texas_100x.csv",
    ROOT / "data/texas_100x.csv",
    ROOT / "../data/texas_100x.csv",
]

RANDOM_STATE = 42
K_CS = 20
FAIR_LAMBDA = 1.0
FAIR_A_SR = 0.8

METRIC_KEYS = ["DI", "SPD", "EOPP", "EOD", "TI", "PP", "CAL"]
ATTRS = ["RACE", "SEX", "ETHNICITY", "AGE_GROUP"]

THRESHOLDS = {
    "DI": 0.80, "SPD": 0.10, "EOPP": 0.10, "EOD": 0.10,
    "TI": 0.10, "PP": 0.10, "CAL": 0.05,
}


def find_data_path():
    for p in DATA_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("texas_100x.csv not found")


def is_fair(metric, value):
    if pd.isna(value):
        return False
    return (value >= THRESHOLDS["DI"]) if metric == "DI" else (abs(value) < THRESHOLDS[metric])


def create_age_groups(age_code):
    if age_code <= 4:
        return "Age_0_17"
    if age_code <= 9:
        return "Age_18_39"
    if age_code <= 12:
        return "Age_40_54"
    if age_code <= 14:
        return "Age_55_64"
    return "Age_65_Plus"


def load_data():
    path = find_data_path()
    print(f"Data: {path}")
    df = pd.read_csv(path)
    df["LOS_BINARY"] = (df["LENGTH_OF_STAY"] > 3).astype(int)
    df["AGE_GROUP"] = df["PAT_AGE"].apply(create_age_groups)
    protected_cols = ["RACE", "SEX_CODE", "ETHNICITY", "AGE_GROUP"]
    exclude_cols = ["LOS_BINARY", "LENGTH_OF_STAY", "THCIC_ID", "RECORD_ID"] + protected_cols
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ("int64", "float64", "object")]
    for col in feature_cols:
        if df[col].dtype == "object":
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    for col in feature_cols:
        if df[col].dtype == "int64":
            df[col] = df[col].astype("int32")
        elif df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
    X = df[feature_cols].fillna(0).values.astype("float32")
    y = df["LOS_BINARY"].values
    hospital_ids = df["THCIC_ID"].values
    idx_tr, idx_te = train_test_split(range(len(df)), test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    X_train = X[idx_tr]; X_test = X[idx_te]
    y_train = y[idx_tr]; y_test = y[idx_te]
    prot_tr = {}
    prot_te = {}
    for col in protected_cols:
        attr = col.replace("_CODE", "")
        enc = LabelEncoder().fit_transform(df[col].astype(str))
        prot_tr[attr] = enc[idx_tr]
        prot_te[attr] = enc[idx_te]
    hosp_tr = hospital_ids[idx_tr]
    hosp_te = hospital_ids[idx_te]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    hosp_all = np.concatenate([hosp_tr, hosp_te])
    prot_all = {a: np.concatenate([prot_tr[a], prot_te[a]]) for a in prot_tr}
    print(f"Rows: {len(X_all):,}  Hospitals: {len(set(hosp_all)):,}  Features: {X_all.shape[1]}")
    return X_all, y_all, hosp_all, prot_all


def fairness_metrics(y_true, y_pred, y_prob, protected):
    groups = np.unique(protected)
    rates = {}
    for g in groups:
        m = protected == g
        y_t = y_true[m]
        y_p = y_pred[m]
        sr = float(np.mean(y_p))
        tpr = float(np.mean(y_p[y_t == 1])) if (y_t == 1).any() else 0.0
        fpr = float(np.mean(y_p[y_t == 0])) if (y_t == 0).any() else 0.0
        ppv = float(np.mean(y_t[y_p == 1])) if (y_p == 1).any() else 0.0
        rates[g] = (sr, tpr, fpr, ppv, int(m.sum()))
    srs = [r[0] for r in rates.values()]
    tprs = [r[1] for r in rates.values()]
    fprs = [r[2] for r in rates.values()]
    ppvs = [r[3] for r in rates.values()]
    max_sr = max(srs)
    di = (min(srs) / max_sr) if max_sr > 0 else 1.0
    spd = max(srs) - min(srs)
    eopp = max(tprs) - min(tprs)
    eod = max(eopp, max(fprs) - min(fprs))
    pp = max(ppvs) - min(ppvs)
    # TI
    preds_per_group = [y_pred[protected == g][: min((protected == g).sum(), 5000)] for g in groups]
    min_len = min(len(p) for p in preds_per_group)
    ti_vals = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            ti_vals.append(float(np.mean(preds_per_group[i][:min_len] != preds_per_group[j][:min_len])))
    ti = float(np.mean(ti_vals)) if ti_vals else 0.0
    # Calibration
    cal_diffs = []
    for g in groups:
        m = protected == g
        p_g = y_prob[m]
        y_g = y_true[m]
        bins = np.linspace(0, 1, 11)
        for b in range(len(bins) - 1):
            in_bin = (p_g >= bins[b]) & (p_g < bins[b + 1])
            if in_bin.sum() >= 10:
                cal_diffs.append(abs(float(np.mean(y_g[in_bin])) - float(np.mean(p_g[in_bin]))))
    cal = max(cal_diffs) if cal_diffs else 0.0
    return {"DI": di, "SPD": spd, "EOPP": eopp, "EOD": eod, "TI": ti, "PP": pp, "CAL": cal}


def fair_multi_weights(race_tr, age_tr, sex_tr, y_tr, lam):
    key = np.array([f"{r}|{a}|{s}" for r, a, s in zip(race_tr, age_tr, sex_tr)])
    uniq = sorted(set(key))
    n = len(y_tr)
    sw = np.ones(n, dtype=float)
    total0 = int((y_tr == 0).sum())
    total1 = int((y_tr == 1).sum())
    for g in uniq:
        mg = key == g
        ng = int(mg.sum())
        for lab, total_lab in ((0, total0), (1, total1)):
            mgl = mg & (y_tr == lab)
            ngl = int(mgl.sum())
            if ngl > 0:
                expected = (ng / n) * (total_lab / n)
                observed = ngl / n
                raw_w = expected / observed if observed > 0 else 1.0
                sw[mgl] = np.clip(1.0 + lam * (raw_w - 1.0), 0.1, 10.0)
    return sw


def find_sr_threshold(probs, target, lo=0.01, hi=0.99, step=0.01):
    best_t = 0.5
    best_d = abs(float((probs >= 0.5).mean()) - target)
    for t in np.arange(lo, hi, step):
        d = abs(float((probs >= t).mean()) - target)
        if d < best_d:
            best_d, best_t = d, float(t)
    return best_t


def run(X_all, y_all, hosp_all, prot_all):
    print(f"Fair Cross-Site Portability: K={K_CS} GroupKFold  (lam={FAIR_LAMBDA}, a_SR={FAIR_A_SR})")
    gkf = GroupKFold(n_splits=K_CS)
    rows = []
    t0 = time.time()
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_all, y_all, groups=hosp_all)):
        X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
        race_tr = prot_all["RACE"][tr_idx]
        age_tr = prot_all["AGE_GROUP"][tr_idx]
        sex_tr = prot_all["SEX"][tr_idx]
        sw = fair_multi_weights(race_tr, age_tr, sex_tr, y_tr, FAIR_LAMBDA)

        mdl = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=8,
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=-1,
        )
        mdl.fit(X_tr, y_tr, sample_weight=sw)

        X_val, y_val = X_all[val_idx], y_all[val_idx]
        race_val = prot_all["RACE"][val_idx]
        age_val = prot_all["AGE_GROUP"][val_idx]
        sex_val = prot_all["SEX"][val_idx]
        y_prob = mdl.predict_proba(X_val)[:, 1]

        vkeys = np.array([f"{r}|{a}|{s}" for r, a, s in zip(race_val, age_val, sex_val)])
        overall_sr = float((y_prob >= 0.5).mean())
        y_pred = (y_prob >= 0.5).astype(int)
        for g in set(vkeys):
            mask = vkeys == g
            if int(mask.sum()) < 5:
                continue
            t_sr = find_sr_threshold(y_prob[mask], overall_sr)
            t = float(np.clip(0.5 + FAIR_A_SR * (t_sr - 0.5), 0.01, 0.99))
            y_pred[mask] = (y_prob[mask] >= t).astype(int)

        row = {
            "Fold": fold + 1,
            "N_val": int(len(val_idx)),
            "N_hospitals": len(set(hosp_all[val_idx])),
            "Acc": float(accuracy_score(y_val, y_pred)),
            "AUC": float(roc_auc_score(y_val, y_prob)) if len(set(y_val)) > 1 else np.nan,
        }
        for attr in ATTRS:
            attr_val = prot_all[attr][val_idx]
            if len(set(attr_val)) >= 2:
                mc = fairness_metrics(y_val, y_pred, y_prob, attr_val)
                for mk in METRIC_KEYS:
                    row[f"{mk}_{attr}"] = mc[mk]
                    row[f"V_{mk}_{attr}"] = 1 if is_fair(mk, mc[mk]) else 0
            else:
                for mk in METRIC_KEYS:
                    row[f"{mk}_{attr}"] = np.nan
                    row[f"V_{mk}_{attr}"] = np.nan
        rows.append(row)
        if (fold + 1) % 2 == 0:
            print(
                f"  Fold {fold+1}/{K_CS}: Acc={row['Acc']:.4f} "
                f"DI: R={row['DI_RACE']:.3f} S={row['DI_SEX']:.3f} "
                f"E={row['DI_ETHNICITY']:.3f} A={row['DI_AGE_GROUP']:.3f}  "
                f"t+={time.time()-t0:.0f}s"
            )
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "cikm_cross_site_portability_FAIR.csv", index=False)
    print(f"Completed in {time.time()-t0:.1f}s  -> cikm_cross_site_portability_FAIR.csv")
    return df


def build_tables(fair_df):
    # Per-cluster (80 rows)
    rows = []
    for _, r in fair_df.iterrows():
        for attr in ATTRS:
            e = {
                "Cluster": int(r["Fold"]),
                "N_hosp": int(r["N_hospitals"]),
                "Acc": round(r["Acc"], 4),
                "AUC": round(r["AUC"], 4),
                "Attribute": attr,
            }
            nf = 0
            for m in METRIC_KEYS:
                v = r.get(f"{m}_{attr}", np.nan)
                e[m] = round(v, 4) if pd.notna(v) else np.nan
                nf += int(is_fair(m, v))
            e["N_fair"] = f"{nf}/7"
            rows.append(e)
    pc = pd.DataFrame(rows)
    pc.to_csv(TABLES_DIR / "Table6Fair_CrossSite_PerCluster.csv", index=False)

    # Per-metric mean + verdict
    rows = []
    for attr in ATTRS:
        for m in METRIC_KEYS:
            vals = fair_df[f"{m}_{attr}"].dropna()
            if len(vals) == 0:
                continue
            mean_v = float(vals.mean())
            k = int(sum(is_fair(m, v) for v in vals))
            rows.append({
                "Attribute": attr,
                "Metric": m,
                "Mean": round(mean_v, 4),
                "SD": round(float(vals.std(ddof=1)), 4),
                "Threshold": THRESHOLDS[m],
                "Fair_at_mean": "Pass" if is_fair(m, mean_v) else "Fail",
                "Pass_k_over_N": f"{k}/{len(vals)}",
                "Pass_pct": round(100 * k / len(vals), 1),
            })
    mv = pd.DataFrame(rows)
    mv.to_csv(TABLES_DIR / "Table6Fair_CrossSite_MetricVerdicts.csv", index=False)

    # Std vs Fair totals (reload Standard counterparts)
    std_mv_path = TABLES_DIR / "Table6b_CrossSite_MetricVerdicts.csv"
    std_mv = pd.read_csv(std_mv_path) if std_mv_path.exists() else None
    rows = []
    for attr in ATTRS:
        fair_sub = mv[mv["Attribute"] == attr]
        fair_n = int((fair_sub["Fair_at_mean"] == "Pass").sum())
        fair_di = float(fair_sub[fair_sub["Metric"] == "DI"]["Mean"].iloc[0])
        std_n = None
        std_di = None
        if std_mv is not None:
            ssub = std_mv[std_mv["Attribute"] == attr]
            std_n = int((ssub["Fair_at_mean"] == "Pass").sum())
            std_di = float(ssub[ssub["Metric"] == "DI"]["Mean"].iloc[0])
        rows.append({
            "Attribute": attr,
            "Standard_N_fair_at_mean": f"{std_n}/7" if std_n is not None else "—",
            "Fair_N_fair_at_mean": f"{fair_n}/7",
            "Delta": fair_n - (std_n if std_n is not None else 0),
            "Standard_DI_mean": round(std_di, 3) if std_di is not None else None,
            "Fair_DI_mean": round(fair_di, 3),
        })
    totals = pd.DataFrame(rows)
    totals.to_csv(TABLES_DIR / "Table6Fair_StdVsFair_Totals.csv", index=False)
    print("\nAttribute totals (Standard vs Fair, cross-site mean):")
    print(totals.to_string(index=False))


def main():
    X_all, y_all, hosp_all, prot_all = load_data()
    fair_df = run(X_all, y_all, hosp_all, prot_all)
    build_tables(fair_df)


if __name__ == "__main__":
    main()
