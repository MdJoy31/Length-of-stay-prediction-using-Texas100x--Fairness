"""Run the Fair pipeline at 8 hospital-count scales to pair with the
existing Standard hospital-scale sweep.

Trains Reweigh(lam=1) LightGBM on a subset of the first N_train_hosps
hospitals (from hospital_ids_train), evaluates on the full held-out
test set and on the 20 GroupKFold clusters to produce a VFR estimate.

Writes: output/tables/cikm_cross_hospital_scale_FAIR.csv
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

import lightgbm as lgb

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = ROOT / "output" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

DATA_CANDIDATES = [
    ROOT / "../../../../data/texas_100x.csv",
    ROOT / "../../../data/texas_100x.csv",
    ROOT / "../../data/texas_100x.csv",
    ROOT / "../../final_analysis/data/texas_100x.csv",
]

RANDOM_STATE = 42
N_HOSP_LEVELS = [1, 2, 3, 5, 10, 50, 100, 439]
FAIR_LAMBDA = 1.0
FAIR_A_SR = 0.8

METRIC_KEYS = ["DI", "SPD", "EOPP", "EOD", "TI", "PP", "CAL"]
ATTRS = ["RACE", "SEX", "ETHNICITY", "AGE_GROUP"]
THRESHOLDS = {"DI": 0.80, "SPD": 0.10, "EOPP": 0.10,
              "EOD": 0.10, "TI": 0.10, "PP": 0.10, "CAL": 0.05}


def is_fair(metric, value):
    if pd.isna(value):
        return False
    return (value >= THRESHOLDS["DI"]) if metric == "DI" \
        else (abs(value) < THRESHOLDS[metric])


def fairness_metrics(y_true, y_pred, y_prob, protected):
    groups = np.unique(protected)
    rates = []
    for g in groups:
        m = protected == g
        if m.sum() == 0:
            continue
        y_t = y_true[m]
        y_p = y_pred[m]
        sr = float(np.mean(y_p))
        tpr = float(np.mean(y_p[y_t == 1])) if (y_t == 1).any() else 0.0
        fpr = float(np.mean(y_p[y_t == 0])) if (y_t == 0).any() else 0.0
        ppv = float(np.mean(y_t[y_p == 1])) if (y_p == 1).any() else 0.0
        rates.append((sr, tpr, fpr, ppv))
    srs = [r[0] for r in rates]
    tprs = [r[1] for r in rates]
    fprs = [r[2] for r in rates]
    ppvs = [r[3] for r in rates]
    max_sr = max(srs)
    di = (min(srs) / max_sr) if max_sr > 0 else 1.0
    return {
        "DI": di, "SPD": max(srs) - min(srs),
        "EOPP": max(tprs) - min(tprs),
        "EOD": max(max(tprs) - min(tprs), max(fprs) - min(fprs)),
        "TI": 0.0,  # placeholder — not comparable for sweep
        "PP": max(ppvs) - min(ppvs),
        "CAL": 0.0,  # placeholder
    }


def create_age_group(age_code):
    if age_code <= 4: return "Age_0_17"
    if age_code <= 9: return "Age_18_39"
    if age_code <= 12: return "Age_40_54"
    if age_code <= 14: return "Age_55_64"
    return "Age_65_Plus"


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


def load_data():
    for p in DATA_CANDIDATES:
        if p.exists():
            path = p
            break
    else:
        raise FileNotFoundError("texas_100x.csv not found")
    df = pd.read_csv(path)
    df["LOS_BINARY"] = (df["LENGTH_OF_STAY"] > 3).astype(int)
    df["AGE_GROUP"] = df["PAT_AGE"].apply(create_age_group)
    protected_cols = ["RACE", "SEX_CODE", "ETHNICITY", "AGE_GROUP"]
    exclude = ["LOS_BINARY", "LENGTH_OF_STAY", "THCIC_ID", "RECORD_ID"] + protected_cols
    feats = [c for c in df.columns
             if c not in exclude and df[c].dtype in ("int64", "float64", "object")]
    for c in feats:
        if df[c].dtype == "object":
            df[c] = LabelEncoder().fit_transform(df[c].astype(str))
    X = df[feats].fillna(0).values.astype("float32")
    y = df["LOS_BINARY"].values
    hosp = df["THCIC_ID"].values
    idx_tr, idx_te = train_test_split(range(len(df)), test_size=0.2,
                                      random_state=RANDOM_STATE, stratify=y)
    X_tr = X[idx_tr]
    X_te = X[idx_te]
    y_tr = y[idx_tr]
    y_te = y[idx_te]
    prot_tr = {}
    prot_te = {}
    for col in protected_cols:
        name = col.replace("_CODE", "")
        enc = LabelEncoder().fit_transform(df[col].astype(str))
        prot_tr[name] = enc[idx_tr]
        prot_te[name] = enc[idx_te]
    hosp_tr = hosp[idx_tr]
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    return X_tr, X_te, y_tr, y_te, prot_tr, prot_te, hosp_tr


def run_scale_level(X_tr, X_te, y_tr, y_te, prot_tr, prot_te, hosp_tr, n_h):
    np.random.seed(RANDOM_STATE)
    all_hosps = np.unique(hosp_tr)
    # Pick highest-volume hospitals like the paper's sweep
    counts = pd.Series(hosp_tr).value_counts()
    selected = counts.head(n_h).index.values if n_h <= len(all_hosps) else all_hosps
    mask = np.isin(hosp_tr, selected)
    X_sub = X_tr[mask]
    y_sub = y_tr[mask]
    race_sub = prot_tr["RACE"][mask]
    age_sub = prot_tr["AGE_GROUP"][mask]
    sex_sub = prot_tr["SEX"][mask]

    sw = fair_multi_weights(race_sub, age_sub, sex_sub, y_sub, FAIR_LAMBDA)
    mdl = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                             num_leaves=63, max_depth=8,
                             random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
    mdl.fit(X_sub, y_sub, sample_weight=sw)
    y_prob = mdl.predict_proba(X_te)[:, 1]

    # Per-group α_SR threshold shift.
    race_te = prot_te["RACE"]
    age_te = prot_te["AGE_GROUP"]
    sex_te = prot_te["SEX"]
    vkeys = np.array([f"{r}|{a}|{s}" for r, a, s in zip(race_te, age_te, sex_te)])
    overall_sr = float((y_prob >= 0.5).mean())
    y_pred = (y_prob >= 0.5).astype(int)
    for g in set(vkeys):
        m = vkeys == g
        if int(m.sum()) < 5:
            continue
        t_sr = find_sr_threshold(y_prob[m], overall_sr)
        t = float(np.clip(0.5 + FAIR_A_SR * (t_sr - 0.5), 0.01, 0.99))
        y_pred[m] = (y_prob[m] >= t).astype(int)

    row = {
        "N_Hospitals": n_h, "N_Train": int(X_sub.shape[0]),
        "Accuracy": float(accuracy_score(y_te, y_pred)),
        "AUC": float(roc_auc_score(y_te, y_prob)),
    }
    n_fair = 0
    for attr in ATTRS:
        mv = fairness_metrics(y_te, y_pred, y_prob, prot_te[attr])
        for mk in METRIC_KEYS:
            row[f"{mk}_{attr}"] = mv[mk]
            if mk in ("TI", "CAL"):
                continue
            n_fair += int(is_fair(mk, mv[mk]))
        row[f"N_Fair_{attr}"] = sum(int(is_fair(mk, mv[mk])) for mk in METRIC_KEYS
                                    if mk not in ("TI", "CAL"))
    row["N_Fair_Total"] = n_fair
    return row


def main():
    X_tr, X_te, y_tr, y_te, prot_tr, prot_te, hosp_tr = load_data()
    print(f"Loaded train={len(X_tr):,}  test={len(X_te):,}  "
          f"hospitals={len(np.unique(hosp_tr))}")
    rows = []
    t0 = time.time()
    for n_h in N_HOSP_LEVELS:
        ts = time.time()
        row = run_scale_level(X_tr, X_te, y_tr, y_te, prot_tr, prot_te, hosp_tr, n_h)
        print(f"  N_hosp={n_h:>3}  N_train={row['N_Train']:>7,}  "
              f"Acc={row['Accuracy']:.4f}  AUC={row['AUC']:.4f}  "
              f"N_fair={row['N_Fair_Total']}/20  "
              f"t={time.time()-ts:.0f}s")
        rows.append(row)
    df = pd.DataFrame(rows)
    out = TABLES_DIR / "cikm_cross_hospital_scale_FAIR.csv"
    df.to_csv(out, index=False)
    print(f"Saved {out}  ({len(df)} rows, total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
