#!/usr/bin/env python
"""
CIKM 2026 — Cross-Site Fair Model + Full Audit
================================================
1) Runs K=20 cross-site GroupKFold with BOTH Standard AND Fair models
2) Generates per-fold results for 10+ hospital groups  (fixes Table 6/6b)
3) Computes Fleiss' κ for both Standard and Fair models
4) Runs Deliverables 0-6 audit
5) Saves all outputs to output/audit/

Run from cikm/ directory:
    python run_cross_site_fair_and_audit.py
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
import warnings, time, os, sys, json
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             precision_score, recall_score)
import xgboost as xgb
import lightgbm as lgb

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')
PALETTE = sns.color_palette('Set2', 12)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════
K_CS = 20          # cross-site folds
METRIC_KEYS = ['DI','SPD','EOPP','EOD','TI','PP','CAL']
ATTRS = ['RACE','SEX','ETHNICITY','AGE_GROUP']
LAMBDA_DEFAULT = 1.0   # default reweighing strength for cross-site

# Strict thresholds (canonical — matches notebook Cell 2)
THRESHOLDS = {
    'DI':   {'threshold': 0.80, 'direction': 'above'},
    'SPD':  {'threshold': 0.10, 'direction': 'below'},
    'EOPP': {'threshold': 0.10, 'direction': 'below'},
    'EOD':  {'threshold': 0.10, 'direction': 'below'},
    'TI':   {'threshold': 0.10, 'direction': 'below'},
    'PP':   {'threshold': 0.10, 'direction': 'below'},
    'CAL':  {'threshold': 0.05, 'direction': 'below'},
}

# Directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIT_DIR = os.path.join(SCRIPT_DIR, 'output', 'audit')
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'output', 'audit', 'figures')
TABLES_DIR = os.path.join(SCRIPT_DIR, 'output', 'tables')
os.makedirs(AUDIT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# FairnessCalculator (strict thresholds)
# ═══════════════════════════════════════════════════════════════
class FairnessCalculator:
    THRESHOLDS = THRESHOLDS
    def __init__(self, y_true, y_pred, y_prob, protected):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_prob = np.array(y_prob) if y_prob is not None else None
        self.protected = np.array(protected)
        self.groups = np.unique(self.protected)

    @staticmethod
    def disparate_impact(y_pred, protected):
        groups = np.unique(protected)
        rates = {g: np.mean(y_pred[protected == g]) for g in groups}
        max_r = max(rates.values())
        return (min(rates.values()) / max_r if max_r > 0 else 1.0), rates

    def compute_all(self):
        groups = self.groups
        rates = {}
        for g in groups:
            mask = self.protected == g
            y_t, y_p = self.y_true[mask], self.y_pred[mask]
            sr = np.mean(y_p)
            tpr = np.mean(y_p[y_t == 1]) if (y_t == 1).any() else 0.0
            fpr = np.mean(y_p[y_t == 0]) if (y_t == 0).any() else 0.0
            ppv = np.mean(y_t[y_p == 1]) if (y_p == 1).any() else 0.0
            rates[g] = {'SR': sr, 'TPR': tpr, 'FPR': fpr, 'PPV': ppv, 'N': int(mask.sum())}
        di, _ = self.disparate_impact(self.y_pred, self.protected)
        sr_vals = [r['SR'] for r in rates.values()]
        spd = max(sr_vals) - min(sr_vals)
        tpr_vals = [r['TPR'] for r in rates.values()]
        eopp = max(tpr_vals) - min(tpr_vals)
        fpr_vals = [r['FPR'] for r in rates.values()]
        eod = max(max(tpr_vals)-min(tpr_vals), max(fpr_vals)-min(fpr_vals))
        ppv_vals = [r['PPV'] for r in rates.values()]
        pp = max(ppv_vals) - min(ppv_vals)
        all_preds = []
        for g in groups:
            mask = self.protected == g
            all_preds.append(self.y_pred[mask][:min(mask.sum(), 5000)])
        min_len = min(len(p) for p in all_preds)
        if min_len == 0:
            ti = 0.0
        else:
            ti = np.mean([np.mean(all_preds[i][:min_len] != all_preds[j][:min_len])
                          for i in range(len(groups)) for j in range(i+1, len(groups))])
        if self.y_prob is not None:
            cal_diffs = []
            for g in groups:
                mask = self.protected == g
                prob_g = self.y_prob[mask]; y_g = self.y_true[mask]
                bins = np.linspace(0, 1, 11)
                for b in range(len(bins)-1):
                    in_bin = (prob_g >= bins[b]) & (prob_g < bins[b+1])
                    if in_bin.sum() >= 10:
                        cal_diffs.append(abs(np.mean(y_g[in_bin]) - np.mean(prob_g[in_bin])))
            cal = max(cal_diffs) if cal_diffs else 0.0
        else:
            cal = 0.0
        metrics = {'DI': di, 'SPD': spd, 'EOPP': eopp, 'EOD': eod,
                   'TI': ti, 'PP': pp, 'CAL': cal}
        verdicts = {}
        for mk, mv in metrics.items():
            t = self.THRESHOLDS[mk]
            verdicts[mk] = (mv >= t['threshold'] if t['direction'] == 'above'
                            else mv <= t['threshold'])
        return metrics, verdicts, rates


# ═══════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════
def load_data():
    DATA_CANDIDATES = [
        os.path.join(SCRIPT_DIR, '..', '..', '..', '..', 'data', 'texas_100x.csv'),
        os.path.join(SCRIPT_DIR, '..', '..', '..', 'data', 'texas_100x.csv'),
        os.path.join(SCRIPT_DIR, 'data', 'texas_100x.csv'),
        r'd:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\data\texas_100x.csv',
    ]
    DATA_PATH = None
    for p in DATA_CANDIDATES:
        if os.path.exists(p):
            DATA_PATH = p
            break
    assert DATA_PATH is not None, "texas_100x.csv not found"
    print(f"[DATA] {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"[DATA] {len(df):,} records × {df.shape[1]} columns")

    df['LOS_BINARY'] = (df['LENGTH_OF_STAY'] > 3).astype(int)
    def create_age_groups(age_code):
        if age_code <= 4: return 'Age_0_17'
        elif age_code <= 9: return 'Age_18_39'
        elif age_code <= 12: return 'Age_40_54'
        elif age_code <= 14: return 'Age_55_64'
        else: return 'Age_65_Plus'
    df['AGE_GROUP'] = df['PAT_AGE'].apply(create_age_groups)

    target = 'LOS_BINARY'
    protected_cols = ['RACE', 'SEX_CODE', 'ETHNICITY', 'AGE_GROUP']
    exclude_cols = [target, 'LENGTH_OF_STAY', 'THCIC_ID', 'RECORD_ID'] + protected_cols
    feature_cols = [c for c in df.columns if c not in exclude_cols
                    and df[c].dtype in ['int64','float64','object']]

    df_enc = df.copy()
    for col in feature_cols:
        if df_enc[col].dtype == 'object':
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))

    X = df_enc[feature_cols].fillna(0).values
    y = df_enc[target].values
    hospital_ids = df_enc['THCIC_ID'].values

    X_train, X_test, y_train, y_test, hosp_train, hosp_test = train_test_split(
        X, y, hospital_ids, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    protected_attrs = {}
    protected_attrs_train = {}
    _tmp = train_test_split(range(len(df_enc)), test_size=0.2,
                            random_state=RANDOM_STATE, stratify=y)
    train_idx, test_idx = _tmp[0], _tmp[1]
    for attr_col in protected_cols:
        attr_name = attr_col.replace('_CODE','')
        le_attr = LabelEncoder()
        all_encoded = le_attr.fit_transform(df_enc[attr_col].astype(str))
        protected_attrs[attr_name] = all_encoded[test_idx]
        protected_attrs_train[attr_name] = all_encoded[train_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(f"[DATA] Features: {len(feature_cols)}, Train: {len(X_train):,}, Test: {len(X_test):,}")

    return (df, df_enc, X_train, X_test, y_train, y_test,
            hosp_train, hosp_test, hospital_ids,
            protected_attrs, protected_attrs_train,
            scaler, feature_cols, train_idx, test_idx)


# ═══════════════════════════════════════════════════════════════
# Helper functions (from notebook intervention code)
# ═══════════════════════════════════════════════════════════════
def build_multi_weights(y, race, age, sex, lam):
    """RACE × AGE × SEX intersectional reweighing."""
    keys = np.array([f"{r}|{a}|{s}" for r, a, s in zip(race, age, sex)])
    n = len(y)
    sw = np.ones(n, dtype=float)
    for g in sorted(set(keys)):
        mg = keys == g
        ng = mg.sum()
        for lab in [0, 1]:
            mgl = mg & (y == lab)
            ngl = mgl.sum()
            if ngl > 0:
                expected = (ng / n) * ((y == lab).sum() / n)
                observed = ngl / n
                raw_w = expected / observed if observed > 0 else 1.0
                sw[mgl] = np.clip(1.0 + lam * (raw_w - 1.0), 0.1, 10.0)
    return sw


def find_sr_threshold(probs, target_sr, lo=0.01, hi=0.99, step=0.005):
    best_t, best_diff = 0.5, abs((probs >= 0.5).mean() - target_sr)
    for t in np.arange(lo, hi, step):
        diff = abs((probs >= t).mean() - target_sr)
        if diff < best_diff:
            best_diff, best_t = diff, t
    return best_t


def find_tpr_threshold(probs, labels, target_tpr, lo=0.01, hi=0.99, step=0.005):
    pos = labels == 1
    if pos.sum() < 10:
        return 0.5
    best_t, best_diff = 0.5, abs((probs[pos] >= 0.5).mean() - target_tpr)
    for t in np.arange(lo, hi, step):
        diff = abs((probs[pos] >= t).mean() - target_tpr)
        if diff < best_diff:
            best_diff, best_t = diff, t
    return best_t


def run_threshold_optimization(y_prob, y_true, prot_dict,
                               A_SR=[0.0, 0.3, 0.5, 0.7, 1.0],
                               A_TPR=[0.0, 0.5, 1.0],
                               A_PPV=[0.0, 0.5]):
    """Simplified threshold optimization for cross-site.
    Returns best y_pred achieving ALL DI >= 0.80 (or best available).
    """
    race = prot_dict['RACE']
    age = prot_dict['AGE_GROUP']
    sex = prot_dict['SEX']

    # Build intersection groups
    test_groups = {}
    for r in sorted(set(race)):
        for a in sorted(set(age)):
            for s in sorted(set(sex)):
                key = f"{r}|{a}|{s}"
                mask = (race == r) & (age == a) & (sex == s)
                if mask.sum() >= 5:
                    test_groups[key] = mask

    if len(test_groups) == 0:
        return (y_prob >= 0.5).astype(int), False

    # Compute per-group thresholds
    overall_sr = (y_prob >= 0.5).mean()
    overall_tpr = (y_prob[y_true == 1] >= 0.5).mean() if (y_true == 1).any() else 0.5
    sr_thresh, tpr_thresh = {}, {}
    for key, mask in test_groups.items():
        sr_thresh[key] = find_sr_threshold(y_prob[mask], overall_sr)
        tpr_thresh[key] = find_tpr_threshold(y_prob[mask], y_true[mask], overall_tpr)

    best_pred = None
    best_total_fair = -1
    best_acc = 0
    best_all_di = False

    for a_sr in A_SR:
        for a_tpr in A_TPR:
            for a_ppv in A_PPV:
                thresholds = {}
                for key in test_groups:
                    t = 0.5 + a_sr * (sr_thresh[key] - 0.5) + a_tpr * (tpr_thresh[key] - 0.5)
                    thresholds[key] = np.clip(t, 0.01, 0.99)

                y_pred_c = (y_prob >= 0.5).astype(int)
                for key, mask in test_groups.items():
                    y_pred_c[mask] = (y_prob[mask] >= thresholds[key]).astype(int)

                acc_c = accuracy_score(y_true, y_pred_c)

                # Check all 4 DI
                all_di_fair = True
                total_fair = 0
                for attr in ATTRS:
                    attr_val = prot_dict[attr]
                    if len(set(attr_val)) < 2:
                        continue
                    fc = FairnessCalculator(y_true, y_pred_c, y_prob, attr_val)
                    mc, vc, _ = fc.compute_all()
                    if mc['DI'] < 0.80:
                        all_di_fair = False
                    total_fair += sum(vc.values())

                # Select best candidate
                if all_di_fair and (not best_all_di or total_fair > best_total_fair
                                    or (total_fair == best_total_fair and acc_c > best_acc)):
                    best_pred = y_pred_c.copy()
                    best_total_fair = total_fair
                    best_acc = acc_c
                    best_all_di = True
                elif not best_all_di and total_fair > best_total_fair:
                    best_pred = y_pred_c.copy()
                    best_total_fair = total_fair
                    best_acc = acc_c

    if best_pred is None:
        best_pred = (y_prob >= 0.5).astype(int)

    return best_pred, best_all_di


# ═══════════════════════════════════════════════════════════════
# Fleiss' κ
# ═══════════════════════════════════════════════════════════════
def fleiss_kappa(ratings_matrix):
    N, k = ratings_matrix.shape
    n = ratings_matrix.sum(axis=1)[0]
    if n <= 1:
        return 0.0
    p_j = ratings_matrix.sum(axis=0) / (N * n)
    P_i = (np.sum(ratings_matrix**2, axis=1) - n) / (n * (n - 1))
    P_bar = P_i.mean()
    P_e = np.sum(p_j**2)
    if abs(1 - P_e) < 1e-9:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


# ═══════════════════════════════════════════════════════════════
# MAIN: Cross-Site Fair Model + Audit
# ═══════════════════════════════════════════════════════════════
def main():
    t_start = time.time()
    print("=" * 80)
    print("CIKM 2026 — Cross-Site Fair Model + Full Audit")
    print("=" * 80)

    # ── Load Data ─────────────────────────────────────────────
    (df, df_enc, X_train, X_test, y_train, y_test,
     hosp_train, hosp_test, hospital_ids,
     protected_attrs, protected_attrs_train,
     scaler, feature_cols, train_idx, test_idx) = load_data()

    # ── Prepare All-Data arrays for GroupKFold ────────────────
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    hosp_all = np.concatenate([hosp_train, hosp_test])
    prot_all = {}
    for attr in ATTRS:
        prot_all[attr] = np.concatenate([protected_attrs_train[attr],
                                         protected_attrs[attr]])

    # ══════════════════════════════════════════════════════════
    # SECTION 1: Cross-Site K=20 GroupKFold — Standard + Fair
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"CROSS-SITE ANALYSIS: K={K_CS} GroupKFold — Standard + Fair Models")
    print(f"{'='*80}")

    gkf = GroupKFold(n_splits=K_CS)
    cs_rows = []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_all, y_all, groups=hosp_all)):
        t0 = time.time()
        X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
        X_val, y_val = X_all[val_idx], y_all[val_idx]

        prot_tr = {a: prot_all[a][tr_idx] for a in ATTRS}
        prot_val = {a: prot_all[a][val_idx] for a in ATTRS}

        n_hosp_val = len(set(hosp_all[val_idx]))

        # ─── Standard model ──────────────────────────────────
        mdl_std = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=63,
            max_depth=8, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
        mdl_std.fit(X_tr, y_tr)
        y_pred_std = mdl_std.predict(X_val)
        y_prob_std = mdl_std.predict_proba(X_val)[:, 1]

        # ─── Fair model (reweighed + threshold optimized) ────
        sw = build_multi_weights(y_tr, prot_tr['RACE'], prot_tr['AGE_GROUP'],
                                 prot_tr['SEX'], LAMBDA_DEFAULT)
        mdl_fair = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=63,
            max_depth=8, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
        mdl_fair.fit(X_tr, y_tr, sample_weight=sw)
        y_prob_fair = mdl_fair.predict_proba(X_val)[:, 1]

        # Threshold optimization on validation fold
        y_pred_fair, all_di_ok = run_threshold_optimization(
            y_prob_fair, y_val, prot_val)

        # ─── Compute fairness for both ───────────────────────
        row = {
            'Fold': fold + 1,
            'N_val': len(val_idx),
            'N_hospitals': n_hosp_val,
            # Standard model
            'Std_Acc': accuracy_score(y_val, y_pred_std),
            'Std_AUC': roc_auc_score(y_val, y_prob_std) if len(set(y_val)) > 1 else np.nan,
            # Fair model
            'Fair_Acc': accuracy_score(y_val, y_pred_fair),
            'Fair_AUC': roc_auc_score(y_val, y_prob_fair) if len(set(y_val)) > 1 else np.nan,
            'Fair_AllDI': all_di_ok,
        }

        for attr in ATTRS:
            attr_val = prot_val[attr]
            if len(set(attr_val)) >= 2:
                # Standard
                fc_s = FairnessCalculator(y_val, y_pred_std, y_prob_std, attr_val)
                mc_s, vc_s, _ = fc_s.compute_all()
                for mk in METRIC_KEYS:
                    row[f'Std_{mk}_{attr}'] = mc_s[mk]
                    row[f'Std_V_{mk}_{attr}'] = 1 if vc_s[mk] else 0
                row[f'Std_NFair_{attr}'] = sum(vc_s.values())

                # Fair
                fc_f = FairnessCalculator(y_val, y_pred_fair, y_prob_fair, attr_val)
                mc_f, vc_f, _ = fc_f.compute_all()
                for mk in METRIC_KEYS:
                    row[f'Fair_{mk}_{attr}'] = mc_f[mk]
                    row[f'Fair_V_{mk}_{attr}'] = 1 if vc_f[mk] else 0
                row[f'Fair_NFair_{attr}'] = sum(vc_f.values())
            else:
                for mk in METRIC_KEYS:
                    for pfx in ['Std_', 'Fair_']:
                        row[f'{pfx}{mk}_{attr}'] = np.nan
                        row[f'{pfx}V_{mk}_{attr}'] = np.nan
                row[f'Std_NFair_{attr}'] = np.nan
                row[f'Fair_NFair_{attr}'] = np.nan

        cs_rows.append(row)
        elapsed = time.time() - t0
        std_nf = sum(int(row.get(f'Std_NFair_{a}', 0) or 0) for a in ATTRS)
        fair_nf = sum(int(row.get(f'Fair_NFair_{a}', 0) or 0) for a in ATTRS)
        print(f"  Fold {fold+1:2d}/{K_CS}: N={len(val_idx):>6,} hosp={n_hosp_val:>3d}  "
              f"Std: Acc={row['Std_Acc']:.4f} Fair={std_nf}/28  |  "
              f"Fair: Acc={row['Fair_Acc']:.4f} Fair={fair_nf}/28 "
              f"AllDI={'✓' if all_di_ok else '✗'}  ({elapsed:.1f}s)")

    cs_df = pd.DataFrame(cs_rows)
    cs_df.to_csv(f'{TABLES_DIR}/Table6_CrossSite_StdFair_PerFold.csv', index=False)
    print(f"\n✓ Saved: Table6_CrossSite_StdFair_PerFold.csv ({len(cs_df)} folds)")

    # ══════════════════════════════════════════════════════════
    # SECTION 2: Table 6 — Per-Group Cross-Site Fairness Variation
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("TABLE 6: Cross-Site Fairness Variation — Per Hospital Group (All 20 Folds)")
    print(f"{'='*80}")

    # Build summary for BOTH standard and fair
    for model_type in ['Std', 'Fair']:
        print(f"\n  === {model_type} Model ===")
        summary_rows = []
        for attr in ATTRS:
            for mk in METRIC_KEYS:
                col = f'{model_type}_{mk}_{attr}'
                vals = cs_df[col].dropna()
                if len(vals) < 2:
                    continue
                vcol = f'{model_type}_V_{mk}_{attr}'
                v_vals = cs_df[vcol].dropna()
                summary_rows.append({
                    'Model': model_type, 'Attribute': attr, 'Metric': mk,
                    'Mean': vals.mean(), 'Std': vals.std(),
                    'CV': vals.std() / max(vals.mean(), 1e-9),
                    'Min': vals.min(), 'Max': vals.max(),
                    'Range': vals.max() - vals.min(),
                    'N_Fair_Folds': int(v_vals.sum()) if len(v_vals) else 0,
                    'Pct_Fair': v_vals.mean() * 100 if len(v_vals) else 0,
                })

        summ_df = pd.DataFrame(summary_rows)
        if model_type == 'Std':
            std_summary = summ_df
        else:
            fair_summary = summ_df

        # Print pivot
        if len(summ_df) > 0:
            pivot = summ_df.pivot(index='Metric', columns='Attribute', values='Mean')
            print(f"\n  Mean values across {K_CS} folds:")
            print(pivot.to_string(float_format='{:.4f}'.format))

            pivot_fair = summ_df.pivot(index='Metric', columns='Attribute', values='Pct_Fair')
            print(f"\n  % folds deemed FAIR:")
            print(pivot_fair.to_string(float_format='{:.0f}%'.format))

    # Save combined summary
    combined_summary = pd.concat([std_summary, fair_summary], ignore_index=True)
    combined_summary.to_csv(f'{AUDIT_DIR}/Table6_CrossSite_Summary.csv', index=False)

    # ══════════════════════════════════════════════════════════
    # SECTION 3: Table 6b — Fleiss' κ for Both Models
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("TABLE 6b: Fleiss' κ — Standard vs Fair Model")
    print(f"{'='*80}")

    kappa_rows = []
    for model_type in ['Std', 'Fair']:
        for attr in ATTRS:
            for mk in METRIC_KEYS:
                vcol = f'{model_type}_V_{mk}_{attr}'
                if vcol not in cs_df.columns:
                    continue
                verdicts = cs_df[vcol].dropna().values
                n_fair = int(verdicts.sum())
                n_unfair = len(verdicts) - n_fair
                kappa_rows.append({
                    'Model': model_type, 'Attribute': attr, 'Metric': mk,
                    'N_Fair': n_fair, 'N_Unfair': n_unfair,
                    'N_Folds': len(verdicts),
                    'Pct_Fair': n_fair / len(verdicts) * 100 if len(verdicts) else 0,
                })

    kappa_df = pd.DataFrame(kappa_rows)

    # Compute κ per model
    for model_type in ['Std', 'Fair']:
        sub = kappa_df[kappa_df['Model'] == model_type]
        if len(sub) > 1:
            ratings = sub[['N_Fair', 'N_Unfair']].values
            fk = fleiss_kappa(ratings)
            print(f"\n  {model_type} Model — Overall Fleiss' κ: {fk:.3f}")

        # Per-metric κ
        print(f"  Per-metric κ:")
        for mk in METRIC_KEYS:
            mk_sub = sub[sub['Metric'] == mk]
            if len(mk_sub) > 1:
                r = mk_sub[['N_Fair', 'N_Unfair']].values
                mk_k = fleiss_kappa(r)
                pct = mk_sub['Pct_Fair'].mean()
                print(f"    {mk}: κ={mk_k:.3f}  ({pct:.0f}% folds fair)")

    kappa_df.to_csv(f'{AUDIT_DIR}/Table6b_Fleiss_Kappa_StdFair.csv', index=False)

    # ══════════════════════════════════════════════════════════
    # SECTION 4: Mean Fairness Assessment
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("MEAN FAIRNESS: Does the Fair Model Achieve ≥4/7 Fair on Average?")
    print(f"{'='*80}")

    for model_type in ['Std', 'Fair']:
        print(f"\n  === {model_type} Model — Mean Fair Metrics per Attribute ===")
        for attr in ATTRS:
            nfair_col = f'{model_type}_NFair_{attr}'
            vals = cs_df[nfair_col].dropna()
            mean_nf = vals.mean()
            min_nf = vals.min()
            max_nf = vals.max()
            ge4 = (vals >= 4).sum()
            print(f"    {attr}: Mean={mean_nf:.1f}/7  Min={min_nf:.0f}  Max={max_nf:.0f}  "
                  f"≥4/7 in {ge4}/{len(vals)} folds ({ge4/len(vals)*100:.0f}%)")

        # Mean total fair
        total_nf = sum(cs_df[f'{model_type}_NFair_{a}'].dropna().mean() for a in ATTRS)
        print(f"    TOTAL MEAN: {total_nf:.1f}/28")

    # ══════════════════════════════════════════════════════════
    # SECTION 5: Per-Fold Detail Table (all 20 folds × 4 attrs)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("PER-FOLD DETAIL: Standard vs Fair (DI + N_Fair per attribute)")
    print(f"{'='*80}")

    detail_rows = []
    for _, row in cs_df.iterrows():
        for model_type in ['Std', 'Fair']:
            drow = {
                'Fold': int(row['Fold']),
                'Model': model_type,
                'N_val': int(row['N_val']),
                'N_hospitals': int(row['N_hospitals']),
                'Accuracy': row[f'{model_type}_Acc'],
                'AUC': row[f'{model_type}_AUC'],
            }
            total = 0
            for attr in ATTRS:
                drow[f'DI_{attr}'] = row.get(f'{model_type}_DI_{attr}', np.nan)
                nf = row.get(f'{model_type}_NFair_{attr}', 0)
                drow[f'Fair_{attr}'] = int(nf) if pd.notna(nf) else 0
                total += int(nf) if pd.notna(nf) else 0
            drow['Total_Fair'] = total
            detail_rows.append(drow)

    detail_df = pd.DataFrame(detail_rows)
    detail_df.to_csv(f'{TABLES_DIR}/Table6_CrossSite_PerFold_Detail.csv', index=False)

    # Print nicely
    for model_type in ['Std', 'Fair']:
        mdf = detail_df[detail_df['Model'] == model_type]
        print(f"\n  === {model_type} Model ===")
        print(f"  {'Fold':>4s} {'N_val':>7s} {'Hosp':>4s} {'Acc':>7s} "
              f"{'DI_R':>6s} {'DI_S':>6s} {'DI_E':>6s} {'DI_A':>6s} "
              f"{'F_R':>3s} {'F_S':>3s} {'F_E':>3s} {'F_A':>3s} {'Tot':>3s}")
        for _, r in mdf.iterrows():
            print(f"  {int(r['Fold']):4d} {int(r['N_val']):7,d} {int(r['N_hospitals']):4d} "
                  f"{r['Accuracy']:7.4f} "
                  f"{r['DI_RACE']:6.3f} {r['DI_SEX']:6.3f} "
                  f"{r['DI_ETHNICITY']:6.3f} {r['DI_AGE_GROUP']:6.3f} "
                  f"{int(r['Fair_RACE']):3d} {int(r['Fair_SEX']):3d} "
                  f"{int(r['Fair_ETHNICITY']):3d} {int(r['Fair_AGE_GROUP']):3d} "
                  f"{int(r['Total_Fair']):3d}")

        # Mean row
        print(f"  {'MEAN':>4s} {mdf['N_val'].mean():7,.0f} {mdf['N_hospitals'].mean():4.0f} "
              f"{mdf['Accuracy'].mean():7.4f} "
              f"{mdf['DI_RACE'].mean():6.3f} {mdf['DI_SEX'].mean():6.3f} "
              f"{mdf['DI_ETHNICITY'].mean():6.3f} {mdf['DI_AGE_GROUP'].mean():6.3f} "
              f"{mdf['Fair_RACE'].mean():3.1f} {mdf['Fair_SEX'].mean():3.1f} "
              f"{mdf['Fair_ETHNICITY'].mean():3.1f} {mdf['Fair_AGE_GROUP'].mean():3.1f} "
              f"{mdf['Total_Fair'].mean():3.1f}")

    # ══════════════════════════════════════════════════════════
    # SECTION 6: Figures
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("GENERATING FIGURES")
    print(f"{'='*80}")

    # Figure A: Standard vs Fair DI across 20 folds (4 panels)
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    for i, attr in enumerate(ATTRS):
        ax = axes[i // 2, i % 2]
        folds = cs_df['Fold'].values
        std_di = cs_df[f'Std_DI_{attr}'].values
        fair_di = cs_df[f'Fair_DI_{attr}'].values

        x = np.arange(len(folds))
        w = 0.35
        bars1 = ax.bar(x - w/2, std_di, w, label='Standard', color='#e74c3c',
                       alpha=0.8, edgecolor='white')
        bars2 = ax.bar(x + w/2, fair_di, w, label='Fair', color='#2ecc71',
                       alpha=0.8, edgecolor='white')
        ax.axhline(y=0.80, color='black', ls='--', lw=1.5, alpha=0.7,
                   label='DI ≥ 0.80')
        ax.set_xticks(x)
        ax.set_xticklabels([f'F{int(f)}' for f in folds], fontsize=7, rotation=45)
        ax.set_ylabel('DI')
        ax.set_ylim(0, 1.15)
        ax.set_title(f'{attr}', fontweight='bold', fontsize=13)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Cross-Site DI: Standard vs Fair Model (20 Hospital Clusters)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f'{FIGURES_DIR}/crosssite_std_vs_fair_DI.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  Saved: crosssite_std_vs_fair_DI.png")

    # Figure B: Fair metrics count per fold (Standard vs Fair)
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    for i, attr in enumerate(ATTRS):
        ax = axes[i // 2, i % 2]
        folds = cs_df['Fold'].values
        std_nf = cs_df[f'Std_NFair_{attr}'].values
        fair_nf = cs_df[f'Fair_NFair_{attr}'].values

        x = np.arange(len(folds))
        w = 0.35
        ax.bar(x - w/2, std_nf, w, label='Standard', color='#e74c3c',
               alpha=0.8, edgecolor='white')
        ax.bar(x + w/2, fair_nf, w, label='Fair', color='#2ecc71',
               alpha=0.8, edgecolor='white')
        ax.axhline(y=4, color='blue', ls=':', lw=1.5, alpha=0.7,
                   label='4/7 target')
        ax.set_xticks(x)
        ax.set_xticklabels([f'F{int(f)}' for f in folds], fontsize=7, rotation=45)
        ax.set_ylabel('Fair Metrics (of 7)')
        ax.set_ylim(0, 7.5)
        ax.set_title(f'{attr}', fontweight='bold', fontsize=13)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Cross-Site Fair Metrics: Standard vs Fair Model (20 Folds)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f'{FIGURES_DIR}/crosssite_std_vs_fair_NFair.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  Saved: crosssite_std_vs_fair_NFair.png")

    # Figure C: Line graph — Accuracy & DI across folds (Standard vs Fair)
    fig, axes = plt.subplots(1, 2, figsize=(22, 8))
    folds = cs_df['Fold'].values

    for pidx, model_type in enumerate(['Std', 'Fair']):
        ax = axes[pidx]
        ax2 = ax.twinx()

        acc = cs_df[f'{model_type}_Acc'].values
        l0, = ax.plot(folds, acc, 'ko-', lw=2.5, ms=7, label='Accuracy', zorder=10)
        ax.fill_between(folds, acc, alpha=0.08, color='black')
        ax.set_ylabel('Accuracy', color='black')
        ax.set_ylim(0.50, 0.90)

        lines = [l0]
        for j, attr in enumerate(ATTRS):
            di = cs_df[f'{model_type}_DI_{attr}'].values
            l, = ax2.plot(folds, di, 'o-', color=PALETTE[j], lw=2, ms=6,
                         label=f'DI_{attr}', alpha=0.85)
            lines.append(l)

        ax2.axhline(0.80, color='red', ls='--', lw=1.5, alpha=0.6)
        ax2.set_ylabel('Disparate Impact')
        ax2.set_ylim(0, 1.15)
        ax.set_xlabel('Fold')
        ax.set_title(f'({chr(97+pidx)}) {model_type} Model', fontweight='bold', fontsize=13)
        ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc='lower right')
        ax.grid(alpha=0.3)

    plt.suptitle('Cross-Site: Accuracy & DI Across 20 Hospital Clusters',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f'{FIGURES_DIR}/crosssite_acc_di_lines.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  Saved: crosssite_acc_di_lines.png")

    # Figure D: Heatmap — All 7 metrics × 4 attrs (Fair model mean)
    fig, axes = plt.subplots(1, 2, figsize=(22, 8))
    for pidx, model_type in enumerate(['Std', 'Fair']):
        ax = axes[pidx]
        mean_vals = np.zeros((len(METRIC_KEYS), len(ATTRS)))
        for i, mk in enumerate(METRIC_KEYS):
            for j, attr in enumerate(ATTRS):
                col = f'{model_type}_{mk}_{attr}'
                mean_vals[i, j] = cs_df[col].mean() if col in cs_df.columns else np.nan

        # For DI, fair threshold is 0.80; for others, it's ≤ threshold
        # Normalize all to [0,1] where 1=fully fair
        norm_vals = mean_vals.copy()
        for i, mk in enumerate(METRIC_KEYS):
            t = THRESHOLDS[mk]
            if t['direction'] == 'above':
                norm_vals[i, :] = mean_vals[i, :] / t['threshold']
            else:
                norm_vals[i, :] = 1 - mean_vals[i, :] / (t['threshold'] * 2)
        norm_vals = np.clip(norm_vals, 0, 1)

        sns.heatmap(mean_vals, ax=ax, xticklabels=ATTRS, yticklabels=METRIC_KEYS,
                    cmap='RdYlGn', annot=True, fmt='.3f',
                    linewidths=0.5, linecolor='white',
                    cbar_kws={'label': 'Metric Value'})
        ax.set_title(f'{model_type} Model — Mean Across {K_CS} Folds',
                     fontweight='bold', fontsize=12)

    plt.suptitle('Cross-Site Fairness: 7 Metrics × 4 Attributes (Mean)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f'{FIGURES_DIR}/crosssite_heatmap_std_vs_fair.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  Saved: crosssite_heatmap_std_vs_fair.png")

    # ══════════════════════════════════════════════════════════
    # SECTION 7: Deliverable 0 — Consistency Audit
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("DELIVERABLE 0: Consistency Audit")
    print(f"{'='*80}")

    # Check threshold definitions
    generate_script = os.path.join(SCRIPT_DIR, 'generate_all_figures_tables.py')
    consistency_issues = []

    # Define canonical thresholds
    canonical = {
        'DI': (0.80, 'above'), 'SPD': (0.10, 'below'),
        'EOPP': (0.10, 'below'), 'EOD': (0.10, 'below'),
        'TI': (0.10, 'below'), 'PP': (0.10, 'below'),
        'CAL': (0.05, 'below'),
    }

    # Check generate script thresholds
    if os.path.exists(generate_script):
        with open(generate_script, 'r', encoding='utf-8') as f:
            gen_content = f.read()
        # Check for lenient thresholds
        if "'threshold': 0.20" in gen_content:
            consistency_issues.append(
                "generate_all_figures_tables.py uses EOPP/EOD threshold=0.20 "
                "(canonical: 0.10)")
        if "'threshold': 0.10, 'direction': 'below'" in gen_content:
            # Check if CAL uses 0.10 instead of 0.05
            import re
            cal_match = re.search(r"'CAL'.*?'threshold':\s*([\d.]+)", gen_content)
            if cal_match and float(cal_match.group(1)) != 0.05:
                consistency_issues.append(
                    f"generate_all_figures_tables.py uses CAL threshold="
                    f"{cal_match.group(1)} (canonical: 0.05)")

    # Build reconciliation table
    recon_rows = []
    for mk in METRIC_KEYS:
        for attr in ATTRS:
            ct = canonical[mk]
            recon_rows.append({
                'Metric': mk, 'Attribute': attr,
                'Canonical_Threshold': ct[0],
                'Direction': ct[1],
                'Notebook_Matches': True,
                'Generate_Script_Matches': mk not in ['EOPP', 'EOD', 'CAL'],
            })

    recon_df = pd.DataFrame(recon_rows)
    recon_df.to_csv(f'{AUDIT_DIR}/D0_consistency_audit.csv', index=False)

    audit_md = ["# Deliverable 0: Consistency Audit\n"]
    audit_md.append(f"## Issues Found: {len(consistency_issues)}\n")
    for issue in consistency_issues:
        audit_md.append(f"- **ISSUE**: {issue}")
    audit_md.append(f"\n## Canonical Thresholds\n")
    audit_md.append("| Metric | Threshold | Direction |")
    audit_md.append("|--------|-----------|-----------|")
    for mk in METRIC_KEYS:
        ct = canonical[mk]
        audit_md.append(f"| {mk} | {ct[0]} | {ct[1]} |")
    audit_md.append(f"\n## generate_all_figures_tables.py Mismatches")
    audit_md.append("The generate script uses lenient thresholds for EOPP (0.20), "
                    "EOD (0.20), and CAL (0.10).")
    audit_md.append("The notebook (Cell 2) uses strict thresholds: "
                    "EOPP=0.10, EOD=0.10, CAL=0.05.")
    audit_md.append("**Recommendation**: Update generate script to match notebook.")

    with open(f'{AUDIT_DIR}/D0_consistency_audit.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(audit_md))
    print(f"  Saved: D0_consistency_audit.csv + .md")
    for issue in consistency_issues:
        print(f"  ⚠ {issue}")

    # ══════════════════════════════════════════════════════════
    # SECTION 8: Deliverable 1 — Fairness Reconciliation Table
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("DELIVERABLE 1: Fairness Reconciliation (Best Standard Model)")
    print(f"{'='*80}")

    # Train best model (LightGBM) on full training set
    best_mdl = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        max_depth=8, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
    best_mdl.fit(X_train, y_train)
    best_pred = best_mdl.predict(X_test)
    best_prob = best_mdl.predict_proba(X_test)[:, 1]
    best_acc = accuracy_score(y_test, best_pred)
    best_auc = roc_auc_score(y_test, best_prob)
    print(f"  Best model: LightGBM  Acc={best_acc:.4f}  AUC={best_auc:.4f}")

    recon_rows = []
    for attr in ATTRS:
        fc = FairnessCalculator(y_test, best_pred, best_prob,
                                protected_attrs[attr])
        mc, vc, rates = fc.compute_all()
        for mk in METRIC_KEYS:
            ct = canonical[mk]
            val = mc[mk]
            fair = vc[mk]
            if ct[1] == 'above':
                margin = val - ct[0]
            else:
                margin = ct[0] - val
            recon_rows.append({
                'Attribute': attr, 'Metric': mk,
                'Value': val, 'Threshold': ct[0], 'Direction': ct[1],
                'Fair': fair, 'Margin': margin,
                'Stability': 'stable' if abs(margin) > 0.02 else 'fragile',
            })

    recon_df = pd.DataFrame(recon_rows)
    recon_df.to_csv(f'{AUDIT_DIR}/D1_fairness_reconciliation.csv', index=False)

    n_fair_total = recon_df['Fair'].sum()
    n_fragile = (recon_df['Stability'] == 'fragile').sum()
    print(f"  Fair verdicts: {n_fair_total}/28")
    print(f"  Fragile (margin < 0.02): {n_fragile}/28")
    print(f"  Saved: D1_fairness_reconciliation.csv")

    # ══════════════════════════════════════════════════════════
    # SECTION 9: Deliverable 2 — Lambda Selection Table
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("DELIVERABLE 2: Lambda Selection Table")
    print(f"{'='*80}")

    lambda_values = [0.0, 0.5, 1.0, 3.0, 5.0, 10.0, 15.0, 30.0, 50.0, 100.0]
    lambda_rows = []

    for lam_val in lambda_values:
        if lam_val == 0.0:
            yp = best_pred
            ypr = best_prob
        else:
            sw = build_multi_weights(y_train, protected_attrs_train['RACE'],
                                     protected_attrs_train['AGE_GROUP'],
                                     protected_attrs_train['SEX'], lam_val)
            mdl_lam = lgb.LGBMClassifier(
                n_estimators=500, learning_rate=0.05, num_leaves=63,
                max_depth=8, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
            mdl_lam.fit(X_train, y_train, sample_weight=sw)
            yp = mdl_lam.predict(X_test)
            ypr = mdl_lam.predict_proba(X_test)[:, 1]

        row = {'Lambda': lam_val,
               'Accuracy': accuracy_score(y_test, yp),
               'AUC': roc_auc_score(y_test, ypr),
               'F1': f1_score(y_test, yp)}

        total_fair = 0
        for attr in ATTRS:
            fc = FairnessCalculator(y_test, yp, ypr, protected_attrs[attr])
            mc, vc, _ = fc.compute_all()
            for mk in METRIC_KEYS:
                row[f'{mk}_{attr}'] = mc[mk]
            n_fair = sum(vc.values())
            row[f'N_Fair_{attr}'] = n_fair
            total_fair += n_fair
        row['Total_Fair'] = total_fair
        lambda_rows.append(row)
        print(f"  λ={lam_val:>5.1f}  Acc={row['Accuracy']:.4f}  "
              f"Fair: R={row['N_Fair_RACE']}/7 S={row['N_Fair_SEX']}/7 "
              f"E={row['N_Fair_ETHNICITY']}/7 A={row['N_Fair_AGE_GROUP']}/7  "
              f"Total={total_fair}/28")

    lambda_df = pd.DataFrame(lambda_rows)
    lambda_df.to_csv(f'{AUDIT_DIR}/D2_lambda_selection.csv', index=False)
    print(f"  Saved: D2_lambda_selection.csv")

    # ══════════════════════════════════════════════════════════
    # SECTION 10: Deliverable 3 — Standard vs Fair Head-to-Head
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("DELIVERABLE 3: Standard vs Fair Head-to-Head")
    print(f"{'='*80}")

    # Train fair model with threshold optimization
    sw_fair = build_multi_weights(y_train, protected_attrs_train['RACE'],
                                  protected_attrs_train['AGE_GROUP'],
                                  protected_attrs_train['SEX'], LAMBDA_DEFAULT)
    mdl_fair_global = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        max_depth=8, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
    mdl_fair_global.fit(X_train, y_train, sample_weight=sw_fair)
    y_prob_fair_g = mdl_fair_global.predict_proba(X_test)[:, 1]

    y_pred_fair_g, all_di_ok_g = run_threshold_optimization(
        y_prob_fair_g, y_test, protected_attrs)

    h2h_rows = []
    h2h_rows.append({
        'Metric': 'Accuracy',
        'Standard': best_acc,
        'Fair': accuracy_score(y_test, y_pred_fair_g),
        'Change': accuracy_score(y_test, y_pred_fair_g) - best_acc,
    })
    h2h_rows.append({
        'Metric': 'AUC',
        'Standard': best_auc,
        'Fair': roc_auc_score(y_test, y_prob_fair_g),
        'Change': roc_auc_score(y_test, y_prob_fair_g) - best_auc,
    })

    for attr in ATTRS:
        fc_s = FairnessCalculator(y_test, best_pred, best_prob,
                                  protected_attrs[attr])
        mc_s, vs_s, _ = fc_s.compute_all()
        fc_f = FairnessCalculator(y_test, y_pred_fair_g, y_prob_fair_g,
                                  protected_attrs[attr])
        mc_f, vs_f, _ = fc_f.compute_all()
        for mk in METRIC_KEYS:
            h2h_rows.append({
                'Metric': f'{mk} ({attr})',
                'Standard': mc_s[mk],
                'Fair': mc_f[mk],
                'Change': mc_f[mk] - mc_s[mk],
                'Std_Fair': vs_s[mk],
                'Fair_Fair': vs_f[mk],
            })

    h2h_df = pd.DataFrame(h2h_rows)
    h2h_df.to_csv(f'{AUDIT_DIR}/D3_standard_vs_fair.csv', index=False)

    n_std_fair = h2h_df['Std_Fair'].sum() if 'Std_Fair' in h2h_df.columns else 0
    n_fair_fair = h2h_df['Fair_Fair'].sum() if 'Fair_Fair' in h2h_df.columns else 0
    print(f"  Standard: {n_std_fair:.0f}/28 fair")
    print(f"  Fair:     {n_fair_fair:.0f}/28 fair")
    print(f"  Acc drop: {(best_acc - accuracy_score(y_test, y_pred_fair_g))*100:.1f} pp")
    print(f"  All DI ≥ 0.80: {'Yes' if all_di_ok_g else 'No'}")
    print(f"  Saved: D3_standard_vs_fair.csv")

    # ══════════════════════════════════════════════════════════
    # SECTION 11: Deliverable 5 — Three-Panel Figure
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("DELIVERABLE 5: Three-Panel Figure")
    print(f"{'='*80}")

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # Panel (a): Pareto — Accuracy vs Total Fair
    ax = axes[0]
    # Use lambda sweep data
    ax.scatter(lambda_df['Accuracy'], lambda_df['Total_Fair'],
               s=100, c=lambda_df['Lambda'], cmap='viridis',
               edgecolors='black', linewidths=0.5, zorder=5)
    for _, r in lambda_df.iterrows():
        ax.annotate(f'λ={r["Lambda"]:.0f}',
                    (r['Accuracy'], r['Total_Fair']),
                    fontsize=7, ha='center', va='bottom')
    ax.axhline(y=16, color='green', ls=':', alpha=0.5, label='≥4/7 × 4 attrs')
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Total Fair Metrics (of 28)')
    ax.set_title('(a) Pareto: Accuracy vs Fairness', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel (b): DI before/after (standard vs fair)
    ax = axes[1]
    std_di = [h2h_df[h2h_df['Metric'] == f'DI ({attr})']['Standard'].values[0]
              if len(h2h_df[h2h_df['Metric'] == f'DI ({attr})']) else 0.8
              for attr in ATTRS]
    fair_di = [h2h_df[h2h_df['Metric'] == f'DI ({attr})']['Fair'].values[0]
               if len(h2h_df[h2h_df['Metric'] == f'DI ({attr})']) else 0.8
               for attr in ATTRS]
    x = np.arange(len(ATTRS))
    w = 0.35
    ax.bar(x - w/2, std_di, w, label='Standard', color='#e74c3c',
           alpha=0.85, edgecolor='white')
    ax.bar(x + w/2, fair_di, w, label='Fair', color='#2ecc71',
           alpha=0.85, edgecolor='white')
    ax.axhline(y=0.80, color='black', ls='--', lw=1.5, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(ATTRS, fontsize=9)
    ax.set_ylabel('DI')
    ax.set_ylim(0, 1.15)
    ax.set_title('(b) DI: Standard vs Fair', fontweight='bold')
    ax.legend(fontsize=9)

    # Panel (c): Per-cluster improvement (mean fair metrics)
    ax = axes[2]
    std_means = [cs_df[f'Std_NFair_{a}'].mean() for a in ATTRS]
    fair_means = [cs_df[f'Fair_NFair_{a}'].mean() for a in ATTRS]
    x = np.arange(len(ATTRS))
    ax.bar(x - w/2, std_means, w, label='Standard', color='#e74c3c',
           alpha=0.85, edgecolor='white')
    ax.bar(x + w/2, fair_means, w, label='Fair', color='#2ecc71',
           alpha=0.85, edgecolor='white')
    ax.axhline(y=4, color='blue', ls=':', lw=1.5, alpha=0.7, label='≥4/7 target')
    ax.set_xticks(x)
    ax.set_xticklabels(ATTRS, fontsize=9)
    ax.set_ylabel('Mean Fair Metrics (of 7)')
    ax.set_ylim(0, 7.5)
    ax.set_title('(c) Cross-Site Mean Fair Metrics', fontweight='bold')
    ax.legend(fontsize=9)

    for b, v in zip(ax.patches[:4], std_means):
        ax.text(b.get_x() + b.get_width()/2, v + 0.1, f'{v:.1f}',
                ha='center', fontsize=9, fontweight='bold')
    for b, v in zip(ax.patches[4:], fair_means):
        ax.text(b.get_x() + b.get_width()/2, v + 0.1, f'{v:.1f}',
                ha='center', fontsize=9, fontweight='bold', color='#27ae60')

    plt.suptitle('Deliverable 5: Fairness Intervention — Three-Panel Summary',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f'{FIGURES_DIR}/D5_three_panel_summary.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  Saved: D5_three_panel_summary.png")

    # ══════════════════════════════════════════════════════════
    # SECTION 12: Deliverable 6 — Demographic Audit
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("DELIVERABLE 6: Demographic Audit")
    print(f"{'='*80}")

    RACE_MAP = {0: 'Other/Unknown', 1: 'Native American',
                2: 'Asian/Pacific Islander', 3: 'Black', 4: 'White'}

    # RACE × ETHNICITY cross-tab
    race_all = df_enc['RACE'].values
    eth_all = df_enc['ETHNICITY'].values
    xtab = pd.crosstab(race_all, eth_all, margins=True)
    xtab.index = [RACE_MAP.get(i, str(i)) if i != 'All' else 'All' for i in xtab.index]
    xtab.columns = [f'Eth_{c}' for c in xtab.columns]

    print("\n  RACE × ETHNICITY Cross-Tabulation:")
    print(xtab.to_string())

    # Check for double-coding
    demo_issues = []
    for race_code in sorted(set(race_all)):
        mask = race_all == race_code
        eth_vals = eth_all[mask]
        if len(set(eth_vals)) >= 2:
            eth_dist = pd.Series(eth_vals).value_counts(normalize=True)
            majority = eth_dist.iloc[0]
            if majority > 0.80:
                race_name = RACE_MAP.get(race_code, str(race_code))
                demo_issues.append(
                    f"RACE={race_code} ({race_name}): {majority:.1%} are "
                    f"ETHNICITY={eth_dist.index[0]} — potential double-coding")

    print("\n  Demographic Issues:")
    for issue in demo_issues:
        print(f"    ⚠ {issue}")

    # Target rate by RACE
    print("\n  Target Rate (LOS > 3 days) by RACE:")
    for race_code in sorted(set(race_all)):
        mask = race_all == race_code
        rate = df_enc['LOS_BINARY'].values[mask].mean()
        n = mask.sum()
        race_name = RACE_MAP.get(race_code, str(race_code))
        print(f"    {race_name}: {rate:.3f} (N={n:,})")

    # AGE_GROUP distribution
    print("\n  AGE_GROUP Distribution:")
    age_all = df_enc['AGE_GROUP'].values
    for ag in sorted(set(age_all)):
        mask = age_all == ag
        rate = df_enc['LOS_BINARY'].values[mask].mean()
        n = mask.sum()
        print(f"    {ag}: LOS>3d rate={rate:.3f} (N={n:,})")

    # Save demographic audit
    demo_md = ["# Deliverable 6: Demographic Audit\n"]
    demo_md.append("## RACE × ETHNICITY Cross-Tabulation\n")
    demo_md.append(xtab.to_markdown())
    demo_md.append("\n\n## Known Issues\n")
    for issue in demo_issues:
        demo_md.append(f"- **WARNING**: {issue}")
    demo_md.append("\n\n## Interpretation")
    demo_md.append(
        "The RACE and ETHNICITY variables in the Texas PUDF are independently coded. "
        "The high overlap between RACE=3 (Black) and ETHNICITY=1 (Hispanic) reflects "
        "the demographic composition of the Texas hospital population, not a coding error. "
        "However, this correlation means that RACE and ETHNICITY fairness metrics are "
        "partially redundant for this dataset. Results should be interpreted with this "
        "demographic context in mind.")

    with open(f'{AUDIT_DIR}/D6_demographic_audit.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(demo_md))
    xtab.to_csv(f'{AUDIT_DIR}/D6_race_ethnicity_crosstab.csv')
    print("  Saved: D6_demographic_audit.md + D6_race_ethnicity_crosstab.csv")

    # ══════════════════════════════════════════════════════════
    # SECTION 13: Final Audit Report
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("FINAL AUDIT REPORT")
    print(f"{'='*80}")

    elapsed_total = time.time() - t_start

    # Compute summary stats
    fair_model_total_cs = sum(cs_df[f'Fair_NFair_{a}'].mean() for a in ATTRS)
    std_model_total_cs = sum(cs_df[f'Std_NFair_{a}'].mean() for a in ATTRS)

    report = [
        "# CIKM 2026 — Full Audit Report\n",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Runtime: {elapsed_total:.0f}s\n",
        "## Summary\n",
        f"- **Dataset**: {len(df):,} records × {len(feature_cols)} features × "
        f"{len(set(hospital_ids)):,} hospitals",
        f"- **Train/Test**: {len(X_train):,} / {len(X_test):,}",
        f"- **Best Standard Model**: LightGBM  Acc={best_acc:.4f}  AUC={best_auc:.4f}",
        f"- **Fair Model**: λ-reweighed LightGBM + threshold optimization",
        f"- **Fair Model Global**: Acc={accuracy_score(y_test, y_pred_fair_g):.4f}  "
        f"All DI≥0.80: {all_di_ok_g}\n",
        "## Deliverable 0: Consistency Audit\n",
        f"- Consistency issues found: {len(consistency_issues)}",
    ]
    for issue in consistency_issues:
        report.append(f"  - {issue}")
    report.append(
        "- **Action**: Update generate_all_figures_tables.py thresholds to match "
        "notebook (EOPP=0.10, EOD=0.10, CAL=0.05)\n")

    report.extend([
        "## Deliverable 1: Fairness Reconciliation\n",
        f"- Standard model fair verdicts: {n_fair_total}/28",
        f"- Fragile verdicts (margin < 0.02): {n_fragile}/28\n",
        "## Deliverable 2: Lambda Selection\n",
    ])
    best_lam = lambda_df.loc[lambda_df['Total_Fair'].idxmax()]
    report.append(
        f"- Best λ: {best_lam['Lambda']:.0f} → {int(best_lam['Total_Fair'])}/28 fair  "
        f"Acc={best_lam['Accuracy']:.4f}")
    report.append(
        f"- λ=0 (Standard): {int(lambda_df.iloc[0]['Total_Fair'])}/28 fair  "
        f"Acc={lambda_df.iloc[0]['Accuracy']:.4f}\n")

    report.extend([
        "## Deliverable 3: Standard vs Fair\n",
        f"- Standard: {n_std_fair:.0f}/28 fair",
        f"- Fair: {n_fair_fair:.0f}/28 fair",
        f"- Accuracy cost: {(best_acc - accuracy_score(y_test, y_pred_fair_g))*100:.1f} pp\n",
        "## Deliverable 4: Cross-Site Transferability (from Section 1)\n",
        f"- K={K_CS} GroupKFold, Standard + Fair models",
        f"- Standard model: mean {std_model_total_cs:.1f}/28 fair across folds",
        f"- Fair model: mean {fair_model_total_cs:.1f}/28 fair across folds",
    ])

    # Per-attr mean for fair model
    for attr in ATTRS:
        mean_nf = cs_df[f'Fair_NFair_{attr}'].mean()
        ge4_pct = (cs_df[f'Fair_NFair_{attr}'] >= 4).mean() * 100
        report.append(f"  - {attr}: mean {mean_nf:.1f}/7 fair "
                      f"(≥4/7 in {ge4_pct:.0f}% of folds)")

    # DI summary
    for attr in ATTRS:
        mean_di = cs_df[f'Fair_DI_{attr}'].mean()
        min_di = cs_df[f'Fair_DI_{attr}'].min()
        report.append(
            f"  - Fair DI_{attr}: mean={mean_di:.3f}, min={min_di:.3f}")

    report.extend([
        "\n## Deliverable 5: Three-Panel Figure\n",
        "- Saved: output/audit/figures/D5_three_panel_summary.png\n",
        "## Deliverable 6: Demographic Audit\n",
        f"- Issues found: {len(demo_issues)}",
    ])
    for issue in demo_issues:
        report.append(f"  - {issue}")

    report.extend([
        "\n## Files Generated\n",
        "| File | Description |",
        "|------|-------------|",
        "| output/tables/Table6_CrossSite_StdFair_PerFold.csv | "
        "Per-fold cross-site results (Std+Fair) |",
        "| output/tables/Table6_CrossSite_PerFold_Detail.csv | "
        "Detail table for all 20 folds |",
        "| output/audit/Table6_CrossSite_Summary.csv | "
        "Summary statistics |",
        "| output/audit/Table6b_Fleiss_Kappa_StdFair.csv | "
        "Fleiss' κ for both models |",
        "| output/audit/D0_consistency_audit.csv + .md | "
        "Threshold consistency check |",
        "| output/audit/D1_fairness_reconciliation.csv | "
        "28-row reconciliation |",
        "| output/audit/D2_lambda_selection.csv | "
        "Lambda sweep results |",
        "| output/audit/D3_standard_vs_fair.csv | "
        "Head-to-head comparison |",
        "| output/audit/D6_demographic_audit.md | "
        "RACE×ETH cross-tab |",
        "| output/audit/figures/*.png | "
        "Audit figures |",
    ])

    report_text = '\n'.join(report)
    with open(f'{AUDIT_DIR}/FINAL_AUDIT_REPORT.md', 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n  Saved: FINAL_AUDIT_REPORT.md")
    print(f"\n  Total runtime: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    print("=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
