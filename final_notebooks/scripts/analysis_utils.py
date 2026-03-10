"""
analysis_utils.py — Helper functions for fairness analysis pipeline.
Texas-100X LOS Prediction Fairness Study.

Functions:
    compute_metrics()       — Accuracy, F1, AUC, Precision, Recall
    compute_fairness()      — DI, WTPR, SPD, EOD, PPV Ratio per attribute
    bootstrap_ci()          — Bootstrap 95% CI for per-group TPR
    intersectional_audit()  — RACE×SEX intersectional fairness
    hospital_analysis()     — Per-hospital fairness variance
    run_fairlearn_baseline()— Fairlearn ThresholdOptimizer baseline (if available)
"""
import numpy as np
import json, os, time, platform
from datetime import datetime
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_score, recall_score, brier_score_loss)

# ───────────────────────────────────────────────────────────────────────
# Core metrics
# ───────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob=None):
    """Compute standard classification metrics.

    Returns dict with: accuracy, f1, precision, recall, auc, brier, ece.
    """
    m = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred)),
        'recall': float(recall_score(y_true, y_pred)),
    }
    if y_prob is not None:
        m['auc'] = float(roc_auc_score(y_true, y_prob))
        m['brier'] = float(brier_score_loss(y_true, y_prob))
        m['ece'] = float(_expected_calibration_error(y_true, y_prob))
    return m


def _expected_calibration_error(y_true, y_prob, n_bins=10):
    """Expected Calibration Error (ECE) with equal-width bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        avg_conf = y_prob[mask].mean()
        avg_acc = y_true[mask].mean()
        ece += mask.sum() * abs(avg_acc - avg_conf)
    return ece / len(y_true)


# ───────────────────────────────────────────────────────────────────────
# Fairness metrics
# ───────────────────────────────────────────────────────────────────────
def compute_fairness(y_true, y_pred, protected_attr):
    """Compute fairness metrics for one protected attribute.

    Returns dict with: DI, SPD, EOD, WTPR, PPV_Ratio, FPR_diff, EqOdds_proxy,
                       per_group_tpr, per_group_fpr, per_group_sr.
    """
    groups = sorted(set(protected_attr))
    sr, tpr, fpr, ppv = {}, {}, {}, {}

    for g in groups:
        mask = protected_attr == g
        n = mask.sum()
        if n == 0:
            continue
        sr[g] = float(y_pred[mask].mean())

        pos = mask & (y_true == 1)
        neg = mask & (y_true == 0)

        tpr[g] = float(y_pred[pos].mean()) if pos.sum() > 0 else 0.0
        fpr[g] = float(y_pred[neg].mean()) if neg.sum() > 0 else 0.0

        pred_pos = mask & (y_pred == 1)
        ppv[g] = float(y_true[pred_pos].mean()) if pred_pos.sum() > 0 else 0.0

    sr_vals = [v for v in sr.values() if v > 0]
    tpr_vals = list(tpr.values())
    fpr_vals = list(fpr.values())
    ppv_vals = [v for v in ppv.values() if v > 0]

    di = min(sr_vals) / max(sr_vals) if sr_vals and max(sr_vals) > 0 else 0.0
    spd = max(sr_vals) - min(sr_vals) if sr_vals else 0.0
    eod = max(tpr_vals) - min(tpr_vals) if tpr_vals else 0.0
    wtpr = min(tpr_vals) if tpr_vals else 0.0
    ppv_ratio = min(ppv_vals) / max(ppv_vals) if ppv_vals and max(ppv_vals) > 0 else 0.0
    fpr_diff = max(fpr_vals) - min(fpr_vals) if fpr_vals else 0.0
    eq_odds = max(eod, fpr_diff)

    return {
        'DI': round(di, 4),
        'SPD': round(spd, 4),
        'EOD': round(eod, 4),
        'WTPR': round(wtpr, 4),
        'PPV_Ratio': round(ppv_ratio, 4),
        'FPR_diff': round(fpr_diff, 4),
        'EqOdds_proxy': round(eq_odds, 4),
        'per_group_tpr': {str(k): round(v, 4) for k, v in tpr.items()},
        'per_group_fpr': {str(k): round(v, 4) for k, v in fpr.items()},
        'per_group_sr': {str(k): round(v, 4) for k, v in sr.items()},
    }


# ───────────────────────────────────────────────────────────────────────
# Bootstrap CI
# ───────────────────────────────────────────────────────────────────────
def bootstrap_ci(y_true, y_pred, protected_attr, n_bootstrap=200, seed=42):
    """Bootstrap 95% CI for per-group TPR.

    Returns dict: {group: {'mean': float, 'ci_low': float, 'ci_high': float}}.
    """
    rng = np.random.RandomState(seed)
    groups = sorted(set(protected_attr))
    tpr_samples = {g: [] for g in groups}

    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        for g in groups:
            mask = (protected_attr[idx] == g) & (y_true[idx] == 1)
            if mask.sum() > 0:
                tpr_samples[g].append(float(y_pred[idx][mask].mean()))

    result = {}
    for g in groups:
        samples = np.array(tpr_samples[g])
        if len(samples) > 0:
            result[str(g)] = {
                'mean': round(float(samples.mean()), 4),
                'ci_low': round(float(np.percentile(samples, 2.5)), 4),
                'ci_high': round(float(np.percentile(samples, 97.5)), 4),
                'ci_width': round(float(np.percentile(samples, 97.5) - np.percentile(samples, 2.5)), 4),
            }
    return result


# ───────────────────────────────────────────────────────────────────────
# Intersectional audit
# ───────────────────────────────────────────────────────────────────────
def intersectional_audit(y_true, y_pred, race_attr, sex_attr):
    """RACE×SEX intersectional fairness audit.

    Returns dict with per-intersection metrics and overall intersectional DI.
    """
    races = sorted(set(race_attr))
    sexes = sorted(set(sex_attr))

    intersections = {}
    sr_vals = []

    for r in races:
        for s in sexes:
            mask = (race_attr == r) & (sex_attr == s)
            n = int(mask.sum())
            if n < 10:
                continue
            sr = float(y_pred[mask].mean())
            sr_vals.append(sr)

            pos = mask & (y_true == 1)
            tpr = float(y_pred[pos].mean()) if pos.sum() > 0 else 0.0

            intersections[f"{r} x {s}"] = {
                'n': n,
                'selection_rate': round(sr, 4),
                'tpr': round(tpr, 4),
                'positive_rate': round(float(y_true[mask].mean()), 4),
            }

    sr_nonzero = [v for v in sr_vals if v > 0]
    intersectional_di = round(min(sr_nonzero) / max(sr_nonzero), 4) if sr_nonzero and max(sr_nonzero) > 0 else 0.0

    return {
        'intersections': intersections,
        'intersectional_DI': intersectional_di,
        'n_intersections': len(intersections),
    }


# ───────────────────────────────────────────────────────────────────────
# Hospital analysis
# ───────────────────────────────────────────────────────────────────────
def hospital_analysis(y_true, y_pred, y_prob, hospital_ids, protected_attr,
                      n_hospitals=30, min_size=200, seed=42):
    """Per-hospital fairness variance analysis.

    Returns dict with per-hospital metrics and cross-hospital variance.
    """
    rng = np.random.RandomState(seed)
    unique_hosp, counts = np.unique(hospital_ids, return_counts=True)
    large = unique_hosp[counts >= min_size]

    if len(large) > n_hospitals:
        selected = rng.choice(large, n_hospitals, replace=False)
    else:
        selected = large

    per_hospital = {}
    di_vals, f1_vals, acc_vals = [], [], []

    for h in selected:
        mask = hospital_ids == h
        n = int(mask.sum())
        yt = y_true[mask]
        yp = y_pred[mask]

        # Skip if only one class
        if len(set(yt)) < 2:
            continue

        acc = float(accuracy_score(yt, yp))
        f1v = float(f1_score(yt, yp))

        pa = protected_attr[mask]
        if len(set(pa)) < 2:
            di = 1.0
        else:
            fm = compute_fairness(yt, yp, pa)
            di = fm['DI']

        per_hospital[str(h)] = {
            'n': n, 'accuracy': round(acc, 4),
            'f1': round(f1v, 4), 'DI': round(di, 4),
        }
        di_vals.append(di)
        f1_vals.append(f1v)
        acc_vals.append(acc)

    return {
        'n_hospitals': len(per_hospital),
        'per_hospital': per_hospital,
        'DI_mean': round(float(np.mean(di_vals)), 4) if di_vals else None,
        'DI_std': round(float(np.std(di_vals)), 4) if di_vals else None,
        'DI_min': round(float(np.min(di_vals)), 4) if di_vals else None,
        'DI_max': round(float(np.max(di_vals)), 4) if di_vals else None,
        'F1_mean': round(float(np.mean(f1_vals)), 4) if f1_vals else None,
        'Acc_mean': round(float(np.mean(acc_vals)), 4) if acc_vals else None,
    }


# ───────────────────────────────────────────────────────────────────────
# Fairlearn baseline (optional)
# ───────────────────────────────────────────────────────────────────────
def run_fairlearn_baseline(y_true, y_prob, protected_attr, constraint='demographic_parity'):
    """Run Fairlearn ThresholdOptimizer if available.

    Returns dict with post-processed predictions and metrics, or None if unavailable.
    """
    try:
        from fairlearn.postprocessing import ThresholdOptimizer
        from sklearn.base import BaseEstimator, ClassifierMixin

        # Wrap probabilities as a dummy estimator
        class ProbWrapper(BaseEstimator, ClassifierMixin):
            def __init__(self, probs):
                self.probs = probs
                self.classes_ = np.array([0, 1])
            def fit(self, X, y): return self
            def predict(self, X): return (self.probs >= 0.5).astype(int)
            def predict_proba(self, X): return np.column_stack([1-self.probs, self.probs])

        wrapper = ProbWrapper(y_prob)
        to = ThresholdOptimizer(
            estimator=wrapper, constraints=constraint,
            objective='accuracy_score', prefit=True
        )

        X_dummy = np.zeros((len(y_true), 1))
        to.fit(X_dummy, y_true, sensitive_features=protected_attr)
        y_pred_fl = to.predict(X_dummy, sensitive_features=protected_attr)

        metrics = compute_metrics(y_true, y_pred_fl, y_prob)
        fairness = compute_fairness(y_true, y_pred_fl, protected_attr)

        return {'method': 'fairlearn_ThresholdOptimizer',
                'constraint': constraint,
                'metrics': metrics, 'fairness': fairness}

    except ImportError:
        return {'error': 'fairlearn not installed. Install with: pip install fairlearn'}
    except Exception as e:
        return {'error': str(e)}


# ───────────────────────────────────────────────────────────────────────
# Run header and JSON output
# ───────────────────────────────────────────────────────────────────────
def make_run_header():
    """Create a reproducibility header for the results JSON."""
    import sklearn, numpy
    header = {
        'timestamp': datetime.now().isoformat(),
        'python_version': platform.python_version(),
        'numpy_version': numpy.__version__,
        'sklearn_version': sklearn.__version__,
        'platform': platform.platform(),
        'random_seed': 42,
    }
    try:
        import torch
        header['torch_version'] = torch.__version__
        header['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            header['gpu'] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        import xgboost
        header['xgboost_version'] = xgboost.__version__
    except ImportError:
        pass
    try:
        import lightgbm
        header['lightgbm_version'] = lightgbm.__version__
    except ImportError:
        pass
    return header


def save_results(results, filepath):
    """Save results dict to JSON with pretty formatting."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"Results saved to: {filepath}")
