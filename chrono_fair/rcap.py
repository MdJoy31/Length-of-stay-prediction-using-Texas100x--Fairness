"""
Regression Counterfactual Allocation Parity (RCAP).

For a regressor f(x, a) predicting a continuous output (length of stay),
let F_hat_{Yhat|A=a} be the empirical CDF of the predicted output within
group a. For patient i in group a, the predicted-rank positions under the
factual and the counterfactual protected attribute are

    u_i(a)  = F_hat_{Yhat|A=a }( f(x_i, a ) )      in [0, 1]
    u_i(a') = F_hat_{Yhat|A=a'}( f(x_i, a') )      in [0, 1]

Each u_i is the patient's percentile position in the group-conditional
predicted-output distribution. The per-patient rank shift is

    Delta_rank(x_i, a, a') = u_i(a) - u_i(a').

The aggregate RCAP statistic is the Wasserstein-1 distance between the two
rank-position distributions:

    RCAP(a, a') = W_1( {u_i(a) : i in a}, {u_i(a') : i in a} ).

RCAP measures how far a patient's predicted length-of-stay rank position
shifts under the protected-attribute counterfactual. Rank position, not
raw error alone, determines prioritisation in length-of-stay-based bed
allocation. RCAP is reported on the [0, 1] rank scale; a value of 0.0163
corresponds to a shift of 1.63 percentile points.

Public API
----------
- rank_positions     : per-patient ECDF rank position u_i
- rank_shift         : per-patient rank shift u_i(a) - u_i(a')
- rcap_w1            : W_1 between two rank-position distributions
- rcap_w1_ci         : bootstrap (W_1, ci_low, ci_high)
- intersectional_rcap : RCAP across all group pairs of an attribute
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Callable, Tuple


def _ecdf_rank(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Empirical CDF value (rank position) in [0, 1] for each query point.

    Returns F_hat(query), where F_hat is the empirical CDF of `reference`.
    """
    s = np.sort(reference)
    idx = np.searchsorted(s, query, side='right')
    return idx / max(1, len(s))


def rank_positions(y_hat: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Per-patient rank position u_i = F_hat_reference(y_hat)."""
    return _ecdf_rank(reference, y_hat)


def rank_shift(
    y_hat_actual: np.ndarray,
    y_hat_cf: np.ndarray,
    ref_dist_actual: np.ndarray,
    ref_dist_cf: np.ndarray,
) -> np.ndarray:
    """Per-patient rank shift Delta_rank = u_i(a) - u_i(a').

    u_i(a)  ranks the factual prediction against ref_dist_actual.
    u_i(a') ranks the counterfactual prediction against ref_dist_cf.
    """
    u_a = _ecdf_rank(ref_dist_actual, y_hat_actual)
    u_ap = _ecdf_rank(ref_dist_cf, y_hat_cf)
    return u_a - u_ap


def rcap_w1(u_a: np.ndarray, u_aprime: np.ndarray) -> float:
    """Wasserstein-1 distance between two rank-position distributions.

    For one-dimensional samples the W_1 distance equals the mean absolute
    difference of the order statistics, which is computed exactly here by
    sorting both samples. This is the distance between the distributions
    {u_i(a)} and {u_i(a')}, not the mean of the paired per-patient shift.
    """
    if len(u_a) == 0 or len(u_aprime) == 0:
        return 0.0
    try:
        from scipy.stats import wasserstein_distance
        return float(wasserstein_distance(u_a, u_aprime))
    except Exception:
        # Exact 1-D fallback: equalise lengths by quantile interpolation
        n = max(len(u_a), len(u_aprime))
        q = (np.arange(n) + 0.5) / n
        return float(np.mean(np.abs(
            np.quantile(u_a, q) - np.quantile(u_aprime, q))))


def rcap_w1_ci(u_a: np.ndarray, u_aprime: np.ndarray,
                n_boot: int = 1000, alpha: float = 0.05,
                seed: int = 0) -> Tuple[float, float, float]:
    """Bootstrap (W_1, ci_low, ci_high) for RCAP.

    The two rank-position arrays are resampled with replacement; the W_1
    distance is recomputed each replicate.
    """
    point = rcap_w1(u_a, u_aprime)
    if len(u_a) == 0 or len(u_aprime) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        ra = rng.choice(u_a, size=len(u_a), replace=True)
        rb = rng.choice(u_aprime, size=len(u_aprime), replace=True)
        boots[b] = rcap_w1(ra, rb)
    return (point,
            float(np.quantile(boots, alpha / 2)),
            float(np.quantile(boots, 1 - alpha / 2)))


# Backwards-compatible alias for the bootstrap helper used by older code.
def wasserstein_rcap_ci(delta: np.ndarray, n_boot: int = 1000,
                          alpha: float = 0.05,
                          seed: int = 0) -> Tuple[float, float, float]:
    """Deprecated. Bootstrap CI for the mean absolute paired rank shift.

    Retained so older experiment scripts keep running. New code should use
    rcap_w1_ci, which computes the W_1 distance between the two
    rank-position distributions rather than the mean paired difference.
    """
    if len(delta) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.choice(delta, size=len(delta), replace=True)
        boots[b] = float(np.mean(np.abs(s)))
    return (float(np.mean(np.abs(delta))),
            float(np.quantile(boots, alpha / 2)),
            float(np.quantile(boots, 1 - alpha / 2)))


def intersectional_rcap(
    df: pd.DataFrame,
    y_hat_col: str,
    sensitive_col: str,
    counterfactual_fn: Callable,
) -> pd.DataFrame:
    """All-pairs RCAP across the levels of `sensitive_col`.

    Parameters
    ----------
    df : DataFrame with the factual predictions in `y_hat_col`.
    counterfactual_fn : callable (df_sub, swap_to) -> predicted values when
        the sensitive attribute is set to `swap_to`.

    For each ordered pair (a, a') the function computes the rank-position
    arrays u(a) and u(a') for the patients in group a and returns the W_1
    RCAP statistic between them.
    """
    groups = sorted(df[sensitive_col].unique())
    rows = []
    for a in groups:
        mask_a = (df[sensitive_col] == a).values
        if mask_a.sum() < 5:
            continue
        y_a = df.loc[mask_a, y_hat_col].values
        ref_a = y_a
        for ap in groups:
            if a == ap:
                continue
            df_sub = df.loc[mask_a].copy()
            y_a_cf = counterfactual_fn(df_sub, swap_to=ap)
            mask_ap = (df[sensitive_col] == ap).values
            ref_ap = (df.loc[mask_ap, y_hat_col].values
                       if mask_ap.sum() > 0 else y_a_cf)
            u_a = _ecdf_rank(ref_a, y_a)
            u_ap = _ecdf_rank(ref_ap, y_a_cf)
            rows.append({
                'group_a': a,
                'group_aprime': ap,
                'n': int(mask_a.sum()),
                'mean_rank_shift': float(np.mean(u_a - u_ap)),
                'abs_mean_rank_shift': float(np.mean(np.abs(u_a - u_ap))),
                'rcap_W1': rcap_w1(u_a, u_ap),
            })
    return pd.DataFrame(rows)


__all__ = ['rank_positions', 'rank_shift', 'rcap_w1', 'rcap_w1_ci',
            'wasserstein_rcap_ci', 'intersectional_rcap']
