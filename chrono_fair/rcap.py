"""
Week 3b — Regression Counterfactual Allocation Parity (RCAP).

For a regression model f(x, a) predicting (e.g.) length of stay y, RCAP
measures how many *rank positions* in the group-conditional distribution
a patient would shift if their sensitive attribute were counterfactually
swapped.

Formally, for sensitive attributes a, a' and prediction y_hat = f(x, a):

    Delta_RCAP(x, a, a') = F^{-1}_{Y|A=a}(y_hat(x,a))
                          - F^{-1}_{Y|A=a'}(y_hat(x,a'))

where F_{Y|A=a} is the empirical CDF of predictions within group a.

The aggregate RCAP statistic between two groups is the Wasserstein-1 distance
between the counterfactual rank distributions:

    RCAP(a, a') = W_1( {Delta_RCAP(x_i, a, a') : i in a} )

Intuitively, RCAP answers: "if we flipped the protected attribute, by how
many bed-allocation positions would this patient be reassigned?" This is the
clinically meaningful question for triage and resource allocation in LOS
prediction -- traditional regression-fairness metrics (group-MAE gap) do
not capture it.

Public API
----------
- rank_shift               : per-patient counterfactual rank shift
- wasserstein_rcap         : aggregate RCAP across a group pair
- intersectional_rcap      : RCAP across all group pairs of an attribute
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Callable, Dict


def _ecdf_rank(values: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Empirical CDF rank in [0, 1] for each query against `values`."""
    s = np.sort(values)
    idx = np.searchsorted(s, query, side='right')
    return idx / max(1, len(s))


def rank_shift(
    y_hat_actual: np.ndarray,
    y_hat_cf: np.ndarray,
    ref_dist_actual: np.ndarray,
    ref_dist_cf: np.ndarray,
) -> np.ndarray:
    """Per-patient counterfactual rank shift Delta_RCAP."""
    r_actual = _ecdf_rank(ref_dist_actual, y_hat_actual)
    r_cf = _ecdf_rank(ref_dist_cf, y_hat_cf)
    return r_actual - r_cf


def wasserstein_rcap(delta: np.ndarray) -> float:
    """W_1 distance between the delta distribution and the symmetric null at 0."""
    if len(delta) == 0:
        return 0.0
    return float(np.mean(np.abs(delta)))


def wasserstein_rcap_ci(delta: np.ndarray, n_boot: int = 1000,
                          alpha: float = 0.05,
                          seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap (mean, ci_low, ci_high) for RCAP W_1."""
    if len(delta) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(delta, size=len(delta), replace=True)
        boots[b] = np.mean(np.abs(sample))
    return (float(np.mean(np.abs(delta))),
            float(np.quantile(boots, alpha / 2)),
            float(np.quantile(boots, 1 - alpha / 2)))


def intersectional_rcap(
    df: pd.DataFrame,
    y_hat_col: str,
    sensitive_col: str,
    counterfactual_fn: Callable,
) -> pd.DataFrame:
    """All-pairs RCAP across levels of `sensitive_col`.

    Parameters
    ----------
    df : DataFrame containing original features + predictions in `y_hat_col`.
    counterfactual_fn : callable (df_sub, swap_to) -> predicted values when
        the sensitive attribute is set to `swap_to`.
    """
    groups = sorted(df[sensitive_col].unique())
    rows = []
    for a in groups:
        mask_a = (df[sensitive_col] == a).values
        if mask_a.sum() < 5:
            continue
        y_a = df.loc[mask_a, y_hat_col].values
        for ap in groups:
            if a == ap:
                continue
            df_sub = df.loc[mask_a].copy()
            y_a_cf = counterfactual_fn(df_sub, swap_to=ap)
            mask_ap = (df[sensitive_col] == ap).values
            ref_a = df.loc[mask_a, y_hat_col].values
            ref_ap = df.loc[mask_ap, y_hat_col].values if mask_ap.sum() > 0 else y_a_cf
            delta = rank_shift(y_a, y_a_cf, ref_a, ref_ap)
            rows.append({
                'group_a': a,
                'group_aprime': ap,
                'n': int(mask_a.sum()),
                'mean_rank_shift': float(np.mean(delta)),
                'abs_mean_rank_shift': float(np.mean(np.abs(delta))),
                'rcap_W1': wasserstein_rcap(delta),
            })
    return pd.DataFrame(rows)


__all__ = ['rank_shift', 'wasserstein_rcap', 'wasserstein_rcap_ci',
            'intersectional_rcap']
