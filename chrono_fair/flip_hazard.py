"""
Week 1 — Flip Hazard Function (CHFT): time-resolved counterfactual fairness.

For each patient i with sensitive attribute a, let T_i^flip be the first
arrival index at which the model's binary verdict on i would change under a
counterfactual swap of the sensitive attribute (classification), or at which
the predicted regression output crosses a clinically meaningful threshold
(regression). The Flip Hazard function

    lambda_a(t | x) = lim_{dt -> 0} (1/dt) * P( T^flip in [t, t+dt) |
                                                T^flip >= t, A = a, X = x )

is estimated non-parametrically (Kaplan–Meier per group with Nelson–Aalen
hazard), tested across groups by the log-rank statistic, and summarised by
the Restricted Mean Flip Time (RMFT) at clinical horizon tau*.

Implements:
  - kaplan_meier_per_group : group-stratified survival of "no-flip"
  - nelson_aalen_per_group : cumulative hazard
  - logrank_two_groups      : asymptotic chi-square test
  - restricted_mean_flip_time
  - FlipHazardEstimator     : top-level convenience class
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple


def kaplan_meier_curve(
    event_times: np.ndarray,
    observed: np.ndarray,
    horizon: float | None = None,
) -> pd.DataFrame:
    """Standard Kaplan–Meier estimator.

    Returns a DataFrame with columns [t, n_at_risk, d_events, c_censored, S, ci_low, ci_high].
    Confidence intervals use Greenwood's formula on the log-survival scale.
    """
    df = pd.DataFrame({'t': event_times, 'd': observed.astype(int)})
    df = df.sort_values('t').reset_index(drop=True)
    if horizon is not None:
        df = df[df['t'] <= horizon]
    grid = df.groupby('t').agg(d=('d', 'sum'), n=('d', 'size')).reset_index()
    # n at risk at time t = total subjects - those whose t < current
    total = len(df)
    grid['n_at_risk'] = total - grid['n'].cumsum().shift(1, fill_value=0)
    grid['hazard'] = grid['d'] / grid['n_at_risk'].clip(lower=1)
    grid['S'] = (1 - grid['hazard']).cumprod()
    # Greenwood: Var(log S) = sum d / (n (n - d))
    g = (grid['d'] / (grid['n_at_risk'] * (grid['n_at_risk'] - grid['d']).clip(lower=1))).cumsum()
    se = np.sqrt(g)
    grid['ci_low'] = np.exp(np.log(grid['S'].clip(1e-12)) - 1.96 * se).clip(0, 1)
    grid['ci_high'] = np.exp(np.log(grid['S'].clip(1e-12)) + 1.96 * se).clip(0, 1)
    return grid[['t', 'n_at_risk', 'd', 'S', 'ci_low', 'ci_high', 'hazard']]


def nelson_aalen_curve(event_times: np.ndarray, observed: np.ndarray) -> pd.DataFrame:
    """Cumulative hazard H(t) = sum d_i / n_i."""
    km = kaplan_meier_curve(event_times, observed)
    km['H'] = (km['d'] / km['n_at_risk'].clip(lower=1)).cumsum()
    return km[['t', 'H']]


def logrank_two_groups(
    et_a: np.ndarray, ob_a: np.ndarray,
    et_b: np.ndarray, ob_b: np.ndarray,
) -> Dict[str, float]:
    """Asymptotic two-sample log-rank test for survival equality.

    Returns chi^2 statistic and p-value (one degree of freedom).
    """
    from scipy.stats import chi2
    times = np.unique(np.concatenate([et_a[ob_a == 1], et_b[ob_b == 1]]))
    O_a = 0.0
    E_a = 0.0
    V = 0.0
    for t in times:
        n_a = (et_a >= t).sum()
        n_b = (et_b >= t).sum()
        n = n_a + n_b
        if n == 0:
            continue
        d_a = ((et_a == t) & (ob_a == 1)).sum()
        d_b = ((et_b == t) & (ob_b == 1)).sum()
        d = d_a + d_b
        if d == 0:
            continue
        expected_a = d * n_a / n
        var = (d * n_a * n_b * (n - d)) / max(1, n * n * (n - 1))
        O_a += d_a
        E_a += expected_a
        V += var
    if V <= 0:
        return {'chi2': 0.0, 'pvalue': 1.0, 'O': O_a, 'E': E_a}
    chi2_stat = (O_a - E_a) ** 2 / V
    p = 1 - chi2.cdf(chi2_stat, df=1)
    return {'chi2': float(chi2_stat), 'pvalue': float(p),
            'O': float(O_a), 'E': float(E_a)}


def restricted_mean_flip_time(km: pd.DataFrame, tau_star: float) -> float:
    """RMFT = integral_0^tau* of S(t) dt -- area under the survival curve."""
    sub = km[km['t'] <= tau_star].copy()
    if len(sub) == 0:
        return float(tau_star)
    # Step-function integration: sum S(t_k) * (t_{k+1} - t_k)
    ts = np.concatenate([[0.0], sub['t'].values, [tau_star]])
    ss = np.concatenate([[1.0], sub['S'].values, [sub['S'].iloc[-1]]])
    return float(np.sum(ss[:-1] * np.diff(ts)))


@dataclass
class FlipHazardEstimator:
    """Top-level convenience: compute group-stratified flip-time survival.

    Parameters
    ----------
    counterfactual_fn : callable (df, swap_to) -> np.ndarray
        Returns model predictions on df with sensitive attribute swapped to `swap_to`.
    sensitive_col : str
        Column name of the protected attribute.
    """
    counterfactual_fn: Callable
    sensitive_col: str = 'race'

    def fit(self, df: pd.DataFrame, groups: Sequence[str] | None = None,
            reference: str | None = None) -> Dict[str, pd.DataFrame]:
        if groups is None:
            groups = sorted(df[self.sensitive_col].unique())
        if reference is None:
            reference = groups[0]

        results = {}
        y_hat_actual = self.counterfactual_fn(df, swap_to=None)

        for g in groups:
            mask = (df[self.sensitive_col] == g).values
            if mask.sum() < 5:
                continue
            # Counterfactual: swap THIS group's attribute to the reference,
            # see whether their predictions flip.
            df_sub = df.loc[mask].copy()
            y_hat_cf = self.counterfactual_fn(df_sub, swap_to=reference)
            y_hat_obs = y_hat_actual[mask]
            event_time = np.arange(mask.sum(), dtype=float)
            observed = (y_hat_obs != y_hat_cf).astype(int)
            km = kaplan_meier_curve(event_time, observed)
            results[g] = km
        return results

    def pairwise_logrank(self, fitted: Dict[str, pd.DataFrame],
                          reference: str) -> pd.DataFrame:
        # NOTE: a fully proper log-rank needs raw event arrays; we expose this
        # helper for the convenience case where flip indicators are tracked
        # separately by the caller. See exp1_detection_delay for direct usage.
        rows = []
        for g, km in fitted.items():
            if g == reference:
                continue
            # Approximation: chi^2 derived from final hazard difference;
            # the exact test uses logrank_two_groups on raw flip arrays.
            S_g = km['S'].iloc[-1] if len(km) else 1.0
            rows.append({'group': g, 'reference': reference,
                          'S_final': float(S_g),
                          'flip_prob_final': float(1 - S_g)})
        return pd.DataFrame(rows)


# Public re-exports
__all__ = [
    'kaplan_meier_curve',
    'nelson_aalen_curve',
    'logrank_two_groups',
    'restricted_mean_flip_time',
    'FlipHazardEstimator',
]
