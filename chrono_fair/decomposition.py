"""
Week 3a — Aleatoric/Epistemic decomposition of the Flip Hazard.

For a deep ensemble {f_theta_k}_{k=1}^K of classifiers trained on different
bootstrap resamples (or with different random seeds), let

    p_k(x) = f_theta_k(x)     in [0, 1]
    p_bar(x) = (1/K) * sum_k p_k(x)

The total predictive uncertainty H[p_bar] = -p_bar log p_bar - (1-p_bar) log(1-p_bar)
admits the standard Information-Theoretic decomposition (Depeweg 2018;
Houlsby 2011; Smith & Gal 2018; reaffirmed in the ICLR 2025 review of the
aleatoric/epistemic dichotomy):

    H[p_bar]      =        H[E_k p_k]
    Aleatoric     = E_k H[p_k]                  (irreducible)
    Epistemic     = H[E_k p_k] - E_k H[p_k]      (reducible by more data)

We extend this to *Flip Hazard*: define the binary flip indicator under
counterfactual swap, and decompose its hazard rate into:

  * Aleatoric flip rate   : ensemble *agrees* the verdict flips
                            -> data-level / label-level bias
  * Epistemic flip rate   : ensemble *disagrees* on whether it flips
                            -> capacity / sample-size limited

Public API
----------
- ensemble_decompose : returns per-instance (aleatoric, epistemic) flip mass
- recommend_action   : maps the dominant component to a clinical action
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict


def _h(p: np.ndarray) -> np.ndarray:
    """Binary entropy in nats, with safe clipping."""
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -p * np.log(p) - (1 - p) * np.log(1 - p)


def ensemble_decompose(
    probs_actual: np.ndarray,        # shape (K, n) -- ensemble member probs on x
    probs_cf: np.ndarray,             # shape (K, n) -- on counterfactual x
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Decompose flip probability into aleatoric and epistemic mass.

    Approach: for each ensemble member k, compute flip_k = 1{y_k != y_k_cf}.
    Then:
       p_flip_bar(i)     = mean_k flip_k(i)                  -- ensemble-mean flip
       aleatoric_flip(i) = mean_k flip_k(i) * (1 - flip_k(i)) -- always 0 for binary,
                            so we use the *probability* form instead:
       aleatoric(i)      = mean_k H[p_k(i)]                 -- per-member uncertainty
       epistemic(i)      = Var_k[p_k(i)]                    -- between-member spread
    The flip-rate is split proportionally to where each member's flip lies.
    """
    K, n = probs_actual.shape
    y_pred_act = (probs_actual >= threshold).astype(int)
    y_pred_cf = (probs_cf >= threshold).astype(int)
    flip_k = (y_pred_act != y_pred_cf).astype(int)         # (K, n)
    p_flip_bar = flip_k.mean(axis=0)                        # (n,)

    # Per-instance ensemble probability of "active" class
    p_act_bar = probs_actual.mean(axis=0)
    aleatoric_unc = _h(probs_actual).mean(axis=0)           # E_k H[p_k]
    total_unc = _h(p_act_bar)                                # H[E_k p_k]
    epistemic_unc = np.clip(total_unc - aleatoric_unc, 0, None)

    # Split flip mass: proportion attributable to each component, normalised so
    # aleatoric_flip + epistemic_flip = p_flip_bar.
    denom = np.clip(aleatoric_unc + epistemic_unc, 1e-9, None)
    aleatoric_share = aleatoric_unc / denom
    epistemic_share = epistemic_unc / denom
    aleatoric_flip = p_flip_bar * aleatoric_share
    epistemic_flip = p_flip_bar * epistemic_share

    return pd.DataFrame({
        'p_flip_bar': p_flip_bar,
        'aleatoric_unc': aleatoric_unc,
        'epistemic_unc': epistemic_unc,
        'aleatoric_flip': aleatoric_flip,
        'epistemic_flip': epistemic_flip,
        'ensemble_disagreement': flip_k.std(axis=0),
    })


def aggregate_by_group(
    decomp: pd.DataFrame, group: pd.Series
) -> pd.DataFrame:
    """Group-wise mean of aleatoric/epistemic flip mass."""
    d = decomp.copy()
    d['group'] = group.values
    g = d.groupby('group').agg(
        flip_rate=('p_flip_bar', 'mean'),
        aleatoric_flip=('aleatoric_flip', 'mean'),
        epistemic_flip=('epistemic_flip', 'mean'),
        n=('p_flip_bar', 'size'),
    ).reset_index()
    g['aleatoric_share'] = g['aleatoric_flip'] / g['flip_rate'].clip(1e-9)
    g['epistemic_share'] = g['epistemic_flip'] / g['flip_rate'].clip(1e-9)
    return g


def recommend_action(row: pd.Series) -> Dict[str, str]:
    """Map the dominant component to an automatic governance recommendation."""
    if row['flip_rate'] < 1e-3:
        return {'cause': 'none', 'action': 'no action -- below detection floor'}
    if row['epistemic_share'] > 0.6:
        return {
            'cause': 'epistemic',
            'action': (f"Collect ~{int(np.ceil(row['n'] * 1.5))} additional "
                        f"samples from this stratum; retrain with class-"
                        f"reweighted loss."),
        }
    if row['aleatoric_share'] > 0.6:
        return {
            'cause': 'aleatoric',
            'action': (f"Audit label pipeline for this stratum: "
                        f"investigate coder bias and feature missingness; "
                        f"model retraining alone will NOT close this gap."),
        }
    return {
        'cause': 'mixed',
        'action': ("Run a joint mitigation: targeted data audit + "
                    "ensemble-stabilised retraining."),
    }


__all__ = ['ensemble_decompose', 'aggregate_by_group', 'recommend_action']
