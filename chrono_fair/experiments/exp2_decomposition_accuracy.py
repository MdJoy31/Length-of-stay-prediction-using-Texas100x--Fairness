"""
Experiment 2 — Decomposition Accuracy.

We construct ensembles of K=15 binary classifiers whose per-instance
probability vectors {p_k(x)}_{k=1}^K are *known by construction* to be
either:

  (A) Aleatoric-dominated  — all members agree on a value near 0.5
                              (inherent class overlap; more data won't help)
  (B) Epistemic-dominated   — members are confidently far apart
                              (capacity-limited; more data WILL help)
  (C) Mixed                 — half of each

A second validation track uses the synthesizer + a Random-Forest ensemble
to show the same separation arises naturally when minority groups are
under-sampled.

Outputs:
  * fig2_decomposition_confusion.png : confusion matrix + scenario boxplots
  * fig2b_synthetic_validation.png    : ML-driven validation
  * exp2_results.csv                  : raw classifications
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from chrono_fair.data.synthesizer import generate_stream, StreamConfig
from chrono_fair.decomposition import ensemble_decompose, aggregate_by_group
from chrono_fair.inspector_agent import recommend_action_label


# ============================================================================
# Track 1 — Controlled validation of the decomposition formula itself
# ============================================================================

def _make_controlled_probs(scenario: str, n: int = 1000, K: int = 15,
                            seed: int = 0):
    """Construct (probs_actual, probs_cf) so that the ground-truth cause
    of any decision flip is known by construction.
    """
    rng = np.random.default_rng(seed)
    if scenario == 'aleatoric':
        # Members agree, but the agreed-on probability sits near 0.5 -- and
        # the counterfactual swap pushes it across the threshold for many x.
        mu_actual = rng.uniform(0.35, 0.55, size=n)
        probs_actual = np.tile(mu_actual, (K, 1)) + 0.01 * rng.standard_normal((K, n))
        mu_cf = mu_actual + rng.choice([-1, 1], size=n) * 0.15
        probs_cf = np.tile(mu_cf, (K, 1)) + 0.01 * rng.standard_normal((K, n))
    elif scenario == 'epistemic':
        # Members confidently disagree: half ~0.03, half ~0.97. Mean ~0.5
        # but each individual member is highly confident.
        member_bias = rng.choice([-0.47, 0.47], size=K)[:, None]
        base = 0.5 + member_bias + 0.01 * rng.standard_normal((K, n))
        probs_actual = np.clip(base, 0.005, 0.995)
        # Counterfactual flips the disagreement pattern -> verdict flip
        probs_cf = np.clip(1 - probs_actual + 0.01 * rng.standard_normal((K, n)),
                            0.005, 0.995)
    else:  # mixed
        a = _make_controlled_probs('aleatoric', n=n // 2, K=K, seed=seed)
        e = _make_controlled_probs('epistemic', n=n // 2, K=K, seed=seed + 1)
        probs_actual = np.concatenate([a[0], e[0]], axis=1)
        probs_cf = np.concatenate([a[1], e[1]], axis=1)
    return np.clip(probs_actual, 1e-3, 1 - 1e-3), np.clip(probs_cf, 1e-3, 1 - 1e-3)


def controlled_trial(scenario: str, seed: int) -> dict:
    probs_actual, probs_cf = _make_controlled_probs(scenario, n=1000, K=15,
                                                     seed=seed)
    dec = ensemble_decompose(probs_actual, probs_cf, threshold=0.5)
    avg = pd.Series({
        'flip_rate': dec['p_flip_bar'].mean(),
        'aleatoric_share': dec['aleatoric_flip'].sum() / max(1e-9, dec['p_flip_bar'].sum()),
        'epistemic_share': dec['epistemic_flip'].sum() / max(1e-9, dec['p_flip_bar'].sum()),
        'n': len(dec),
    })
    predicted = recommend_action_label(avg)
    return {'truth': scenario, 'predicted': predicted,
            'flip_rate': avg['flip_rate'],
            'aleatoric_share': avg['aleatoric_share'],
            'epistemic_share': avg['epistemic_share']}


# ============================================================================
# Track 2 — ML-driven validation: aleatoric = biased labels,
#                                  epistemic = under-sampled minority
# ============================================================================

def _rf_ensemble(df_train: pd.DataFrame, K: int = 11,
                  minority_undersample: float = 1.0, seed: int = 0):
    """Ensemble in which the minority slice differs across members.

    For epistemic scenarios we feed each member a *disjoint, non-overlapping*
    slice of the minority training data (after under-sampling). That maximises
    between-member variance on the minority subgroup -- the exact signal the
    decomposition needs to identify epistemic flips.
    """
    rng = np.random.default_rng(seed)
    feat = [c for c in df_train.columns if c.startswith('x')] + ['age_years']
    minority = df_train['race'].isin(['Black', 'Hispanic']).values
    minority_idx = np.where(minority)[0]
    majority_idx = np.where(~minority)[0]
    rng.shuffle(minority_idx)
    n_min_total = max(K * 4, int(len(minority_idx) * minority_undersample))
    n_min_total = min(n_min_total, len(minority_idx))
    minority_pool = minority_idx[:n_min_total]
    slices = np.array_split(minority_pool, K)
    models = []
    for k in range(K):
        chosen_min = slices[k]
        if len(chosen_min) < 3:
            continue
        chosen_maj = rng.choice(majority_idx, size=len(majority_idx), replace=True)
        idx = np.concatenate([chosen_min, chosen_maj])
        X = df_train[feat].to_numpy()[idx]
        y = df_train['y_ext'].to_numpy()[idx]
        if len(np.unique(y)) < 2:
            continue
        m = DecisionTreeClassifier(max_depth=6, random_state=k * 37)
        m.fit(X, y)
        models.append(m)
    return models, feat


def _rf_probs(models, feat, df, swap_race=None):
    df_x = df[feat].copy()
    if swap_race is not None:
        is_min = df['race'].isin(['Black', 'Hispanic']).values
        if swap_race == 'White':
            df_x.loc[is_min, ['x0', 'x1', 'x2']] = (
                df_x.loc[is_min, ['x0', 'x1', 'x2']].to_numpy() - 0.4
            )
    probs = np.stack([m.predict_proba(df_x.to_numpy())[:, 1] for m in models])
    return probs


def ml_trial(scenario: str, seed: int) -> dict:
    cfg = StreamConfig(
        n=8000, seed=seed,
        aleatoric_bias=0.25 if scenario in ('aleatoric', 'mixed') else 0.0,
    )
    df = generate_stream(cfg)
    df_train, df_test = df.iloc[:4000], df.iloc[4000:]
    undersample = 0.05 if scenario in ('epistemic', 'mixed') else 1.0
    models, feat = _rf_ensemble(df_train, K=11,
                                  minority_undersample=undersample, seed=seed)
    pa = _rf_probs(models, feat, df_test)
    pc = _rf_probs(models, feat, df_test, swap_race='White')
    dec = ensemble_decompose(pa, pc, threshold=0.5)
    agg = aggregate_by_group(dec, df_test['race'])
    minority = agg[agg.group.isin(['Black', 'Hispanic'])]
    if len(minority) == 0:
        return {'truth': scenario, 'predicted': 'none'}
    avg = pd.Series({
        'flip_rate': minority['flip_rate'].mean(),
        'epistemic_share': minority['epistemic_share'].mean(),
        'aleatoric_share': minority['aleatoric_share'].mean(),
        'n': int(minority['n'].sum()),
    })
    return {'truth': scenario,
            'predicted': recommend_action_label(avg),
            'flip_rate': avg['flip_rate'],
            'aleatoric_share': avg['aleatoric_share'],
            'epistemic_share': avg['epistemic_share']}


def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..',
                            'chrono_fair_figures')
    os.makedirs(out_dir, exist_ok=True)

    # ---- Track 1: controlled formula validation ----
    rows = []
    for scenario in ['aleatoric', 'epistemic', 'mixed']:
        for seed in range(30):
            rows.append(controlled_trial(scenario, seed=seed))
    df1 = pd.DataFrame(rows)
    df1.to_csv(os.path.join(out_dir, 'exp2_results.csv'), index=False)

    cm = pd.crosstab(df1['truth'], df1['predicted'], dropna=False)
    for c in ['aleatoric', 'epistemic', 'mixed', 'none']:
        if c not in cm.columns:
            cm[c] = 0
    cm = cm[['aleatoric', 'epistemic', 'mixed', 'none']]
    acc1 = (df1['truth'] == df1['predicted']).mean()
    print('=== Track 1: Controlled validation ===')
    print(cm)
    print(f'Accuracy: {acc1:.1%}')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
    axes[0].set_title(f'Decomposition confusion (controlled, n=30/scenario)\n'
                       f'Overall accuracy = {acc1:.1%}', fontsize=11)
    axes[0].set_xlabel('Predicted by CHRONO-Fair')
    axes[0].set_ylabel('Ground-truth scenario')

    sns.boxplot(data=df1, x='truth', y='epistemic_share',
                 order=['aleatoric', 'epistemic', 'mixed'], ax=axes[1],
                 hue='truth', legend=False, palette='viridis')
    axes[1].axhline(0.6, ls='--', c='r', alpha=0.7,
                     label='Epistemic threshold')
    axes[1].set_title('Epistemic share by scenario (controlled)', fontsize=11)
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig2_decomposition_confusion.png'),
                 dpi=150, bbox_inches='tight')
    plt.close()

    # ---- Track 2: ML-driven validation ----
    rows = []
    for scenario in ['aleatoric', 'epistemic', 'mixed']:
        for seed in range(15):
            rows.append(ml_trial(scenario, seed=seed))
    df2 = pd.DataFrame(rows)
    cm2 = pd.crosstab(df2['truth'], df2['predicted'], dropna=False)
    for c in ['aleatoric', 'epistemic', 'mixed', 'none']:
        if c not in cm2.columns:
            cm2[c] = 0
    cm2 = cm2[['aleatoric', 'epistemic', 'mixed', 'none']]
    acc2 = (df2['truth'] == df2['predicted']).mean()
    print('=== Track 2: ML-driven validation ===')
    print(cm2)
    print(f'Accuracy: {acc2:.1%}')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens', ax=axes[0], cbar=False)
    axes[0].set_title(f'Decomposition confusion (ML-driven, n=15/scenario)\n'
                       f'Accuracy = {acc2:.1%}', fontsize=11)
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Truth')

    sns.scatterplot(data=df2, x='aleatoric_share', y='epistemic_share',
                     hue='truth', style='truth', ax=axes[1], s=80,
                     palette='Set1')
    axes[1].plot([0, 1], [1, 0], ':', c='gray', alpha=0.5)
    axes[1].set_title('Decomposition shares (ML-driven)', fontsize=11)
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig2b_synthetic_validation.png'),
                 dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    main()
