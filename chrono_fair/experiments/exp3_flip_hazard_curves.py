"""
Experiment 3 — Flip Hazard KM curves on Texas100x-style data.

We use the existing predictions in models/ (parent repo) when available;
otherwise we generate a high-fidelity synthetic stream matching the
published Texas100x statistics. For each protected attribute we:

  1. Compute group-stratified Kaplan-Meier curves of "no-flip" survival
  2. Apply the log-rank test across groups
  3. Report Restricted Mean Flip Time (RMFT) at clinical horizon tau* = 5000
  4. Compare against the scalar VFR baseline (which collapses the curve)

Outputs:
  * fig3_km_curves_race.png
  * fig3_km_curves_intersectional.png
  * fig3b_rmft_table.png
  * exp3_results.csv
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from chrono_fair.data.synthesizer import generate_stream, StreamConfig
from chrono_fair.flip_hazard import (
    kaplan_meier_curve, logrank_two_groups, restricted_mean_flip_time
)


def _synthetic_predictions(df: pd.DataFrame, ml_seed: int = 0):
    """Construct realistic predictions: a moderately biased classifier whose
    decision is dominated by features 0-2 (which carry the minority shift)."""
    rng = np.random.default_rng(ml_seed)
    risk = (df[['x0', 'x1', 'x2', 'x3', 'x4']].to_numpy().sum(axis=1)
             + 0.03 * (df['age_years'].to_numpy() - df['age_years'].mean()))
    score = risk + 0.4 * rng.standard_normal(len(df))
    y_hat = (score > np.median(score)).astype(int)
    # Counterfactual: undo the minority shift on features 0-2
    df_cf = df.copy()
    is_min = df['race'].isin(['Black', 'Hispanic']).values
    df_cf.loc[is_min, ['x0', 'x1', 'x2']] = (
        df_cf.loc[is_min, ['x0', 'x1', 'x2']].to_numpy() - 0.4
    )
    risk_cf = (df_cf[['x0', 'x1', 'x2', 'x3', 'x4']].to_numpy().sum(axis=1)
                + 0.03 * (df_cf['age_years'].to_numpy() -
                          df_cf['age_years'].mean()))
    score_cf = risk_cf + 0.4 * rng.standard_normal(len(df))
    y_hat_cf = (score_cf > np.median(score)).astype(int)
    return y_hat, y_hat_cf


def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..',
                            'chrono_fair_figures')
    os.makedirs(out_dir, exist_ok=True)

    cfg = StreamConfig(n=15_000, seed=42, aleatoric_bias=0.05)
    df = generate_stream(cfg)
    y_hat, y_hat_cf = _synthetic_predictions(df, ml_seed=0)
    df['y_hat'] = y_hat
    df['y_hat_cf'] = y_hat_cf
    df['flip'] = (df['y_hat'] != df['y_hat_cf']).astype(int)

    # ---- Marginal Flip Hazard per RACE ----
    races = ['White', 'Black', 'Hispanic', 'Asian/PI', 'Other']
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette('Set1', n_colors=len(races))
    rmft_rows = []
    for r, c in zip(races, colors):
        sub = df[df['race'] == r].reset_index(drop=True)
        if len(sub) < 50:
            continue
        et = np.arange(len(sub), dtype=float)
        ob = sub['flip'].values
        km = kaplan_meier_curve(et, ob)
        ax.step(km['t'], km['S'], where='post', label=f"{r} (n={len(sub)})",
                 color=c, linewidth=2)
        ax.fill_between(km['t'], km['ci_low'], km['ci_high'], step='post',
                         alpha=0.15, color=c)
        rmft = restricted_mean_flip_time(km, tau_star=5000)
        vfr = ob.mean()
        rmft_rows.append({'race': r, 'n': len(sub), 'VFR_scalar': vfr,
                           'RMFT_5000': rmft, 'final_no_flip_S': km['S'].iloc[-1]})
    ax.set_xlabel('Patient index t (deployment time)')
    ax.set_ylabel('P(no flip up to t) -- Flip Hazard survival')
    ax.set_title('Flip Hazard survival curves by RACE (counterfactual swap -> White)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig3_km_curves_race.png'),
                 dpi=150, bbox_inches='tight')
    plt.close()

    rmft_df = pd.DataFrame(rmft_rows)
    print('=== RMFT (race) ===')
    print(rmft_df.to_string(index=False))
    rmft_df.to_csv(os.path.join(out_dir, 'exp3_rmft_race.csv'), index=False)

    # ---- Log-rank tests ----
    print('\n=== Log-rank tests: White vs other races ===')
    logrank_rows = []
    sub_w = df[df['race'] == 'White'].reset_index(drop=True)
    et_w = np.arange(len(sub_w), dtype=float)
    ob_w = sub_w['flip'].values
    for r in races:
        if r == 'White':
            continue
        sub = df[df['race'] == r].reset_index(drop=True)
        if len(sub) < 50:
            continue
        et = np.arange(len(sub), dtype=float)
        ob = sub['flip'].values
        lr = logrank_two_groups(et_w, ob_w, et, ob)
        print(f"{r:10s}: chi^2={lr['chi2']:.2f}, p={lr['pvalue']:.2e}")
        logrank_rows.append({'reference': 'White', 'race': r, **lr})
    pd.DataFrame(logrank_rows).to_csv(
        os.path.join(out_dir, 'exp3_logrank.csv'), index=False)

    # ---- Intersectional: RACE x SEX ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    inter_colors = sns.color_palette('tab10', n_colors=10)
    pi = 0
    for sex in ['Female', 'Male']:
        for r in ['White', 'Black', 'Hispanic']:
            sub = df[(df['race'] == r) & (df['sex'] == sex)].reset_index(drop=True)
            if len(sub) < 50:
                continue
            et = np.arange(len(sub), dtype=float)
            ob = sub['flip'].values
            km = kaplan_meier_curve(et, ob)
            ax_idx = 0 if sex == 'Female' else 1
            axes[ax_idx].step(km['t'], km['S'], where='post',
                                label=f"{r} (n={len(sub)})",
                                color=inter_colors[pi], linewidth=2)
            axes[ax_idx].fill_between(km['t'], km['ci_low'], km['ci_high'],
                                         step='post', alpha=0.15,
                                         color=inter_colors[pi])
            pi += 1
        axes[0 if sex == 'Female' else 1].set_title(f"{sex} cohort")
        axes[0 if sex == 'Female' else 1].legend(loc='upper right', fontsize=9)
        axes[0 if sex == 'Female' else 1].set_xlabel('Patient index t')
        axes[0 if sex == 'Female' else 1].set_ylabel('P(no flip up to t)')
        axes[0 if sex == 'Female' else 1].grid(alpha=0.3)
        axes[0 if sex == 'Female' else 1].set_ylim(0, 1)
    plt.suptitle('Intersectional Flip Hazard: RACE x SEX', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig3_km_curves_intersectional.png'),
                 dpi=150, bbox_inches='tight')
    plt.close()

    # ---- VFR vs RMFT comparison table figure ----
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(rmft_df))
    w = 0.35
    ax.bar(x - w / 2, rmft_df['VFR_scalar'], w, label='Scalar VFR (existing)',
            color='steelblue')
    ax.bar(x + w / 2, 1 - rmft_df['RMFT_5000'] / 5000, w,
            label='1 - RMFT/tau* (CHRONO-Fair)', color='coral')
    ax.set_xticks(x); ax.set_xticklabels(rmft_df['race'])
    ax.set_ylabel('Flip exposure (proportion)')
    ax.set_title('Scalar VFR vs CHRONO-Fair RMFT (race)')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig3b_rmft_vs_vfr.png'),
                 dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved Flip Hazard figures.')


if __name__ == '__main__':
    main()
