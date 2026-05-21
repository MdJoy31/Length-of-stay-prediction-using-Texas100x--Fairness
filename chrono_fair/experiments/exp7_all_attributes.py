"""
Experiment 7. Four-attribute audit: race, ethnicity, sex, age group.

A single stream of n = 15,000 patients is generated. Predictions and
counterfactual predictions are produced as in Experiment 3. For each of the
four protected attributes documented in the parent repository, the Flip
Hazard estimator is run, the marginal flip rate per group is reported, and
the log-rank chi-square statistic against the modal group is computed.

The output is a single figure with four sub-panels and a CSV summary that the
paper's Discussion references when it claims multi-attribute generalisation.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from chrono_fair.data.synthesizer import generate_stream, StreamConfig
from chrono_fair.flip_hazard import (
    kaplan_meier_curve, logrank_two_groups
)


def _predictions(df, seed=0):
    rng = np.random.default_rng(seed)
    risk = (df[['x0', 'x1', 'x2', 'x3', 'x4']].to_numpy().sum(axis=1)
             + 0.03 * (df['age_years'].to_numpy() - df['age_years'].mean()))
    score = risk + 0.4 * rng.standard_normal(len(df))
    y_hat = (score > np.median(score)).astype(int)
    df_cf = df.copy()
    is_min = df['race'].isin(['Black', 'Hispanic']).values
    df_cf.loc[is_min, ['x0', 'x1', 'x2']] = (
        df_cf.loc[is_min, ['x0', 'x1', 'x2']].to_numpy() - 0.4
    )
    risk_cf = (df_cf[['x0', 'x1', 'x2', 'x3', 'x4']].to_numpy().sum(axis=1)
                + 0.03 * (df_cf['age_years'].to_numpy() - df_cf['age_years'].mean()))
    score_cf = risk_cf + 0.4 * rng.standard_normal(len(df))
    y_hat_cf = (score_cf > np.median(score)).astype(int)
    return y_hat, y_hat_cf


def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..',
                            'chrono_fair_figures')
    os.makedirs(out_dir, exist_ok=True)
    df = generate_stream(StreamConfig(n=15_000, seed=42, aleatoric_bias=0.05))
    y_hat, y_hat_cf = _predictions(df, seed=0)
    df['flip'] = (y_hat != y_hat_cf).astype(int)

    attrs = [('race', ['White', 'Black', 'Hispanic', 'Asian/PI', 'Other'],
              'White'),
              ('ethnicity', ['Non-Hispanic', 'Hispanic'], 'Non-Hispanic'),
              ('sex', ['Female', 'Male'], 'Female'),
              ('age_group',
               ['Pediatric', 'Young Adult', 'Middle-aged', 'Elderly'],
               'Middle-aged')]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    summary_rows = []
    for (ax, (attr_name, levels, ref)) in zip(axes.flat, attrs):
        palette = sns.color_palette('Set1', n_colors=len(levels))
        for g, c in zip(levels, palette):
            sub = df[df[attr_name] == g].reset_index(drop=True)
            if len(sub) < 30:
                continue
            et = np.arange(len(sub), dtype=float)
            ob = sub['flip'].values
            km = kaplan_meier_curve(et, ob)
            ax.step(km['t'], km['S'], where='post',
                     label=f"{g} (n={len(sub)})", color=c, linewidth=2)
            ax.fill_between(km['t'], km['ci_low'], km['ci_high'],
                              step='post', alpha=0.15, color=c)
            ref_sub = df[df[attr_name] == ref].reset_index(drop=True)
            if g != ref and len(ref_sub) > 30:
                et_r = np.arange(len(ref_sub), dtype=float)
                ob_r = ref_sub['flip'].values
                lr = logrank_two_groups(et_r, ob_r, et, ob)
            else:
                lr = {'chi2': 0.0, 'pvalue': 1.0}
            summary_rows.append({'attribute': attr_name, 'group': g,
                                  'reference': ref, 'n': len(sub),
                                  'flip_rate': float(ob.mean()),
                                  'logrank_chi2': float(lr['chi2']),
                                  'logrank_p': float(lr['pvalue'])})
        ax.set_title(f"{attr_name} (reference = {ref})")
        ax.set_xlabel('Patient index t')
        ax.set_ylabel('P(no flip up to t)')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)
    plt.suptitle('Four-attribute Flip Hazard audit (Section 8.8)',
                  fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig8_four_attribute_audit.png'),
                 dpi=150, bbox_inches='tight')
    plt.close()

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(out_dir, 'exp7_four_attributes.csv'),
                    index=False)
    print('=== Four-attribute audit summary ===')
    print(summary.to_string(index=False))
    print('Wrote fig8_four_attribute_audit.png')


if __name__ == '__main__':
    main()
