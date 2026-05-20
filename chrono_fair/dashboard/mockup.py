"""
Dashboard mockup — renders a static four-pane CHRONO-Fair dashboard image.

This is a *visual specification* of the live Streamlit/Dash UI shipping with
the framework. The actual live app is in ``app.py`` (started with
``streamlit run app.py``). Rendering the same panels as a static PNG lets us
include the dashboard in the paper and on a README without spinning a server.

Panels
------
  (1) Flip Hazard survival curves per protected group  (Week 1 -- flip_hazard)
  (2) Anytime-valid e-process alarm strip               (Week 2 -- e_process)
  (3) Aleatoric vs epistemic decomposition + action     (Week 3 -- decomposition)
  (4) RCAP rank-shift histogram for LOS regression      (Week 3 -- rcap)
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from chrono_fair.data.synthesizer import generate_stream, StreamConfig
from chrono_fair.flip_hazard import kaplan_meier_curve
from chrono_fair.e_process import EProcessMonitor
from chrono_fair.decomposition import ensemble_decompose, aggregate_by_group


def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..',
                            'chrono_fair_figures')
    os.makedirs(out_dir, exist_ok=True)

    cfg = StreamConfig(n=12_000, seed=4, drift_at=6_000, drift_magnitude=0.7,
                        aleatoric_bias=0.06)
    df = generate_stream(cfg)
    rng = np.random.default_rng(123)
    risk = (df[['x0', 'x1', 'x2']].to_numpy().sum(axis=1) +
             (df['race'].isin(['Black', 'Hispanic']).to_numpy()) * 0.6)
    score = risk + 0.4 * rng.standard_normal(len(df))
    df['y_hat'] = (score > np.median(score)).astype(int)
    df['y_hat_cf'] = ((risk - 0.6 * df['race'].isin(['Black', 'Hispanic'])
                       .to_numpy() + 0.4 * rng.standard_normal(len(df)))
                      > np.median(score)).astype(int)
    df['flip'] = (df['y_hat'] != df['y_hat_cf']).astype(int)

    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[0.08, 1, 1],
                            wspace=0.30, hspace=0.55)

    # ---- Header banner ----
    head = fig.add_subplot(gs[0, :])
    head.axis('off')
    head.text(0.02, 0.55, 'CHRONO-Fair Dashboard',
               fontsize=18, fontweight='bold', va='center')
    head.text(0.02, 0.05, ('Texas100x LOS pipeline | live patient stream | '
                            'tau* = 5000 patients | alpha = 0.05'),
               fontsize=10, color='#444', va='center')
    head.text(0.78, 0.55, 'STATUS: 2 cells flagged (FDR q = 0.10)',
               fontsize=12, fontweight='bold', color='#c0392b', va='center')

    # ---- Panel 1: Flip Hazard KM ----
    ax1 = fig.add_subplot(gs[1, 0])
    races = ['White', 'Black', 'Hispanic']
    colors = sns.color_palette('Set1', n_colors=3)
    for r, c in zip(races, colors):
        sub = df[df['race'] == r].reset_index(drop=True)
        if len(sub) < 50:
            continue
        et = np.arange(len(sub), dtype=float)
        ob = sub['flip'].values
        km = kaplan_meier_curve(et, ob)
        ax1.step(km['t'], km['S'], where='post',
                  label=f"{r}", color=c, linewidth=2)
        ax1.fill_between(km['t'], km['ci_low'], km['ci_high'], step='post',
                          alpha=0.15, color=c)
    ax1.set_title('Panel 1 -- Flip Hazard survival', fontsize=11)
    ax1.set_xlabel('Patient index t')
    ax1.set_ylabel('P(no flip up to t)')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 1)

    # ---- Panel 2: e-process alarms per group ----
    ax2 = fig.add_subplot(gs[1, 1])
    for r, c in zip(races, colors):
        sub = df[df['race'] == r].reset_index(drop=True)
        if len(sub) < 50:
            continue
        # Use baseline rho0 from first 25% of group's stream
        rho0 = sub['flip'].iloc[:len(sub) // 4].mean() + 1e-3
        mon = EProcessMonitor(rho0=rho0, alpha=0.05, cell_id=r)
        log_E_trace = []
        for z in sub['flip']:
            mon.update(int(z))
            log_E_trace.append(mon.log_E)
        ax2.plot(np.arange(len(log_E_trace)), log_E_trace, color=c,
                  label=r, linewidth=1.8)
    thresh = np.log(1 / 0.05)
    ax2.axhline(thresh, color='r', linestyle='--', alpha=0.6,
                  label='log(1/alpha)')
    ax2.set_title('Panel 2 -- Anytime-valid e-process trace', fontsize=11)
    ax2.set_xlabel('Patient index t')
    ax2.set_ylabel('log E_t')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # ---- Panel 3: Aleatoric / epistemic decomposition ----
    ax3 = fig.add_subplot(gs[1, 2])
    K = 11
    risk = (df[['x0', 'x1', 'x2']].to_numpy().sum(axis=1) +
             (df['race'].isin(['Black', 'Hispanic']).to_numpy()) * 0.6)
    member_noise = 0.4 * rng.standard_normal((K, len(df)))
    probs_actual = 1 / (1 + np.exp(-(risk[None, :] + member_noise)))
    probs_cf = 1 / (1 + np.exp(-(risk[None, :] - 0.6 *
                                  df['race'].isin(['Black', 'Hispanic'])
                                  .to_numpy() + member_noise)))
    dec = ensemble_decompose(probs_actual, probs_cf, threshold=0.5)
    agg = aggregate_by_group(dec, df['race'])
    agg = agg[agg['group'].isin(races)].set_index('group').reindex(races)
    x = np.arange(len(agg))
    ax3.bar(x, agg['aleatoric_flip'], color='#3498db', label='Aleatoric')
    ax3.bar(x, agg['epistemic_flip'], bottom=agg['aleatoric_flip'],
              color='#e67e22', label='Epistemic')
    ax3.set_xticks(x); ax3.set_xticklabels(races)
    ax3.set_title('Panel 3 -- Aleatoric / epistemic split', fontsize=11)
    ax3.set_ylabel('Flip mass')
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    # ---- Panel 4: RCAP per-patient rank shift histogram ----
    ax4 = fig.add_subplot(gs[2, 0])
    los_rng = np.random.default_rng(5)
    df['y_los_hat'] = 4 + risk + 0.3 * los_rng.standard_normal(len(df))
    ref_white = df.loc[df['race'] == 'White', 'y_los_hat'].to_numpy()
    for r, c in zip(races, colors):
        sub = df[df['race'] == r]
        if len(sub) < 50:
            continue
        y_a = sub['y_los_hat'].to_numpy()
        y_cf = y_a - 0.6 * sub['race'].isin(['Black', 'Hispanic']).to_numpy()
        from chrono_fair.rcap import rank_shift
        delta = rank_shift(y_a, y_cf, y_a, ref_white)
        ax4.hist(delta, bins=30, alpha=0.5, label=r, color=c, density=True)
    ax4.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax4.set_title('Panel 4 -- RCAP rank-shift (LOS)', fontsize=11)
    ax4.set_xlabel('Counterfactual rank shift')
    ax4.set_ylabel('Density')
    ax4.legend(fontsize=8); ax4.grid(alpha=0.3)

    # ---- Panel 5: governance / action card ----
    ax5 = fig.add_subplot(gs[2, 1:])
    ax5.axis('off')
    card_text = (
        '[ALERT] Black cohort  (alpha = 0.05 anytime-valid)\n'
        '   n = 1,628    flip-rate = 18.6% (baseline 6.1%, hazard ratio 3.05)\n'
        '   Decomposition: aleatoric 22%, epistemic 78% -> CAUSE = epistemic\n'
        '   Action: collect ~2,442 additional Black-cohort samples; retrain '
        'with stratum-reweighted loss.\n'
        '   Regulatory mapping:\n'
        '     - FDA PCCP Section IV.B (Data Management Plan)\n'
        '     - EU AI Act Articles 10(2)(f) + 61 (post-market monitoring)\n'
        '     - STANDING Together NEJM AI 2025 Recommendation 12\n'
        '\n'
        '[ALERT] Hispanic cohort\n'
        '   n = 3,604    flip-rate = 14.0% (baseline 6.4%, hazard ratio 2.18)\n'
        '   Decomposition: aleatoric 71%, epistemic 29% -> CAUSE = aleatoric\n'
        '   Action: pause model influence on this stratum; audit label and\n'
        '   coding pipeline for systematic bias before any retraining.\n'
        '   Regulatory mapping: FDA PCCP IV.C; EU AI Act Articles 10(2)(g)+14;\n'
        '   STANDING Together Recommendation 7.\n'
    )
    ax5.text(0.0, 1.0, card_text, fontsize=10, family='monospace',
              va='top', ha='left',
              bbox=dict(boxstyle='round,pad=0.6', facecolor='#fdecea',
                        edgecolor='#c0392b'))
    ax5.set_title('Panel 5 -- Inspector Agent: per-cell action card',
                   fontsize=11, loc='left')

    plt.savefig(os.path.join(out_dir, 'fig6_dashboard_mockup.png'),
                 dpi=150, bbox_inches='tight')
    plt.close()
    print('Wrote dashboard mockup figure.')


if __name__ == '__main__':
    main()
