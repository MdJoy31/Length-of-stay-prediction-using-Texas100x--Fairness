"""
Experiment 4 — RCAP on Length-of-Stay regression.

We train an LR regressor on synthesised Texas100x-style LOS data and audit
RCAP (Regression Counterfactual Allocation Parity) versus the standard
group-MAE-gap baseline. RCAP measures, for each minority patient, how many
quantile positions they shift in the predicted-LOS distribution under a
counterfactual race swap. Aggregated by Wasserstein-1.

Outputs:
  * fig4_rcap_rank_shift.png    : histogram of per-patient rank shifts by group
  * fig4b_rcap_vs_mae_gap.png   : RCAP vs naive MAE gap by group
  * exp4_results.csv             : aggregated statistics
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from chrono_fair.data.synthesizer import generate_stream, StreamConfig
from chrono_fair.rcap import intersectional_rcap, rank_shift


def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..',
                            'chrono_fair_figures')
    os.makedirs(out_dir, exist_ok=True)

    cfg = StreamConfig(n=20_000, seed=11, aleatoric_bias=0.05)
    df = generate_stream(cfg)
    df_tr = df.iloc[:8000]
    df_te = df.iloc[8000:].reset_index(drop=True)
    feat = [c for c in df.columns if c.startswith('x')] + ['age_years']

    model = Ridge(alpha=1.0).fit(df_tr[feat].to_numpy(), df_tr['y_los'].to_numpy())
    df_te['y_hat'] = model.predict(df_te[feat].to_numpy())

    def cf_fn(df_sub, swap_to):
        d = df_sub.copy()
        is_min = d['race'].isin(['Black', 'Hispanic']).values
        if swap_to == 'White':
            d.loc[is_min, ['x0', 'x1', 'x2']] = (
                d.loc[is_min, ['x0', 'x1', 'x2']].to_numpy() - 0.4
            )
        elif swap_to in ('Black', 'Hispanic'):
            non_min = ~is_min
            d.loc[non_min, ['x0', 'x1', 'x2']] = (
                d.loc[non_min, ['x0', 'x1', 'x2']].to_numpy() + 0.4
            )
        return model.predict(d[feat].to_numpy())

    rcap = intersectional_rcap(df_te, y_hat_col='y_hat', sensitive_col='race',
                                counterfactual_fn=cf_fn)
    print('=== RCAP (race) ===')
    print(rcap.to_string(index=False))
    rcap.to_csv(os.path.join(out_dir, 'exp4_rcap.csv'), index=False)

    # ---- Bootstrap 95% CI for RCAP against White reference ----
    # RCAP is the Wasserstein-1 distance between the rank-position
    # distributions u(a) and u(a'), computed by rcap_w1_ci.
    from chrono_fair.rcap import rcap_w1_ci, rank_positions
    races_for_ci = ['White', 'Black', 'Hispanic', 'Asian/PI', 'Other']
    ci_rows = []
    ref_white_y = df_te.loc[df_te['race'] == 'White', 'y_hat'].to_numpy()
    for r in races_for_ci:
        sub = df_te[df_te['race'] == r]
        if len(sub) < 5:
            continue
        y_a = sub['y_hat'].to_numpy()
        y_cf = cf_fn(sub, swap_to='White')
        u_a = rank_positions(y_a, y_a)               # factual rank position
        u_ap = rank_positions(y_cf, ref_white_y)     # counterfactual rank position
        seed = int(np.frombuffer(r.encode()[:4].ljust(4, b'_'),
                                  dtype=np.uint32)[0]) % 2**31
        w1, lo, hi = rcap_w1_ci(u_a, u_ap, n_boot=1000, alpha=0.05, seed=seed)
        ci_rows.append({'race': r, 'n': int(len(sub)),
                         'rcap_W1': float(w1),
                         'ci_low_95': float(lo),
                         'ci_high_95': float(hi)})
    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(os.path.join(out_dir, 'exp4_rcap_ci.csv'), index=False)
    print('=== RCAP 95% CI (against White) ===')
    print(ci_df.to_string(index=False))

    # ---- Per-patient rank shifts: histograms by group ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    races = ['White', 'Black', 'Hispanic', 'Asian/PI']
    palette = sns.color_palette('Set1', n_colors=len(races))
    ref_dist_white = df_te.loc[df_te['race'] == 'White', 'y_hat'].to_numpy()
    for r, c in zip(races, palette):
        sub = df_te[df_te['race'] == r]
        if len(sub) < 50:
            continue
        y_a = sub['y_hat'].to_numpy()
        y_cf = cf_fn(sub, swap_to='White')
        delta = rank_shift(y_a, y_cf, y_a, ref_dist_white)
        axes[0].hist(delta, bins=40, alpha=0.5, label=f"{r} (n={len(sub)})",
                      color=c, density=True)
    axes[0].axvline(0, color='k', linestyle='--', alpha=0.6,
                     label='No counterfactual shift')
    axes[0].set_xlabel('Rank shift Delta_RCAP')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Per-patient counterfactual rank-shift distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Cumulative
    for r, c in zip(races, palette):
        sub = df_te[df_te['race'] == r]
        if len(sub) < 50:
            continue
        y_a = sub['y_hat'].to_numpy()
        y_cf = cf_fn(sub, swap_to='White')
        delta = rank_shift(y_a, y_cf, y_a, ref_dist_white)
        srt = np.sort(np.abs(delta))
        axes[1].step(srt, np.arange(len(srt)) / max(1, len(srt)),
                      where='post', label=r, color=c, linewidth=2)
    axes[1].set_xlabel('|Delta_RCAP|')
    axes[1].set_ylabel('Empirical CDF')
    axes[1].set_title('Cumulative |rank shift| by group')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig4_rcap_rank_shift.png'),
                 dpi=150, bbox_inches='tight')
    plt.close()

    # ---- RCAP vs MAE-gap by group ----
    naive = []
    for r in races:
        sub = df_te[df_te['race'] == r]
        if len(sub) < 50:
            continue
        mae = mean_absolute_error(sub['y_los'], sub['y_hat'])
        naive.append({'race': r, 'mae': mae})
    naive_df = pd.DataFrame(naive)
    rcap_white = rcap[rcap['group_aprime'] == 'White'].copy()
    rcap_white = rcap_white.rename(columns={'group_a': 'race'})
    joined = naive_df.merge(rcap_white[['race', 'rcap_W1']], on='race', how='left')
    joined['mae_gap_vs_white'] = joined['mae'] - joined.loc[joined['race'] == 'White', 'mae'].iloc[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(joined))
    w = 0.4
    ax.bar(x - w / 2, joined['mae_gap_vs_white'].fillna(0), w,
            label='MAE gap vs White (days)', color='steelblue')
    ax2 = ax.twinx()
    ax2.bar(x + w / 2, joined['rcap_W1'].fillna(0), w,
             label='RCAP W_1 (rank units)', color='coral')
    ax.set_xticks(x); ax.set_xticklabels(joined['race'])
    ax.set_ylabel('MAE gap (days)', color='steelblue')
    ax2.set_ylabel('RCAP W_1 (rank units)', color='coral')
    ax.set_title('Naive MAE gap underreports allocation disparity captured by RCAP')
    fig.legend(loc='upper left', bbox_to_anchor=(0.13, 0.95))
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig4b_rcap_vs_mae_gap.png'),
                 dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved RCAP figures.')
    joined.to_csv(os.path.join(out_dir, 'exp4_rcap_vs_mae.csv'), index=False)
    print('=== RCAP vs MAE gap ===')
    print(joined.to_string(index=False))


if __name__ == '__main__':
    main()
