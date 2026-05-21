"""
Experiment 6. Sensitivity analysis: detection delay against drift magnitude.

Under H_1 the post-drift rate is rho_1 in {0.06, 0.08, 0.10, 0.12, 0.15, 0.20}
while rho_0 = 0.05. For each magnitude, 30 Monte-Carlo runs of length
n = 12,000 with drift onset at t = 6,000 are generated. The empirical
detection rate and mean delay of the CHRONO-Fair e-process (alpha = 0.05) are
reported. The figure shows the trade-off between drift magnitude and time to
alarm.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from chrono_fair.e_process import EProcessMonitor


def one_run(seed: int, rho1: float, drift_at: int = 6_000,
             n: int = 12_000, rho0: float = 0.05) -> dict:
    rng = np.random.default_rng(seed)
    mon = EProcessMonitor(rho0=rho0, alpha=0.05)
    for t in range(n):
        p = rho0 if t < drift_at else rho1
        mon.update(int(rng.random() < p))
        if mon.alarm_at is not None and mon.alarm_at >= drift_at:
            break
    delay = (None if mon.alarm_at is None or mon.alarm_at < drift_at
              else mon.alarm_at - drift_at)
    return {'rho1': rho1, 'seed': seed, 'delay': delay,
             'detected': delay is not None}


def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..',
                            'chrono_fair_figures')
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for rho1 in [0.06, 0.08, 0.10, 0.12, 0.15, 0.20]:
        for seed in range(30):
            rows.append(one_run(seed=seed, rho1=rho1))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'exp6_sensitivity.csv'), index=False)

    agg = (df.groupby('rho1')
            .agg(detection_rate=('detected', 'mean'),
                  mean_delay=('delay', lambda s: s.dropna().mean()),
                  median_delay=('delay', lambda s: s.dropna().median()),
                  n_detected=('detected', 'sum'),
                  n_total=('detected', 'size'))
            .reset_index())
    print('=== Sensitivity to drift magnitude ===')
    print(agg.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    axes[0].plot(agg['rho1'], agg['detection_rate'], marker='o',
                  linewidth=2, color='steelblue')
    axes[0].axhline(1.0, ls='--', color='gray', alpha=0.5)
    axes[0].set_xlabel('Post-drift rate rho_1 (baseline rho_0 = 0.05)')
    axes[0].set_ylabel('Detection rate (30 runs each)')
    axes[0].set_title('Detection rate against drift magnitude')
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(0, 1.05)

    sns.boxplot(data=df.dropna(subset=['delay']),
                 x='rho1', y='delay', ax=axes[1], hue='rho1',
                 legend=False, palette='viridis')
    axes[1].set_title('Detection delay against drift magnitude')
    axes[1].set_ylabel('Patients to alarm (alarms only)')
    axes[1].set_xlabel('rho_1')
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig7_sensitivity.png'), dpi=150,
                 bbox_inches='tight')
    plt.close()
    print('Wrote fig7_sensitivity.png')


if __name__ == '__main__':
    main()
