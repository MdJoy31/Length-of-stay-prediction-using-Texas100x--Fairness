"""
Experiment E / 12. Full module ablation of the e-process monitor.

Each design choice in the e-process is removed in turn and the effect on
the H_0 false-alarm rate and the H_1 detection delay is recorded. The
ablations are:

  - full          : the framework as specified (one-sided GROW, warm-up 100,
                    shrinkage 0.25).
  - no_warmup     : warm-up set to 0; the monitor bets from patient 1.
  - no_grow       : a fixed bet lambda = 0.5 instead of the GROW rule.
  - no_shrink     : GROW with shrinkage 1.0 instead of 0.25.
  - two_sided     : the one-sided clip is removed; lambda may go negative.

The two_sided ablation reproduces the defect that the baseline-miscalibration
stress test exposed and motivates the one-sided design.

Each setting is run on 100 H_0 streams and 100 H_1 streams of length
n = 12,000 with drift onset at t = 6,000 and a post-drift rate of 0.15.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from chrono_fair.e_process import EProcessMonitor


def _run(mode, seed, rho0=0.05, rho1=0.15, n=12_000, drift_at=6_000):
    rng = np.random.default_rng(seed)
    warm = 0 if mode == 'no_warmup' else 100
    use_grow = mode != 'no_grow'
    mon = EProcessMonitor(rho0=rho0, alpha=0.05, use_grow=use_grow,
                            lambda_init=0.5)
    mon._abl_warm = warm
    mon._abl_shrink = 1.0 if mode == 'no_shrink' else 0.25
    mon._abl_two_sided = (mode == 'two_sided')
    for t in range(n):
        p = rho0 if t < drift_at else rho1
        z = int(rng.random() < p)
        _ablated_update(mon, z)
        if mon.alarm_at is not None:
            break
    return mon.alarm_at


def _ablated_update(mon, z):
    """Replicates EProcessMonitor.update with ablation hooks."""
    lo, hi = mon.feasible()
    warm = getattr(mon, '_abl_warm', 100)
    shrink = getattr(mon, '_abl_shrink', 0.25)
    two_sided = getattr(mon, '_abl_two_sided', False)
    lam_lo = lo + 1e-6 if two_sided else 0.0
    lam_hi = hi - 1e-6
    if mon.use_grow:
        if mon.n >= warm:
            z_bar = mon.sum_z / max(1, mon.n)
            raw = (z_bar - mon.rho0) / max(1e-9, mon.rho0 * (1 - mon.rho0))
            lam = float(np.clip(shrink * raw, lam_lo, lam_hi))
        else:
            lam = 0.0
    else:
        lam = float(np.clip(mon.lambda_init, lam_lo, lam_hi))
    factor = max(1.0 + lam * (z - mon.rho0), 1e-12)
    mon.log_E += float(np.log(factor))
    mon.n += 1
    mon.sum_z += z
    if mon.log_E >= np.log(1.0 / mon.alpha) and mon.alarm_at is None:
        mon.alarm_at = mon.n


def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..',
                            'chrono_fair_figures')
    os.makedirs(out_dir, exist_ok=True)
    modes = ['full', 'no_warmup', 'no_grow', 'no_shrink', 'two_sided']
    rows = []
    for mode in modes:
        fa = 0
        for seed in range(100):
            a = _run(mode, seed, rho1=0.05, drift_at=12_000)  # H0
            if a is not None:
                fa += 1
        delays, det = [], 0
        for seed in range(100):
            a = _run(mode, seed + 5000)                       # H1
            if a is not None and a >= 6000:
                delays.append(a - 6000)
                det += 1
        rows.append({'ablation': mode,
                      'false_alarm_rate_H0': fa / 100,
                      'detection_rate_H1': det / 100,
                      'mean_delay_H1': float(np.mean(delays)) if delays else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'exp12_ablation.csv'), index=False)
    print('=== Experiment E/12: full module ablation ===')
    print(df.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    axes[0].bar(df['ablation'], df['false_alarm_rate_H0'], color='crimson')
    axes[0].axhline(0.05, ls='--', color='black', label='nominal alpha = 0.05')
    axes[0].set_ylabel('H0 false-alarm rate (100 streams)')
    axes[0].set_title('False-alarm rate per ablation')
    axes[0].legend(); axes[0].grid(alpha=0.3, axis='y')
    axes[0].tick_params(axis='x', rotation=20)
    delay_plot = df['mean_delay_H1'].fillna(0)
    axes[1].bar(df['ablation'], delay_plot, color='steelblue')
    axes[1].set_ylabel('H1 mean detection delay (patients)')
    axes[1].set_title('Detection delay per ablation')
    axes[1].grid(alpha=0.3, axis='y')
    axes[1].tick_params(axis='x', rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig13_ablation_full.png'), dpi=150,
                 bbox_inches='tight')
    plt.close()
    print('Wrote fig13_ablation_full.png')


if __name__ == '__main__':
    main()
