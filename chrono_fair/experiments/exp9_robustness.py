"""
Experiment 9. Robustness stress tests for the e-process monitor.

Three reviewer-requested stress tests are run on the synthesiser.

  (F) Baseline miscalibration. The true pre-drift flip rate is 0.05. The
      monitor is given a baseline rho_0 in {0.03, 0.04, 0.05, 0.06, 0.07}.
      Under H_0 (no drift) the false-alarm rate is recorded. Under H_1
      (drift to 0.15 at t = 6000) the detection rate and delay are recorded.

  (G) Small-subgroup reliability. A single cell is monitored with per-cell
      stream length n in {200, 500, 1000, 2000, 5000, 10000}. The post-drift
      rate is 0.15 with onset at the stream midpoint. Detection rate, delay,
      and the H_0 false-alarm rate are recorded per size.

  (H) Label / counterfactual delay. The flip indicator is delayed by d
      patients (the counterfactual verdict is only resolved d arrivals
      later). d in {0, 50, 100, 250, 500}. Detection delay is recorded.

All numbers are produced by the released synthesiser. No external data.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from chrono_fair.e_process import EProcessMonitor


def _run_stream(seed, rho0_true, rho0_assumed, n, drift_at, rho1, delay=0):
    rng = np.random.default_rng(seed)
    mon = EProcessMonitor(rho0=rho0_assumed, alpha=0.05)
    buffer = []
    for t in range(n):
        p = rho0_true if t < drift_at else rho1
        z = int(rng.random() < p)
        buffer.append(z)
        if len(buffer) > delay:
            mon.update(buffer.pop(0))
            if mon.alarm_at is not None:
                break
    return mon.alarm_at


def stress_miscalibration(out_dir):
    rows = []
    for rho0_assumed in [0.03, 0.04, 0.05, 0.06, 0.07]:
        fa = 0
        for seed in range(100):
            a = _run_stream(seed, 0.05, rho0_assumed, 10000, 10000, 0.05)
            if a is not None:
                fa += 1
        delays, det = [], 0
        for seed in range(100):
            a = _run_stream(seed + 999, 0.05, rho0_assumed, 12000, 6000, 0.15)
            if a is not None and a >= 6000:
                delays.append(a - 6000)
                det += 1
        rows.append({'rho0_assumed': rho0_assumed,
                      'false_alarm_rate_H0': fa / 100,
                      'detection_rate_H1': det / 100,
                      'mean_delay_H1': float(np.mean(delays)) if delays else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'exp9F_miscalibration.csv'), index=False)
    print('=== (F) Baseline miscalibration (true rho_0 = 0.05) ===')
    print(df.to_string(index=False))
    return df


def stress_small_subgroup(out_dir):
    rows = []
    for n in [200, 500, 1000, 2000, 5000, 10000]:
        drift_at = n // 2
        fa = 0
        for seed in range(100):
            a = _run_stream(seed, 0.05, 0.05, n, n, 0.05)
            if a is not None:
                fa += 1
        delays, det = [], 0
        for seed in range(100):
            a = _run_stream(seed + 999, 0.05, 0.05, n, drift_at, 0.15)
            if a is not None and a >= drift_at:
                delays.append(a - drift_at)
                det += 1
        rows.append({'cell_n': n, 'false_alarm_rate_H0': fa / 100,
                      'detection_rate_H1': det / 100,
                      'mean_delay_H1': float(np.mean(delays)) if delays else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'exp9G_small_subgroup.csv'), index=False)
    print('=== (G) Small-subgroup reliability (post-drift rate 0.15) ===')
    print(df.to_string(index=False))
    return df


def stress_label_delay(out_dir):
    rows = []
    for d in [0, 50, 100, 250, 500]:
        delays, det = [], 0
        for seed in range(100):
            a = _run_stream(seed + 999, 0.05, 0.05, 12000, 6000, 0.15, delay=d)
            if a is not None and a >= 6000:
                # Wall-clock detection delay adds the label delay d, since the
                # counterfactual verdict for patient t resolves only at t + d.
                delays.append(a - 6000 + d)
                det += 1
        rows.append({'label_delay': d, 'detection_rate': det / 100,
                      'mean_wallclock_delay': float(np.mean(delays)) if delays else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'exp9H_label_delay.csv'), index=False)
    print('=== (H) Label / counterfactual delay ===')
    print(df.to_string(index=False))
    return df


def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..',
                            'chrono_fair_figures')
    os.makedirs(out_dir, exist_ok=True)
    dfF = stress_miscalibration(out_dir)
    dfG = stress_small_subgroup(out_dir)
    dfH = stress_label_delay(out_dir)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    axes[0].plot(dfF['rho0_assumed'], dfF['false_alarm_rate_H0'],
                  marker='o', color='crimson', label='False alarm (H0)')
    axes[0].plot(dfF['rho0_assumed'], dfF['detection_rate_H1'],
                  marker='s', color='steelblue', label='Detection (H1)')
    axes[0].axvline(0.05, ls='--', color='gray', alpha=0.6,
                     label='True rho_0')
    axes[0].set_xlabel('Assumed baseline rho_0 (true = 0.05)')
    axes[0].set_ylabel('Rate over 100 streams')
    axes[0].set_title('(F) Baseline miscalibration')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(dfG['cell_n'], dfG['detection_rate_H1'],
                  marker='o', color='steelblue', label='Detection rate')
    axes[1].plot(dfG['cell_n'], dfG['false_alarm_rate_H0'],
                  marker='s', color='crimson', label='False-alarm rate')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Per-cell stream length n (log scale)')
    axes[1].set_ylabel('Rate over 100 streams')
    axes[1].set_title('(G) Small-subgroup reliability')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(dfH['label_delay'], dfH['mean_wallclock_delay'],
                  marker='o', color='darkgreen')
    axes[2].set_xlabel('Label / counterfactual delay (patients)')
    axes[2].set_ylabel('Mean detection delay (patients)')
    axes[2].set_title('(H) Label-delay effect')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig10_robustness.png'), dpi=150,
                 bbox_inches='tight')
    plt.close()
    print('Wrote fig10_robustness.png')


if __name__ == '__main__':
    main()
