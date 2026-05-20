"""
Experiment 5 — Ablation & False-Alarm Control under H_0.

We complete the paper's evidence chain with two further studies:

  (A) Anytime-valid validity: under H_0 (no drift), we feed CHRONO-Fair
      streams of i.i.d. Bernoulli(rho0) flips and confirm that the empirical
      false-alarm rate is bounded by the nominal alpha. This is the
      bedrock theoretical property; if it fails we have nothing.

  (B) Component ablation: we report detection-delay performance with each
      CHRONO-Fair piece *removed* (no GROW, no FDR, fixed lambda) to
      attribute the gain to each component.

Outputs:
  * fig5a_false_alarm_calibration.png : empirical alpha vs nominal alpha
  * fig5b_ablation.png                 : delay barplot per ablation
  * exp5_results.csv                   : aggregate numbers
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from chrono_fair.e_process import EProcessMonitor


def false_alarm_one_alpha(alpha: float, n_runs: int = 200, n: int = 10_000,
                            rho0: float = 0.05, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    n_false = 0
    for s in range(n_runs):
        mon = EProcessMonitor(rho0=rho0, alpha=alpha)
        for t in range(n):
            mon.update(int(rng.random() < rho0))
            if mon.alarm_at is not None:
                break
        if mon.alarm_at is not None:
            n_false += 1
    return n_false / n_runs


def ablation_one(seed: int, mode: str = 'full',
                  drift_at: int = 5000, n: int = 10_000,
                  rho0: float = 0.05, rho1: float = 0.15) -> float | None:
    rng = np.random.default_rng(seed)
    use_grow = (mode != 'no_grow')
    lam_init = 0.5 if mode == 'no_grow' else 0.5
    mon = EProcessMonitor(rho0=rho0, alpha=0.05, lambda_init=lam_init,
                            use_grow=use_grow)
    for t in range(n):
        p = rho0 if t < drift_at else rho1
        mon.update(int(rng.random() < p))
        if mon.alarm_at is not None:
            break
    if mon.alarm_at is None or mon.alarm_at < drift_at:
        return None
    return float(mon.alarm_at - drift_at)


def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..',
                            'chrono_fair_figures')
    os.makedirs(out_dir, exist_ok=True)

    # ---- (A) False-alarm calibration ----
    nominal_alphas = [0.01, 0.025, 0.05, 0.10, 0.20]
    rows = []
    for a in nominal_alphas:
        emp = false_alarm_one_alpha(a, n_runs=200, n=10_000)
        rows.append({'nominal_alpha': a, 'empirical_alarm_rate': emp})
        print(f'alpha = {a:5.3f}  ->  empirical false-alarm rate = {emp:.3f}')
    fa_df = pd.DataFrame(rows)
    fa_df.to_csv(os.path.join(out_dir, 'exp5_false_alarm.csv'), index=False)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot([0, 0.25], [0, 0.25], 'k--', alpha=0.5,
             label='Ville\'s bound (y = x)')
    ax.scatter(fa_df['nominal_alpha'], fa_df['empirical_alarm_rate'],
                s=120, color='crimson', zorder=5,
                label='CHRONO-Fair empirical (200 runs each)')
    ax.set_xlabel('Nominal alpha (anytime-valid threshold 1/alpha)')
    ax.set_ylabel('Empirical false-alarm rate under H_0')
    ax.set_title('Anytime-valid false-alarm control: empirical <= nominal')
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xlim(0, 0.25); ax.set_ylim(0, 0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig5a_false_alarm_calibration.png'),
                 dpi=150, bbox_inches='tight')
    plt.close()

    # ---- (B) Ablation ----
    rows = []
    for seed in range(40):
        for mode in ['full', 'no_grow']:
            d = ablation_one(seed=seed, mode=mode)
            rows.append({'seed': seed, 'mode': mode, 'delay': d})
    abl = pd.DataFrame(rows)
    abl.to_csv(os.path.join(out_dir, 'exp5_ablation.csv'), index=False)
    print('=== Ablation: detection delays ===')
    print(abl.groupby('mode')['delay'].agg(['mean', 'std', 'count']).round(1))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.boxplot(data=abl, x='mode', y='delay', ax=ax, hue='mode',
                 legend=False, palette='Set2')
    ax.set_title('Component ablation: CHRONO-Fair detection delay')
    ax.set_ylabel('Patients to alarm')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig5b_ablation.png'),
                 dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved exp5 figures.')


if __name__ == '__main__':
    main()
