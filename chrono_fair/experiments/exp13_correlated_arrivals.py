"""
Experiment I / 13. Correlated arrivals and seasonality.

The anytime-valid guarantee of the e-process assumes the flip indicator
sequence is, under H_0, a sequence with conditional mean rho_0. Real
admission streams are not independent: admission waves, ward rounds, and
seasonal case-mix shifts induce positive autocorrelation. This experiment
measures how the H_0 false-alarm rate and the H_1 detection delay change
when the flip indicator is generated with autocorrelation.

A first-order autoregressive latent process drives the flip probability.
The autocorrelation coefficient phi is varied in {0.0, 0.3, 0.6, 0.9}.
phi = 0.0 reproduces the independent stream. The marginal flip rate is held
at rho_0 = 0.05 under H_0 and rises to 0.15 after t = 6000 under H_1.
Each setting uses 200 H_0 streams and 200 H_1 streams of length 12,000.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from chrono_fair.e_process import EProcessMonitor


def _ar1_flip_stream(rng, n, base_rate, phi):
    """Generate a binary flip stream with AR(1)-driven probability.

    A latent Gaussian AR(1) process u_t is mapped through a logistic link so
    that the marginal flip rate matches base_rate. phi controls the
    autocorrelation; phi = 0 gives independent draws.
    """
    # latent AR(1) with unit marginal variance
    u = np.zeros(n)
    eps = rng.standard_normal(n) * np.sqrt(1 - phi ** 2)
    for t in range(1, n):
        u[t] = phi * u[t - 1] + eps[t]
    # shift the logit so the marginal mean equals base_rate
    from scipy.special import expit, logit
    intercept = logit(base_rate)
    prob = expit(intercept + 0.6 * u)            # 0.6 scales the swing
    # rescale to keep the empirical mean close to base_rate
    prob = prob * (base_rate / max(prob.mean(), 1e-9))
    prob = np.clip(prob, 0, 1)
    return (rng.random(n) < prob).astype(int)


def _run(seed, phi, drift, n=12_000, drift_at=6_000, rho0=0.05, rho1=0.15):
    rng = np.random.default_rng(seed)
    pre = _ar1_flip_stream(rng, drift_at, rho0, phi)
    if drift:
        post = _ar1_flip_stream(rng, n - drift_at, rho1, phi)
    else:
        post = _ar1_flip_stream(rng, n - drift_at, rho0, phi)
    stream = np.concatenate([pre, post])
    mon = EProcessMonitor(rho0=rho0, alpha=0.05)
    for t, z in enumerate(stream):
        mon.update(int(z))
        if mon.alarm_at is not None:
            break
    return mon.alarm_at


def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..',
                            'chrono_fair_figures')
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for phi in [0.0, 0.3, 0.6, 0.9]:
        fa = 0
        for seed in range(200):
            a = _run(seed, phi, drift=False)
            if a is not None:
                fa += 1
        delays, det = [], 0
        for seed in range(200):
            a = _run(seed + 9000, phi, drift=True)
            if a is not None and a >= 6000:
                delays.append(a - 6000)
                det += 1
        rows.append({'phi': phi, 'false_alarm_rate_H0': fa / 200,
                      'detection_rate_H1': det / 200,
                      'mean_delay_H1': float(np.mean(delays)) if delays else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'exp13_correlated.csv'), index=False)
    print('=== Experiment I/13: correlated arrivals ===')
    print(df.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].plot(df['phi'], df['false_alarm_rate_H0'], marker='o',
                  color='crimson')
    axes[0].axhline(0.05, ls='--', color='black', label='nominal alpha')
    axes[0].set_xlabel('AR(1) autocorrelation phi')
    axes[0].set_ylabel('H0 false-alarm rate (200 streams)')
    axes[0].set_title('False-alarm rate under correlated arrivals')
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(df['phi'], df['mean_delay_H1'], marker='s',
                  color='steelblue')
    axes[1].set_xlabel('AR(1) autocorrelation phi')
    axes[1].set_ylabel('H1 mean detection delay (patients)')
    axes[1].set_title('Detection delay under correlated arrivals')
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig14_correlated_arrivals.png'),
                 dpi=150, bbox_inches='tight')
    plt.close()
    print('Wrote fig14_correlated_arrivals.png')


if __name__ == '__main__':
    main()
