"""
Experiment C / 10. Multiple-testing procedures for the intersectional monitor.

The intersectional monitor runs one e-process per cell. When several cells
are inspected at once, a multiple-testing correction is needed. Three
procedures are compared on a stream of m = 20 cells, of which k = 4 drift.

  - Step-wise Benjamini-Hochberg on p_t = 1/E_t. Cross-sectional at the
    inspection step. This is the procedure in the current framework.
  - e-BH (Wang and Ramdas, JRSS-B 2022). The Benjamini-Hochberg procedure
    applied directly to e-values. It controls the false-discovery rate for
    arbitrarily dependent e-values, which the p-value BH does not.
  - Bonferroni on p_t = 1/E_t. The conservative reference.

The experiment reports, at the final inspection step, the empirical
false-discovery proportion and the power (fraction of truly drifting cells
detected), averaged over 200 simulated streams. The point is not that one
procedure wins outright. The point is to show, with numbers, that step-wise
BH is a cross-sectional control and that e-BH is the procedure with a
dependence-robust guarantee, so the paper should describe the limitation
honestly.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from chrono_fair.e_process import EProcessMonitor


def _stream(seed, m=30, k=6, n=4500, drift_at=1500, rho0=0.05, rho1=0.12):
    """Marginal-signal regime. The post-drift rate 0.085 and the short
    800-patient post-drift window keep the drifting-cell e-values close to
    the rejection boundary, so the three procedures separate."""
    rng = np.random.default_rng(seed)
    drift_cells = set(range(k))            # first k cells drift
    monitors = [EProcessMonitor(rho0=rho0, alpha=0.05) for _ in range(m)]
    for t in range(n):
        for c in range(m):
            p = rho1 if (c in drift_cells and t >= drift_at) else rho0
            monitors[c].update(int(rng.random() < p))
    e_vals = np.array([np.exp(min(mon.log_E, 700.0)) for mon in monitors])
    return e_vals, drift_cells


def _reject_bh(p, q=0.10):
    m = len(p)
    order = np.argsort(p)
    sp = p[order]
    thr = q * np.arange(1, m + 1) / m
    below = np.where(sp <= thr)[0]
    k = below.max() + 1 if len(below) else 0
    rej = np.zeros(m, dtype=bool)
    rej[order[:k]] = True
    return rej


def _reject_ebh(e, q=0.10):
    # Wang and Ramdas e-BH: sort e descending, reject the largest k with
    # e_(k) >= m / (q k).
    m = len(e)
    order = np.argsort(-e)
    se = e[order]
    thr = m / (q * np.arange(1, m + 1))
    ok = np.where(se >= thr)[0]
    k = ok.max() + 1 if len(ok) else 0
    rej = np.zeros(m, dtype=bool)
    rej[order[:k]] = True
    return rej


def _reject_bonferroni(p, alpha=0.05):
    m = len(p)
    return p <= alpha / m


def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..',
                            'chrono_fair_figures')
    os.makedirs(out_dir, exist_ok=True)
    m, k = 30, 6
    rows = []
    for seed in range(200):
        e_vals, drift_cells = _stream(seed, m=m, k=k)
        p_vals = np.clip(1.0 / np.maximum(e_vals, 1e-12), 0, 1)
        truth = np.array([c in drift_cells for c in range(m)])
        for name, rej in [('Step-wise BH', _reject_bh(p_vals)),
                           ('e-BH', _reject_ebh(e_vals)),
                           ('Bonferroni', _reject_bonferroni(p_vals))]:
            n_rej = int(rej.sum())
            n_false = int((rej & ~truth).sum())
            n_true = int((rej & truth).sum())
            fdp = n_false / n_rej if n_rej > 0 else 0.0
            power = n_true / k
            rows.append({'seed': seed, 'procedure': name,
                          'FDP': fdp, 'power': power, 'n_rejected': n_rej})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'exp10_online_fdr.csv'), index=False)
    agg = df.groupby('procedure').agg(
        mean_FDP=('FDP', 'mean'),
        mean_power=('power', 'mean'),
        mean_rejections=('n_rejected', 'mean')).reset_index()
    print('=== Experiment C/10: multiple-testing procedures ===')
    print(f'm = {m} cells, k = {k} drifting, q = 0.10, 200 streams')
    print(agg.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    order = ['Step-wise BH', 'e-BH', 'Bonferroni']
    agg_i = agg.set_index('procedure').reindex(order)
    axes[0].bar(order, agg_i['mean_FDP'], color=['steelblue', 'seagreen', 'gray'])
    axes[0].axhline(0.10, ls='--', color='crimson', label='q = 0.10 target')
    axes[0].set_ylabel('Mean false-discovery proportion')
    axes[0].set_title('FDP across procedures')
    axes[0].legend(); axes[0].grid(alpha=0.3, axis='y')
    axes[1].bar(order, agg_i['mean_power'],
                 color=['steelblue', 'seagreen', 'gray'])
    axes[1].set_ylabel('Mean power (fraction of drifts detected)')
    axes[1].set_title('Power across procedures')
    axes[1].grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig11_online_fdr.png'), dpi=150,
                 bbox_inches='tight')
    plt.close()
    print('Wrote fig11_online_fdr.png')


if __name__ == '__main__':
    main()
