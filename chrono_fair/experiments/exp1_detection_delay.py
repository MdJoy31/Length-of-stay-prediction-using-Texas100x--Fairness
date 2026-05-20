"""
Experiment 1 — Detection Delay: CHRONO-Fair vs ADWIN vs Periodic Audit.

We inject a fairness drift at a known patient index `T*` and measure how
many patients each method takes to fire an alarm. Compares:

  - CHRONO-Fair (anytime-valid e-process, this paper)
  - ADWIN (Bifet & Gavalda 2007) -- one-sided, on the fairness gap time series
  - Periodic batch audit (every K patients, two-proportion z-test) -- the
    Davis et al. JAMIA 2025 baseline.
  - Static VFR (computed once at deployment, never updated) -- the user's
    own prior method.

Outputs:
  * fig1_detection_delay.png : barplot mean delay + boxplot per method
  * exp1_results.csv          : raw delays across 50 Monte-Carlo runs
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from chrono_fair.data.synthesizer import generate_stream, StreamConfig
from chrono_fair.e_process import EProcessMonitor


# ----- baselines -----
class ADWIN:
    """ADWIN-like detector (Bifet & Gavalda 2007), checked every 25 steps.

    Uses cumulative-sum trick so split comparison is O(splits) per check
    instead of O(window). We further sub-sample candidate splits.
    """

    def __init__(self, delta: float = 0.05, check_every: int = 25):
        self.delta = delta
        self.check_every = check_every
        self.cumsum = [0]
        self.n = 0
        self.alarm_at: int | None = None
        self._since_check = 0

    def update(self, z: int, t: int) -> bool:
        self.cumsum.append(self.cumsum[-1] + z)
        self.n += 1
        self._since_check += 1
        if self.n < 60 or self._since_check < self.check_every:
            return False
        self._since_check = 0
        c = self.cumsum
        n = self.n
        # Candidate splits: subsample
        splits = list(range(20, n - 20, max(1, n // 30)))
        for split in splits:
            mu0 = c[split] / split
            mu1 = (c[n] - c[split]) / (n - split)
            n0, n1 = split, n - split
            m = 1 / (1 / n0 + 1 / n1)
            eps_cut = np.sqrt((2 / m) * np.log(2 * n / self.delta))
            if abs(mu0 - mu1) > eps_cut and self.alarm_at is None:
                self.alarm_at = t
                # Shrink: drop pre-split prefix
                self.cumsum = [0] + [c[i] - c[split]
                                       for i in range(split + 1, n + 1)]
                self.n = n - split
                return True
        return False


class PeriodicAudit:
    """Two-proportion z-test every K patients, comparing recent vs baseline."""

    def __init__(self, baseline_rate: float, batch: int = 500, alpha: float = 0.05):
        self.p0 = baseline_rate
        self.batch = batch
        self.alpha = alpha
        self.buffer: list[int] = []
        self.n_seen = 0
        self.alarm_at: int | None = None

    def update(self, z: int, t: int) -> bool:
        self.buffer.append(z)
        self.n_seen += 1
        if len(self.buffer) >= self.batch:
            from scipy.stats import norm
            arr = np.array(self.buffer)
            p_hat = arr.mean()
            se = np.sqrt(self.p0 * (1 - self.p0) / len(arr))
            z_stat = (p_hat - self.p0) / max(se, 1e-9)
            p_val = 2 * (1 - norm.cdf(abs(z_stat)))
            self.buffer = []
            if p_val < self.alpha and self.alarm_at is None:
                self.alarm_at = t
                return True
        return False


# ----- one run -----
def one_run(seed: int, drift_at: int = 5000, n: int = 10_000,
             rho0: float = 0.05, rho1: float = 0.15) -> dict:
    rng = np.random.default_rng(seed)
    chrono = EProcessMonitor(rho0=rho0, alpha=0.05)
    adwin = ADWIN(delta=0.05)
    periodic = PeriodicAudit(baseline_rate=rho0, batch=500, alpha=0.05)
    static_vfr_alarm = None
    static_baseline = rho0
    static_buffer: list[int] = []

    for t in range(n):
        p = rho0 if t < drift_at else rho1
        z = int(rng.random() < p)
        chrono.update(z)
        adwin.update(z, t)
        periodic.update(z, t)
        # Static VFR: never updates after deployment day -- definitionally
        # cannot raise an alarm. We record it for completeness as +inf delay.
        static_buffer.append(z)

    def delay(alarm: int | None) -> float:
        if alarm is None or alarm < drift_at:
            return float('nan')   # missed detection
        return float(alarm - drift_at)

    return {
        'CHRONO-Fair': delay(chrono.alarm_at),
        'ADWIN': delay(adwin.alarm_at),
        'Periodic-500': delay(periodic.alarm_at),
        'Static-VFR': float('nan'),  # never alarms
    }


def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..',
                            'chrono_fair_figures')
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for seed in range(50):
        rows.append({'seed': seed, **one_run(seed=seed)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'exp1_detection_delay.csv'), index=False)

    melt = df.melt(id_vars='seed', var_name='method', value_name='delay')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    order = ['CHRONO-Fair', 'ADWIN', 'Periodic-500', 'Static-VFR']
    sns.boxplot(data=melt, x='method', y='delay', order=order, ax=axes[0],
                 palette='viridis')
    axes[0].set_title('Detection delay (patients) after drift onset', fontsize=11)
    axes[0].set_ylabel('Patients to alarm')
    axes[0].set_xlabel('')
    axes[0].grid(alpha=0.3)

    means = melt.groupby('method')['delay'].agg(['mean', 'std']).reindex(order)
    means['mean'].plot(kind='bar', yerr=means['std'], ax=axes[1],
                        color=sns.color_palette('viridis', 4))
    axes[1].set_title('Mean detection delay +/- 1 SD (n=50 runs)', fontsize=11)
    axes[1].set_ylabel('Patients to alarm')
    axes[1].grid(alpha=0.3)
    axes[1].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig1_detection_delay.png'), dpi=150,
                 bbox_inches='tight')
    plt.close()

    print('=== Experiment 1: Detection Delay ===')
    print(melt.groupby('method')['delay'].agg(['mean', 'median', 'std', 'count']).round(1))
    print('Wrote', os.path.join(out_dir, 'fig1_detection_delay.png'))


if __name__ == '__main__':
    main()
