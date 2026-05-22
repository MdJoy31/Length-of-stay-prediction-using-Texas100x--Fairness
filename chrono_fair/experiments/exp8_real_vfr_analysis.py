"""
Experiment 8. Real Texas-100X verdict-stability analysis: point estimate
versus bootstrap confidence interval versus Verdict Flip Rate (VFR).

This experiment uses the real fairness results computed by the parent
repository on the full Texas-100X discharge dataset (925,128 records,
185,026-record held-out test set). The inputs are read from
results/paper_analysis_all.json, which stores:

  * bootstrap_ci  : 1000-iteration bootstrap mean and 95% CI for each
                    accuracy metric and each fairness metric.
  * fluctuation   : 20 hospital-network resamples of DI, WTPR, SPD, EOD,
                    and PPV ratio for RACE, ETHNICITY, SEX, AGE_GROUP.

Three verdict procedures are compared on the same real numbers.

  (P) Point estimate. A metric is declared fair if its single full-sample
      value satisfies the 0.8 rule (the 80 percent rule for DI, WTPR,
      PPV ratio; difference metrics SPD and EOD are excluded here).
  (C) Bootstrap CI. The verdict is called unstable if the 95% CI
      straddles the 0.8 threshold.
  (V) Verdict Flip Rate. VFR is the fraction of the 20 resamples whose
      verdict differs from the majority verdict.

No data are synthesised in this experiment. Every number is read from the
parent repository's stored real-data outputs.
"""
from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


THRESHOLD = 0.80
RATIO_METRICS = ['DI', 'WTPR', 'PPV_Ratio']   # 0.8-rule applies to ratios


def _verdict(value: float) -> str:
    return 'fair' if value >= THRESHOLD else 'unfair'


def main():
    repo_root = os.path.join(os.path.dirname(__file__), '..', '..')
    analysis_path = os.path.join(repo_root, 'results', 'paper_analysis_all.json')
    out_dir = os.path.join(repo_root, 'chrono_fair_figures')
    os.makedirs(out_dir, exist_ok=True)

    with open(analysis_path) as fh:
        analysis = json.load(fh)
    fluct = analysis['fluctuation']
    bc = analysis['bootstrap_ci']

    # Index the bootstrap CI table by metric name
    ci_by_metric = {}
    for i in bc['Metric']:
        ci_by_metric[bc['Metric'][i]] = {
            'mean': bc['Mean'][i],
            'ci_low': bc['CI_Low'][i],
            'ci_high': bc['CI_High'][i],
        }

    rows = []
    for attr in fluct:
        for met in RATIO_METRICS:
            if met not in fluct[attr]:
                continue
            arr = np.asarray(fluct[attr][met], dtype=float)
            point = float(arr.mean())
            point_verdict = _verdict(point)
            verds = [_verdict(v) for v in arr]
            n_fair = verds.count('fair')
            n_unfair = verds.count('unfair')
            vfr = min(n_fair, n_unfair) / len(arr)
            # bootstrap CI verdict stability
            ci_key = f'{attr}_{met}'
            ci = ci_by_metric.get(ci_key)
            if ci is not None:
                straddles = ci['ci_low'] < THRESHOLD < ci['ci_high']
                ci_low, ci_high = ci['ci_low'], ci['ci_high']
            else:
                # CI not stored for this metric; bootstrap from resamples
                boot = np.array([np.random.default_rng(0).choice(
                    arr, size=len(arr), replace=True).mean()
                    for _ in range(1000)])
                ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
                straddles = ci_low < THRESHOLD < ci_high
            rows.append({
                'attribute': attr,
                'metric': met,
                'point_value': round(point, 4),
                'point_verdict': point_verdict,
                'ci_low': round(ci_low, 4),
                'ci_high': round(ci_high, 4),
                'ci_straddles_0.8': straddles,
                'resample_fair': n_fair,
                'resample_unfair': n_unfair,
                'VFR': round(vfr, 3),
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'exp8_real_vfr.csv'), index=False)
    print('=== Experiment 8: real Texas-100X verdict stability ===')
    print(df.to_string(index=False))

    # Agreement analysis between the three procedures
    df['ci_unstable'] = df['ci_straddles_0.8']
    df['vfr_unstable'] = df['VFR'] > 0
    n = len(df)
    point_silent = int((~df['ci_unstable'] & (df['VFR'] == 0)).sum())
    flagged_by_ci = int(df['ci_unstable'].sum())
    flagged_by_vfr = int(df['vfr_unstable'].sum())
    agree = int((df['ci_unstable'] == df['vfr_unstable']).sum())
    print()
    print(f'Metric-attribute combinations audited : {n}')
    print(f'Flagged unstable by bootstrap CI       : {flagged_by_ci}')
    print(f'Flagged unstable by VFR > 0            : {flagged_by_vfr}')
    print(f'CI and VFR agree on stability          : {agree}/{n}')
    print(f'Point estimate gives a definite verdict for all {n}, '
           f'including the {flagged_by_vfr} that VFR flags as unstable.')

    # ---- Figure: forest-style plot of point, CI, resample spread ----
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    df_sorted = df.sort_values('VFR', ascending=True).reset_index(drop=True)
    y = np.arange(len(df_sorted))
    for i, r in df_sorted.iterrows():
        colour = 'crimson' if r['VFR'] > 0 else 'steelblue'
        ax.plot([r['ci_low'], r['ci_high']], [i, i], color=colour,
                 linewidth=2.5, alpha=0.7)
        ax.plot(r['point_value'], i, 'o', color=colour, markersize=8)
    ax.axvline(THRESHOLD, color='black', linestyle='--', linewidth=1.3,
                label='0.8 fairness threshold')
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['attribute']}_{r['metric']}"
                         for _, r in df_sorted.iterrows()], fontsize=9)
    ax.set_xlabel('Fairness metric value (0.8-rule)')
    ax.set_title('Real Texas-100X: point estimate, 95% bootstrap CI, and VFR\n'
                  'red = VFR > 0 (verdict flips across resamples)')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig9_real_vfr_forest.png'), dpi=150,
                 bbox_inches='tight')
    plt.close()
    print('Wrote fig9_real_vfr_forest.png')


if __name__ == '__main__':
    main()
