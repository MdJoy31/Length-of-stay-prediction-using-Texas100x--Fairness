"""
Experiment 16. Multi-model, multi-attribute fairness comparison on real
Texas-100X.

This experiment reads the real fairness numbers that the parent repository
computed on the full Texas-100X discharge dataset (925,128 records,
185,026-record held-out test set). Eight model variants were trained:
logistic regression, random forest, gradient boosting (CPU), XGBoost (GPU),
LightGBM (GPU), a PyTorch DNN, a stacking ensemble, and an LGB+XGB blend.
For each model, six fairness metrics were computed across four protected
attributes:

  - DI         disparate impact (the 80% rule applies)
  - WTPR       worst-group true-positive rate (the 0.8 rule applies)
  - SPD        statistical-parity difference (smaller absolute is better)
  - EOD        equalised-odds difference  (smaller absolute is better)
  - PPV_Ratio  positive-predictive-value ratio (the 0.8 rule applies)
  - Eq_Odds    equalised-odds aggregate    (smaller absolute is better)

This experiment does not retrain any model. It loads the stored real-data
numbers, formats them into reviewer-ready tables, and writes both wide and
long CSV summaries plus a comparison figure. The point is to show on real
Texas-100X numbers how the same model can pass on one protected attribute
and fail on another, and how different model families differ in their
per-attribute fairness profile.

Run: python -m chrono_fair.experiments.exp16_multi_model_fairness
"""
from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METRICS = ['DI', 'WTPR', 'SPD', 'EOD', 'PPV_Ratio', 'Eq_Odds']
ATTRS = ['RACE', 'ETHNICITY', 'SEX', 'AGE_GROUP']
RATIO_METRICS = {'DI', 'WTPR', 'PPV_Ratio'}    # 0.8 rule: higher is fairer
GAP_METRICS = {'SPD', 'EOD', 'Eq_Odds'}        # smaller absolute is fairer

# Display order chosen to put a classic linear baseline first, two boosted
# trees, and a DNN at the end so readers can read across families.
MODEL_ORDER = ['Logistic_Regression', 'Random_Forest', 'Gradient_Boosting',
                'XGBoost_GPU', 'LightGBM_GPU', 'PyTorch_DNN']
MODEL_LABELS = {'Logistic_Regression': 'LR',
                 'Random_Forest': 'RF',
                 'Gradient_Boosting': 'GB',
                 'XGBoost_GPU': 'XGBoost',
                 'LightGBM_GPU': 'LightGBM',
                 'PyTorch_DNN': 'DNN'}


def _verdict(value: float, metric: str) -> str:
    if metric in RATIO_METRICS:
        return 'fair' if value >= 0.80 else 'unfair'
    return 'fair' if abs(value) <= 0.10 else 'unfair'   # 0.1 SPD/EOD rule


def main():
    repo_root = os.path.join(os.path.dirname(__file__), '..', '..')
    src = os.path.join(repo_root, 'results', 'summary.json')
    out_dir = os.path.join(repo_root, 'chrono_fair_figures')
    os.makedirs(out_dir, exist_ok=True)
    with open(src) as fh:
        full = json.load(fh)
    fairness = full.get('fairness', {})
    perf = full.get('results', {})

    # ---- Long-form table: (model, attribute, metric, value, verdict) ----
    rows = []
    for model in MODEL_ORDER:
        if model not in fairness:
            continue
        for attr in ATTRS:
            for met in METRICS:
                v = fairness[model].get(attr, {}).get(met)
                if v is None:
                    continue
                rows.append({'model': MODEL_LABELS.get(model, model),
                              'attribute': attr, 'metric': met,
                              'value': round(float(v), 4),
                              'verdict': _verdict(float(v), met),
                              'test_AUC': round(float(perf.get(model, {})
                                                  .get('test_auc', float('nan'))),
                                                 4)})
    long = pd.DataFrame(rows)
    long.to_csv(os.path.join(out_dir, 'exp16_multi_model_long.csv'),
                 index=False)

    # ---- Wide table per metric: rows=model, cols=attribute ----
    for met in METRICS:
        sub = long[long['metric'] == met]
        pivot = sub.pivot(index='model', columns='attribute',
                           values='value').reindex(
                               [MODEL_LABELS[m] for m in MODEL_ORDER
                                if m in fairness])
        pivot = pivot.reindex(columns=ATTRS)
        pivot.to_csv(os.path.join(out_dir,
                                    f'exp16_wide_{met}.csv'))
        print(f'\n=== {met} (real Texas-100X test set) ===')
        print(pivot.round(4).to_string())

    # ---- Verdict table per attribute (0.8 rule / 0.1 gap rule) ----
    verdict_rows = []
    for model in MODEL_ORDER:
        if model not in fairness:
            continue
        for attr in ATTRS:
            rec = {'model': MODEL_LABELS.get(model, model), 'attribute': attr}
            for met in METRICS:
                v = fairness[model].get(attr, {}).get(met)
                if v is None:
                    rec[met] = '-'
                else:
                    rec[met] = _verdict(float(v), met)[0].upper()
            verdict_rows.append(rec)
    vt = pd.DataFrame(verdict_rows)
    vt.to_csv(os.path.join(out_dir, 'exp16_verdict_grid.csv'), index=False)
    print('\n=== Verdict grid (F = fair, U = unfair; 0.8 rule for ratios, '
           '0.1 gap rule for SPD/EOD/Eq_Odds) ===')
    print(vt.to_string(index=False))

    # ---- Cell-level summary: fair count per (model, attribute) ----
    cell_summary = (vt.set_index(['model', 'attribute'])
                       .apply(lambda r: (r == 'F').sum(), axis=1)
                       .reset_index(name='fair_metrics'))
    cell_summary['unfair_metrics'] = 6 - cell_summary['fair_metrics']
    cell_summary.to_csv(os.path.join(out_dir, 'exp16_cell_summary.csv'),
                         index=False)
    print('\n=== Fair-metric count per cell (out of 6) ===')
    print(cell_summary.pivot(index='model', columns='attribute',
                               values='fair_metrics').to_string())

    # ---- Figure: heatmap of DI across models x attributes ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    plot_metrics = [('DI', 0.8, '0.8 rule'), ('WTPR', 0.8, '0.8 rule'),
                     ('SPD', 0.1, '0.1 gap'), ('EOD', 0.1, '0.1 gap')]
    for ax, (met, thr, rule) in zip(axes.flat, plot_metrics):
        sub = long[long['metric'] == met]
        pivot = sub.pivot(index='model', columns='attribute',
                           values='value').reindex(
            index=[MODEL_LABELS[m] for m in MODEL_ORDER if m in fairness],
            columns=ATTRS)
        im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn'
                        if met in RATIO_METRICS else 'RdYlGn_r')
        ax.set_xticks(range(len(ATTRS)))
        ax.set_xticklabels(ATTRS, rotation=20, fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_title(f'{met}  ({rule})', fontsize=11)
        # annotate values
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.iat[i, j]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                          fontsize=8, color='black')
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle('Multi-model multi-attribute fairness on real Texas-100X',
                  fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig17_multi_model_fairness.png'),
                 dpi=150, bbox_inches='tight')
    plt.close()
    print('\nWrote fig17_multi_model_fairness.png')


if __name__ == '__main__':
    main()
