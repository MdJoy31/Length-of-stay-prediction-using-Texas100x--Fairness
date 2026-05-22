"""
Experiment D / 11. Counterfactual-definition sensitivity.

The flip indicator depends on how the counterfactual prediction is formed.
Three definitions are compared on the synthetic Texas-100X stream.

  - Naive swap. The protected attribute label is changed and the feature
    vector X is left unchanged. This ignores the marginal effect of the
    protected attribute on observable features.
  - Feature-shift. The known marginal shift that the protected attribute
    induces on features x0 to x2 is removed. This is the definition used by
    the CHRONO-Fair framework, following Maughan and Near.
  - Proxy-adjusted. The feature-shift is applied, and in addition a proxy
    feature (x3, here standing in for an attribute-correlated proxy such as
    neighbourhood or insurance) is partly reverted.

For each definition the experiment reports the group flip rate and the rank
order of the five racial groups by flip rate. The point is to quantify how
much the alarm and the subgroup ranking depend on the counterfactual
definition, which the paper currently states as a limitation without a
number.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from chrono_fair.data.synthesizer import generate_stream, StreamConfig


def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..',
                            'chrono_fair_figures')
    os.makedirs(out_dir, exist_ok=True)
    df = generate_stream(StreamConfig(n=20_000, seed=7, aleatoric_bias=0.05))
    feat = [c for c in df.columns if c.startswith('x')] + ['age_years']
    df_tr, df_te = df.iloc[:8000], df.iloc[8000:].reset_index(drop=True)
    model = LogisticRegression(max_iter=300).fit(
        df_tr[feat].to_numpy(), df_tr['y_ext'].to_numpy())

    y_hat = model.predict(df_te[feat].to_numpy())
    is_min = df_te['race'].isin(['Black', 'Hispanic']).to_numpy()

    def cf_predict(kind):
        X = df_te[feat].copy()
        if kind == 'naive_swap':
            pass  # X unchanged; only the (already encoded) label notion swaps
        elif kind == 'feature_shift':
            X.loc[is_min, ['x0', 'x1', 'x2']] = (
                X.loc[is_min, ['x0', 'x1', 'x2']].to_numpy() - 0.4)
        elif kind == 'proxy_adjusted':
            X.loc[is_min, ['x0', 'x1', 'x2']] = (
                X.loc[is_min, ['x0', 'x1', 'x2']].to_numpy() - 0.4)
            X.loc[is_min, 'x3'] = (
                X.loc[is_min, 'x3'].to_numpy() - 0.2)
        return model.predict(X.to_numpy())

    races = ['White', 'Black', 'Hispanic', 'Asian/PI', 'Other']
    rows = []
    for kind in ['naive_swap', 'feature_shift', 'proxy_adjusted']:
        y_cf = cf_predict(kind)
        flip = (y_hat != y_cf).astype(int)
        for r in races:
            mask = (df_te['race'] == r).to_numpy()
            if mask.sum() < 5:
                continue
            rows.append({'counterfactual': kind, 'race': r,
                          'n': int(mask.sum()),
                          'flip_rate': float(flip[mask].mean())})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, 'exp11_cf_sensitivity.csv'), index=False)
    pivot = res.pivot(index='race', columns='counterfactual',
                       values='flip_rate').reindex(races)
    print('=== Experiment D/11: counterfactual-definition sensitivity ===')
    print(pivot.round(4).to_string())
    # rank agreement
    for kind in pivot.columns:
        ranking = pivot[kind].sort_values(ascending=False).index.tolist()
        print(f'  {kind:16s} ranking high->low flip rate: {ranking}')

    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(races))
    w = 0.26
    for i, kind in enumerate(['naive_swap', 'feature_shift', 'proxy_adjusted']):
        ax.bar(x + (i - 1) * w, pivot[kind].values, w, label=kind)
    ax.set_xticks(x); ax.set_xticklabels(races)
    ax.set_ylabel('Group flip rate')
    ax.set_title('Counterfactual-definition sensitivity of the flip rate\n'
                  '($n = 12{,}000$ test stream)')
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig12_cf_sensitivity.png'), dpi=150,
                 bbox_inches='tight')
    plt.close()
    print('Wrote fig12_cf_sensitivity.png')


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    main()
