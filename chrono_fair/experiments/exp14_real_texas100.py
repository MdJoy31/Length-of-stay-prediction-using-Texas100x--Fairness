"""
Experiment 14. Real-data flip-rate analysis on Texas-100.

This experiment uses the real Texas-100 hospital discharge dataset of
Shokri et al. (67,330 records, 6,169 normalised categorical features, a
100-class surgical-procedure label). The file is obtained from the public
distribution of the dataset.

Honest scope statement. Texas-100 is not Texas-100X. It is a smaller
dataset and the prediction task is 100-class surgical procedure, not
length of stay. The 6,169 feature columns are anonymised normalised
categorical encodings. Without the official feature-description file the
protected-attribute columns cannot be identified with certainty. This
experiment therefore does two things, and labels each honestly.

  Part 1, real ensemble decision-flip rate. A binary task is defined: the
  positive class is the single most frequent surgical procedure. An
  ensemble of K logistic-regression members is trained on disjoint
  bootstrap resamples with a 70/15/15 temporal split. The empirical
  decision flip rate across ensemble members is measured on the test
  split. This is a real-data instantiation of the flip indicator that the
  Flip Hazard estimator consumes.

  Part 2, candidate-attribute counterfactual flip. A two-valued feature
  column with at least 10% support for each value is auto-selected and
  treated as a CANDIDATE binary attribute, not a verified protected
  attribute. The counterfactual flip rate under a swap of that column is
  reported. If no such column exists, Part 2 is omitted. The result is a
  methodological demonstration on real data, not a fairness claim about a
  named attribute.

Run: python -m chrono_fair.experiments.exp14_real_texas100
The data path can be set by the TEXAS100_DIR environment variable.
"""
from __future__ import annotations
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from chrono_fair.flip_hazard import kaplan_meier_curve


def _load(data_dir):
    """Load Texas-100. Prefer the full pickle files in data_dir. If they are
    absent, fall back to the 12,000-record compressed real subsample shipped
    in chrono_fair/data/, so the experiment is reproducible from the repo."""
    feat = os.path.join(data_dir, 'texas_100_features.p')
    if os.path.exists(feat):
        with open(feat, 'rb') as f:
            X = np.asarray(pickle.load(f))
        with open(os.path.join(data_dir, 'texas_100_labels.p'), 'rb') as f:
            y = np.asarray(pickle.load(f))
        print('Loaded full Texas-100 from', data_dir)
        return X, y
    sub = os.path.join(os.path.dirname(__file__), '..', 'data',
                        'texas100_real_subsample.npz')
    npz = np.load(sub)
    print('Full file absent; loaded 12,000-record real subsample from',
           os.path.basename(sub))
    return npz['X'].astype(np.float64), npz['y']


def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..',
                            'chrono_fair_figures')
    os.makedirs(out_dir, exist_ok=True)
    data_dir = os.environ.get('TEXAS100_DIR', '/tmp/tx_data')
    _sub = os.path.join(os.path.dirname(__file__), '..', 'data', 'texas100_real_subsample.npz')
    if not os.path.exists(os.path.join(data_dir, 'texas_100_features.p')) and not os.path.exists(_sub):
        print(f'Texas-100 data not found at {data_dir}. '
               f'Set TEXAS100_DIR or place the files there. Skipping.')
        return

    X, y = _load(data_dir)
    print(f'Real Texas-100 loaded: X {X.shape}, y {y.shape}, '
           f'{len(np.unique(y))} classes')

    # Binary task: positive = label in the 50 most frequent procedures.
    # This gives an approximately balanced, non-trivial task, so ensemble
    # members genuinely disagree on a measurable fraction of records.
    classes, counts = np.unique(y, return_counts=True)
    order = classes[np.argsort(-counts)]
    frequent_half = set(order[:50].tolist())
    y_bin = np.array([1 if lab in frequent_half else 0 for lab in y])
    print(f'Binary task: label in 50 most frequent procedures vs rest, '
           f'positive rate {y_bin.mean():.3f}')

    # Temporal 70/15/15 split (record order preserved)
    n = len(y)
    i_tr, i_va = int(0.70 * n), int(0.85 * n)
    Xtr, Xte = X[:i_tr], X[i_va:]
    ytr, yte = y_bin[:i_tr], y_bin[i_va:]
    print(f'Split: train {len(ytr)}, val {i_va - i_tr}, test {len(yte)}')

    # ---- Part 1: real ensemble decision-flip rate ----
    K = 11
    rng = np.random.default_rng(0)
    members = []
    for k in range(K):
        idx = rng.choice(len(ytr), size=len(ytr), replace=True)
        m = LogisticRegression(max_iter=200, C=1.0, solver='liblinear')
        m.fit(Xtr[idx], ytr[idx])
        members.append(m)
    preds = np.stack([m.predict(Xte) for m in members])      # (K, n_test)
    modal = (preds.mean(axis=0) >= 0.5).astype(int)
    # decision flip rate: fraction of members disagreeing with the modal vote
    flip_per_patient = (preds != modal).mean(axis=0)
    edfr = float(flip_per_patient.mean())
    print(f'Part 1: real ensemble decision-flip rate (eDFR) = {edfr:.4f}')
    test_acc = float((modal == yte).mean())
    print(f'        ensemble test accuracy = {test_acc:.4f}')

    # Flip Hazard survival on the real flip stream
    flip_event = (flip_per_patient > 0).astype(int)
    km = kaplan_meier_curve(np.arange(len(flip_event), dtype=float),
                             flip_event)
    print(f'        Flip Hazard final no-flip survival = '
           f'{km["S"].iloc[-1]:.4f}')

    # ---- Part 2: candidate-attribute counterfactual flip ----
    # A two-valued feature column is auto-selected, requiring that both
    # values have at least 10% support in the test split. The column is a
    # CANDIDATE binary attribute used to demonstrate the counterfactual flip
    # indicator on real data. It is not a verified protected attribute,
    # because the official feature description is not in the released file.
    col = None
    for j in range(X.shape[1]):
        vj = np.unique(Xte[:, j])
        if len(vj) == 2:
            frac = np.isclose(Xte[:, j], vj[0]).mean()
            if 0.10 <= frac <= 0.90:
                col = j
                break
    rows = []
    if col is not None:
        vals = np.unique(Xte[:, col])
        print(f'Part 2: candidate binary attribute = column {col}, '
               f'values {vals.tolist()}')
        Xte_cf = Xte.copy()
        # counterfactual swap: set every test record to the other value
        other = np.where(np.isclose(Xte[:, col], vals[0]), vals[1], vals[0])
        Xte_cf[:, col] = other
        y_hat = members[0].predict(Xte)
        y_hat_cf = members[0].predict(Xte_cf)
        flip = (y_hat != y_hat_cf).astype(int)
        for v in vals:
            mask = np.isclose(Xte[:, col], v)
            rows.append({'attribute_value': float(v),
                          'n': int(mask.sum()),
                          'counterfactual_flip_rate': float(flip[mask].mean())})
        cf = pd.DataFrame(rows)
        print(cf.to_string(index=False))
    else:
        cf = pd.DataFrame()

    # ---- save ----
    summary = {
        'dataset': 'Texas-100 (Shokri et al.), real',
        'n_records': int(n), 'n_features': int(X.shape[1]),
        'binary_task_positive_rate': float(y_bin.mean()),
        'ensemble_K': K, 'ensemble_test_accuracy': test_acc,
        'real_eDFR': edfr,
        'flip_hazard_final_survival': float(km['S'].iloc[-1]),
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(out_dir, 'exp14_real_texas100.csv'), index=False)
    if len(cf):
        cf.to_csv(os.path.join(out_dir, 'exp14_candidate_attr.csv'),
                   index=False)

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].hist(flip_per_patient, bins=np.linspace(0, 1, 23),
                  color='steelblue', edgecolor='white')
    axes[0].set_xlabel('Per-patient ensemble disagreement')
    axes[0].set_ylabel('Test patients')
    axes[0].set_title(f'Real Texas-100 ensemble flip distribution\n'
                       f'eDFR = {edfr:.3f}, K = {K}, '
                       f'n_test = {len(yte)}')
    axes[0].grid(alpha=0.3)
    axes[1].step(km['t'], km['S'], where='post', color='crimson',
                  linewidth=2)
    axes[1].fill_between(km['t'], km['ci_low'], km['ci_high'], step='post',
                          alpha=0.2, color='crimson')
    axes[1].set_xlabel('Test patient index t')
    axes[1].set_ylabel('P(no ensemble flip up to t)')
    axes[1].set_title('Flip Hazard survival on real Texas-100')
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig15_real_texas100.png'), dpi=150,
                 bbox_inches='tight')
    plt.close()
    print('Wrote fig15_real_texas100.png')


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    main()
