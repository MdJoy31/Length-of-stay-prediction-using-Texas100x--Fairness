"""
Experiment 17. AFCE pre-deployment fairness correction, then CHRONO-Fair
post-deployment monitoring (two-stage workflow).

Conceptual position. CHRONO-Fair is a fairness MONITOR, not a fairness
mitigator. In a deployment that takes fairness seriously, the model that
the monitor watches is not a raw classifier but a pre-corrected model
produced by an upstream fairness pipeline. In the parent repository, that
upstream pipeline is the Adaptive Fairness-Constrained Ensemble (AFCE) of
the VFR-Audit prior work, which combines (i) fairness-through-awareness
features, (ii) an accurate LGB+XGB ensemble, (iii) additive per-attribute
threshold offsets, (iv) hospital-cluster calibration, and (v) Pareto
trade-off control via an alpha-search over the per-attribute correction
strength. The pipeline therefore has three coordinated levers:

  * alpha    per-attribute correction strength, 0 (none) to 1 (full)
  * lambda   trade-off control between accuracy and fairness
  * threshold-sweep  the additive per-attribute decision threshold

This experiment loads the AFCE result JSON that the parent repository
produced on the full real Texas-100X test set and reports the BEFORE and
AFTER fairness numbers per protected attribute. The point is to show that
the AFCE pipeline brings the deployed verdict to fair (on RACE,
ETHNICITY, SEX) before CHRONO-Fair begins monitoring, and that AGE_GROUP
is the residual attribute where the corrected verdict is still unfair
under the 0.8 rule, so AGE_GROUP is the cell most at risk of an
alarm-worthy flip in the streaming regime.
"""
from __future__ import annotations
import os
import json
import pandas as pd


def main():
    repo_root = os.path.join(os.path.dirname(__file__), '..', '..')
    out_dir = os.path.join(repo_root, 'chrono_fair_figures')
    os.makedirs(out_dir, exist_ok=True)

    # Canonical Standard-vs-Fair table from the CIKM VFR-Audit notebook.
    canonical = os.path.join(repo_root, 'research question 1 version 3 '
                              'final result and output', 'Final_Notebook',
                              'output', 'tables',
                              'paper_table8a_accuracy_vs_fairness_subgroup.csv')
    if os.path.exists(canonical):
        df = pd.read_csv(canonical)
        df.to_csv(os.path.join(out_dir, 'exp17_canonical_std_vs_fair.csv'),
                   index=False)
        print('=== Canonical Standard-vs-Fair on real Texas-100X '
               '(CIKM VFR-Audit notebook) ===')
        print(df.to_string(index=False))
        n_fair_di = int((df['Fair DI'] >= 0.80).sum())
        print(f'\nFair model passes the 0.8 DI rule on '
               f'{n_fair_di} / {len(df)} protected attributes.')
        print(f'Accuracy: Std {df["Std Acc"].iloc[0]:.4f}, '
               f'Fair {df["Fair Acc"].iloc[0]:.4f}, '
               f'Delta = {float(df["Δ Acc (pp)"].iloc[0]):.2f} pp')
        return

    # Fallback to afce_results.json if the canonical table is not present.
    src = os.path.join(repo_root, 'results', 'afce_results.json')
    if not os.path.exists(src):
        print('Neither canonical CIKM table nor afce_results.json found. '
               'Skipping.')
        return
    with open(src) as fh:
        afce = json.load(fh)
    rows = []
    for attr, m in afce.get('fairness', {}).items():
        rows.append({
            'attribute':   attr,
            'DI_before':   round(float(m.get('DI_before', 0)), 4),
            'DI_after':    round(float(m.get('DI', 0)), 4),
            'fair_after':  m.get('fair'),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'exp17_afce_workflow.csv'), index=False)
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
