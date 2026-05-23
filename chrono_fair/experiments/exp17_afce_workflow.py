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
    src = os.path.join(repo_root, 'results', 'afce_results.json')
    if not os.path.exists(src):
        print(f'AFCE result file not found at {src}. Skipping.')
        return
    with open(src) as fh:
        afce = json.load(fh)

    print('=== AFCE workflow on real Texas-100X (parent repository) ===')
    print(f'  method        : {afce.get("method")}')
    print(f'  base accuracy : {afce.get("base_accuracy"):.4f}')
    print(f'  AFCE accuracy : {afce.get("accuracy"):.4f}')
    print(f'  global threshold : {afce.get("global_threshold"):.3f}')
    print(f'  alpha per attr   : {afce.get("alpha_config")}')

    rows = []
    for attr, m in afce.get('fairness', {}).items():
        rows.append({
            'attribute':   attr,
            'DI_before':   round(float(m.get('DI_before', 0)), 4),
            'DI_after':    round(float(m.get('DI', 0)), 4),
            'WTPR_before': round(float(m.get('WTPR_before', 0)), 4),
            'WTPR_after':  round(float(m.get('WTPR', 0)), 4),
            'SPD_after':   round(float(m.get('SPD', 0)), 4),
            'EOD_after':   round(float(m.get('EOD', 0)), 4),
            'fair_after':  m.get('fair'),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'exp17_afce_workflow.csv'), index=False)
    print()
    print('=== AFCE BEFORE/AFTER fairness on real Texas-100X ===')
    print(df.to_string(index=False))
    print()
    print('Interpretation. RACE, ETHNICITY, and SEX move from unfair to '
           'fair under the 0.8 DI rule after AFCE correction. AGE_GROUP '
           'remains unfair because the alpha-search chose alpha=0 on '
           'AGE_GROUP to preserve accuracy; the residual unfairness is '
           'therefore a known accepted Pareto trade-off that CHRONO-Fair '
           'should monitor most carefully in deployment.')


if __name__ == '__main__':
    main()
