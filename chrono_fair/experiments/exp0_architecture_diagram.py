"""Render the CHRONO-Fair system architecture diagram (Figure 1 of the paper).

The diagram is rendered procedurally via matplotlib so it stays in sync with
the codebase and ships in the repo without binary tooling like Inkscape.
"""
from __future__ import annotations
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mp


def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..',
                            'chrono_fair_figures')
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis('off')

    def box(x, y, w, h, label, color, edgecolor='#222'):
        rect = mp.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.06',
                                   linewidth=1.4, edgecolor=edgecolor,
                                   facecolor=color)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha='center', va='center',
                  fontsize=10, fontweight='bold', wrap=True)

    def arr(x1, y1, x2, y2, label=None, color='#222', curve=0.0):
        a = mp.FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle='-|>', mutation_scale=14,
                                 linewidth=1.4, color=color,
                                 connectionstyle=f"arc3,rad={curve}")
        ax.add_patch(a)
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18, label,
                      ha='center', fontsize=8, color=color,
                      bbox=dict(boxstyle='round,pad=0.15',
                                facecolor='white', edgecolor='none', alpha=0.85))

    # Left: data
    box(0.2, 5.0, 2.4, 1.0,
         'Clinical stream\n(EHR / FHIR Subscription)\nTexas100x | MIMIC-IV',
         '#d6eaf8')
    box(0.2, 3.5, 2.4, 1.0,
         'Deployed model f_theta\nLOS classification / regression',
         '#d6eaf8')
    box(0.2, 2.0, 2.4, 1.0,
         'K-member ensemble\n{f_theta_k}_k=1..K (bootstrap)',
         '#d6eaf8')

    # Center: metrics
    box(3.5, 5.0, 3.2, 1.0,
         'Flip Hazard estimator\nKM + Nelson-Aalen + log-rank',
         '#fad7a0')
    box(3.5, 3.5, 3.2, 1.0,
         'Anytime-valid e-process\nGROW + intersectional FDR',
         '#fad7a0')
    box(3.5, 2.0, 3.2, 1.0,
         'Aleatoric / Epistemic decomp.\nensemble entropy split',
         '#fad7a0')
    box(3.5, 0.5, 3.2, 1.0,
         'RCAP (LOS regression)\nWasserstein-1 rank shift',
         '#fad7a0')

    # Right: outputs
    box(7.7, 5.0, 2.5, 1.0,
         'Flip Hazard curves\n+ RMFT, hazard ratios',
         '#abebc6')
    box(7.7, 3.5, 2.5, 1.0,
         'Per-cell alarms\n(FDR-controlled)',
         '#abebc6')
    box(7.7, 2.0, 2.5, 1.0,
         'Root-cause label\n{epistemic, aleatoric, mixed}',
         '#abebc6')
    box(7.7, 0.5, 2.5, 1.0,
         'Rank-shift histograms\n+ MAE-gap',
         '#abebc6')

    # Inspector agent
    box(10.5, 2.5, 2.3, 2.5,
         'Inspector Agent\n(Claude / templated)\n\n* alarm narrative\n'
         '* recommended action\n* FDA PCCP +\n  EU AI Act mapping',
         '#fadbd8')

    box(10.5, 0.5, 2.3, 1.5,
         'Clinician /\nGovernance UI\n(Streamlit dashboard)',
         '#e8daef')

    # Arrows
    arr(2.6, 5.5, 3.5, 5.5)
    arr(2.6, 4.0, 3.5, 4.0)
    arr(2.6, 2.5, 3.5, 2.5)
    arr(2.6, 4.0, 3.5, 1.0, curve=-0.2)
    arr(6.7, 5.5, 7.7, 5.5)
    arr(6.7, 4.0, 7.7, 4.0)
    arr(6.7, 2.5, 7.7, 2.5)
    arr(6.7, 1.0, 7.7, 1.0)
    arr(10.2, 4.0, 10.5, 4.0)
    arr(10.2, 2.5, 10.5, 3.0, curve=-0.2)
    arr(11.6, 2.5, 11.6, 2.0)

    # Header label
    ax.text(6.5, 6.7, 'CHRONO-Fair: Counterfactual Hazard-Rate ONline '
                       'Surveillance of Fairness',
              ha='center', fontsize=13, fontweight='bold')

    plt.savefig(os.path.join(out_dir, 'fig0_architecture.png'),
                 dpi=150, bbox_inches='tight')
    plt.close()
    print('Wrote architecture figure.')


if __name__ == '__main__':
    main()
