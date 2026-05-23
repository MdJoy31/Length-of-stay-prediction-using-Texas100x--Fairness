"""Render the CHRONO-Fair system architecture diagram (Figure 2 of the paper).

The diagram is rendered procedurally via matplotlib so it stays in sync with
the codebase. It shows the patient stream feeding the deployed model, the
factual/counterfactual flip indicator, the four CHRONO-Fair modules, and the
Inspector report that aggregates the four outputs for governance review.
"""
from __future__ import annotations
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mp


def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..',
                            'chrono_fair_figures')
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.5)
    ax.axis('off')

    def box(x, y, w, h, label, color, edgecolor='#222', fontsize=12):
        rect = mp.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08',
                                   linewidth=1.4, edgecolor=edgecolor,
                                   facecolor=color)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha='center', va='center',
                  fontsize=fontsize, fontweight='bold')

    def arr(x1, y1, x2, y2, color='#222', curve=0.0):
        a = mp.FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle='-|>', mutation_scale=18,
                                 linewidth=1.6, color=color,
                                 connectionstyle=f"arc3,rad={curve}")
        ax.add_patch(a)

    # Column 1: input
    box(0.2, 5.4, 2.5, 1.0, 'Patient stream', '#d6eaf8', fontsize=13)
    box(0.2, 3.9, 2.5, 1.0, 'Deployed model', '#d6eaf8', fontsize=13)
    box(0.2, 2.4, 2.5, 1.0, 'Factual prediction', '#d6eaf8', fontsize=12)
    box(0.2, 0.9, 2.5, 1.0, 'Counterfactual\nprediction', '#d6eaf8',
         fontsize=12)

    # Column 2: flip indicator (vertical center)
    box(3.4, 3.0, 2.4, 1.2, 'Flip indicator\n$Z_i \\in \\{0, 1\\}$',
         '#fdebd0', fontsize=13)

    # Column 3: four modules
    box(6.7, 5.4, 3.0, 1.0, 'Flip Hazard', '#fad7a0', fontsize=13)
    box(6.7, 3.9, 3.0, 1.0, 'Anytime-valid\ne-process', '#fad7a0',
         fontsize=13)
    box(6.7, 2.4, 3.0, 1.0, 'Diagnostic\nattribution', '#fad7a0',
         fontsize=13)
    box(6.7, 0.9, 3.0, 1.0, 'RCAP\n(LOS regression)', '#fad7a0',
         fontsize=13)

    # Column 4: inspector and governance
    box(10.6, 3.2, 3.2, 2.2, 'Inspector report', '#abebc6', fontsize=14)
    box(10.6, 0.9, 3.2, 1.5, 'Governance\nreview', '#e8daef', fontsize=14)

    # Arrows: input chain
    arr(1.45, 5.4, 1.45, 4.9)
    arr(1.45, 3.9, 1.45, 3.4)
    arr(1.45, 2.4, 1.45, 1.9)

    # Arrows from factual + counterfactual into flip indicator
    arr(2.7, 2.9, 3.4, 3.4)
    arr(2.7, 1.4, 3.4, 3.2, curve=-0.15)

    # Arrows from flip indicator to four modules
    arr(5.8, 3.9, 6.7, 5.7, curve=0.2)
    arr(5.8, 3.7, 6.7, 4.3, curve=0.05)
    arr(5.8, 3.5, 6.7, 2.7, curve=-0.1)
    arr(5.8, 3.3, 6.7, 1.2, curve=-0.25)

    # Arrows from four modules to Inspector report
    arr(9.7, 5.7, 10.6, 5.0, curve=-0.1)
    arr(9.7, 4.3, 10.6, 4.4, curve=0.0)
    arr(9.7, 2.7, 10.6, 3.8, curve=0.1)
    arr(9.7, 1.2, 10.6, 3.4, curve=0.2)

    # Arrow from Inspector report to governance review
    arr(12.2, 3.2, 12.2, 2.4)

    ax.text(7.0, 6.95,
              'CHRONO-Fair: patient stream to Inspector report',
              ha='center', fontsize=15, fontweight='bold')

    plt.savefig(os.path.join(out_dir, 'fig0_architecture.png'),
                 dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, 'fig0_architecture.pdf'),
                 bbox_inches='tight')
    plt.close()
    print('Wrote fig0_architecture.{png,pdf}')


if __name__ == '__main__':
    main()
