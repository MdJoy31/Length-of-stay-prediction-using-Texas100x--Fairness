"""Render the VFR-Audit to CHRONO-Fair conceptual diagram (paper Figure 2).

The diagram contrasts the static batch resampling of VFR-Audit with the
streaming patient-arrival monitoring of CHRONO-Fair, and shows how the
scalar VFR maps onto the time-resolved Flip Hazard survival curve.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp


def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..',
                            'chrono_fair_figures')
    os.makedirs(out_dir, exist_ok=True)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

    # ---------- Left: VFR-Audit, static batch resampling ----------
    axL.set_xlim(0, 10); axL.set_ylim(0, 10); axL.axis('off')
    axL.text(5, 9.5, 'VFR-Audit (prior work): static batch audit',
              ha='center', fontsize=12, fontweight='bold')
    rng = np.random.default_rng(0)
    for i in range(5):
        y = 7.6 - i * 1.05
        rect = mp.FancyBboxPatch((0.6, y), 4.4, 0.8,
                                   boxstyle='round,pad=0.03',
                                   facecolor='#d6eaf8', edgecolor='#333')
        axL.add_patch(rect)
        axL.text(2.8, y + 0.4, f'Resample {i+1}: fixed cohort',
                  ha='center', va='center', fontsize=8)
        verdict = 'fair' if rng.random() > 0.3 else 'unfair'
        col = '#1a7d3c' if verdict == 'fair' else '#c0392b'
        axL.text(6.0, y + 0.4, verdict, ha='center', va='center',
                  fontsize=9, color=col, fontweight='bold')
    axL.add_patch(mp.FancyBboxPatch((6.9, 3.7), 2.7, 3.4,
                  boxstyle='round,pad=0.05', facecolor='#fef3cd',
                  edgecolor='#333'))
    axL.text(8.25, 6.4, 'VFR', ha='center', fontsize=12, fontweight='bold')
    axL.text(8.25, 5.7, 'one scalar', ha='center', fontsize=8)
    axL.text(8.25, 5.2, 'fraction of', ha='center', fontsize=8)
    axL.text(8.25, 4.8, 'resamples', ha='center', fontsize=8)
    axL.text(8.25, 4.4, 'that flip', ha='center', fontsize=8)
    axL.annotate('', xy=(6.9, 5.4), xytext=(6.3, 5.4),
                  arrowprops=dict(arrowstyle='-|>', color='#333'))
    axL.text(5, 1.9, 'Computed once, before deployment.',
              ha='center', fontsize=9, style='italic')
    axL.text(5, 1.3, 'No time axis. No alarm. No cause.',
              ha='center', fontsize=9, style='italic', color='#c0392b')

    # ---------- Right: CHRONO-Fair, streaming monitoring ----------
    axR.set_xlim(0, 10); axR.set_ylim(0, 10); axR.axis('off')
    axR.text(5, 9.5, 'CHRONO-Fair: streaming patient-arrival monitoring',
              ha='center', fontsize=12, fontweight='bold')
    # patient stream
    for i in range(9):
        x = 0.6 + i * 1.0
        c = mp.Circle((x, 8.3), 0.28, facecolor='#d6eaf8', edgecolor='#333')
        axR.add_patch(c)
    axR.annotate('', xy=(9.7, 8.3), xytext=(0.3, 8.3),
                  arrowprops=dict(arrowstyle='-|>', color='#888'))
    axR.text(5, 7.6, 'patients arrive over time t', ha='center', fontsize=8,
              style='italic')
    # Flip Hazard survival curve
    t = np.linspace(0, 9, 200)
    S = np.exp(-0.18 * t)
    axR.plot(0.6 + t, 3.0 + 3.6 * S, color='#c0392b', linewidth=2.4)
    axR.text(5.0, 6.9, 'Flip Hazard survival S(t): P(no verdict flip by t)',
              ha='center', fontsize=9)
    axR.plot([0.6, 9.6], [3.0, 3.0], color='#333', linewidth=0.8)
    axR.plot([0.6, 0.6], [3.0, 6.8], color='#333', linewidth=0.8)
    # alarm marker
    axR.plot(0.6 + 6.2, 3.0 + 3.6 * np.exp(-0.18 * 6.2), 'v',
              color='black', markersize=11)
    axR.text(0.6 + 6.2, 3.0 + 3.6 * np.exp(-0.18 * 6.2) + 0.5,
              'e-process alarm', ha='center', fontsize=8)
    axR.text(5, 2.0, 'Scalar VFR is recovered as 1 - S at the audit horizon.',
              ha='center', fontsize=9, style='italic')
    axR.text(5, 1.4, 'Adds: when (time), where (cell), why (decomposition).',
              ha='center', fontsize=9, style='italic', color='#1a7d3c')

    # bridging arrow between panels
    fig.text(0.5, 0.5, 'extends', ha='center', fontsize=10,
              fontweight='bold', rotation=0,
              bbox=dict(boxstyle='rarrow,pad=0.3', facecolor='#fadbd8',
                        edgecolor='#333'))

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig_concept_vfr_to_chrono.png'),
                 dpi=150, bbox_inches='tight')
    plt.close()
    print('Wrote fig_concept_vfr_to_chrono.png')


if __name__ == '__main__':
    main()
