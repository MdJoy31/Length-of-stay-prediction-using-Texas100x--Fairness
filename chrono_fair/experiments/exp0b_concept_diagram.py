"""Render the VFR-Audit to CHRONO-Fair conceptual diagram (paper Figure 1).

The diagram contrasts the static batch resampling of VFR-Audit with the
streaming patient-arrival monitoring of CHRONO-Fair, and shows how the
scalar VFR maps onto the time-resolved Flip Hazard survival curve.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.gridspec import GridSpec


def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..',
                            'chrono_fair_figures')
    os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 6.4))
    gs = GridSpec(1, 3, width_ratios=[1.0, 0.18, 1.0], wspace=0.05,
                  figure=fig)
    axL = fig.add_subplot(gs[0, 0])
    axM = fig.add_subplot(gs[0, 1])
    axR = fig.add_subplot(gs[0, 2])

    # ---------- Left panel: VFR-Audit, static batch resampling ----------
    axL.set_xlim(0, 10); axL.set_ylim(0, 10); axL.axis('off')
    axL.text(5, 9.55, 'VFR-Audit (prior work): static batch audit',
              ha='center', fontsize=14, fontweight='bold')
    rng = np.random.default_rng(0)
    for i in range(5):
        y = 7.6 - i * 1.05
        rect = mp.FancyBboxPatch((0.4, y), 4.2, 0.8,
                                   boxstyle='round,pad=0.03',
                                   facecolor='#d6eaf8', edgecolor='#333',
                                   linewidth=1.2)
        axL.add_patch(rect)
        axL.text(2.5, y + 0.4, f'Resample {i+1}: fixed cohort',
                  ha='center', va='center', fontsize=11)
        verdict = 'fair' if rng.random() > 0.3 else 'unfair'
        col = '#1a7d3c' if verdict == 'fair' else '#c0392b'
        axL.text(5.6, y + 0.4, verdict, ha='center', va='center',
                  fontsize=12, color=col, fontweight='bold')
    # VFR scalar box on the right side of the left panel
    axL.add_patch(mp.FancyBboxPatch((6.8, 3.7), 2.9, 3.4,
                  boxstyle='round,pad=0.05', facecolor='#fef3cd',
                  edgecolor='#333', linewidth=1.2))
    axL.text(8.25, 6.5, 'VFR', ha='center', fontsize=14, fontweight='bold')
    axL.text(8.25, 5.8, 'one scalar', ha='center', fontsize=11)
    axL.text(8.25, 5.3, 'fraction of', ha='center', fontsize=11)
    axL.text(8.25, 4.85, 'resamples', ha='center', fontsize=11)
    axL.text(8.25, 4.4, 'that flip', ha='center', fontsize=11)
    axL.annotate('', xy=(6.8, 5.4), xytext=(6.2, 5.4),
                  arrowprops=dict(arrowstyle='-|>', color='#333', lw=1.6))
    axL.text(5, 2.0, 'Computed once, before deployment.',
              ha='center', fontsize=12, style='italic')
    axL.text(5, 1.3, 'No time axis. No alarm. No diagnostic triage.',
              ha='center', fontsize=12, style='italic', color='#c0392b')

    # ---------- Middle panel: bridging arrow ----------
    axM.set_xlim(0, 1); axM.set_ylim(0, 10); axM.axis('off')
    # Big rightward arrow centred vertically in the gap between panels
    arrow = mp.FancyArrowPatch((0.05, 5.0), (0.95, 5.0),
                                 arrowstyle='-|>', mutation_scale=28,
                                 linewidth=3.0, color='#c0392b')
    axM.add_patch(arrow)
    axM.text(0.5, 6.4, 'extends', ha='center', va='center',
              fontsize=13, fontweight='bold', color='#c0392b')
    axM.text(0.5, 3.6, 'static verdict\ninstability\nbecomes\ntime-resolved\nmonitoring',
              ha='center', va='top', fontsize=10, style='italic',
              color='#444')

    # ---------- Right panel: CHRONO-Fair, streaming monitoring ----------
    axR.set_xlim(0, 10); axR.set_ylim(0, 10); axR.axis('off')
    axR.text(5, 9.55, 'CHRONO-Fair: streaming patient-arrival monitoring',
              ha='center', fontsize=14, fontweight='bold')
    # patient stream
    for i in range(9):
        x = 0.6 + i * 1.0
        c = mp.Circle((x, 8.3), 0.30, facecolor='#d6eaf8', edgecolor='#333',
                       linewidth=1.2)
        axR.add_patch(c)
    axR.annotate('', xy=(9.7, 8.3), xytext=(0.3, 8.3),
                  arrowprops=dict(arrowstyle='-|>', color='#888', lw=1.6))
    axR.text(5, 7.55, 'patients arrive over time t', ha='center',
              fontsize=11, style='italic')
    # Flip Hazard survival curve
    t = np.linspace(0, 9, 200)
    S = np.exp(-0.18 * t)
    axR.plot(0.6 + t, 3.0 + 3.6 * S, color='#c0392b', linewidth=2.8)
    axR.text(5.0, 6.95, 'Flip Hazard survival S(t)',
              ha='center', fontsize=12)
    axR.plot([0.6, 9.6], [3.0, 3.0], color='#333', linewidth=1.0)
    axR.plot([0.6, 0.6], [3.0, 6.8], color='#333', linewidth=1.0)
    # alarm marker
    axR.plot(0.6 + 6.2, 3.0 + 3.6 * np.exp(-0.18 * 6.2), 'v',
              color='black', markersize=14)
    axR.text(0.6 + 6.2, 3.0 + 3.6 * np.exp(-0.18 * 6.2) + 0.55,
              'anytime-valid alarm', ha='center', fontsize=11)
    axR.text(5, 2.0, 'Scalar VFR is recovered as 1 - S at the audit horizon.',
              ha='center', fontsize=12, style='italic')
    axR.text(5, 1.3, 'Adds: when (time), where (cell), how to triage.',
              ha='center', fontsize=12, style='italic', color='#1a7d3c')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig_concept_vfr_to_chrono.png'),
                 dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, 'fig_concept_vfr_to_chrono.pdf'),
                 bbox_inches='tight')
    plt.close()
    print('Wrote fig_concept_vfr_to_chrono.{png,pdf}')


if __name__ == '__main__':
    main()
