"""
F6 cleaned up — pure scatter + side legend. NO inline text labels (those caused
overlap). Each dot is identified by its color via the legend.
"""
import pandas as pd, numpy as np, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from pathlib import Path

ROOT = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm")
TAB  = ROOT / "output_final" / "tables"
FIG_OUT = ROOT / "paper_images" / "revisions"
FIG_OUT2 = ROOT / "output_final" / "figures" / "revisions"
FIG_OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT2.mkdir(parents=True, exist_ok=True)

mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['savefig.dpi'] = 220

T = pd.read_csv(TAB / 'T_per_model_before_after.csv')
print(f'loaded T_per_model_before_after.csv  ({len(T)} models)')

distinct_colors = [
    '#e6194B', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4',
    '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324',
]
T = T.reset_index(drop=True)

fig = plt.figure(figsize=(16, 7))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.28])
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axL = fig.add_subplot(gs[0, 2]); axL.axis('off')

def panel(ax, y_before_col, y_after_col, ylabel, title, ylim, show_floor=False):
    for k, r in T.iterrows():
        c = distinct_colors[k % len(distinct_colors)]
        x0, y0 = r['Acc_before'], r[y_before_col]
        x1, y1 = r['Acc_after'],  r[y_after_col]
        ax.scatter(x0, y0, s=140, facecolor='white', edgecolor=c, linewidth=2.2, marker='o', zorder=4)
        ax.scatter(x1, y1, s=220, facecolor=c, edgecolor='black', linewidth=1.2, marker='o', zorder=5)
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='-|>', color=c, alpha=0.7, lw=2.0), zorder=3)
    ax.axhline(0.80, color='red', linestyle='--', linewidth=2.0, alpha=0.85, label='Four-fifths rule (DI = 0.80)')
    ax.axhline(1.00, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    if show_floor:
        ax.axhspan(0.0, 0.32, color='red', alpha=0.10, zorder=1)
        # Place annotation in the empty top-right zone (above DI=0.8 line and to the right of dots)
        ax.text(0.885, 0.95,
                'Structural floor\n(Age base-rate gap = 0.399)\nrequires canonical Phase 5b\ngreedy refinement',
                ha='right', va='top', fontsize=8.5, fontweight='bold', color='#8b0000',
                bbox=dict(boxstyle='round,pad=0.45', facecolor='white', edgecolor='#8b0000', alpha=0.95))
    ax.set_xlabel('Test-set accuracy', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=11.5, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_ylim(*ylim)
    ax.set_xlim(0.77, 0.895)

panel(axA, 'DI_RACE_before', 'DI_RACE_after',
      'DI · Race axis (closer to 1 = fairer)',
      '(a) Race-axis threshold shift: works universally\nAll 12 models move above DI = 0.80',
      (0.45, 1.05))
panel(axB, 'DI_AGE_GROUP_before', 'DI_AGE_GROUP_after',
      'DI · Age axis',
      '(b) Age-axis threshold shift (naïve, per-attribute α only): structurally infeasible\nNo model crosses DI = 0.80',
      (0.0, 1.05), show_floor=True)

legend_handles = []
for k, r in T.iterrows():
    c = distinct_colors[k % len(distinct_colors)]
    legend_handles.append(Line2D([0], [0], marker='o', color='black', markerfacecolor=c,
                                  markersize=11, linewidth=0, label=f'{r["Model"]}'))
legend_handles.append(Line2D([0], [0], marker='', color='none', label=''))
legend_handles.append(Line2D([0], [0], marker='o', color='black', markerfacecolor='white',
                              markersize=11, linewidth=0, label='open = before'))
legend_handles.append(Line2D([0], [0], marker='o', color='black', markerfacecolor='black',
                              markersize=11, linewidth=0, label='filled = after'))
legend_handles.append(Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='4/5 rule'))
axL.legend(handles=legend_handles, loc='center left', fontsize=10, frameon=True,
            title='Classifiers (12) + markers', title_fontsize=11)

plt.suptitle('F6 · Per-model trade-off across 12 classifiers · Race-axis intervention works universally · Age-axis needs greedy refinement',
             fontsize=12.5, fontweight='bold', y=1.02)
plt.tight_layout()
for path in [FIG_OUT, FIG_OUT2]:
    plt.savefig(path / 'F6_per_model_tradeoff.png', bbox_inches='tight', dpi=220)
plt.close()
print('F6 saved (no inline labels)')
