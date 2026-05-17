"""
F8 · Four-model full-Phase-5b summary
=====================================
Reads T_4model_before_after.csv (4 models × DI for 4 attributes × before/after
under the FULL Phase 5b pipeline) and renders a clean two-panel figure:

  (a) Grouped bars: per-model × per-attribute DI before/after with the
      four-fifths line. Shows that ALL FOUR DIs satisfy the rule after the
      full pipeline (matching manuscript Table 2 for XGBoost canonical).
  (b) Accuracy / cost summary: per-model accuracy before/after with the cost.
"""
import pandas as pd, numpy as np, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from pathlib import Path

ROOT = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm")
TAB  = ROOT / "output_final" / "tables"
FIG_OUT = ROOT / "paper_images" / "revisions"
FIG_OUT2 = ROOT / "output_final" / "figures" / "revisions"
FIG_OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT2.mkdir(parents=True, exist_ok=True)

mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['savefig.dpi'] = 220

T = pd.read_csv(TAB / 'T_4model_before_after.csv')
print(f"loaded {len(T)} models")

ATTRS = ['Race', 'Sex', 'Eth', 'Age']
attr_colors = {'Race': '#3a78b8', 'Sex': '#e07b39', 'Eth': '#f4ca7c', 'Age': '#5fa55a'}

fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(2, 1, height_ratios=[2.4, 1.0], hspace=0.35)

# (a) DI before/after per model × per attribute
axA = fig.add_subplot(gs[0, 0])
model_centers = np.arange(len(T))
group_w = 0.85
bar_w = group_w / (len(ATTRS) * 2 + 1)

for i, (_, r) in enumerate(T.iterrows()):
    for j, a in enumerate(ATTRS):
        x = model_centers[i] + (j - len(ATTRS)/2 + 0.5) * (bar_w * 2.05)
        b_val = r[f'DI_{a}_before']
        a_val = r[f'DI_{a}_after']
        col_b = attr_colors[a]
        # Before bar (lighter)
        axA.bar(x - bar_w/2, b_val, bar_w, color=col_b, alpha=0.40, edgecolor='black', linewidth=0.3)
        # After bar (full)
        axA.bar(x + bar_w/2, a_val, bar_w, color=col_b, alpha=1.00, edgecolor='black', linewidth=0.3)
        # Value labels
        axA.text(x - bar_w/2, b_val + 0.012, f'{b_val:.2f}', ha='center', fontsize=8, color='gray')
        axA.text(x + bar_w/2, a_val + 0.012, f'{a_val:.2f}', ha='center', fontsize=8, fontweight='bold')

axA.axhline(0.80, color='red', linestyle='--', linewidth=2.0, alpha=0.8, label='Four-fifths rule (DI ≥ 0.80)')
axA.set_xticks(model_centers)
axA.set_xticklabels(T['Model'].values, fontsize=12, fontweight='bold')
axA.set_ylabel('Disparate Impact', fontsize=12, fontweight='bold')
axA.set_ylim(0, 1.15)
axA.set_title('(a) Disparate Impact per protected attribute · before (light) vs after (saturated) intervention\n'
             'Cross-model verification: does Phase 5b generalise beyond XGBoost?',
             fontsize=11.5, fontweight='bold')
axA.grid(axis='y', alpha=0.3)

# Build legend pairs
legend_pairs = []
for a in ATTRS:
    legend_pairs.append(Patch(facecolor=attr_colors[a], alpha=0.40, edgecolor='black', label=f'{a} · before'))
    legend_pairs.append(Patch(facecolor=attr_colors[a], alpha=1.00, edgecolor='black', label=f'{a} · after'))
legend_pairs.append(Patch(facecolor='none', edgecolor='red', linestyle='--', label='4/5 rule'))
axA.legend(handles=legend_pairs, loc='upper right', fontsize=8.5, ncol=3, frameon=True, columnspacing=0.8)

# (b) Accuracy + cost
axB = fig.add_subplot(gs[1, 0])
x = np.arange(len(T))
axB.bar(x - 0.18, T['Acc_before'].values, 0.36, color='#888888', edgecolor='black', linewidth=0.4, label='Acc · before')
axB.bar(x + 0.18, T['Acc_after'].values,  0.36, color='#3a78b8', edgecolor='black', linewidth=0.4, label='Acc · after')
for i in range(len(T)):
    axB.text(i - 0.18, T['Acc_before'].iloc[i] + 0.008, f'{T["Acc_before"].iloc[i]:.4f}',
              ha='center', fontsize=9)
    axB.text(i + 0.18, T['Acc_after'].iloc[i] + 0.008, f'{T["Acc_after"].iloc[i]:.4f}',
              ha='center', fontsize=9, fontweight='bold')
    cost = T['Acc_cost'].iloc[i]
    color = 'red' if cost > 0 else 'green'
    axB.text(i, max(T['Acc_before'].iloc[i], T['Acc_after'].iloc[i]) + 0.05,
              f'cost {cost*100:+.2f} pp', ha='center', fontsize=10, color=color, fontweight='bold')
axB.set_xticks(x)
axB.set_xticklabels(T['Model'].values, fontsize=12, fontweight='bold')
axB.set_ylabel('Accuracy', fontsize=11)
axB.set_ylim(0.50, 1.05)
axB.set_title('(b) Test-set accuracy: before vs after intervention · cost reported above each pair',
             fontsize=12, fontweight='bold')
axB.grid(axis='y', alpha=0.3)
axB.legend(loc='lower right', fontsize=10)

plt.suptitle('F8 · Cross-model verification (4 classifiers, DI only) — complement to F7 (1 model, all 7 metrics)',
             fontsize=12.5, fontweight='bold', y=1.00)
for path in [FIG_OUT, FIG_OUT2]:
    plt.savefig(path / 'F8_4model_summary.png', bbox_inches='tight', dpi=220)
plt.close()
print('F8 saved')
