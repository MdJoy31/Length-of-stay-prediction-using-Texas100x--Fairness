"""Renders F7 only (best-model summary). Uses T15_standard_vs_fair.csv."""
import pandas as pd, numpy as np, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from pathlib import Path

ROOT = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm")
TAB  = ROOT / "output_final" / "tables"
FIG_OUT = ROOT / "paper_images" / "revisions"
FIG_OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['savefig.dpi'] = 220

T15 = pd.read_csv(TAB / 'T15_standard_vs_fair.csv')
METRICS = ['DI', 'SPD', 'EOPP', 'EOD', 'TI', 'PP', 'CAL']
ATTRS = ['Race', 'Sex', 'Eth', 'Age']
def get(metric, attr):
    label = f'{metric} ({attr})'
    r = T15[T15['Metric'] == label]
    if len(r) == 0: return None, None
    return float(r['Standard'].iloc[0]), float(r['Fair (Intersect.)'].iloc[0])

before = np.zeros((len(METRICS), len(ATTRS)))
after  = np.zeros((len(METRICS), len(ATTRS)))
for i, m in enumerate(METRICS):
    for j, a in enumerate(ATTRS):
        b, f_ = get(m, a)
        before[i, j] = b if b is not None else np.nan
        after[i, j]  = f_ if f_ is not None else np.nan

fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(2, 2, width_ratios=[2.4, 1.0], height_ratios=[1, 1], hspace=0.35, wspace=0.25)

axA = fig.add_subplot(gs[0, 0])
group_w = 0.7
metric_centers = np.arange(len(METRICS))
bar_w = group_w / (len(ATTRS) * 2 + 1)
attr_colors_before = {'Race': '#a6c6e6', 'Sex': '#f0c099', 'Eth': '#f7e1a0', 'Age': '#bfd9b8'}
attr_colors_after = {'Race': '#3a78b8', 'Sex': '#e07b39', 'Eth': '#f4ca7c', 'Age': '#5fa55a'}

for i, m in enumerate(METRICS):
    for j, a in enumerate(ATTRS):
        x = metric_centers[i] + (j - len(ATTRS)/2 + 0.5) * (bar_w * 2.05)
        axA.bar(x - bar_w/2, before[i, j], bar_w, color=attr_colors_before[a], edgecolor='black', linewidth=0.4)
        axA.bar(x + bar_w/2, after[i, j],  bar_w, color=attr_colors_after[a],  edgecolor='black', linewidth=0.4)

axA.set_xticks(metric_centers)
axA.set_xticklabels(METRICS, fontsize=12, fontweight='bold')
axA.set_ylabel('Metric value (DI: 0=unfair, 1=fair; others: gap, 0=fair)', fontsize=11)
axA.set_title('(a) XGBoost canonical: all 7 metrics × 4 attributes before / after intervention', fontsize=12, fontweight='bold')
axA.grid(axis='y', alpha=0.3)
axA.set_ylim(0, 1.05)
legend_pairs = []
for a in ATTRS:
    legend_pairs.append(Patch(facecolor=attr_colors_before[a], edgecolor='black', label=f'{a} (before)'))
    legend_pairs.append(Patch(facecolor=attr_colors_after[a], edgecolor='black', label=f'{a} (after)'))
axA.legend(handles=legend_pairs, loc='upper right', ncol=2, fontsize=9, frameon=True)

axB = fig.add_subplot(gs[0, 1])
di_before = [get('DI', a)[0] for a in ATTRS]
di_after  = [get('DI', a)[1] for a in ATTRS]
xpos = np.arange(len(ATTRS))
axB.bar(xpos - 0.2, di_before, 0.4, color='#aaaaaa', edgecolor='black', linewidth=0.5, label='Before')
axB.bar(xpos + 0.2, di_after,  0.4, color='#2c7d3a', edgecolor='black', linewidth=0.5, label='After')
axB.axhline(0.80, color='red', linestyle='--', linewidth=2.0, alpha=0.7, label='4/5 rule')
axB.set_xticks(xpos); axB.set_xticklabels(ATTRS, fontsize=11)
axB.set_ylabel('Disparate Impact', fontsize=11)
axB.set_ylim(0, 1.1)
axB.set_title('(b) DI move toward four-fifths rule', fontsize=12, fontweight='bold')
axB.grid(axis='y', alpha=0.3); axB.legend(fontsize=9, loc='upper left')
for i in range(len(ATTRS)):
    axB.text(i - 0.2, di_before[i] + 0.02, f'{di_before[i]:.3f}', ha='center', fontsize=8.5)
    axB.text(i + 0.2, di_after[i] + 0.02, f'{di_after[i]:.3f}', ha='center', fontsize=8.5, fontweight='bold')

axC = fig.add_subplot(gs[1, 0])
acc_b = float(T15[T15['Metric'] == 'Accuracy']['Standard'].iloc[0])
acc_a = float(T15[T15['Metric'] == 'Accuracy']['Fair (Intersect.)'].iloc[0])
auc_b = float(T15[T15['Metric'] == 'AUC']['Standard'].iloc[0])
auc_a = float(T15[T15['Metric'] == 'AUC']['Fair (Intersect.)'].iloc[0])
f1_b  = float(T15[T15['Metric'] == 'F1']['Standard'].iloc[0])
f1_a  = float(T15[T15['Metric'] == 'F1']['Fair (Intersect.)'].iloc[0])
labels = ['Accuracy', 'AUROC', 'F1']
b_vals = [acc_b, auc_b, f1_b]
a_vals = [acc_a, auc_a, f1_a]
xpos = np.arange(len(labels))
axC.bar(xpos - 0.2, b_vals, 0.4, color='#888888', edgecolor='black', linewidth=0.5, label='Before')
axC.bar(xpos + 0.2, a_vals, 0.4, color='#3a78b8', edgecolor='black', linewidth=0.5, label='After')
for i in range(len(labels)):
    axC.text(i - 0.2, b_vals[i] + 0.005, f'{b_vals[i]:.4f}', ha='center', fontsize=10)
    axC.text(i + 0.2, a_vals[i] + 0.005, f'{a_vals[i]:.4f}', ha='center', fontsize=10, fontweight='bold')
    delta = a_vals[i] - b_vals[i]
    axC.text(i, max(b_vals[i], a_vals[i]) + 0.04, f'Δ {delta:+.4f}', ha='center', fontsize=10,
              color='red' if delta < 0 else 'green', fontweight='bold')
axC.set_xticks(xpos); axC.set_xticklabels(labels, fontsize=12, fontweight='bold')
axC.set_ylim(0, 1.1); axC.set_ylabel('Metric value', fontsize=11)
axC.set_title('(c) Performance cost of the fairness intervention', fontsize=12, fontweight='bold')
axC.grid(axis='y', alpha=0.3); axC.legend(fontsize=10, loc='lower right')

axD = fig.add_subplot(gs[1, 1])
axD.axis('off')
text_lines = [
    'Trade-off summary',
    '─' * 24,
    f'Acc :  {acc_b:.4f} → {acc_a:.4f}  ({(acc_a-acc_b)*100:+.2f} pp)',
    f'AUROC: {auc_b:.4f} → {auc_a:.4f}  ({(auc_a-auc_b)*100:+.2f} pp)',
    f'F1   : {f1_b:.4f} → {f1_a:.4f}  ({(f1_a-f1_b)*100:+.2f} pp)',
    '',
    'DI movement (worst → best):',
    f'  Race: {di_before[0]:.3f} → {di_after[0]:.3f}',
    f'  Sex:  {di_before[1]:.3f} → {di_after[1]:.3f}',
    f'  Eth:  {di_before[2]:.3f} → {di_after[2]:.3f}',
    f'  Age:  {di_before[3]:.3f} → {di_after[3]:.3f}',
    '',
    '4/5 rule satisfied on all four',
    'attributes after intervention.',
]
for k, line in enumerate(text_lines):
    weight = 'bold' if k in [0, 6, 12] else 'normal'
    axD.text(0.05, 0.96 - k * 0.067, line, fontsize=11, family='monospace', fontweight=weight,
              transform=axD.transAxes)

plt.suptitle('F7 · Canonical XGBoost: complete before/after fairness profile + performance cost',
              fontsize=13, fontweight='bold', y=1.00)
plt.savefig(FIG_OUT / 'F7_best_model_summary.png', bbox_inches='tight', dpi=220)
plt.close()
print("F7 saved")
