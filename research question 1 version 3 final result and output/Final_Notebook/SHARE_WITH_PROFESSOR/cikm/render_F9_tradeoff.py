"""
F9 · Fairness trade-off line graph (single-column).
x-axis: DI target (0.80 to 0.90)
y1:     accuracy cost (pp)
y2:     VFR (per axis)
"""
import pandas as pd, numpy as np, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

ROOT = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm")
TAB = ROOT / "output_final" / "tables"
OUT = ROOT / "paper_images" / "most_updated"
OUT2 = ROOT / "output_final" / "figures" / "most_updated"
OUT.mkdir(parents=True, exist_ok=True)
OUT2.mkdir(parents=True, exist_ok=True)

mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['savefig.dpi'] = 300

T = pd.read_csv(TAB / 'T_tradeoff_curve.csv')
print(T.to_string(index=False))

fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.5, 4.0))

# Panel (a): accuracy cost vs DI target (left y) + VFR (right y)
ax1 = axA
ax1.plot(T['DI_target'], T['Acc_cost_pp'], marker='o', markersize=8, linewidth=2.5,
         color='#c0392b', label='Accuracy cost')
ax1.set_xlabel('DI target (intervention dial)', fontsize=10, fontweight='bold')
ax1.set_ylabel('Accuracy cost (pp)', color='#c0392b', fontsize=10, fontweight='bold')
ax1.tick_params(axis='y', labelcolor='#c0392b', labelsize=9)
ax1.tick_params(axis='x', labelsize=9)
ax1.grid(alpha=0.3)
ax1.set_ylim(4, 8)
for i, r in T.iterrows():
    ax1.text(r['DI_target'], r['Acc_cost_pp'] + 0.15, f'{r["Acc_cost_pp"]:.2f}',
             ha='center', fontsize=8, color='#c0392b', fontweight='bold')

ax2 = ax1.twinx()
ax2.plot(T['DI_target'], T['VFR_Race'], marker='s', markersize=7, linewidth=2.0,
         color='#3a78b8', linestyle='--', label='VFR · DI Race')
ax2.plot(T['DI_target'], T['VFR_Age'], marker='^', markersize=7, linewidth=2.0,
         color='#5fa55a', linestyle='--', label='VFR · DI Age')
ax2.set_ylabel('VFR (verdict flip rate)', color='#3a78b8', fontsize=10, fontweight='bold')
ax2.tick_params(axis='y', labelcolor='#3a78b8', labelsize=9)
ax2.set_ylim(0, 0.55)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8, frameon=True)
axA.set_title('(a) The intervention dial · cost vs verdict stability', fontsize=10, fontweight='bold')

# Panel (b): post-intervention DI per attribute as function of target
for attr, color in [('Race', '#c0392b'), ('Sex', '#e07b39'), ('Eth', '#f4ca7c'), ('Age', '#5fa55a')]:
    axB.plot(T['DI_target'], T[f'DI_{attr}_post'], marker='o', markersize=7, linewidth=2.0,
             color=color, label=attr)
axB.axhline(0.80, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='4/5 rule')
axB.set_xlabel('DI target', fontsize=10, fontweight='bold')
axB.set_ylabel('Post-intervention DI', fontsize=10, fontweight='bold')
axB.set_ylim(0.78, 1.02)
axB.grid(alpha=0.3)
axB.legend(loc='lower right', fontsize=8, frameon=True, ncol=2)
axB.set_title('(b) Per-attribute DI as function of target', fontsize=10, fontweight='bold')
axB.tick_params(labelsize=9)

plt.suptitle('F9 · Fairness intervention trade-off dial · canonical XGBoost\n'
             'Increasing DI target trades accuracy for verdict stability',
             fontsize=10.5, fontweight='bold', y=1.05)
plt.tight_layout()
for p in [OUT, OUT2]:
    plt.savefig(p / 'F9_tradeoff_curve.png', bbox_inches='tight', dpi=300)
plt.close()
print('F9 saved')
