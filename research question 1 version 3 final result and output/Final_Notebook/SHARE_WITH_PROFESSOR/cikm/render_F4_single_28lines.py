"""
F4 in MANUSCRIPT style: single-panel with 28 lines (7 metrics × 4 attributes),
each line a (metric, attribute) cell. Color by metric family, linestyle by
attribute. Real data from T_axis2_real_CV.csv.
"""
import pandas as pd, numpy as np, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from pathlib import Path

ROOT = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm")
TAB = ROOT / "output_final" / "tables"
OUT = ROOT / "paper_images" / "most_updated"
OUT2 = ROOT / "output_final" / "figures" / "most_updated"
OUT.mkdir(parents=True, exist_ok=True)
OUT2.mkdir(parents=True, exist_ok=True)

mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['savefig.dpi'] = 300

METRICS = ['DI', 'SPD', 'EOPP', 'EOD', 'TI', 'PP', 'CAL']
ATTRS = ['RACE', 'SEX', 'ETHNICITY', 'AGE_GROUP']
ATTR_LABELS = {'RACE': 'Race', 'SEX': 'Sex', 'ETHNICITY': 'Eth', 'AGE_GROUP': 'Age'}
METRIC_COLORS = {
    'DI':   '#c0392b', 'SPD':  '#e67e22', 'EOPP': '#3a78b8', 'EOD':  '#16a085',
    'TI':   '#8e44ad', 'PP':   '#2c7d3a', 'CAL':  '#34495e',
}
ATTR_LINESTYLES = {'RACE': '-', 'SEX': '--', 'ETHNICITY': ':', 'AGE_GROUP': '-.'}

T = pd.read_csv(TAB / 'T_axis2_real_CV.csv')

fig, ax = plt.subplots(figsize=(8.5, 5.5))

for m in METRICS:
    for a in ATTRS:
        sub = T[(T['metric'] == m) & (T['attribute'] == a)].sort_values('N')
        cvs = sub['CV'].values
        Ns = sub['N'].values
        cvs = np.where(cvs > 0, cvs, 1e-4)
        ax.plot(Ns, cvs, marker='o', markersize=3.5, linewidth=1.5,
                color=METRIC_COLORS[m], linestyle=ATTR_LINESTYLES[a], alpha=0.85)

# CV=0.05 reference line
ax.axhline(0.05, color='black', linestyle='-', linewidth=1.6, alpha=0.9)
ax.text(1.05e3, 0.06, 'CV = 0.05 stability cutoff', fontsize=8.5, color='black', fontweight='bold')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Audit cohort N (log scale)', fontsize=11, fontweight='bold')
ax.set_ylabel('Coefficient of variation (log scale)', fontsize=11, fontweight='bold')
ax.set_title('F4 · CV vs audit-size N · 28 (metric × attribute) cells on canonical XGBoost\n'
             'Color = metric family · Line style = protected attribute · Real data (R=30 reps per N)',
             fontsize=10.5, fontweight='bold')
ax.grid(alpha=0.3, which='both')
ax.tick_params(labelsize=9)

# Two legends: one for metrics (color), one for attributes (linestyle)
metric_handles = [Line2D([0], [0], color=METRIC_COLORS[m], lw=2, marker='o', markersize=4, label=m) for m in METRICS]
attr_handles = [Line2D([0], [0], color='black', linestyle=ATTR_LINESTYLES[a], lw=1.5,
                        label=ATTR_LABELS[a]) for a in ATTRS]
leg1 = ax.legend(handles=metric_handles, loc='upper right', fontsize=9, frameon=True,
                  title='Metric (color)', title_fontsize=9, bbox_to_anchor=(1.0, 1.0))
ax.add_artist(leg1)
ax.legend(handles=attr_handles, loc='upper right', fontsize=9, frameon=True,
          title='Attribute (line)', title_fontsize=9, bbox_to_anchor=(1.0, 0.65))

plt.tight_layout()
for p in [OUT, OUT2]:
    plt.savefig(p / 'F4_cv_subplots.png', bbox_inches='tight', dpi=300)
plt.close()
print('F4 saved · single-panel 28 lines · real data')
