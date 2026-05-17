"""
F4 (4-subplot one per attribute, single-column fit) + F5 (per-fold violin)
from canonical C4 data (T_axis2_real_CV.csv and T_axis3_real_per_fold.csv
after re-computation on Phase 5b yhat).
"""
import pandas as pd, numpy as np, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
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
METRIC_THRESHOLDS = {'DI': 0.80, 'SPD': 0.10, 'EOPP': 0.10, 'EOD': 0.10, 'TI': 0.10, 'PP': 0.10, 'CAL': 0.05}
METRIC_COLORS = {
    'DI':   '#c0392b', 'SPD':  '#e67e22', 'EOPP': '#3a78b8', 'EOD':  '#16a085',
    'TI':   '#8e44ad', 'PP':   '#2c7d3a', 'CAL':  '#34495e',
}
ATTR_COLORS = {'RACE': '#c0392b', 'SEX': '#3a78b8', 'ETHNICITY': '#e07b39', 'AGE_GROUP': '#5fa55a'}

# ===================================================================
# F4 · 4 subplots (one per attribute), single-column fit, no overlap
# ===================================================================
T2 = pd.read_csv(TAB / 'T_axis2_real_CV.csv')
print(f'F4 source: {len(T2)} rows · attrs: {T2["attribute"].unique()}')

fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.8), sharex=True, sharey=False)
axes = axes.flatten()
for idx, attr in enumerate(ATTRS):
    ax = axes[idx]
    for m in METRICS:
        sub = T2[(T2['attribute'] == attr) & (T2['metric'] == m)].sort_values('N')
        cvs = sub['CV'].values
        Ns = sub['N'].values
        cvs = np.where(cvs > 0, cvs, 1e-4)
        ax.plot(Ns, cvs, marker='o', markersize=3.5, linewidth=1.4, color=METRIC_COLORS[m], label=m)
    ax.axhline(0.05, color='black', linestyle='--', linewidth=1.2, alpha=0.7)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_title(ATTR_LABELS[attr], fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3, which='both')
    ax.tick_params(labelsize=8)
    if idx >= 2: ax.set_xlabel('N (log)', fontsize=9)
    if idx % 2 == 0: ax.set_ylabel('CV (log)', fontsize=9)

handles = [Line2D([0], [0], color=METRIC_COLORS[m], lw=2, marker='o', markersize=4, label=m) for m in METRICS]
handles.append(Line2D([0], [0], color='black', linestyle='--', lw=1.2, label='CV=0.05'))
fig.legend(handles=handles, loc='lower center', ncol=8, fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.06))
plt.suptitle('F4 · CV vs audit-size N · one panel per protected attribute · canonical C4 · single-column',
             fontsize=10, fontweight='bold', y=1.02)
plt.tight_layout()
plt.subplots_adjust(bottom=0.13)
for p in [OUT, OUT2]:
    plt.savefig(p / 'F4_cv_subplots.png', bbox_inches='tight', dpi=300)
plt.close()
print('F4 saved · 4 subplots single-column')

# ===================================================================
# F5 · per-fold violin from canonical C4 data
# ===================================================================
T3 = pd.read_csv(TAB / 'T_axis3_real_per_fold.csv')
print(f'F5 source: {len(T3)} rows · folds {T3["fold"].max()+1}')
# Verify DI values approximate canonical (should be ~0.80 for Race/Age)
di_summary = T3[T3['metric']=='DI'].groupby('attribute')['metric_value'].mean()
print(f'F5 DI per-attr mean: {di_summary.to_dict()}')

fig, ax = plt.subplots(figsize=(7.5, 4.2))
positions = []; group_centers = []
n_attr = len(ATTRS)
group_width = 1.0; gap = 0.25
violin_w = (group_width - 0.1) / n_attr

for gi, m in enumerate(METRICS):
    gc = gi * (group_width + gap); group_centers.append(gc)
    for ai, a in enumerate(ATTRS):
        x = gc + (ai - (n_attr - 1) / 2) * violin_w
        sub = T3[(T3['metric'] == m) & (T3['attribute'] == a)]
        vals = sub['metric_value'].values
        if len(vals) < 2: continue
        parts = ax.violinplot([vals], positions=[x], widths=violin_w * 0.88, showmedians=True)
        for body in parts['bodies']:
            body.set_facecolor(ATTR_COLORS[a]); body.set_alpha(0.7)
            body.set_edgecolor('black'); body.set_linewidth(0.5)
        for k in ['cmedians', 'cmaxes', 'cmins', 'cbars']:
            if k in parts: parts[k].set_color('black'); parts[k].set_linewidth(0.8)
    thr = METRIC_THRESHOLDS[m]
    x0 = gc - group_width/2 + 0.05; x1 = gc + group_width/2 - 0.05
    ax.hlines(thr, x0, x1, color='black', linestyle='--', linewidth=1.4, alpha=0.85)
    # τ value label next to the threshold line
    ax.text(x1 + 0.02, thr, f'τ={thr}', va='center', ha='left', fontsize=8.5,
            fontweight='bold', color='black',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85))

ax.set_xticks(group_centers); ax.set_xticklabels(METRICS, fontsize=10, fontweight='bold')
ax.set_ylabel('Metric value · 20 hospital folds × 4 attributes\n(80 verdicts per metric)',
              fontsize=9, fontweight='bold')
ax.set_title('F5 · Cross-hospital metric distribution on CANONICAL XGBoost C4 · K=20 GroupKFold',
             fontsize=10, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.tick_params(axis='y', labelsize=8)

y_max = T3['metric_value'].max() * 1.1
ax.set_ylim(-0.05, max(1.05, y_max))

legend_handles = [Patch(facecolor=ATTR_COLORS[a], edgecolor='black', label=ATTR_LABELS[a], alpha=0.7) for a in ATTRS]
legend_handles.append(Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='Operational threshold τ'))
ax.legend(handles=legend_handles, loc='upper right', fontsize=7.5, frameon=True, title='Attribute', title_fontsize=8, ncol=2)
plt.tight_layout()
for p in [OUT, OUT2]:
    plt.savefig(p / 'F5_hospital_violin_v2.png', bbox_inches='tight', dpi=300)
plt.close()
print('F5 saved · canonical C4 per-fold')
