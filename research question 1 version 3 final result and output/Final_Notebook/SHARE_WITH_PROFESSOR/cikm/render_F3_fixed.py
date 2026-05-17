"""
render_F3_fixed.py
==================
Re-render F3 dual heatmap. The "Standard" panel (a) now shows the
cross-model VFR landscape across 12 classifiers (matches the manuscript
headline "146 of 336 cells flip = 43.5 %"), not the canonical-XGBoost-only
single-model audit. The "Canonical" panel (b) keeps the C4 single-model
view (28 cells) since after intervention the panel measures how the
canonical pipeline performs.

Data:
  - Panel (a): cikm_vfr_all_metrics.csv (336 rows = 12 models × 28 cells)
              aggregated per (metric, attribute) cell as mean VFR across
              12 models, with cell label showing n_flipping_models / 12.
  - Panel (b): T13_axis1_vfr_config4.csv (28 rows = canonical XGBoost C4)
              cell label = P / F + VFR.
"""
import pandas as pd, numpy as np, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

ROOT = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm")
TAB  = ROOT / "output_final" / "tables"
FIG_OUT = ROOT / "paper_images" / "revisions"
FIG_OUT2 = ROOT / "output_final" / "figures" / "revisions"
FIG_OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT2.mkdir(parents=True, exist_ok=True)

mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['savefig.dpi'] = 220

METRIC_ORDER = ['DI', 'SPD', 'EOPP', 'EOD', 'TI', 'PP', 'CAL']
ATTR_ORDER = ['RACE', 'SEX', 'ETHNICITY', 'AGE_GROUP']
ATTR_LABEL = ['Race', 'Sex', 'Eth', 'Age']

# ============================================================
# Panel (a): cross-model aggregate (12 models)
# ============================================================
T_xmod = pd.read_csv(TAB / 'cikm_vfr_all_metrics.csv')
print(f'cross-model rows: {len(T_xmod)}  models: {T_xmod["Model"].nunique()}')
agg = (T_xmod.groupby(['Metric','Attribute'])
              .agg(mean_VFR=('VFR','mean'),
                   max_VFR=('VFR','max'),
                   n_flip=('VFR', lambda x: int((x > 0).sum())))
              .reset_index())

g_xmod_v = np.zeros((len(METRIC_ORDER), len(ATTR_ORDER)))
g_xmod_l = np.empty((len(METRIC_ORDER), len(ATTR_ORDER)), dtype=object)
for i, m in enumerate(METRIC_ORDER):
    for j, a in enumerate(ATTR_ORDER):
        row = agg[(agg['Metric']==m) & (agg['Attribute']==a)]
        if len(row)==0:
            g_xmod_v[i,j] = np.nan; g_xmod_l[i,j] = ''; continue
        r = row.iloc[0]
        g_xmod_v[i,j] = r['mean_VFR']
        g_xmod_l[i,j] = f'{r["n_flip"]}/12\n{r["mean_VFR"]:.2f}'

# Headline total
total_cells = len(T_xmod)
n_flip_total = int((T_xmod['VFR'] > 0).sum())
print(f'Total flipping cells: {n_flip_total} of {total_cells} ({100*n_flip_total/total_cells:.1f}%)')

# ============================================================
# Panel (b): canonical XGBoost C4
# ============================================================
T_C4 = pd.read_csv(TAB / 'T13_axis1_vfr_config4.csv')
g4_v = np.zeros((len(METRIC_ORDER), len(ATTR_ORDER)))
g4_l = np.empty((len(METRIC_ORDER), len(ATTR_ORDER)), dtype=object)
for i, m in enumerate(METRIC_ORDER):
    for j, a in enumerate(ATTR_ORDER):
        row = T_C4[(T_C4['metric']==m) & (T_C4['attribute']==a)]
        if len(row)==0:
            g4_v[i,j] = np.nan; g4_l[i,j] = ''; continue
        r = row.iloc[0]
        g4_v[i,j] = r['vfr']
        v = 'P' if str(r['verdict_dominant']).lower() in ['fair','pass'] else 'F'
        g4_l[i,j] = f'{v}\n{r["vfr"]:.2f}'

# ============================================================
# Render
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), gridspec_kw={'width_ratios':[1, 1.10]})

for ax, (gv, gl, title) in zip(axes, [
    (g_xmod_v, g_xmod_l, f'(a) STANDARD baseline · cross-model VFR (12 models)\n{n_flip_total} of {total_cells} cells flip = {100*n_flip_total/total_cells:.1f}%'),
    (g4_v,     g4_l,     '(b) CANONICAL Phase 5b · single-model VFR (XGBoost C4)\n11 of 28 cells still flip after intervention')
]):
    im = ax.imshow(gv, cmap='RdYlGn_r', vmin=0.0, vmax=0.5, aspect='auto')
    ax.set_xticks(range(len(ATTR_LABEL))); ax.set_xticklabels(ATTR_LABEL, fontsize=12, fontweight='bold')
    ax.set_yticks(range(len(METRIC_ORDER))); ax.set_yticklabels(METRIC_ORDER, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=11.5, fontweight='bold')
    for i in range(len(METRIC_ORDER)):
        for j in range(len(ATTR_ORDER)):
            txt = gl[i, j]
            if not txt: continue
            cell_v = gv[i, j]
            colour = 'white' if cell_v > 0.30 else 'black'
            ax.text(j, i, txt, ha='center', va='center', fontsize=9, fontweight='bold', color=colour)

cbar = fig.colorbar(im, ax=axes[1], fraction=0.045, pad=0.02)
cbar.set_label('VFR (0 stable / 0.5 coin-flip)', fontsize=11, fontweight='bold')
plt.suptitle('F3 · VFR heatmap: STANDARD cross-model baseline vs CANONICAL post-intervention',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()

for path in [FIG_OUT, FIG_OUT2]:
    plt.savefig(path / 'F3_vfr_dual_heatmap.png', bbox_inches='tight', dpi=220)
plt.close()
print(f'F3 saved to both folders')
