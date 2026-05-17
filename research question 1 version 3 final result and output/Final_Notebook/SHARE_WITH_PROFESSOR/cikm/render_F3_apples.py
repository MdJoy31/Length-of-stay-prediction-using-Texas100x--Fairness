"""
F3 — apples-to-apples canonical XGBoost C1 vs C4 (both 28 cells).

Same model, same audit grid, same colour scale. Cell label = P/F + VFR.
The visual narrative:
  Panel (a) C1 Standard: most cells are F (stably unfair), Race-axis has high-VFR
  Panel (b) C4 Canonical: most cells move to P, but Race-DI/SPD/EOPP/EOD still high-VFR
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

def grid(T):
    g_vfr = np.zeros((len(METRIC_ORDER), len(ATTR_ORDER)))
    g_lbl = np.empty((len(METRIC_ORDER), len(ATTR_ORDER)), dtype=object)
    flip = 0
    for i, m in enumerate(METRIC_ORDER):
        for j, a in enumerate(ATTR_ORDER):
            row = T[(T['metric']==m) & (T['attribute']==a)]
            if len(row)==0:
                g_vfr[i,j] = np.nan; g_lbl[i,j] = ''; continue
            r = row.iloc[0]
            g_vfr[i,j] = r['vfr']
            v = 'P' if str(r['verdict_dominant']).lower() in ['fair','pass'] else 'F'
            g_lbl[i,j] = f'{v}\n{r["vfr"]:.3f}'
            if r['vfr'] > 0: flip += 1
    return g_vfr, g_lbl, flip

T_C1 = pd.read_csv(TAB / 'T13_axis1_vfr_config1.csv')
T_C4 = pd.read_csv(TAB / 'T13_axis1_vfr_config4.csv')
g1_v, g1_l, n1 = grid(T_C1)
g4_v, g4_l, n4 = grid(T_C4)
print(f'C1 flipping cells: {n1}/28   C4 flipping cells: {n4}/28')

fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), gridspec_kw={'width_ratios':[1, 1.10]})

for ax, (gv, gl, title) in zip(axes, [
    (g1_v, g1_l, f'(a) STANDARD · canonical XGBoost (C1, no intervention)\n{n1} of 28 cells flip (VFR > 0)'),
    (g4_v, g4_l, f'(b) CANONICAL · canonical XGBoost (C4, Real+VFR pipeline)\n{n4} of 28 cells flip (VFR > 0)'),
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
            ax.text(j, i, txt, ha='center', va='center', fontsize=9.5, fontweight='bold', color=colour)

cbar = fig.colorbar(im, ax=axes[1], fraction=0.045, pad=0.02)
cbar.set_label('VFR (0 stable / 0.5 coin-flip)', fontsize=11, fontweight='bold')
plt.suptitle('F3 · Canonical XGBoost VFR landscape: Standard (C1) vs Canonical Phase 5b (C4)\n'
             'Same model · same 28 audit cells · same colour scale',
             fontsize=12.5, fontweight='bold', y=1.03)
plt.tight_layout()
for path in [FIG_OUT, FIG_OUT2]:
    plt.savefig(path / 'F3_vfr_dual_heatmap.png', bbox_inches='tight', dpi=220)
plt.close()
print('F3 saved')
